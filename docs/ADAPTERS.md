# Adapters: what is optional, what it costs, and what to install

The core package has **zero runtime dependencies** and that is a load-bearing
property, not a badge. A component that owns block/allow decisions has to be
auditable by a security team, and every transitive dependency is code that team
has to trust and patch. So nothing below is required, nothing below is imported
by the core, and every seam has a working option that needs only the standard
library.

## The seams

Six extension points. Each is a `Protocol` -- implement it yourself and ignore
everything here if you prefer.

| Seam | Protocol | Where it plugs in |
|---|---|---|
| Metrics | `MetricsSink` | `GuardrailPipeline(metrics=...)` |
| Session ceilings | `CounterStore` | `SessionLedger(store=...)` |
| Audit trail | `AuditSink` | `GuardrailPipeline(audit=...)` |
| Model tier | `ModelScorer` | `ModelDetector(scorer=...)` |
| Intent tier | `IntentClassifier` | `ScopeGate(classifier=...)` |
| Free-text PII | `EntityRecognizer` | `EntityDetector(recognizer=...)` |
| Embeddings | `TextEmbedder` | `SemanticScorer(embedder=...)` |

## Install matrix

```bash
pip install agent_guardrails                 # core + every stdlib adapter
pip install 'agent_guardrails[redis]'        # cross-host session ceilings
pip install 'agent_guardrails[prometheus]'   # Prometheus metrics
pip install 'agent_guardrails[otel]'         # OpenTelemetry metrics
pip install 'agent_guardrails[spacy]'        # NER: names, locations
pip install 'agent_guardrails[presidio]'     # NER + national-ID recognisers
pip install 'agent_guardrails[onnx]'         # local model tier (no torch)
pip install 'agent_guardrails[fastapi]'      # ASGI middleware
```

`spacy` and `presidio` need a second, non-pip step that pip cannot express:

```bash
python -m spacy download en_core_web_sm      # spacy adapter
python -m spacy download en_core_web_lg      # presidio's default
```

Imports are lazy (PEP 562). `import agent_guardrails.integrations` pulls in none
of these -- a test asserts that `redis`, `prometheus_client`, `opentelemetry`,
`spacy`, `onnxruntime` and `fastapi` are absent from `sys.modules` afterwards.
Touching a name whose extra is missing raises `ImportError` naming the exact
install command.

## Metrics

| Adapter | Needs | Cost/call | Use when |
|---|---|---|---|
| `NullMetrics` | stdlib | 0.19 us | you have no telemetry yet |
| `InMemoryMetrics` | stdlib | 3.7 us | tests, assertions on counters |
| `ConsoleMetricsSink` | stdlib | 5.9 us | local development |
| `LoggingMetricsSink` (level off) | stdlib | 0.30 us | wired but idle |
| `LoggingMetricsSink` (writing) | stdlib | 74 us | low QPS, or an existing log pipeline |
| `StatsdMetricsSink` | stdlib | 38 us | real aggregation, no new dependency |
| `StatsdMetricsSink(max_datagram_bytes=1400)` | stdlib | 9.1 us | as above, at volume |
| `PrometheusMetricsSink` | `[prometheus]` | -- | you already scrape Prometheus |
| `OtelMetricsSink` | `[otel]` | -- | you already run OTel collectors |

Measured on Python 3.12, `python benchmarks/adapter_bench.py`. Two things worth
internalising from those numbers:

* `observability.emit()` makes about four sink calls per verdict. An unsampled
  writing logging sink therefore adds ~300 us per verdict, which can exceed the
  entire deterministic scan. Use `sample_rate`, or send to statsd/OTel.
* An unbatched statsd send is one syscall per measurement.
  `max_datagram_bytes=1400` packs ~40 metrics per datagram for ~4x less
  overhead, at the cost of losing whatever is buffered if the process dies.
  That trade is fine for metrics and would not be for audit.

Prometheus has a structural mismatch worth knowing: collectors fix their label
names at registration, while `MetricsSink` accepts free-form tags per call. The
adapter pins each metric's labels to the first call it sees and projects later
calls onto that set, counting mismatches in `label_mismatches` rather than
raising into the enforcement path. **Never** tag with `session_id`, `run_id` or
`principal_id`: each distinct value is a permanent time series.

## Session ceilings (`CounterStore`)

This is the seam where the default is quietly wrong at scale.
`InMemoryCounterStore` is per-process, so **N workers enforce roughly N times
the configured ceiling**. Sticky sessions hide that; they do not fix it.

| Adapter | Needs | Cost/incr | Correct across |
|---|---|---|---|
| `InMemoryCounterStore` | stdlib | 4.2 us | one process |
| `SqliteCounterStore` | stdlib | 181 us (~5.5k/s) | all processes on one host |
| `RedisCounterStore` | `[redis]` | one round trip | all hosts |

`SqliteCounterStore` is a real fix, not a toy: each increment is an UPSERT
inside `BEGIN IMMEDIATE`, WAL enabled, busy-timeout set. A test starts four
processes doing 250 increments each and asserts the total is exactly 1000 --
which a read-modify-write implementation fails under contention.

Its limits, stated plainly: one host only, and **not** on NFS/EFS, where
sqlite's locking is unreliable. 5.5k increments/sec is the ceiling on ordinary
disk -- ample next to any LLM call, insufficient as a general-purpose counter.

`RedisCounterStore` maps the protocol onto one pipelined
`MULTI/HINCRBY.../EXPIRE/HGETALL/EXEC`: one round trip, no lost updates, sliding
TTL so an active session cannot win a fresh budget mid-conversation. It raises
on outage rather than silently reverting to an unenforced ceiling. If
availability must outrank the ceiling, wrap it in `FallbackCounterStore` --
which weakens enforcement to per-process during an outage and counts it in
`degraded_calls` so the condition is alertable.

## Audit

| Adapter | Needs | Cost/record | Chain survives |
|---|---|---|---|
| `JsonlAuditSink` | stdlib | fast | in-process only |
| `SqliteAuditSink` | stdlib | 367 us | restarts and concurrent writers |

JSONL keeps its chain head in memory, so a restart forks the chain and two
workers can link two records against the same predecessor. `SqliteAuditSink`
reads and advances the head inside one `BEGIN IMMEDIATE` transaction; a test
runs three processes writing 25 records each and asserts all 75 verify as one
unbroken chain. Its `records()` output is shaped for `verify_chain`, so the
verifier is backend-agnostic.

`synchronous=FULL` is the default. Neither backend is tamper-*proof*: anyone who
can write the file can rewrite history. The chain makes edits *detectable* by an
auditor holding an external copy of a later hash.

## Model and intent tiers

Zero-dependency first, because you probably do not need an SDK:

* `HttpModelScorer` -- POST JSON to any moderation endpoint or inference server
  (TGI, vLLM, Triton, Ollama, an internal classifier). You supply `parse=`
  because no two APIs agree on a response shape and a wrong guess yields
  silent zeros, i.e. a guardrail that always allows.
* `OpenAIModerationScorer` -- preset for the `/v1/moderations` shape. Note the
  real gap: OpenAI moderation has no notion of prompt injection or scope, so
  `INJECTION`/`JAILBREAK` get no signal from it. Keep the deterministic rules.
* `HttpIntentClassifier`, `HttpEntityRecognizer` -- same transport for the other
  two seams.
* `KeywordIntentClassifier`, `GazetteerEntityRecognizer` -- not models. Lexical
  stand-ins so the seams can be exercised in CI with no weights present. They
  abstain (`unclear`, or no spans) rather than guess, because `ScopeGate` widens
  scope on any non-`out_of_scope` label at confidence >= 0.6.

Three properties the HTTP adapters enforce:

1. Blocking `urllib` calls run in `asyncio.to_thread` with a hard
   `asyncio.wait_for` budget -- never on the event loop.
2. Transport errors **raise**. `ModelDetector` resolves them through the
   layer's `fail_closed` setting; swallowing them here would silently downgrade
   a regulated deployment to fail-open.
3. Cleartext is refused. `https` is required unless `allow_insecure_http=True`
   is passed explicitly, and a credential header is never sent over plaintext
   under any setting.

## Free-text PII

The bundled rules cover PII with *structure* (email, phone, Luhn-checked cards,
IBAN, national IDs, IPs). They cannot cover "Priya Sharma was discharged on
Tuesday". Under GDPR or HIPAA scope, treat NER as required.

| Adapter | Needs | Adds |
|---|---|---|
| `GazetteerEntityRecognizer` | stdlib | title cues + your own name list |
| `HttpEntityRecognizer` | stdlib | a remote NER/DLP service |
| `SpacyEntityRecognizer` | `[spacy]` + model | PERSON, LOCATION, DATE |
| `PresidioEntityRecognizer` | `[presidio]` + model | the above + country ID recognisers, checksums, context boosting |

Model choice is a latency decision: `en_core_web_sm` is fast and misses names in
noisy text; `en_core_web_trf` is materially more accurate and one to two orders
of magnitude slower. The spaCy adapter excludes tagger/parser/lemmatizer at load
(they contribute nothing to NER) and truncates at `nlp.max_length`.

Offsets are the whole contract: a recogniser must return spans indexing the
*exact normalized string it was given*. A service that re-tokenises before
computing offsets produces misplaced redactions -- a leak, not a formatting bug.
`HttpEntityRecognizer` drops out-of-range spans rather than trusting them.

Data-flow warning: `HttpEntityRecognizer` sends candidate PII to a third party.
That is what a DLP contract usually permits and what a residency requirement
usually forbids. Prefer in-process spaCy/Presidio where residency is
constrained.

## The local model tier: "any model" done safely

"Any model should just work" is right as a goal and dangerous as an
implementation. Taken literally it means fetching whatever a repo name resolves
to today and deserialising it -- and `.bin`/`.pt`/`.pth`/`.ckpt` are **pickle
streams that execute arbitrary code on load**. The guardrail is the last
component in a system that should be doing that.

So `[onnx]` supports any model that fits a verifiable contract:

```python
from agent_guardrails.integrations.onnx_model import FileSpec, OnnxModelSpec, OnnxModelScorer
from agent_guardrails.types import Category

spec = OnnxModelSpec(
    repo_id="your-org/injection-classifier-onnx",
    revision="e3b0c44298fc1c149afbf4c8996fb92427ae41e4",   # full commit SHA
    onnx_file=FileSpec("model.onnx", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"),
    tokenizer_file=FileSpec("tokenizer.json", "60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752"),
    label_map={"BENIGN": Category.OUT_OF_SCOPE, "INJECTION": Category.INJECTION},
    max_length=256,
    activation="sigmoid",   # sigmoid = multi-label head, softmax = single-label
)
scorer = OnnxModelScorer(spec)          # verifies on download and on every load
```

Enforced, with tests:

* **Format allowlist.** `.onnx`/`.json`/`.txt`/`.model` only. Pickle-bearing
  suffixes are refused *by name* with an explanation, before a byte is fetched;
  the ONNX protobuf marker is checked after.
* **Immutable revision.** A full 40-hex commit SHA. `main` is refused -- it
  re-points, turning "the model we validated" into "whatever was pushed last
  night".
* **SHA-256 per file**, verified on download *and* on every load from cache. A
  mismatch raises and deletes the file; there is no "probably fine" path.
  Download is verify-then-rename, so a truncated fetch never lands at the final
  path.
* **Explicit `label_map`.** Nothing can infer that `LABEL_1` means
  `Category.INJECTION`. An output-count mismatch raises instead of silently
  zipping short.
* **`allow_download=False`** for air-gapped deployments.
* **CPU provider, `intra_op_threads=1`** by default: under N gunicorn workers,
  onnxruntime's default of "all cores" gives N x cores threads fighting, and p99
  gets worse as you add workers.
* Inference runs in `asyncio.to_thread` -- a 5-40 ms blocking forward pass on
  the event loop would wreck tail latency.

`OnnxEmbedder` gives the `TextEmbedder` seam the same treatment (mean pooling
over the attention mask, L2-normalised) so the semantic tier can run locally
under the identical contract.

**What a pinned SHA does not prove.** It proves you get the same bytes every
time. It says nothing about whether those bytes are a well-behaved classifier.
Evaluate any model on your own corpus, record which revision you evaluated, and
re-evaluate when you re-pin. Nor does ONNX eliminate risk entirely: a graph can
reference custom operators, which is why only `CPUExecutionProvider` is enabled
and no custom-op libraries are registered.

## Choosing, in one pass

* One process, no telemetry yet: change nothing. The defaults are honest.
* One host, several workers: `SqliteCounterStore` + `SqliteAuditSink`. Still
  zero dependencies.
* Several hosts: `[redis]` for the ceiling. Everything else can stay stdlib.
* Existing Prometheus/OTel: install that one extra; keep statsd otherwise.
* GDPR/HIPAA text: `[presidio]` (or `[spacy]`), in-process, no egress.
* Want a model tier: try `HttpModelScorer` against a service you already run
  before installing `[onnx]`.
