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
pip install charterguard                 # core + every stdlib adapter
pip install 'charterguard[redis]'        # cross-host session ceilings
pip install 'charterguard[prometheus]'   # Prometheus metrics
pip install 'charterguard[otel]'         # OpenTelemetry metrics
pip install 'charterguard[spacy]'        # NER: names, locations
pip install 'charterguard[presidio]'     # NER + national-ID recognisers
pip install 'charterguard[onnx]'         # local model tier (no torch)
pip install 'charterguard[fastapi]'      # ASGI middleware
pip install 'charterguard[langchain]'    # LangChain runnables, tools, retriever
pip install 'charterguard[langgraph]'    # LangGraph: nodes need no import;
                                             # only approval_node() uses it
```

`spacy` and `presidio` need a second, non-pip step that pip cannot express:

```bash
python -m spacy download en_core_web_sm      # spacy adapter
python -m spacy download en_core_web_lg      # presidio's default
```

Imports are lazy (PEP 562). `import charterguard.integrations` pulls in none
of these -- a test asserts that `redis`, `prometheus_client`, `opentelemetry`,
`spacy`, `onnxruntime`, `fastapi`, `langchain_core` and `langgraph` are absent
from `sys.modules` afterwards.
Touching a name whose extra is missing raises `ImportError` naming the exact
install command.

The `langgraph` extra is the odd one out: it installs the framework, but the
adapter itself imports nothing from it (see below), so the nodes work on any
langgraph version and are unit-tested with the package absent.

## LangGraph

A LangGraph node is any callable over the state dict and an edge condition is
any callable returning a node name, so `integrations.langgraph` imports **no
third-party package at all**. That is a deliberate constraint, not a shortcut:
it means these nodes are version-proof against langgraph's fast-moving API and
are tested with neither langgraph nor langchain-core installed. Where
langchain-core *is* present, the nodes emit real `AIMessage`/`ToolMessage`
objects; where it is not, they emit role dicts, which langgraph coerces anyway.

```python
from langgraph.graph import StateGraph, MessagesState
from charterguard.integrations.langgraph import (
    GuardrailState, guard_input_node, guard_output_node,
    guard_tool_calls_node, guard_tool_results_node,
    guardrail_router, refusal_node, thread_session_id,
)

class State(MessagesState, GuardrailState):   # adds `guardrail` + `safe_output`
    pass

g = StateGraph(State)
g.add_node("guard_in",    guard_input_node(pipeline))
g.add_node("model",       call_model)
g.add_node("guard_calls", guard_tool_calls_node(pipeline))
g.add_node("tools",       ToolNode(tools))
g.add_node("guard_res",   guard_tool_results_node(pipeline))
g.add_node("guard_out",   guard_output_node(pipeline))
g.add_node("refuse",      refusal_node())

g.set_entry_point("guard_in")
g.add_conditional_edges("guard_in", guardrail_router(ok="model", blocked="refuse"))
g.add_edge("model", "guard_calls")
g.add_conditional_edges("guard_calls", guardrail_router(ok="tools", blocked=END))
g.add_edge("tools", "guard_res")
g.add_edge("guard_res", "model")
g.add_edge("guard_out", END)
```

Five design decisions in there are worth stating, because each one is a place a
hand-rolled version usually goes wrong.

**Guards record; the router enforces.** A guard node never diverts flow itself.
It writes a record to `state["guardrail"]` and the conditional edge decides.
This is what makes a phased rollout possible: ship the guards, log the FLAG and
BLOCK outcomes for a release, then wire the `blocked=` branch once the rate is
understood. It also keeps the graph's control flow visible in the graph, rather
than hidden inside a node.

**State stays JSON-serialisable.** The record holds `action`, `policy_id`,
category names, the rendered refusal and the audit row -- strings and numbers,
never a `Verdict`. Checkpointers persist state, and a custom dataclass in state
forces every backend (and every `astream` consumer) to handle it. A test asserts
`json.dumps` on the record.

**A blocked tool call answers *every* `tool_call_id`.** The chat-completions
contract requires exactly one result per requested call, and `ToolNode` runs all
calls on the message rather than a subset. So a partial block would either
strand an id -- a provider 400 on the very next turn -- or execute the refused
call anyway. `guard_tool_calls_node` short-circuits all of them with refusal
`ToolMessage`s, which keeps the transcript valid and lets the model recover
instead of ending the run.

**A quarantined tool result is neutralised, not dropped, and does not refuse the
turn.** Same contract: deleting a `ToolMessage` strands its call id. The content
is replaced by the provenance envelope so embedded instructions read as data,
`blocked` stays `false`, and the router keeps going. Retrieved *documents* are
different -- they carry no id the provider will demand an answer for -- so
`guard_retrieval_node` drops poisoned chunks outright.

**`safe_output` is separate from message history.** `guard_output_node` publishes
the text that is safe to show and leaves the transcript alone by default,
because what the user may see and what the model should remember are different
questions -- silently rewriting history changes what the next turn reasons over.
Pass `rewrite=True` to replace in place; it requires a message `id`, and raises
a clear error without one, since `add_messages` merges by id and an id-less
replacement would *append* the safe copy while leaving the original in history.

Finally, wire the session: `thread_session_id(config)` maps LangGraph's
`thread_id` to the guardrail session, which is what per-session volume ceilings
and injection-attempt counters key on. Left unset, every thread in the process
shares `"default"`, so one busy conversation trips the ceiling for everyone and
a per-user abuse signal is averaged away. The nodes call it automatically from
the `config` LangGraph passes them.

## LangChain

`integrations.langchain` requires `langchain-core`. The guard decisions live in a
dependency-free private module shared with the LangGraph adapter, so both
frameworks quarantine on identical rules and emit identical audit rows -- and
the behaviour is tested without either package installed.

```python
from charterguard.integrations.langchain import (
    GuardedRetriever, guard_input, guard_output, guarded_tools,
)

chain = guard_input(pipeline) | prompt | model | guard_output(pipeline)
answer = await chain.ainvoke("...", config={"configurable": {"session_id": "u-1"}})

retriever = GuardedRetriever(retriever=base_retriever, pipeline=pipeline)
agent = create_react_agent(model, guarded_tools(tools, pipeline))
```

`guard_input` accepts every shape a chain actually carries -- a string, a
message, a message list, a `PromptValue`, or a dict of chain variables -- and
returns the same shape with redactions applied.

The two guards default to **opposite** block behaviours, which is intentional.
`guard_input` raises `GuardrailBlocked`, because a chain that forgets to check a
return value must not be able to feed a blocked prompt to the model. By
`guard_output` a user is already waiting on the call, so it substitutes the
rendered refusal instead: an exception there surfaces as a 500, which is a worse
answer than a refusal. Both take `on_block="raise"|"refuse"` to override.

`GuardedRetriever` wraps the retriever rather than the prompt, so a new chain
built on the same retriever inherits the guard instead of forgetting it. This is
the layer that most needs it in a RAG app: a document is attacker-reachable
through whatever writes to the index, and nothing in the user's turn signals
that a retrieved chunk contains instructions.

`guard_tool` guards both halves of a tool call, which protect different things.
The pre-check enforces the charter and argument policy *before* any side effect
runs. The post-check treats the result as untrusted content -- redacting secrets
and wrapping the body in a provenance envelope -- so text a tool returns reads
as data rather than as instructions. The wrapper keeps the original name,
description and `args_schema`, so the model's view is unchanged and no
re-binding is needed. Refused calls return the refusal string rather than
raising, because an agent receives that as the tool's result and can adapt,
whereas an exception in a tool node usually ends the run. Guarding is async end
to end and the wrapper exposes no sync `func`: a blocking bridge inside a
running event loop is a deadlock waiting to happen, and agent executors use
`ainvoke` regardless.


## Streaming

A whole-response guard is the wrong shape for a streamed answer, and the failure
is quiet rather than loud. `guard_output` is a `RunnableLambda`, and a lambda
cannot return before it has seen all of its input: drop one into a streaming
chain and the entire response is buffered, so the user waits for the full
generation *and* gets no incremental enforcement. Use the streaming guards
instead.

```python
# LangChain -- a RunnableGenerator, so chunks flow as they arrive
from charterguard.integrations.langchain import guard_output_stream
chain = guard_input(pipeline) | prompt | model | guard_output_stream(pipeline)
async for chunk in chain.astream("...", config={"configurable": {"session_id": "u-1"}}):
    await send(chunk)

# LangGraph -- wraps stream_mode="messages"
from charterguard.integrations.langgraph import guard_message_stream
stream = graph.astream(state, config, stream_mode="messages")
async for chunk in guard_message_stream(pipeline, stream, session_id=sid):
    await send(chunk.text)          # blocked chunks carry the refusal
```

Both are built on `AsyncStreamGuard`, which shares its hold-back state machine
with the sync `StreamGuard` rather than reimplementing it -- a test asserts the
two release identical text for identical input, because two copies of this logic
would drift and only one of them would be the tested one.

**What hold-back buys.** Text is released only once `holdback_chars` of newer
text has been scanned behind it, so a secret split across two chunks
(`"...AKIA" + "IOSFODNN7EXAMPLE..."`) is caught before any of it is emitted.
Per-chunk scanning cannot see that pattern at all.

**The limit, stated honestly.** Already-released text cannot be recalled. The
window bounds how much text can be emitted *unscanned*; it is not a retraction
mechanism. A larger window tightens that bound and adds latency.

### `coalesce_chars`: the lever that actually controls cost

Every scan re-reads the whole buffer -- that is what catches boundary-spanning
patterns -- so scanning once per token makes per-token cost grow with response
length. Measured with `benchmarks/adapter_bench.py`, scanning per token versus
rescanning once per 32 characters:

| Response | per-token scan | `coalesce_chars=32` | Speedup |
|---|---|---|---|
| 32 tokens | 62.8 us/token | 22.1 us/token | 2.8x |
| 128 tokens | 74.6 us/token | 10.3 us/token | 7.2x |
| 512 tokens | 98.8 us/token | 0.6 us/token | 166x |

Note which direction the per-token number moves without coalescing: it *rises*
with length, which is the signature of an O(n^2) total. That is why the adapter
stream guards default to `coalesce_chars=32` while the core `StreamGuard`
defaults to `0` for backwards compatibility.

It is worth being precise about which knob does what, because it is easy to
reach for the wrong one: `holdback_chars` changes *when* text is released,
`coalesce_chars` changes *how often* it is scanned. Only the second affects cost.
Coalescing cannot leak, because nothing is released before it has been scanned --
it trades release granularity, never safety, and a test asserts a split secret is
still caught at every coalescing setting.

Streamed output is charged to the session ledger when you pass `session_id`
(the adapters do this from config automatically). Before, a streamed response
charged no volume at all, which meant per-session output ceilings never fired on
the largest responses -- exactly the ones they exist for.

## Human approval (LangGraph interrupts)

The security docs are blunt that detection is not containment: least-privilege
tools and human approval for irreversible actions are what bound the damage when
injected instructions *are* followed. `approval_node` implements the second of
those, because LangGraph has durable interrupts and so is the one place it can be
done properly.

```python
from charterguard.integrations.langgraph import approval_node, guardrail_router

g.add_node("guard_calls", guard_tool_calls_node(pipeline))
g.add_node("approve", approval_node())            # on_flagged=True to review FLAGs too
g.add_edge("guard_calls", "approve")
g.add_conditional_edges("approve", guardrail_router(ok="tools", blocked="refuse"))
```

On a block it calls `interrupt(payload)`, which suspends the graph, persists the
payload to the checkpoint, and surfaces it to whatever is driving the run. Resume
with `Command(resume=True)` to approve, or `Command(resume={"approved": True,
"reviewer": "sre@corp", "note": "incident 4821"})` to record who decided and why.
Approval clears `blocked`, so the existing router carries the run onward with no
extra wiring, while the original verdict fields stay on the record for audit.

Three properties worth knowing:

**It fails closed.** `False`, `None`, `"nope"`, `{}`, `0` and any unrecognised
object are all rejections. The alternative -- treating an unparsed resume as
consent -- means a malformed or replayed resume silently authorises the action
the guardrail stopped.

**The node does nothing but read state and interrupt.** `interrupt()` suspends by
raising, and on resume LangGraph re-executes the node *from the top*. Any scan,
ledger write or audit emission placed here would run twice and be double-counted.
Guarding belongs in the guard node before it; the decision recorded here is
durable because the checkpointer persists state.

**The dependency is per-call, not per-import.** `interrupt` is resolved lazily,
so importing the adapter stays free, and passing `interrupt_fn=...` lets you use
your own approval mechanism (or a stub, in tests) with no langgraph at all.


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
from charterguard.integrations.onnx_model import FileSpec, OnnxModelSpec, OnnxModelScorer
from charterguard.types import Category

spec = OnnxModelSpec(
    repo_id="your-org/injection-classifier-onnx",
    revision="e3b0c44298fc1c149afbf4c8996fb92427ae41e4",   # full commit SHA
    onnx_file=FileSpec("model.onnx", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"),
    tokenizer_file=FileSpec("tokenizer.json", "60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752"),
    label_map={"BENIGN": None, "INJECTION": Category.INJECTION},   # None = ignored output
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

### Ready-made preset

`charterguard.integrations.onnx_presets` ships one fully pinned spec, so you do not have
to compute digests to try the tier:

```python
from charterguard import GuardrailConfig, GuardrailPipeline
from charterguard.integrations.onnx_presets import injection_scorer

pipeline = GuardrailPipeline(GuardrailConfig.strict(), model=injection_scorer())
verdict = await pipeline.acheck_input(text)   # the model tier is async-only
```

| Preset | Model | License | Size | Labels |
| --- | --- | --- | --- | --- |
| `protectai-deberta-v3-injection-v2` (default) | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) @ `90c9989b`, ONNX export | Apache-2.0 | ~740 MB | `SAFE` (ignored), `INJECTION` |

The digests were computed from downloaded files and match the hub's LFS object ids.
Files are cached under `~/.cache/charterguard` (override with `cache_dir=` or
`CHARTERGUARD_MODEL_CACHE`). For air-gapped hosts, copy the cache directory over and
pass `allow_download=False`. The model is English-centric and runs about 60 ms per
input on one CPU core. See [accuracy](accuracy.md#with-the-model-tier) for measured
numbers.

A `label_map` entry of `None` marks an output the model produces but that should not
score anything, such as a `SAFE` class. Do not map it to an unrelated category.

## Choosing, in one pass

* One process, no telemetry yet: change nothing. The defaults are honest.
* One host, several workers: `SqliteCounterStore` + `SqliteAuditSink`. Still
  zero dependencies.
* Several hosts: `[redis]` for the ceiling. Everything else can stay stdlib.
* Existing Prometheus/OTel: install that one extra; keep statsd otherwise.
* GDPR/HIPAA text: `[presidio]` (or `[spacy]`), in-process, no egress.
* Want a model tier: try `HttpModelScorer` against a service you already run
  before installing `[onnx]`.

---

# Operations tier: rule health and policy rollout

Two additions that need no dependency and answer questions the rest of the
package deliberately leaves open.

## `RuleHealth` -- which rules earn their place

`tools/rule_coverage.py` answers "is every rule exercised?" against a corpus.
`RuleHealth` answers it against production traffic, which is where the real
distribution lives.

```python
from charterguard import RuleHealth
from charterguard.patterns import RULES

health = RuleHealth(RULES)          # the inventory is the point -- see below
verdict = pipeline.check_input(text)
health.record(verdict.findings)     # one dict increment per finding
...
print(health.report())
health.emit(metrics)                # per-rule gauges, bounded cardinality
```

The trap this exists to avoid: a bare `defaultdict(int)` of fire counts can
*never* report a dead rule, because a rule that never fires never gets a key, so
`{rid for rid, n in fires.items() if n == 0}` is always empty. The inventory has
to be seeded from the rule set. A test asserts that a 99-rule pack with one
firing rule reports 98 dead ones.

`dead_rules()` returns nothing below `min_verdicts` (default 10,000), because
"never fired" is the expected state for most of the pack after 200 verdicts and
reporting it would train people to ignore the output.

What it does **not** report is rule *shadowing*, in the sense of one rule
preventing another from matching. This engine makes that impossible by
construction: every alternation branch is matched independently, and a broad rule
that later fails validation cannot consume a region a narrower rule should own.
Hunting a failure mode the architecture excludes produces noise. What it reports
instead is **redundant pairs** -- rules that never fire except alongside another
rule, which is two patterns to review for one signal.

## `ShadowPolicy` -- the counterfactual for 100% of traffic

`decide()` is a pure function of findings, config and layer. So a candidate
policy can be evaluated against findings that have *already* been computed,
instead of splitting traffic and scanning twice.

```python
from charterguard import ShadowPolicy

shadow = ShadowPolicy(baseline=cfg, candidates={"stricter": GuardrailConfig.strict()})
verdict = pipeline.check_input(text)      # one scan, behaviour unchanged
shadow.observe(verdict)
...
print(shadow.report()["stricter"].summary())
# stricter: 97.4% agreement over 40213 verdicts, +812 newly blocked, +0 newly allowed
```

This is strictly better than a 50/50 A/B split for *policy* changes: no user is
exposed to the untested config, and you get the counterfactual for every request
rather than half of them.

Measured cost, because the first draft of this claimed "the cost of a dict
lookup" and a performance test caught the lie:

| Operation | Cost |
|---|---|
| `decide()`, 0 findings | 8.9 µs |
| `decide()`, 1 finding | 16.6 µs |
| `decide()`, 5 findings | 36.2 µs |
| full `check_input`, clean query | 70.9 µs |
| full `check_input`, injection attempt | 42.0 µs |

One candidate therefore costs a quarter to a half of a scan on a verdict that
*has* findings -- real, not free. What makes it cheap in aggregate is a provable
short circuit: a verdict with no findings decides to ALLOW under **every**
config, so clean traffic (the overwhelming majority) skips `decide()` entirely.
A test verifies that across all three bundled configs times every layer, because
that property is the licence for the shortcut.

Note the third row above: the injection attempt scans *faster* than the clean
query, because `stop_at` short-circuits on the first CRITICAL finding. The
expensive path is benign traffic, which is the right way round for DoS
resistance.

Divergences are counted directionally -- `would_newly_block` versus
`would_newly_allow` -- because the two are not symmetric. Newly blocked traffic
is a support ticket; newly allowed traffic is an incident. A single "divergence
rate" hides that distinction, so this class refuses to report one.

`ShadowPolicy` records counts and category names only, never request text.
Storing the prompts that diverged would build precisely the corpus of sensitive
content the component exists to keep out of logs.

**Scope limit.** This evaluates *policy* only. A new rule, a changed normalizer,
or a model tier that was not consulted changes the findings themselves, and
findings are the input here. For those, run the real pipeline twice offline over
a corpus.
