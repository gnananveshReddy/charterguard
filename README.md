# charterguard

A framework-agnostic guardrail pipeline for LLM agent harnesses: prompt-injection
and jailbreak detection, harmful-content and destructive-command rules, scope
enforcement against an agent charter, untrusted-content envelopes, tool-output
inspection, secret and PII redaction, grounding checks, session volume ceilings,
and audit-ready verdicts.

Zero runtime dependencies. Python 3.11+. FastAPI is an optional integration, not
a requirement.

## Why this exists

Two ideas drive the whole design.

**Controls do not live in prompts.** A system prompt is a request, not a control.
Every decision here is made in code, outside the model, and returns a typed
`Verdict` that can be logged, replayed, and audited.

**Everything the model did not author is untrusted.** Retrieved documents, tool
results, web pages, and file contents are inspected before they reach the context
window, wrapped in a delimiter-hardened envelope that marks them as data, and
quarantined if they carry instructions.

## Install

Copy `src/charterguard` into your project, or install the package:

```bash
pip install -e .                 # core + every stdlib adapter, no dependencies
pip install -e ".[fastapi]"      # ASGI middleware
pip install -e ".[langchain]"    # LangChain runnables, guarded tools + retriever
pip install -e ".[langgraph]"    # LangGraph nodes (the adapter needs no import)
pip install -e ".[redis]"        # session ceilings that hold across hosts
pip install -e ".[prometheus]"   # or .[otel] -- metrics egress
pip install -e ".[spacy]"        # or .[presidio] -- free-text PII (names)
pip install -e ".[onnx]"         # local model tier: onnxruntime, no torch
```

Every extra is genuinely optional and none is imported by the core. All six
extension points have a working standard-library option -- metrics to
logging/console/StatsD, session ceilings and audit to sqlite, model/intent/NER
over plain HTTP. See [docs/ADAPTERS.md](docs/ADAPTERS.md) for the full matrix
with measured per-call costs.

## Quick start

```python
from charterguard import GuardrailPipeline, GuardrailConfig, Charter

charter = Charter(
    agent_id="policy-assistant",
    version="3",
    purpose="Answer questions about internal policy documents.",
    in_scope=("policy", "standard", "control", "benefit"),
    out_of_scope=("legal advice", "compensation decisions"),
    allowed_tools=("search_docs", "get_policy"),
)

guardrails = GuardrailPipeline(GuardrailConfig.report_only(), charter=charter)

# 1. Inbound turn
verdict = guardrails.check_input(user_message, session_id=session_id)
if verdict.blocked:
    return guardrails.refusal(verdict)

# 2. Retrieved content, before it touches the context window
for chunk in retrieved:
    result = guardrails.inspect_untrusted(chunk.text, source=chunk.uri, kind="retrieval")
    if result.safe_to_use:
        context.append(result.context_block())

# 3. Tool calls
call_verdict = guardrails.check_tool_call("search_docs", arguments, session_id=session_id)

# 4. Tool results
tool_result = guardrails.inspect_tool_output(raw_output, tool_name="search_docs")

# 5. Outbound answer
out = guardrails.check_output(answer, allowed_citations=[c.uri for c in retrieved])
final_text = guardrails.refusal(out) if out.blocked else (out.sanitized or answer)
```

## Trust boundaries

| Boundary | Call | What it is for |
| --- | --- | --- |
| Inbound turn | `check_input` | Injection, jailbreak, harmful intent, destructive commands |
| Scope | `check_scope` | Charter adherence, out-of-scope requests |
| Tool invocation | `check_tool_call` | Tool allow-list, argument inspection, call ceilings |
| Untrusted content | `inspect_untrusted` | Retrieved or fetched content, envelope wrapping, quarantine |
| Tool output | `inspect_tool_output` | Same as above, plus secret and PII redaction before context insertion |
| Outbound answer | `check_output` | Secret and PII redaction, grounding, cumulative volume |

`acheck_input` and `acheck_scope` are async variants for the model-assisted tier.
`check_input_or_raise` raises `GuardrailBlocked` instead of returning a verdict.

## FastAPI integration

```python
from fastapi import Depends, FastAPI
from charterguard import GuardrailPipeline, Verdict
from charterguard.integrations.fastapi import (
    guard_response,
    input_guard,
    install_exception_handler,
)

app = FastAPI()
pipeline = GuardrailPipeline()
install_exception_handler(app)          # renders GuardrailBlocked as HTTP 422 + audit
guard = input_guard(pipeline, enforce_scope=True)

@app.post("/turn")
async def turn(body: dict, verdict: Verdict = Depends(guard)):
    answer = await run_agent(body["message"])
    text, out_verdict = guard_response(pipeline, answer)
    return {"answer": text, "policy_id": out_verdict.policy_id}
```

The verdict is injected into the handler even when nothing is blocked, so
handlers can log FLAG outcomes during a phased rollout.

Construction is cheap (~150 us) and compiled rule sets are shared process-wide,
so building a pipeline per request in a dependency is fine. Scanning is pure and
thread-safe; only the session ledger holds mutable state, behind a lock.

## LangChain and LangGraph

Both frameworks are supported, and the guard decisions are shared between them,
so a graph and a chain quarantine on the same rules and emit the same audit rows.

```python
# LangChain: guards compose with `|`
from charterguard.integrations.langchain import guard_input, guard_output
chain = guard_input(pipeline) | prompt | model | guard_output(pipeline)

# LangGraph: guards are nodes, and the router enforces
from charterguard.integrations.langgraph import (
    guard_input_node, guardrail_router, refusal_node,
)
g.add_node("guard_in", guard_input_node(pipeline))
g.add_conditional_edges("guard_in", guardrail_router(ok="model", blocked="refuse"))
```

Streaming needs the streaming guard, not `guard_output`: a `RunnableLambda`
buffers the whole response before it can return, so a streamed chain silently
loses streaming and gets no incremental enforcement. Use
`guard_output_stream` (LangChain) or `guard_message_stream` (LangGraph); both
enforce as tokens arrive via a hold-back window, so a secret split across two
chunks is caught before any of it is emitted.

Blocked actions do not have to dead-end. `approval_node` converts a block into a
durable LangGraph `interrupt()`, so an irreversible action can be escalated to a
human reviewer and resumed with `Command(resume=...)`. It fails closed: an
unrecognised resume value is a rejection.

The LangGraph adapter imports **no third-party package**: nodes are plain
callables over state, so they work across langgraph versions and are tested with
the framework absent. Guards write a JSON-serialisable record to state (a
`Verdict` in state would break checkpointers) and the router decides, which lets
you ship in observe-only mode first.

Two contract details the adapters get right, and hand-rolled versions usually do
not: a blocked tool call still answers *every* `tool_call_id` (a stranded id is a
provider 400 on the next turn), and a quarantined tool result is neutralised
in place rather than deleted, without refusing the turn. Poisoned *retrieved
documents* are dropped instead, since they carry no id awaiting an answer.

See [docs/ADAPTERS.md](docs/ADAPTERS.md#langgraph) for the full wiring, the
`safe_output`-versus-history split, and the guarded retriever and tool wrappers.


## Untrusted content and envelopes

`inspect_untrusted` returns an `UntrustedResult`:

* `verdict` — the typed decision, with findings and audit payload.
* `envelope` — the text to actually append to the context window, wrapped in
  delimiters that declare it as data and carry a nonce so injected content cannot
  forge a boundary.
* `context_block()` — the envelope, or a neutral placeholder note when the
  content was quarantined. It never returns the original text for quarantined
  content.
* `safe_to_use` / `quarantined` — whether the content was admitted.

Pair this with `pipeline.system_rules()`, which emits the envelope-handling rules
the model needs, and `verify_prefix()`, which asserts your assembled system
prefix still contains them.

## Rollout modes

Policy is configured per category, and every category has an independent mode:

* `Mode.REPORT` — detect and record, never alter behaviour. Start here.
* `Mode.FLAG` — record and mark the verdict; the caller decides.
* `Mode.BLOCK` — refuse or quarantine.

```python
GuardrailConfig.report_only()                                   # shadow mode
GuardrailConfig()                                               # sensible default
GuardrailConfig.strict()                                        # everything blocking
GuardrailConfig().with_mode(Category.INJECTION, Mode.BLOCK)     # one category
```

Defaults are deliberate: secrets and exfiltration block, PII is redacted rather
than blocked, and restricted topics only report.

**Talking about attacks is not an attack.** On the user's own turn, an
injection phrase quoted in order to discuss it ("Explain 'ignore previous
instructions' attacks") is lowered to MEDIUM, so it flags rather than blocks,
and the bare word "jailbreak" is recorded but never blocks. Retrieved content
and tool output are never dampened -- there, the attacker controls the quotes.
Set `dampen_quoted_mentions=False` to turn this off.

## Audit

`verdict.to_audit()` returns a JSON-safe payload. The audit integration turns
verdicts into storage rows and hash-chained events:

```python
from charterguard.integrations.audit import event_payload, guardrail_rows

rows = guardrail_rows(verdict, run_id=run_id, step=3,
                      principal_id=user_id, agent_identity="policy-assistant@3")
payload, row_hash = event_payload(verdict, prev_hash=prev_hash)
```

Redaction keeps a short digest of what it removed, so an auditor can prove a
credential passed through a given step without the credential being stored.

## Performance

The scan engine is built around one observation: Python's `re` has no
multi-pattern automaton, so a 40-branch alternation costs roughly 1.5 us per
character — about 9 ms for a 4 KB document, which puts guardrails on the critical
path and effectively forbids scanning retrieved context.

Three changes remove that cost:

* **Single normalization pass.** Confusable folding, invisible-character
  stripping, and case folding happen once, in slices rather than per character,
  with a sparse offset map so audit spans still point at the original bytes.
* **Literal prefilter.** A necessary literal condition is derived statically from
  each rule pattern; one trie pass reports which literals are present, and only
  rules that can possibly match are executed, in windows around the triggering
  literal. See `prefilter.py`.
* **Content-hash cache.** Retrieval returns the same chunks repeatedly; the
  second scan of identical bytes is a dictionary lookup.

Measured with `python benchmarks/bench.py` (CPython 3.11, single core, p50):

| Operation | p50 | p95 |
| --- | --- | --- |
| construct pipeline | 150 us | 178 us |
| normalize 4 KB | 138 us | 254 us |
| `check_input` clean turn | 11 us | 18 us |
| `check_input` attack turn | 12 us | 24 us |
| `inspect_untrusted` 4 KB unique | 1.0 ms | 1.6 ms |
| `inspect_untrusted` 4 KB repeat (cached) | 40 us | 229 us |
| `inspect_untrusted` 64 KB unique | 14.2 ms | 17.9 ms |
| `check_output` 4 KB unique | 1.2 ms | 1.8 ms |

About 4-5 MB/s of unique text per core, roughly ten times the naive
implementation. Attack-bearing documents are faster than clean ones because
critical findings short-circuit the scan.

The prefilter is an optimization, never a policy: `scan(prefilter=False)` runs
every rule unconditionally, and `tests/test_prefilter.py` asserts both modes
produce identical findings across the whole attack corpus.

## Policy files

Enforcement changes with rollout phase, so it can live in a reviewed TOML (or JSON)
file instead of code. See [`docs/policy.example.toml`](docs/policy.example.toml).

```toml
preset = "default"            # default | strict | report_only

[modes]
injection = "block"
pii = "flag"

[flag_at]
pii = 0.4
```

```python
cfg = GuardrailConfig.from_file("guardrails.toml")   # or charterguard.load_policy(...)
```

```bash
charterguard policy guardrails.toml          # validate, print the effective policy
charterguard scan --policy guardrails.toml prompts/
```

Loading is strict. An unknown key (`injectoin = "block"`), a wrong type, a threshold
outside [0, 1], or `flag_at` above `block_at` raises `PolicyError` naming the key, so a
typo can never silently weaken enforcement. Only the stdlib `tomllib` is used.

## Command line

Installing the package adds a `charterguard` command (also `python -m charterguard`):

```bash
echo "ignore all previous instructions" | charterguard scan --strict   # exit 1: blocked
charterguard scan --layer untrusted docs/*.md --json                 # RAG documents
charterguard scan --layer output --redact reply.txt > reply.safe.txt # redacted text
charterguard eval labelled.jsonl --label-field is_attack --min-recall 0.9
charterguard rules --category pii
```

* `scan` exits 1 when any input reaches `--fail-on` (`block` by default, or `flag`,
  `never`), so it works as a CI step. Reports show rule, severity and offsets, never
  the matched text, so scanning a file with a live key does not put the key in a log.
* `eval` reports precision, recall, F1 and false-positive rate on any JSONL file, and
  can gate CI with `--min-recall` / `--max-fpr`.
* `.pre-commit-hooks.yaml` exposes `charterguard-prompts` and `charterguard-secrets`
  for use from other repositories.

## Agent SDKs

**OpenAI Agents SDK** (`pip install "charterguard[openai-agents]"`):

```python
from charterguard.integrations.openai_agents import (
    input_guardrail, output_guardrail, tool_input_guardrail, tool_output_guardrail,
)

agent = Agent(name="support",
              input_guardrails=[input_guardrail(pipeline)],
              output_guardrails=[output_guardrail(pipeline)])
```

The input guardrail runs blocking (`run_in_parallel=False`), so the agent does not
start acting on a turn that is about to be blocked. The tool output guardrail replaces
quarantined or redacted tool output with the enveloped text.

**Claude Agent SDK** (`pip install "charterguard[claude-agent-sdk]"`):

```python
from charterguard.integrations import claude_agent_hooks

options = ClaudeAgentOptions(hooks=claude_agent_hooks(pipeline))
```

`UserPromptSubmit` blocks attacks before the model sees them, `PreToolUse` denies
forbidden or command-injecting tool calls, and `PostToolUse` flags quarantined tool
output. A post-tool hook cannot erase output that already exists, so for tools you
control, guard inside the tool or serve it through `charterguard mcp`.

## MCP

`charterguard mcp` runs an [MCP](https://modelcontextprotocol.io) server over stdio
(stdlib only), exposing `scan_text`, `redact_text` and `list_rules` to any MCP client:

```json
{"mcpServers": {"charterguard": {"command": "charterguard", "args": ["mcp", "--policy", "guardrails.toml"]}}}
```

For the other direction, results from *other* MCP servers are untrusted input. Run
them through the tool-output guard before the model sees them:

```python
from charterguard.integrations import guard_mcp_result

safe = guard_mcp_result(pipeline, result, tool_name="search")
```

It fails closed. Every readable part (text, embedded resources, link descriptions,
`structuredContent`) is scanned as one payload, so an attack split across blocks is
caught. The model gets a single redacted, enveloped text block. Raw
`structuredContent`, unknown or binary blocks and server `_meta` are never forwarded,
media only with `allow_media=True`, and a quarantined result is marked `isError`.

## Detection accuracy

Measured with `python benchmarks/accuracy.py`: rule tiers only, no model scorer
configured, and no rule was fitted to these datasets. "Detected" means the verdict is
flag or block; the block columns use the strict preset (the default preset flags
injection rather than blocking it).

| Dataset | n (attacks) | Precision | Recall | FPR | Block recall (strict) | Block FPR (strict) |
| --- | --- | --- | --- | --- | --- | --- |
| bundled corpus (`tests/corpus`) | 96 (43) | 0.96 | 1.00 | 0.04 | 1.00 | 0.00 |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 662 (263) | 1.00 | 0.13 | 0.00 | 0.12 | 0.00 |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | 1306 (666) | 0.91 | 0.48 | 0.05 | 0.40 | 0.00 |

What these numbers say:

* **The rule tier is precise, not complete.** On the public sets it almost never
  blocks benign text, but it misses most attacks phrased in ways no rule anticipates.
* **deepset/prompt-injections** is roughly half German, and many of its "injections"
  are persona or opinion prompts ("You are Volkswagen. What do you think of
  Mercedes?") that no pattern-based detector should block. The remaining English
  misses are mostly paraphrases ("remove all previous tasks out of your head").
* **jailbreak-classification** is long DAN-style role-play. The rules catch the
  well-known personas and "no restrictions" framing, not novel stories.

That gap is what the classifier tier (`ModelScorer`, escalation bands) is for. Run the
benchmark with your own scorer configured to measure the full cascade.

## Tests

```bash
python tests/run_tests.py     # dependency-free runner, no pytest required
pytest                        # same tests, if pytest is installed
python benchmarks/bench.py
python benchmarks/accuracy.py  # fetches public datasets once, then cached
```

The test suite covers normalization and offset mapping, injection and jailbreak
detection, encoded payloads, obfuscation (leetspeak, letter spacing,
hyphenation, rot13), secret and PII precision (Luhn, Verhoeff, entropy
gates), rule shadowing, scope enforcement, grounding, volume ceilings, prefilter
equivalence, model-tier escalation bands and sampling, the multilingual pack, the
NER seam, streaming enforcement, audit-chain integrity, metrics, the LangChain and
LangGraph adapters, and the pipeline itself.

Two suites are worth calling out because they find real bugs rather than
confirming intent:

* `test_fuzz.py` -- property tests over seeded random unicode: no input raises,
  spans always index the original text, normalization is idempotent, and no rule
  backtracks catastrophically. This caught an O(n^2) banner pattern and a
  case-folding order bug that made normalization non-idempotent.
* `test_langchain_langgraph.py` -- runs with neither framework installed, by
  duck-typing messages and documents. Asserts the contract properties that break
  agents in production: every `tool_call_id` gets a result even when a call is
  refused, guarded state survives `json.dumps` for checkpointers, and an
  in-place message rewrite refuses to run without an `id`. Streaming coverage
  asserts a secret split across chunks never reaches the user at any coalescing
  setting, and that human approval fails closed on an unrecognised resume value.
* `test_corpus.py` -- recall per attack family plus a false-positive budget on
  benign text. This caught five entirely undetected families (interrogative
  prompt leaks, "enable developer mode", forged chat-template tokens, URL
  beacons carrying credential references, and active-content data URIs).

```bash
python tools/rule_coverage.py                 # which rules the corpus exercises
python benchmarks/semantic_bench.py           # cascade efficacy + latency levers
python tools/rule_coverage.py --fail-under=25 # gate CI on it
```

Rule coverage answers a question line coverage cannot: is every rule reachable,
and does anything actually fire it? A rule that never fires is either dead
(shadowed, or broken by normalization) or untested.

## Extending it

* **Rules.** Add `Rule` entries and pass them as `rules=` to `RuleDetector`.
  Patterns must use non-capturing groups only — this is enforced at import time.
  Patterns run against normalized text, so no `(?i)` and no `[A-Za-z]`.
* **Model tier.** Implement the `ModelScorer` protocol (one `async score`
  method) to wrap a classifier, moderation endpoint, or LLM judge. **No model
  ships with this package** -- the protocol is a seam, not an implementation.
  Ready-made adapters: `HttpModelScorer` and `OpenAIModerationScorer` (stdlib
  `urllib`, so no vendor SDK), or `OnnxModelScorer` for a pinned local model.
  Escalation has two bands: the uncertainty band between `escalate_low` and
  `escalate_high`, and -- with `escalate_clean=True` -- the clean band below
  `escalate_low`, which is where a novel attack that trips no rule lands.
  Clean-band coverage is priced per layer, because a user turn is one call per
  request while retrieval is N:

  ```python
  cfg = (
      GuardrailConfig.strict()                      # input: 100% clean-band
      .with_model_tier(Layer.UNTRUSTED,             # retrieval: sample 5%
                       ModelTier(enabled=True, escalate_clean=True,
                                 clean_sample_rate=0.05))
  )
  pipeline = GuardrailPipeline(cfg, model=MyScorer())
  verdict = await pipeline.ainspect_untrusted(chunk)
  ```

  Sampling is keyed on a content hash, not a PRNG, so a given document makes the
  same decision on every replica and retry.
* **Embedding tier (stage 2).** `SemanticScorer` implements the same
  `ModelScorer` protocol, so it drops into the existing cascade with no new
  plumbing: rules (microseconds) -> embeddings (milliseconds) -> LLM judge
  (hundreds of milliseconds, rare). Attach any encoder via `TextEmbedder`
  (batch-first), seed a `PrototypeBank` per category, and wrap an expensive judge
  with `CascadingScorer` to promote only undecided cases.

  ```python
  bank = PrototypeBank(encoder.dim)
  for text in defamation_seeds:
      bank.add_dense(Category.HARMFUL, "defamation", await encoder.embed_one(text))
  fast = SemanticScorer(encoder, bank.finalize())
  pipeline = GuardrailPipeline(cfg, model=CascadingScorer(fast, llm_judge))
  ```

  Measured levers, encoder-independent (`benchmarks/semantic_bench.py`):
  batching 20 retrieval chunks into one forward pass is **6x** faster than 20
  calls; the content-hash embedding cache is **7x** faster again on repeats; and
  with numpy present the prototype match is **~30x** faster than the pure-Python
  inverted index (92 us vs 2.8 ms at 30 prototypes; 3.0 ms vs 88 ms at 3000).
  numpy is used when available and never required.

  The arithmetic that decides whether the tier pays for itself is
  `total = embed + match + promotion_rate x judge`. At a 4 ms encoder and a 300 ms
  judge, a 5% promotion rate costs 19 ms per item (94% saving) and a 100%
  promotion rate costs 304 ms (**1% slower** than just calling the judge). A cheap
  tier that cannot decide is pure overhead — `cascade_breakeven()` computes the
  threshold and `CascadingScorer.promotion_rate` reports the live value.

  The bundled `HashingEmbedder` is a dependency-free character n-gram model. It is
  **lexical, not semantic**: measured on JailbreakBench harmful goals with 30 seed
  prototypes, its zero-false-positive recall on 64 held-out goals was 5%, against
  3% for rules alone. A real sentence encoder is what makes this tier work; the
  fallback exists so the plumbing is testable and benchmarkable with no model
  installed.
* **Free-text PII.** Structured PII is covered by rules. Names and health facts
  in prose are not: that needs NER. Install `.[spacy]` or `.[presidio]` and use
  `SpacyEntityRecognizer` / `PresidioEntityRecognizer`, point
  `HttpEntityRecognizer` at a DLP service, or implement `EntityRecognizer`
  yourself -- findings flow through the same policy, redaction and audit path as
  rule findings. Offsets must index the normalized text they were given;
  misaligned spans mask the wrong characters, which is a leak rather than a
  formatting bug.
* **Session ledger.** The bundled store is process-local, so **N workers enforce
  roughly N times the configured ceiling** -- `SessionLedger.is_shared` tells you
  which you have. `SqliteCounterStore` fixes that across processes on one host
  with no dependency (UPSERT inside `BEGIN IMMEDIATE`; a test proves four
  processes x 250 increments lands exactly 1000). `RedisCounterStore`
  (`.[redis]`) fixes it across hosts in one pipelined round trip.
* **Audit.** `JsonlAuditSink` keeps its chain head in memory, so concurrent
  writers can fork the chain. `SqliteAuditSink` advances the head inside a
  transaction and survives restarts -- three processes writing 25 records each
  verify as one unbroken chain. `verify_chain` accepts either backend.
* **Metrics.** `LoggingMetricsSink` (structured records through stdlib
  `logging`), `ConsoleMetricsSink` (development), `StatsdMetricsSink` (real
  aggregation over UDP, no client library), plus `PrometheusMetricsSink` and
  `OtelMetricsSink` behind extras. Every verdict emits block rate, latency,
  escalation rate and per-rule counters, tagged with low-cardinality labels only
  -- no content, no session ids. `emit()` makes ~4 sink calls per verdict, so an
  unsampled writing logging sink costs ~300 us per verdict: sample it, or use
  StatsD with `max_datagram_bytes=1400` (9.1 us/call vs 38 unbatched).
* **Local model tier.** `.[onnx]` runs any ONNX classifier under a verifiable
  contract: full 40-hex commit SHA (branch names refused), SHA-256 per file
  checked on download *and* every cache load, format allowlist that refuses
  `.bin`/`.pt`/`.pth`/`.ckpt` by name because they are pickle streams that
  execute code on load, and an explicit `label_map` because no code can infer
  that `LABEL_1` means `Category.INJECTION`. No torch, no transformers, no
  bundled weights.
* **Obfuscation.** `ObfuscationDetector` (on by default,
  `enable_obfuscation_scan=False` to disable) re-scans a de-obfuscated view of
  the text, only around the characters it changed, and reports a match only
  when it overlaps a change -- so a plain attack is never double-counted.
  Gates run on a `bytes.translate` shape of the text: ~20 us per 4 KB when
  nothing is obfuscated, about 8% of the rule scan.
* **Streaming.** `StreamGuard(pipeline, holdback_chars=256)` enforces on a token
  stream: text is released only after it has been scanned, so a secret spanning
  two chunks is still redacted and a BLOCK terminates the stream.
* **Async.** The `a*` entry points offload scans above
  `budget.offload_threshold_chars` to a worker thread, so a 64 KB scan does not
  stall the event loop of an async server.
* **Charter.** `Charter.fingerprint()` is stable, so scope decisions can be
  attributed to an exact charter version in the audit trail.

## Security assumptions and limits

* Deterministic rules catch known attack shapes. They are not a substitute for a
  moderation model on adversarial free text; attach the model tier for that.
* Detection is not containment. Guardrails reduce the probability that injected
  instructions are followed; least-privilege tools, row-level authorization, and
  human approval for irreversible actions are what bound the damage when they
  are.
* Documents above `budget.max_chars` are sampled head and tail, not fully
  scanned. The verdict reports `truncated=True` and a `struct.oversized` finding.
* Encoded-payload scanning decodes to `budget.decode_depth`. Deeper nesting is
  reported as suspicious rather than decoded.
* Obfuscation scanning undoes leetspeak, letter spacing (`i g n o r e`,
  `i.g.n.o.r.e`), hyphenated line breaks and rot13 before re-running the
  injection rules. Letter spacing needs some other separator between words;
  uniformly single-spaced letters cannot be segmented without a dictionary.
* PII rules are tuned for false-positive cost, with checksum and entropy gates.
  They cover PII that has *structure*. Free-text PII needs the NER seam.
* Multilingual coverage is seed-level: common override phrasings in about a dozen
  languages, plus native-script override rules for ten Indian languages and
  romanized Hinglish (verb + "instructions" + "previous/all", either order).
  Paraphrase in any language still needs a classifier.
* Confusable folding uses an explicit table for high-frequency scripts plus NFKD
  for the long tail. It is not the complete Unicode confusables set.
* The session ledger is process-local unless you supply a `CounterStore`; N
  replicas then enforce roughly N times the configured ceiling.
* `integrations.audit` shapes and now also *persists* records via `AuditSink`.
  `JsonlAuditSink` gives local durability and a verifiable chain; it is not
  tamper-proof against someone who can write to the file. Point it at an
  append-only store for that.

See [SECURITY.md](SECURITY.md) for the threat model, the full limitations list,
and a pre-production hardening checklist.
