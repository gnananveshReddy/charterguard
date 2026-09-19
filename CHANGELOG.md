# Changelog

All notable changes to this component. Versions follow semantic versioning:
a major bump means a verdict that used to ALLOW may now BLOCK, or a public
signature changed. Detection-rule additions are minor bumps, because a new rule
changes verdicts on traffic that previously passed -- treat every minor upgrade
as a policy change and shadow it before enforcing (see `ShadowPolicy`).


### Added
- **Async streaming enforcement.** `AsyncStreamGuard` mirrors `StreamGuard` for
  async token streams. Both now share one hold-back state machine, so they
  cannot drift; a test asserts they release identical text for identical input.
- **Streaming through the framework adapters.** `guard_output_stream` (LangChain,
  a `RunnableGenerator`) and `guard_message_stream` (LangGraph, for
  `stream_mode="messages"`). Previously the only output guard was a
  `RunnableLambda`, which silently buffered an entire streamed response and gave
  no incremental enforcement.
- **Human approval for blocked actions.** `approval_node` converts a guardrail
  block into a durable LangGraph `interrupt()`, so irreversible actions can be
  escalated to a reviewer instead of dead-ending. Fails closed: an unrecognised
  resume value is a rejection. `interrupt_fn` is injectable, so it is testable
  and usable without langgraph.
- `coalesce_chars` on both stream guards: rescan once per N new characters
  instead of once per token. Every scan re-reads the whole buffer (that is what
  catches boundary-spanning patterns), so per-token scanning made cost grow with
  response length. Measured on a 512-token response: **166x** faster at
  `coalesce_chars=32`, with identical output. Nothing is released before it is
  scanned, so this trades release granularity, never safety. The adapter stream
  guards default to 32; core `StreamGuard` stays at 0 for compatibility.
- `StreamGuard` and `AsyncStreamGuard` accept `session_id` and
  `allowed_citations`, so streamed output is charged to the right session ledger
  and can be grounding-checked. Previously a stream charged no volume at all.
- `py.typed`, `LICENSE`, `CHANGELOG.md`, and GitHub Actions CI.
- CI runs the suite **with optional dependencies installed** (fastapi,
  langchain, redis, spacy, onnx matrix legs) in addition to the dependency-free
  leg. The adapter code paths were previously never executed anywhere.
- `benchmarks/adapter_bench.py` covers the framework adapters: per-node guard
  overhead and streaming hold-back cost.

### Changed
- `pyproject.toml` declares `license = { file = "LICENSE" }` instead of a bare
  `text = "Proprietary"` with no file behind it.

### Fixed
- Async generators in the streaming path are now closed deterministically. A
  blocked or abandoned stream previously left the upstream generator for the
  garbage collector, which finalises it on an arbitrary later event loop and logs
  "Task was destroyed but it is pending".
- Removed 16 unused imports across the core, tools and benchmarks.



### Added
- **LangChain adapter** (`integrations.langchain`): `guard_input`,
  `guard_output`, `guard_retrieval`, `GuardedRetriever`, `guard_tool`,
  `guarded_tools`. Requires `langchain-core`.
- **LangGraph adapter** (`integrations.langgraph`): guard nodes for input,
  output, tool calls, tool results and retrieval, plus `guardrail_router`,
  `refusal_node`, `GuardrailState` and `thread_session_id`. Imports no
  third-party package: nodes are plain callables over state.
- Guard decisions shared between both adapters, so a chain and a graph
  quarantine on identical rules and emit identical audit rows.
- `langchain` and `langgraph` extras.

### Notes
- A blocked tool call answers *every* `tool_call_id`, because the chat
  completions contract requires one result per call and a stranded id is a
  provider error on the next turn.
- Guarded state is JSON-serialisable, because checkpointers persist it.



### Added
- `RuleHealth` -- per-rule fire/precision telemetry to retire rules that do not
  earn their place.
- `ShadowPolicy` -- counterfactual evaluation of a policy change over 100% of
  traffic, reported directionally (`would_newly_block` vs `would_newly_allow`)
  because the two are not symmetric.
- Inverted scanner dispatch and prefilter equivalence tests.



- Core pipeline: input, scope, untrusted/retrieval, tool-call, tool-output and
  output layers; normalization with offset mapping; injection, jailbreak,
  encoded-payload, structural, secret and PII detectors; grounding validation;
  session volume ceilings; redaction with digest placeholders; provenance
  envelopes; streaming output guard; audit chain; observability.
- Optional adapters: FastAPI, redis, prometheus, otel, spacy, presidio, onnx,
  plus stdlib-only metrics, sqlite ledger/audit, and HTTP model/intent/NER.
