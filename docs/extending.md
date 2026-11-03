# Extending it

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
