# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately to the owning team (internal:
`#agent-platform-security`; external: the address in `pyproject.toml`). Please do
not open a public issue for an unpatched detection bypass.

Include a minimal reproducing input, the version and `POLICY_VERSION`, the
`Verdict` you received, and the one you expected. A failing test against
`tests/corpus/attacks.jsonl` is the fastest possible report.

Target response: acknowledgement within 2 business days, triage within 5. Bypass
reports are triaged as follows.

| Class | Example | Severity |
| --- | --- | --- |
| Enforcement bypass | input that should BLOCK returns ALLOW at the same config | High |
| Redaction failure | a credential survives `check_output` | High |
| Scanner DoS | input that makes a scan superlinear in length | High |
| Audit integrity | a decision that produces no record, or a forgeable chain | High |
| Coverage gap | a novel attack family with no rule | Medium |
| False positive | benign text blocked | Medium |

## What this component is, and is not

It is a **detection and enforcement layer**. It reduces the probability that
injected instructions are followed and that sensitive data leaves the process.

It is **not containment**. If an injection succeeds anyway, what bounds the
damage is least-privilege tool scopes, row-level authorization at the data layer,
egress allowlisting, and human approval for irreversible actions. Deploy this
alongside those controls, never instead of them.

## Threat model

**In scope.** Direct and indirect prompt injection; instruction override in
retrieved documents, tool results, and file contents; jailbreak framings;
delimiter and chat-template forgery; unicode confusable, invisible-character,
bidi and diacritic evasion; base64/hex/percent-encoded payloads; secret and PII
egress; destructive command generation; markdown/URL exfiltration beacons;
cumulative slow exfiltration across turns; scope violations against a charter.

**Out of scope.** Model weights and training-data attacks; the security of the
tools the agent calls; transport and secret storage; authentication and
authorization of the calling principal; adversarial ML attacks against an
attached classifier; anything requiring the model's internal state.

## Known limitations

Each of these is a deliberate, documented boundary rather than an oversight.

1. **Deterministic rules do not generalize.** Paraphrase and novel phrasing
   evade regex by construction. `ModelTier` with `escalate_clean=True` is the
   mitigation, and it requires you to attach a classifier: no weights ship here.
2. **No bundled semantic model.** `ModelScorer` is a protocol, not an
   implementation. Without an attached scorer, semantic detection does not exist
   in this package.
3. **Free-text PII needs NER.** Structured PII (email, card, national IDs) is
   covered by checksum-gated rules. A name or health fact in prose is not. Wire
   a recognizer through `agent_guardrails.pii.EntityRecognizer`.
4. **Multilingual coverage is seed-level.** The bundled pack covers common
   override phrasings in about a dozen languages. Full multilingual detection
   requires a multilingual classifier.
5. **Confusable folding is not the full Unicode confusables set.** An explicit
   table covers the high-frequency scripts, NFKD covers the accented and
   width/style long tail, and combining marks in U+0300-U+036F are dropped. A
   determined attacker can find a lookalike outside that coverage.
6. **Grounding is heuristic.** `GroundingChecker` compares claim tokens and
   numbers against sources. It detects unsupported specifics; it is not an
   entailment model and will not catch a fluent unsupported paraphrase.
7. **The session ledger is process-local by default.** N replicas enforce
   roughly N times the configured ceiling. Implement `CounterStore` against
   Redis for a real shared ceiling; `SessionLedger.is_shared` reports which mode
   you are in.
8. **Truncation is sampled.** Documents above `budget.max_chars` are scanned
   head and tail; the verdict reports `truncated=True` and a `struct.oversized`
   finding. Treat oversized untrusted documents as lower trust.
9. **Decode depth is bounded.** Nesting beyond `budget.decode_depth` is reported
   as suspicious rather than decoded.
10. **The embedding tier is not adversarially robust.** Text can be crafted to
    preserve harmful meaning while moving its embedding away from your
    prototypes. It also still requires a curated seed set per category — a softer
    version of the enumeration problem, not an escape from it — and threshold
    calibration against a false-positive budget. Treat it as a recall amplifier
    over rules, never as a verdict. The bundled `HashingEmbedder` is lexical and
    measured at 5% zero-FP recall on JailbreakBench harmful goals; it is a test
    double and a floor, not semantic detection.
11. **Streaming trades latency for enforcement.** `StreamGuard` holds back
    `holdback_chars` so a boundary-straddling pattern is caught before release.
    Without hold-back, a streamed secret is already gone.

## Failure policy

Set `ModelTier.fail_closed=True` for regulated deployments: a classifier timeout
then produces a CRITICAL finding rather than an implicit allow. The default is
fail-open, which is correct for productivity agents and wrong for anything
handling regulated data. Choose explicitly — the default is not a recommendation.

## Hardening checklist

Before enabling enforcement on production traffic:

- [ ] `GuardrailConfig.strict()` (or per-category `Mode.BLOCK`) is set.
- [ ] `ModelTier.fail_closed=True` if the agent touches regulated data.
- [ ] A `ModelScorer` is attached, or you have accepted rule-only coverage.
- [ ] A shared `CounterStore` backs the ledger if you run more than one worker
      (`SqliteCounterStore` for one host, `RedisCounterStore` across hosts).
      With the in-memory default, N workers enforce N x the intended ceiling.
- [ ] If a model is loaded locally, its `revision` is a commit SHA and every
      file has a pinned SHA-256 (see "Model supply chain" below).
- [ ] An `AuditSink` is wired, and its retention matches your policy.
- [ ] Metrics are exported and a **block-rate drop** alert exists -- a guardrail
      that silently stops firing looks exactly like a quiet week.
- [ ] Redaction placeholders are reviewed: they carry a digest, not the value.
- [ ] `python tests/run_tests.py` and `python tools/rule_coverage.py` pass in CI.
- [ ] Streamed responses go through `guard_output_stream` /
      `guard_message_stream`, not `guard_output` -- a whole-response guard on a
      stream observes rather than enforces, and buffers the response besides.
- [ ] Irreversible tool calls require a human: wire `approval_node` (LangGraph)
      or an equivalent gate. Guardrails lower the probability of a bad action;
      only approval and least privilege bound its cost.
- [ ] An external attack corpus has been run via `GUARDRAILS_ATTACK_CORPUS`.
- [ ] The rule pack has an owner and a review cadence. Detection decays.

## Model supply chain

The optional local model tier (`agent_guardrails[onnx]`) is deliberately narrow,
because "load any model" is a vulnerability rather than a feature.

**Refused, by name, before any byte is fetched:** `.bin`, `.pt`, `.pth`,
`.ckpt`, `.pkl`, `.joblib`, `.h5`. These are pickle or pickle-equivalent formats
and deserialising one executes arbitrary code. A component that owns the
block/allow decision must not be the thing that runs an untrusted checkpoint.

**Required for every model:**

* a full 40-hex commit SHA as `revision` -- branch and tag names re-point
  silently, which turns "the model we validated" into "whatever was pushed last
  night";
* a SHA-256 for every file, verified on download **and** on every load from
  cache, with mismatches deleting the file and raising rather than proceeding;
* an explicit `label_map`, because output labels cannot be mapped to guardrail
  categories by inference and a wrong guess yields a detector that reports the
  wrong category or nothing at all;
* operator-configured repo IDs. Never build a model spec from request data.

Downloads are verify-then-rename, so a truncated or tampered fetch never appears
at the final path. `allow_download=False` supports air-gapped deployments.
Execution is restricted to `CPUExecutionProvider` with no custom-op libraries
registered, since an ONNX graph can reference custom operators.

**What this does not give you.** A pinned digest proves byte-for-byte
reproducibility. It says nothing about model quality or behaviour. Evaluate any
model on your own corpus, record the revision you evaluated, and re-evaluate
whenever you re-pin. Weight provenance remains a trust decision about the
publisher.

**Persistence scope.** `SqliteCounterStore` and `SqliteAuditSink` are correct
across processes on **one host** and are not supported on NFS/EFS, where
sqlite's locking is unreliable. Neither audit backend is tamper-proof: anyone
who can write the file can rewrite history. The hash chain makes edits
*detectable* by an auditor holding an external copy of a later hash.

**Third-party egress.** `HttpModelScorer`, `HttpIntentClassifier` and
`HttpEntityRecognizer` send prompt or document content to whatever endpoint you
configure -- often exactly the PII the guardrail exists to protect. `https` is
required unless `allow_insecure_http=True` is passed explicitly, and credential
headers are never sent over plaintext. Where data residency is constrained, use
the in-process spaCy/Presidio or ONNX adapters instead.

## Reproducible verification

```bash
python tests/run_tests.py                    # 166 tests, no dependencies
python tools/rule_coverage.py --json         # which rules the corpus exercises
python benchmarks/bench.py                   # latency and throughput
GUARDRAILS_ATTACK_CORPUS=/path/to.jsonl python tests/run_tests.py
GUARDRAILS_HARM_CORPUS=/path/to.jsonl python benchmarks/semantic_bench.py
```
