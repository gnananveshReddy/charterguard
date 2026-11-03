# How it compares

Guardrail tools take different approaches, and most production stacks combine
more than one. This page describes **approaches**, not feature checklists:
projects change quickly, so check each one's current docs before you choose.
Corrections are welcome as a PR.

## Where charterguard sits

charterguard is a **deterministic, in-process first layer** for agent
harnesses. It runs in microseconds with zero runtime dependencies, makes every
decision in code with a typed, auditable `Verdict`, and is built around trust
boundaries: user input, untrusted retrieved content, tool calls, tool output
and the final answer. A classifier is optional, plugged in through
`ModelScorer` (for example the `.[onnx]` tier), and called only when the rules
are uncertain.

It is **not** a moderation model and not a semantic detector. On public
prompt-injection sets its rules are precise but miss most paraphrased attacks
(see [Detection accuracy](accuracy.md)). Closing that gap is the model tier's job.

## Approaches

| Approach | Examples | Strengths | Costs |
| --- | --- | --- | --- |
| Deterministic rules plus trust-boundary plumbing | **charterguard** | Microsecond latency, no dependencies, explainable rule ids, redaction with exact spans, envelopes and quarantine for untrusted content, session ceilings, audit chain | Misses paraphrase and novel phrasing without a classifier |
| Classifier-based scanners | LLM Guard (Protect AI), Meta Prompt Guard / Llama Guard models | Generalize beyond known phrasings | Model download and runtime (often a transformer stack), per-call inference latency, a score to threshold instead of a rule to read |
| Programmable dialogue rails | NVIDIA NeMo Guardrails | Expressive flows (Colang), topical and dialogue control | Many rails call an LLM, which adds latency and cost; a larger framework to adopt |
| Output validation frameworks | Guardrails AI | Structured-output validation, a hub of validators | Focused on validating outputs rather than untrusted-content boundaries |
| Hosted detection APIs | Lakera Guard and similar | No model to operate; vendor-maintained detection | Network round trip; text leaves your process; commercial terms |
| LLM-as-judge or canary approaches | Rebuff (prototype; check its maintenance status) | Can catch semantic attacks | An LLM call per check, plus its own injection surface |

## Using them together

A common layering:

1. **charterguard first**, on every boundary: it's cheap enough to run on every
   retrieved chunk and tool result, and it blocks known attack shapes, secrets and
   structured PII deterministically.
2. **A classifier as the escalation tier**, attached through `ModelScorer` and
   called only in the uncertain band (`ModelTier` escalation thresholds), so
   inference cost is paid on a small fraction of traffic.
3. **Containment**, whatever detection you use: least-privilege tools,
   per-session ceilings and human approval for irreversible actions. Detection
   lowers the chance that an injected instruction is followed; containment limits
   the damage when one is.
