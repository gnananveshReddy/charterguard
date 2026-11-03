# Security assumptions and limits

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
  `i.g.n.o.r.e`), hyphenated line breaks, quote or emphasis marks wrapping a
  word (`your "system" prompt`) and rot13 before re-running the injection rules. Letter spacing needs some other separator between words;
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

See [SECURITY.md](https://github.com/gnananveshReddy/charterguard/blob/main/SECURITY.md) for the threat model, the full limitations list,
and a pre-production hardening checklist.
