# Evaluation corpora

`attacks.jsonl` and `benign.jsonl` are a **bundled smoke corpus**, hand-written to
cover one case per known attack family (direct override, prompt leak, role-play,
fake system framing, delimiter escape, homoglyph, invisible characters, bidi,
diacritics, nine languages, encoded payloads, exfiltration, tool abuse) plus
benign text that deliberately contains trigger-adjacent vocabulary.

**This is not a public benchmark and its recall number is not a benchmark score.**
It is small, it was written by the same people who wrote the rules, and it
therefore measures "did a regression break a family we already handle", not
"how does this compare to published detectors".

For a defensible number, run an external corpus. `test_corpus.py` reads any
JSONL with a `text` field, so an external set can be dropped in without code
changes:

    GUARDRAILS_ATTACK_CORPUS=/path/to/external.jsonl python tests/run_tests.py

Public sets worth running before a production rollout, none of which are vendored
here because their licences and distribution terms differ:

  * JailbreakBench (JBB-Behaviors)
  * HarmBench
  * Lakera Gandalf / PINT prompt-injection sets
  * AdvBench harmful-behaviours split
  * garak probe suites (NVIDIA)

Expect deterministic rules to score well on template-derived families and poorly
on paraphrase-heavy ones. That gap is the model tier's job, not the rule tier's.
