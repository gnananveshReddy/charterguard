# Development

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

## Contributing

Missed attacks and false positives are the most useful contributions; the issue
templates ask for exactly what is needed to fix them. See
[CONTRIBUTING.md](https://github.com/gnananveshReddy/charterguard/blob/main/CONTRIBUTING.md) for setup and the rule-change checklist, and
[SECURITY.md](https://github.com/gnananveshReddy/charterguard/blob/main/SECURITY.md) for reporting bypasses privately.
