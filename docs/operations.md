# Operating it

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
