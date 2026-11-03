# Contributing to charterguard

Thanks for helping. The most valuable contributions to a guardrail library are
**evidence**: an attack it misses, or benign text it blocks. Code is welcome too.

## Security issues first

A way to make an input that *should* be blocked come back allowed, a secret that
survives redaction, or an input that makes a scan slow is a **vulnerability**.
Report it privately as described in [SECURITY.md](SECURITY.md), not as an issue.

## Development setup

```bash
git clone https://github.com/gnananveshReddy/charterguard
cd charterguard
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                                  # ruff on every commit
```

The core has **zero runtime dependencies**, and the test suite passes with nothing
installed:

```bash
python tests/run_tests.py     # dependency-free runner
pytest                        # same tests
ruff check .
mypy src/charterguard
```

CI runs the suite bare on Python 3.11–3.13, once per optional extra, and with every
extra installed, to show that imports stay lazy.

## Reporting a missed attack or a false positive

Use the issue templates. The fastest path to a fix is a pull request that adds the
case to the corpus:

* missed attack: a line in `tests/corpus/attacks.jsonl`,
  `{"family": "<family>", "text": "..."}`
* false positive: a line in `tests/corpus/benign.jsonl`

`tests/test_corpus.py` fails until the rules handle it, which makes the fix
reviewable.

## Changing or adding a rule

Rules live in `src/charterguard/patterns.py`. Every rule change needs:

1. **A test that fails before and passes after.** Put attack phrasings in the
   attack corpus or a focused test, and add the benign look-alikes you thought of
   to `benign.jsonl`. A rule without a look-alike test is a future false positive.
2. **Prefilter soundness.** `tests/test_prefilter.py` scans with the literal
   prefilter on and off and requires identical results. Keep every gap bounded
   (`\S{0,16}`, not `\S*`) so no match is wider than the prefilter window.
3. **No catastrophic backtracking.** `tests/test_fuzz.py` checks scan time on
   adversarial inputs. Avoid nested quantifiers over overlapping classes.
4. **Severity that matches confidence.** CRITICAL short-circuits the scan and
   blocks at default thresholds; reserve it for phrasings with no benign reading.
5. A `CHANGELOG.md` entry under `[Unreleased]`.

Run `python benchmarks/accuracy.py` before and after a rule change, and include
both tables in the PR. A precision drop on the public sets needs a justification.

## Contributing a rule pack

Rules for one domain (a product, an industry, a regulation) belong in a pack,
not in the core rule set. To propose a bundled pack:

1. Add `src/charterguard/rulepacks/<name>.toml` (see the README's "Rule packs"
   section for the format, and `agent-tools.toml` as a worked example).
2. Give every rule at least two `examples` and at least one `counterexample`
   that a careless pattern would match. The examples are executed every time the
   pack loads.
3. Run `charterguard pack check <name>` and the test suite. `tests/test_rulepack.py`
   loads every bundled pack.
4. Scope `layers` tightly. Most agent-facing rules belong on `untrusted` and
   `tool_output` only, so users can still ask about the topic.

## Non-English rules

Native speakers are especially welcome. Indian-language vocabulary lives in
`src/charterguard/_indic.py` as plain word lists (qualifiers, nouns, verbs,
negations), so no regex is needed to extend it. Please include benign sentences
that use the same words ("follow all previous instructions"), and any negated
forms.

## Pull requests

* Keep a PR to one concern; stacked PRs are fine.
* Lint, types and tests must pass; CI enforces all three.
* New public API needs type hints, a docstring, and a README or
  `docs/ADAPTERS.md` mention.
* Adapters import their framework lazily, inside the factory, and must be
  testable without it.
* Reports, logs and exceptions must never contain matched secret or PII text.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
