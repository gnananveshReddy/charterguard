## What and why

<!-- One concern per PR. Link the issue it closes. -->

## Checklist

- [ ] Tests that fail before this change and pass after it
- [ ] Rule changes: attack phrasings **and** benign look-alikes added (`tests/corpus/*.jsonl` or a focused test)
- [ ] Rule changes: `python benchmarks/accuracy.py` before/after tables pasted below
- [ ] No reports, logs or exceptions contain matched secret / PII text
- [ ] New third-party imports are behind an optional extra and imported lazily
- [ ] `ruff check .`, `mypy src/charterguard`, `python tests/run_tests.py` pass
- [ ] `CHANGELOG.md` entry under `[Unreleased]`

## Accuracy (rule changes only)

<!-- before / after output of benchmarks/accuracy.py -->
