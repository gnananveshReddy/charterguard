#!/usr/bin/env python3
"""Rule coverage: which rules does the corpus actually exercise?

Line coverage tells you a module was imported. For a detection component the
question that matters is different: *is every rule reachable, and does anything
in our corpus fire it?* A rule that never fires is either dead (a pattern bug,
shadowed by a broader rule, or unreachable after normalization) or untested --
both are silent failures that line coverage reports as green.

Usage:

    python tools/rule_coverage.py                     # bundled corpus
    python tools/rule_coverage.py path/to/corpus.jsonl # external corpus
    python tools/rule_coverage.py --json               # machine-readable

Pass `--fail-under=PCT` to gate CI. Without it the command only reports, because
a small corpus legitimately leaves niche rules unexercised and a hard gate would
just train people to ignore the output.

Short-circuiting is disabled for attribution: in production the scan stops at the
first CRITICAL finding (correctly -- the verdict cannot change), but that hides
every other rule the sample would have matched.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_guardrails import GuardrailPipeline  # noqa: E402
from agent_guardrails.patterns import RULES  # noqa: E402
from agent_guardrails.types import Layer  # noqa: E402

CORPORA = (
    ROOT / "tests" / "corpus" / "attacks.jsonl",
    ROOT / "tests" / "corpus" / "benign.jsonl",
)


def load(paths) -> list[str]:
    out: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line)["text"])
    return out


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    fail_under = next(
        (float(a.split("=", 1)[1]) for a in argv if a.startswith("--fail-under=")), None
    )
    args = [a for a in argv if not a.startswith("--")]
    texts = load(args or CORPORA)
    if not texts:
        print("no corpus samples found", file=sys.stderr)
        return 2

    from dataclasses import replace as _replace

    from agent_guardrails.config import GuardrailConfig

    pipeline = GuardrailPipeline(
        _replace(GuardrailConfig(), short_circuit_on_critical=False)
    )
    hits: Counter[str] = Counter()
    for text in texts:
        for layer_fn in (pipeline.check_input, pipeline.check_output):
            for finding in layer_fn(text).findings:
                if finding.rule_id:
                    hits[finding.rule_id] += 1
        for finding in pipeline.inspect_untrusted(text).verdict.findings:
            if finding.rule_id:
                hits[finding.rule_id] += 1

    all_ids = [r.id for r in RULES]
    unexercised = sorted(set(all_ids) - set(hits))
    covered = len(all_ids) - len(unexercised)
    report = {
        "samples": len(texts),
        "rules_total": len(all_ids),
        "rules_exercised": covered,
        "coverage_pct": round(100.0 * covered / len(all_ids), 1),
        "unexercised": unexercised,
        "top_hits": hits.most_common(10),
    }
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"samples          : {report['samples']}")
        print(f"rules            : {report['rules_total']}")
        print(f"exercised        : {report['rules_exercised']} ({report['coverage_pct']}%)")
        print(f"unexercised      : {len(unexercised)}")
        for rule_id in unexercised:
            print(f"  - {rule_id}")
    if fail_under is not None and report["coverage_pct"] < fail_under:
        print(f"FAIL: coverage {report['coverage_pct']}% < {fail_under}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
