"""Corpus evaluation: recall on attack families, and false positives on benign.

Numbers here are a regression signal, not a benchmark score. The corpus is small
and was written alongside the rules, so it measures "did a change break a family
we already handled". `tests/corpus/README.md` explains how to point this at an
external public corpus, which is where a defensible number comes from.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from charterguard import GuardrailPipeline
from charterguard.types import Action

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def _load(name: str) -> list[dict]:
    """Bundled corpus by default; an external file when one is configured."""
    path: Path | None = None
    if name == "attacks":
        override = os.environ.get("GUARDRAILS_ATTACK_CORPUS", "").strip()
        if override:
            candidate = Path(override)
            if candidate.is_file():
                path = candidate
    if path is None:
        path = CORPUS_DIR / f"{name}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _detected(pipeline: GuardrailPipeline, text: str) -> bool:
    """Detection at any boundary counts: these are the layers an attack traverses."""
    if pipeline.check_input(text).action is not Action.ALLOW:
        return True
    result = pipeline.inspect_untrusted(text)
    return result.quarantined or bool(result.verdict.findings)


def test_attack_corpus_recall():
    p = GuardrailPipeline()
    attacks = _load("attacks")
    missed = [a for a in attacks if not _detected(p, a["text"])]
    recall = 1 - len(missed) / len(attacks)
    assert recall >= 0.90, (
        f"recall {recall:.0%} (missed: {[m['family'] for m in missed]})"
    )


def test_no_attack_family_is_entirely_missed():
    """Aggregate recall can hide a whole family regressing to zero."""
    p = GuardrailPipeline()
    by_family: dict[str, list[bool]] = {}
    for item in _load("attacks"):
        by_family.setdefault(item["family"], []).append(_detected(p, item["text"]))
    dead = [family for family, hits in by_family.items() if not any(hits)]
    assert not dead, f"families with zero detection: {dead}"


def test_benign_corpus_has_no_false_blocks():
    """Trigger-adjacent vocabulary ("ignore the noise", "previous instructions
    to the vendor") must not block: a guardrail that blocks real work is turned
    off by the people it protects."""
    p = GuardrailPipeline()
    blocked = [b["text"] for b in _load("benign") if p.check_input(b["text"]).blocked]
    assert not blocked, f"false positives: {blocked}"


def test_benign_false_flag_rate_is_low():
    p = GuardrailPipeline()
    benign = _load("benign")
    flagged = [b["text"] for b in benign if p.check_input(b["text"]).action is not Action.ALLOW]
    assert len(flagged) / len(benign) <= 0.15, f"flag rate too high: {flagged}"
