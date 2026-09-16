"""Findings -> action. The only place an enforcement decision is made.

Separated from detection on purpose: detection is a security concern that
changes with the threat landscape, enforcement is a product concern that changes
with rollout phase. Keeping them apart is what makes phased enablement
(report -> flag -> block) a config change rather than a code change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .config import GuardrailConfig
from .types import Action, Category, Finding, Layer, Mode

_ACTION_RANK = {Action.ALLOW: 0, Action.FLAG: 1, Action.BLOCK: 2}


def noisy_or(scores: Iterable[float]) -> float:
    """Combine independent-ish evidence for one category.

    Three medium hits should outrank one medium hit without any single rule
    being able to saturate the score on its own. Capped below 1.0 so a
    deterministic result can still be distinguished from a certainty.
    """
    product = 1.0
    for s in scores:
        product *= 1.0 - max(0.0, min(s, 1.0))
    return min(1.0 - product, 0.99)


def category_scores(findings: Sequence[Finding]) -> dict[Category, float]:
    grouped: dict[Category, list[float]] = {}
    for f in findings:
        grouped.setdefault(f.category, []).append(f.score)
    return {c: noisy_or(v) for c, v in grouped.items()}


def dedupe(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    """Collapse rules that matched the same span; keep the most severe."""
    best: dict[tuple[int, int, Category], Finding] = {}
    for f in findings:
        key = (f.start, f.end, f.category)
        current = best.get(key)
        if current is None or (f.severity, f.score) > (current.severity, current.score):
            best[key] = f
    return tuple(sorted(best.values(), key=lambda f: (f.start, -int(f.severity))))


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    score: float
    policy_id: str
    scores: Mapping[Category, float]
    deciding_category: Category | None = None


def decide(
    findings: Sequence[Finding],
    config: GuardrailConfig,
    layer: Layer,
) -> Decision:
    scores = category_scores(findings)
    action = Action.ALLOW
    deciding: Category | None = None
    top_score = 0.0

    for category, score in scores.items():
        mode = config.mode(category)
        flag_at = config.flag_at.get(category, 0.55)
        block_at = config.block_at.get(category, 0.80)

        if mode is Mode.REPORT:
            candidate = Action.ALLOW          # observe only; audit still records it
        elif mode is Mode.FLAG:
            candidate = Action.FLAG if score >= flag_at else Action.ALLOW
        else:
            if score >= block_at:
                candidate = Action.BLOCK
            elif score >= flag_at:
                candidate = Action.FLAG
            else:
                candidate = Action.ALLOW

        if _ACTION_RANK[candidate] > _ACTION_RANK[action] or (
            candidate is action and score > top_score
        ):
            if _ACTION_RANK[candidate] >= _ACTION_RANK[action] and candidate is not Action.ALLOW:
                deciding = category
            action = candidate if _ACTION_RANK[candidate] > _ACTION_RANK[action] else action
        top_score = max(top_score, score)

    policy_id = (
        f"guardrails.{action.value}.{deciding.value}"
        if deciding is not None and action is not Action.ALLOW
        else "guardrails.ok"
    )
    return Decision(action, top_score, policy_id, scores, deciding)
