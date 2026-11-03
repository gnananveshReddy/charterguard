"""Optional model-backed classifier tier.

Two properties make this safe to put in a production hot path:

  1. Cascade. The model is consulted when the deterministic score lands in the
     uncertainty band (default 0.25-0.85), and -- if `escalate_clean` is set for
     the layer -- on a sampled share of traffic that tripped no rule at all,
     which is where a novel attack shows up. Obvious-bad traffic short-circuits
     before the model and never pays the latency or the token cost.
  2. Explicit failure policy. Timeouts and exceptions resolve to `fail_closed`
     from config -- open for productivity-tier agents, closed for regulated ones.
"""
from __future__ import annotations

import asyncio
from typing import Protocol, cast
from collections.abc import Mapping, Sequence

from ..config import GuardrailConfig
from ..normalize import NormalizedText
from ..scanner import ResultCache, content_key
from ..types import Category, Finding, Layer, Severity


class ModelScorer(Protocol):
    """Adapter boundary: wrap any classifier, moderation endpoint, or LLM judge."""

    async def score(
        self, text: str, categories: Sequence[Category], layer: Layer
    ) -> Mapping[Category, float]:
        ...


def _severity_for(score: float) -> Severity:
    if score >= 0.90:
        return Severity.CRITICAL
    if score >= 0.75:
        return Severity.HIGH
    return Severity.MEDIUM


class ModelDetector:
    name = "model"

    __slots__ = ("_scorer", "_cache", "categories")

    def __init__(
        self,
        scorer: ModelScorer,
        *,
        categories: Sequence[Category] = (
            Category.INJECTION,
            Category.JAILBREAK,
            Category.HARMFUL,
            Category.SELF_HARM,
            Category.MALWARE,
        ),
        cache_size: int = 1024,
    ) -> None:
        self._scorer = scorer
        self._cache = ResultCache(cache_size)
        self.categories = tuple(categories)

    def should_escalate(
        self,
        deterministic_score: float,
        config: GuardrailConfig,
        layer: Layer | None = None,
        *,
        text: str | None = None,
    ) -> bool:
        """Two bands: uncertainty always, clean band when configured.

        Clean-band sampling is deterministic in the content hash, so a given
        document escalates consistently across replicas and retries instead of
        flickering with a PRNG.
        """
        tier = config.model_for(layer)
        if not tier.enabled:
            return False
        if tier.escalate_low <= deterministic_score < tier.escalate_high:
            return True
        if deterministic_score < tier.escalate_low and tier.escalate_clean:
            if tier.clean_sample_rate >= 1.0:
                return True
            if tier.clean_sample_rate <= 0.0 or text is None:
                return False
            bucket = int(content_key(text)[:8], 16) / 0xFFFFFFFF
            return bucket < tier.clean_sample_rate
        return False

    async def ascan(
        self,
        nt: NormalizedText,
        layer: Layer,
        config: GuardrailConfig,
        *,
        deterministic_score: float = 0.0,
    ) -> list[Finding]:
        tier = config.model_for(layer)
        text = nt.text[: tier.max_chars]
        key = f"{layer.value}|{content_key(text)}"
        cached = self._cache.get(key)
        if cached is not None:
            return list(cast("tuple[Finding, ...]", cached))

        try:
            scores = await asyncio.wait_for(
                self._scorer.score(text, self.categories, layer), timeout=tier.timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - deliberate policy decision
            if not tier.fail_closed:
                return [
                    Finding(
                        rule_id="model.unavailable",
                        category=Category.INJECTION,
                        severity=Severity.LOW,
                        score=0.0,
                        detector=self.name,
                        description=f"classifier unavailable, failing open: {type(exc).__name__}",
                    )
                ]
            return [
                Finding(
                    rule_id="model.unavailable",
                    category=Category.INJECTION,
                    severity=Severity.CRITICAL,
                    score=1.0,
                    detector=self.name,
                    description=f"classifier unavailable, failing closed: {type(exc).__name__}",
                )
            ]

        findings = [
            Finding(
                rule_id=f"model.{category.value}",
                category=category,
                severity=_severity_for(raw),
                # Blend against the deterministic score. Weighting the model
                # against itself (the obvious typo) is a no-op that silently
                # discards the rule-tier signal.
                score=tier.weight * raw + (1 - tier.weight) * deterministic_score,
                detector=self.name,
                description=f"classifier score {raw:.2f} for {category.value}",
            )
            for category, raw in scores.items()
            if raw >= min(config.flag_at.get(category, 0.55), 0.5)
        ]
        self._cache.put(key, tuple(findings))
        return findings
