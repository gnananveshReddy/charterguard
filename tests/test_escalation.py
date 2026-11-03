"""Model-tier escalation: the bands, the sampling, and the blend.

The bug these tests exist to prevent: a clean-looking novel attack scores 0.0 on
the deterministic tier, and a cascade that only escalates inside an uncertainty
band therefore never asks the model about exactly the input the model is for.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace


from charterguard import GuardrailPipeline
from charterguard.config import GuardrailConfig, ModelTier
from charterguard.detectors.model import ModelDetector
from charterguard.normalize import normalize
from charterguard.types import Category, Layer


class StubScorer:
    """Records every call so escalation can be asserted, not inferred."""

    def __init__(self, score: float = 0.9) -> None:
        self.calls: list[tuple[str, Layer]] = []
        self._score = score

    async def score(self, text, categories, layer):
        self.calls.append((text, layer))
        return {Category.INJECTION: self._score}


class HangingScorer:
    async def score(self, text, categories, layer):
        await asyncio.sleep(10)
        return {}


class BrokenScorer:
    async def score(self, text, categories, layer):
        raise RuntimeError("classifier down")


def _tier(**kw) -> ModelTier:
    return ModelTier(**{"enabled": True, **kw})


# --------------------------------------------------------------------- bands
def test_uncertainty_band_escalates():
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=_tier())
    assert det.should_escalate(0.5, cfg, Layer.INPUT, text="hello")


def test_clean_score_does_not_escalate_by_default():
    """Default stays cheap: no cost on traffic that tripped nothing."""
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=_tier())
    assert not det.should_escalate(0.0, cfg, Layer.INPUT, text="hello")


def test_clean_band_escalates_when_enabled():
    """The whole point: novel attack, zero rule hits, still gets a second look."""
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=_tier(escalate_clean=True))
    assert det.should_escalate(0.0, cfg, Layer.INPUT, text="a novel attack")


def test_high_score_never_escalates():
    """Already blocked: paying for a model call cannot change the verdict."""
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=_tier(escalate_clean=True))
    assert not det.should_escalate(0.99, cfg, Layer.INPUT, text="obvious attack")


def test_disabled_tier_never_escalates():
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=ModelTier(enabled=False, escalate_clean=True))
    assert not det.should_escalate(0.0, cfg, Layer.INPUT, text="x")


# ------------------------------------------------------------------ sampling
def test_clean_sampling_is_deterministic_per_document():
    """Same document, same decision -- across retries and across replicas."""
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=_tier(escalate_clean=True, clean_sample_rate=0.5))
    first = [det.should_escalate(0.0, cfg, Layer.UNTRUSTED, text=f"doc-{i}") for i in range(50)]
    again = [det.should_escalate(0.0, cfg, Layer.UNTRUSTED, text=f"doc-{i}") for i in range(50)]
    assert first == again


def test_clean_sampling_rate_is_roughly_honoured():
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=_tier(escalate_clean=True, clean_sample_rate=0.2))
    hits = sum(
        det.should_escalate(0.0, cfg, Layer.UNTRUSTED, text=f"chunk-{i}") for i in range(400)
    )
    assert 40 <= hits <= 120, hits  # 20% +/- sampling noise


def test_zero_sample_rate_disables_clean_escalation():
    det = ModelDetector(StubScorer())
    cfg = replace(GuardrailConfig(), model=_tier(escalate_clean=True, clean_sample_rate=0.0))
    assert not det.should_escalate(0.0, cfg, Layer.UNTRUSTED, text="chunk")


# ----------------------------------------------------------------- per layer
def test_per_layer_tiers_are_independent():
    """A user turn is one call per request; retrieval is N. Different budgets."""
    det = ModelDetector(StubScorer())
    cfg = GuardrailConfig().with_model_tier(
        Layer.INPUT, _tier(escalate_clean=True, clean_sample_rate=1.0)
    ).with_model_tier(
        Layer.UNTRUSTED, _tier(escalate_clean=False)
    )
    assert det.should_escalate(0.0, cfg, Layer.INPUT, text="novel")
    assert not det.should_escalate(0.0, cfg, Layer.UNTRUSTED, text="novel")


def test_strict_enables_full_input_and_sampled_untrusted():
    cfg = GuardrailConfig.strict()
    assert cfg.model_for(Layer.INPUT).escalate_clean
    assert cfg.model_for(Layer.INPUT).clean_sample_rate == 1.0
    untrusted = cfg.model_for(Layer.UNTRUSTED)
    assert untrusted.escalate_clean and untrusted.clean_sample_rate < 1.0
    assert cfg.model.fail_closed and cfg.model.enabled


# -------------------------------------------------------------------- scoring
def test_blend_uses_the_deterministic_score():
    """Regression: blending the model score against itself discards the rules."""
    scorer = StubScorer(score=1.0)
    det = ModelDetector(scorer)
    cfg = replace(GuardrailConfig(), model=_tier(weight=0.5))
    nt = normalize("some text")
    low = asyncio.run(det.ascan(nt, Layer.INPUT, cfg, deterministic_score=0.0))
    det2 = ModelDetector(StubScorer(score=1.0))
    high = asyncio.run(det2.ascan(nt, Layer.INPUT, cfg, deterministic_score=0.8))
    assert high[0].score > low[0].score


def test_text_is_truncated_to_max_chars():
    scorer = StubScorer()
    det = ModelDetector(scorer)
    cfg = replace(GuardrailConfig(), model=_tier(max_chars=100))
    asyncio.run(det.ascan(normalize("x" * 5_000), Layer.INPUT, cfg))
    assert len(scorer.calls[0][0]) <= 100


def test_cache_key_is_layer_scoped():
    """Same text at two layers must not reuse one verdict: policy differs."""
    scorer = StubScorer()
    det = ModelDetector(scorer)
    cfg = replace(GuardrailConfig(), model=_tier())
    nt = normalize("same text")
    asyncio.run(det.ascan(nt, Layer.INPUT, cfg))
    asyncio.run(det.ascan(nt, Layer.UNTRUSTED, cfg))
    assert [layer for _, layer in scorer.calls] == [Layer.INPUT, Layer.UNTRUSTED]
    asyncio.run(det.ascan(nt, Layer.INPUT, cfg))
    assert len(scorer.calls) == 2  # third call served from cache


def test_timeout_fails_open_or_closed_per_policy():
    det = ModelDetector(HangingScorer())
    nt = normalize("text")
    open_cfg = replace(GuardrailConfig(), model=_tier(timeout_s=0.01, fail_closed=False))
    closed_cfg = replace(GuardrailConfig(), model=_tier(timeout_s=0.01, fail_closed=True))
    opened = asyncio.run(det.ascan(nt, Layer.INPUT, open_cfg))
    assert opened[0].score == 0.0
    det2 = ModelDetector(HangingScorer())
    closed = asyncio.run(det2.ascan(nt, Layer.INPUT, closed_cfg))
    assert closed[0].score == 1.0


def test_exception_is_handled_like_a_timeout():
    det = ModelDetector(BrokenScorer())
    cfg = replace(GuardrailConfig(), model=_tier(fail_closed=True))
    findings = asyncio.run(det.ascan(normalize("text"), Layer.INPUT, cfg))
    assert findings[0].rule_id == "model.unavailable"


# ------------------------------------------------------------------- pipeline
def test_pipeline_escalates_clean_untrusted_content():
    """End to end: a rule-clean retrieval chunk still reaches the classifier."""
    scorer = StubScorer(score=0.95)
    cfg = GuardrailConfig().with_model_tier(
        Layer.UNTRUSTED, _tier(escalate_clean=True, clean_sample_rate=1.0)
    )
    p = GuardrailPipeline(cfg, model=scorer)
    body = "The vendor invoice totals 412 euro and is due next Friday."
    assert p.inspect_untrusted(body).verdict.score == 0.0  # deterministic tier: nothing
    result = asyncio.run(p.ainspect_untrusted(body))
    assert scorer.calls, "clean untrusted content never reached the model tier"
    assert result.verdict.meta.get("model_escalated") is True
    assert any(f.detector == "model" for f in result.verdict.findings)


def test_pipeline_input_escalation_reports_in_meta():
    scorer = StubScorer(score=0.9)
    cfg = GuardrailConfig().with_model_tier(
        Layer.INPUT, _tier(escalate_clean=True, clean_sample_rate=1.0)
    )
    p = GuardrailPipeline(cfg, model=scorer)
    verdict = asyncio.run(p.acheck_input("what is the refund policy"))
    assert verdict.meta.get("model_escalated") is True


def test_no_scorer_means_no_escalation_cost():
    p = GuardrailPipeline(GuardrailConfig.strict())
    verdict = asyncio.run(p.acheck_input("hello"))
    assert "model_escalated" not in verdict.meta
