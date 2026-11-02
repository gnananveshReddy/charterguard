"""Policy configuration. Per-category modes and thresholds, plus scan budgets."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from .types import Category, Layer, Mode

POLICY_VERSION = "1.1.1"

# Categories that should reach BLOCK first when you tighten a phased rollout:
# high precision, low legitimate-traffic overlap.
_DEFAULT_MODES: dict[Category, Mode] = {
    Category.INJECTION: Mode.FLAG,
    Category.JAILBREAK: Mode.FLAG,
    Category.HARMFUL: Mode.FLAG,
    Category.SELF_HARM: Mode.FLAG,
    Category.MALWARE: Mode.FLAG,
    Category.COMMAND: Mode.FLAG,
    Category.RESTRICTED_TOPIC: Mode.REPORT,
    Category.SECRET: Mode.BLOCK,       # never let a credential leave
    Category.PII: Mode.FLAG,           # redacted rather than blocked by default
    Category.EXFIL: Mode.BLOCK,
    Category.OUT_OF_SCOPE: Mode.BLOCK,
    Category.UNGROUNDED: Mode.FLAG,
    Category.VOLUME: Mode.BLOCK,
}

_DEFAULT_FLAG_AT: dict[Category, float] = {c: 0.55 for c in Category}
_DEFAULT_BLOCK_AT: dict[Category, float] = {c: 0.80 for c in Category}


@dataclass(frozen=True, slots=True)
class ScanBudget:
    """Bounded work per call. Guardrails must never be the latency story."""

    max_chars: int = 200_000
    tail_chars: int = 8_000       # attackers append payloads at the end
    max_findings: int = 64
    decode_depth: int = 2         # base64/hex/percent nesting
    min_encoded_run: int = 24
    max_decoded_chars: int = 16_000
    cache_size: int = 4_096       # retrieval returns the same chunks constantly
    # Above this size the async entry points move the scan to a worker thread
    # rather than blocking the event loop; below it, the thread handoff costs
    # more than the scan it defers.
    offload_threshold_chars: int = 4_096


@dataclass(frozen=True, slots=True)
class ModelTier:
    """Optional classifier tier.

    Escalation has two bands, not one:

      * uncertainty band [escalate_low, escalate_high): the deterministic score
        is ambiguous, so a second opinion changes the answer;
      * clean band [0, escalate_low): the deterministic rules found *nothing*.
        This is exactly where a novel attack lands, so refusing to escalate here
        makes the model tier useless for the case it is best at. `escalate_clean`
        opens that band, and `clean_sample_rate` prices it -- 1.0 for a low
        volume hop like a user turn, 0.02 for retrieval chunks where full
        coverage would multiply cost by the fan-out.

    Sampling is keyed on a content hash rather than a PRNG, so the same document
    always makes the same decision and the result cache stays coherent.
    """

    enabled: bool = False
    timeout_s: float = 1.5
    fail_closed: bool = False     # True for regulated deployments
    escalate_low: float = 0.25    # below: deterministic ALLOW ...
    escalate_high: float = 0.85   # above: trust the deterministic BLOCK
    escalate_clean: bool = False  # ... unless this is on: cover novel attacks
    clean_sample_rate: float = 1.0
    weight: float = 0.65          # blend weight for the model score
    max_chars: int = 8_000        # cap tokens per classifier call


@dataclass(frozen=True, slots=True)
class RedactionConfig:
    enabled: bool = True
    placeholder: str = "[REDACTED:{kind}]"
    keep_digest: bool = True      # prove what passed without storing it
    digest_chars: int = 12
    redact_categories: tuple[Category, ...] = (Category.SECRET, Category.PII)


@dataclass(frozen=True, slots=True)
class VolumeCeilings:
    """Slow-exfiltration defence: cumulative per session, not per request."""

    max_rows: int = 5_000
    max_bytes: int = 8_000_000
    max_pii_values: int = 50
    max_external_fetches: int = 25
    max_tool_calls: int = 200


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    modes: Mapping[Category, Mode] = field(default_factory=lambda: dict(_DEFAULT_MODES))
    flag_at: Mapping[Category, float] = field(default_factory=lambda: dict(_DEFAULT_FLAG_AT))
    block_at: Mapping[Category, float] = field(default_factory=lambda: dict(_DEFAULT_BLOCK_AT))
    budget: ScanBudget = ScanBudget()
    model: ModelTier = ModelTier()
    # Per-layer overrides. A single user turn is one call per request; retrieved
    # chunks are N per request. They should not share an escalation budget.
    model_by_layer: Mapping[Layer, ModelTier] = field(default_factory=dict)
    redaction: RedactionConfig = RedactionConfig()
    volume: VolumeCeilings = VolumeCeilings()
    enable_encoded_scan: bool = True
    short_circuit_on_critical: bool = True
    policy_version: str = POLICY_VERSION

    def model_for(self, layer: Layer | None = None) -> ModelTier:
        """Escalation policy for one layer, falling back to the global tier."""
        if layer is None:
            return self.model
        return self.model_by_layer.get(layer, self.model)

    def with_model_tier(self, layer: Layer, tier: ModelTier) -> GuardrailConfig:
        by_layer = dict(self.model_by_layer)
        by_layer[layer] = tier
        return replace(self, model_by_layer=by_layer)

    def mode(self, category: Category) -> Mode:
        return self.modes.get(category, Mode.FLAG)

    def with_mode(self, category: Category, mode: Mode) -> GuardrailConfig:
        modes = dict(self.modes)
        modes[category] = mode
        return replace(self, modes=modes)

    def with_all_modes(self, mode: Mode) -> GuardrailConfig:
        return replace(self, modes={c: mode for c in Category})

    @classmethod
    def report_only(cls) -> GuardrailConfig:
        """Phase 1 of rollout: observe precision before enforcing anything."""
        return cls().with_all_modes(Mode.REPORT)

    @classmethod
    def strict(cls) -> GuardrailConfig:
        """Phase 3+: block everything high-confidence, fail closed.

        Also turns on clean-band escalation, asymmetrically: full coverage on the
        one-per-turn inbound hop, 5% sampling on the high-fan-out untrusted hops
        so novel-attack coverage does not scale with retrieval breadth.
        """
        cfg = cls().with_all_modes(Mode.BLOCK)
        # enabled=True is inert unless a ModelScorer is passed to the pipeline
        # (no scorer -> no ModelDetector -> no escalation), so strict mode can
        # turn the tier on without making a model a hard dependency.
        tier = replace(cfg.model, fail_closed=True, enabled=True)
        cfg = replace(cfg, model=tier)
        cfg = cfg.with_model_tier(
            Layer.INPUT, replace(tier, escalate_clean=True, clean_sample_rate=1.0)
        )
        for layer in (Layer.UNTRUSTED, Layer.TOOL_OUTPUT):
            cfg = cfg.with_model_tier(
                layer, replace(tier, escalate_clean=True, clean_sample_rate=0.05)
            )
        return cfg
