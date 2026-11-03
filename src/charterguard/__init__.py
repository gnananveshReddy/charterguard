"""charterguard -- a framework-agnostic guardrail pipeline for agent harnesses.

Zero runtime dependencies. Drop it into any Python project; wire the five trust
boundaries; keep enforcement out of your prompts.

    from charterguard import GuardrailPipeline, GuardrailConfig, Charter

    pipeline = GuardrailPipeline(
        GuardrailConfig.report_only(),        # phase 1 of rollout
        charter=Charter(
            agent_id="policy-qa",
            version="1.0",
            purpose="answering questions about internal IT policy",
            in_scope=("policy", "standard", "procedure", "control"),
            out_of_scope=("salary", "performance review", "termination"),
            allowed_tools=("search_policy", "read_document"),
        ),
        system_prefix=SYSTEM_PROMPT,
    )

    v = pipeline.check_input(user_turn)
    if v.blocked:
        return pipeline.refusal(v)
"""
from __future__ import annotations

from .config import (
    GuardrailConfig,
    ModelTier,
    RedactionConfig,
    ScanBudget,
    VolumeCeilings,
)
from .detectors import (
    Detector,
    EncodedPayloadDetector,
    ModelDetector,
    ModelScorer,
    ObfuscationDetector,
    RuleDetector,
    StructuralDetector,
    severity_score,
)
from .envelope import (
    SYSTEM_ENVELOPE_RULES,
    Envelope,
    EnvelopeFactory,
    SystemPrefixGuard,
)
from .exceptions import GuardrailBlocked, GuardrailError
from .grounding import GroundingChecker, GroundingReport
from .ledger import CounterStore, InMemoryCounterStore, SessionLedger
from .normalize import NormalizedText, normalize
from .observability import InMemoryMetrics, MetricsSink, NullMetrics
from .pii import Entity, EntityDetector, EntityRecognizer
from .streaming import AsyncStreamGuard, StreamChunk, StreamGuard
from .patterns import RULES, Rule, rules_for
from .pipeline import GuardrailPipeline, UntrustedResult
from .policy import Decision, category_scores, decide, noisy_or
from .redact import Redaction, RedactionResult, redact
from .refusals import refusal_for
from .scanner import CompiledRuleSet, ResultCache, compile_rules, content_key
from .scope import Charter, IntentClassifier, ScopeGate
from .types import (
    Action,
    Category,
    Finding,
    Layer,
    Mode,
    Severity,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # pipeline
    "GuardrailPipeline",
    "UntrustedResult",
    # config
    "GuardrailConfig",
    "ScanBudget",
    "ModelTier",
    "RedactionConfig",
    "VolumeCeilings",
    # types
    "Action",
    "Category",
    "Finding",
    "Layer",
    "Mode",
    "Severity",
    "Verdict",
    # scope
    "Charter",
    "ScopeGate",
    "IntentClassifier",
    # detection
    "Detector",
    "RuleDetector",
    "StructuralDetector",
    "EncodedPayloadDetector",
    "ObfuscationDetector",
    "ModelDetector",
    "ModelScorer",
    # free-text PII seam (bring your own recognizer)
    "Entity",
    "EntityRecognizer",
    "EntityDetector",
    # embedding tier (stage 2 of the cascade)
    "TextEmbedder",
    "HashingEmbedder",
    "PrototypeBank",
    "SemanticScorer",
    "CascadingScorer",
    "cascade_latency_ms",
    "cascade_breakeven",
    "severity_score",
    "RULES",
    "Rule",
    "rules_for",
    "CompiledRuleSet",
    "compile_rules",
    "ResultCache",
    "content_key",
    "normalize",
    "NormalizedText",
    # streaming, observability, ledger backends
    "StreamGuard",
    "AsyncStreamGuard",
    "StreamChunk",
    "MetricsSink",
    "NullMetrics",
    "InMemoryMetrics",
    "CounterStore",
    "InMemoryCounterStore",
    # envelope
    "EnvelopeFactory",
    "Envelope",
    "SystemPrefixGuard",
    "SYSTEM_ENVELOPE_RULES",
    # policy / output
    "decide",
    "Decision",
    "category_scores",
    "noisy_or",
    "redact",
    "Redaction",
    "RedactionResult",
    "GroundingChecker",
    "GroundingReport",
    "SessionLedger",
    "refusal_for",
    # errors
    "GuardrailError",
    "GuardrailBlocked",
]


# --------------------------------------------------------------------- lazy tier
# The embedding tier lives behind PEP 562 lazy attribute access. `charterguard`
# stays a cold, dependency-free import; `charterguard.semantic` (and the numpy
# accelerator it will use if present) is loaded on first touch of one of these
# names. Measured: eager import cost 1398 ms and left numpy in sys.modules, lazy
# import costs 3 ms and does not.
_LAZY = {
    # operations tier: rule health and counterfactual policy evaluation
    "RuleHealth": "ruleops",
    "RuleStat": "ruleops",
    "ShadowPolicy": "shadow",
    "ShadowDelta": "shadow",
    "TextEmbedder": "semantic",
    "HashingEmbedder": "semantic",
    "PrototypeBank": "semantic",
    "SemanticScorer": "semantic",
    "CascadingScorer": "semantic",
    "cascade_latency_ms": "semantic",
    "cascade_breakeven": "semantic",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache: the lazy path is paid once
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY))
