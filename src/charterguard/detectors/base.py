"""Detector protocol plus the deterministic rule detector.

Detectors never normalize: they receive an already-normalized text object. This
keeps the expensive pass at exactly one per call regardless of detector count.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Mapping, Sequence

from ..config import GuardrailConfig
from ..normalize import NormalizedText
from ..patterns import Rule, rules_for
from ..scanner import CompiledRuleSet, compile_rules
from ..types import Category, Finding, Layer, Severity
from ..validators import digits, gstin_ok, iban_ok, india_mobile_ok, looks_random, luhn_ok, verhoeff_ok

# Severity -> base confidence. Deterministic rules are high-precision by
# construction, so even MEDIUM sits above a typical 0.55 flag threshold.
_SEVERITY_SCORE: Mapping[Severity, float] = {
    Severity.LOW: 0.35,
    Severity.MEDIUM: 0.60,
    Severity.HIGH: 0.80,
    Severity.CRITICAL: 0.95,
}


def severity_score(severity: Severity) -> float:
    return _SEVERITY_SCORE[severity]


def _rhs(matched: str) -> str:
    """Right-hand side of a `key = value` credential match."""
    for sep in ("=", ":"):
        idx = matched.find(sep)
        if idx != -1:
            return matched[idx + 1 :].strip().strip("\"'` ")
    return matched


def _phone_ok(matched: str) -> bool:
    d = digits(matched)
    if not 10 <= len(d) <= 15:
        return False
    return len(set(d)) > 2  # reject 0000000000 / sequential padding


_VALIDATORS = {
    "luhn": luhn_ok,
    "verhoeff": verhoeff_ok,
    "entropy": lambda m: looks_random(_rhs(m)),
    "phone": _phone_ok,
    "gstin": gstin_ok,
    "india_mobile": india_mobile_ok,
    "iban": iban_ok,
}


@runtime_checkable
class Detector(Protocol):
    name: str

    def scan(self, nt: NormalizedText, layer: Layer, config: GuardrailConfig) -> list[Finding]:
        ...


class RuleDetector:
    """One compiled rule set per layer, with validator-gated precision.

    Instantiate once per process. `scan` is pure and thread-safe.
    """

    __slots__ = ("name", "_by_layer", "_stop_at")

    def __init__(
        self,
        name: str = "rules",
        *,
        rules: Sequence[Rule] | None = None,
        categories: Sequence[Category] | None = None,
        stop_at: Severity | None = Severity.CRITICAL,
    ) -> None:
        self.name = name
        source = tuple(rules) if rules is not None else None
        self._by_layer: dict[Layer, CompiledRuleSet] = {}
        for layer in Layer:
            selected = rules_for(
                layer=layer,
                categories=categories,
                **({"rules": source} if source is not None else {}),
            )
            if selected:
                self._by_layer[layer] = compile_rules(selected)
        self._stop_at = stop_at

    def rule_count(self, layer: Layer) -> int:
        rs = self._by_layer.get(layer)
        return len(rs) if rs else 0

    @staticmethod
    def _keep(rule: Rule, matched: str) -> bool:
        if rule.validate is None:
            return True
        validator = _VALIDATORS.get(rule.validate)
        return True if validator is None else bool(validator(matched))

    def scan(self, nt: NormalizedText, layer: Layer, config: GuardrailConfig) -> list[Finding]:
        ruleset = self._by_layer.get(layer)
        if ruleset is None or not nt.text:
            return []
        matches = ruleset.scan(
            nt.text,
            limit=config.budget.max_findings,
            stop_at=self._stop_at if config.short_circuit_on_critical else None,
            keep=self._keep,
        )
        findings: list[Finding] = []
        for m in matches:
            start, end = nt.span(m.start, m.end)
            findings.append(
                Finding(
                    rule_id=m.rule.id,
                    category=m.rule.category,
                    severity=m.rule.severity,
                    score=severity_score(m.rule.severity),
                    detector=self.name,
                    start=start,
                    end=end,
                    excerpt=nt.excerpt(m.start, m.end),
                    description=m.rule.description,
                )
            )
        return findings


def layered_rulesets(rules: Sequence[Rule], categories: Sequence[Category]) -> dict[Layer, CompiledRuleSet]:
    """One compiled rule set per layer for a detector that re-scans derived
    text (decoded payloads, de-obfuscated views): the rules for that layer
    plus every untrusted-layer rule, so a pack rule scoped to any layer is
    applied to its layer's derived text as well."""
    untrusted = rules_for(layer=Layer.UNTRUSTED, categories=categories, rules=rules)
    out: dict[Layer, CompiledRuleSet] = {}
    for layer in Layer:
        own = rules_for(layer=layer, categories=categories, rules=rules)
        merged = tuple({r.id: r for r in (*untrusted, *own)}.values())
        if merged:
            out[layer] = compile_rules(merged)
    return out
