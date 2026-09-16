"""Charter and scope gate: the anti-drift control.

Scope adherence is enforced twice, both times outside the model:
  * before the loop starts, on the user request (cheap, saves a whole run);
  * on every tool name the model proposes, against the charter's tool list.

The charter is a versioned, fingerprinted object. The fingerprint goes into
every audit row, so "which rules was this agent operating under" is answerable
months later even after the charter has changed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, Sequence

from .config import GuardrailConfig
from .normalize import normalize
from .patterns import Rule
from .policy import decide, dedupe
from .scanner import compile_rules
from .types import Action, Category, Finding, Layer, Severity, Timer, Verdict

_DEFAULT_REFUSAL = (
    "That request is outside what I'm set up to handle. I cover: {purpose} "
    "Ask me something in that area, or route this to a team or agent that owns it."
)


@dataclass(frozen=True, slots=True)
class Charter:
    agent_id: str
    version: str
    purpose: str
    in_scope: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    require_in_scope_signal: bool = True
    refusal_template: str = _DEFAULT_REFUSAL

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "agent_id": self.agent_id,
                "version": self.version,
                "purpose": self.purpose,
                "in_scope": list(self.in_scope),
                "out_of_scope": list(self.out_of_scope),
                "allowed_tools": sorted(self.allowed_tools),
            },
            sort_keys=True,
        )
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def refusal(self) -> str:
        return self.refusal_template.format(purpose=self.purpose)


class IntentClassifier(Protocol):
    """Optional model tier for the ambiguous band only."""

    async def classify(
        self, text: str, in_scope: Sequence[str], out_of_scope: Sequence[str]
    ) -> tuple[str, float]:
        ...


def _phrase_rules(phrases: Sequence[str], prefix: str, severity: Severity) -> tuple[Rule, ...]:
    """Compile charter phrases through the same engine as the security rules:
    one alternation pass instead of a Python loop over substrings."""
    rules: list[Rule] = []
    for i, phrase in enumerate(phrases):
        cleaned = " ".join(phrase.lower().split())
        if not cleaned:
            continue
        pattern = r"\b" + re.escape(cleaned).replace(r"\ ", r"\s+") + r"\b"
        rules.append(
            Rule(
                id=f"{prefix}.{i}",
                category=Category.OUT_OF_SCOPE,
                severity=severity,
                pattern=pattern,
                description=f"charter phrase: {cleaned}",
            )
        )
    return tuple(rules)


class ScopeGate:
    __slots__ = ("charter", "_in", "_out", "_classifier")

    def __init__(self, charter: Charter, *, classifier: IntentClassifier | None = None) -> None:
        self.charter = charter
        in_rules = _phrase_rules(charter.in_scope, "scope.in", Severity.LOW)
        out_rules = _phrase_rules(charter.out_of_scope, "scope.out", Severity.HIGH)
        self._in = compile_rules(in_rules) if in_rules else None
        self._out = compile_rules(out_rules) if out_rules else None
        self._classifier = classifier

    # ---------------------------------------------------------------- request
    def check(self, text: str, config: GuardrailConfig) -> Verdict:
        timer = Timer()
        nt = normalize(text, max_chars=8_000)
        findings: list[Finding] = []
        in_hits: list[str] = []

        if self._out is not None:
            for m in self._out.scan(nt.text, limit=8):
                s, e = nt.span(m.start, m.end)
                findings.append(
                    Finding(
                        rule_id="scope.explicit_out_of_scope",
                        category=Category.OUT_OF_SCOPE,
                        severity=Severity.HIGH,
                        score=0.88,
                        detector="scope",
                        start=s,
                        end=e,
                        excerpt=nt.excerpt(m.start, m.end),
                        description=f"matched charter exclusion ({m.rule.description})",
                    )
                )

        if self._in is not None:
            in_hits = [m.rule.description for m in self._in.scan(nt.text, limit=8)]

        if not in_hits and self.charter.require_in_scope_signal and not findings:
            findings.append(
                Finding(
                    rule_id="scope.no_in_charter_signal",
                    category=Category.OUT_OF_SCOPE,
                    # Deliberately mid-band: this is the case that should escalate
                    # to a classifier rather than hard-refuse a legitimate rephrase.
                    severity=Severity.MEDIUM,
                    score=0.62,
                    detector="scope",
                    description="no in-charter intent signal found in the request",
                )
            )

        found = dedupe(findings)
        d = decide(found, config, Layer.SCOPE)
        return Verdict(
            layer=Layer.SCOPE,
            action=d.action,
            score=d.score,
            findings=found,
            policy_id=d.policy_id,
            latency_us=timer.us(),
            meta={
                "charter": self.charter.agent_id,
                "charter_version": self.charter.version,
                "charter_fingerprint": self.charter.fingerprint(),
                "in_scope_hits": in_hits,
                "ambiguous": bool(found) and d.score < 0.8,
            },
        )

    async def acheck(self, text: str, config: GuardrailConfig) -> Verdict:
        """Deterministic first; classifier consulted only in the ambiguous band."""
        verdict = self.check(text, config)
        if self._classifier is None or not verdict.meta.get("ambiguous"):
            return verdict
        try:
            label, confidence = await self._classifier.classify(
                text, self.charter.in_scope, self.charter.out_of_scope
            )
        except Exception:  # noqa: BLE001 - classifier is advisory here
            return verdict
        if label != "out_of_scope" and confidence >= 0.6:
            # Classifier rescues a legitimate request the lexical gate missed.
            return Verdict(
                layer=Layer.SCOPE,
                action=Action.ALLOW,
                score=0.0,
                findings=(),
                policy_id="guardrails.ok",
                latency_us=verdict.latency_us,
                meta={**verdict.meta, "classifier": label, "confidence": confidence},
            )
        findings = verdict.findings + (
            Finding(
                rule_id="scope.classifier",
                category=Category.OUT_OF_SCOPE,
                severity=Severity.HIGH,
                score=confidence,
                detector="intent_classifier",
                description=f"classifier label={label} confidence={confidence:.2f}",
            ),
        )
        d = decide(findings, config, Layer.SCOPE)
        return Verdict(
            layer=Layer.SCOPE,
            action=d.action,
            score=d.score,
            findings=findings,
            policy_id=d.policy_id,
            latency_us=verdict.latency_us,
            meta={**verdict.meta, "classifier": label, "confidence": confidence},
        )

    # ---------------------------------------------------------------- tool
    def check_tool(self, tool_name: str, config: GuardrailConfig) -> Verdict:
        """Absence beats refusal: a tool outside the charter is unavailable,
        not merely discouraged."""
        timer = Timer()
        if not self.charter.allowed_tools or tool_name in self.charter.allowed_tools:
            return Verdict(layer=Layer.SCOPE, action=Action.ALLOW, latency_us=timer.us())
        finding = Finding(
            rule_id="scope.tool_not_in_charter",
            category=Category.OUT_OF_SCOPE,
            severity=Severity.CRITICAL,
            score=0.99,
            detector="scope",
            description=(
                f"tool '{tool_name}' is not in charter {self.charter.agent_id}"
                f"@{self.charter.version}"
            ),
        )
        return Verdict(
            layer=Layer.SCOPE,
            action=Action.BLOCK,
            score=0.99,
            findings=(finding,),
            policy_id="guardrails.block.out_of_scope",
            latency_us=timer.us(),
            meta={"tool": tool_name, "allowed": list(self.charter.allowed_tools)},
        )
