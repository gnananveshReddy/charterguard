"""Core value types. Dependency-free, immutable, cheap to construct."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Mapping


class Category(str, Enum):
    """What a finding is about. Policy is configured per category."""

    INJECTION = "injection"
    JAILBREAK = "jailbreak"
    HARMFUL = "harmful"
    SELF_HARM = "self_harm"
    MALWARE = "malware"
    COMMAND = "command"
    RESTRICTED_TOPIC = "restricted_topic"
    SECRET = "secret"  # noqa: S105 - category name, not a credential
    PII = "pii"
    EXFIL = "exfil"
    OUT_OF_SCOPE = "out_of_scope"
    UNGROUNDED = "ungrounded"
    VOLUME = "volume"


class Layer(str, Enum):
    """Where in the loop the check ran. Every verdict records this."""

    INPUT = "input"
    SCOPE = "scope"
    UNTRUSTED = "untrusted"
    TOOL_OUTPUT = "tool_output"
    OUTPUT = "output"


class Action(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


class Mode(str, Enum):
    """Phased rollout: report -> flag -> block. Never start at BLOCK."""

    REPORT = "report"
    FLAG = "flag"
    BLOCK = "block"


class Severity(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule or model hit. Spans point into the ORIGINAL text."""

    rule_id: str
    category: Category
    severity: Severity
    score: float
    detector: str
    start: int = 0
    end: int = 0
    excerpt: str = ""
    description: str = ""
    decoded_from: str | None = None

    def to_audit(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": int(self.severity),
            "score": round(self.score, 4),
            "detector": self.detector,
            "span": [self.start, self.end],
            "excerpt": self.excerpt[:200],
            "decoded_from": self.decoded_from,
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """The single return type of every guardrail check."""

    layer: Layer
    action: Action
    score: float = 0.0
    findings: tuple[Finding, ...] = ()
    policy_id: str = "guardrails.ok"
    latency_us: int = 0
    truncated: bool = False
    sanitized: str | None = None
    quarantined: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action is Action.BLOCK

    @property
    def categories(self) -> tuple[Category, ...]:
        seen: dict[Category, None] = {}
        for f in self.findings:
            seen[f.category] = None
        return tuple(seen)

    @property
    def top(self) -> Finding | None:
        return max(self.findings, key=lambda f: (f.severity, f.score), default=None)

    def to_audit(self) -> dict[str, Any]:
        top = self.top
        return {
            "layer": self.layer.value,
            "action": self.action.value,
            "policy_id": self.policy_id,
            "score": round(self.score, 4),
            "latency_us": self.latency_us,
            "truncated": self.truncated,
            "quarantined": self.quarantined,
            "top_category": top.category.value if top else None,
            "finding_count": len(self.findings),
            "findings": [f.to_audit() for f in self.findings[:20]],
            "meta": dict(self.meta),
        }


class Timer:
    """Microsecond timer. perf_counter_ns avoids float drift on hot paths."""

    __slots__ = ("_t0",)

    def __init__(self) -> None:
        self._t0 = time.perf_counter_ns()

    def us(self) -> int:
        return (time.perf_counter_ns() - self._t0) // 1000
