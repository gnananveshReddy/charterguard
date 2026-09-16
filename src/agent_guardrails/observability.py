"""Metrics and trace hooks.

Guardrails fail silently in the worst way: a config regression that stops
blocking looks identical to a quiet week. These hooks exist so block rate,
escalation rate, latency and cache hit rate are observable per layer and per
category, and so a drop can page someone.

`MetricsSink` is a protocol -- there is no vendored client and no dependency on
a metrics library. `NullMetrics` is the default (no overhead beyond a call),
`InMemoryMetrics` is for tests and local runs, and a StatsD/OpenTelemetry
adapter is a dozen lines against the same three methods.
"""
from __future__ import annotations

import threading
from typing import Mapping, Protocol

from .types import Action, Category, Verdict

__all__ = ["MetricsSink", "NullMetrics", "InMemoryMetrics", "verdict_tags"]


class MetricsSink(Protocol):
    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None: ...

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None: ...

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None: ...


class NullMetrics:
    """Default sink. Cheaper than branching on `if metrics is not None`."""

    __slots__ = ()

    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        return None

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        return None

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        return None


class InMemoryMetrics:
    """Aggregating sink for tests, benchmarks and single-process debugging."""

    __slots__ = ("counters", "observations", "gauges", "_lock")

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.observations: dict[str, list[float]] = {}
        self.gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, tags: Mapping[str, str] | None) -> str:
        if not tags:
            return name
        return name + "{" + ",".join(f"{k}={v}" for k, v in sorted(tags.items())) + "}"

    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        k = self._key(name, tags)
        with self._lock:
            self.counters[k] = self.counters.get(k, 0) + value

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        k = self._key(name, tags)
        with self._lock:
            self.observations.setdefault(k, []).append(value)

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        with self._lock:
            self.gauges[self._key(name, tags)] = value


def verdict_tags(verdict: Verdict) -> dict[str, str]:
    """Low-cardinality tags only -- never session ids, never content."""
    top: Category | None = verdict.findings[0].category if verdict.findings else None
    return {
        "layer": verdict.layer.value,
        "action": verdict.action.value,
        "category": top.value if top else "none",
    }


def emit(metrics: MetricsSink, verdict: Verdict, *, escalated: bool = False) -> None:
    tags = verdict_tags(verdict)
    metrics.incr("guardrail.verdict", tags=tags)
    metrics.observe(
        "guardrail.latency_ms", verdict.latency_us / 1000.0, tags={"layer": tags["layer"]}
    )
    if verdict.action is not Action.ALLOW:
        metrics.incr("guardrail.enforced", tags=tags)
    if escalated:
        metrics.incr("guardrail.model_escalation", tags={"layer": tags["layer"]})
    for finding in verdict.findings:
        metrics.incr(
            "guardrail.finding",
            tags={
                "layer": tags["layer"],
                "category": finding.category.value,
                "severity": finding.severity.name.lower(),
                "rule": finding.rule_id or "n/a",
            },
        )
