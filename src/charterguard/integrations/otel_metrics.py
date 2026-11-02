"""OpenTelemetry `MetricsSink`. Requires `charterguard[otel]`.

Prefer this over the Prometheus adapter when the rest of the service is already
instrumented with OTel: guardrail measurements then land in the same pipeline,
carry the same resource attributes, and -- because OTel metrics and traces share
context -- a latency spike in the model tier can be correlated with the trace
that caused it.

Unlike Prometheus, OTel accepts arbitrary attributes per record, so no label
projection is needed. Cardinality discipline still applies: attributes with
unbounded values (session ids, user ids) explode the backend's series count
whichever protocol carries them.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = ["OtelMetricsSink"]

_INSTALL = (
    "OtelMetricsSink needs the OpenTelemetry API. Install it with:\n"
    "    pip install 'charterguard[otel]'"
)


class OtelMetricsSink:
    """`MetricsSink` over an OpenTelemetry meter.

    Args:
        meter: an existing meter. If omitted, one is taken from the globally
            configured provider -- which is a no-op meter if the application
            never installed a real one, so measurements go nowhere silently.
            Passing the meter explicitly avoids that class of confusion.
        unit_ms: treat `observe` values as milliseconds (the convention used by
            `observability.emit`) and record the histogram in ``ms``.
    """

    __slots__ = ("_meter", "_counters", "_hists", "_gauges", "_unit")

    def __init__(self, meter: Any | None = None, *, name: str = "charterguard", unit_ms: bool = True) -> None:
        try:
            from opentelemetry import metrics  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(_INSTALL) from exc
        self._meter = meter if meter is not None else metrics.get_meter(name)
        self._counters: dict[str, Any] = {}
        self._hists: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._unit = "ms" if unit_ms else "1"

    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        c = self._counters.get(name)
        if c is None:
            c = self._counters[name] = self._meter.create_counter(name, unit="1")
        c.add(value, dict(tags or {}))

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        h = self._hists.get(name)
        if h is None:
            h = self._hists[name] = self._meter.create_histogram(name, unit=self._unit)
        h.record(value, dict(tags or {}))

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        g = self._gauges.get(name)
        if g is None:
            # `create_gauge` is only in newer API versions; an UpDownCounter is
            # the portable fallback, and set-semantics are approximated by
            # recording the delta from the last known value.
            factory = getattr(self._meter, "create_gauge", None)
            if factory is not None:
                g = self._gauges[name] = ("gauge", factory(name, unit="1"))
            else:
                g = self._gauges[name] = ("updown", self._meter.create_up_down_counter(name, unit="1"), {})
        if g[0] == "gauge":
            g[1].set(value, dict(tags or {}))
        else:
            key = tuple(sorted((tags or {}).items()))
            last = g[2].get(key, 0.0)
            g[1].add(value - last, dict(tags or {}))
            g[2][key] = value
