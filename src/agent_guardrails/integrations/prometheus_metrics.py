"""Prometheus `MetricsSink`. Requires `agent_guardrails[prometheus]`.

One structural mismatch has to be handled rather than wished away: the
guardrail `MetricsSink` protocol accepts free-form tags per call, while a
Prometheus collector fixes its label names at registration. A metric registered
with ``{layer}`` and later called with ``{layer, category}`` raises.

This adapter pins each metric's label names to the first call it sees, then
projects every later call onto that set -- missing labels become ``""``, extra
labels are dropped and counted in ``label_mismatches``. That keeps a metrics
call from ever raising into the enforcement path, while making the
misconfiguration visible instead of silent.

Cardinality warning, because this is how Prometheus deployments die: never pass
`session_id`, `run_id`, `principal_id` or raw text as a tag. Each distinct
value is a permanent time series. The bundled `observability.emit` tags with
layer/action/category only, which is bounded and safe.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = ["PrometheusMetricsSink"]

_INSTALL = (
    "PrometheusMetricsSink needs prometheus-client. Install it with:\n"
    "    pip install 'agent_guardrails[prometheus]'"
)

# Latency in this system spans sub-millisecond rule scans to multi-second model
# calls, so the default prometheus buckets (which top out at 10s but start at
# 5ms) hide the deterministic tier entirely.
DEFAULT_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class PrometheusMetricsSink:
    """`MetricsSink` writing to a prometheus_client registry."""

    __slots__ = ("_ns", "_registry", "_buckets", "_counters", "_hists", "_gauges", "_labels", "label_mismatches", "_m", "_scale")

    def __init__(
        self,
        *,
        namespace: str = "guardrail",
        registry: Any | None = None,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
        observe_unit: str = "seconds",
    ) -> None:
        try:
            import prometheus_client  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(_INSTALL) from exc
        if observe_unit not in ("seconds", "milliseconds"):
            raise ValueError("observe_unit must be 'seconds' or 'milliseconds'")
        self._m = prometheus_client
        self._ns = namespace
        self._registry = registry if registry is not None else prometheus_client.REGISTRY
        self._buckets = buckets
        self._counters: dict[str, Any] = {}
        self._hists: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._labels: dict[str, tuple[str, ...]] = {}
        self.label_mismatches = 0
        # `observability.emit` reports latency in milliseconds; Prometheus
        # convention is base units (seconds). Convert once, here, rather than
        # shipping a metric whose name lies about its unit.
        self._scale = 0.001 if observe_unit == "seconds" else 1.0

    # ------------------------------------------------------------- internals
    @staticmethod
    def _safe(name: str) -> str:
        return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)

    def _label_values(self, name: str, tags: Mapping[str, str] | None) -> tuple[str, ...]:
        declared = self._labels[name]
        tags = tags or {}
        if set(tags) - set(declared):
            self.label_mismatches += 1
        return tuple(str(tags.get(k, "")) for k in declared)

    def _get(self, kind: str, name: str, tags: Mapping[str, str] | None) -> Any:
        store = {"c": self._counters, "h": self._hists, "g": self._gauges}[kind]
        metric = store.get(name)
        if metric is None:
            labels = tuple(sorted((tags or {}).keys()))
            self._labels[name] = labels
            full = f"{self._ns}_{self._safe(name)}"
            if kind == "c":
                metric = self._m.Counter(full + "_total", name, labels, registry=self._registry)
            elif kind == "h":
                metric = self._m.Histogram(
                    full, name, labels, buckets=self._buckets, registry=self._registry
                )
            else:
                metric = self._m.Gauge(full, name, labels, registry=self._registry)
            store[name] = metric
        return metric

    # ------------------------------------------------------------ sink API
    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        m = self._get("c", name, tags)
        vals = self._label_values(name, tags)
        (m.labels(*vals) if vals else m).inc(value)

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        m = self._get("h", name, tags)
        vals = self._label_values(name, tags)
        (m.labels(*vals) if vals else m).observe(value * self._scale)

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        m = self._get("g", name, tags)
        vals = self._label_values(name, tags)
        (m.labels(*vals) if vals else m).set(value)
