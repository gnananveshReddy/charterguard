"""Cumulative session ledger -- the slow-exfiltration control.

Per-request checks cannot see an attacker extracting 100 rows per turn across
40 turns. This tracks cumulative volume per session and fires once a ceiling is
crossed, which is also the cheapest circuit breaker for runaway loops.

Storage sits behind the `CounterStore` protocol. The bundled store is
thread-safe but **process-local**, which means ceilings are enforced per replica:
N replicas allow roughly N times the configured volume. Implement `CounterStore`
against Redis (or any atomic counter service) for a genuinely shared ceiling --
the protocol is one atomic multi-field increment, so it is a single round trip.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol
from collections.abc import Mapping

from .config import VolumeCeilings
from .types import Category, Finding, Severity

_FIELDS = ("rows", "bytes", "pii_values", "external_fetches", "tool_calls")


@dataclass
class _Counters:
    rows: int = 0
    bytes: int = 0
    pii_values: int = 0
    external_fetches: int = 0
    tool_calls: int = 0
    started: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in _FIELDS}


class CounterStore(Protocol):
    """Pluggable counter storage behind the ledger.

    The default store is per-process. That is correct for one worker and *wrong*
    for a horizontally scaled service: 20 replicas running the in-memory store
    enforce roughly 20x the intended per-session ceiling, because each replica
    only observes its own slice of the traffic. Sticky sessions hide this rather
    than fix it.

    Implement this protocol against Redis (``HINCRBY`` per-session hash plus
    ``EXPIRE``) or an equivalent service for a real shared ceiling. The contract
    is deliberately one atomic multi-field increment returning post-increment
    totals, so it maps to a single round trip and cannot lose updates to a
    read-modify-write race between concurrent writers.
    """

    def incr(self, session_id: str, deltas: Mapping[str, int]) -> Mapping[str, int]:
        """Atomically apply `deltas`; return every counter after the update."""
        ...

    def get(self, session_id: str) -> Mapping[str, int]: ...

    def reset(self, session_id: str) -> None: ...


class InMemoryCounterStore:
    """Process-local store: TTL eviction, bounded cardinality, thread-safe."""

    __slots__ = ("_data", "_lock", "_ttl_s", "_max_sessions")

    def __init__(self, *, ttl_s: float = 3600.0, max_sessions: int = 10_000) -> None:
        self._data: dict[str, _Counters] = {}
        self._lock = threading.Lock()
        self._ttl_s = ttl_s
        self._max_sessions = max_sessions

    def incr(self, session_id: str, deltas: Mapping[str, int]) -> Mapping[str, int]:
        now = time.monotonic()
        with self._lock:
            c = self._data.get(session_id)
            if c is None:
                if len(self._data) >= self._max_sessions:
                    self._evict_locked(now)
                c = self._data[session_id] = _Counters()
            for name, delta in deltas.items():
                if delta:
                    setattr(c, name, getattr(c, name) + delta)
            c.last_seen = now
            return c.as_dict()

    def get(self, session_id: str) -> Mapping[str, int]:
        with self._lock:
            c = self._data.get(session_id)
            return c.as_dict() if c else {f: 0 for f in _FIELDS}

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def _evict_locked(self, now: float) -> None:
        stale = [k for k, v in self._data.items() if now - v.last_seen > self._ttl_s]
        for k in stale:
            del self._data[k]
        if len(self._data) >= self._max_sessions:
            oldest = sorted(self._data.items(), key=lambda kv: kv[1].last_seen)
            for k, _ in oldest[: max(1, len(oldest) // 10)]:
                del self._data[k]


class SessionLedger:
    """Per-session volume accounting; storage delegated to a `CounterStore`."""

    __slots__ = ("_store",)

    def __init__(
        self,
        *,
        ttl_s: float = 3600.0,
        max_sessions: int = 10_000,
        store: CounterStore | None = None,
    ) -> None:
        self._store = store or InMemoryCounterStore(ttl_s=ttl_s, max_sessions=max_sessions)

    @property
    def store(self) -> CounterStore:
        return self._store

    @property
    def is_shared(self) -> bool:
        """False for the bundled store: ceilings then apply per replica, not per
        session cluster-wide. Surface this in readiness checks."""
        return not isinstance(self._store, InMemoryCounterStore)

    def record(
        self,
        session_id: str,
        *,
        rows: int = 0,
        bytes_: int = 0,
        pii_values: int = 0,
        external_fetches: int = 0,
        tool_calls: int = 0,
    ) -> Mapping[str, int]:
        return self._store.incr(
            session_id,
            {
                "rows": rows,
                "bytes": bytes_,
                "pii_values": pii_values,
                "external_fetches": external_fetches,
                "tool_calls": tool_calls,
            },
        )

    def snapshot(self, session_id: str) -> Mapping[str, int]:
        return self._store.get(session_id)

    def reset(self, session_id: str) -> None:
        self._store.reset(session_id)

    def check(self, session_id: str, ceilings: VolumeCeilings) -> Finding | None:
        totals = self.snapshot(session_id)
        breaches = [
            (name, totals[name], limit)
            for name, limit in (
                ("rows", ceilings.max_rows),
                ("bytes", ceilings.max_bytes),
                ("pii_values", ceilings.max_pii_values),
                ("external_fetches", ceilings.max_external_fetches),
                ("tool_calls", ceilings.max_tool_calls),
            )
            if totals[name] > limit
        ]
        if not breaches:
            return None
        detail = ", ".join(f"{n}={v} > {lim}" for n, v, lim in breaches)
        return Finding(
            rule_id="volume.session_ceiling",
            category=Category.VOLUME,
            severity=Severity.CRITICAL,
            score=0.97,
            detector="ledger",
            description=f"cumulative session ceiling exceeded ({detail})",
        )
