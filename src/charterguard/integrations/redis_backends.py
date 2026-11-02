"""Cross-host ledger backend. Requires `charterguard[redis]`.

This is the store that makes a per-session ceiling *actually* a ceiling when
the service runs on more than one host. The contract is one atomic multi-field
increment returning post-increment totals, which maps onto a single pipelined
``MULTI/HINCRBY.../EXPIRE/HGETALL/EXEC`` -- one round trip, no read-modify-write
window, no lost updates between replicas.

Deliberate design choices:

  * **Sliding TTL, not a fixed one.** ``EXPIRE`` is refreshed on every
    increment, so an idle session evaporates and an active one never does. A
    fixed TTL would reset a live session's ceiling mid-conversation, which is
    the failure mode where a long exfiltration attempt gets a fresh budget.
  * **No client-side arithmetic.** All addition happens server-side.
  * **Failure is loud.** A Redis outage raises; it does not silently degrade to
    an unenforced ceiling. Wrap it in `FallbackCounterStore` below if you have
    decided, explicitly, that availability outranks the ceiling.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..ledger import _FIELDS, CounterStore

__all__ = ["RedisCounterStore", "FallbackCounterStore"]

_INSTALL = (
    "RedisCounterStore needs the redis client. Install it with:\n"
    "    pip install 'charterguard[redis]'"
)


class RedisCounterStore:
    """`CounterStore` backed by one Redis hash per session.

    Args:
        client: an existing `redis.Redis` instance. Passing your own client is
            the supported path -- connection pooling, TLS, sentinel and cluster
            configuration belong to your infrastructure, not to this adapter.
        url: alternatively, a connection URL (``redis://`` / ``rediss://``).
        ttl_s: sliding expiry for a session's counters.
        key_prefix: namespace, so guardrail keys are separable from app keys.
    """

    __slots__ = ("_r", "_ttl_s", "_prefix")

    def __init__(
        self,
        client: Any | None = None,
        *,
        url: str | None = None,
        ttl_s: int = 3600,
        key_prefix: str = "gr:ledger:",
    ) -> None:
        if client is None:
            try:
                import redis  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(_INSTALL) from exc
            if url is None:
                raise ValueError("pass either an existing client or a redis url")
            client = redis.Redis.from_url(url, decode_responses=True)
        self._r = client
        self._ttl_s = int(ttl_s)
        self._prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def incr(self, session_id: str, deltas: Mapping[str, int]) -> Mapping[str, int]:
        key = self._key(session_id)
        applied = {f: int(deltas.get(f, 0) or 0) for f in _FIELDS}
        pipe = self._r.pipeline(transaction=True)
        for field, delta in applied.items():
            pipe.hincrby(key, field, delta)
        pipe.expire(key, self._ttl_s)
        results = pipe.execute()
        # HINCRBY replies arrive in field order; the trailing reply is EXPIRE.
        return {f: int(v) for f, v in zip(applied.keys(), results[: len(applied)])}

    def get(self, session_id: str) -> Mapping[str, int]:
        raw = self._r.hgetall(self._key(session_id)) or {}
        # A client built without decode_responses=True returns bytes keys;
        # normalise both shapes rather than requiring one client config.
        decoded = {
            (k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()
        }
        return {f: int(decoded.get(f, 0) or 0) for f in _FIELDS}

    def reset(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))


class FallbackCounterStore:
    """Wrap a remote store with a local one for when the remote is unreachable.

    Read the tradeoff before using this. On a Redis outage the ceiling silently
    becomes per-process again -- which is exactly the weakening this store
    exists to prevent. It is the right choice for a productivity agent where an
    outage must not take the product down, and the wrong choice for a regulated
    one, where the correct response to "I cannot enforce the limit" is to stop.

    `degraded_calls` counts fallbacks so the condition is alertable rather than
    invisible.
    """

    __slots__ = ("_primary", "_fallback", "degraded_calls", "_on_error")

    def __init__(
        self,
        primary: CounterStore,
        fallback: CounterStore | None = None,
        *,
        on_error=None,  # noqa: ANN001 - optional callable(exc) for logging/metrics
    ) -> None:
        from ..ledger import InMemoryCounterStore

        self._primary = primary
        self._fallback = fallback if fallback is not None else InMemoryCounterStore()
        self.degraded_calls = 0
        self._on_error = on_error

    def _degrade(self, exc: Exception) -> None:
        self.degraded_calls += 1
        if self._on_error is not None:
            self._on_error(exc)

    def incr(self, session_id: str, deltas: Mapping[str, int]) -> Mapping[str, int]:
        try:
            return self._primary.incr(session_id, deltas)
        except Exception as exc:  # noqa: BLE001 - deliberate availability tradeoff
            self._degrade(exc)
            return self._fallback.incr(session_id, deltas)

    def get(self, session_id: str) -> Mapping[str, int]:
        try:
            return self._primary.get(session_id)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            return self._fallback.get(session_id)

    def reset(self, session_id: str) -> None:
        try:
            self._primary.reset(session_id)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
        self._fallback.reset(session_id)
