"""Metrics sinks that need nothing but the standard library.

Three of them, in increasing order of production seriousness:

  * `ConsoleMetricsSink` -- human-readable lines for local development.
  * `LoggingMetricsSink` -- one structured record per measurement, emitted
    through `logging` so an existing log pipeline (CloudWatch, Loki, Splunk,
    Datadog agent) picks it up with no new egress path and no new dependency.
  * `StatsdMetricsSink` -- real aggregation. StatsD is a plain-text UDP
    protocol, so a client library is not required to speak it; this is ~40
    lines of `socket` and gives you percentiles from a datagram the process
    never waits on.

A caution that applies to the first two and is measured in
`benchmarks/adapter_bench.py`: `observability.emit()` makes several sink calls
per verdict, and a logging sink formats and writes on every one. Emitting at
INFO from a hot path is materially more expensive than `NullMetrics`. Both
logging sinks therefore check `isEnabledFor` before doing any work, and expose
`sample_rate` so a high-QPS deployment can keep counters exact while thinning
the per-measurement records.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Mapping

__all__ = [
    "LoggingMetricsSink",
    "ConsoleMetricsSink",
    "StatsdMetricsSink",
]

# `logging` raises KeyError if an `extra` key collides with a LogRecord
# attribute (`name`, `message`, `msg`, `args`, `levelname`, `module`, ...), so
# every field this module injects is prefixed rather than named naturally.
_PREFIX = "gr_"


def _fmt_tags(tags: Mapping[str, str] | None) -> str:
    if not tags:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(tags.items()))


class LoggingMetricsSink:
    """Structured measurement records through stdlib `logging`.

    Wire your aggregator to the logger name (default
    ``agent_guardrails.metrics``). Each record carries ``gr_metric``,
    ``gr_kind``, ``gr_value`` and ``gr_tags`` in ``extra``, so a JSON formatter
    emits queryable fields instead of a prose string.
    """

    __slots__ = ("_log", "_level", "_sample_rate", "_n")

    def __init__(
        self,
        logger_name: str = "agent_guardrails.metrics",
        *,
        level: int = logging.INFO,
        sample_rate: float = 1.0,
    ) -> None:
        self._log = logging.getLogger(logger_name)
        self._level = level
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        self._n = 0

    def _emit(self, kind: str, name: str, value: float, tags: Mapping[str, str] | None) -> None:
        # Cheapest possible early exit: no formatting, no dict copy, no I/O.
        if not self._log.isEnabledFor(self._level):
            return
        if self._sample_rate < 1.0:
            self._n += 1
            # Deterministic 1-in-N rather than a PRNG: reproducible, and no
            # `random` call on the hot path.
            if (self._n % max(1, int(1.0 / self._sample_rate))) != 0:
                return
        self._log.log(
            self._level,
            "%s %s=%s %s",
            kind,
            name,
            value,
            _fmt_tags(tags),
            extra={
                _PREFIX + "metric": name,
                _PREFIX + "kind": kind,
                _PREFIX + "value": value,
                _PREFIX + "tags": dict(tags or {}),
            },
        )

    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        self._emit("counter", name, value, tags)

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self._emit("observation", name, value, tags)

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self._emit("gauge", name, value, tags)


class ConsoleMetricsSink:
    """Development sink: one aligned line per measurement on stderr.

    Not for production -- it writes unconditionally and does no sampling. Its
    purpose is to make "is my policy actually firing?" answerable in one run
    without standing up any infrastructure.
    """

    __slots__ = ("_stream", "_started")

    def __init__(self, stream=None) -> None:  # noqa: ANN001 - any write()-able
        import sys

        self._stream = stream if stream is not None else sys.stderr
        self._started = time.monotonic()

    def _write(self, kind: str, name: str, value: float, tags: Mapping[str, str] | None) -> None:
        elapsed = time.monotonic() - self._started
        self._stream.write(f"[{elapsed:8.3f}s] {kind:<11} {name:<28} {value:>10} {_fmt_tags(tags)}\n")

    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        self._write("counter", name, value, tags)

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self._write("observation", name, round(value, 3), tags)

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self._write("gauge", name, round(value, 3), tags)


class StatsdMetricsSink:
    """StatsD over UDP. Real aggregation, zero pip dependencies.

    StatsD's wire format is one line of text per metric
    (``name:value|type|#tag:v``), so `socket` is the only client needed. UDP is
    fire-and-forget: a dead collector costs a syscall, never a stall, which is
    the property you want from a guardrail hot path. Datadog's agent, Telegraf
    and statsd-exporter all accept this.

    The tag syntax differs by collector (`dogstatsd` appends ``|#k:v``, plain
    StatsD has no tags), so `tag_style` is explicit rather than guessed.
    """

    __slots__ = ("_addr", "_sock", "_prefix", "_tag_style", "_sample_rate", "_buf", "_buf_bytes", "_max_bytes", "_lock")

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8125,
        *,
        prefix: str = "",
        tag_style: str = "dogstatsd",
        sample_rate: float = 1.0,
        max_datagram_bytes: int = 0,
    ) -> None:
        if tag_style not in ("dogstatsd", "none"):
            raise ValueError("tag_style must be 'dogstatsd' or 'none'")
        if max_datagram_bytes and max_datagram_bytes > 8_192:
            # Beyond the typical 8 KiB collector read buffer (and past an MTU on
            # a non-loopback path) the datagram is silently truncated, which
            # loses metrics instead of speeding them up.
            raise ValueError("max_datagram_bytes above 8192 risks silent truncation")
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._prefix = prefix
        self._tag_style = tag_style
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        # Batching is opt-in. Measured: an unbatched send is ~34 us -- one
        # syscall per measurement, and `observability.emit` makes about four per
        # verdict, which is comparable to an entire deterministic scan. Buffering
        # into 1400-byte datagrams amortises the syscall roughly 20:1. The cost
        # is bounded loss: whatever sits in the buffer when the process dies is
        # gone, which is acceptable for metrics and would not be for audit.
        self._max_bytes = int(max_datagram_bytes)
        self._buf: list[str] = []
        self._buf_bytes = 0
        self._lock = threading.Lock()

    def _send(self, name: str, value: float, kind: str, tags: Mapping[str, str] | None) -> None:
        line = f"{self._prefix}{name}:{value}|{kind}"
        if self._sample_rate < 1.0:
            line += f"|@{self._sample_rate}"
        if tags and self._tag_style == "dogstatsd":
            line += "|#" + ",".join(f"{k}:{v}" for k, v in sorted(tags.items()))
        if self._max_bytes <= 0:
            self._emit_raw(line)
            return
        with self._lock:
            # +1 for the newline separator StatsD uses between packed metrics.
            if self._buf and self._buf_bytes + len(line) + 1 > self._max_bytes:
                payload = "\n".join(self._buf)
                self._buf = [line]
                self._buf_bytes = len(line)
            else:
                self._buf.append(line)
                self._buf_bytes += len(line) + 1
                return
        self._emit_raw(payload)

    def _emit_raw(self, payload: str) -> None:
        try:
            self._sock.sendto(payload.encode("utf-8"), self._addr)
        except OSError:
            # Metrics must never break enforcement. A full socket buffer or an
            # unroutable collector is dropped silently and deliberately.
            pass

    def incr(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        self._send(name, value, "c", tags)

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        # `ms` (timer) rather than `h`: timers are universally supported and
        # give percentiles, which is what latency questions actually need.
        self._send(name, round(value, 3), "ms", tags)

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        self._send(name, round(value, 3), "g", tags)

    def flush(self) -> None:
        """Send any buffered measurements. Call on shutdown, or periodically
        from a timer if traffic is bursty -- otherwise the tail of a quiet period
        sits in the buffer until the next metric arrives."""
        with self._lock:
            if not self._buf:
                return
            payload = "\n".join(self._buf)
            self._buf = []
            self._buf_bytes = 0
        self._emit_raw(payload)

    def close(self) -> None:
        self.flush()
        self._sock.close()
