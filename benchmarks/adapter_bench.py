#!/usr/bin/env python3
"""What do the optional adapters actually cost?

Run:  python benchmarks/adapter_bench.py

Two questions this answers, both of which are easy to get wrong by intuition:

  1. Metrics sinks sit on the enforcement path and are called several times per
     verdict. A logging sink that formats and writes on every call is not free,
     and the honest number matters when choosing a default.
  2. `SqliteCounterStore` buys a correct cross-process ceiling. That correctness
     is not free either -- it is a transaction and an fsync-class write per
     increment. The number below decides whether you put it inline or behind a
     batching layer.
"""
from __future__ import annotations

import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_guardrails.integrations.heuristics import (  # noqa: E402
    GazetteerEntityRecognizer,
    KeywordIntentClassifier,
)
from agent_guardrails.integrations.logging_metrics import (  # noqa: E402
    ConsoleMetricsSink,
    LoggingMetricsSink,
    StatsdMetricsSink,
)
from agent_guardrails.integrations.sqlite_backends import (  # noqa: E402
    SqliteAuditSink,
    SqliteCounterStore,
)
from agent_guardrails.ledger import InMemoryCounterStore  # noqa: E402
from agent_guardrails.observability import InMemoryMetrics, NullMetrics  # noqa: E402
from agent_guardrails.types import Action, Layer, Verdict  # noqa: E402

N = 20_000


def timed(label: str, fn, n: int = N) -> float:
    fn()  # warm
    start = time.perf_counter()
    for _ in range(n):
        fn()
    per_call_us = (time.perf_counter() - start) / n * 1e6
    print(f"  {label:<34} {per_call_us:>9.3f} us/call")
    return per_call_us


def bench_metrics() -> None:
    print("\nMetrics sinks (one incr call; emit() makes ~4 per verdict)")
    tags = {"layer": "input", "action": "block"}

    null = NullMetrics()
    timed("NullMetrics", lambda: null.incr("m", 1, tags))

    mem = InMemoryMetrics()
    timed("InMemoryMetrics", lambda: mem.incr("m", 1, tags))

    quiet = logging.getLogger("bench.quiet")
    quiet.setLevel(logging.CRITICAL)
    quiet.addHandler(logging.NullHandler())
    sink = LoggingMetricsSink("bench.quiet", level=logging.INFO)
    timed("LoggingMetricsSink (level off)", lambda: sink.incr("m", 1, tags))

    loud = logging.getLogger("bench.loud")
    loud.setLevel(logging.INFO)
    devnull = open("/dev/null", "w")
    handler = logging.StreamHandler(devnull)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    loud.addHandler(handler)
    live = LoggingMetricsSink("bench.loud", level=logging.INFO)
    timed("LoggingMetricsSink (writing)", lambda: live.incr("m", 1, tags), n=5_000)

    sampled = LoggingMetricsSink("bench.loud", level=logging.INFO, sample_rate=0.01)
    timed("LoggingMetricsSink (1% sampled)", lambda: sampled.incr("m", 1, tags), n=5_000)

    console = ConsoleMetricsSink(devnull)
    timed("ConsoleMetricsSink", lambda: console.incr("m", 1, tags), n=5_000)

    # Bind a receiver: sending to a closed port provokes ICMP
    # port-unreachable, which makes the send path look 2-3x worse than it is.
    import socket

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    rx.bind(("127.0.0.1", 0))
    port = rx.getsockname()[1]

    statsd = StatsdMetricsSink("127.0.0.1", port)
    timed("StatsdMetricsSink (1 syscall/metric)", lambda: statsd.incr("m", 1, tags), n=20_000)
    statsd.close()

    packed = StatsdMetricsSink("127.0.0.1", port, max_datagram_bytes=1400)
    timed("StatsdMetricsSink (batched 1400B)", lambda: packed.incr("m", 1, tags), n=20_000)
    packed.close()
    rx.close()
    devnull.close()


def bench_counters() -> None:
    print("\nCounter stores (one atomic multi-field increment)")
    mem = InMemoryCounterStore()
    timed("InMemoryCounterStore", lambda: mem.incr("s", {"rows": 1, "bytes": 10}), n=20_000)

    tmp = Path(tempfile.mkdtemp(prefix="gr-bench-"))
    store = SqliteCounterStore(tmp / "c.db", purge_every=1_000_000)
    per_call = timed(
        "SqliteCounterStore (WAL, txn)", lambda: store.incr("s", {"rows": 1, "bytes": 10}), n=3_000
    )
    store.close()
    print(f"    -> {1e6 / per_call:,.0f} increments/sec ceiling on this disk")


def bench_audit() -> None:
    print("\nAudit sinks (one chained record)")
    tmp = Path(tempfile.mkdtemp(prefix="gr-bench-"))
    verdict = Verdict(layer=Layer.INPUT, action=Action.BLOCK, score=0.9)
    sink = SqliteAuditSink(tmp / "a.db")
    timed("SqliteAuditSink (synchronous=FULL)", lambda: sink.write(verdict), n=500)
    sink.close()
    relaxed = SqliteAuditSink(tmp / "b.db")
    relaxed._conn().execute("PRAGMA synchronous = NORMAL")
    timed("SqliteAuditSink (synchronous=NORMAL)", lambda: relaxed.write(verdict), n=1_000)
    relaxed.close()
    print("    NORMAL is faster and can lose the last records on power loss.")
    print("    FULL is the default here because an audit trail that loses its")
    print("    tail is not an audit trail.")


def bench_heuristics() -> None:
    print("\nDependency-free model-seam stand-ins")
    import asyncio

    clf = KeywordIntentClassifier()
    in_scope = ["reset a password", "check order status", "update billing address"]
    out_scope = ["give medical advice", "recommend a stock", "write legal contracts"]
    text = "hi, I need to check the status of my order from last week please"
    loop = asyncio.new_event_loop()
    timed(
        "KeywordIntentClassifier.classify",
        lambda: loop.run_until_complete(clf.classify(text, in_scope, out_scope)),
        n=5_000,
    )
    loop.close()

    doc = "Dr. Priya Sharma met Mr. Chen in Bangalore to review the chart. " * 8
    rec = GazetteerEntityRecognizer(names=["Sharma", "Chen"])
    timed("GazetteerEntityRecognizer.recognize", lambda: rec.recognize(doc), n=5_000)


def main() -> int:
    print(f"python {sys.version.split()[0]} -- adapter overhead")
    bench_metrics()
    bench_counters()
    bench_audit()
    bench_heuristics()
    print(
        "\nReading these numbers:\n"
        "  * A disabled logging sink is close to NullMetrics, so leaving\n"
        "    LoggingMetricsSink wired at DEBUG costs nothing in production.\n"
        "  * A writing logging sink is the expensive option; sample it or send\n"
        "    to statsd/OTel instead of formatting a record per measurement.\n"
        "  * Unbatched statsd is one syscall per measurement. max_datagram_bytes\n"
        "    packs ~40 metrics per datagram and is ~4x cheaper per call.\n"
        "  * The sqlite counter store is orders of magnitude slower than the\n"
        "    in-memory one and still fast relative to a single LLM call. It is\n"
        "    the price of a ceiling that actually holds across workers.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
