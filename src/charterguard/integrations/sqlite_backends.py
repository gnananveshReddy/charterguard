"""Multi-process ledger and audit backends on stdlib `sqlite3`.

The bundled `InMemoryCounterStore` is process-local, which means a service on
four gunicorn workers enforces roughly four times its configured per-session
ceiling. That is the single most common way a volume control silently stops
working. Redis fixes it across hosts; `sqlite3` fixes it across *processes on
one host* with no dependency at all, which covers the very common "N workers,
one container/VM" deployment.

Correctness notes, because a counter store that loses updates is worse than no
counter store:

  * Every increment runs inside ``BEGIN IMMEDIATE``, so concurrent writers
    serialise at the database rather than racing a read-modify-write.
  * The increment is an UPSERT (``SET rows = rows + excluded.rows``), so the
    arithmetic happens in SQL, not in Python between two round trips.
  * WAL mode plus ``busy_timeout`` lets readers proceed during a write and
    makes contention a wait rather than an immediate ``SQLITE_BUSY``.
  * Connections are thread-local; sqlite3 connections are not safe to share
    across threads.

Scope limit, stated plainly: this is one host. A shared filesystem (NFS, EFS)
is *not* a supported substitute for a network counter service -- sqlite's
locking is unreliable there. For multi-host, implement `CounterStore` against
Redis (see `redis_backends.py`).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..ledger import _FIELDS
from ..types import Verdict
from .audit import GENESIS_HASH, event_payload, guardrail_rows

__all__ = ["SqliteCounterStore", "SqliteAuditSink"]

# Column names are interpolated into SQL (sqlite cannot bind identifiers), so
# they must come only from this module-level constant and be plain identifiers.
# Values are always bound with "?".
if not all(f.isidentifier() and f.isascii() for f in _FIELDS):  # pragma: no cover
    raise RuntimeError(f"ledger field names must be SQL-safe identifiers: {_FIELDS!r}")
_COLS = ", ".join(_FIELDS)


def _enable_wal(conn: sqlite3.Connection, timeout_s: float) -> None:
    """Switch to WAL, retrying while another connection holds the lock.

    Changing the journal mode can fail with SQLITE_BUSY at once, without
    consulting the busy handler, when several processes open a fresh database
    together. Retry with backoff until the busy timeout is spent.
    """
    deadline = time.monotonic() + timeout_s
    delay = 0.005
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) and "busy" not in str(exc):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.1)


class SqliteCounterStore:
    """`CounterStore` shared by every process on one host.

    Args:
        path: database file. Use a real path on local disk, not a network mount.
        ttl_s: sessions untouched for this long are purged opportunistically.
        busy_timeout_s: how long a writer waits for a competing writer.
    """

    __slots__ = ("_path", "_local", "_ttl_s", "_busy_timeout_s", "_purge_every", "_since_purge", "_lock")

    def __init__(
        self,
        path: str | os.PathLike[str] = "guardrail_counters.db",
        *,
        ttl_s: float = 3600.0,
        busy_timeout_s: float = 5.0,
        purge_every: int = 500,
    ) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._ttl_s = ttl_s
        self._busy_timeout_s = busy_timeout_s
        self._purge_every = purge_every
        self._since_purge = 0
        self._lock = threading.Lock()
        self._migrate()

    # ------------------------------------------------------------ connection
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=self._busy_timeout_s, isolation_level=None)
            conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_s * 1000)}")
            if self._path != ":memory:":
                # WAL survives across connections and processes; skip for
                # :memory:, where it is meaningless.
                _enable_wal(conn, self._busy_timeout_s)
            conn.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = conn
        return conn

    def _migrate(self) -> None:
        cols = ", ".join(f"{f} INTEGER NOT NULL DEFAULT 0" for f in _FIELDS)
        conn = self._conn()
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS counters ("
            f"session_id TEXT PRIMARY KEY, {cols}, "
            f"started REAL NOT NULL, last_seen REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS counters_last_seen ON counters (last_seen)")

    # ------------------------------------------------------ CounterStore API
    def incr(self, session_id: str, deltas: Mapping[str, int]) -> Mapping[str, int]:
        applied = {f: int(deltas.get(f, 0) or 0) for f in _FIELDS}
        now = time.time()
        conn = self._conn()
        placeholders = ", ".join("?" for _ in _FIELDS)
        updates = ", ".join(f"{f} = {f} + excluded.{f}" for f in _FIELDS)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                f"INSERT INTO counters (session_id, {_COLS}, started, last_seen) "  # noqa: S608 - identifiers validated at import
                f"VALUES (?, {placeholders}, ?, ?) "
                f"ON CONFLICT(session_id) DO UPDATE SET {updates}, last_seen = excluded.last_seen",
                (session_id, *(applied[f] for f in _FIELDS), now, now),
            )
            row = conn.execute(
                f"SELECT {_COLS} FROM counters WHERE session_id = ?", (session_id,)  # noqa: S608
            ).fetchone()
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        self._maybe_purge(now)
        return dict(zip(_FIELDS, row, strict=True))

    def get(self, session_id: str) -> Mapping[str, int]:
        row = self._conn().execute(
            f"SELECT {_COLS} FROM counters WHERE session_id = ?", (session_id,)  # noqa: S608
        ).fetchone()
        return dict(zip(_FIELDS, row, strict=True)) if row else {f: 0 for f in _FIELDS}

    def reset(self, session_id: str) -> None:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM counters WHERE session_id = ?", (session_id,))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------ housekeeping
    def _maybe_purge(self, now: float) -> None:
        """Amortised TTL eviction. Without this the table grows without bound,
        which is a slow disk leak rather than a crash -- the kind of failure
        nobody notices for a quarter."""
        with self._lock:
            self._since_purge += 1
            if self._since_purge < self._purge_every:
                return
            self._since_purge = 0
        try:
            self._conn().execute("DELETE FROM counters WHERE last_seen < ?", (now - self._ttl_s,))
        except sqlite3.OperationalError:
            pass  # a competing writer holds the lock; next call retries

    def session_count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM counters").fetchone()[0])

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class SqliteAuditSink:
    """`AuditSink` with the hash chain persisted in sqlite.

    Preferred over `JsonlAuditSink` when several processes write audit records
    for the same deployment: the chain head is read and advanced inside one
    ``BEGIN IMMEDIATE`` transaction, so two workers cannot fork the chain by
    linking against the same predecessor. JSONL cannot offer that -- its head
    lives in process memory.

    Still not tamper-proof. Anyone who can write the database can rewrite
    history; the chain makes edits *detectable* by an auditor who holds an
    external copy of a later hash, which is a different and weaker guarantee.
    """

    __slots__ = ("_path", "_local", "_lock", "_busy_timeout_s")

    def __init__(
        self,
        path: str | os.PathLike[str] = "guardrail_audit.db",
        *,
        busy_timeout_s: float = 5.0,
    ) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._busy_timeout_s = busy_timeout_s
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS audit ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
            "prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL UNIQUE, "
            "run_id TEXT, step INTEGER, principal_id TEXT, agent_identity TEXT, "
            "event TEXT NOT NULL, findings TEXT NOT NULL)"
        )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=self._busy_timeout_s, isolation_level=None)
            conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_s * 1000)}")
            if self._path != ":memory:":
                _enable_wal(conn, self._busy_timeout_s)
            conn.execute("PRAGMA synchronous = FULL")  # audit durability over speed
            self._local.conn = conn
        return conn

    @property
    def head(self) -> str:
        row = self._conn().execute("SELECT row_hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS_HASH

    def write(
        self,
        verdict: Verdict,
        *,
        run_id: str = "",
        step: int = 0,
        principal_id: str = "",
        agent_identity: str = "",
    ) -> str:
        conn = self._conn()
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT row_hash FROM audit ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                prev = row[0] if row else GENESIS_HASH
                payload, row_hash = event_payload(verdict, prev_hash=prev)
                findings = guardrail_rows(
                    verdict,
                    run_id=run_id,
                    step=step,
                    principal_id=principal_id,
                    agent_identity=agent_identity,
                )
                conn.execute(
                    "INSERT INTO audit (ts, prev_hash, row_hash, run_id, step, principal_id, "
                    "agent_identity, event, findings) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        prev,
                        row_hash,
                        run_id,
                        step,
                        principal_id,
                        agent_identity,
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        json.dumps(findings, sort_keys=True, separators=(",", ":")),
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return row_hash

    def flush(self) -> None:
        return None  # every write is already committed

    def records(self) -> Iterable[dict[str, Any]]:
        """Chain records in order, shaped like `JsonlAuditSink.records()` so
        `verify_chain` accepts either backend unchanged."""
        for row in self._conn().execute(
            "SELECT prev_hash, row_hash, run_id, step, principal_id, agent_identity, event, "
            "findings FROM audit ORDER BY seq"
        ):
            yield {
                "prev_hash": row[0],
                "row_hash": row[1],
                "run_id": row[2],
                "step": row[3],
                "principal_id": row[4],
                "agent_identity": row[5],
                "event": json.loads(row[6]),
                "findings": json.loads(row[7]),
            }

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
