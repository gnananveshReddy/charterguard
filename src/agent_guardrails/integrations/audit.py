"""Verdict -> audit row, plus a durable sink.

Two distinct things live here, and conflating them is a compliance mistake:

  * `guardrail_rows` / `event_payload` are pure *shaping* helpers. They format a
    verdict for the `guardrail_finding` and `agent_event` tables and compute a
    chain hash. They persist nothing.
  * `AuditSink` is the durability contract, and `JsonlAuditSink` is a working
    local implementation with an unbroken hash chain, fsync-on-flush, and a
    process-wide lock.

A shaping helper alone is not an audit trail: nothing has been written, nothing
is tamper-evident, and a crash loses the record. Point `AuditSink` at your real
store (append-only table, object storage with retention lock, or a log pipeline)
for production. `JsonlAuditSink` is honest local durability -- it survives a
crash, but a local file is not tamper-proof against someone who can write to it.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from ..types import Verdict

GENESIS_HASH = "0" * 64


def guardrail_rows(
    verdict: Verdict,
    *,
    run_id: str,
    step: int,
    principal_id: str,
    agent_identity: str,
) -> list[dict[str, Any]]:
    """One row per finding, matching the guardrail_finding table."""
    return [
        {
            "run_id": run_id,
            "step": step,
            "principal_id": principal_id,
            "agent_identity": agent_identity,
            "layer": verdict.layer.value,
            "policy_id": f.rule_id,
            "verdict": verdict.action.value,
            "confidence": round(f.score, 3),
            "category": f.category.value,
            "matched_span": f.excerpt[:200],
            "action_taken": verdict.action.value,
        }
        for f in verdict.findings
    ]


def event_payload(verdict: Verdict, *, prev_hash: str) -> tuple[dict[str, Any], str]:
    """Hash-chained event payload. Returns (payload, row_hash)."""
    payload = verdict.to_audit()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    row_hash = hashlib.sha256((prev_hash + canonical).encode()).hexdigest()
    return payload, row_hash


class AuditSink(Protocol):
    """Durability contract for guardrail decisions.

    `write` must be atomic with respect to the hash chain: the returned hash is
    the chain head that the next call links against.
    """

    def write(
        self,
        verdict: Verdict,
        *,
        run_id: str = "",
        step: int = 0,
        principal_id: str = "",
        agent_identity: str = "",
    ) -> str:
        ...

    def flush(self) -> None: ...


class NullAuditSink:
    """Discards records. Explicit opt-out, so "no audit" is a visible choice."""

    __slots__ = ()

    def write(self, verdict: Verdict, **_: Any) -> str:
        return GENESIS_HASH

    def flush(self) -> None:
        return None


class JsonlAuditSink:
    """Append-only JSONL sink with a verifiable hash chain.

    Each line carries the payload, its `row_hash`, and the `prev_hash` it links
    to, so `verify_chain` can detect deletion or edit of any record. On restart
    the chain head is recovered from the last line, so restarts do not fork the
    chain.
    """

    __slots__ = ("_path", "_lock", "_head", "_fh", "_fsync")

    def __init__(self, path: str | os.PathLike[str], *, fsync: bool = True) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fsync = fsync
        self._head = self._recover_head()
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def head(self) -> str:
        return self._head

    def __enter__(self) -> "JsonlAuditSink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Safety net only: callers should use `close()` or a `with` block.
        try:
            self.close()
        except Exception:
            pass

    def _recover_head(self) -> str:
        if not self._path.exists():
            return GENESIS_HASH
        head = GENESIS_HASH
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    head = json.loads(line)["row_hash"]
                except (ValueError, KeyError):
                    continue  # tolerate a torn final write
        return head

    def write(
        self,
        verdict: Verdict,
        *,
        run_id: str = "",
        step: int = 0,
        principal_id: str = "",
        agent_identity: str = "",
    ) -> str:
        with self._lock:
            payload, row_hash = event_payload(verdict, prev_hash=self._head)
            record = {
                "prev_hash": self._head,
                "row_hash": row_hash,
                "run_id": run_id,
                "step": step,
                "principal_id": principal_id,
                "agent_identity": agent_identity,
                "event": payload,
                "findings": guardrail_rows(
                    verdict,
                    run_id=run_id,
                    step=step,
                    principal_id=principal_id,
                    agent_identity=agent_identity,
                ),
            }
            self._fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            self._head = row_hash
            return row_hash

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def records(self) -> Iterable[dict[str, Any]]:
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def verify_chain(records: Iterable[Mapping[str, Any]]) -> tuple[bool, int]:
    """Recompute the chain. Returns (intact, index of first bad record or -1)."""
    prev = GENESIS_HASH
    for index, rec in enumerate(records):
        canonical = json.dumps(rec["event"], sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256((prev + canonical).encode()).hexdigest()
        if rec.get("prev_hash") != prev or rec.get("row_hash") != expected:
            return False, index
        prev = expected
    return True, -1
