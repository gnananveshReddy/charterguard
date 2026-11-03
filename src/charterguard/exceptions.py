"""Exceptions. Carry the verdict so callers can audit without re-scanning."""
from __future__ import annotations

from .types import Verdict


class GuardrailError(Exception):
    pass


class GuardrailBlocked(GuardrailError):
    """Raised only by the `*_or_raise` helpers. The default API returns verdicts."""

    def __init__(self, verdict: Verdict, message: str | None = None) -> None:
        self.verdict = verdict
        super().__init__(message or f"blocked by {verdict.policy_id}")
