"""Egress redaction.

Applied to model output and to tool output before it enters the context window.
Redaction replaces the value with a typed placeholder plus a short digest, so
audit can prove that a specific secret was seen and suppressed without ever
persisting the secret itself.

Spans arrive in ORIGINAL coordinates from the normalizer, so replacement is
exact -- no second scan of the raw string, no offset guesswork.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Sequence

from .config import RedactionConfig
from .types import Finding

_KIND_BY_PREFIX = {
    "sec.": "SECRET",
    "pii.email": "EMAIL",
    "pii.phone": "PHONE",
    "pii.card": "CARD",
    "pii.ssn": "SSN",
    "pii.aadhaar": "AADHAAR",
    "pii.pan": "PAN",
    "pii.gstin": "GSTIN",
    "pii.upi": "UPI",
    "pii.passport": "PASSPORT",
    "pii.iban": "IBAN",
    "pii.mrn": "MRN",
    "pii.dob": "DOB",
    "pii.ipv4": "IP",
}


def _kind_for(rule_id: str) -> str:
    for prefix, kind in _KIND_BY_PREFIX.items():
        if rule_id.startswith(prefix):
            return kind
    return "SENSITIVE"


def value_digest(value: str, chars: int = 12) -> str:
    return hashlib.blake2b(value.encode("utf-8", "ignore"), digest_size=16).hexdigest()[:chars]


@dataclass(frozen=True, slots=True)
class Redaction:
    kind: str
    rule_id: str
    start: int
    end: int
    digest: str


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    redactions: tuple[Redaction, ...]

    @property
    def count(self) -> int:
        return len(self.redactions)

    @property
    def changed(self) -> bool:
        return bool(self.redactions)

    def to_audit(self) -> list[dict[str, object]]:
        return [
            {
                "kind": r.kind,
                "rule_id": r.rule_id,
                "span": [r.start, r.end],
                "value_digest": r.digest,
            }
            for r in self.redactions
        ]


def redact(
    text: str,
    findings: Sequence[Finding],
    config: RedactionConfig,
) -> RedactionResult:
    if not config.enabled or not findings:
        return RedactionResult(text, ())

    targets = [
        f
        for f in findings
        if f.category in config.redact_categories and f.end > f.start
    ]
    if not targets:
        return RedactionResult(text, ())

    # Right-to-left replacement keeps earlier spans valid. Overlapping spans
    # are merged into their UNION: dropping a partially overlapping finding
    # would leave its tail in the clear ("1234-4111-1111-1111-1111": a phone
    # match on the head, a card on the rest). The widest finding at each start
    # names the merged span.
    targets.sort(key=lambda f: (f.start, -(f.end - f.start)))
    merged: list[tuple[int, int, Finding]] = []
    for f in targets:
        if merged and f.start < merged[-1][1]:
            s, e, owner = merged[-1]
            merged[-1] = (s, max(e, f.end), owner)
            continue
        merged.append((f.start, f.end, f))

    out = text
    redactions: list[Redaction] = []
    for start, end, f in reversed(merged):
        raw = text[start:end]
        kind = _kind_for(f.rule_id)
        digest = value_digest(raw, config.digest_chars)
        placeholder = config.placeholder.format(kind=kind)
        if config.keep_digest:
            placeholder = f"{placeholder[:-1]}:{digest}]" if placeholder.endswith("]") else placeholder
        out = out[:start] + placeholder + out[end:]
        redactions.append(Redaction(kind, f.rule_id, start, end, digest))

    redactions.reverse()
    return RedactionResult(out, tuple(redactions))
