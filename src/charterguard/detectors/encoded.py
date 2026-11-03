"""Encoded-payload detector.

Injection is routinely wrapped in base64 / hex / percent-encoding so plain
pattern matching sees noise. This finds candidate encoded runs, decodes them
under a strict budget, and re-scans the decoded text with the injection rules.

Critical detail: this detector reads `nt.original`, NOT `nt.text`. The
normalizer casefolds, which destroys base64. Scanning normalized text here
silently misses every mixed-case base64 payload -- an easy bypass to ship by
accident, and the reason this detector owns its own text handling.

Cost control: base64 candidates are gated by `is_encoded_run` before any decode
is attempted, decoding is depth- and size-bounded, fan-out per level is capped,
and the whole detector can be disabled in config.
"""
from __future__ import annotations

import base64
import binascii
import re
from urllib.parse import unquote

from ..config import GuardrailConfig, ScanBudget
from ..normalize import NormalizedText, normalize
from ..patterns import rules_for
from ..scanner import compile_rules
from ..types import Category, Finding, Layer, Severity
from ..validators import is_encoded_run

# Fast-fail gate: one C-speed search decides whether any decode work is
# worthwhile at all. Ordinary prose has no 24-character unbroken token, so this
# skips the entire detector on the overwhelming majority of documents.
_ANY_LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_\-]{24,}|(?:%[0-9a-fA-F]{2}){6,}")

# Leading lookaheads make each candidate scan fail on the first character
# instead of entering the nested quantifier and backtracking.
_B64_RUN = re.compile(r"(?=[A-Za-z0-9+/_\-]{24})[A-Za-z0-9+/_\-]{24,4096}={0,2}")
_HEX_RUN = re.compile(r"(?=[0-9a-fA-F]{24})(?:[0-9a-fA-F]{2}){12,2048}")
_PCT_RUN = re.compile(r"(?:%[0-9a-fA-F]{2}){6,512}")

# Only instruction-shaped categories are worth re-scanning after a decode.
_DECODED_CATEGORIES = (
    Category.INJECTION,
    Category.EXFIL,
    Category.JAILBREAK,
    Category.COMMAND,
    Category.SECRET,
)


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isprintable() or ch in "\n\t\r" for ch in text) / len(text)


def _try_b64(chunk: str) -> str | None:
    padded = chunk + "=" * (-len(chunk) % 4)
    try:
        raw = base64.b64decode(padded, validate=False)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 8:
        return None
    text = raw.decode("utf-8", "ignore")
    return text if _printable_ratio(text) >= 0.85 else None


def _try_hex(chunk: str) -> str | None:
    try:
        raw = bytes.fromhex(chunk)
    except ValueError:
        return None
    if len(raw) < 8:
        return None
    text = raw.decode("utf-8", "ignore")
    return text if _printable_ratio(text) >= 0.85 else None


def _try_pct(chunk: str) -> str | None:
    out = unquote(chunk)
    return out if out != chunk and _printable_ratio(out) >= 0.85 else None


class EncodedPayloadDetector:
    name = "encoded"

    __slots__ = ("_ruleset",)

    def __init__(self) -> None:
        self._ruleset = compile_rules(
            rules_for(layer=Layer.UNTRUSTED, categories=_DECODED_CATEGORIES)
        )

    def scan(self, nt: NormalizedText, layer: Layer, config: GuardrailConfig) -> list[Finding]:
        if not config.enable_encoded_scan:
            return []
        budget = config.budget
        # Case must be preserved for base64, so work from the original text.
        source = nt.original
        if not _ANY_LONG_TOKEN.search(source):
            return []

        findings: list[Finding] = []
        seen: set[str] = set()
        if len(source) > budget.max_chars:
            source = (
                source[: budget.max_chars - budget.tail_chars] + source[-budget.tail_chars :]
            )

        for depth, decoded, start, end in self._walk(source, budget.decode_depth, budget, seen):
            inner = normalize(decoded, max_chars=budget.max_decoded_chars)
            for m in self._ruleset.scan(inner.text, limit=4):
                findings.append(
                    Finding(
                        rule_id=m.rule.id,
                        category=m.rule.category,
                        # Concealment is aggravating: an injection hidden inside
                        # base64 is never an accident.
                        severity=Severity.CRITICAL
                        if m.rule.severity >= Severity.HIGH
                        else Severity.HIGH,
                        score=0.90,
                        detector=self.name,
                        start=start,
                        end=end,
                        excerpt=inner.excerpt(m.start, m.end),
                        description=f"{m.rule.description} (decoded, depth {depth})",
                        decoded_from="base64/hex/percent",
                    )
                )
                if len(findings) >= 8:
                    return findings
        return findings

    def _walk(self, text: str, depth: int, budget: ScanBudget, seen: set[str]):
        """Yield (depth, decoded_text, run_start, run_end) for each decodable run."""
        if depth <= 0 or not text:
            return
        level = budget.decode_depth - depth + 1
        produced = 0
        for pattern, decoder in (
            (_PCT_RUN, _try_pct),
            (_B64_RUN, _try_b64),
            (_HEX_RUN, _try_hex),
        ):
            for m in pattern.finditer(text):
                chunk = m.group(0)
                if len(chunk) < budget.min_encoded_run:
                    continue
                if decoder is _try_b64 and not is_encoded_run(chunk):
                    continue
                out = decoder(chunk)
                if not out or out == chunk:
                    continue
                key = out[:64]
                if key in seen:
                    continue
                seen.add(key)
                out = out[: budget.max_decoded_chars]
                yield level, out, m.start(), m.end()
                produced += 1
                if produced >= 4:  # cap decode fan-out per level
                    return
                yield from self._walk(out, depth - 1, budget, seen)
