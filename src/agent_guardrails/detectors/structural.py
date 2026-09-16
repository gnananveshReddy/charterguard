"""Structural signals -- no rule text involved.

Catches the obfuscation layer itself rather than the payload: invisible
character smuggling, confusable-heavy text, and forged envelope delimiters.
These are strong signals precisely because legitimate enterprise content
almost never contains them in volume.
"""
from __future__ import annotations

from ..config import GuardrailConfig
from ..normalize import NormalizedText
from ..types import Category, Finding, Layer, Severity

_INVISIBLE_ABS = 4          # a handful of soft hyphens is normal; a dozen is not
_INVISIBLE_RATIO = 0.02
_CONFUSABLE_RATIO = 0.08


class StructuralDetector:
    name = "structural"

    def scan(self, nt: NormalizedText, layer: Layer, config: GuardrailConfig) -> list[Finding]:
        findings: list[Finding] = []
        n = max(len(nt.original), 1)

        if nt.invisible_chars >= _INVISIBLE_ABS and nt.invisible_chars / n >= _INVISIBLE_RATIO:
            findings.append(
                Finding(
                    rule_id="struct.invisible",
                    category=Category.INJECTION,
                    severity=Severity.HIGH,
                    score=0.82,
                    detector=self.name,
                    description=(
                        f"{nt.invisible_chars} zero-width/bidi control characters "
                        "-- consistent with hidden-instruction smuggling"
                    ),
                )
            )

        if nt.confusable_chars / n >= _CONFUSABLE_RATIO and nt.confusable_chars >= 8:
            findings.append(
                Finding(
                    rule_id="struct.confusable",
                    category=Category.INJECTION,
                    severity=Severity.MEDIUM,
                    score=0.62,
                    detector=self.name,
                    description=(
                        f"{nt.confusable_chars} homoglyph/compatibility characters "
                        "-- consistent with filter evasion"
                    ),
                )
            )

        if nt.truncated and layer in (Layer.UNTRUSTED, Layer.TOOL_OUTPUT):
            findings.append(
                Finding(
                    rule_id="struct.oversized",
                    category=Category.VOLUME,
                    severity=Severity.LOW,
                    score=0.30,
                    detector=self.name,
                    description="Content exceeded the scan budget; only head+tail inspected",
                )
            )
        return findings
