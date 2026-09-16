"""Free-text PII: the adapter, and an honest statement of the gap.

What the bundled rules do: match PII that has *structure* -- emails, phone
numbers, card numbers (Luhn-checked), national IDs, IBANs, IPs. Precision is
high because the format itself is the signal.

What they cannot do: recognise PII that has no format. "Priya Sharma was
discharged on Tuesday" contains a name and a health fact, and no regex
distinguishes it from "Priya Sharma reviewed the pull request". Deciding that
requires named-entity recognition, and NER requires a model -- which means
weights, a runtime, and a licence, none of which belong in a dependency-free
package that a security team has to audit line by line.

So this module ships the *seam* rather than a guess: implement
`EntityRecognizer` over spaCy, Presidio, a cloud DLP API, or an internal
service, wrap it in `EntityDetector`, and pass it to the pipeline as a detector.
Everything downstream -- policy modes, redaction, ledger accounting, audit --
already treats the resulting findings identically to rule findings.

Do not read the absence of a bundled recogniser as "free-text PII is covered".
It is not. Under GDPR or HIPAA scope, treat NER as required and wire it here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import GuardrailConfig
from .normalize import NormalizedText
from .types import Category, Finding, Layer, Severity

__all__ = ["Entity", "EntityRecognizer", "EntityDetector"]


@dataclass(frozen=True, slots=True)
class Entity:
    """One recognised span, in NORMALIZED offsets.

    Offsets must index the normalized text handed to the recogniser. The
    pipeline maps them back to original offsets for audit and redaction, so a
    recogniser that returns original offsets will produce misplaced spans.
    """

    label: str            # "PERSON", "MEDICAL", "LOCATION", ...
    start: int
    end: int
    score: float = 0.8


class EntityRecognizer(Protocol):
    """Bring your own NER.

    Implementations must be thread-safe -- the sync scan path may run inside a
    worker thread via the async entry points -- and should be bounded in
    latency, because they execute inline with enforcement.
    """

    def recognize(self, text: str) -> Sequence[Entity]:
        ...


# Labels that carry regulatory weight by default. Everything else is recorded
# at LOW so it is visible in audit without driving enforcement.
_SEVERITY_BY_LABEL = {
    "PERSON": Severity.MEDIUM,
    "MEDICAL": Severity.HIGH,
    "HEALTH": Severity.HIGH,
    "FINANCIAL": Severity.HIGH,
    "CREDENTIAL": Severity.HIGH,
    "GOV_ID": Severity.HIGH,
    "LOCATION": Severity.LOW,
    "ORG": Severity.LOW,
}


class EntityDetector:
    """Adapts an `EntityRecognizer` to the `Detector` protocol."""

    __slots__ = ("_recognizer", "_min_score", "_labels")

    def __init__(
        self,
        recognizer: EntityRecognizer,
        *,
        min_score: float = 0.5,
        labels: Sequence[str] | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._min_score = min_score
        self._labels = frozenset(labels) if labels else None

    def scan(
        self, nt: NormalizedText, layer: Layer, config: GuardrailConfig
    ) -> list[Finding]:
        out: list[Finding] = []
        for ent in self._recognizer.recognize(nt.text):
            if ent.score < self._min_score:
                continue
            if self._labels is not None and ent.label not in self._labels:
                continue
            start, end = nt.span(ent.start, ent.end)
            out.append(
                Finding(
                    rule_id=f"pii.ner.{ent.label.lower()}",
                    category=Category.PII,
                    severity=_SEVERITY_BY_LABEL.get(ent.label.upper(), Severity.LOW),
                    score=ent.score,
                    detector="ner",
                    start=start,
                    end=end,
                    excerpt=f"<{ent.label.lower()}>",  # never echo the value itself
                    description=f"NER entity: {ent.label}",
                )
            )
            if len(out) >= config.budget.max_findings:
                break
        return out
