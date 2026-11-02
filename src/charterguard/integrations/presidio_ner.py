"""Microsoft Presidio `EntityRecognizer`. Requires `charterguard[presidio]`.

Choose Presidio over raw spaCy when you want the *recognisers*, not just the
NER: it bundles country-specific ID logic (US SSN, UK NHS number, AU TFN, IN
Aadhaar/PAN), checksum validation, and a context-word booster that raises a
span's score when supporting words are nearby ("SSN:" next to a 9-digit
number).

Overlap with the bundled rules is real and worth deciding deliberately.
`charterguard` already detects structured PII (email, phone, card with
Luhn, IBAN, IP) with no dependency. Two sane configurations:

  * **Names only** (default here): pass `entities=("PERSON",)` and keep the
    bundled rules for structured PII. Least duplication, least latency.
  * **Presidio for everything**: pass a wide `entities` tuple and accept that
    findings will be reported twice, once per detector, which the pipeline
    dedupes by span but which does double the scan cost.

Presidio loads a spaCy model underneath, so its first call carries model load
time and its memory footprint is the model's. Construct it once, at startup.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..pii import Entity

__all__ = ["PresidioEntityRecognizer", "PRESIDIO_LABEL_MAP"]

_INSTALL = (
    "PresidioEntityRecognizer needs presidio-analyzer and a spaCy model. Install with:\n"
    "    pip install 'charterguard[presidio]'\n"
    "    python -m spacy download en_core_web_lg"
)

PRESIDIO_LABEL_MAP: Mapping[str, str] = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "NRP": "PERSON",
    "MEDICAL_LICENSE": "MEDICAL",
    "US_SSN": "NATIONAL_ID",
    "UK_NHS": "MEDICAL",
    "IN_AADHAAR": "NATIONAL_ID",
    "IN_PAN": "NATIONAL_ID",
    "AU_TFN": "NATIONAL_ID",
    "DATE_TIME": "DATE",
}


class PresidioEntityRecognizer:
    """Args:
        entities: Presidio entity types to request. Narrow this -- asking for
            every recogniser runs every recogniser.
        language: must match a configured NLP engine language.
        min_score: Presidio returns real confidence scores, so this threshold
            is meaningful (unlike the spaCy adapter). 0.5 is Presidio's own
            default notion of "probable"; 0.85 is roughly "checksum-validated".
        analyzer: inject a pre-configured `AnalyzerEngine` to control the NLP
            engine, add custom recognisers, or load a non-default model.
    """

    __slots__ = ("_analyzer", "_entities", "_language", "_min_score", "_map")

    def __init__(
        self,
        *,
        entities: Sequence[str] = ("PERSON", "LOCATION"),
        language: str = "en",
        min_score: float = 0.5,
        label_map: Mapping[str, str] | None = None,
        analyzer: Any | None = None,
    ) -> None:
        if analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(_INSTALL) from exc
            analyzer = AnalyzerEngine()
        self._analyzer = analyzer
        self._entities = list(entities)
        self._language = language
        self._min_score = min_score
        self._map = dict(label_map or PRESIDIO_LABEL_MAP)

    def recognize(self, text: str) -> Sequence[Entity]:
        results = self._analyzer.analyze(
            text=text,
            entities=self._entities or None,
            language=self._language,
            score_threshold=self._min_score,
        )
        out: list[Entity] = []
        for r in results:
            label = self._map.get(r.entity_type, r.entity_type)
            out.append(Entity(label=label, start=int(r.start), end=int(r.end), score=float(r.score)))
        return out
