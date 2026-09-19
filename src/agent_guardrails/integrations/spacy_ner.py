"""spaCy `EntityRecognizer`. Requires `agent_guardrails[spacy]` plus a model.

Install is two steps, and the second one is the one people forget:

    pip install 'agent_guardrails[spacy]'
    python -m spacy download en_core_web_sm

Model choice is a real precision/latency decision, not a default to accept:
`en_core_web_sm` is fast and misses names in noisy text; `en_core_web_trf` is
transformer-based, materially more accurate on person names, and one to two
orders of magnitude slower. For a request-path guardrail on a CPU box, `sm` or
`md` is usually the only viable choice, and its recall gap is the reason
`min_score` and the policy mode matter.

Two properties this adapter enforces:

  * **Offsets index the string it was given.** spaCy reports character offsets
    into the exact input, which is the normalized text -- so spans line up with
    what the pipeline maps back to the original. Do not pre-process the text
    before handing it here.
  * **Bounded work per call.** `nlp.max_length` is respected by truncating, and
    non-NER pipeline components are excluded at load, which is typically a 2-4x
    latency win because the tagger, parser and lemmatizer are pure overhead for
    this use.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..pii import Entity

__all__ = ["SpacyEntityRecognizer", "SPACY_LABEL_MAP"]

_INSTALL = (
    "SpacyEntityRecognizer needs spaCy and a model. Install with:\n"
    "    pip install 'agent_guardrails[spacy]'\n"
    "    python -m spacy download en_core_web_sm"
)

# spaCy's OntoNotes labels, filtered to the ones that are actually personal
# data. NORP/ORG/PRODUCT are excluded by default: an organisation name is not
# PII on its own, and including it makes redaction unusable on business text.
SPACY_LABEL_MAP: Mapping[str, str] = {
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE",
    "TIME": "DATE",
}

# Components that do not contribute to NER. Excluding rather than disabling
# means they are never loaded, so the memory is never allocated.
_EXCLUDE = ("lemmatizer", "tagger", "parser", "attribute_ruler", "senter", "textcat")


class SpacyEntityRecognizer:
    """Args:
        model: spaCy model name or path.
        label_map: spaCy label -> guardrail entity label. Defaults to
            `SPACY_LABEL_MAP`; pass a wider map to include ORG/NORP knowingly.
        min_score: spaCy's `ent` objects carry no probability under the default
            pipeline, so a fixed confidence is assigned per label instead of a
            fabricated per-span score. This is honest: the number downstream is
            a policy weight, not a model probability.
        max_chars: hard truncation, defaulting to the model's own max_length.
    """

    __slots__ = ("_nlp", "_map", "_score", "_max_chars")

    def __init__(
        self,
        model: str = "en_core_web_sm",
        *,
        label_map: Mapping[str, str] | None = None,
        score: float = 0.8,
        max_chars: int | None = None,
        exclude: Iterable[str] = _EXCLUDE,
    ) -> None:
        try:
            import spacy  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(_INSTALL) from exc
        try:
            self._nlp = spacy.load(model, exclude=list(exclude))
        except OSError as exc:  # model not downloaded
            raise ImportError(
                f"spaCy model {model!r} is not installed. Run:\n"
                f"    python -m spacy download {model}"
            ) from exc
        if "ner" not in self._nlp.pipe_names:
            raise ValueError(
                f"spaCy model {model!r} has no 'ner' component; it cannot recognise entities"
            )
        self._map = dict(label_map or SPACY_LABEL_MAP)
        self._score = score
        self._max_chars = max_chars or int(getattr(self._nlp, "max_length", 100_000))

    def recognize(self, text: str) -> Sequence[Entity]:
        if len(text) > self._max_chars:
            # Truncate rather than raise: a guardrail that throws on a long
            # document fails the request instead of scanning the part it can.
            text = text[: self._max_chars]
        doc = self._nlp(text)
        out: list[Entity] = []
        for ent in doc.ents:
            label = self._map.get(ent.label_)
            if label is None:
                continue
            out.append(Entity(label=label, start=ent.start_char, end=ent.end_char, score=self._score))
        return out

    def recognize_batch(self, texts: Sequence[str]) -> list[Sequence[Entity]]:
        """Batched variant. spaCy's `nlp.pipe` is substantially faster than a
        loop over `nlp()`; use it for offline scanning of stored documents."""
        results: list[Sequence[Entity]] = []
        for doc in self._nlp.pipe([t[: self._max_chars] for t in texts]):
            results.append(
                [
                    Entity(self._map[e.label_], e.start_char, e.end_char, self._score)
                    for e in doc.ents
                    if e.label_ in self._map
                ]
            )
        return results
