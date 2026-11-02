"""Embedding tier: the middle stage of a three-stage cascade.

Why this stage exists
---------------------
Deterministic rules match *shape*. Measured against JailbreakBench's harmful
behaviour goals -- plain requests with no override phrasing, no encoding, no
forged delimiter -- the rule tier fires on 3% of them, because there is no shape
to match. An LLM judge closes that gap and costs hundreds of milliseconds of
autoregressive generation to answer a yes/no question.

An embedding is a single encoder forward pass: one shot, no token-by-token
decode. That is the whole latency argument, and it is a real one. But the encoder
is not where an embedding tier goes wrong in production -- the plumbing around it
is. This module is mostly about the plumbing:

  * `PrototypeBank` scores a query against every category prototype using a
    pruned inverted index, so match cost scales with the number of non-zero
    dimensions in the query rather than with prototype count. numpy is used when
    present and is never required.
  * Embeddings are cached by content hash, because retrieval returns the same
    chunks constantly and re-encoding them is pure waste.
  * The API is batch-first (`aembed` takes a list). Retrieval fan-out is the
    dominant cost centre, and N chunks in one forward pass is several times
    cheaper than N sequential calls.
  * `CascadingScorer` promotes to an expensive judge only inside a band where the
    cheap tier is genuinely undecided, which is what keeps the LLM tier rare.

What it does not do
-------------------
It is not adversarially robust: text can be crafted to preserve harmful meaning
while moving its embedding, and this tier will not stop that. It still needs a
curated seed set per category -- a softer version of the enumeration problem, not
an escape from it -- and it still needs threshold calibration against a
false-positive budget. Treat it as a recall amplifier over rules, not a verdict.

No dependency is added to the core: `TextEmbedder` is a protocol. `HashingEmbedder`
is a dependency-free fallback that captures lexical overlap only; a real
sentence encoder is what makes this tier semantic.
"""
from __future__ import annotations

import asyncio
import math
import re
from array import array
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .scanner import ResultCache, content_key
from .types import Category, Layer

try:  # optional accelerator, never a requirement
    import numpy as _np
except ImportError:  # pragma: no cover - exercised in dependency-free envs
    _np = None

__all__ = [
    "TextEmbedder",
    "HashingEmbedder",
    "PrototypeBank",
    "SemanticScorer",
    "CascadingScorer",
    "numpy_available",
]


def numpy_available() -> bool:
    return _np is not None


class TextEmbedder(Protocol):
    """Batch-first embedding adapter.

    Batch-first is deliberate: the single largest latency lever for retrieval is
    encoding N chunks in one forward pass instead of N.
    """

    dim: int

    async def aembed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


# --------------------------------------------------------------------- fallback
_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Dependency-free hashed character n-gram embedder.

    Honest about what it is: lexical, not semantic. It captures paraphrase only
    to the extent that the paraphrase shares character n-grams, so "fabricate a
    false story" and "invent a fake news article" are near-orthogonal to it. It
    exists so the tier can be exercised, tested and benchmarked with no model
    present, and as a genuinely bundleable floor for deployments that cannot take
    a dependency at all.
    """

    __slots__ = ("dim", "_n", "_word_weight")

    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self._n = ngram
        self._word_weight = 2.0  # whole tokens are more informative than n-grams

    def _features(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = defaultdict(float)
        lowered = text.lower()
        for token in _TOKEN.findall(lowered):
            counts[hash(("w", token)) % self.dim] += self._word_weight
        squashed = re.sub(r"\s+", " ", lowered)
        lo, hi = self._n
        for size in range(lo, hi + 1):
            for i in range(0, max(0, len(squashed) - size + 1)):
                counts[hash(squashed[i : i + size]) % self.dim] += 1.0
        # sublinear scaling then L2 normalize: long documents must not dominate
        out = {d: 1.0 + math.log(v) for d, v in counts.items() if v > 0}
        norm = math.sqrt(sum(v * v for v in out.values())) or 1.0
        return {d: v / norm for d, v in out.items()}

    def embed_sparse(self, text: str) -> dict[int, float]:
        return self._features(text)

    async def aembed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = []
        for text in texts:
            dense = [0.0] * self.dim
            for d, v in self._features(text).items():
                dense[d] = v
            vectors.append(dense)
        return vectors


# ---------------------------------------------------------------------- bank
@dataclass(slots=True)
class _Prototype:
    category: Category
    label: str
    dims: array           # int32 indices, pruned
    weights: array        # float32, L2-renormalized after pruning


class PrototypeBank:
    """Category prototypes with a pruned inverted index.

    The naive implementation is a dense matrix-vector product per query. That is
    fine with numpy and slow in pure Python, so prototypes are pruned to their
    `top_k` heaviest dimensions and inverted into `dim -> [(proto, weight)]`.
    Query cost then scales with the query's non-zero count times the average
    posting-list length, not with prototype count -- the same reasoning as the
    literal prefilter in front of the rule tier.

    Pruning is lossy by construction. `top_k` trades recall for latency, and
    `tests/test_semantic.py` pins the loss on the bundled seeds.
    """

    __slots__ = ("_protos", "_index", "_dim", "_top_k", "_matrix", "_meta", "_dirty")

    def __init__(self, dim: int, *, top_k: int = 96) -> None:
        self._dim = dim
        self._top_k = top_k
        self._protos: list[_Prototype] = []
        self._index: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._matrix = None            # numpy dense path, built lazily
        self._meta: list[tuple[Category, str]] = []
        self._dirty = True

    def __len__(self) -> int:
        return len(self._protos)

    @property
    def categories(self) -> set[Category]:
        return {p.category for p in self._protos}

    def add_sparse(self, category: Category, label: str, vector: Mapping[int, float]) -> None:
        top = sorted(vector.items(), key=lambda kv: -abs(kv[1]))[: self._top_k]
        norm = math.sqrt(sum(v * v for _, v in top)) or 1.0
        dims = array("i", [d for d, _ in top])
        weights = array("f", [v / norm for _, v in top])
        self._protos.append(_Prototype(category, label, dims, weights))
        self._meta.append((category, label))
        self._dirty = True

    def add_dense(self, category: Category, label: str, vector: Sequence[float]) -> None:
        self.add_sparse(category, label, {i: v for i, v in enumerate(vector) if v})

    def finalize(self) -> PrototypeBank:
        """Build the inverted index (and the numpy matrix when available)."""
        self._index = defaultdict(list)
        for pid, proto in enumerate(self._protos):
            for d, w in zip(proto.dims, proto.weights):
                self._index[d].append((pid, w))
        if _np is not None and self._protos:
            matrix = _np.zeros((len(self._protos), self._dim), dtype=_np.float32)
            for pid, proto in enumerate(self._protos):
                matrix[pid, _np.asarray(proto.dims, dtype=_np.int32)] = _np.asarray(
                    proto.weights, dtype=_np.float32
                )
            self._matrix = matrix
        self._dirty = False
        return self

    # ------------------------------------------------------------------ scoring
    def score_sparse(self, query: Mapping[int, float]) -> dict[Category, float]:
        """Pure-Python path: accumulate over posting lists of present dims only."""
        if self._dirty:
            self.finalize()
        acc: dict[int, float] = defaultdict(float)
        index = self._index
        for d, qv in query.items():
            postings = index.get(d)
            if postings:
                for pid, w in postings:
                    acc[pid] += qv * w
        out: dict[Category, float] = {}
        for pid, sim in acc.items():
            category = self._meta[pid][0]
            if sim > out.get(category, 0.0):
                out[category] = sim
        return out

    def score_dense(self, query: Sequence[float]) -> dict[Category, float]:
        """numpy path: one matrix-vector product over all prototypes."""
        if self._dirty:
            self.finalize()
        if _np is None or self._matrix is None:
            return self.score_sparse({i: v for i, v in enumerate(query) if v})
        sims = self._matrix @ _np.asarray(query, dtype=_np.float32)
        out: dict[Category, float] = {}
        for pid, sim in enumerate(sims.tolist()):
            category = self._meta[pid][0]
            if sim > out.get(category, 0.0):
                out[category] = sim
        return out

    def best_match(self, query: Mapping[int, float]) -> tuple[str, float] | None:
        if self._dirty:
            self.finalize()
        acc: dict[int, float] = defaultdict(float)
        for d, qv in query.items():
            for pid, w in self._index.get(d, ()):
                acc[pid] += qv * w
        if not acc:
            return None
        pid, sim = max(acc.items(), key=lambda kv: kv[1])
        return self._meta[pid][1], sim


# -------------------------------------------------------------------- scorers
class SemanticScorer:
    """`ModelScorer` implementation backed by embeddings + a prototype bank.

    It implements the existing `ModelScorer` protocol on purpose: the cascade,
    the per-layer escalation bands, the clean-band sampling, the failure policy
    and the audit path all already exist, so the embedding tier needs no new
    plumbing in the pipeline.
    """

    __slots__ = ("_embedder", "_bank", "_cache", "_scale", "_floor", "_max_chars")

    def __init__(
        self,
        embedder: TextEmbedder,
        bank: PrototypeBank,
        *,
        cache_size: int = 2048,
        floor: float = 0.30,
        scale: float = 1.35,
        max_chars: int = 2_000,
    ) -> None:
        self._embedder = embedder
        self._bank = bank
        self._cache = ResultCache(cache_size)
        self._floor = floor
        self._scale = scale
        # Truncation is a latency decision AND a fidelity decision. Measured: the
        # bundled pure-Python embedder costs ~190 us on a sentence and ~33 ms on
        # 4 KB, so an untruncated document is the single worst cost in the tier.
        # Real sentence encoders truncate at 256-512 tokens regardless, so a long
        # document is never fully represented by one vector anyway -- chunk it and
        # score the chunks rather than pretending one vector covers 64 KB.
        self._max_chars = max_chars

    @property
    def cache(self) -> ResultCache:
        return self._cache

    def _calibrate(self, similarity: float) -> float:
        """Cosine similarity is not a probability. Map it through a floor and a
        gain so thresholds mean the same thing here as in the rule tier."""
        if similarity <= self._floor:
            return 0.0
        return min(1.0, (similarity - self._floor) * self._scale / (1.0 - self._floor))

    async def score(
        self, text: str, categories: Sequence[Category], layer: Layer
    ) -> Mapping[Category, float]:
        scores = await self.ascore_batch([text], categories, layer)
        return scores[0]

    async def ascore_batch(
        self, texts: Sequence[str], categories: Sequence[Category], layer: Layer
    ) -> list[dict[Category, float]]:
        """One encoder pass for the whole batch; cache hits never reach it."""
        texts = [t[: self._max_chars] for t in texts]
        keys = [content_key(t) for t in texts]
        pending: list[int] = []
        vectors: list[Sequence[float] | None] = []
        for i, key in enumerate(keys):
            cached = self._cache.get(key)
            vectors.append(cached)  # type: ignore[arg-type]
            if cached is None:
                pending.append(i)
        if pending:
            fresh = await self._embedder.aembed([texts[i] for i in pending])
            for slot, vector in zip(pending, fresh):
                vectors[slot] = vector
                self._cache.put(keys[slot], vector)

        wanted = set(categories)
        out: list[dict[Category, float]] = []
        for vector in vectors:
            # Measured: with numpy present the dense matrix product beats the
            # pure-Python inverted index by ~30x at 30 prototypes and ~29x at
            # 3000, because the index still pays Python loop overhead per
            # non-zero query dimension. The index is the no-numpy fallback, not
            # the fast path.
            raw = self._bank.score_dense(vector or [])
            out.append(
                {
                    category: self._calibrate(sim)
                    for category, sim in raw.items()
                    if category in wanted and self._calibrate(sim) > 0.0
                }
            )
        return out


class CascadingScorer:
    """Cheap scorer first; promote to an expensive judge only when undecided.

    This is the third stage. The embedding tier resolves "paraphrase of something
    we have seen" cheaply; the LLM judge is reserved for the band where the cheap
    tier is genuinely uncertain, which is what keeps its call rate -- and its cost
    -- low. `promoted` and `calls` are counters, because the only way to know the
    cascade is working is to watch the promotion rate in production.
    """

    __slots__ = ("_fast", "_slow", "_lo", "_hi", "_timeout", "calls", "promoted")

    def __init__(
        self,
        fast: object,
        slow: object,
        *,
        promote_low: float = 0.25,
        promote_high: float = 0.80,
        slow_timeout_s: float = 2.0,
    ) -> None:
        self._fast = fast
        self._slow = slow
        self._lo = promote_low
        self._hi = promote_high
        self._timeout = slow_timeout_s
        self.calls = 0
        self.promoted = 0

    @property
    def promotion_rate(self) -> float:
        return self.promoted / self.calls if self.calls else 0.0

    async def score(
        self, text: str, categories: Sequence[Category], layer: Layer
    ) -> Mapping[Category, float]:
        self.calls += 1
        fast = dict(await self._fast.score(text, categories, layer))
        top = max(fast.values(), default=0.0)
        if top >= self._hi or top < self._lo:
            return fast  # confident either way: do not pay for the judge
        self.promoted += 1
        try:
            slow = await asyncio.wait_for(
                self._slow.score(text, categories, layer), timeout=self._timeout
            )
        except Exception:  # noqa: BLE001 - the pipeline's fail policy owns this
            return fast
        merged = dict(fast)
        for category, value in slow.items():
            merged[category] = max(merged.get(category, 0.0), value)
        return merged


def cascade_latency_ms(
    *, embed_ms: float, match_ms: float, judge_ms: float, promotion_rate: float
) -> float:
    """Expected per-item latency of a three-stage cascade.

    The arithmetic that decides whether stage 2 is worth adding at all:

        total = embed + match + promotion_rate * judge

    So the embedding tier reduces latency only in proportion to how often it
    keeps the judge out of the path. Measured with the bundled lexical embedder,
    the promotion rate was 100% -- every item landed in the undecided band -- and
    the cascade was 1% *slower* than calling the judge directly. A cheap tier that
    cannot decide is pure overhead. Calibrate the promotion rate on real traffic
    before claiming a saving.
    """
    return embed_ms + match_ms + promotion_rate * judge_ms


def cascade_breakeven(*, embed_ms: float, match_ms: float, judge_ms: float) -> float:
    """Promotion rate above which the cascade costs more than always judging."""
    if judge_ms <= 0:
        return 0.0
    return max(0.0, 1.0 - (embed_ms + match_ms) / judge_ms)
