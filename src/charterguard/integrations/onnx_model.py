"""Universal local-model tier: pinned, checksummed ONNX. `charterguard[onnx]`.

"Any model should just work" is the right goal and the wrong implementation.
Made literal, it means downloading whatever a repo name resolves to today and
handing it to a deserializer -- and the standard PyTorch formats (`.bin`, `.pt`,
`.pth`, `.ckpt`) are pickle streams, which execute arbitrary code on load.
`torch.load` of an untrusted file is remote code execution, full stop. A
guardrail that owns the block/allow decision is the last component in a system
that should be loading one.

So "universal" here means *any model that fits a declared, verifiable contract*:

  1. **Format allowlist.** ONNX graph plus tokenizer JSON. Pickle-bearing
     suffixes are refused by name before a byte is fetched, and the ONNX magic
     bytes are checked after.
  2. **Immutable revision.** `revision` must be a full 40-hex commit SHA. A
     branch name like `main` is refused: it silently re-points, which turns
     "the model we validated" into "whatever was pushed last night".
  3. **SHA-256 per file.** Every file carries an expected digest, verified on
     download and on load from cache. A mismatch raises and deletes the file;
     it never falls back to running it.
  4. **Explicit label map.** Nothing can infer that a model's `LABEL_1` means
     `Category.INJECTION`. The caller states it, or the adapter refuses.

That is genuinely universal across the useful space -- any HF classifier
exported to ONNX, or any internally trained one -- while being honest that
"loads any checkpoint" is a vulnerability, not a feature.

What this does *not* give you: provenance for the weights themselves. A pinned
SHA proves you got the same bytes every time, not that those bytes are a
well-behaved classifier. Evaluate the model on your own corpus before trusting
it, and record which revision was evaluated.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from ..types import Category, Layer

__all__ = [
    "FileSpec",
    "OnnxModelSpec",
    "ModelIntegrityError",
    "ensure_local",
    "OnnxTextClassifier",
    "OnnxModelScorer",
    "OnnxEmbedder",
]

_INSTALL = (
    "The local ONNX tier needs onnxruntime and tokenizers. Install with:\n"
    "    pip install 'charterguard[onnx]'"
)

# Extensions that are safe to fetch: protobuf graphs and JSON/text tokenizers.
ALLOWED_SUFFIXES = frozenset({".onnx", ".onnx_data", ".json", ".txt", ".model", ".vocab"})

# Extensions that are pickle-based or otherwise execute on load. Refused by
# name, with a message that says why, because "just support .bin too" is the
# single most likely future request against this module.
DENIED_SUFFIXES: Mapping[str, str] = {
    ".bin": "PyTorch pickle - executes arbitrary code on load",
    ".pt": "PyTorch pickle - executes arbitrary code on load",
    ".pth": "PyTorch pickle - executes arbitrary code on load",
    ".ckpt": "checkpoint pickle - executes arbitrary code on load",
    ".pkl": "pickle - executes arbitrary code on load",
    ".pickle": "pickle - executes arbitrary code on load",
    ".joblib": "joblib pickle - executes arbitrary code on load",
    ".h5": "Keras HDF5 - can carry executable Lambda layers",
    ".msgpack": "Flax msgpack - not verifiable by this loader",
    ".safetensors": "safe to load, but not executable by onnxruntime; export to ONNX first",
}

_ONNX_MAGIC = b"\x08"  # protobuf field 1 (ir_version), varint - every ONNX file starts here
_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB ceiling per file
_HF_BASE = "https://huggingface.co"


class ModelIntegrityError(RuntimeError):
    """Raised when a model file fails format, size or digest validation.

    Never caught internally: an unverifiable model is a hard stop, not a
    degraded mode.
    """


@dataclass(frozen=True, slots=True)
class FileSpec:
    """One artifact and its expected SHA-256 (lowercase hex, 64 chars).

    Get the digest by downloading once and hashing locally
    (``sha256sum model.onnx``), then commit it. Do not copy a digest from a
    page you did not verify.
    """

    name: str
    sha256: str

    def __post_init__(self) -> None:
        suffix = Path(self.name).suffix.lower()
        if suffix in DENIED_SUFFIXES:
            raise ModelIntegrityError(
                f"refusing {self.name!r}: {DENIED_SUFFIXES[suffix]}. "
                "Export the model to ONNX (optimum-cli export onnx) and pin that instead."
            )
        if suffix not in ALLOWED_SUFFIXES:
            raise ModelIntegrityError(
                f"refusing {self.name!r}: extension {suffix!r} is not in the allowlist "
                f"{sorted(ALLOWED_SUFFIXES)}"
            )
        if ".." in self.name or self.name.startswith("/"):
            raise ModelIntegrityError(f"refusing path-traversing filename {self.name!r}")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower()):
            raise ModelIntegrityError(f"{self.name}: sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class OnnxModelSpec:
    """A fully pinned, verifiable model.

    Args:
        repo_id: HF repo (``org/name``) or any base URL if `base_url` is set.
            Operator-configured only -- never build this from request data.
        revision: full 40-hex commit SHA. Branch names are refused.
        onnx_file: the graph, with digest.
        tokenizer_file: a `tokenizers`-compatible ``tokenizer.json``, with digest.
        label_map: model output label (or ``LABEL_0`` style index name) ->
            guardrail `Category`. Required.
        max_length: token truncation. Bounds worst-case latency; a model that
            accepts 512 tokens will happily spend 10x on a 4k-token document.
        activation: ``sigmoid`` for multi-label heads (independent categories),
            ``softmax`` for single-label heads (mutually exclusive classes).
            Getting this wrong does not error -- it silently miscalibrates every
            threshold in your policy.
    """

    repo_id: str
    revision: str
    onnx_file: FileSpec
    tokenizer_file: FileSpec
    label_map: Mapping[str, Category]
    max_length: int = 256
    activation: str = "sigmoid"
    extra_files: tuple[FileSpec, ...] = field(default_factory=tuple)
    base_url: str = _HF_BASE

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(
            c not in "0123456789abcdef" for c in self.revision.lower()
        ):
            raise ModelIntegrityError(
                f"revision {self.revision!r} is not a full 40-hex commit SHA. "
                "Branch and tag names re-point without notice; pin the commit."
            )
        if not self.label_map:
            raise ModelIntegrityError(
                "label_map is required: a model's output labels cannot be mapped to "
                "guardrail categories by inference, and guessing wrong means a "
                "detector that reports the wrong category or nothing at all."
            )
        if self.activation not in ("sigmoid", "softmax", "none"):
            raise ModelIntegrityError("activation must be 'sigmoid', 'softmax' or 'none'")
        parts = urllib.parse.urlsplit(self.base_url)
        if parts.scheme.lower() != "https" or not parts.hostname:
            raise ModelIntegrityError("base_url must be an https URL with a host")
        if self.max_length < 8 or self.max_length > 8192:
            raise ModelIntegrityError("max_length must be between 8 and 8192")

    @property
    def files(self) -> tuple[FileSpec, ...]:
        return (self.onnx_file, self.tokenizer_file, *self.extra_files)

    def url_for(self, name: str) -> str:
        return f"{self.base_url}/{self.repo_id}/resolve/{self.revision}/{name}"

    def cache_key(self) -> str:
        return f"{self.repo_id.replace('/', '__')}@{self.revision[:12]}"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never off https.

    Model hubs redirect file downloads to a CDN, so redirects must be followed.
    The SHA-256 check already makes a tampered file harmless, but a downgrade to
    http still leaks which model and revision this host loads, and a redirect
    to ``file:`` or another scheme has no legitimate use.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https":
            raise ModelIntegrityError(f"refusing redirect off https to {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsOnlyRedirect)


def _download(url: str, dest: Path, expected: str, timeout_s: float) -> None:
    """Stream to a temp file, verify, then atomically rename.

    The order matters: verify-then-rename means a truncated or tampered
    download never appears at the final path, so a crashed process cannot leave
    behind a file that a later run trusts because it exists.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    tmp = Path(tmp_name)
    os.close(tmp_fd)
    try:
        if urllib.parse.urlsplit(url).scheme.lower() != "https":
            raise ModelIntegrityError(f"refusing non-https download {url!r}")
        req = urllib.request.Request(url, headers={"user-agent": "charterguard"})  # noqa: S310 - https checked
        h = hashlib.sha256()
        total = 0
        # https enforced above and on every redirect hop.
        with _OPENER.open(req, timeout=timeout_s) as resp, tmp.open("wb") as out:  # noqa: S310
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                total += len(block)
                if total > _MAX_BYTES:
                    raise ModelIntegrityError(f"{url} exceeds the {_MAX_BYTES} byte ceiling")
                h.update(block)
                out.write(block)
        actual = h.hexdigest()
        if actual != expected.lower():
            raise ModelIntegrityError(
                f"digest mismatch for {url}\n  expected {expected}\n  actual   {actual}\n"
                "The bytes served are not the bytes you pinned. Not loading."
            )
        if dest.suffix.lower() == ".onnx":
            with tmp.open("rb") as fh:
                if fh.read(1) != _ONNX_MAGIC:
                    raise ModelIntegrityError(
                        f"{url} does not begin with the ONNX protobuf marker; refusing to load"
                    )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def ensure_local(
    spec: OnnxModelSpec,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    timeout_s: float = 60.0,
    allow_download: bool = True,
) -> Path:
    """Materialise `spec` locally, verifying every file. Returns the directory.

    Cached files are re-hashed on every call. That costs a disk read of the
    model at startup and buys detection of on-disk tampering or bit rot -- the
    right trade for a security component that starts once and runs for weeks.

    Set `allow_download=False` for air-gapped or locked-down deployments: files
    must already be present and correct, or this raises.
    """
    root = Path(cache_dir) if cache_dir else Path(
        os.environ.get("AGENT_GUARDRAILS_MODEL_CACHE", Path.home() / ".cache" / "charterguard")
    )
    target = root / spec.cache_key()
    for fs in spec.files:
        path = target / fs.name
        if path.exists():
            actual = _sha256_file(path)
            if actual == fs.sha256.lower():
                continue
            # A wrong file on disk is deleted, not reused and not renamed.
            path.unlink()
            if not allow_download:
                raise ModelIntegrityError(
                    f"{path} failed verification and downloads are disabled"
                )
        if not allow_download:
            raise ModelIntegrityError(f"{path} is missing and downloads are disabled")
        _download(spec.url_for(fs.name), path, fs.sha256, timeout_s)
    return target


class OnnxTextClassifier:
    """Thin, synchronous onnxruntime + tokenizers wrapper.

    Threading note that matters in a web service: onnxruntime defaults to using
    every core for intra-op parallelism. Under gunicorn with N workers that is
    N x cores threads fighting each other, and p99 latency gets worse as you add
    workers. `intra_op_threads=1` is the default here; raise it only for a
    single-process, batch workload.
    """

    __slots__ = ("_sess", "_tok", "_spec", "_input_names", "_labels", "dim")

    def __init__(
        self,
        spec: OnnxModelSpec,
        model_dir: Path,
        *,
        intra_op_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(_INSTALL) from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU provider only: GPU providers pull in a CUDA stack whose failure
        # modes (driver mismatch, OOM) are not worth taking on a request path.
        self._sess = ort.InferenceSession(
            str(model_dir / spec.onnx_file.name),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._tok = Tokenizer.from_file(str(model_dir / spec.tokenizer_file.name))
        self._tok.enable_truncation(max_length=spec.max_length)
        self._spec = spec
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self._labels = list(spec.label_map.keys())

    def _encode(self, texts: Sequence[str]) -> dict[str, Any]:
        import numpy as np

        self._tok.enable_padding()
        encodings = self._tok.encode_batch(list(texts))
        feed: dict[str, Any] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )
        return {k: v for k, v in feed.items() if k in self._input_names}

    def logits(self, texts: Sequence[str]) -> Any:
        feed = self._encode(texts)
        missing = self._input_names - set(feed)
        if missing:
            raise ModelIntegrityError(
                f"model expects inputs {sorted(self._input_names)} but this adapter "
                f"can only supply input_ids/attention_mask/token_type_ids (missing {sorted(missing)})"
            )
        return self._sess.run(None, feed)[0]

    def scores(self, text: str) -> Mapping[str, float]:
        """Label -> probability for one text."""
        import numpy as np

        raw = np.asarray(self.logits([text])[0], dtype="float64")
        if self._spec.activation == "sigmoid":
            probs = 1.0 / (1.0 + np.exp(-raw))
        elif self._spec.activation == "softmax":
            shifted = raw - raw.max()
            exp = np.exp(shifted)
            probs = exp / exp.sum()
        else:
            probs = raw
        n = min(len(probs), len(self._labels))
        if len(probs) != len(self._labels):
            # Loud, because a silent zip() truncation here means some categories
            # are never scored and the tier looks like it is working.
            raise ModelIntegrityError(
                f"model produced {len(probs)} outputs but label_map declares "
                f"{len(self._labels)}; fix the label map to match the model head"
            )
        return {self._labels[i]: float(probs[i]) for i in range(n)}


class OnnxModelScorer:
    """`ModelScorer` over a local ONNX classifier.

    Inference runs in a worker thread (`asyncio.to_thread`) because
    onnxruntime's `run` is blocking C++ that would otherwise stall the event
    loop for the whole forward pass -- typically 5-40 ms on CPU for a small
    encoder, which is enough to wreck tail latency under concurrency.
    """

    __slots__ = ("_clf", "_spec", "_reverse")

    def __init__(self, spec: OnnxModelSpec, *, cache_dir: str | os.PathLike[str] | None = None,
                 allow_download: bool = True, intra_op_threads: int = 1) -> None:
        model_dir = ensure_local(spec, cache_dir, allow_download=allow_download)
        self._clf = OnnxTextClassifier(spec, model_dir, intra_op_threads=intra_op_threads)
        self._spec = spec
        self._reverse: dict[Category, list[str]] = {}
        for label, cat in spec.label_map.items():
            self._reverse.setdefault(cat, []).append(label)

    async def score(
        self, text: str, categories: Sequence[Category], layer: Layer
    ) -> Mapping[Category, float]:
        raw = await asyncio.to_thread(self._clf.scores, text)
        out: dict[Category, float] = {}
        for cat in categories:
            labels = self._reverse.get(cat)
            if not labels:
                continue  # this model has nothing to say about this category
            # max, not mean: several labels mapping to one category are
            # alternative evidence for it, and averaging dilutes a strong hit.
            out[cat] = max(raw.get(lbl, 0.0) for lbl in labels)
        return out


class OnnxEmbedder:
    """`TextEmbedder` for the semantic tier, same safety contract.

    Mean-pools the last hidden state over the attention mask and L2-normalises,
    which is what sentence-transformers models expect. A model exported without
    its pooling layer and used with CLS-token pooling instead will produce
    embeddings that "work" but retrieve badly -- so this stays explicit rather
    than clever.
    """

    __slots__ = ("_clf", "dim")

    def __init__(
        self,
        spec: OnnxModelSpec,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        dim: int | None = None,
        allow_download: bool = True,
        intra_op_threads: int = 1,
    ) -> None:
        model_dir = ensure_local(spec, cache_dir, allow_download=allow_download)
        self._clf = OnnxTextClassifier(spec, model_dir, intra_op_threads=intra_op_threads)
        probe = self._embed_sync(["dimension probe"])
        self.dim = dim or len(probe[0])

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        feed = self._clf._encode(texts)
        hidden = np.asarray(self._clf._sess.run(None, feed)[0], dtype="float32")
        if hidden.ndim != 3:
            raise ModelIntegrityError(
                f"expected last_hidden_state with shape (batch, tokens, dim), got {hidden.shape}. "
                "Export the base encoder, not a classification head."
            )
        mask = feed["attention_mask"].astype("float32")[..., None]
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return (pooled / norms).tolist()

    async def aembed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await asyncio.to_thread(self._embed_sync, list(texts))
