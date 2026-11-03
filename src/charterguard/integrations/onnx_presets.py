"""Ready-made, pinned ONNX model specs for the local model tier.

    from charterguard import GuardrailPipeline
    from charterguard.integrations.onnx_presets import injection_scorer

    pipeline = GuardrailPipeline(model=injection_scorer())

Each preset is an ordinary `OnnxModelSpec`: a full commit SHA and a SHA-256
for every file, computed by downloading the files and hashing them locally (and
cross-checked against the hub's LFS object id). Nothing here is fetched until a
scorer is built, and every load re-verifies the digests.

A preset says "these are the exact bytes we tested", not "this model is right
for your traffic". Measure it on your own data before enforcing on it -- see
docs/accuracy.md for how the bundled numbers were produced.
"""
from __future__ import annotations

import os
from typing import Any

from ..types import Category
from .onnx_model import FileSpec, OnnxModelScorer, OnnxModelSpec

__all__ = ["PROTECTAI_DEBERTA_V3_INJECTION_V2", "PRESETS", "injection_scorer"]

# protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0), its ONNX export.
# A DeBERTa-v3-base binary classifier: SAFE / INJECTION, single-label (softmax).
# English only; ~740 MB; roughly 30-120 ms per 256-token input on one CPU core.
PROTECTAI_DEBERTA_V3_INJECTION_V2 = OnnxModelSpec(
    repo_id="protectai/deberta-v3-base-prompt-injection-v2",
    revision="90c9989b1a342275dd0d1a95aad283c04e075671",
    onnx_file=FileSpec(
        "onnx/model.onnx", "f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c"
    ),
    tokenizer_file=FileSpec(
        "onnx/tokenizer.json", "752fe5f0d5678ad563e1bd2ecc1ddf7a3ba7e2024d0ac1dba1a72975e26dff2f"
    ),
    # Output order is the model's id2label: 0 = SAFE, 1 = INJECTION.
    label_map={"SAFE": None, "INJECTION": Category.INJECTION},
    max_length=256,
    activation="softmax",
)

PRESETS: dict[str, OnnxModelSpec] = {
    "protectai-deberta-v3-injection-v2": PROTECTAI_DEBERTA_V3_INJECTION_V2,
}


def injection_scorer(
    *,
    preset: str = "protectai-deberta-v3-injection-v2",
    cache_dir: str | os.PathLike[str] | None = None,
    allow_download: bool = True,
    **kwargs: Any,
) -> OnnxModelScorer:
    """An `OnnxModelScorer` for a named preset (downloads once, verifies always).

    Set ``allow_download=False`` in locked-down deployments and pre-seed
    ``cache_dir`` with the pinned files; anything missing or altered raises.
    """
    try:
        spec = PRESETS[preset]
    except KeyError:
        raise ValueError(f"unknown preset {preset!r}; available: {sorted(PRESETS)}") from None
    return OnnxModelScorer(spec, cache_dir=cache_dir, allow_download=allow_download, **kwargs)
