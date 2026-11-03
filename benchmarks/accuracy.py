"""Detection accuracy on public, labelled prompt-injection / jailbreak datasets.

    python benchmarks/accuracy.py                 # all datasets, markdown table
    python benchmarks/accuracy.py --json out.json # machine-readable too
    python benchmarks/accuracy.py --offline       # cached data only
    python benchmarks/accuracy.py --model protectai-deberta-v3-injection-v2
                                                  # + the pinned ONNX classifier tier

Datasets are fetched once through the Hugging Face datasets-server JSON API
(stdlib only, no `datasets`/`pyarrow`) and cached under
`benchmarks/.cache/`. Their text is treated as data: it is only ever passed
to `check_input`, never executed or interpreted.

What is measured: `GuardrailPipeline.check_input` with the rule tiers only
(no model scorer configured), in the default and strict presets. A sample is
"detected" when the verdict is not ALLOW (block *or* flag) and, separately,
when it is a block. Numbers are reported per dataset with no tuning on them;
the rules were never fitted to these sets.

With ``--model PRESET`` (needs ``.[onnx]``), each preset also runs with that
ONNX classifier as the model tier, escalated on every input
(``escalate_clean``), through ``acheck_input``. A public classifier may have
been trained on some of these datasets; check its model card before reading
its numbers as held-out.

Read the results with the datasets' labelling in mind -- see the notes printed
with each table. Several label benign role-play ("act as an interviewer") as
injection, and much of deepset/prompt-injections is German, which the rule
tier only partly covers. Those misses are what the classifier tier is for.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from charterguard import GuardrailConfig, GuardrailPipeline  # noqa: E402
from charterguard.types import Action, Layer  # noqa: E402

CACHE = ROOT / ".cache"
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100


@dataclass(frozen=True)
class Dataset:
    name: str
    hf_id: str
    splits: tuple[str, ...]
    text_field: str
    is_attack: Callable[[dict[str, object]], bool]
    note: str


DATASETS = (
    Dataset(
        "deepset/prompt-injections",
        "deepset/prompt-injections",
        ("train", "test"),
        "text",
        lambda row: row.get("label") == 1,
        "Mixed English/German; labels some benign role-play requests as injections.",
    ),
    Dataset(
        "jackhhao/jailbreak-classification",
        "jackhhao/jailbreak-classification",
        ("train", "test"),
        "prompt",
        lambda row: row.get("type") == "jailbreak",
        "Long role-play jailbreaks (DAN-style) vs. benign role-play prompts.",
    ),
)


def _fetch_split(hf_id: str, split: str, *, offline: bool) -> list[dict[str, object]]:
    path = CACHE / f"{hf_id.replace('/', '__')}__{split}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        raise SystemExit(f"--offline but {path.name} is not cached")
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": hf_id, "config": "default", "split": split, "offset": offset, "length": PAGE}
        )
        with urllib.request.urlopen(f"{API}?{query}", timeout=30) as resp:  # noqa: S310 -- fixed https URL
            page = json.load(resp)
        batch = [r["row"] for r in page["rows"]]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= page.get("num_rows_total", 0):
            break
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, attack: bool, detected: bool) -> None:
        if attack and detected:
            self.tp += 1
        elif attack:
            self.fn += 1
        elif detected:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0


def with_input_model(cfg: GuardrailConfig) -> GuardrailConfig:
    """``cfg`` with the model tier consulted on every input turn."""
    from dataclasses import replace

    tier = replace(cfg.model_for(Layer.INPUT), enabled=True, escalate_clean=True, clean_sample_rate=1.0)
    return cfg.with_model_tier(Layer.INPUT, tier)


def evaluate(
    samples: Iterable[tuple[str, bool]], pipeline: GuardrailPipeline, *, use_async: bool = False
) -> dict[str, object]:
    flagged, blocked = Score(), Score()
    latencies: list[float] = []
    n = 0
    for text, attack in samples:
        start = time.perf_counter()
        verdict = asyncio.run(pipeline.acheck_input(text)) if use_async else pipeline.check_input(text)
        latencies.append((time.perf_counter() - start) * 1e6)
        flagged.add(attack, verdict.action is not Action.ALLOW)
        blocked.add(attack, verdict.blocked)
        n += 1
    latencies.sort()
    return {
        "samples": n,
        "attacks": flagged.tp + flagged.fn,
        "flag_or_block": _metrics(flagged),
        "block": _metrics(blocked),
        "latency_us_p50": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_us_p99": round(latencies[int(0.99 * (len(latencies) - 1))], 1) if latencies else 0.0,
    }


def _metrics(s: Score) -> dict[str, float]:
    return {
        "precision": round(s.precision, 3),
        "recall": round(s.recall, 3),
        "f1": round(s.f1, 3),
        "fpr": round(s.fpr, 3),
    }


def _local_corpus() -> list[tuple[str, bool]]:
    corpus = ROOT.parent / "tests" / "corpus"
    out: list[tuple[str, bool]] = []
    for name, attack in (("attacks.jsonl", True), ("benign.jsonl", False)):
        for line in (corpus / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append((json.loads(line)["text"], attack))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="also write results as JSON")
    ap.add_argument("--offline", action="store_true", help="use cached datasets only")
    ap.add_argument("--local-only", action="store_true", help="only the bundled corpus")
    ap.add_argument("--model", help="also run with this ONNX preset as the model tier (needs .[onnx])")
    ap.add_argument("--model-cache", type=Path, help="cache directory for the model files")
    args = ap.parse_args(argv)

    presets: dict[str, tuple[GuardrailConfig, object]] = {
        "default": (GuardrailConfig(), None),
        "strict": (GuardrailConfig.strict(), None),
    }
    if args.model:
        from charterguard.integrations.onnx_presets import injection_scorer

        scorer = injection_scorer(preset=args.model, cache_dir=args.model_cache)
        presets["default + model"] = (with_input_model(GuardrailConfig()), scorer)
        presets["strict + model"] = (with_input_model(GuardrailConfig.strict()), scorer)
    sets: list[tuple[str, str, list[tuple[str, bool]]]] = [
        ("bundled corpus", "Hand-written attacks and deliberately tricky benign texts.", _local_corpus())
    ]
    if not args.local_only:
        for ds in DATASETS:
            rows = [r for split in ds.splits for r in _fetch_split(ds.hf_id, split, offline=args.offline)]
            samples = [(str(r[ds.text_field]), ds.is_attack(r)) for r in rows if r.get(ds.text_field)]
            sets.append((ds.name, ds.note, samples))

    results: dict[str, dict[str, object]] = {}
    print("| Dataset | Preset | n (attacks) | Precision | Recall | F1 | FPR | Block recall | Block FPR | p50 µs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, _note, samples in sets:
        results[name] = {}
        for preset, (cfg, scorer) in presets.items():
            pipeline = GuardrailPipeline(cfg, model=scorer)  # type: ignore[arg-type]
            r = evaluate(samples, pipeline, use_async=scorer is not None)
            results[name][preset] = r
            f, b = r["flag_or_block"], r["block"]
            assert isinstance(f, dict) and isinstance(b, dict)
            print(
                f"| {name} | {preset} | {r['samples']} ({r['attacks']}) | {f['precision']:.2f} | "
                f"{f['recall']:.2f} | {f['f1']:.2f} | {f['fpr']:.2f} | {b['recall']:.2f} | "
                f"{b['fpr']:.2f} | {r['latency_us_p50']} |",
                flush=True,
            )
    print()
    for name, note, _ in sets:
        print(f"* **{name}**: {note}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
