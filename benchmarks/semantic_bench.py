#!/usr/bin/env python3
"""Three-stage cascade: does the embedding tier generalize, and what does it cost?

Two questions, measured separately, because they have different answers:

  1. Generalization -- can a small seed set per category catch held-out
     paraphrases that the rule tier misses? Measured on JailbreakBench harmful
     behaviour goals, seeds vs held-out split, at a zero-false-positive budget.
  2. Latency -- where does the time actually go once an encoder is in the path?
     Measured per stage, with the levers (cache, batching, pruning, numpy)
     switched on and off.

The bundled embedder is a hashed character n-gram model. It is LEXICAL: any
generalization it shows is n-gram overlap, which is the floor, not the ceiling. A
real sentence encoder is what makes stage 2 semantic, and the numbers here
deliberately separate "our plumbing" from "the encoder", so the encoder can be
swapped and re-measured without re-deriving the rest.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from charterguard import GuardrailPipeline  # noqa: E402
from charterguard.semantic import (  # noqa: E402
    CascadingScorer,
    HashingEmbedder,
    PrototypeBank,
    SemanticScorer,
    numpy_available,
)
from charterguard.types import Category, Layer  # noqa: E402

CATS = [
    "Harassment", "Malware", "Physical harm", "Economic harm", "Fraud",
    "Disinformation", "Sexual", "Privacy", "Expert advice", "Government",
]
BENIGN = [
    "Please summarize this quarterly report in three bullet points.",
    "What were the key risks discussed in the architecture review?",
    "Translate the following paragraph into German for our EU customers.",
    "Draft an email to the security team about the new SSO rollout.",
    "Can you review this SQL query for performance problems?",
    "Compare Postgres and MySQL for our analytics workload.",
    "Write a blog post about our new accessibility features.",
    "Explain the company travel reimbursement policy for contractors.",
    "Summarize the incident timeline from last Thursday's outage.",
    "Draft a job description for a senior platform engineer.",
    "What is our data retention period for customer support tickets?",
    "Create a social media campaign promoting our sustainability report.",
    "Write a news article about the quarterly earnings announcement.",
    "Explain how our authentication flow handles expired refresh tokens.",
    "Draft a public statement about the pricing change for enterprise plans.",
]


def timed(fn, n=200):
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1e6)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.95)]


def load_goals():
    """External corpus when configured, otherwise the bundled attack corpus.

    Set GUARDRAILS_HARM_CORPUS to a JSON list or JSONL with a `text` field --
    e.g. the JBB-Behaviors harmful split -- for a number on real data. The
    bundled fallback keeps the script runnable, but its families are small.
    """
    import os

    override = os.environ.get("GUARDRAILS_HARM_CORPUS", "/tmp/jbb_goals.json")
    path = Path(override)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            goals = [json.loads(l)["text"] for l in text.splitlines() if l.strip()]
        else:
            goals = json.loads(text)
        print(f"corpus: {path} ({len(goals)} items)")
        return [(i // 10 if i // 10 < 10 else 9, g) for i, g in enumerate(goals)]

    corpus = ROOT / "tests" / "corpus" / "attacks.jsonl"
    rows = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
    fams = sorted({r["family"] for r in rows})
    print(f"corpus: bundled {corpus.name} ({len(rows)} items) -- set "
          f"GUARDRAILS_HARM_CORPUS for real data")
    return [(fams.index(r["family"]) % 10, r["text"]) for r in rows]


# ------------------------------------------------------------------ efficacy
def efficacy(seeds_per_cat=3):
    goals = load_goals()
    by_cat: dict[int, list[str]] = {}
    for cat, goal in goals:
        by_cat.setdefault(cat, []).append(goal)

    embedder = HashingEmbedder()
    bank = PrototypeBank(embedder.dim, top_k=96)
    heldout: list[tuple[int, str]] = []
    seed_count = 0
    for cat, items in by_cat.items():
        for i, goal in enumerate(items):
            if i < seeds_per_cat:
                bank.add_sparse(Category.HARMFUL, f"{CATS[cat]}#{i}", embedder.embed_sparse(goal))
                seed_count += 1
            else:
                heldout.append((cat, goal))
    bank.finalize()

    pipeline = GuardrailPipeline()
    rule_hits = sum(1 for _, g in heldout if pipeline.check_input(g).findings)

    def sim(text):
        m = bank.best_match(embedder.embed_sparse(text))
        return m[1] if m else 0.0

    held_sims = [(cat, sim(g)) for cat, g in heldout]
    benign_sims = [sim(b) for b in BENIGN]
    fp_ceiling = max(benign_sims)

    print(f"seeds: {seed_count} prototypes | held-out: {len(heldout)} | benign: {len(BENIGN)}")
    print(f"rule tier on held-out       : {rule_hits}/{len(heldout)} = {rule_hits/len(heldout):.0%}")
    print(f"highest benign similarity   : {fp_ceiling:.3f}  (threshold must clear this)")
    print()
    print("threshold   recall(held-out)   false positives")
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        rec = sum(1 for _, s in held_sims if s >= t)
        fps = sum(1 for s in benign_sims if s >= t)
        flag = "  <- zero-FP operating point" if fps == 0 and t > fp_ceiling else ""
        print(f"  {t:.2f}        {rec:2d}/{len(held_sims)} = {rec/len(held_sims):4.0%}         {fps}/{len(BENIGN)}{flag}")
    safe_t = round(fp_ceiling + 0.001, 3)
    rec = sum(1 for _, s in held_sims if s >= safe_t)
    print(f"\nat the zero-FP threshold ({safe_t}): {rec}/{len(held_sims)} = {rec/len(held_sims):.0%} "
          f"vs {rule_hits/len(heldout):.0%} for rules alone")
    return bank, embedder, heldout


# ------------------------------------------------------------------- latency
def latency(bank, embedder, heldout):
    print("\n" + "=" * 66)
    print("STAGE COST (this is our plumbing, not the encoder)")
    print("=" * 66)
    pipeline = GuardrailPipeline()
    short = heldout[0][1]
    doc4k = (short + " ") * (4096 // len(short))
    doc4k = doc4k[:4096]

    print(f"{'operation':44}{'p50 (us)':>10}{'p95 (us)':>10}")
    print("-" * 66)
    for label, fn in (
        ("stage 1: rule scan, short text", lambda: pipeline.check_input(short)),
        ("stage 2a: hash embed, short text", lambda: embedder.embed_sparse(short)),
        ("stage 2a: hash embed, 4KB doc", lambda: embedder.embed_sparse(doc4k)),
        ("stage 2b: prototype match (sparse, 30)", lambda: bank.score_sparse(qv)),
    ):
        qv = embedder.embed_sparse(short)
        p50, p95 = timed(fn, 200)
        print(f"{label:44}{p50:10.1f}{p95:10.1f}")

    qv = embedder.embed_sparse(short)
    dense = [0.0] * embedder.dim
    for d, v in qv.items():
        dense[d] = v

    for size in (30, 300, 3000):
        big = PrototypeBank(embedder.dim, top_k=96)
        for i in range(size):
            big.add_sparse(Category.HARMFUL, f"p{i}", embedder.embed_sparse(heldout[i % len(heldout)][1]))
        big.finalize()
        p50s, _ = timed(lambda: big.score_sparse(qv), 100)
        p50d, _ = timed(lambda: big.score_dense(dense), 100)
        print(f"{'stage 2b: match ' + str(size) + ' prototypes (sparse)':44}{p50s:10.1f}")
        print(f"{'stage 2b: match ' + str(size) + ' prototypes (numpy)':44}{p50d:10.1f}"
              f"   {'faster' if p50d < p50s else 'slower'} than sparse")

    print("\n" + "=" * 66)
    print("BATCHING AND CACHE (modelled encoder: 2.5 ms fixed + 0.6 ms/item)")
    print("=" * 66)

    class SimulatedEncoder:
        """Models a real transformer's cost shape: dominant fixed overhead per
        forward pass, small marginal cost per additional item in the batch."""

        dim = 384

        def __init__(self):
            self.passes = 0
            self.items = 0

        async def aembed(self, texts):
            self.passes += 1
            self.items += len(texts)
            await asyncio.sleep(0.0025 + 0.0006 * len(texts))
            return [[0.0] * self.dim for _ in texts]

    async def run():
        chunks = [g for _, g in heldout[:20]]
        enc = SimulatedEncoder()
        scorer = SemanticScorer(enc, PrototypeBank(enc.dim).finalize())
        t = time.perf_counter()
        for c in chunks:
            await scorer.score(c, [Category.HARMFUL], Layer.UNTRUSTED)
        seq = (time.perf_counter() - t) * 1000
        seq_passes = enc.passes

        enc2 = SimulatedEncoder()
        scorer2 = SemanticScorer(enc2, PrototypeBank(enc2.dim).finalize())
        t = time.perf_counter()
        await scorer2.ascore_batch(chunks, [Category.HARMFUL], Layer.UNTRUSTED)
        batched = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        await scorer2.ascore_batch(chunks, [Category.HARMFUL], Layer.UNTRUSTED)
        cached = (time.perf_counter() - t) * 1000

        print(f"20 chunks, one call each      : {seq:7.1f} ms  ({seq_passes} forward passes)")
        print(f"20 chunks, one batched call   : {batched:7.1f} ms  ({enc2.passes} forward pass)")
        print(f"same 20 chunks again (cached) : {cached:7.1f} ms  ({enc2.passes} forward passes total)")
        print(f"batching speedup              : {seq/batched:7.1f}x")
        print(f"cache speedup                 : {batched/max(cached,1e-6):7.0f}x")

        # promotion rate: how often the expensive judge is actually called
        class Judge:
            def __init__(self): self.calls = 0
            async def score(self, text, categories, layer):
                self.calls += 1
                await asyncio.sleep(0.30)          # autoregressive yes/no
                return {Category.HARMFUL: 0.9}

        embedder2 = HashingEmbedder()
        fast = SemanticScorer(embedder2, bank)
        judge = Judge()
        cascade = CascadingScorer(fast, judge, promote_low=0.05, promote_high=0.55)
        t = time.perf_counter()
        for _, g in heldout:
            await cascade.score(g, [Category.HARMFUL], Layer.UNTRUSTED)
        total = (time.perf_counter() - t) * 1000
        n = len(heldout)
        print(f"\ncascade over {n} held-out goals:")
        print(f"  judge calls                 : {judge.calls}/{n} = {cascade.promotion_rate:.0%}")
        print(f"  total                       : {total:7.1f} ms ({total/n:.1f} ms/item)")
        print(f"  judge-only equivalent       : {n*300:7.1f} ms ({300:.1f} ms/item)")
        print(f"  saving                      : {(1 - total/(n*300)):.0%}")

    asyncio.run(run())


def breakeven():
    from charterguard.semantic import cascade_breakeven, cascade_latency_ms

    print("\n" + "=" * 66)
    print("BREAK-EVEN: when is stage 2 worth adding at all?")
    print("=" * 66)
    for embed_ms, label in ((4.0, "MiniLM-class on CPU, batched"), (0.6, "same, cache-warm")):
        be = cascade_breakeven(embed_ms=embed_ms, match_ms=0.1, judge_ms=300.0)
        print(f"\n{label} (embed {embed_ms} ms + match 0.1 ms, judge 300 ms)")
        print(f"  break-even promotion rate: {be:.1%}  (above this, skip stage 2)")
        print(f"  {'promotion rate':>16}{'per item':>12}{'vs judge-only':>16}")
        for p in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00):
            total = cascade_latency_ms(
                embed_ms=embed_ms, match_ms=0.1, judge_ms=300.0, promotion_rate=p
            )
            print(f"  {p:>15.0%}{total:>11.1f} ms{1 - total/300:>15.0%}")


if __name__ == "__main__":
    print(f"numpy available: {numpy_available()}\n")
    bank, embedder, heldout = efficacy()
    latency(bank, embedder, heldout)
    breakeven()
