# Detection accuracy

Measured with `python benchmarks/accuracy.py`: rule tiers only, no model scorer
configured. No rule was fitted to jailbreak-classification. Some deepset phrasings
were added from its **train** split (see below), so its held-out **test** split is
reported on its own line. "Detected" means the verdict is
flag or block; the block columns use the strict preset (the default preset flags
injection rather than blocking it).

| Dataset | n (attacks) | Precision | Recall | FPR | Block recall (strict) | Block FPR (strict) |
| --- | --- | --- | --- | --- | --- | --- |
| bundled corpus (`tests/corpus`) | 96 (43) | 0.96 | 1.00 | 0.04 | 1.00 | 0.00 |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) (train + test) | 662 (263) | 1.00 | 0.35 | 0.00 | 0.34 | 0.00 |
| deepset/prompt-injections, **test split only** (held out) | 116 (60) | 1.00 | 0.20 | 0.00 | 0.20 | 0.00 |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | 1306 (666) | 0.91 | 0.50 | 0.05 | 0.43 | 0.00 |

What these numbers say:

* **The rule tier is precise, not complete.** On the public sets it almost never
  blocks benign text, but it misses most attacks phrased in ways no rule anticipates.
* **deepset/prompt-injections** is roughly half German, and many of its "injections"
  are persona or opinion prompts ("You are Volkswagen. What do you think of
  Mercedes?") that no pattern-based detector should block. Override shapes mined
  from the train split ("ignore all preceding orders", "vergiss alle Aufträge",
  "oubliez toutes les instructions") took held-out test recall from 0.05 to 0.20
  with no new false positives; what remains is mostly persona and opinion prompts.
* **jailbreak-classification** is long DAN-style role-play. The rules catch the
  well-known personas and "no restrictions" framing, not novel stories.

That gap is what the classifier tier (`ModelScorer`, escalation bands) is for. Run the
benchmark with your own scorer configured to measure the full cascade.
