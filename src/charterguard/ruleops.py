"""Rule health: which rules earn their place, and which are dead weight?

`SECURITY.md` requires that the rule pack have an owner and a review cadence,
because detection decays -- attack phrasing moves, normalization changes, and a
rule that was load-bearing last year can stop matching anything without any test
turning red. `tools/rule_coverage.py` answers that question *offline*, against a
corpus. This module answers it against *production traffic*, which is the only
place the real distribution lives.

Two things it reports, and one it deliberately does not:

  * **Dead rules.** Rules that have never fired. Note the trap in the obvious
    implementation: a counter keyed by rule id only ever gains keys for rules
    that *did* fire, so `{rid for rid, n in fires.items() if n == 0}` is always
    empty. The inventory has to be seeded from the rule set, which is why the
    constructor takes the rules rather than discovering them.
  * **Redundant pairs.** Rules that never fire except together with another
    rule, on the same span. That is a maintenance cost: two patterns to review,
    one signal.

It does **not** report "shadowing" in the sense of one rule preventing another
from matching, because this engine makes that impossible by construction -- every
variant is matched independently, and a broad rule that later fails validation
cannot consume a region a narrower rule should own (see `scanner`). Looking for
a failure mode the architecture excludes would produce noise, not insight.

Cost is one dict increment per finding, so this is safe to leave on. Findings
are already capped by `ScanBudget.max_findings`.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from .patterns import Rule
from .types import Finding, Severity

__all__ = ["RuleHealth", "RuleStat"]


@dataclass(frozen=True, slots=True)
class RuleStat:
    rule_id: str
    severity: str
    fires: int
    solo_fires: int          # fired with no other rule on the same verdict
    validator_rejects: int   # candidate matched but a validator refused it

    @property
    def precision_signal(self) -> float:
        """Share of candidate matches that survived validation.

        Not precision -- nobody labelled this traffic. A low value means the
        pattern is broad and the validator is carrying it, which is worth
        knowing before someone "simplifies" the validator away.
        """
        total = self.fires + self.validator_rejects
        return self.fires / total if total else 1.0


class RuleHealth:
    """Live rule-firing tracker.

    Args:
        rules: the rule set in force. Required, and the whole point: the
            inventory is what makes "never fired" expressible.
    """

    __slots__ = ("_inventory", "_fires", "_solo", "_rejects", "_pairs", "verdicts")

    def __init__(self, rules: Sequence[Rule]) -> None:
        self._inventory: dict[str, Severity] = {r.id: r.severity for r in rules}
        self._fires: dict[str, int] = dict.fromkeys(self._inventory, 0)
        self._solo: dict[str, int] = dict.fromkeys(self._inventory, 0)
        self._rejects: dict[str, int] = dict.fromkeys(self._inventory, 0)
        self._pairs: dict[tuple[str, str], int] = {}
        self.verdicts = 0

    # ------------------------------------------------------------- recording
    def record(self, findings: Iterable[Finding]) -> None:
        """Tally one verdict's findings. Call it with `verdict.findings`."""
        self.verdicts += 1
        ids = [f.rule_id for f in findings if f.rule_id]
        if not ids:
            return
        distinct = sorted(set(ids))
        for rid in distinct:
            if rid in self._fires:
                self._fires[rid] += 1
            else:
                # A rule id outside the inventory means the pipeline is running a
                # different rule set than the one this tracker was built with.
                # Counting it anyway is more useful than dropping it silently.
                self._inventory.setdefault(rid, Severity.LOW)
                self._fires[rid] = 1
                self._solo.setdefault(rid, 0)
                self._rejects.setdefault(rid, 0)
        if len(distinct) == 1:
            self._solo[distinct[0]] += 1
        else:
            for i, left in enumerate(distinct):
                for right in distinct[i + 1 :]:
                    key = (left, right)
                    self._pairs[key] = self._pairs.get(key, 0) + 1

    def record_reject(self, rule_id: str) -> None:
        """Record a candidate that matched but failed its validator (Luhn,
        Verhoeff, entropy). Wire this from the `keep=` hook to see which patterns
        depend on validation to stay usable."""
        if rule_id in self._rejects:
            self._rejects[rule_id] += 1

    # -------------------------------------------------------------- reporting
    def dead_rules(self, *, min_verdicts: int = 10_000) -> tuple[str, ...]:
        """Rules that have never fired -- empty until the sample is large enough.

        The guard matters: on 200 verdicts, "never fired" is the expected state
        for most of a 99-rule pack, and reporting it would train people to ignore
        this output.
        """
        if self.verdicts < min_verdicts:
            return ()
        return tuple(sorted(rid for rid, n in self._fires.items() if n == 0))

    def redundant_pairs(self, *, min_fires: int = 50) -> tuple[tuple[str, str, float], ...]:
        """Pairs where one rule almost never fires without the other.

        Returns (subsumed_rule, companion_rule, co_fire_rate). A rate near 1.0
        means the first rule contributed no independent signal in this sample.
        """
        out: list[tuple[str, str, float]] = []
        for (left, right), together in self._pairs.items():
            for rid, other in ((left, right), (right, left)):
                total = self._fires.get(rid, 0)
                if total >= min_fires and together / total >= 0.98:
                    out.append((rid, other, together / total))
        return tuple(sorted(out, key=lambda t: -t[2]))

    def stats(self) -> tuple[RuleStat, ...]:
        return tuple(
            RuleStat(
                rule_id=rid,
                severity=self._inventory[rid].name,
                fires=self._fires[rid],
                solo_fires=self._solo.get(rid, 0),
                validator_rejects=self._rejects.get(rid, 0),
            )
            for rid in sorted(self._fires)
        )

    def top(self, n: int = 10) -> tuple[RuleStat, ...]:
        return tuple(sorted(self.stats(), key=lambda s: -s.fires)[:n])

    def coverage(self) -> float:
        """Share of the rule pack that has fired at least once."""
        if not self._fires:
            return 0.0
        return sum(1 for n in self._fires.values() if n) / len(self._fires)

    def emit(self, metrics, *, prefix: str = "guardrail.rules") -> None:
        """Export to a `MetricsSink`.

        Per-rule series are gauges keyed by rule id -- bounded cardinality,
        because the rule pack is a fixed, small inventory. This is the one place
        a rule id as a label is safe.
        """
        metrics.gauge(f"{prefix}.coverage", self.coverage())
        metrics.gauge(f"{prefix}.verdicts", self.verdicts)
        for stat in self.stats():
            metrics.gauge(f"{prefix}.fires", stat.fires, {"rule": stat.rule_id})

    def report(self, *, min_verdicts: int = 10_000) -> str:
        lines = [
            f"rule health over {self.verdicts} verdicts -- "
            f"{self.coverage():.1%} of {len(self._fires)} rules fired at least once",
        ]
        for stat in self.top(5):
            lines.append(
                f"  {stat.rule_id:<28} {stat.fires:>7} fires  "
                f"({stat.solo_fires} solo, {stat.validator_rejects} rejected)"
            )
        dead = self.dead_rules(min_verdicts=min_verdicts)
        if dead:
            lines.append(f"  dead ({len(dead)}): {', '.join(dead[:8])}")
        elif self.verdicts < min_verdicts:
            lines.append(f"  dead rules: not reported below {min_verdicts} verdicts")
        for rid, other, rate in self.redundant_pairs()[:5]:
            lines.append(f"  redundant: {rid} fires with {other} {rate:.0%} of the time")
        return "\n".join(lines)
