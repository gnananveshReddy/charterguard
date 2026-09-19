"""Counterfactual policy evaluation: what would config B have decided?

The expensive half of a guardrail is *detection* -- normalization, the trie
probe, regex execution, and any model escalation. `decide()` is a pure function
of findings, config and layer, so a candidate policy can be evaluated against
findings that have already been computed.

That asymmetry makes the usual A/B design wasteful. Splitting traffic and running
a second pipeline doubles detection cost, exposes half your users to the untested
config, and still only tells you what happened to *that* half. Scanning once and
re-deciding gives the counterfactual for **100% of traffic at zero risk**.

Measured, rather than assumed -- because the first draft of this docstring
claimed "the cost of a dict lookup" and a perf test caught it:

    decide(0 findings)                8.9 us
    decide(1 finding)                16.6 us
    decide(5 findings)               36.2 us
    full check_input, clean query    70.9 us
    full check_input, injection      42.0 us   (stop_at short-circuits)

So one candidate costs roughly a quarter to a half of a scan on a verdict that
*has* findings -- real, not free. What makes it cheap in aggregate is the short
circuit below: a verdict with no findings decides to ALLOW under every possible
config, so the overwhelmingly common clean case is skipped outright and only
interesting traffic is re-decided. Keep the candidate list to one or two; this is
a rollout tool, not a permanent fixture.

    shadow = ShadowPolicy(baseline=cfg, candidates={"stricter": tighter_cfg})
    verdict = pipeline.check_input(text)      # one scan, unchanged behaviour
    shadow.observe(verdict)                   # ~5 us
    ...
    shadow.report()["stricter"].would_newly_block   # decide with evidence

What this cannot tell you: anything that depends on detection rather than
policy. A new *rule*, a different normalizer, or a model tier that was not
consulted changes the findings themselves, and findings are the input here. For
those, run the real pipeline twice offline over a corpus -- `tools/`, not this.

Deliberately absent: request text. This class records counts and rule ids only.
Storing the prompts that diverged would build exactly the corpus of sensitive
content that the component exists to keep out of logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .config import GuardrailConfig
# Reuse the policy engine's own ordering rather than restating it: there is no
# REDACT action here, and a duplicated rank table would silently disagree the
# first time the enum changes.
from .policy import _ACTION_RANK as _RANK
from .policy import decide
from .types import Action, Verdict

__all__ = ["ShadowDelta", "ShadowPolicy"]


@dataclass(slots=True)
class ShadowDelta:
    """Divergence tally for one candidate config against the baseline."""

    name: str
    observed: int = 0
    agreed: int = 0
    # Directional, because the two directions have opposite risk profiles: newly
    # blocked traffic is a support cost, newly allowed traffic is an incident.
    would_newly_block: int = 0
    would_newly_allow: int = 0
    escalated: int = 0            # any strictness increase short of ALLOW->BLOCK
    relaxed: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    transitions: dict[str, int] = field(default_factory=dict)

    @property
    def agreement_rate(self) -> float:
        return self.agreed / self.observed if self.observed else 1.0

    @property
    def newly_allowed_rate(self) -> float:
        """The number to justify a rollout against. A candidate that newly
        allows anything must be explained finding-by-finding, not averaged."""
        return self.would_newly_allow / self.observed if self.observed else 0.0

    def summary(self) -> str:
        return (
            f"{self.name}: {self.agreement_rate:.1%} agreement over {self.observed} verdicts, "
            f"+{self.would_newly_block} newly blocked, +{self.would_newly_allow} newly allowed"
        )


class ShadowPolicy:
    """Evaluate candidate configs against live findings without changing behaviour.

    Args:
        baseline: the config actually in force. Passing the same object the
            pipeline uses is the point -- if it drifts, the comparison is a lie.
        candidates: named configs to evaluate. Keep this small; each one costs a
            `decide()` call per verdict.
        max_findings: verdicts with more findings than this are counted but not
            re-decided, so a pathological document cannot make the shadow path
            the expensive one.
    """

    __slots__ = ("_baseline", "_candidates", "_deltas", "_max_findings", "skipped", "_all_allow")

    def __init__(
        self,
        baseline: GuardrailConfig,
        candidates: Mapping[str, GuardrailConfig],
        *,
        max_findings: int = 64,
    ) -> None:
        if not candidates:
            raise ValueError("at least one candidate config is required")
        self._baseline = baseline
        self._candidates = dict(candidates)
        self._deltas = {name: ShadowDelta(name) for name in self._candidates}
        self._max_findings = max_findings
        self.skipped = 0
        self._all_allow = {name: Action.ALLOW for name in self._candidates}

    def observe(self, verdict: Verdict) -> Mapping[str, Action]:
        """Record what each candidate would have done. Returns their actions.

        Safe to call on every verdict in production: no I/O, no allocation
        beyond a few small dicts, and it cannot raise into the request path for
        any verdict the pipeline itself produced.
        """
        if not verdict.findings:
            # Provable short circuit, not an approximation: `category_scores(())`
            # is empty, so `decide` never enters its per-category loop and
            # returns ALLOW with no deciding category for *every* config and
            # every layer. A test asserts that across all three bundled configs
            # x all layers. This is what keeps the shadow path off the hot path,
            # since clean traffic dominates.
            for delta in self._deltas.values():
                delta.observed += 1
                delta.agreed += 1
            return self._all_allow

        if len(verdict.findings) > self._max_findings:
            self.skipped += 1
            return {}

        actual = verdict.action
        out: dict[str, Action] = {}
        for name, cfg in self._candidates.items():
            shadow = decide(verdict.findings, cfg, verdict.layer)
            out[name] = shadow.action
            delta = self._deltas[name]
            delta.observed += 1
            if shadow.action is actual:
                delta.agreed += 1
                continue

            key = f"{actual.value}->{shadow.action.value}"
            delta.transitions[key] = delta.transitions.get(key, 0) + 1
            if shadow.deciding_category is not None:
                cat = shadow.deciding_category.value
                delta.by_category[cat] = delta.by_category.get(cat, 0) + 1

            if shadow.action is Action.BLOCK and actual is not Action.BLOCK:
                delta.would_newly_block += 1
            elif actual is Action.BLOCK and shadow.action is not Action.BLOCK:
                delta.would_newly_allow += 1
            elif _RANK[shadow.action] > _RANK[actual]:
                delta.escalated += 1
            else:
                delta.relaxed += 1
        return out

    def report(self) -> Mapping[str, ShadowDelta]:
        return dict(self._deltas)

    def reset(self) -> None:
        self._deltas = {name: ShadowDelta(name) for name in self._candidates}
        self.skipped = 0

    def emit(self, metrics, *, prefix: str = "guardrail.shadow") -> None:
        """Push the tallies to any `MetricsSink` as gauges.

        Gauges rather than counters: these are cumulative totals held in this
        object, so re-sending them as counters would double-count on every call.
        """
        for name, delta in self._deltas.items():
            tags = {"candidate": name}
            metrics.gauge(f"{prefix}.observed", delta.observed, tags)
            metrics.gauge(f"{prefix}.agreement_rate", delta.agreement_rate, tags)
            metrics.gauge(f"{prefix}.newly_blocked", delta.would_newly_block, tags)
            metrics.gauge(f"{prefix}.newly_allowed", delta.would_newly_allow, tags)
