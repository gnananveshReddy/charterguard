"""GuardrailPipeline -- the façade the harness actually calls.

Five entry points, one per trust boundary:

    check_input          user turn, before the model sees it
    check_scope          charter adherence, before the loop spends a token
    check_tool_call      proposed tool name + arguments, before the gateway runs it
    inspect_untrusted    retrieval chunks / web content / memory, before context
    inspect_tool_output  tool results, before they become an observation
    check_output         model answer, before it leaves the process

Everything returns a `Verdict`. Nothing raises unless you use `*_or_raise`.
Every verdict is audit-ready via `.to_audit()`.

Performance notes:
  * one normalization pass per call, shared by all detectors;
  * content-hash cache shared across layers, so a chunk scanned during
    retrieval is free when it reappears in tool output;
  * short-circuit on CRITICAL;
  * the async variants are the only ones that can touch a model.

Scanning is CPU-bound and synchronous. Calling `check_input` directly from an
async handler blocks the event loop for the duration of the scan, which at 64 KB
is milliseconds of stall for every other coroutine on that worker. The `a*`
variants offload to a worker thread (`asyncio.to_thread`) once the payload
exceeds `ScanBudget.offload_threshold_chars`; the regex work releases the GIL
often enough for that to be a real win, and small payloads skip the hop because
the thread handoff would cost more than the scan.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from .config import GuardrailConfig
from .detectors import (
    Detector,
    EncodedPayloadDetector,
    ModelDetector,
    ModelScorer,
    RuleDetector,
    StructuralDetector,
)
from .envelope import EnvelopeFactory, SystemPrefixGuard
from .exceptions import GuardrailBlocked
from .grounding import GroundingChecker
from .ledger import SessionLedger
from .observability import MetricsSink, NullMetrics, emit as emit_metrics
from .normalize import normalize
from .policy import decide, dedupe
from .redact import RedactionResult, redact
from .refusals import refusal_for
from .scanner import ResultCache, content_key
from .scope import Charter, IntentClassifier, ScopeGate
from .types import Action, Category, Finding, Layer, Severity, Timer, Verdict

_DEFAULT_SESSION = "default"

# Categories redaction can neutralise in place. Everything else -- injection,
# exfiltration, destructive commands -- must quarantine instead.
_SENSITIVE = (Category.SECRET, Category.PII)


@dataclass(frozen=True, slots=True)
class UntrustedResult:
    """Result of admitting untrusted content into the context window."""

    verdict: Verdict
    envelope: str
    content_id: str
    quarantined: bool

    @property
    def safe_to_use(self) -> bool:
        return not self.quarantined

    def context_block(self) -> str:
        """What to actually append. Quarantined content is replaced by a note,
        never by the original text."""
        return self.envelope


class GuardrailPipeline:
    __slots__ = (
        "config",
        "scope_gate",
        "envelopes",
        "ledger",
        "grounding",
        "prefix_guard",
        "_detectors",
        "_model",
        "_cache",
        "_metrics",
        "_audit",
    )

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        *,
        charter: Charter | None = None,
        model: ModelScorer | None = None,
        intent_classifier: IntentClassifier | None = None,
        detectors: Sequence[Detector] | None = None,
        ledger: SessionLedger | None = None,
        envelopes: EnvelopeFactory | None = None,
        grounding: GroundingChecker | None = None,
        system_prefix: str | None = None,
        metrics: MetricsSink | None = None,
        audit_sink: object | None = None,
    ) -> None:
        self.config = config or GuardrailConfig()
        self.scope_gate = (
            ScopeGate(charter, classifier=intent_classifier) if charter is not None else None
        )
        self.envelopes = envelopes or EnvelopeFactory()
        self.ledger = ledger or SessionLedger()
        self.grounding = grounding or GroundingChecker()
        self.prefix_guard = SystemPrefixGuard(system_prefix) if system_prefix else None
        self._detectors: tuple[Detector, ...] = tuple(detectors) if detectors is not None else (
            RuleDetector(),
            StructuralDetector(),
            EncodedPayloadDetector(),
        )
        self._model = ModelDetector(model) if model is not None else None
        self._cache = ResultCache(self.config.budget.cache_size)
        self._metrics = metrics or NullMetrics()
        # Duck-typed on purpose: integrations.audit.AuditSink, or anything with
        # .write(verdict). None means decisions are not persisted, which is a
        # deliberate choice the caller makes rather than a silent default.
        self._audit = audit_sink

    # ------------------------------------------------------------------ core
    def _scan(self, text: str, layer: Layer) -> tuple[tuple[Finding, ...], bool]:
        cfg = self.config
        if not text:
            return (), False
        key = f"{layer.value}|{cfg.policy_version}|{content_key(text)}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        nt = normalize(text, max_chars=cfg.budget.max_chars, tail_chars=cfg.budget.tail_chars)
        findings: list[Finding] = []
        for detector in self._detectors:
            findings.extend(detector.scan(nt, layer, cfg))
            if cfg.short_circuit_on_critical and any(
                f.severity is Severity.CRITICAL for f in findings
            ):
                break
        result = (dedupe(findings), nt.truncated)
        self._cache.put(key, result)
        return result

    def _verdict(
        self,
        layer: Layer,
        findings: Sequence[Finding],
        timer: Timer,
        *,
        truncated: bool = False,
        sanitized: str | None = None,
        quarantined: bool = False,
        meta: dict[str, object] | None = None,
    ) -> Verdict:
        found = dedupe(findings)
        d = decide(found, self.config, layer)
        verdict = Verdict(
            layer=layer,
            action=d.action,
            score=d.score,
            findings=found,
            policy_id=d.policy_id,
            latency_us=timer.us(),
            truncated=truncated,
            sanitized=sanitized,
            quarantined=quarantined,
            meta=meta or {},
        )
        # Single choke point: every entry point funnels through here, so metrics
        # and audit cannot drift out of sync with enforcement.
        emit_metrics(self._metrics, verdict, escalated=bool((meta or {}).get("model_escalated")))
        if self._audit is not None:
            self._audit.write(verdict)
        return verdict

    # ------------------------------------------------------------------ setup
    def system_rules(self) -> str:
        """Append to the system prompt once, ahead of any untrusted block."""
        return self.envelopes.system_rules()

    def verify_prefix(self, prefix: str) -> Verdict:
        timer = Timer()
        if self.prefix_guard is None:
            return Verdict(layer=Layer.INPUT, action=Action.ALLOW, latency_us=timer.us())
        finding = self.prefix_guard.verify(prefix)
        return self._verdict(Layer.INPUT, (finding,) if finding else (), timer)

    # ------------------------------------------------------------------ input
    def check_input(self, text: str, *, session_id: str = _DEFAULT_SESSION) -> Verdict:
        timer = Timer()
        findings, truncated = self._scan(text, Layer.INPUT)
        return self._verdict(
            Layer.INPUT, findings, timer, truncated=truncated, meta={"session_id": session_id}
        )

    async def acheck_input(self, text: str, *, session_id: str = _DEFAULT_SESSION) -> Verdict:
        verdict = await self._offload(self.check_input, text, session_id=session_id)
        return await self._maybe_escalate(verdict, text, Layer.INPUT)

    # ------------------------------------------------------------ async infra
    async def _offload(self, fn, text: str, **kwargs) -> Verdict:
        """Run a sync scan without stalling the event loop on large payloads."""
        if len(text) < self.config.budget.offload_threshold_chars:
            return fn(text, **kwargs)
        return await asyncio.to_thread(fn, text, **kwargs)

    async def _maybe_escalate(self, verdict: Verdict, text: str, layer: Layer) -> Verdict:
        """Consult the model tier when this layer's policy says to.

        Passes the deterministic score so the model blends against it, and the
        text so clean-band sampling is deterministic per document.
        """
        if self._model is None or not self._model.should_escalate(
            verdict.score, self.config, layer, text=text
        ):
            return verdict
        timer = Timer()
        nt = normalize(text, max_chars=self.config.budget.max_chars)
        extra = await self._model.ascan(
            nt, layer, self.config, deterministic_score=verdict.score
        )
        return self._verdict(
            layer,
            tuple(verdict.findings) + tuple(extra),
            timer,
            truncated=verdict.truncated,
            sanitized=verdict.sanitized,
            quarantined=verdict.quarantined,
            meta={**dict(verdict.meta), "model_escalated": True},
        )

    async def ainspect_untrusted(self, body: str, **kwargs) -> UntrustedResult:
        """Async retrieval inspection: offloaded, and model-escalated per policy.

        This is the layer that most needs the model tier -- a novel injection in
        a retrieved document trips no deterministic rule -- and also the layer
        where escalation cost multiplies by retrieval fan-out, which is why the
        default clean-band sample rate here is a few percent rather than 100%.
        """
        return await self._ainspect(body, Layer.UNTRUSTED, kwargs)

    async def ainspect_tool_output(self, payload: object, **kwargs) -> UntrustedResult:
        text = payload if isinstance(payload, str) else repr(payload)
        result = await self._offload(self.inspect_tool_output, text, **kwargs)
        extra = await self._escalation_findings(
            result.verdict.score, text, Layer.TOOL_OUTPUT
        )
        if not extra:
            return result
        return self.inspect_untrusted(
            text,
            kind="tool_result",
            source=kwargs.get("tool_name", "unknown"),
            session_id=kwargs.get("session_id", _DEFAULT_SESSION),
            rows=kwargs.get("rows", 0),
            trust="low",
            redact_sensitive=kwargs.get("redact_secrets", True),
            layer=Layer.TOOL_OUTPUT,
            extra_findings=extra,
        )

    async def _ainspect(self, body: str, layer: Layer, kwargs: dict) -> UntrustedResult:
        result = await self._offload(self.inspect_untrusted, body, **kwargs)
        extra = await self._escalation_findings(result.verdict.score, body, layer)
        if not extra:
            return result
        # Re-run the admission path with the model findings folded in: the
        # ledger is idempotent for this call because the second pass records the
        # same content_id, and _scan is served from cache.
        return self.inspect_untrusted(body, **{**kwargs, "extra_findings": extra})

    async def _escalation_findings(
        self, deterministic_score: float, text: str, layer: Layer
    ) -> tuple[Finding, ...]:
        if self._model is None or not self._model.should_escalate(
            deterministic_score, self.config, layer, text=text
        ):
            return ()
        nt = normalize(text, max_chars=self.config.budget.max_chars)
        return tuple(
            await self._model.ascan(
                nt, layer, self.config, deterministic_score=deterministic_score
            )
        )

    async def acheck_output(self, text: str, **kwargs) -> Verdict:
        verdict = await self._offload(self.check_output, text, **kwargs)
        return await self._maybe_escalate(verdict, text, Layer.OUTPUT)

    # ------------------------------------------------------------------ scope
    def check_scope(self, text: str) -> Verdict:
        if self.scope_gate is None:
            return Verdict(layer=Layer.SCOPE, action=Action.ALLOW)
        return self.scope_gate.check(text, self.config)

    async def acheck_scope(self, text: str) -> Verdict:
        if self.scope_gate is None:
            return Verdict(layer=Layer.SCOPE, action=Action.ALLOW)
        return await self.scope_gate.acheck(text, self.config)

    def check_tool_call(
        self, tool_name: str, arguments: object = None, *, session_id: str = _DEFAULT_SESSION
    ) -> Verdict:
        """Charter check on the tool, injection/command check on the arguments.

        Arguments are model-authored text that frequently originates in
        retrieved content, so they are scanned like any other untrusted string.
        """
        timer = Timer()
        if self.scope_gate is not None:
            scope_verdict = self.scope_gate.check_tool(tool_name, self.config)
            if scope_verdict.blocked:
                return scope_verdict
        findings: list[Finding] = []
        if arguments is not None:
            rendered = arguments if isinstance(arguments, str) else repr(arguments)
            arg_findings, _ = self._scan(rendered, Layer.TOOL_OUTPUT)
            findings.extend(
                f for f in arg_findings if f.category in (Category.COMMAND, Category.EXFIL)
            )
        self.ledger.record(session_id, tool_calls=1)
        volume = self.ledger.check(session_id, self.config.volume)
        if volume is not None:
            findings.append(volume)
        return self._verdict(Layer.TOOL_OUTPUT, findings, timer, meta={"tool": tool_name})

    # -------------------------------------------------------------- untrusted
    def inspect_untrusted(
        self,
        body: str,
        *,
        kind: str = "retrieval",
        source: str = "unknown",
        content_id: str | None = None,
        session_id: str = _DEFAULT_SESSION,
        rows: int = 0,
        trust: str = "none",
        redact_sensitive: bool = False,
        layer: Layer = Layer.UNTRUSTED,
        extra_findings: Sequence[Finding] = (),
    ) -> UntrustedResult:
        """Scan, redact, wrap, and decide whether content may enter the context.

        Order is deliberate: redaction runs BEFORE the enforcement decision, so a
        useful tool result carrying one credential is delivered with the
        credential removed rather than discarded wholesale. Injection and
        exfiltration findings are never redactable and still quarantine.
        """
        timer = Timer()
        scanned, truncated = self._scan(body, layer)
        # `extra_findings` carries model-tier hits from the async path, so the
        # redact -> wrap -> quarantine sequence stays in exactly one place
        # instead of being re-implemented for escalated calls.
        findings = list(scanned) + list(extra_findings)

        redactions: list[dict[str, object]] = []
        context_body = body
        if redact_sensitive:
            sensitive = [f for f in findings if f.category in _SENSITIVE]
            if sensitive:
                red = redact(body, sensitive, self.config.redaction)
                context_body = red.text
                redactions = red.to_audit()

        envelope = self.envelopes.wrap(
            context_body, kind=kind, source=source, content_id=content_id, trust=trust
        )
        audit_findings = tuple(findings) + envelope.findings

        pii_count = sum(1 for f in findings if f.category is Category.PII)
        self.ledger.record(
            session_id,
            rows=rows,
            bytes_=len(body),
            pii_values=pii_count,
            external_fetches=1 if kind in ("web", "external") else 0,
        )
        volume = self.ledger.check(session_id, self.config.volume)
        if volume is not None:
            audit_findings = audit_findings + (volume,)

        # Anything already neutralised by redaction must not also drive a block.
        decision_findings = (
            tuple(f for f in audit_findings if f.category not in _SENSITIVE)
            if redactions
            else audit_findings
        )
        d = decide(dedupe(decision_findings), self.config, layer)
        quarantined = d.action is Action.BLOCK

        text = (
            envelope.text
            if not quarantined
            else self.envelopes.wrap(
                f"[quarantined: content from {source} was withheld by policy {d.policy_id}]",
                kind=kind,
                source=source,
                content_id=envelope.content_id,
                trust="quarantined",
            ).text
        )
        verdict = Verdict(
            layer=layer,
            action=d.action,
            score=d.score,
            findings=dedupe(audit_findings),
            policy_id=d.policy_id,
            latency_us=timer.us(),
            truncated=truncated,
            sanitized=None if quarantined else envelope.text,
            quarantined=quarantined,
            meta={
                "kind": kind,
                "source": source,
                "content_id": envelope.content_id,
                "escaped_delimiters": envelope.escaped_delimiters,
                "bytes": len(body),
                "redactions": redactions,
                "redaction_count": len(redactions),
                **({"model_escalated": True} if extra_findings else {}),
            },
        )
        return UntrustedResult(verdict, text, envelope.content_id, quarantined)

    def inspect_tool_output(
        self,
        payload: object,
        *,
        tool_name: str,
        session_id: str = _DEFAULT_SESSION,
        rows: int = 0,
        redact_secrets: bool = True,
    ) -> UntrustedResult:
        """Tool results are untrusted input. Secrets are redacted here so a
        misbehaving or compromised upstream cannot push a live credential into
        the context window, where it would persist for the rest of the run."""
        text = payload if isinstance(payload, str) else repr(payload)
        return self.inspect_untrusted(
            text,
            kind="tool_result",
            source=tool_name,
            session_id=session_id,
            rows=rows,
            trust="low",
            redact_sensitive=redact_secrets,
            layer=Layer.TOOL_OUTPUT,
        )

    # ----------------------------------------------------------------- output
    def check_output(
        self,
        text: str,
        *,
        session_id: str = _DEFAULT_SESSION,
        allowed_citations: Sequence[str] | None = None,
    ) -> Verdict:
        """Redact, validate grounding, enforce cumulative volume. Last gate."""
        timer = Timer()
        findings, truncated = self._scan(text, Layer.OUTPUT)
        all_findings = list(findings)

        red: RedactionResult = redact(text, findings, self.config.redaction)

        if allowed_citations is not None:
            ground = self.grounding.check(text, allowed_citations, self.config)
            all_findings.extend(ground.findings)

        pii_count = sum(1 for f in findings if f.category is Category.PII)
        self.ledger.record(session_id, bytes_=len(text), pii_values=pii_count)
        volume = self.ledger.check(session_id, self.config.volume)
        if volume is not None:
            all_findings.append(volume)

        verdict = self._verdict(
            Layer.OUTPUT,
            all_findings,
            timer,
            truncated=truncated,
            sanitized=red.text,
            meta={
                "redactions": red.to_audit(),
                "redaction_count": red.count,
                "session_id": session_id,
            },
        )
        if verdict.blocked:
            return Verdict(
                layer=verdict.layer,
                action=verdict.action,
                score=verdict.score,
                findings=verdict.findings,
                policy_id=verdict.policy_id,
                latency_us=verdict.latency_us,
                truncated=verdict.truncated,
                sanitized=None,  # suppressed: nothing leaves on a block
                meta=verdict.meta,
            )
        return verdict

    # ----------------------------------------------------------------- extras
    def refusal(self, verdict: Verdict) -> str:
        charter_refusal = (
            self.scope_gate.charter.refusal() if self.scope_gate is not None else None
        )
        return refusal_for(verdict, charter_refusal=charter_refusal)

    def check_input_or_raise(self, text: str, *, session_id: str = _DEFAULT_SESSION) -> Verdict:
        verdict = self.check_input(text, session_id=session_id)
        if verdict.blocked:
            raise GuardrailBlocked(verdict, self.refusal(verdict))
        return verdict

    @property
    def cache_stats(self) -> dict[str, float | int]:
        return {
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "hit_rate": round(self._cache.hit_rate, 4),
        }

    def rule_count(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for detector in self._detectors:
            if isinstance(detector, RuleDetector):
                for layer in Layer:
                    out[layer.value] = detector.rule_count(layer)
        return out
