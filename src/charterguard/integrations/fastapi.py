"""FastAPI wiring.

Kept out of the core package so `charterguard` stays dependency-free:
importing this module is the only thing that requires fastapi.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

try:
    from fastapi import Request, status
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("charterguard.integrations.fastapi requires fastapi") from exc

from ..exceptions import GuardrailBlocked
from ..pipeline import GuardrailPipeline
from ..types import Action, Verdict


def install_exception_handler(app: Any) -> None:
    @app.exception_handler(GuardrailBlocked)
    async def _handler(_: Request, exc: GuardrailBlocked) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "guardrail_blocked",
                "policy_id": exc.verdict.policy_id,
                "message": str(exc),
                "audit": exc.verdict.to_audit(),
            },
        )


def input_guard(
    pipeline: GuardrailPipeline,
    *,
    text_field: str = "message",
    session_field: str = "session_id",
    enforce_scope: bool = False,
) -> Callable[..., Any]:
    """Dependency that blocks or flags the incoming turn.

    The returned verdict is injected into the handler even when it is not
    blocking, so the handler can attach it to its audit row and see FLAG
    outcomes during a phased rollout.

    Usage:
        guard = input_guard(pipeline, enforce_scope=True)

        @app.post("/turn")
        async def turn(body: TurnBody, verdict: Verdict = Depends(guard)):
            ...
    """

    async def _dependency(request: Request) -> Verdict:
        body = await request.json()
        if isinstance(body, dict):
            text = str(body.get(text_field, "") or "")
            session_id = str(body.get(session_field, "default") or "default")
        else:
            text, session_id = str(body), "default"

        verdict = await pipeline.acheck_input(text, session_id=session_id)
        if verdict.blocked:
            raise GuardrailBlocked(verdict, pipeline.refusal(verdict))

        if enforce_scope:
            scope_verdict = await pipeline.acheck_scope(text)
            if scope_verdict.blocked:
                raise GuardrailBlocked(scope_verdict, pipeline.refusal(scope_verdict))
            if scope_verdict.action is not Action.ALLOW:
                # Surface the scope finding without discarding the input verdict.
                return replace(
                    verdict,
                    action=scope_verdict.action,
                    findings=verdict.findings + scope_verdict.findings,
                    score=max(verdict.score, scope_verdict.score),
                    policy_id=scope_verdict.policy_id,
                )
        return verdict

    return _dependency


def guard_response(
    pipeline: GuardrailPipeline,
    text: str,
    *,
    allowed_citations: Sequence[str] | None = None,
) -> tuple[str, Verdict]:
    """Run the outbound checks and return the text that is safe to send.

    Returns the redacted or refusal text rather than raising, because an outbound
    block must still produce a well-formed response to the user.
    """
    verdict = pipeline.check_output(
        text,
        allowed_citations=None if allowed_citations is None else tuple(allowed_citations),
    )
    if verdict.blocked:
        return pipeline.refusal(verdict), verdict
    return (verdict.sanitized if verdict.sanitized is not None else text), verdict
