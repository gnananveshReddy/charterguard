"""Model, intent and NER adapters over plain HTTP -- still zero dependencies.

`urllib.request` can talk to any moderation endpoint, inference server
(TGI, vLLM, Triton, TorchServe, Ollama, llama.cpp) or internal classifier
service. That removes the last common reason to add a vendor SDK: you do not
need `openai` to POST JSON to an OpenAI-compatible URL.

Three engineering points that matter more than the transport choice:

  * **Blocking I/O off the event loop.** `urllib` is synchronous. Calling it
    directly from `async def score` would stall every other request on the
    loop for the duration of the round trip. Every call here goes through
    `asyncio.to_thread`, and the thread is bounded by an explicit timeout.
  * **Fail policy belongs to the caller.** These adapters raise on transport
    failure; `ModelDetector` already resolves timeouts and exceptions through
    `fail_closed` for the layer. Swallowing errors here would silently
    downgrade a regulated deployment to fail-open.
  * **Cleartext is refused.** Prompts are the most sensitive payload in the
    system and often carry the PII the guardrail exists to protect. `https` is
    required unless `allow_insecure_http=True` is passed explicitly, and an
    auth header is never sent over a plaintext URL under any setting.

Response shapes are not guessable, so the mapping is explicit: supply
`parse=` (and `build_payload=` if the request shape is unusual). Two presets
are included for the shapes people actually hit.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from collections.abc import Callable, Mapping, Sequence

from ..pii import Entity
from ..types import Category, Layer

__all__ = [
    "HttpError",
    "HttpModelScorer",
    "OpenAIModerationScorer",
    "HttpIntentClassifier",
    "HttpEntityRecognizer",
]

_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "proxy-authorization"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    urllib follows 301/302/303 by default and re-sends ordinary request headers,
    including ``Authorization`` and ``x-api-key``, to the new location. For a
    client that carries a credential and prompt content, a redirect is either a
    misconfigured URL or an attempt to move both somewhere else -- possibly to
    plaintext http. Failing loudly makes the operator configure the final URL.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


# A private opener: installing a global opener would change urllib behaviour for
# the host application.
_OPENER = urllib.request.build_opener(_NoRedirect)


def _validate_endpoint(url: str, headers: Mapping[str, str] | None, allow_insecure_http: bool) -> None:
    """Allow ``https``, and ``http`` only on explicit opt-in. Nothing else.

    A prefix check is not enough: with the opt-in set, ``file://``, ``ftp://``
    or a custom scheme would otherwise reach ``urlopen``, which reads local
    files for ``file:``.
    """
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in {"https", "http"}:
        raise ValueError(f"unsupported URL scheme {parts.scheme!r} in {url!r}; use https")
    if not parts.hostname:
        raise ValueError(f"URL has no host: {url!r}")
    if scheme == "http":
        if not allow_insecure_http:
            raise ValueError(
                f"refusing to send prompt content in cleartext to {url!r}; "
                "pass allow_insecure_http=True only for localhost or an in-mesh sidecar"
            )
        if {k.lower() for k in (headers or {})} & _CREDENTIAL_HEADERS:
            raise ValueError("refusing to send a credential header over plaintext http; use https")


class HttpError(RuntimeError):
    """Transport or protocol failure. Propagated so the layer's `fail_closed`
    policy -- not this module -- decides what it means."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _JsonClient:
    """Minimal retrying JSON client. Shared by the adapters below."""

    __slots__ = ("url", "headers", "timeout_s", "max_retries", "backoff_s", "max_chars")

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 2.0,
        max_retries: int = 1,
        backoff_s: float = 0.15,
        max_chars: int = 20_000,
        allow_insecure_http: bool = False,
    ) -> None:
        _validate_endpoint(url, headers, allow_insecure_http)
        self.url = url
        self.headers = {"content-type": "application/json", **dict(headers or {})}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_chars = max_chars

    def _post_sync(self, payload: Mapping[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")  # noqa: S310 - scheme validated in __init__
            try:
                # Scheme and host were validated in __init__; redirects are refused.
                with _OPENER.open(req, timeout=self.timeout_s) as resp:  # noqa: S310
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # An HTTPError carries the open response; close it or the
                # socket leaks on every error or redirect status.
                exc.close()
                if 300 <= exc.code < 400:
                    raise HttpError(
                        f"{self.url} redirected ({exc.code}); refusing to follow -- "
                        "configure the final URL",
                        status=exc.code,
                    ) from exc
                last = HttpError(f"{self.url} returned {exc.code}", status=exc.code)
                if exc.code not in _RETRY_STATUS:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = HttpError(f"{self.url} unreachable: {exc}")
            except json.JSONDecodeError as exc:
                raise HttpError(f"{self.url} returned non-JSON body") from exc
            if attempt < self.max_retries:
                # Blocking sleep is correct here: we are already inside a worker
                # thread, so this parks the thread, not the event loop.
                import time

                time.sleep(self.backoff_s * (2 ** attempt))
        raise last or HttpError("request failed")

    async def post(self, payload: Mapping[str, Any]) -> Any:
        # Hard ceiling on wall time even if a socket ignores its timeout.
        budget = self.timeout_s * (self.max_retries + 1) + self.backoff_s * 4
        return await asyncio.wait_for(asyncio.to_thread(self._post_sync, payload), timeout=budget)

    def clip(self, text: str) -> str:
        return text if len(text) <= self.max_chars else text[: self.max_chars]


class HttpModelScorer:
    """`ModelScorer` over any JSON classifier endpoint.

    Args:
        parse: maps the decoded response to ``{Category: score}``. Required,
            because no two moderation APIs agree on a response shape and a
            wrong guess silently produces zeros -- a guardrail that always
            allows. Scores outside [0, 1] are clamped.
        build_payload: request body builder. Defaults to
            ``{"text": ..., "categories": [...], "layer": ...}``.
    """

    __slots__ = ("_client", "_parse", "_build")

    def __init__(
        self,
        url: str,
        *,
        parse: Callable[[Any, Sequence[Category]], Mapping[Category, float]],
        build_payload: Callable[[str, Sequence[Category], Layer], Mapping[str, Any]] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 2.0,
        max_retries: int = 1,
        max_chars: int = 20_000,
        allow_insecure_http: bool = False,
    ) -> None:
        self._client = _JsonClient(
            url,
            headers=headers,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_chars=max_chars,
            allow_insecure_http=allow_insecure_http,
        )
        self._parse = parse
        self._build = build_payload or self._default_payload

    @staticmethod
    def _default_payload(
        text: str, categories: Sequence[Category], layer: Layer
    ) -> Mapping[str, Any]:
        return {
            "text": text,
            "categories": [c.value for c in categories],
            "layer": layer.value,
        }

    async def score(
        self, text: str, categories: Sequence[Category], layer: Layer
    ) -> Mapping[Category, float]:
        raw = await self._client.post(self._build(self._client.clip(text), categories, layer))
        scores = self._parse(raw, categories)
        return {c: min(1.0, max(0.0, float(s))) for c, s in scores.items()}


def _openai_parse(raw: Any, categories: Sequence[Category]) -> Mapping[Category, float]:
    """Map OpenAI moderation `category_scores` onto guardrail categories.

    The mapping is many-to-one and lossy in one direction that matters:
    OpenAI has no notion of prompt injection or scope violation, so
    INJECTION/JAILBREAK never receive a signal here. Use this tier for harm
    categories and keep the deterministic rules for injection.
    """
    try:
        cs = raw["results"][0]["category_scores"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HttpError("unexpected OpenAI moderation response shape") from exc
    groups: dict[Category, tuple[str, ...]] = {
        Category.SELF_HARM: ("self-harm", "self-harm/intent", "self-harm/instructions"),
        Category.HARMFUL: (
            "violence",
            "violence/graphic",
            "harassment",
            "harassment/threatening",
            "hate",
            "hate/threatening",
            "illicit",
            "illicit/violent",
            "sexual/minors",
        ),
    }
    out: dict[Category, float] = {}
    for cat in categories:
        keys = groups.get(cat)
        if not keys:
            continue
        vals = [float(cs[k]) for k in keys if k in cs]
        if vals:
            out[cat] = max(vals)
    return out


class OpenAIModerationScorer(HttpModelScorer):
    """Preset for OpenAI's `/v1/moderations` response shape.

    Works against OpenAI and any gateway that mirrors it (Azure OpenAI,
    LiteLLM, an internal proxy). Sends only the text; no other request field is
    populated, so nothing about the session leaves the process.
    """

    def __init__(
        self,
        api_key: str,
        *,
        url: str = "https://api.openai.com/v1/moderations",
        model: str = "omni-moderation-latest",
        timeout_s: float = 2.0,
        max_retries: int = 1,
        max_chars: int = 20_000,
    ) -> None:
        super().__init__(
            url,
            parse=_openai_parse,
            build_payload=lambda text, cats, layer: {"model": model, "input": text},
            headers={"authorization": f"Bearer {api_key}"},
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_chars=max_chars,
        )


class HttpIntentClassifier:
    """`IntentClassifier` over an HTTP endpoint.

    Must return one of ``in_scope`` / ``out_of_scope`` / ``unclear`` with a
    confidence; `ScopeGate` treats any non-`out_of_scope` label at confidence
    >= 0.6 as a rescue for a request the lexical gate flagged as ambiguous, so a
    miscalibrated endpoint here widens scope. Keep the threshold in mind when
    choosing a model.
    """

    __slots__ = ("_client", "_parse")

    def __init__(
        self,
        url: str,
        *,
        parse: Callable[[Any], tuple[str, float]] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 2.0,
        max_retries: int = 1,
        allow_insecure_http: bool = False,
    ) -> None:
        self._client = _JsonClient(
            url,
            headers=headers,
            timeout_s=timeout_s,
            max_retries=max_retries,
            allow_insecure_http=allow_insecure_http,
        )
        self._parse = parse or self._default_parse

    @staticmethod
    def _default_parse(raw: Any) -> tuple[str, float]:
        label = str(raw.get("label", "unclear"))
        if label not in ("in_scope", "out_of_scope", "unclear"):
            label = "unclear"
        return label, float(raw.get("confidence", 0.0))

    async def classify(
        self, text: str, in_scope: Sequence[str], out_of_scope: Sequence[str]
    ) -> tuple[str, float]:
        raw = await self._client.post(
            {
                "text": self._client.clip(text),
                "in_scope": list(in_scope),
                "out_of_scope": list(out_of_scope),
            }
        )
        return self._parse(raw)


class HttpEntityRecognizer:
    """`EntityRecognizer` over a remote NER or DLP service.

    Offsets are the whole contract: the service must return spans indexing the
    *exact string it was given*, which is the normalized text. A service that
    strips, lowercases or re-tokenises before computing offsets will produce
    misaligned redactions -- a leak, not a formatting bug. Spans that fall
    outside the input are dropped rather than trusted.

    Note the data-flow consequence: this sends candidate PII to a third party.
    That is often exactly what a DLP contract permits, and exactly what a
    residency requirement forbids. Decide before wiring it, and prefer the
    in-process spaCy/Presidio adapters where residency is constrained.
    """

    __slots__ = ("_client", "_parse", "_min_score")

    def __init__(
        self,
        url: str,
        *,
        parse: Callable[[Any], Sequence[Entity]] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 2.0,
        min_score: float = 0.35,
        allow_insecure_http: bool = False,
    ) -> None:
        self._client = _JsonClient(
            url, headers=headers, timeout_s=timeout_s, allow_insecure_http=allow_insecure_http
        )
        self._parse = parse or self._default_parse
        self._min_score = min_score

    @staticmethod
    def _default_parse(raw: Any) -> Sequence[Entity]:
        return [
            Entity(
                label=str(e["label"]),
                start=int(e["start"]),
                end=int(e["end"]),
                score=float(e.get("score", 0.8)),
            )
            for e in raw.get("entities", [])
        ]

    def recognize(self, text: str) -> Sequence[Entity]:
        """Synchronous, per the `EntityRecognizer` protocol.

        This blocks the calling thread on network I/O. Inside an async service,
        run the pipeline call itself in a worker (`asyncio.to_thread`) or prefer
        an in-process recogniser; do not call this on the event loop.
        """
        raw = self._client._post_sync({"text": self._client.clip(text)})
        n = len(text)
        return [
            e
            for e in self._parse(raw)
            if e.score >= self._min_score and 0 <= e.start < e.end <= n
        ]
