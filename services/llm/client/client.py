"""Async client for an OpenAI-compatible chat endpoint.

Talks raw HTTP rather than through a provider SDK: the layer needs a hard
timeout, exactly one retry, and a circuit breaker wrapped tightly around the
transport, and it is easier to be sure of all three when the request is a single
``httpx`` call. It also means the same client serves a hosted gateway, a
self-hosted vLLM container, or anything else that speaks the same shape.

Every request asks for structured output, by whichever mechanism the endpoint
supports. No free-text parsing anywhere in this layer.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Optional, Type, TypeVar

import httpx

from .. import prompts
from ..schemas.base import AbstainReason, Outcome, TaskOutput
from .breaker import CircuitBreaker
from .config import LLMSettings
from .observability import CallRecord, MetricsSink, NullMetrics
from .schema_tools import json_schema_for, response_format_for, schema_hint

T = TypeVar("T", bound=TaskOutput)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class UpstreamUnavailable(RuntimeError):
    """The endpoint could not be reached, or refused to answer.

    Distinct from a bad *answer*: a bad answer is an abstention, this is a
    routing signal telling the caller to try the fallback provider.
    """


def extract_json(content: str) -> str:
    """Pull the JSON object out of a response body.

    With strict structured output the content is already bare JSON and this is a
    no-op. In ``prompt`` mode a model may wrap it in a markdown fence or bracket
    it with commentary, so we recover the first balanced object rather than
    failing the whole call over a stray "Here you go:".
    """
    text = content.strip()
    if not text:
        return text

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text


class LLMClient:
    """Schema-constrained async client with retries and a circuit breaker."""

    def __init__(
        self,
        settings: Optional[LLMSettings] = None,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        metrics: Optional[MetricsSink] = None,
        breaker: Optional[CircuitBreaker] = None,
        source: str = "primary",
    ) -> None:
        self.settings = settings or LLMSettings()
        self.metrics: MetricsSink = metrics or NullMetrics()
        self.source = source
        self.breaker = breaker or CircuitBreaker(
            threshold=self.settings.breaker_threshold,
            cooldown_s=self.settings.breaker_cooldown_s,
        )
        self._http = httpx.AsyncClient(
            base_url=self.settings.base_url.rstrip("/"),
            timeout=httpx.Timeout(self.settings.timeout_s),
            headers=self.settings.headers(),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ health

    async def health(self) -> bool:
        """Cheap liveness probe against the served-models endpoint."""
        try:
            response = await self._http.get("/models", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def warm(self, task: str = "sentiment") -> bool:
        """One tiny real call, to fail fast on a bad key or an unknown model.

        Against a hosted endpoint there is no cold start to avoid; the value is
        finding out at boot rather than mid-decision that the credentials are
        wrong or the model slug has been retired.
        """
        try:
            await self._post(
                {
                    "model": self.settings.model,
                    "messages": [
                        {"role": "system", "content": prompts.load(task).text},
                        {"role": "user", "content": "warmup"},
                    ],
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
                timeout=60.0,
            )
            self.breaker.record_success()
            return True
        except (httpx.HTTPError, UpstreamUnavailable):
            return False

    # ------------------------------------------------------------------- calls

    async def complete(
        self,
        task: str,
        schema: Type[T],
        user_content: str,
        *,
        prompt_version: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_confidence: Optional[float] = None,
        extra_system: str = "",
        escalated: Optional[bool] = None,
    ) -> Outcome[T]:
        """Run one schema-constrained task and return a value or an abstention.

        Raises :class:`UpstreamUnavailable` when the endpoint itself is the
        problem; every other failure mode — unparseable JSON, a value the
        validators reject, a confidence below the task floor — comes back as an
        abstaining :class:`Outcome`.
        """
        if not self.breaker.allows_calls():
            raise UpstreamUnavailable(
                f"circuit breaker open after {self.breaker.consecutive_failures} failures"
            )

        prompt = prompts.load(task, prompt_version)
        system = prompt.text
        if self.settings.structured_mode == "prompt":
            system = f"{system}\n\n{schema_hint(schema)}"
        if extra_system:
            system = f"{system}\n\n{extra_system}"

        floor = (
            min_confidence
            if min_confidence is not None
            else self.settings.confidence_floor(task)
        )

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        self._apply_structured_output(payload, schema)

        started = time.monotonic()
        try:
            body = await self._post(payload)
        except UpstreamUnavailable as exc:
            self.breaker.record_failure()
            self._emit(task, prompt.ref, time.monotonic() - started, ok=False, error=str(exc))
            raise

        self.breaker.record_success()
        latency = time.monotonic() - started
        usage = body.get("usage") or {}
        common = {
            "latency_s": latency,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "model": self.settings.model,
            "prompt_sha": prompt.ref,
            "source": self.source,
        }

        content = extract_json(self._extract_content(body))
        try:
            value = schema.model_validate_json(content)
        except Exception as exc:  # pydantic ValidationError, JSON errors, both
            # Structured output should make this rare. When it happens anyway it
            # is exactly the failure this layer exists to contain: convert it
            # into a non-vote, never a default.
            outcome: Outcome[T] = Outcome.abstain(
                AbstainReason.SCHEMA_FAIL, detail=str(exc)[:500], **common
            )
            self._emit(
                task,
                prompt.ref,
                latency,
                ok=True,
                reason=AbstainReason.SCHEMA_FAIL,
                escalated=escalated,
                usage=common,
            )
            return outcome

        reason = value.abstain_reason(floor)
        if reason is not None:
            outcome = Outcome.abstain(
                reason,
                detail=f"confidence={value.confidence:.2f} floor={floor:.2f}",
                **common,
            )
        else:
            outcome = Outcome.voted(value, **common)

        self._emit(
            task, prompt.ref, latency, ok=True, reason=reason, escalated=escalated, usage=common
        )
        return outcome

    # --------------------------------------------------------------- internals

    def _apply_structured_output(self, payload: dict[str, Any], schema: Type[T]) -> None:
        mode = self.settings.structured_mode
        if mode == "json_schema":
            payload["response_format"] = response_format_for(schema)
        elif mode == "guided_json":
            payload["guided_json"] = json_schema_for(schema, strict=False)
        elif mode == "prompt":
            pass  # the contract went into the system prompt instead
        else:
            raise ValueError(f"unknown structured_mode {mode!r}")

    async def _post(
        self, payload: dict[str, Any], timeout: Optional[float] = None
    ) -> dict[str, Any]:
        """POST with one retry. Retries transport errors, 5xx and 429; not other 4xx.

        A 400 means the request is wrong — a bad model slug, or a schema the
        provider will not accept — and retrying it just spends the timeout
        budget twice before failing the same way. A 429 is the opposite: the
        request is fine and the only problem is timing.
        """
        attempts = self.settings.max_retries + 1
        last: Exception | None = None

        for attempt in range(attempts):
            try:
                response = await self._http.post(
                    "/chat/completions",
                    json=payload,
                    timeout=timeout or self.settings.timeout_s,
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise UpstreamUnavailable(
                        f"upstream {response.status_code}: {response.text[:200]}"
                    )
                if response.status_code >= 400:
                    raise UpstreamUnavailable(
                        f"request rejected {response.status_code}: {response.text[:200]}"
                    )
                return response.json()
            except UpstreamUnavailable as exc:
                last = exc
                if "request rejected" in str(exc):
                    raise
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last = UpstreamUnavailable(f"{type(exc).__name__}: {exc}")

            if attempt < attempts - 1:
                await asyncio.sleep(0.25 * (attempt + 1))

        raise last or UpstreamUnavailable("unknown transport failure")

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def _emit(
        self,
        task: str,
        prompt_ref: str,
        latency: float,
        *,
        ok: bool,
        reason: Optional[AbstainReason] = None,
        error: str = "",
        escalated: Optional[bool] = None,
        usage: Optional[dict[str, Any]] = None,
    ) -> None:
        usage = usage or {}
        self.metrics.record(
            CallRecord(
                task=task,
                model=self.settings.model,
                source=self.source,
                latency_s=latency,
                ok=ok,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                abstain_reason=reason,
                prompt_ref=prompt_ref,
                error=error,
                escalated=escalated,
                ts=time.time(),
            )
        )
