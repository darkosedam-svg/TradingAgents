"""Async client for the local vLLM endpoint.

Talks the OpenAI-compatible ``/chat/completions`` surface over plain HTTP rather
than through the ``openai`` SDK: the layer needs a hard timeout, exactly one
retry, and a circuit breaker wrapped tightly around the transport, and it is
easier to be sure of all three when the request is a single ``httpx`` call.
Every request is schema-constrained via vLLM's ``guided_json`` — no free-text
parsing anywhere in this layer.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional, Type, TypeVar

import httpx

from .. import prompts
from ..schemas.base import AbstainReason, Outcome, TaskOutput
from .breaker import CircuitBreaker
from .config import LLMSettings
from .observability import CallRecord, MetricsSink, NullMetrics

T = TypeVar("T", bound=TaskOutput)


class LocalUnavailable(RuntimeError):
    """The local endpoint could not be reached, or refused to answer.

    Distinct from a bad *answer*: a bad answer is an abstention, this is a
    routing signal telling the caller to fall back to the hosted API.
    """


class LLMClient:
    """Schema-constrained async client with retries and a circuit breaker."""

    def __init__(
        self,
        settings: Optional[LLMSettings] = None,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        metrics: Optional[MetricsSink] = None,
        breaker: Optional[CircuitBreaker] = None,
        source: str = "local",
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
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
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
            response = await self._http.get("/models", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def warm(self, task: str = "sentiment") -> bool:
        """Force weights and the prefix cache resident before first real use.

        Cold start is 20-40s. Call this on boot and gate routing on it; never
        let it happen inside a decision.
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
                timeout=90.0,
            )
            self.breaker.record_success()
            return True
        except (httpx.HTTPError, LocalUnavailable):
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

        Raises :class:`LocalUnavailable` when the endpoint itself is the
        problem; every other failure mode — unparseable JSON, a value the
        validators reject, a confidence below the task floor — comes back as an
        abstaining :class:`Outcome`.
        """
        if not self.breaker.allows_local():
            raise LocalUnavailable(
                f"circuit breaker open after {self.breaker.consecutive_failures} failures"
            )

        prompt = prompts.load(task, prompt_version)
        system = f"{prompt.text}\n\n{extra_system}".strip() if extra_system else prompt.text
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
            # vLLM extension: XGrammar constrains decoding to this schema, so
            # the response is guaranteed to parse as JSON of the right shape.
            "guided_json": schema.model_json_schema(),
        }

        started = time.monotonic()
        try:
            body = await self._post(payload)
        except LocalUnavailable as exc:
            self.breaker.record_failure()
            self._emit(
                task,
                prompt.ref,
                time.monotonic() - started,
                ok=False,
                error=str(exc),
            )
            raise

        self.breaker.record_success()
        latency = time.monotonic() - started
        usage = body.get("usage") or {}
        common = {
            "latency_s": latency,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "model": self.settings.model,
            "prompt_sha": prompt.ref,
            "source": self.source,
        }

        content = self._extract_content(body)
        try:
            value = schema.model_validate_json(content)
        except Exception as exc:  # pydantic ValidationError, JSON errors, both
            # Guided decoding should make this impossible. When it happens
            # anyway it is exactly the failure this layer exists to contain:
            # convert it into a non-vote, never a default.
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
                reason, detail=f"confidence={value.confidence:.2f} floor={floor:.2f}", **common
            )
        else:
            outcome = Outcome.voted(value, **common)

        self._emit(
            task,
            prompt.ref,
            latency,
            ok=True,
            reason=reason,
            escalated=escalated,
            usage=common,
        )
        return outcome

    # --------------------------------------------------------------- internals

    async def _post(
        self, payload: dict[str, Any], timeout: Optional[float] = None
    ) -> dict[str, Any]:
        """POST with one retry. Retries transport errors and 5xx, nothing else.

        A 4xx means the request is wrong; retrying it just spends the timeout
        budget twice before failing the same way.
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
                if response.status_code >= 500:
                    raise LocalUnavailable(
                        f"upstream {response.status_code}: {response.text[:200]}"
                    )
                if response.status_code >= 400:
                    raise LocalUnavailable(
                        f"request rejected {response.status_code}: {response.text[:200]}"
                    )
                return response.json()
            except LocalUnavailable as exc:
                last = exc
                if "request rejected" in str(exc):
                    raise
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last = LocalUnavailable(f"{type(exc).__name__}: {exc}")

            if attempt < attempts - 1:
                await asyncio.sleep(0.25 * (attempt + 1))

        raise last or LocalUnavailable("unknown transport failure")

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
