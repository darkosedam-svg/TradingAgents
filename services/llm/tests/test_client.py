import asyncio
import json

import httpx
import pytest

from services.llm.client import (
    BreakerState,
    CircuitBreaker,
    InMemoryMetrics,
    LLMClient,
    LLMSettings,
    LocalUnavailable,
)
from services.llm.schemas import AbstainReason, SentimentVote


def completion(payload: dict, prompt_tokens: int = 100, completion_tokens: int = 20) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def make_client(handler, **overrides) -> LLMClient:
    settings = LLMSettings(timeout_s=1.0, max_retries=1, **overrides)
    return LLMClient(
        settings,
        transport=httpx.MockTransport(handler),
        metrics=InMemoryMetrics(),
    )


async def _one_call(client: LLMClient, content: str = "headline"):
    async with client:
        return await client.complete("sentiment", SentimentVote, content)


def test_happy_path_returns_a_vote():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=completion(
                {
                    "sentiment": "bullish",
                    "confidence": 0.82,
                    "assets": ["ETH"],
                    "rationale": "inflows",
                    "insufficient_data": False,
                }
            ),
        )

    outcome = asyncio.run(_one_call(make_client(handler)))

    assert outcome.abstained is False
    assert outcome.value.sentiment == "bullish"
    assert outcome.prompt_tokens == 100 and outcome.completion_tokens == 20
    assert outcome.prompt_sha.startswith("sentiment.v1@")

    # Every call is schema-constrained; nothing goes out without guided_json.
    body = seen[0]
    assert body["guided_json"]["properties"]["sentiment"]
    assert body["temperature"] == 0.0
    assert body["messages"][0]["role"] == "system"


def test_low_confidence_becomes_an_abstention_not_a_vote():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=completion(
                {
                    "sentiment": "bearish",
                    "confidence": 0.31,
                    "assets": ["BTC"],
                    "rationale": "unclear",
                    "insufficient_data": False,
                }
            ),
        )

    outcome = asyncio.run(_one_call(make_client(handler)))
    assert outcome.reason is AbstainReason.LOW_CONFIDENCE
    assert outcome.value is None


def test_malformed_response_becomes_schema_fail():
    def handler(request: httpx.Request) -> httpx.Response:
        # Guided decoding should make this impossible; the layer still has to
        # convert it into a non-vote rather than a default.
        return httpx.Response(200, json=completion({"sentiment": "up", "confidence": 0.99}))

    client = make_client(handler)
    outcome = asyncio.run(_one_call(client))

    assert outcome.reason is AbstainReason.SCHEMA_FAIL
    snapshot = client.metrics.snapshot()["sentiment"]
    assert snapshot["parse_failure_rate"] == 1.0


def test_insufficient_data_flag_wins_over_high_confidence():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=completion(
                {
                    "sentiment": "neutral",
                    "confidence": 0.97,
                    "assets": [],
                    "rationale": "",
                    "insufficient_data": True,
                }
            ),
        )

    outcome = asyncio.run(_one_call(make_client(handler)))
    assert outcome.reason is AbstainReason.INSUFFICIENT_DATA


def test_server_error_is_retried_once_then_raises_and_trips_the_breaker():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="engine busy")

    breaker = CircuitBreaker(threshold=1, cooldown_s=60.0)
    client = LLMClient(
        LLMSettings(timeout_s=1.0, max_retries=1),
        transport=httpx.MockTransport(handler),
        metrics=InMemoryMetrics(),
        breaker=breaker,
    )

    with pytest.raises(LocalUnavailable):
        asyncio.run(_one_call(client))

    assert calls["n"] == 2  # one attempt plus one retry
    assert breaker.state is BreakerState.OPEN


def test_client_refuses_to_call_while_the_breaker_is_open():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not reach the transport")

    breaker = CircuitBreaker(threshold=1, cooldown_s=60.0)
    breaker.record_failure()
    client = LLMClient(
        LLMSettings(),
        transport=httpx.MockTransport(handler),
        breaker=breaker,
    )

    with pytest.raises(LocalUnavailable, match="circuit breaker open"):
        asyncio.run(_one_call(client))


def test_client_errors_are_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, text="bad guided_json")

    with pytest.raises(LocalUnavailable, match="request rejected"):
        asyncio.run(_one_call(make_client(handler)))

    assert calls["n"] == 1


def test_timeout_is_retried_then_surfaces_as_unavailable():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(LocalUnavailable):
        asyncio.run(_one_call(make_client(handler)))

    assert calls["n"] == 2


def test_health_and_warm_probe_the_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json=completion({"sentiment": "neutral", "confidence": 0.5}))

    async def scenario():
        async with make_client(handler) as client:
            return await client.health(), await client.warm()

    assert asyncio.run(scenario()) == (True, True)


def test_health_is_false_when_the_endpoint_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def scenario():
        async with make_client(handler) as client:
            return await client.health(), await client.warm()

    assert asyncio.run(scenario()) == (False, False)


def test_metrics_capture_latency_tokens_and_abstentions():
    responses = [
        {"sentiment": "bullish", "confidence": 0.9, "assets": ["ETH"], "rationale": "", "insufficient_data": False},
        {"sentiment": "bearish", "confidence": 0.1, "assets": ["BTC"], "rationale": "", "insufficient_data": False},
    ]
    index = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = responses[index["n"] % len(responses)]
        index["n"] += 1
        return httpx.Response(200, json=completion(payload))

    client = make_client(handler)

    async def scenario():
        async with client:
            await client.complete("sentiment", SentimentVote, "a")
            await client.complete("sentiment", SentimentVote, "b")

    asyncio.run(scenario())
    stats = client.metrics.snapshot()["sentiment"]

    assert stats["calls"] == 2
    assert stats["abstain_rate"] == 0.5
    assert stats["abstain_by_reason"] == {"LOW_CONFIDENCE": 1}
    assert stats["total_tokens"] == 240
    assert stats["p95_latency_s"] >= stats["p50_latency_s"]
