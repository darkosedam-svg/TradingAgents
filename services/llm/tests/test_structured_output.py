import asyncio
import json

import httpx
import pytest

from services.llm.client import LLMClient, LLMSettings, extract_json
from services.llm.client.schema_tools import (
    json_schema_for,
    response_format_for,
    schema_hint,
)
from services.llm.schemas import AbstainReason, ParsedSignal, SentimentVote

from .test_client import completion, make_client


def test_strict_schema_requires_every_property():
    """Strict mode has no optional fields, so defaults must still be listed."""
    schema = json_schema_for(SentimentVote, strict=True)

    assert set(schema["required"]) == set(schema["properties"])
    assert "assets" in schema["required"]  # has a default, still required
    assert schema["additionalProperties"] is False


def test_non_strict_schema_leaves_optionality_alone():
    schema = json_schema_for(SentimentVote, strict=False)
    assert "assets" not in schema.get("required", [])
    assert schema["additionalProperties"] is False


def test_unsupported_validation_keywords_are_stripped():
    """Range checks live in Pydantic, not in the decoder's grammar."""
    schema = json_schema_for(SentimentVote)
    confidence = schema["properties"]["confidence"]

    assert "minimum" not in confidence and "maximum" not in confidence
    assert confidence["type"] == "number"
    # ...but the model still rejects an out-of-range value on the way in.
    with pytest.raises(Exception):
        SentimentVote(sentiment="bullish", confidence=1.5)


def test_stripping_recurses_into_nested_definitions():
    schema = json_schema_for(ParsedSignal)
    blob = json.dumps(schema)

    for keyword in ("maxLength", "maxItems", "exclusiveMinimum", "default"):
        assert keyword not in blob, keyword


def test_enums_and_descriptions_survive():
    schema = json_schema_for(SentimentVote)
    sentiment = schema["properties"]["sentiment"]
    assert set(sentiment["enum"]) == {"bullish", "bearish", "neutral"}
    assert schema["properties"]["assets"]["description"]


def test_response_format_block_shape():
    block = response_format_for(SentimentVote)
    assert block["type"] == "json_schema"
    assert block["json_schema"]["name"] == "SentimentVote"
    assert block["json_schema"]["strict"] is True
    assert block["json_schema"]["schema"]["properties"]["sentiment"]


def test_schema_hint_names_every_field_and_its_options():
    hint = schema_hint(SentimentVote)
    for field in ("sentiment", "assets", "confidence", "insufficient_data"):
        assert field in hint
    assert "'bullish'" in hint
    assert "array of string" in hint


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ("```\n{\"a\": 1}\n```", '{"a": 1}'),
        ('Here you go:\n{"a": 1}\nHope that helps!', '{"a": 1}'),
        ('{"a": {"b": 2}} trailing', '{"a": {"b": 2}}'),
        ('{"a": "}"}', '{"a": "}"}'),
        ('{"a": "say \\"}\\" ok"}', '{"a": "say \\"}\\" ok"}'),
        ("", ""),
        ("no json at all", "no json at all"),
    ],
)
def test_extract_json_recovers_the_object(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_output_is_parseable():
    recovered = extract_json('Sure!\n```json\n{"sentiment": "bullish"}\n```\nDone.')
    assert json.loads(recovered) == {"sentiment": "bullish"}


def sentiment_body() -> dict:
    return completion(
        {
            "sentiment": "bullish",
            "confidence": 0.9,
            "assets": ["ETH"],
            "rationale": "inflows",
            "insufficient_data": False,
        }
    )


def capture(mode: str) -> dict:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=sentiment_body())

    client = make_client(handler, structured_mode=mode)

    async def scenario():
        async with client:
            return await client.complete("sentiment", SentimentVote, "headline")

    outcome = asyncio.run(scenario())
    assert not outcome.abstained
    return seen[0]


def test_json_schema_mode_sends_response_format_only():
    body = capture("json_schema")
    assert "response_format" in body
    assert "guided_json" not in body


def test_guided_json_mode_sends_the_vllm_extension_only():
    body = capture("guided_json")
    assert "guided_json" in body
    assert "response_format" not in body
    # Non-strict: a self-hosted grammar backend handles optional fields fine.
    assert "assets" not in body["guided_json"].get("required", [])


def test_prompt_mode_puts_the_contract_in_the_system_message():
    body = capture("prompt")
    assert "response_format" not in body and "guided_json" not in body

    system = body["messages"][0]["content"]
    assert "single JSON object" in system
    assert "sentiment" in system


def test_prompt_mode_survives_a_fenced_response():
    """The failure mode `prompt` mode exists to tolerate."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.dumps(
            {
                "sentiment": "bearish",
                "confidence": 0.88,
                "assets": ["BTC"],
                "rationale": "outflows",
                "insufficient_data": False,
            }
        )
        return httpx.Response(
            200,
            json=completion({}) | {
                "choices": [
                    {"message": {"content": f"Sure thing!\n```json\n{payload}\n```"}}
                ]
            },
        )

    client = make_client(handler, structured_mode="prompt")

    async def scenario():
        async with client:
            return await client.complete("sentiment", SentimentVote, "headline")

    outcome = asyncio.run(scenario())
    assert not outcome.abstained
    assert outcome.value.sentiment == "bearish"


def test_prose_only_response_abstains_rather_than_guessing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I think it's probably bullish."}}]},
        )

    client = make_client(handler, structured_mode="prompt")

    async def scenario():
        async with client:
            return await client.complete("sentiment", SentimentVote, "headline")

    assert asyncio.run(scenario()).reason is AbstainReason.SCHEMA_FAIL


def test_unknown_structured_mode_is_rejected_loudly():
    client = LLMClient(
        LLMSettings(structured_mode="telepathy"),  # type: ignore[arg-type]
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=sentiment_body())),
    )

    async def scenario():
        async with client:
            return await client.complete("sentiment", SentimentVote, "headline")

    with pytest.raises(ValueError, match="unknown structured_mode"):
        asyncio.run(scenario())
