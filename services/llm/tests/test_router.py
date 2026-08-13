import asyncio

import httpx
import pytest

from services.llm.client import (
    BudgetLedger,
    EscalationPolicy,
    EscalationRateMonitor,
    LLMClient,
    LLMSettings,
    UpstreamUnavailable,
    Router,
    TaskBudget,
)
from services.llm.schemas import AbstainReason, NewsTriage, Outcome, SentimentVote

from .test_client import completion


class FallbackStub:
    """Stands in for the second provider."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, task, schema, user_content, **kwargs):
        self.calls.append((task, user_content))
        return Outcome.voted(
            SentimentVote(sentiment="neutral", confidence=0.9), source="fallback"
        )


def dead_endpoint() -> LLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return LLMClient(
        LLMSettings(timeout_s=0.5, max_retries=0),
        transport=httpx.MockTransport(handler),
    )


def live_endpoint(payload: dict) -> LLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(payload))

    return LLMClient(
        LLMSettings(timeout_s=1.0), transport=httpx.MockTransport(handler)
    )


def test_router_falls_back_to_the_second_provider_when_the_first_is_down():
    fallback = FallbackStub()
    router = Router(dead_endpoint(), fallback)

    outcome = asyncio.run(router.run("sentiment", SentimentVote, "headline"))

    assert outcome.source == "fallback"
    assert fallback.calls == [("sentiment", "headline")]


def test_router_abstains_when_the_endpoint_is_down_and_there_is_no_fallback():
    router = Router(dead_endpoint())
    outcome = asyncio.run(router.run("sentiment", SentimentVote, "headline"))

    assert outcome.reason is AbstainReason.INSUFFICIENT_DATA
    assert "upstream unavailable" in outcome.detail


def test_router_can_be_told_to_raise_instead_of_abstaining():
    router = Router(dead_endpoint(), abstain_on_unavailable=False)
    with pytest.raises(UpstreamUnavailable):
        asyncio.run(router.run("sentiment", SentimentVote, "headline"))


def test_router_charges_the_ledger_and_falls_back_once_it_is_spent():
    fallback = FallbackStub()
    ledger = BudgetLedger(budgets={"sentiment": TaskBudget(max_tokens_per_day=100)})
    router = Router(
        live_endpoint(
            {
                "sentiment": "bullish",
                "confidence": 0.9,
                "assets": ["ETH"],
                "rationale": "",
                "insufficient_data": False,
            }
        ),
        fallback,
        ledger=ledger,
    )

    async def scenario():
        first = await router.run("sentiment", SentimentVote, "a")
        second = await router.run("sentiment", SentimentVote, "b")
        return first, second

    first, second = asyncio.run(scenario())

    assert first.source == "primary"
    assert ledger.spent("sentiment")[0] == 120
    assert second.source == "fallback"  # budget exhausted, routed out


def triage(**overrides) -> Outcome[NewsTriage]:
    payload = {
        "confidence": 0.8,
        "relevance": 0.9,
        "market_ids": ["mkt-1"],
        "direction": "yes",
        "escalate": False,
        "priority": 0.1,
        "rationale": "",
    }
    payload.update(overrides)
    return Outcome.voted(NewsTriage(**payload))


def test_policy_escalates_when_the_model_says_so():
    decision = EscalationPolicy().decide(triage(escalate=True, priority=0.05))
    assert decision.escalate is True
    assert decision.priority >= 0.35  # floored, so it does not sort to the bottom


def test_policy_escalates_on_priority_alone():
    decision = EscalationPolicy(priority_threshold=0.35).decide(triage(priority=0.6))
    assert decision.escalate is True


def test_policy_escalates_a_relevant_item_the_model_deprioritised():
    # Recall over precision: relevance plus a named market is enough.
    decision = EscalationPolicy().decide(triage(priority=0.05, relevance=0.5))
    assert decision.escalate is True
    assert "relevant to 1 market" in decision.why


def test_policy_skips_only_when_nothing_argues_for_escalation():
    decision = EscalationPolicy().decide(
        triage(priority=0.05, relevance=0.05, direction="unclear", market_ids=[])
    )
    assert decision.escalate is False


def test_policy_escalates_on_abstention():
    for reason in AbstainReason:
        decision = EscalationPolicy().decide(Outcome.abstain(reason))
        assert decision.escalate is True, reason
        assert decision.priority == 1.0


def test_policy_can_be_configured_to_drop_abstentions():
    policy = EscalationPolicy(escalate_on_abstain=False, escalate_on_low_confidence=False)
    decision = policy.decide(Outcome.abstain(AbstainReason.SCHEMA_FAIL))
    assert decision.escalate is False


def test_escalation_rate_monitor_flags_a_silent_drop():
    monitor = EscalationRateMonitor(drift_threshold=0.20)
    monitor.record_window(escalated=200, total=1000)
    monitor.record_window(escalated=120, total=1000)

    assert monitor.drift() == pytest.approx(-0.40)
    alarm = monitor.alarm()
    assert alarm is not None and "down 40.0%" in alarm


def test_escalation_rate_monitor_is_quiet_inside_the_band():
    monitor = EscalationRateMonitor(drift_threshold=0.20)
    monitor.record_window(escalated=200, total=1000)
    monitor.record_window(escalated=210, total=1000)
    assert monitor.alarm() is None


def test_escalation_rate_monitor_needs_two_windows():
    monitor = EscalationRateMonitor()
    monitor.record_window(escalated=10, total=100)
    assert monitor.drift() is None
    assert monitor.alarm() is None
