import asyncio
from pathlib import Path

import pytest

from services.llm.client.advisory import AdvisoryEnricher, with_deadline
from services.llm.client.shadow import (
    ShadowLog,
    ShadowPair,
    SpotCheck,
    summarize_disagreements,
)
from services.llm.schemas import AbstainReason


def test_with_deadline_returns_a_fast_answer():
    async def quick():
        return "narrative"

    assert asyncio.run(with_deadline(quick, 1.0)) == "narrative"


def test_with_deadline_gives_up_on_a_slow_answer():
    async def slow():
        await asyncio.sleep(5)
        return "too late"

    async def scenario():
        loop = asyncio.get_running_loop()
        started = loop.time()
        value = await with_deadline(slow, 0.05, default="unset")
        return value, loop.time() - started

    value, elapsed = asyncio.run(scenario())
    assert value == "unset"
    assert elapsed < 1.0  # the deadline is what bounds us, not the coroutine


def test_with_deadline_never_propagates_an_exception():
    async def boom():
        raise RuntimeError("model exploded")

    assert asyncio.run(with_deadline(boom, 1.0, default=None)) is None


def test_enricher_never_blocks_the_entry_decision():
    """The Gate D property, at the unit level: peek() is timing-neutral."""

    async def scenario():
        enricher = AdvisoryEnricher(deadline_s=0.05)

        async def slow():
            await asyncio.sleep(5)
            return "narrative"

        enricher.start("mint-1", slow)

        loop = asyncio.get_running_loop()
        started = loop.time()
        # This is what the entry path does. It must not await anything.
        seen = enricher.peek("mint-1")
        elapsed = loop.time() - started

        assert await enricher.collect("mint-1") is None
        await enricher.drain()
        return seen, elapsed

    seen, elapsed = asyncio.run(scenario())
    assert seen is None
    assert elapsed < 0.01


def test_enricher_attaches_a_result_that_lands_in_time():
    async def scenario():
        enricher = AdvisoryEnricher(deadline_s=1.0)

        async def quick():
            return {"narrative_cluster": "dog-meme"}

        enricher.start("mint-2", quick)
        result = await enricher.collect("mint-2")
        return result, enricher.peek("mint-2")

    result, peeked = asyncio.run(scenario())
    assert result == {"narrative_cluster": "dog-meme"} == peeked


def test_enricher_ignores_a_duplicate_start():
    async def scenario():
        enricher = AdvisoryEnricher(deadline_s=1.0)
        calls = {"n": 0}

        async def counted():
            calls["n"] += 1
            return calls["n"]

        enricher.start("k", counted)
        enricher.start("k", counted)
        await enricher.collect("k")
        await enricher.drain()
        return calls["n"]

    assert asyncio.run(scenario()) == 1


def test_enricher_peek_is_none_for_unknown_keys():
    assert AdvisoryEnricher().peek("never-started") is None


def pairs(agree: int, disagree: int) -> ShadowLog:
    log = ShadowLog()
    for i in range(agree):
        log.record(ShadowPair(item_id=f"a{i}", incumbent="bullish", local="bullish"))
    for i in range(disagree):
        log.record(ShadowPair(item_id=f"d{i}", incumbent="bullish", local="bearish"))
    return log


def test_shadow_log_measures_agreement():
    log = pairs(agree=93, disagree=7)
    assert log.agreement_rate == pytest.approx(0.93)
    assert len(log.disagreements) == 7


def test_shadow_gate_b_needs_volume_as_well_as_agreement():
    thin = pairs(agree=93, disagree=7)
    thin.reviewed_disagreements = 7
    assert not thin.gate().passed

    thick = pairs(agree=558, disagree=42)
    thick.reviewed_disagreements = 42
    assert thick.gate().passed, thick.gate().report()


def test_shadow_gate_b_blocks_when_a_malformed_response_would_have_voted():
    log = pairs(agree=600, disagree=0)
    log.record(
        ShadowPair(
            item_id="bad",
            incumbent="bullish",
            local=None,
            local_reason=AbstainReason.SCHEMA_FAIL.value,
            malformed_propagated=True,
        )
    )
    assert log.malformed_propagated == 1
    assert not log.gate().passed


def test_shadow_log_round_trips_through_disk(tmp_path: Path):
    log = pairs(agree=3, disagree=1)
    path = log.save(tmp_path / "shadow" / "sentiment.jsonl")
    reloaded = ShadowLog.load(path)

    assert len(reloaded.pairs) == 4
    assert reloaded.agreement_rate == log.agreement_rate


def test_mutual_abstention_counts_as_agreement():
    log = ShadowLog()
    log.record(ShadowPair(item_id="1", incumbent=None, local=None))
    assert log.agreement_rate == 1.0


def test_summarize_disagreements_groups_transitions():
    log = ShadowLog()
    log.record(ShadowPair(item_id="1", incumbent="bullish", local="bearish"))
    log.record(ShadowPair(item_id="2", incumbent="bullish", local="bearish"))
    log.record(ShadowPair(item_id="3", incumbent="bearish", local=None))
    log.record(ShadowPair(item_id="4", incumbent="neutral", local="neutral"))

    assert summarize_disagreements(log.pairs) == {
        "bullish -> bearish": 2,
        "bearish -> abstain": 1,
    }


def test_spot_check_samples_deterministically():
    check = SpotCheck(sample_rate=0.02)
    sampled = [i for i in range(200) if check.should_sample(i)]
    assert len(sampled) == 4
    assert sampled == [0, 50, 100, 150]


def test_spot_check_disabled_at_zero_rate():
    assert not SpotCheck(sample_rate=0.0).should_sample(0)


def test_spot_check_tracks_disagreement_rate():
    check = SpotCheck()
    check.record(ShadowPair(item_id="1", incumbent="bullish", local="bullish"))
    check.record(ShadowPair(item_id="2", incumbent="bullish", local="bearish"))
    assert check.disagreement_rate == 0.5
