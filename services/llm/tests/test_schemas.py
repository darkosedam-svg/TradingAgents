import pytest
from pydantic import ValidationError

from services.llm.schemas import (
    AbstainReason,
    NewsTriage,
    Outcome,
    ParsedSignal,
    SentimentVote,
    TokenNarrativeFlags,
)


def test_sentiment_normalizes_and_dedupes_assets():
    vote = SentimentVote(
        sentiment="bullish",
        confidence=0.8,
        assets=["$sol", "SOL", " avax ", ""],
        rationale="listing",
    )
    assert vote.assets == ["SOL", "AVAX"]


def test_sentiment_rejects_prose_in_assets():
    with pytest.raises(ValidationError):
        SentimentVote(sentiment="neutral", confidence=0.5, assets=["the ethereum network"])


def test_sentiment_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        SentimentVote(sentiment="bullish", confidence=0.9, price_target=100)


def test_confidence_must_be_a_probability():
    with pytest.raises(ValidationError):
        SentimentVote(sentiment="bullish", confidence=1.4)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"confidence": 0.9, "insufficient_data": True}, AbstainReason.INSUFFICIENT_DATA),
        ({"confidence": 0.2}, AbstainReason.LOW_CONFIDENCE),
        ({"confidence": 0.9}, None),
    ],
)
def test_abstain_reason_precedence(kwargs, expected):
    vote = SentimentVote(sentiment="neutral", **kwargs)
    assert vote.abstain_reason(0.55) is expected


def test_signal_negative_must_be_clean():
    ok = ParsedSignal(valid=False, confidence=0.9)
    assert ok.asset is None

    with pytest.raises(ValidationError):
        ParsedSignal(valid=False, confidence=0.9, asset="BTC", side="long")


def test_signal_valid_requires_asset_and_side():
    with pytest.raises(ValidationError):
        ParsedSignal(valid=True, confidence=0.9, entry=100)


def test_signal_long_stop_must_sit_below_entry():
    with pytest.raises(ValidationError):
        ParsedSignal(
            valid=True, confidence=0.9, asset="BTC", side="long", entry=100, stop_loss=110
        )


def test_signal_short_targets_must_sit_below_entry():
    with pytest.raises(ValidationError):
        ParsedSignal(
            valid=True,
            confidence=0.9,
            asset="SOL",
            side="short",
            entry=148,
            take_profit=[160],
        )


def test_signal_accepts_consistent_geometry():
    signal = ParsedSignal(
        valid=True,
        confidence=0.9,
        asset="$btc",
        side="long",
        entry=64200,
        stop_loss=62800,
        take_profit=[66500, 68000],
        leverage=5,
    )
    assert signal.asset == "BTC"


def test_signal_rejects_non_positive_prices():
    with pytest.raises(ValidationError):
        ParsedSignal(valid=True, confidence=0.9, asset="BTC", side="long", entry=0)


def test_news_directional_call_needs_a_market():
    with pytest.raises(ValidationError):
        NewsTriage(
            confidence=0.9, relevance=0.8, direction="yes", escalate=True, market_ids=[]
        )

    NewsTriage(
        confidence=0.9, relevance=0.0, direction="unclear", escalate=False, market_ids=[]
    )


def test_news_dedupes_market_ids():
    triage = NewsTriage(
        confidence=0.7,
        relevance=0.5,
        direction="yes",
        escalate=True,
        market_ids=["m-1", " m-1 ", "m-2"],
    )
    assert triage.market_ids == ["m-1", "m-2"]


def test_token_meta_slugifies_and_dedupes():
    flags = TokenNarrativeFlags(
        confidence=0.6,
        narrative_cluster="  Dog Meme ",
        copycat_likelihood=0.9,
        scam_flags=["impersonation", "impersonation"],
    )
    assert flags.narrative_cluster == "dog-meme"
    assert flags.scam_flags == ["impersonation"]


def test_token_meta_rejects_invented_scam_flags():
    with pytest.raises(ValidationError):
        TokenNarrativeFlags(
            confidence=0.6,
            narrative_cluster="abstract",
            copycat_likelihood=0.1,
            scam_flags=["vibes_are_off"],
        )


def test_outcome_must_carry_exactly_one_of_value_or_reason():
    vote = SentimentVote(sentiment="bullish", confidence=0.9)

    assert Outcome.voted(vote).abstained is False
    assert Outcome.abstain(AbstainReason.SCHEMA_FAIL).abstained is True

    with pytest.raises(ValueError):
        Outcome(value=vote, reason=AbstainReason.LOW_CONFIDENCE)
    with pytest.raises(ValueError):
        Outcome(value=None, reason=None)


def test_outcome_totals_tokens():
    vote = SentimentVote(sentiment="bullish", confidence=0.9)
    outcome = Outcome.voted(vote, prompt_tokens=120, completion_tokens=40)
    assert outcome.total_tokens == 160
