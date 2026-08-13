from pathlib import Path

import pytest

from services.decisions.journal import DecisionJournal, Pair
from services.decisions.record import Decision, Domain, Realisation, Side
from services.decisions.scoring import (
    by_domain,
    by_source,
    by_strategy,
    calibration,
    overall,
    summary,
)


def pair(
    *,
    correct: bool = True,
    confidence: float = 0.7,
    strategy: str = "momentum",
    sources: tuple[str, ...] = ("news",),
    domain: Domain = Domain.CRYPTO,
    magnitude: float = 0.02,
) -> Pair:
    decision = Decision(
        domain=domain,
        instrument="SOL-USD",
        side=Side.LONG,
        confidence=confidence,
        rationale="test",
        strategy_id=strategy,
        sources=sources,
    )
    realised = magnitude if correct else -magnitude
    return Pair(
        decision,
        Realisation(decision_id=decision.decision_id, realised_return=realised),
    )


def test_overall_counts_hits_and_returns():
    pairs = [pair(correct=True), pair(correct=True), pair(correct=False)]
    score = overall(pairs)

    assert score.n == 3
    assert score.hit_rate == pytest.approx(2 / 3)
    assert score.total_return == pytest.approx(0.02)
    assert score.sharpe > 0


def test_empty_slice_does_not_divide_by_zero():
    score = overall([])
    assert score.hit_rate == 0.0
    assert score.mean_return == 0.0
    assert score.sharpe == 0.0


def test_sharpe_needs_two_points():
    assert overall([pair()]).sharpe == 0.0


def test_zero_variance_returns_no_sharpe():
    """All-identical returns would divide by zero, not signal infinite skill."""
    assert overall([pair(magnitude=0.02) for _ in range(5)]).sharpe == 0.0


def test_strategies_and_domains_partition_the_journal():
    pairs = [
        pair(strategy="momentum", domain=Domain.CRYPTO),
        pair(strategy="momentum", domain=Domain.CRYPTO, correct=False),
        pair(strategy="meanrev", domain=Domain.EQUITY),
    ]

    strategies = by_strategy(pairs)
    assert strategies["momentum"].n == 2
    assert strategies["momentum"].hit_rate == 0.5
    assert strategies["meanrev"].hit_rate == 1.0

    assert sum(s.n for s in by_domain(pairs).values()) == len(pairs)


def test_sources_overlap_rather_than_partition():
    """A decision citing three sources counts toward all three — the totals
    should exceed the decision count, and that is correct."""
    pairs = [
        pair(sources=("news", "funding", "flow")),
        pair(sources=("news",), correct=False),
    ]
    scores = by_source(pairs)

    assert sum(s.n for s in scores.values()) == 4 > len(pairs)
    assert scores["news"].n == 2
    assert scores["news"].hit_rate == 0.5
    assert scores["flow"].hit_rate == 1.0


def test_calibration_finds_a_wildly_overconfident_system():
    """Says 0.9, right half the time — exactly what the design guards against."""
    pairs = [pair(confidence=0.9, correct=i % 2 == 0) for i in range(20)]
    result = calibration(pairs)

    assert result.overconfidence == pytest.approx(0.4, abs=0.01)
    assert result.brier > 0.25  # worse than always guessing 0.5


def test_calibration_rewards_an_honest_system():
    honest = [pair(confidence=0.9, correct=i < 9) for i in range(10)]
    result = calibration(honest)

    assert result.overconfidence == pytest.approx(0.0, abs=0.01)
    assert result.brier < 0.1


def test_underconfidence_shows_as_a_negative_gap():
    pairs = [pair(confidence=0.3, correct=True) for _ in range(10)]
    assert calibration(pairs).overconfidence < 0


def test_confidence_of_one_lands_in_the_top_bin_not_out_of_range():
    result = calibration([pair(confidence=1.0)], bins=5)
    assert result.bins[-1].n == 1


def test_brier_of_a_perfect_forecaster_is_zero():
    perfect = [pair(confidence=1.0, correct=True) for _ in range(5)]
    assert calibration(perfect).brier == pytest.approx(0.0)


def test_bins_must_be_positive():
    with pytest.raises(ValueError):
        calibration([pair()], bins=0)


def test_summary_is_honest_about_what_it_has_not_done():
    text = summary([pair(correct=i % 3 != 0, strategy=f"s{i%2}") for i in range(12)])

    assert "scored decisions" in text
    assert "By strategy:" in text
    assert "By source:" in text
    assert "Brier" in text
    # Must point at the correction rather than let a raw Sharpe stand alone.
    assert "OverfittingGuard" in text


def test_summary_says_so_when_there_is_nothing_yet():
    assert summary([]) == "No scored decisions yet."


def test_end_to_end_from_journal_to_score(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    for i in range(10):
        made = journal.append(
            Decision(
                domain=Domain.PREDICTION,
                instrument=f"mkt-{i}",
                side=Side.LONG,
                confidence=0.8,
                rationale="test",
                strategy_id="triage",
                sources=("polymarket",),
            )
        )
        journal.record_outcome(
            Realisation(
                decision_id=made.decision_id,
                realised_return=0.01 if i < 6 else -0.01,
            )
        )

    pairs = journal.pairs()
    assert overall(pairs).hit_rate == pytest.approx(0.6)
    assert by_source(pairs)["polymarket"].n == 10
    # Stated 0.8, delivered 0.6 — the gap the loop must not be allowed to ignore.
    assert calibration(pairs).overconfidence == pytest.approx(0.2, abs=0.01)
