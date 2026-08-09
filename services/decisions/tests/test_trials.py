import math

import pytest

from services.decisions.trials import (
    OverfittingGuard,
    TrialRegister,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
)


def test_expected_max_sharpe_rises_with_trials_at_zero_true_skill():
    """The finding this whole module exists for: search alone manufactures
    impressive-looking results."""
    curve = [expected_max_sharpe(n) for n in (1, 2, 5, 10, 50, 100, 1000)]

    assert curve[0] == 0.0  # one draw from a zero-mean distribution
    assert curve == sorted(curve)
    assert curve[-1] > 3.0  # a "Sharpe 3" strategy, from 1000 coin flips


def test_one_trial_returns_the_mean():
    assert expected_max_sharpe(1, mean_sr=0.4) == 0.4


def test_expected_max_scales_with_dispersion():
    assert expected_max_sharpe(20, std_sr=2.0) == pytest.approx(
        2 * expected_max_sharpe(20, std_sr=1.0)
    )


def test_expected_max_shifts_with_mean():
    assert expected_max_sharpe(20, mean_sr=1.0) == pytest.approx(
        expected_max_sharpe(20) + 1.0
    )


def test_trial_count_must_be_positive():
    with pytest.raises(ValueError):
        expected_max_sharpe(0)


def test_deflated_sharpe_falls_as_the_search_widens():
    """Identical performance, more attempts to find it, less believable."""
    scores = [
        deflated_sharpe_ratio(1.5, n_trials=n, n_observations=250)
        for n in (1, 10, 100, 1000)
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.9
    assert scores[-1] < scores[0]


def test_more_evidence_sharpens_the_verdict_in_whichever_direction_it_points():
    """Evidence does not flatter — it confirms. A result above the no-skill
    benchmark becomes more credible with more data; one below it becomes more
    clearly not skill."""
    benchmark = expected_max_sharpe(10)

    above = 2.5
    assert above > benchmark
    ahead = [
        deflated_sharpe_ratio(above, n_trials=10, n_observations=n) for n in (30, 250)
    ]
    assert ahead[1] > ahead[0]

    below = 1.2
    assert below < benchmark
    behind = [
        deflated_sharpe_ratio(below, n_trials=10, n_observations=n) for n in (30, 250)
    ]
    assert behind[1] < behind[0]


def test_a_result_below_the_no_skill_bar_is_not_rescued_by_data():
    """1.2 looks respectable until you remember it took ten attempts to find."""
    assert deflated_sharpe_ratio(1.2, n_trials=10, n_observations=5000) < 0.01


def test_deflated_sharpe_stays_a_probability():
    for sr in (-3.0, 0.0, 0.5, 10.0):
        value = deflated_sharpe_ratio(sr, n_trials=7, n_observations=100)
        assert 0.0 <= value <= 1.0


def test_negative_skew_and_fat_tails_reduce_confidence():
    """A given Sharpe earned with crash risk is worth less."""
    clean = deflated_sharpe_ratio(1.5, n_trials=5, n_observations=500)
    ugly = deflated_sharpe_ratio(
        1.5, n_trials=5, n_observations=500, skew=-1.5, kurtosis=9.0
    )
    assert ugly < clean


def test_deflated_sharpe_needs_two_observations():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(1.0, n_trials=1, n_observations=1)


def test_track_record_length_is_infinite_when_not_ahead():
    assert min_track_record_length(0.0) == math.inf
    assert min_track_record_length(-0.5) == math.inf


def test_track_record_length_shrinks_as_edge_grows():
    weak = min_track_record_length(0.1)
    strong = min_track_record_length(1.0)
    assert weak > strong > 1.0


def test_register_counts_and_will_not_forget():
    register = TrialRegister()
    for i in range(3):
        register.register("momentum", f"variant {i}")
    register.register("meanrev", "first")

    assert register.count == 4
    assert register.count_for("momentum") == 3
    assert register.count_for("meanrev") == 1
    assert not hasattr(register, "reset")


def test_registered_trials_carry_identity():
    register = TrialRegister()
    trial = register.register("momentum", "20/50 crossover")
    assert trial.strategy_id == "momentum"
    assert trial.description == "20/50 crossover"
    assert len(trial.trial_id) == 32
    assert trial.ts.endswith("+00:00")


def test_guard_blocks_on_too_little_evidence():
    guard = OverfittingGuard(min_observations=30)
    verdict = guard.evaluate(3.0, n_observations=10, n_trials=1)

    assert not verdict.passed
    assert "too few observations" in verdict.reason
    assert "FAIL" in verdict.report()


def test_guard_names_search_as_the_problem_when_that_is_the_problem():
    """A wide search that never cleared the no-skill bar. More data won't help,
    and the message must say so rather than suggest waiting."""
    guard = OverfittingGuard()
    verdict = guard.evaluate(1.4, n_observations=250, n_trials=500)

    assert not verdict.passed
    assert verdict.benchmark_sr > verdict.observed_sr
    assert "the search itself is the problem" in verdict.reason
    assert "Keep logging" not in verdict.reason


def test_guard_names_sample_size_when_that_is_the_problem():
    """One honest attempt, real but small edge — the fix is patience, not a
    different strategy, and the message must not blame the search."""
    guard = OverfittingGuard()
    verdict = guard.evaluate(0.10, n_observations=40, n_trials=1)

    assert not verdict.passed
    assert "not enough evidence" in verdict.reason
    assert "search itself" not in verdict.reason
    assert verdict.min_observations > 40


def test_guard_passes_a_strong_result_from_a_narrow_search():
    guard = OverfittingGuard()
    verdict = guard.evaluate(1.5, n_observations=1000, n_trials=2)

    assert verdict.passed, verdict.report()
    assert "clears the bar" in verdict.reason


def test_guard_reads_the_trial_count_off_the_register_by_default():
    register = TrialRegister()
    for i in range(400):
        register.register("grid", f"cell {i}")
    guard = OverfittingGuard(register)

    verdict = guard.evaluate(1.4, n_observations=250)

    assert verdict.n_trials == 400
    assert not verdict.passed


def test_same_result_flips_verdict_purely_on_trial_count():
    """The whole argument in one assertion: identical numbers, opposite call."""
    guard = OverfittingGuard()
    honest = guard.evaluate(1.6, n_observations=800, n_trials=1)
    searched = guard.evaluate(1.6, n_observations=800, n_trials=1000)

    assert honest.passed
    assert not searched.passed
    assert honest.observed_sr == searched.observed_sr
