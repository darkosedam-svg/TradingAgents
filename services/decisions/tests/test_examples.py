"""The three worked examples are documentation, so their conclusions are
asserted rather than merely printed. If a change to the guard flips one of
these verdicts, the prose in `examples.py` has silently become wrong."""

from pathlib import Path

from services.decisions.examples import (
    example_one_honest_idea,
    example_three_right_and_losing,
    example_two_grid_search,
    main,
)


def test_a_real_edge_over_one_year_is_still_not_proven(tmp_path: Path):
    verdict = example_one_honest_idea(tmp_path)

    assert not verdict.passed
    assert verdict.observed_sr > 0  # the edge is real...
    assert verdict.min_observations > verdict.n_observations  # ...and unproven
    assert "Keep logging" in verdict.reason


def test_the_grid_search_only_survives_if_the_search_is_hidden(tmp_path: Path):
    naive, honest = example_two_grid_search(tmp_path)

    # Identical returns, identical sample — the trial count is the only input
    # that differs, and it is the one that decides.
    assert naive.observed_sr == honest.observed_sr
    assert naive.n_observations == honest.n_observations
    assert naive.passed
    assert not honest.passed
    assert "the search itself is the problem" in honest.reason


def test_a_better_than_coin_flip_hit_rate_can_still_be_a_losing_system(
    tmp_path: Path,
):
    verdict = example_three_right_and_losing(tmp_path)

    assert not verdict.passed
    assert verdict.observed_sr < 0
    assert "not positive" in verdict.reason


def test_the_whole_demo_runs(capsys):
    main()
    out = capsys.readouterr().out
    assert out.count("EXAMPLE") == 3
    assert "PASS" in out and "FAIL" in out
