"""Checks on the real-data examples.

Two kinds of assertion here, and the difference matters. Some are structural —
they hold whatever the market did. Others pin the conclusions the prose in
`real_examples.py` draws; those are data-dependent on purpose, so that
re-running the fetcher and getting a different world makes the tests fail
rather than leaving the commentary quietly wrong.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from services.decisions.real_examples import (
    WARMUP,
    Market,
    back_the_favourite,
    example_one_textbook_crossover,
    example_three_real_polymarket,
    example_two_real_grid_search,
    load_markets,
    load_series,
    main,
)


# ------------------------------------------------------------------ structural


def test_price_series_load_and_line_up():
    series = [load_series(s) for s in ("btc", "eth", "sol")]
    for s in series:
        assert len(s.closes) > WARMUP + 50
        assert all(c > 0 for c in s.closes)
        assert s.dates == sorted(s.dates)
    assert len({tuple(s.dates) for s in series}) == 1, "coins must share a calendar"


def test_a_crossover_decision_cannot_see_past_its_own_day():
    """The look-ahead check that actually bites: truncate the future and the
    decision must be identical. A signal that changes is reading ahead."""
    btc = load_series("btc")
    full = btc.crossover(20, 50)

    cutoff = WARMUP + 40
    truncated = replace(btc, dates=btc.dates[: cutoff + 2], closes=btc.closes[: cutoff + 2])

    assert truncated.crossover(20, 50) == [row for row in full if row[0] <= cutoff]


def test_market_rows_are_forecasts_not_resolutions():
    """If `p_yes` were ever 0 or 1 the dataset would be the answer key rather
    than the forecast, and every calibration number would be meaningless."""
    markets = load_markets()
    assert len(markets) > 100
    for m in markets:
        assert 0.0 < m.p_yes < 1.0
        assert m.hours_before_end > 0
        assert m.volume_usd >= 1_000


def test_backing_a_favourite_pays_the_spread_and_loses_the_ticket():
    win = Market("a", "2026-01-01", 5_000, p_yes=0.80, resolved_yes=True, hours_before_end=24)
    loss = replace(win, resolved_yes=False)

    paid, payoff, right = back_the_favourite(win, taker_cost=0.01)
    assert right and paid == pytest.approx(0.81)
    assert payoff == pytest.approx(0.19 / 0.81)

    _, payoff, right = back_the_favourite(loss, taker_cost=0.01)
    assert not right and payoff == -1.0


def test_the_underdog_side_is_backed_when_it_is_the_favourite():
    market = Market("a", "2026-01-01", 5_000, p_yes=0.2, resolved_yes=False, hours_before_end=24)
    paid, payoff, right = back_the_favourite(market, taker_cost=0.0)
    assert right, "p_yes 0.2 means NO is the favourite, and NO was correct"
    assert paid == pytest.approx(0.8)


# ------------------------------------------------- conclusions the prose makes


def test_the_textbook_crossover_shows_nothing_over_this_window(tmp_path: Path):
    verdict = example_one_textbook_crossover(tmp_path)

    assert verdict.n_trials == 1
    assert not verdict.passed
    assert verdict.n_observations == len(load_series("btc").closes) - WARMUP - 1


def test_the_grid_winner_only_survives_with_the_search_hidden(tmp_path: Path):
    naive, honest, holdout = example_two_real_grid_search(tmp_path)

    assert naive.observed_sr == honest.observed_sr
    assert naive.n_observations == honest.n_observations
    assert honest.n_trials > 100
    # Counting the search can only ever lower the verdict, never raise it.
    assert honest.dsr < naive.dsr
    assert honest.benchmark_sr > naive.benchmark_sr
    assert not honest.passed
    assert set(holdout) == {"ETH", "SOL"}
    assert not any(v.passed for v in holdout.values()), (
        "the prose says neither holdout clears the bar; rewrite it if that changed"
    )


def test_a_high_hit_rate_on_real_markets_is_still_a_losing_system(tmp_path: Path):
    verdict = example_three_real_polymarket(tmp_path)
    markets = load_markets()
    hit_rate = sum(
        back_the_favourite(m, taker_cost=0.01)[2] for m in markets
    ) / len(markets)

    assert hit_rate > 0.7, "the headline is that accuracy is high"
    assert verdict.observed_sr < 0, "...and the money still goes the other way"
    assert not verdict.passed


def test_losing_is_not_a_fee_problem(tmp_path: Path):
    """The prose claims this loses even with no trading cost at all. If a data
    refresh made that false, the claim has to change."""
    markets = load_markets()
    free = [back_the_favourite(m, taker_cost=0.0)[1] for m in markets]
    assert sum(free) < 0


def test_the_whole_real_demo_runs(capsys):
    main()
    out = capsys.readouterr().out
    assert out.count("EXAMPLE") == 3
    assert "Kraken daily closes" in out
