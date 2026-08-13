"""The paper loop's correctness is mostly about *timing* — which bar a decision
is allowed to see, and when it may be scored. Synthetic series here rather than
the committed CSVs, so each case can put the calendar exactly where it needs it.
"""

from pathlib import Path

import pytest

from services.decisions.journal import DecisionJournal
from services.decisions.paper import (
    AlwaysLong,
    Coinflip,
    Crossover,
    already_ran,
    backfill,
    default_strategies,
    emit,
    resolve,
    run_once,
    status,
)
from services.decisions.prices import Series
from services.decisions.record import Domain, Side
from services.decisions.trials import TrialRegister


def ramp(n: int = 60, *, start_day: int = 1, step: float = 1.0) -> Series:
    """A calendar-day series with steadily rising closes."""
    from datetime import date, timedelta

    first = date(2026, 1, start_day)
    return Series(
        symbol="TEST",
        dates=[(first + timedelta(days=i)).isoformat() for i in range(n)],
        closes=[100.0 + i * step for i in range(n)],
    )


def weekdays_only(n: int = 40) -> Series:
    """An equity-style series: no weekend bars."""
    from datetime import date, timedelta

    dates, closes, day, i = [], [], date(2026, 1, 5), 0  # a Monday
    while len(dates) < n:
        if day.weekday() < 5:
            dates.append(day.isoformat())
            closes.append(100.0 + i)
            i += 1
        day += timedelta(days=1)
    return Series("EQ", dates, closes, Domain.EQUITY, 252)


def fixture(tmp_path: Path, series: Series):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    register = TrialRegister(tmp_path / "t.jsonl")
    return journal, register, {"TEST-USD": series}


# --------------------------------------------------------------------- timing


def test_a_decision_is_never_written_against_an_unfinished_bar(tmp_path: Path):
    """Today's daily candle is a live price, not a close. Trading it and then
    scoring against tomorrow's real close compares two different things."""
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)

    emit(journal, universe, [AlwaysLong()], register, today=series.dates[-1])

    written = journal.decisions()
    assert len(written) == 1
    assert written[0].meta["as_of"] == series.dates[-2], "used the partial bar"


def test_nothing_is_emitted_before_the_first_bar_closes(tmp_path: Path):
    series = ramp(5)
    journal, register, universe = fixture(tmp_path, series)

    emitted, abstained = emit(
        journal, universe, [AlwaysLong()], register, today=series.dates[0]
    )

    assert (emitted, abstained) == (0, 1)
    assert journal.decisions() == []


def test_a_decision_cannot_be_resolved_in_the_run_that_created_it(tmp_path: Path):
    """The whole ordering guarantee in one test."""
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)

    report = run_once(
        journal, universe, [AlwaysLong()], register, today=series.dates[-1]
    )

    assert report.emitted == 1
    assert report.resolved == 0
    assert report.still_pending == 1


def test_the_outcome_arrives_on_the_next_run(tmp_path: Path):
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)

    run_once(journal, universe, [AlwaysLong()], register, today=series.dates[-2])
    second = run_once(
        journal, universe, [AlwaysLong()], register, today=series.dates[-1]
    )

    assert second.resolved == 1
    pair = journal.pairs()[0]
    assert pair.realised_return == pytest.approx(
        series.closes[-2] / series.closes[-3] - 1
    )


def test_a_friday_call_is_judged_on_monday(tmp_path: Path):
    """A one-day horizon over a weekend has no next-day bar. Skipping to the
    next session is right; leaving it pending forever is not."""
    series = weekdays_only()
    journal = DecisionJournal(tmp_path / "d.jsonl")
    register = TrialRegister()
    universe = {"EQ-USD": series}

    friday = next(d for d in series.dates if d == "2026-01-09")
    saturday = "2026-01-10"
    emit(journal, universe, [AlwaysLong()], register, today=saturday)
    assert journal.decisions()[0].meta["as_of"] == friday

    resolved, _ = resolve(journal, universe)
    assert resolved == 1
    assert "2026-01-12" in journal.outcomes()[0].notes  # the Monday


def test_a_decision_with_no_exit_bar_yet_stays_pending(tmp_path: Path):
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)
    emit(journal, universe, [AlwaysLong()], register, today=series.dates[-1])

    # The exit bar is the final one, which exists — trim it away and the
    # decision must wait rather than be scored against something else.
    trimmed = Series("TEST", series.dates[:-1], series.closes[:-1])
    resolved, stuck = resolve(journal, {"TEST-USD": trimmed})

    assert (resolved, stuck) == (0, [])
    assert len(journal.pending()) == 1


def test_an_instrument_that_vanished_is_reported_not_guessed(tmp_path: Path):
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)
    emit(journal, universe, [AlwaysLong()], register, today=series.dates[-1])

    resolved, stuck = resolve(journal, {})

    assert resolved == 0
    assert stuck == ["TEST-USD"]


# ---------------------------------------------------------------- idempotence


def test_running_twice_in_a_day_does_not_double_count(tmp_path: Path):
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)
    today = series.dates[-1]

    run_once(journal, universe, [AlwaysLong()], register, today=today)
    second = run_once(journal, universe, [AlwaysLong()], register, today=today)

    assert second.emitted == 0
    assert len(journal.decisions()) == 1
    assert already_ran(journal, series.dates[-2])


def test_a_strategy_run_daily_is_still_one_trial(tmp_path: Path):
    """A trial is a distinct attempt, not a distinct run. Counting runs would
    inflate the correction until nothing could ever pass."""
    series = ramp(40)
    journal, register, universe = fixture(tmp_path, series)

    for day in series.dates[-5:]:
        run_once(journal, universe, [AlwaysLong()], register, today=day)

    assert register.count == 1
    assert len(journal.decisions()) > 1


def test_the_trial_count_survives_a_restart(tmp_path: Path):
    register = TrialRegister(tmp_path / "t.jsonl")
    register.register_once("a", "first idea")
    register.register_once("b", "second idea")

    assert TrialRegister(tmp_path / "t.jsonl").count == 2


# ----------------------------------------------------------------- baselines


def test_the_coin_flip_cannot_be_rerolled(tmp_path: Path):
    """Re-rolling a baseline until it looks bad is the same cheat as re-rolling
    a strategy until it looks good."""
    series = ramp(30)
    flip = Coinflip()
    calls = [flip.signal(series, 10).side for _ in range(5)]

    assert len(set(calls)) == 1


def test_the_coin_flip_actually_varies_across_days():
    series = ramp(60)
    flip = Coinflip()
    sides = {flip.signal(series, i).side for i in range(50)}
    assert sides == {Side.LONG, Side.SHORT}


def test_the_defaults_ship_with_both_baselines():
    ids = {s.strategy_id for s in default_strategies()}
    assert "baseline-hold" in ids and "baseline-coinflip" in ids


def test_a_crossover_abstains_without_enough_history():
    series = ramp(30)
    assert Crossover(20, 50).signal(series, 10) is None
    assert Crossover(5, 10).signal(series, 20) is not None


def test_a_rising_market_makes_the_crossover_long():
    series = ramp(60)
    assert Crossover(5, 20).signal(series, 50).side is Side.LONG


# ------------------------------------------------------------------ backfill


def test_backfill_marks_its_decisions_as_hindsight(tmp_path: Path):
    series = ramp(60)
    journal, register, universe = fixture(tmp_path, series)

    backfill(journal, universe, [AlwaysLong()], register, days=10)

    assert journal.decisions()
    assert all(d.meta["replayed"] for d in journal.decisions())


def test_backfill_refuses_to_contaminate_a_live_journal(tmp_path: Path):
    """Months of hindsight sitting in the same file as a forward record would
    be indistinguishable to the guard. It must not be possible by accident."""
    series = ramp(60)
    journal, register, universe = fixture(tmp_path, series)
    run_once(journal, universe, [AlwaysLong()], register, today=series.dates[-1])

    with pytest.raises(ValueError, match="live decision"):
        backfill(journal, universe, [AlwaysLong()], register, days=10)


def test_backfill_scores_what_it_replays(tmp_path: Path):
    series = ramp(60)
    journal, register, universe = fixture(tmp_path, series)

    report = backfill(journal, universe, default_strategies(), register, days=20)

    assert report.emitted > 0
    assert report.resolved > 0
    assert len(journal.pairs()) == report.resolved


# -------------------------------------------------------------------- status


def test_status_says_so_before_anything_has_resolved(tmp_path: Path):
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)
    run_once(journal, universe, [AlwaysLong()], register, today=series.dates[-1])

    text = status(journal, register)
    assert "No scored decisions yet" in text
    assert "1 pending" in text


def test_status_will_not_print_a_deflated_sharpe_off_three_points(tmp_path: Path):
    series = ramp(60)
    journal, register, universe = fixture(tmp_path, series)
    backfill(journal, universe, [AlwaysLong()], register, days=4)

    text = status(journal, register)
    assert "n/a" in text, "a DSR from a handful of points invites misreading"
    assert "nothing clears the bar" in text


def test_status_names_the_baselines_and_the_correlation_caveat(tmp_path: Path):
    series = ramp(120)
    journal, register, universe = fixture(tmp_path, series)
    backfill(journal, universe, default_strategies(), register, days=60)

    text = status(journal, register)
    assert "(baseline)" in text
    assert "Against the baselines:" in text
    assert "correlated" in text
    assert "Calibration:" in text


def test_an_outcome_is_never_taken_from_an_unclosed_bar(tmp_path: Path):
    """The mirror of the emission rule. Scoring yesterday's call against
    today's live price books a partial day as a full day, biased in whichever
    direction the market happens to be moving when cron fires."""
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)
    yesterday, today = series.dates[-2], series.dates[-1]

    emit(journal, universe, [AlwaysLong()], register, today=yesterday)
    assert journal.decisions()[0].meta["as_of"] == series.dates[-3]

    # The bar that would settle it is today's, which has not closed.
    resolved, _ = resolve(journal, universe, today=yesterday)
    assert resolved == 0

    # Once it closes, the same call resolves.
    resolved, _ = resolve(journal, universe, today=today)
    assert resolved == 1


def test_run_once_passes_its_clock_to_resolution(tmp_path: Path):
    series = ramp(30)
    journal, register, universe = fixture(tmp_path, series)

    run_once(journal, universe, [AlwaysLong()], register, today=series.dates[-2])
    same_day = run_once(
        journal, universe, [AlwaysLong()], register, today=series.dates[-2]
    )

    assert same_day.resolved == 0, "resolved against a bar that had not closed"
    assert same_day.still_pending == 1


# ----------------------------------------------------------------------- cli


def test_the_cli_refuses_a_future_today(capsys):
    """--today moves the line between a closed bar and a live one. A future
    date would write decisions against prices nobody could have traded, and
    the record has no way to tell that apart afterwards."""
    from services.decisions.__main__ import main

    with pytest.raises(SystemExit) as exit_info:
        main(["run", "--today", "2099-01-01", "--offline"])

    assert exit_info.value.code != 0
    assert "future" in capsys.readouterr().err


def test_the_cli_refuses_a_malformed_today(capsys):
    from services.decisions.__main__ import main

    with pytest.raises(SystemExit):
        main(["run", "--today", "13-08-2026", "--offline"])
    assert "YYYY-MM-DD" in capsys.readouterr().err


def test_the_cli_drives_the_loop_at_a_chosen_date(tmp_path: Path, monkeypatch):
    """Rehearsing the scheduled job, and catching up a run the scheduler
    dropped, are the same operation."""
    from services.decisions import __main__ as cli

    series = ramp(40)
    monkeypatch.setattr(cli, "_universe", lambda symbols, home: {"TEST-USD": series})

    assert cli.main(["run", "--home", str(tmp_path), "--offline",
                     "--today", series.dates[-3]]) == 0
    first = DecisionJournal(tmp_path / "decisions.jsonl").decisions()
    assert [d.meta["as_of"] for d in first] == [series.dates[-4]] * len(first)

    assert cli.main(["run", "--home", str(tmp_path), "--offline",
                     "--today", series.dates[-2]]) == 0
    journal = DecisionJournal(tmp_path / "decisions.jsonl")
    assert len(journal.pairs()) == len(first), "the earlier day should now resolve"
    assert len(journal.pending()) == len(first), "and a new day should be pending"
