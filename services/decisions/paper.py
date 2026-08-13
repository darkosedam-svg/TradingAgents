"""Paper trading: the loop that turns a package into a track record.

Step 3 of the [build order](../../docs/unified-agent-findings.md) is *pick one
market and prove an edge exists, paper-traded, scored against reality for
months.* That is not a thing more code can do — it is a thing time does. What
code can do is make the daily act cost nothing, so it actually happens.

One run does two things, in this order:

1. **Resolve** every decision whose horizon has now elapsed, by looking up the
   close on the day it was judged against.
2. **Emit** today's decisions and journal them.

Resolve first, deliberately. A decision written today must not be able to see
an outcome recorded in the same pass, and doing the two in this order makes
that structural rather than careful.

Every strategy is registered as a trial exactly once, via
:meth:`TrialRegister.register_once` — running the same strategy again tomorrow
is not a new roll of the dice. And two deliberately stupid strategies run
alongside the real ones, because the research finding that survived 3–0 was
*keep a dumb baseline in every comparison*. A strategy that cannot beat a coin
flip and a buy-and-hold has not earned a place in anything.

Nothing here executes. Decisions land in the journal; what reads the journal is
somebody else's problem, and today that is a person.

Stdlib only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

from .journal import DecisionJournal
from .prices import Series, compounded, sharpe
from .record import Decision, Realisation, Side
from .scoring import by_strategy, calibration, overall
from .trials import (
    OverfittingGuard,
    TrialRegister,
    Verdict,
    no_skill_dispersion,
)

HORIZON_DAYS = 1


@dataclass(frozen=True)
class Signal:
    """What a strategy thinks, before anything turns it into a position."""

    side: Side
    confidence: float
    rationale: str


class Strategy(Protocol):
    strategy_id: str
    description: str

    def signal(self, series: Series, i: int) -> Optional[Signal]:
        """The call for day ``i``, or ``None`` to abstain.

        Abstention is a first-class answer. A strategy without enough history,
        or without a view, returns ``None`` and no decision is written — which
        is different from, and much better than, writing a coin flip.
        """


@dataclass(frozen=True)
class Crossover:
    """The moving-average crossover, run forward instead of backwards."""

    fast: int
    slow: int

    @property
    def strategy_id(self) -> str:
        return f"sma-{self.fast}-{self.slow}"

    @property
    def description(self) -> str:
        return f"SMA {self.fast}/{self.slow} crossover, daily close, 1-day horizon"

    def signal(self, series: Series, i: int) -> Optional[Signal]:
        if i < self.slow - 1:
            return None
        position = series.position_at(self.fast, self.slow, i)
        gap = series.sma(self.fast, i) / series.sma(self.slow, i) - 1
        return Signal(
            side=Side.LONG if position > 0 else Side.SHORT,
            # The crossover has no probability of its own. The distance between
            # the averages is all it knows, so that is what gets written down —
            # and calibration will judge whether it meant anything.
            confidence=min(0.5 + abs(gap) * 4, 0.95),
            rationale=(
                f"SMA{self.fast} {'above' if position > 0 else 'below'} "
                f"SMA{self.slow} by {gap:+.2%}"
            ),
        )


@dataclass(frozen=True)
class AlwaysLong:
    """Buy and hold. The baseline that most strategies quietly lose to."""

    strategy_id: str = "baseline-hold"
    description: str = "always long, no view at all"

    def signal(self, series: Series, i: int) -> Optional[Signal]:
        return Signal(Side.LONG, 0.5, "baseline: hold")


@dataclass
class Coinflip:
    """A coin flip. If something cannot beat this, it is not a strategy.

    Seeded by ``(strategy_id, date)`` so a rerun of the same day produces the
    same call — otherwise the baseline could be re-rolled until it looked bad,
    which is the same cheat as re-rolling the strategy until it looks good.
    """

    strategy_id: str = "baseline-coinflip"
    description: str = "fair coin, seeded by date"

    def signal(self, series: Series, i: int) -> Optional[Signal]:
        rng = random.Random(f"{self.strategy_id}:{series.symbol}:{series.dates[i]}")
        return Signal(
            Side.LONG if rng.random() < 0.5 else Side.SHORT, 0.5, "baseline: coin flip"
        )


def default_strategies() -> list[Strategy]:
    """One idea and the two baselines it has to beat."""
    return [Crossover(20, 50), AlwaysLong(), Coinflip()]


# ------------------------------------------------------------------ the run


@dataclass
class RunReport:
    as_of: str
    resolved: int = 0
    emitted: int = 0
    abstained: int = 0
    still_pending: int = 0
    unresolvable: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"paper run {self.as_of}",
            f"  resolved   {self.resolved}",
            f"  emitted    {self.emitted}"
            + (f"  ({self.abstained} abstained)" if self.abstained else ""),
            f"  pending    {self.still_pending}",
        ]
        if self.unresolvable:
            lines.append(
                f"  {len(self.unresolvable)} decision(s) cannot be resolved: "
                + ", ".join(sorted(set(self.unresolvable))[:5])
            )
        return "\n".join(lines)


def _entry(decision: Decision) -> tuple[Optional[str], Optional[float]]:
    meta = decision.meta or {}
    return meta.get("as_of"), meta.get("close")


def resolve(
    journal: DecisionJournal,
    universe: dict[str, Series],
    *,
    horizon_days: int = HORIZON_DAYS,
    today: Optional[str] = None,
) -> tuple[int, list[str]]:
    """Record the outcome of every decision whose horizon has elapsed.

    The exit price is looked up by date, not by offset, so a gap in the series
    means the decision stays pending rather than being scored against the wrong
    day. Pending is an honest state; silently scoring against whatever bar
    happened to be next is not.

    An outcome is only taken from a bar that has *closed*. Today's candle is a
    live price, and scoring yesterday's call against it books a partial day's
    move as a full day's result — a bias that runs in whichever direction the
    market happens to be going at the moment cron fires.
    """
    cutoff = today or datetime.now(timezone.utc).date().isoformat()
    resolved = 0
    stuck: list[str] = []
    for decision in journal.pending():
        series = universe.get(decision.instrument)
        as_of, entry_close = _entry(decision)
        if series is None or as_of is None or not entry_close:
            stuck.append(decision.instrument)
            continue
        due = (date.fromisoformat(as_of) + timedelta(days=horizon_days)).isoformat()
        judged = series.first_close_on_or_after(due)
        if judged is None or judged[0] >= cutoff:
            continue  # not yet, or not yet closed; try again next run
        judged_on, exit_close = judged
        journal.record_outcome(
            Realisation(
                decision_id=decision.decision_id,
                realised_return=exit_close / entry_close - 1,
                notes=f"{as_of} → {judged_on}",
            )
        )
        resolved += 1
    return resolved, stuck


def emit(
    journal: DecisionJournal,
    universe: dict[str, Series],
    strategies: Sequence[Strategy],
    register: TrialRegister,
    *,
    horizon_days: int = HORIZON_DAYS,
    today: Optional[str] = None,
    at: Optional[dict[str, int]] = None,
    replayed: bool = False,
) -> tuple[int, int]:
    """Write one decision per (strategy, instrument), on the last *closed* bar.

    ``at`` overrides which bar to use per instrument — the backfill needs it,
    live running does not.
    """
    emitted = abstained = 0
    for strategy in strategies:
        register.register_once(strategy.strategy_id, strategy.description)
        for instrument, series in sorted(universe.items()):
            i = (
                at[instrument]
                if at is not None
                else series.last_complete_index(today)
            )
            if i is None:
                abstained += 1
                continue
            signal = strategy.signal(series, i)
            if signal is None:
                abstained += 1
                continue
            journal.append(
                Decision(
                    domain=series.domain,
                    instrument=instrument,
                    side=signal.side,
                    confidence=signal.confidence,
                    rationale=signal.rationale,
                    strategy_id=strategy.strategy_id,
                    sources=("daily-close",),
                    horizon_s=horizon_days * 86_400,
                    # The entry price is known now, which is exactly why it may
                    # be recorded now. The exit price is not, which is why it
                    # arrives later in its own entry.
                    meta={
                        "as_of": series.dates[i],
                        "close": series.closes[i],
                        **({"replayed": True} if replayed else {}),
                    },
                )
            )
            emitted += 1
    return emitted, abstained


def already_ran(journal: DecisionJournal, as_of: str) -> bool:
    """Has a run already covered this date? Emitting twice would double-count
    every decision into the score, which flatters or damns a strategy purely
    on how often cron fired."""
    return any((d.meta or {}).get("as_of") == as_of for d in journal.decisions())


def run_once(
    journal: DecisionJournal,
    universe: dict[str, Series],
    strategies: Sequence[Strategy],
    register: TrialRegister,
    *,
    horizon_days: int = HORIZON_DAYS,
    today: Optional[str] = None,
    force: bool = False,
) -> RunReport:
    """Resolve what is due, then emit today. Safe to run more than once a day."""
    closed = [
        series.dates[i]
        for series in universe.values()
        if (i := series.last_complete_index(today)) is not None
    ]
    report = RunReport(as_of=max(closed) if closed else "no closed bar yet")

    report.resolved, report.unresolvable = resolve(
        journal, universe, horizon_days=horizon_days, today=today
    )
    if closed and (force or not already_ran(journal, report.as_of)):
        report.emitted, report.abstained = emit(
            journal,
            universe,
            strategies,
            register,
            horizon_days=horizon_days,
            today=today,
        )
    report.still_pending = len(journal.pending())
    return report


def backfill(
    journal: DecisionJournal,
    universe: dict[str, Series],
    strategies: Sequence[Strategy],
    register: TrialRegister,
    *,
    days: int = 90,
    horizon_days: int = HORIZON_DAYS,
) -> RunReport:
    """Replay history through the same loop, to see it work before waiting.

    **This is a backtest wearing the runner's clothes.** Every decision it
    writes is tagged ``replayed``, and it refuses to touch a journal that
    already holds live entries — mixing the two would let months of hindsight
    masquerade as a forward track record, and the guard cannot tell them apart.

    Useful for checking the wiring and for a first read on a strategy. Not
    useful as evidence, for exactly the reasons in the examples.
    """
    live = [d for d in journal.decisions() if not (d.meta or {}).get("replayed")]
    if live:
        raise ValueError(
            f"{journal.path} already holds {len(live)} live decision(s). "
            "Backfill writes hindsight; keep it in its own journal so a replay "
            "can never be mistaken for a track record."
        )

    report = RunReport(as_of=f"replay of the last {days} days")
    for series in universe.values():
        end = series.last_complete_index()
        if end is None:
            continue
        for i in range(max(0, end - days + 1), end + 1):
            emitted, abstained = emit(
                journal,
                {k: v for k, v in universe.items() if v is series},
                strategies,
                register,
                horizon_days=horizon_days,
                at={k: i for k, v in universe.items() if v is series},
                replayed=True,
            )
            report.emitted += emitted
            report.abstained += abstained

    report.resolved, report.unresolvable = resolve(
        journal, universe, horizon_days=horizon_days
    )
    report.still_pending = len(journal.pending())
    return report


# ---------------------------------------------------------------- the report


def status(
    journal: DecisionJournal,
    register: TrialRegister,
    *,
    baseline_ids: Iterable[str] = ("baseline-hold", "baseline-coinflip"),
    min_dsr: float = 0.95,
) -> str:
    """Where the track record stands, and what the guard makes of it."""
    pairs = journal.pairs()
    if not pairs:
        pending = len(journal.pending())
        return (
            "No scored decisions yet."
            + (f" {pending} pending — outcomes arrive a horizon later." if pending else "")
        )

    baselines = set(baseline_ids)
    scores = by_strategy(pairs)
    guard = OverfittingGuard(register, min_dsr=min_dsr)
    # These are per-observation Sharpes, so the paper's annualised default of
    # 1.0 would reject everything. There is no grid to measure here, so use the
    # no-skill sampling spread instead.
    dispersion = no_skill_dispersion(max(2, min(s.n for s in scores.values())))

    total = overall(pairs)
    calib = calibration(pairs)
    lines = [
        f"{total.n} scored decisions, {len(journal.pending())} pending",
        f"registered trials: {register.count}",
        "",
        f"{'strategy':<22} {'n':>5} {'hit':>7} {'sharpe':>8} {'total':>9} {'DSR':>7}",
    ]
    verdicts: dict[str, Verdict] = {}
    for name, score in scores.items():
        verdict = guard.evaluate(
            score.sharpe,
            n_observations=score.n,
            sr_std_across_trials=dispersion,
        )
        verdicts[name] = verdict
        mark = "  (baseline)" if name in baselines else ""
        # A deflated Sharpe off a handful of observations is a number, not
        # information. Printing it invites someone to read 0.7 as encouraging.
        dsr = (
            f"{verdict.dsr:>7.3f}"
            if score.n >= guard.min_observations
            else f"{'n/a':>7}"
        )
        lines.append(
            f"{name:<22} {score.n:>5} {score.hit_rate:>6.1%} {score.sharpe:>+8.4f} "
            f"{compounded(score.returns):>+8.1%} {dsr}{mark}"
        )

    real = {n: s for n, s in scores.items() if n not in baselines}
    beaten = [
        name
        for name, score in real.items()
        if all(
            score.sharpe > scores[b].sharpe for b in baselines if b in scores
        )
    ]
    lines += ["", "Against the baselines:"]
    if not real:
        lines.append("  nothing but baselines so far.")
    elif beaten:
        lines.append(
            "  beats every baseline on Sharpe: " + ", ".join(sorted(beaten))
        )
        lines.append(
            "  — necessary, not sufficient. The DSR column is the one that decides."
        )
    else:
        lines.append(
            "  none of them beat both baselines. Nothing here is worth tuning yet;"
        )
        lines.append("  a better parameter is a new trial, and trials are what cost you.")

    passing = [n for n, v in verdicts.items() if v.passed and n not in baselines]
    lines += ["", "Guard:"]
    if passing:
        lines.append("  CLEARS THE BAR: " + ", ".join(sorted(passing)))
        lines.append("  Read the caveats in the README before acting on it.")
    else:
        soonest = min(
            (v for n, v in verdicts.items() if n not in baselines),
            key=lambda v: v.min_observations,
            default=None,
        )
        lines.append("  nothing clears the bar yet.")
        if soonest is not None and soonest.min_observations != float("inf"):
            lines.append(
                f"  best case needs about {soonest.min_observations:,.0f} observations; "
                f"you have {soonest.n_observations}."
            )

    lines += [
        "",
        f"Calibration: overconfidence {calib.overconfidence:+.3f}, Brier {calib.brier:.3f}",
        "  Watch that first number over weeks. Growing means the system is",
        "  learning to sound certain rather than to be right.",
        "",
        "Caveat: trading several instruments on the same days pools correlated",
        "observations, so n overstates how much independent evidence you have.",
        "The DSR column is optimistic by an amount this does not measure.",
    ]
    return "\n".join(lines)


__all__ = [
    "AlwaysLong",
    "Coinflip",
    "Crossover",
    "HORIZON_DAYS",
    "RunReport",
    "Signal",
    "Strategy",
    "already_ran",
    "default_strategies",
    "emit",
    "resolve",
    "run_once",
    "status",
]
