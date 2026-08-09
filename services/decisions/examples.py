"""Three worked examples, run end to end.

Not tests and not toys — three situations you will actually be in, played
through the journal, the scorer and the guard so you can see what each one says
before it is your own money on the line:

1. **One honest idea.** A single strategy, never re-tuned, with a small real
   edge. The kind of result that looks like nothing and might be everything.
2. **The grid search.** Four hundred parameter cells over data with *zero* edge
   by construction. The best cell looks excellent. It is noise.
3. **Right, and losing anyway.** A prediction-market triage that calls 62% of
   markets correctly and still bleeds, because it crosses the spread.

Deterministic — every example seeds its own generator, so the numbers below
reproduce exactly. Run them::

    python -m services.decisions.examples

Stdlib only, like the rest of the package.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from statistics import fmean, stdev

from .journal import DecisionJournal, Pair
from .record import Decision, Domain, Realisation, Side
from .scoring import calibration, overall
from .trials import (
    OverfittingGuard,
    TrialRegister,
    Verdict,
    measure_trial_dispersion,
)

RULE = "─" * 72


def _heading(number: int, title: str, subtitle: str) -> str:
    return f"\n{RULE}\nEXAMPLE {number} — {title}\n{subtitle}\n{RULE}"


def _draw_returns(
    rng: random.Random,
    n: int,
    *,
    p_win: float,
    win: float,
    loss: float,
    spread: float = 0.35,
) -> list[float]:
    """Signed returns for ``n`` calls with a true win probability of ``p_win``.

    Magnitudes vary around ``win``/``loss`` so the sample is not two-valued;
    ``spread`` is the relative noise on each magnitude.
    """
    out: list[float] = []
    for _ in range(n):
        won = rng.random() < p_win
        base = win if won else loss
        magnitude = abs(rng.gauss(base, base * spread))
        out.append(magnitude if won else -magnitude)
    return out


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    spread = stdev(returns)
    return fmean(returns) / spread if spread > 0 else 0.0


def _journal_run(
    path: Path,
    signed_returns: list[float],
    *,
    domain: Domain,
    instrument: str,
    strategy_id: str,
    sources: tuple[str, ...],
    confidence_of,
    rng: random.Random,
) -> list[Pair]:
    """Write a full run through the real journal — decisions first, outcomes
    later, exactly as they would arrive in production."""
    journal = DecisionJournal(path)
    for signed in signed_returns:
        side = Side.LONG if rng.random() < 0.7 else Side.SHORT
        decision = journal.append(
            Decision(
                domain=domain,
                instrument=instrument,
                side=side,
                confidence=confidence_of(signed > 0),
                rationale="see strategy definition",
                strategy_id=strategy_id,
                sources=sources,
                horizon_s=86_400,
            )
        )
        # The journal stores the market's move; `Pair.signed_return` negates it
        # for a short, so this is what makes the two agree.
        realised = signed if side is Side.LONG else -signed
        journal.record_outcome(
            Realisation(decision_id=decision.decision_id, realised_return=realised)
        )
    return journal.pairs()


# --------------------------------------------------------------------------- 1


def example_one_honest_idea(workdir: Path) -> Verdict:
    """A real but small edge, found once and never re-tuned.

    Daily Sharpe near 0.06 is roughly Sharpe 1.0 annualised — a genuinely good
    strategy. A year of it is still not enough to know that.
    """
    rng = random.Random(20260809)
    print(_heading(1, "One honest idea", "SOL-USD momentum, 260 daily calls, one trial"))

    returns = _draw_returns(rng, 260, p_win=0.53, win=0.020, loss=0.020)
    pairs = _journal_run(
        workdir / "honest.jsonl",
        returns,
        domain=Domain.CRYPTO,
        instrument="SOL-USD",
        strategy_id="momentum-20-50",
        sources=("funding", "spot-volume"),
        # Honest: says 0.55, is right about 53% of the time.
        confidence_of=lambda correct: 0.55,
        rng=rng,
    )

    score = overall(pairs)
    calib = calibration(pairs)
    print(
        f"\n{score.n} journalled decisions, {score.n} scored\n"
        f"  hit rate        {score.hit_rate:.1%}\n"
        f"  mean return     {score.mean_return:+.4f} per call\n"
        f"  per-obs Sharpe  {score.sharpe:.3f}   "
        f"(≈{score.sharpe * (252 ** 0.5):.2f} annualised)\n"
        f"  overconfidence  {calib.overconfidence:+.3f}   Brier {calib.brier:.3f}"
    )

    register = TrialRegister()
    register.register("momentum-20-50", "the only thing we tried")
    verdict = OverfittingGuard(register).evaluate(
        score.sharpe, n_observations=score.n
    )
    print("\n" + verdict.report())
    print(
        "\n  Read it as: nothing is wrong with the strategy. One year of daily\n"
        "  calls simply cannot separate a Sharpe-1 edge from luck. The honest\n"
        "  answer is 'not yet', and the honest action is to keep logging."
    )
    return verdict


# --------------------------------------------------------------------------- 2


def example_two_grid_search(workdir: Path) -> tuple[Verdict, Verdict]:
    """400 parameter cells over data with no edge whatsoever.

    Nothing here can work — ``p_win`` is exactly 0.5 in every cell. The best
    cell will nonetheless look like a discovery, which is the entire finding.
    """
    rng = random.Random(11111)
    print(
        _heading(
            2,
            "The grid search",
            "400 (fast, slow) moving-average cells, 250 days each, zero true edge",
        )
    )

    register = TrialRegister()
    cell_sharpes: list[float] = []
    best = (-99.0, "", [])
    for fast in range(5, 105, 5):
        for slow in range(50, 1050, 50):
            label = f"ma-{fast}-{slow}"
            register.register("grid", label)
            returns = _draw_returns(rng, 250, p_win=0.50, win=0.018, loss=0.018)
            sharpe = _sharpe(returns)
            cell_sharpes.append(sharpe)
            if sharpe > best[0]:
                best = (sharpe, label, returns)

    best_sr, best_label, best_returns = best
    dispersion = measure_trial_dispersion(cell_sharpes)

    pairs = _journal_run(
        workdir / "grid.jsonl",
        best_returns,
        domain=Domain.EQUITY,
        instrument="SPY",
        strategy_id=best_label,
        sources=("price",),
        confidence_of=lambda correct: 0.65,
        rng=rng,
    )
    score = overall(pairs)

    print(
        f"\n  cells tried            {register.count}\n"
        f"  best cell              {best_label}\n"
        f"  its hit rate           {score.hit_rate:.1%}\n"
        f"  its per-obs Sharpe     {best_sr:.3f}   "
        f"(≈{best_sr * (252 ** 0.5):.2f} annualised)\n"
        f"  Sharpe spread across all 400 cells: {dispersion:.4f}"
    )

    guard = OverfittingGuard(register, sr_std_across_trials=dispersion)

    naive = guard.evaluate(best_sr, n_observations=score.n, n_trials=1)
    print(
        "\n  What you would conclude if you reported only the winner:\n"
        + "\n".join("  " + line for line in naive.report().splitlines())
    )

    honest = guard.evaluate(best_sr, n_observations=score.n)
    print(
        "\n  What you conclude having counted the search:\n"
        + "\n".join("  " + line for line in honest.report().splitlines())
    )
    print(
        "\n  Same returns. Same 250 days. Same backtest. The only thing that\n"
        "  changed is whether the other 399 attempts were disclosed — and the\n"
        f"  true edge here is exactly zero, so {'the second' if not honest.passed else 'the first'} verdict is the correct one."
    )
    return naive, honest


# --------------------------------------------------------------------------- 3


def example_three_right_and_losing(workdir: Path) -> Verdict:
    """A Polymarket triage that is right more often than not, and still loses.

    Wins collect the remaining spread; losses pay the full ticket. The research
    finding this reproduces: winners on that venue are the ones posting limit
    orders, and this system is a taker.
    """
    rng = random.Random(4242)
    print(
        _heading(
            3,
            "Right, and losing anyway",
            "Polymarket news triage, 180 markets, right about 3 calls in 5",
        )
    )

    # +5% when right, −10% when wrong: at any hit rate under two-thirds this
    # loses money, and the loss is a property of the payoff, not of the sample.
    returns = _draw_returns(rng, 180, p_win=0.62, win=0.050, loss=0.100, spread=0.20)
    pairs = _journal_run(
        workdir / "polymarket.jsonl",
        returns,
        domain=Domain.PREDICTION,
        instrument="various",
        strategy_id="news-triage",
        sources=("newswire", "orderbook"),
        # The failure mode the calibration table exists to catch: a system that
        # has learned to sound certain.
        confidence_of=lambda correct: 0.85,
        rng=rng,
    )

    score = overall(pairs)
    calib = calibration(pairs)
    breakeven = 0.100 / (0.050 + 0.100)
    print(
        f"\n  decisions scored   {score.n}\n"
        f"  hit rate           {score.hit_rate:.1%}   ← comfortably better than a coin flip\n"
        f"  break-even needs   {breakeven:.1%}   ← and short of this\n"
        f"  total return       {score.total_return:+.3f}\n"
        f"  mean per call      {score.mean_return:+.4f}\n"
        f"  per-obs Sharpe     {score.sharpe:+.3f}"
    )
    print("\n" + "\n".join("  " + line for line in calib.table().splitlines()))

    register = TrialRegister()
    register.register("news-triage", "single configuration")
    verdict = OverfittingGuard(register).evaluate(score.sharpe, n_observations=score.n)
    print("\n" + verdict.report())
    print(
        "\n  Two separate problems, and hit rate hides both. The payoff is\n"
        "  asymmetric — a win takes the remaining spread, a loss pays the whole\n"
        f"  ticket — so being right {score.hit_rate:.0%} of the time is still short of the\n"
        f"  {breakeven:.0%} this bet needs. And the system states 0.85 while delivering\n"
        f"  {score.hit_rate:.2f}: {calib.overconfidence:+.2f} of pure overconfidence, which is exactly the\n"
        "  number that would have sized the positions up."
    )
    return verdict


def main() -> None:
    print("Three runs through services/decisions — journal, score, guard.")
    print("All figures generated from seeded random data; no market data involved.")
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        example_one_honest_idea(workdir)
        example_two_grid_search(workdir)
        example_three_right_and_losing(workdir)
    print(
        f"\n{RULE}\nOne verdict of three would have been a green light if the trial\n"
        "count had been thrown away — and that one is the fabricated edge.\n"
        f"{RULE}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
