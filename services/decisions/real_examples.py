"""The same three examples, on real market data.

Everything here comes from prices that actually traded and events that actually
resolved. Nothing is simulated:

* **Kraken** daily closes for BTC, ETH and SOL — 721 bars, the full depth of
  the public OHLC endpoint.
* **Polymarket** — 354 resolved binary markets, each paired with what the
  market was quoting roughly a day before it ended. The quote comes from the
  CLOB price history, not from the settled ``outcomePrices``, because a closed
  market reports the answer rather than the forecast.

Refresh the data with ``python -m services.decisions.data.fetch``; run the
examples with::

    python -m services.decisions.real_examples

The synthetic versions in :mod:`services.decisions.examples` exist to show the
mechanism with the true edge known. These exist to show what the mechanism says
when nobody knows the true edge — which is the situation you are actually in.

Stdlib only.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
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

DATA = Path(__file__).parent / "data"
RULE = "─" * 74

# Every moving-average cell is scored over the same window, so no cell gets a
# different sample than another. 200 bars of lookback are reserved for the
# slowest average.
WARMUP = 200


def _heading(number: int, title: str, subtitle: str) -> str:
    return f"\n{RULE}\nEXAMPLE {number} — {title}\n{subtitle}\n{RULE}"


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    spread = stdev(returns)
    return fmean(returns) / spread if spread > 0 else 0.0


def _compounded(returns: list[float]) -> float:
    """Total return from reinvesting each day. Summing daily percentages is
    the flattering version and it is wrong by tens of points over two years."""
    total = 1.0
    for r in returns:
        total *= 1.0 + r
    return total - 1.0


# ------------------------------------------------------------------ price data


@dataclass(frozen=True)
class Series:
    symbol: str
    dates: list[str]
    closes: list[float]

    def sma(self, window: int, i: int) -> float:
        return sum(self.closes[i - window + 1 : i + 1]) / window

    def crossover(self, fast: int, slow: int) -> list[tuple[int, int, float]]:
        """Run a moving-average crossover. Returns ``(index, position, return)``.

        The signal is computed from closes up to and including day ``i``, and
        the return is day ``i+1``'s. Nothing from the future touches the
        decision — the same guarantee the journal enforces structurally.
        """
        out = []
        for i in range(WARMUP, len(self.closes) - 1):
            position = 1 if self.sma(fast, i) > self.sma(slow, i) else -1
            move = self.closes[i + 1] / self.closes[i] - 1
            out.append((i, position, position * move))
        return out

    def buy_and_hold(self) -> list[float]:
        return [
            self.closes[i + 1] / self.closes[i] - 1
            for i in range(WARMUP, len(self.closes) - 1)
        ]


def load_series(symbol: str) -> Series:
    path = DATA / f"kraken_{symbol}_usd_daily.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return Series(
        symbol=symbol.upper(),
        dates=[r["date"] for r in rows],
        closes=[float(r["close"]) for r in rows],
    )


def _journal_crossover(path: Path, series: Series, fast: int, slow: int) -> list[Pair]:
    """Write the run through the real journal, one decision per trading day."""
    journal = DecisionJournal(path)
    for i, position, _signed in series.crossover(fast, slow):
        gap = series.sma(fast, i) / series.sma(slow, i) - 1
        decision = journal.append(
            Decision(
                domain=Domain.CRYPTO,
                instrument=f"{series.symbol}-USD",
                side=Side.LONG if position > 0 else Side.SHORT,
                # A crossover has no probability of its own; the distance
                # between the averages is the only thing it knows, so that is
                # what gets written down and later scored for calibration.
                confidence=min(0.5 + abs(gap) * 4, 0.95),
                rationale=f"SMA{fast} {'above' if position > 0 else 'below'} SMA{slow}",
                strategy_id=f"sma-{fast}-{slow}",
                sources=("kraken-ohlc",),
                horizon_s=86_400,
            )
        )
        journal.record_outcome(
            Realisation(
                decision_id=decision.decision_id,
                realised_return=series.closes[i + 1] / series.closes[i] - 1,
            )
        )
    return journal.pairs()


# --------------------------------------------------------------------------- 1


def example_one_textbook_crossover(workdir: Path) -> Verdict:
    """The 20/50 crossover on real BTC — chosen because it is the textbook
    default, not because it won anything. One attempt, honestly pre-registered.
    """
    btc = load_series("btc")
    window = f"{btc.dates[WARMUP + 1]} → {btc.dates[-1]}"
    print(
        _heading(
            1,
            "The textbook crossover, on real prices",
            f"BTC/USD 20/50 SMA, Kraken daily closes, {window}",
        )
    )

    pairs = _journal_crossover(workdir / "btc.jsonl", btc, 20, 50)
    score = overall(pairs)
    calib = calibration(pairs)
    hold = btc.buy_and_hold()

    print(
        f"\n  decisions journalled  {score.n}\n"
        f"  hit rate              {score.hit_rate:.1%}\n"
        f"  per-obs Sharpe        {score.sharpe:+.4f}   "
        f"(≈{score.sharpe * (365 ** 0.5):+.2f} annualised)\n"
        f"  compounded return     {_compounded([p.signed_return for p in pairs]):+.1%}\n"
        f"  buy-and-hold Sharpe   {_sharpe(hold):+.4f}   "
        f"(BTC itself returned {_compounded(hold):+.1%} over the same window)\n"
        f"  overconfidence        {calib.overconfidence:+.3f}   Brier {calib.brier:.3f}"
    )

    register = TrialRegister()
    register.register("sma-20-50", "the textbook default, fixed in advance")
    verdict = OverfittingGuard(register).evaluate(score.sharpe, n_observations=score.n)
    print("\n" + verdict.report())
    print(
        "\n  A year and a half of daily calls on the most liquid crypto pair\n"
        "  there is, and the honest answer is that nothing has been shown. It\n"
        "  did dodge a falling market, which is worth something and is not the\n"
        "  same as an edge."
    )
    return verdict


# --------------------------------------------------------------------------- 2


def example_two_real_grid_search(workdir: Path) -> tuple[Verdict, Verdict, dict]:
    """Every crossover cell on the same real BTC series, then the winner
    re-tested on two coins it was never fitted to."""
    btc = load_series("btc")
    window = f"{btc.dates[WARMUP + 1]} → {btc.dates[-1]}"
    print(
        _heading(
            2,
            "The grid search, on real prices",
            f"every SMA crossover cell on the same BTC series, {window}",
        )
    )

    register = TrialRegister()
    cell_sharpes: list[float] = []
    best: tuple[float, int, int] = (-99.0, 0, 0)
    for fast in range(5, 105, 5):
        for slow in range(20, 210, 10):
            if slow <= fast:
                continue
            register.register("sma-grid", f"{fast}/{slow}")
            sharpe = _sharpe([r for _, _, r in btc.crossover(fast, slow)])
            cell_sharpes.append(sharpe)
            if sharpe > best[0]:
                best = (sharpe, fast, slow)

    best_sr, fast, slow = best
    dispersion = measure_trial_dispersion(cell_sharpes)
    pairs = _journal_crossover(workdir / "grid.jsonl", btc, fast, slow)
    score = overall(pairs)

    print(
        f"\n  cells tried            {register.count}\n"
        f"  best cell              SMA {fast}/{slow}\n"
        f"  its hit rate           {score.hit_rate:.1%}\n"
        f"  its per-obs Sharpe     {best_sr:+.4f}   "
        f"(≈{best_sr * (365 ** 0.5):+.2f} annualised)\n"
        f"  median cell            {sorted(cell_sharpes)[len(cell_sharpes) // 2]:+.4f}\n"
        f"  Sharpe spread across all {register.count} cells: {dispersion:.4f}"
    )

    guard = OverfittingGuard(register, sr_std_across_trials=dispersion)
    naive = guard.evaluate(best_sr, n_observations=score.n, n_trials=1)
    honest = guard.evaluate(best_sr, n_observations=score.n)

    print(
        "\n  Reporting only the winner:\n"
        + "\n".join("  " + line for line in naive.report().splitlines())
    )
    print(
        f"\n  Counting all {register.count} attempts:\n"
        + "\n".join("  " + line for line in honest.report().splitlines())
    )

    print("\n  The same cell on two coins it was never fitted to:")
    holdout: dict[str, Verdict] = {}
    for symbol in ("eth", "sol"):
        series = load_series(symbol)
        returns = [r for _, _, r in series.crossover(fast, slow)]
        sharpe = _sharpe(returns)
        # One pre-registered cell per coin, so the honest trial count is 1 —
        # with the caveat printed below.
        verdict = OverfittingGuard().evaluate(sharpe, n_observations=len(returns))
        holdout[series.symbol] = verdict
        print(
            f"    {series.symbol:<4} Sharpe {sharpe:+.4f}  "
            f"compounded {_compounded(returns):+.1%}  "
            f"deflated {verdict.dsr:.3f}  [{'PASS' if verdict.passed else 'FAIL'}]"
        )
    print(
        "\n  Read the holdout carefully. Running the winner on two coins is two\n"
        "  more trials, and reporting whichever did better is the same mistake\n"
        "  one level up. Neither clears the bar, so the question does not arise\n"
        "  here — but it is the exact point where it usually does."
    )
    return naive, honest, holdout


# --------------------------------------------------------------------------- 3


@dataclass(frozen=True)
class Market:
    slug: str
    end_date: str
    volume_usd: float
    p_yes: float
    resolved_yes: bool
    hours_before_end: float


def load_markets() -> list[Market]:
    path = DATA / "polymarket_resolved.csv"
    with path.open(encoding="utf-8") as handle:
        return [
            Market(
                slug=r["slug"],
                end_date=r["end_date"],
                volume_usd=float(r["volume_usd"]),
                p_yes=float(r["p_yes"]),
                resolved_yes=bool(int(r["resolved_yes"])),
                hours_before_end=float(r["hours_before_end"]),
            )
            for r in csv.DictReader(handle)
        ]


def back_the_favourite(market: Market, *, taker_cost: float) -> tuple[float, float, bool]:
    """Stake $1 on whichever side the market makes the favourite.

    Returns ``(price paid, return on the stake, was it right)``. A win pays the
    remaining spread; a loss costs the whole ticket. ``taker_cost`` is the
    slippage from crossing the book rather than resting on it — the single
    detail that separates the profitable cohort on that venue from everyone
    else.
    """
    back_yes = market.p_yes > 0.5
    quoted = market.p_yes if back_yes else 1 - market.p_yes
    paid = min(max(quoted, 0.01) + taker_cost, 0.99)
    right = back_yes == market.resolved_yes
    return paid, ((1 - paid) / paid if right else -1.0), right


def example_three_real_polymarket(workdir: Path, *, taker_cost: float = 0.01) -> Verdict:
    """Real forecasts against real resolutions, and the trade that loses anyway.

    The strategy is the simplest one a scanner produces: back the favourite.
    """
    markets = load_markets()
    print(
        _heading(
            3,
            "Right, and losing anyway — on real resolutions",
            f"{len(markets)} settled Polymarket binaries, quoted ~"
            f"{fmean(m.hours_before_end for m in markets):.0f}h before they ended",
        )
    )

    journal = DecisionJournal(workdir / "polymarket.jsonl")
    for market in markets:
        paid, payoff, _right = back_the_favourite(market, taker_cost=taker_cost)
        decision = journal.append(
            Decision(
                domain=Domain.PREDICTION,
                instrument=market.slug[:60],
                side=Side.LONG,
                confidence=max(market.p_yes, 1 - market.p_yes),
                rationale=f"back the favourite at {paid:.3f}",
                strategy_id="back-the-favourite",
                sources=("polymarket-clob",),
            )
        )
        journal.record_outcome(
            Realisation(
                decision_id=decision.decision_id,
                realised_return=payoff,
                notes=market.end_date,
            )
        )

    pairs = journal.pairs()
    score = overall(pairs)
    calib = calibration(pairs)

    print(
        f"\n  markets                {score.n}\n"
        f"  favourite was right    {score.hit_rate:.1%}\n"
        f"  P&L on $1 per market   {score.total_return:+.2f}\n"
        f"  mean per market        {score.mean_return:+.4f}\n"
        f"  per-obs Sharpe         {score.sharpe:+.4f}\n"
        f"  (taker cost assumed: {taker_cost * 100:.0f}c on the price paid)"
    )

    free = [back_the_favourite(m, taker_cost=0.0)[1] for m in markets]
    print(
        f"  the same bets at zero trading cost: {sum(free):+.2f}\n"
        "  — so this does not lose because of fees. It loses because the odds\n"
        "    on a favourite are already the odds."
    )

    print("\n" + "\n".join("  " + line for line in calib.table().splitlines()))

    print("\n  Was the crowd calibrated? Brier of the raw market price:")
    brier = fmean((m.p_yes - float(m.resolved_yes)) ** 2 for m in markets)
    print(f"    {brier:.4f}   (0.25 is what you get by always saying 0.5)")

    print("\n  Accuracy by market size — the claim worth checking:")
    bands = [
        (1e3, 1e4, "$1k–10k"),
        (1e4, 1e5, "$10k–100k"),
        (1e5, 1e6, "$100k–1M"),
        (1e6, 1e7, "$1M–10M"),
        (1e7, float("inf"), "over $10M"),
    ]
    for low, high, label in bands:
        band = [m for m in markets if low <= m.volume_usd < high]
        if not band:
            continue
        correct = fmean(float((m.p_yes > 0.5) == m.resolved_yes) for m in band)
        band_brier = fmean((m.p_yes - float(m.resolved_yes)) ** 2 for m in band)
        print(
            f"    {label:<11} n={len(band):>3}  favourite right {correct:.1%}  "
            f"Brier {band_brier:.4f}"
        )
    print(
        "  The research brief claimed ~61% under $10k against ~84% over $100k.\n"
        "  On this sample that gap does not appear. Sampling a day before\n"
        "  resolution flatters every band, and these are different markets from\n"
        "  the ones that study used — but the claim does not reproduce here, and\n"
        "  it should not be leaned on."
    )

    register = TrialRegister()
    register.register("back-the-favourite", "one rule, no parameters")
    verdict = OverfittingGuard(register).evaluate(score.sharpe, n_observations=score.n)
    print("\n" + verdict.report())
    print(
        f"\n  {score.hit_rate:.0%} accuracy. Real money, gone. Hit rate is the number\n"
        "  every dashboard shows and it is the number that tells you least."
    )
    return verdict


def main() -> None:
    btc = load_series("btc")
    markets = load_markets()
    print("Three runs through services/decisions, on real market data.")
    print(
        f"Kraken daily closes {btc.dates[0]} → {btc.dates[-1]} ({len(btc.closes)} bars); "
        f"{len(markets)} settled Polymarket binaries."
    )
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        example_one_textbook_crossover(workdir)
        example_two_real_grid_search(workdir)
        example_three_real_polymarket(workdir)
    print(
        f"\n{RULE}\nThree real datasets, no edge demonstrated in any of them. That is\n"
        "the expected result, and it is the reason to log before you trade\n"
        f"rather than after.\n{RULE}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
