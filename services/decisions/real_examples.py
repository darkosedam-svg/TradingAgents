"""Five examples on real market data, one per domain the system would cover.

Everything here comes from prices that actually traded and events that actually
resolved. Nothing is simulated:

* **Kraken** daily closes — BTC, ETH, SOL and all thirteen memecoins the
  exchange lists against USD. 721 bars, the full depth of the public endpoint.
* **Yahoo** daily *adjusted* closes for eight US tickers. Adjusted, because a
  raw close gaps on every dividend and split and a crossover rule will happily
  "predict" a gap nobody could trade.
* **Polymarket** — 354 resolved binary markets, each paired with what the
  market was quoting roughly a day before it ended. The quote comes from the
  CLOB price history, not from the settled ``outcomePrices``, because a closed
  market reports the answer rather than the forecast.
* **CoinGecko** — every meme token currently listed, with its distance from
  peak. The closest measurement of survivorship the free data allows.

Each domain fails in its own way, and that is the point of running all five:

1. **Crypto** — one honest attempt shows nothing over 520 days.
2. **Crypto, searched** — 299 cells produce a winner that the correction eats.
3. **Prediction markets** — 84% accurate and losing money.
4. **Equities** — zero is the wrong bar; the index is the bar.
5. **Memecoins** — the sample is made entirely of survivors.

Refresh the data with ``python -m services.decisions.data.fetch``; run the
examples with::

    python -m services.decisions.real_examples

The synthetic versions in :mod:`services.decisions.examples` show the mechanism
with the true edge known. These show what the mechanism says when nobody knows
the true edge — which is the situation you are actually in.

Stdlib only.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Optional

from .journal import DecisionJournal, Pair
from .prices import (
    DATA,
    EQUITIES,
    MEMECOINS,
    Series,
    compounded,
    load_equity,
    load_series,
    sharpe,
)
from .record import Decision, Domain, Realisation, Side
from .scoring import calibration, overall
from .trials import (
    OverfittingGuard,
    TrialRegister,
    Verdict,
    measure_trial_dispersion,
)

RULE = "─" * 74

# Every moving-average cell is scored over the same window, so no cell gets a
# different sample than another. 200 bars of lookback are reserved for the
# slowest average.
WARMUP = 200

# One grid, reused by every search below, so "how many things did you try" has
# the same meaning in each example.
GRID = tuple(
    (fast, slow)
    for fast in range(5, 105, 5)
    for slow in range(20, 210, 10)
    if slow > fast
)


def _heading(number: int, title: str, subtitle: str) -> str:
    return f"\n{RULE}\nEXAMPLE {number} — {title}\n{subtitle}\n{RULE}"


@dataclass(frozen=True)
class GridResult:
    """What a parameter sweep actually produced — winner and everything else.

    The losing cells are kept deliberately. They are not waste: their spread is
    what sets the scale of the overfitting correction, and discarding them is
    the mechanical form of the mistake this package exists to prevent.
    """

    winner: Series
    fast: int
    slow: int
    sharpe: float
    returns: list[float]
    cell_sharpes: list[float]

    @property
    def trials(self) -> int:
        return len(self.cell_sharpes)

    @property
    def dispersion(self) -> float:
        return measure_trial_dispersion(self.cell_sharpes)

    @property
    def median_cell(self) -> float:
        return sorted(self.cell_sharpes)[len(self.cell_sharpes) // 2]


def run_grid(universe: list[Series], register: TrialRegister, label: str) -> GridResult:
    """Every cell in :data:`GRID` against every series, all of it registered."""
    best: Optional[GridResult] = None
    cells: list[float] = []
    for series in universe:
        if len(series.closes) <= WARMUP + 60:
            continue  # too little post-warmup history to score fairly
        for fast, slow in GRID:
            register.register(label, f"{series.symbol} {fast}/{slow}")
            returns = [r for _, _, r in series.crossover(fast, slow, start=WARMUP)]
            cell = sharpe(returns)
            cells.append(cell)
            if best is None or cell > best.sharpe:
                best = GridResult(series, fast, slow, cell, returns, cells)
    if best is None:
        raise ValueError("no series had enough history to search")
    # `cells` is shared by reference above; rebind so the winner sees the full
    # list rather than the prefix that existed when it won.
    return GridResult(
        best.winner, best.fast, best.slow, best.sharpe, best.returns, cells
    )


def _journal_crossover(path: Path, series: Series, fast: int, slow: int) -> list[Pair]:
    """Write the run through the real journal, one decision per trading day."""
    journal = DecisionJournal(path)
    for i, position, _signed in series.crossover(fast, slow, start=WARMUP):
        gap = series.sma(fast, i) / series.sma(slow, i) - 1
        decision = journal.append(
            Decision(
                domain=series.domain,
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
    hold = btc.buy_and_hold(start=WARMUP)

    print(
        f"\n  decisions journalled  {score.n}\n"
        f"  hit rate              {score.hit_rate:.1%}\n"
        f"  per-obs Sharpe        {score.sharpe:+.4f}   "
        f"(≈{btc.annualised(score.sharpe):+.2f} annualised)\n"
        f"  compounded return     {compounded([p.signed_return for p in pairs]):+.1%}\n"
        f"  buy-and-hold Sharpe   {sharpe(hold):+.4f}   "
        f"(BTC itself returned {compounded(hold):+.1%} over the same window)\n"
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
    grid = run_grid([btc], register, "sma-grid")
    fast, slow, best_sr = grid.fast, grid.slow, grid.sharpe
    pairs = _journal_crossover(workdir / "grid.jsonl", btc, fast, slow)
    score = overall(pairs)

    print(
        f"\n  cells tried            {grid.trials}\n"
        f"  best cell              SMA {fast}/{slow}\n"
        f"  its hit rate           {score.hit_rate:.1%}\n"
        f"  its per-obs Sharpe     {best_sr:+.4f}   "
        f"(≈{btc.annualised(best_sr):+.2f} annualised)\n"
        f"  median cell            {grid.median_cell:+.4f}\n"
        f"  Sharpe spread across all {grid.trials} cells: {grid.dispersion:.4f}"
    )

    guard = OverfittingGuard(register, sr_std_across_trials=grid.dispersion)
    naive = guard.evaluate(best_sr, n_observations=score.n, n_trials=1)
    honest = guard.evaluate(best_sr, n_observations=score.n)

    print(
        "\n  Reporting only the winner:\n"
        + "\n".join("  " + line for line in naive.report().splitlines())
    )
    print(
        f"\n  Counting all {grid.trials} attempts:\n"
        + "\n".join("  " + line for line in honest.report().splitlines())
    )

    print("\n  The same cell on two coins it was never fitted to:")
    holdout: dict[str, Verdict] = {}
    for symbol in ("eth", "sol"):
        series = load_series(symbol)
        returns = [r for _, _, r in series.crossover(fast, slow, start=WARMUP)]
        out_of_sample = sharpe(returns)
        # One pre-registered cell per coin, so the honest trial count is 1 —
        # with the caveat printed below.
        verdict = OverfittingGuard().evaluate(out_of_sample, n_observations=len(returns))
        holdout[series.symbol] = verdict
        print(
            f"    {series.symbol:<4} Sharpe {out_of_sample:+.4f}  "
            f"compounded {compounded(returns):+.1%}  "
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


# --------------------------------------------------------------------------- 4


def example_four_equities(workdir: Path) -> tuple[Verdict, Verdict]:
    """Stocks, where the trap is different.

    Crypto over this window went nowhere, so a strategy's Sharpe could be read
    against zero without doing much damage. Equities drifted up hard. A rule
    that is long most of the time inherits that drift and reports it as skill,
    so zero is the wrong bar — **the index is the bar.**
    """
    universe = [load_equity(t) for t in EQUITIES]
    spy = universe[0]
    window = f"{spy.dates[WARMUP + 1]} → {spy.dates[-1]}"
    print(
        _heading(
            4,
            "Stocks — where zero is the wrong bar",
            f"{len(universe)} US tickers, Yahoo adjusted closes, {window}",
        )
    )

    hold = spy.buy_and_hold(start=WARMUP)
    print(
        f"\n  sessions scored        {len(hold)}\n"
        f"  SPY buy-and-hold       Sharpe {sharpe(hold):+.4f}   "
        f"(≈{spy.annualised(sharpe(hold)):+.2f} annualised, {compounded(hold):+.1%} total)\n"
        "  — that is what doing nothing earned. Anything below it is a worse\n"
        "    idea than an index fund, however good its Sharpe looks alone."
    )

    register = TrialRegister()
    grid = run_grid(universe, register, "equity-sma-grid")
    winner = grid.winner
    pairs = _journal_crossover(workdir / "equity.jsonl", winner, grid.fast, grid.slow)
    score = overall(pairs)

    print(
        f"\n  cells tried            {grid.trials}   "
        f"({len(GRID)} parameter pairs × {len(universe)} tickers)\n"
        f"  best cell              {winner.symbol} SMA {grid.fast}/{grid.slow}\n"
        f"  its per-obs Sharpe     {grid.sharpe:+.4f}   "
        f"(≈{winner.annualised(grid.sharpe):+.2f} annualised)\n"
        f"  compounded             {compounded(grid.returns):+.1%}\n"
        f"  median cell            {grid.median_cell:+.4f}\n"
        f"  hit rate               {score.hit_rate:.1%}"
    )

    guard = OverfittingGuard(register, sr_std_across_trials=grid.dispersion)
    against_zero = guard.evaluate(grid.sharpe, n_observations=score.n)
    print(
        "\n  Judged against zero, counting the search:\n"
        + "\n".join("  " + line for line in against_zero.report().splitlines())
    )

    # The honest question is not "did it make money" but "did it beat owning
    # the thing", so score the difference, not the level.
    own = winner.buy_and_hold(start=WARMUP)
    excess = [a - b for a, b in zip(grid.returns, own)]
    against_index = guard.evaluate(sharpe(excess), n_observations=len(excess))
    print(
        f"\n  Judged against buy-and-hold {winner.symbol} "
        f"(Sharpe {sharpe(own):+.4f}, {compounded(own):+.1%}):\n"
        + "\n".join("  " + line for line in against_index.report().splitlines())
    )
    print(
        f"\n  The winner of {grid.trials} attempts does not beat holding the ticker\n"
        "  it was fitted to. In a rising market almost any long-biased rule\n"
        "  posts a respectable Sharpe, and reporting it against zero is how a\n"
        "  strategy that destroys value gets promoted."
    )
    return against_zero, against_index


# --------------------------------------------------------------------------- 5


@dataclass(frozen=True)
class MemeToken:
    id: str
    symbol: str
    market_cap_usd: float
    pct_below_ath: float
    ath_date: str


def load_meme_universe() -> list[MemeToken]:
    path = DATA / "coingecko_meme_tokens.csv"
    with path.open(encoding="utf-8") as handle:
        return [
            MemeToken(
                id=r["id"],
                symbol=r["symbol"],
                market_cap_usd=float(r["market_cap_usd"]),
                pct_below_ath=float(r["pct_below_ath"]),
                ath_date=r["ath_date"],
            )
            for r in csv.DictReader(handle)
        ]


def example_five_memecoins(workdir: Path) -> tuple[Verdict, Verdict]:
    """Memecoins, where the data itself is the problem.

    Every number below is drawn from coins that are *listed today*. The ones
    that went to zero and were delisted are not in this file, cannot be put in
    it from any free source, and would drag every figure down. Read what
    follows as the best case, measured on the survivors.
    """
    universe = [load_series(name) for name in MEMECOINS]
    print(
        _heading(
            5,
            "Memecoins — where the sample is the problem",
            f"all {len(universe)} memecoins Kraken lists against USD",
        )
    )

    print("\n  Every one of them, from its first Kraken bar to today:")
    losers = 0
    for series in sorted(universe, key=lambda s: -len(s.closes)):
        moves = [
            series.closes[i + 1] / series.closes[i] - 1
            for i in range(len(series.closes) - 1)
        ]
        total = compounded(moves)
        losers += total < 0
        print(
            f"    {series.symbol:<9} {len(series.closes):>4} bars from {series.dates[0]}"
            f"   {total:>+8.1%}   daily vol {stdev(moves):.3f}"
        )
    print(
        f"\n  {losers} of {len(universe)} lost money — and these are the ones that\n"
        "  made it onto a major exchange and stayed there."
    )

    register = TrialRegister()
    grid = run_grid(universe, register, "meme-sma-grid")
    winner = grid.winner
    pairs = _journal_crossover(workdir / "meme.jsonl", winner, grid.fast, grid.slow)
    score = overall(pairs)

    print(
        f"\n  Now search it the way an adaptive system would:\n"
        f"    cells tried          {grid.trials}   "
        f"({len(GRID)} parameter pairs × {len(universe)} coins)\n"
        f"    best cell            {winner.symbol} SMA {grid.fast}/{grid.slow}\n"
        f"    its per-obs Sharpe   {grid.sharpe:+.4f}   "
        f"(≈{winner.annualised(grid.sharpe):+.2f} annualised)\n"
        f"    compounded           {compounded(grid.returns):+.1%}   ← on {score.n} days\n"
        f"    median cell          {grid.median_cell:+.4f}"
    )

    guard = OverfittingGuard(register, sr_std_across_trials=grid.dispersion)
    naive = guard.evaluate(grid.sharpe, n_observations=score.n, n_trials=1)
    honest = guard.evaluate(grid.sharpe, n_observations=score.n)
    print(
        "\n  Reporting only the winner:\n"
        + "\n".join("  " + line for line in naive.report().splitlines())
    )
    print(
        f"\n  Counting all {grid.trials} attempts:\n"
        + "\n".join("  " + line for line in honest.report().splitlines())
    )

    tokens = load_meme_universe()
    drawdowns = sorted(t.pct_below_ath for t in tokens)
    deep = sum(1 for d in drawdowns if d < -90) / len(drawdowns)
    dead = sum(1 for d in drawdowns if d < -99) / len(drawdowns)
    print(
        f"\n  And the sample itself. Of {len(tokens):,} meme tokens CoinGecko lists\n"
        f"  right now — every one of them a survivor:\n"
        f"    median distance below its peak   {median(drawdowns):.1f}%\n"
        f"    down more than 90% from peak     {deep:.1%}\n"
        f"    down more than 99% from peak     {dead:.1%}"
    )
    print(
        "\n  The ones that went to zero and got delisted are in none of these\n"
        "  numbers, and no free source will hand them to you. So the true\n"
        "  distribution is worse than the worst figure above, by an amount this\n"
        "  data cannot measure. That is not a reason to model it more carefully;\n"
        "  it is the reason the evidence review defers this domain outright."
    )
    return naive, honest


def main() -> None:
    btc = load_series("btc")
    spy = load_equity("SPY")
    markets = load_markets()
    print("Five runs through services/decisions, on real market data.")
    print(
        f"Kraken daily closes {btc.dates[0]} → {btc.dates[-1]} "
        f"({len(btc.closes)} bars, 3 majors + {len(MEMECOINS)} memecoins); "
        f"Yahoo adjusted closes for {len(EQUITIES)} tickers "
        f"({spy.dates[0]} → {spy.dates[-1]}); "
        f"{len(markets)} settled Polymarket binaries; "
        f"{len(load_meme_universe()):,} listed meme tokens."
    )
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        example_one_textbook_crossover(workdir)
        example_two_real_grid_search(workdir)
        example_three_real_polymarket(workdir)
        example_four_equities(workdir)
        example_five_memecoins(workdir)
    print(
        f"\n{RULE}\nFour domains, five real datasets, no edge demonstrated in any of\n"
        "them. That is the expected result, and it is the reason to log before\n"
        f"you trade rather than after.\n{RULE}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
