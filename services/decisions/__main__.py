"""Command line for the paper-trading loop.

    python -m services.decisions run       # fetch, resolve what's due, emit today
    python -m services.decisions status    # where the track record stands
    python -m services.decisions backfill  # replay history into a fresh journal

The point of ``run`` is that it costs nothing to do daily::

    0 6 * * *  cd /path/to/repo && python -m services.decisions run >> paper.log

Running it twice in a day is safe — the second run resolves whatever is due and
declines to emit again, because double-counting a decision would flatter or
damn a strategy purely on how often cron fired.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .journal import DecisionJournal
from .paper import default_strategies, run_once, status
from .prices import MAJORS, load_series
from .trials import TrialRegister

DEFAULT_HOME = Path("data/paper")


def _paths(home: Path) -> tuple[DecisionJournal, TrialRegister]:
    return DecisionJournal(home / "decisions.jsonl"), TrialRegister(home / "trials.jsonl")


def _universe(symbols: list[str]) -> dict:
    universe = {}
    for symbol in symbols:
        series = load_series(symbol)
        universe[f"{series.symbol}-USD"] = series
    return universe


def _refresh(symbols: list[str]) -> None:
    """Pull today's bars. Kept out of the library so the loop itself never
    depends on the network — an offline run still resolves and still scores."""
    from .data.fetch import MEME_PAIRS, PAIRS, fetch_ohlc

    known = {**PAIRS, **MEME_PAIRS}
    for symbol in symbols:
        if symbol not in known:
            print(f"  {symbol}: no Kraken pair configured, skipping refresh")
            continue
        fetch_ohlc(symbol, known[symbol])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="services.decisions")
    parser.add_argument(
        "command", choices=("run", "status", "backfill"), help="what to do"
    )
    parser.add_argument(
        "--home", type=Path, default=DEFAULT_HOME, help="where the journal lives"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(MAJORS),
        help="Kraken symbols to trade on paper",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the price refresh and use the committed data",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="backfill: how many days of history to replay",
    )
    args = parser.parse_args(argv)

    journal, register = _paths(args.home)

    if args.command == "status":
        print(status(journal, register))
        return 0

    if not args.offline:
        print("refreshing prices:")
        _refresh(args.symbols)

    universe = _universe(args.symbols)
    strategies = default_strategies()

    if args.command == "backfill":
        from .paper import backfill

        report = backfill(journal, universe, strategies, register, days=args.days)
        print(report)
        print()
        print(status(journal, register))
        return 0

    print(run_once(journal, universe, strategies, register))
    print()
    print(status(journal, register))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
