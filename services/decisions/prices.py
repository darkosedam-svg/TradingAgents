"""Daily price series, and the arithmetic every caller needs from them.

Shared by the worked examples and by the paper-trading runner, so that a
strategy scored in a backtest and the same strategy run forward are reading
identical code. When those two diverge is when a backtest stops predicting
anything about the live system.

Stdlib only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Optional

from .record import Domain

DATA = Path(__file__).parent / "data"

# Every memecoin Kraken lists against USD. Thirteen names is the whole
# investable universe on a major exchange, and that is itself a finding.
MEMECOINS = (
    "doge",
    "shib",
    "pepe",
    "wif",
    "bonk",
    "floki",
    "trump",
    "popcat",
    "mog",
    "turbo",
    "pengu",
    "fartcoin",
    "meme",
)

MAJORS = ("btc", "eth", "sol")

EQUITIES = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "KO", "XOM")


def sharpe(returns: list[float]) -> float:
    """Per-observation Sharpe. Never annualised here — see
    :meth:`Series.annualised`, which needs to know the calendar."""
    if len(returns) < 2:
        return 0.0
    spread = stdev(returns)
    return fmean(returns) / spread if spread > 0 else 0.0


def compounded(returns: list[float]) -> float:
    """Total return from reinvesting each period. Summing daily percentages is
    the flattering version and it is wrong by tens of points over two years."""
    total = 1.0
    for r in returns:
        total *= 1.0 + r
    return total - 1.0


@dataclass(frozen=True)
class Series:
    symbol: str
    dates: list[str]
    closes: list[float]
    domain: Domain = Domain.CRYPTO
    # Crypto trades every day; equities do not. Getting this wrong inflates an
    # equity Sharpe by about 20%.
    periods_per_year: int = 365

    def __len__(self) -> int:
        return len(self.closes)

    @property
    def last_date(self) -> str:
        return self.dates[-1]

    def annualised(self, per_observation_sharpe: float) -> float:
        return per_observation_sharpe * self.periods_per_year**0.5

    def index_of(self, date: str) -> Optional[int]:
        try:
            return self.dates.index(date)
        except ValueError:
            return None

    def close_on(self, date: str) -> Optional[float]:
        i = self.index_of(date)
        return None if i is None else self.closes[i]

    def first_close_on_or_after(
        self, date: str, *, within_days: int = 7
    ) -> Optional[tuple[str, float]]:
        """The first close on or after ``date``, within ``within_days``.

        Equities do not trade at weekends, so a one-day horizon starting Friday
        is judged on Monday. The bound matters: without it a stale series would
        silently score a decision against a bar weeks later, which is a
        different question from the one that was asked.
        """
        limit = (_date.fromisoformat(date) + timedelta(days=within_days)).isoformat()
        for i, day in enumerate(self.dates):
            if day >= date:
                return (day, self.closes[i]) if day <= limit else None
        return None

    def last_complete_index(self, today: Optional[str] = None) -> Optional[int]:
        """Index of the last bar that is definitely finished.

        Today's daily candle is a live price, not a close. Emitting a decision
        against it and then resolving against tomorrow's real close compares
        two different kinds of number, and the difference flatters whichever
        direction the day happened to be moving.
        """
        cutoff = today or datetime.now(timezone.utc).date().isoformat()
        for i in range(len(self.dates) - 1, -1, -1):
            if self.dates[i] < cutoff:
                return i
        return None

    def sma(self, window: int, i: int) -> float:
        return sum(self.closes[i - window + 1 : i + 1]) / window

    def position_at(self, fast: int, slow: int, i: int) -> int:
        """The crossover's stance on day ``i``: +1 long, −1 short.

        Reads closes up to and including ``i`` and nothing after. This is the
        single function both the backtest and the live runner call, so a
        discrepancy between them is impossible rather than merely unlikely.
        """
        if i < slow - 1:
            raise ValueError(f"need {slow} bars of history, have {i + 1}")
        return 1 if self.sma(fast, i) > self.sma(slow, i) else -1

    def crossover(
        self, fast: int, slow: int, *, start: Optional[int] = None
    ) -> list[tuple[int, int, float]]:
        """Run the crossover. Returns ``(index, position, signed return)``.

        The signal comes from closes up to and including day ``i``; the return
        is day ``i+1``'s. Nothing from the future touches the decision — the
        same guarantee the journal enforces structurally.
        """
        first = slow - 1 if start is None else start
        return [
            (
                i,
                self.position_at(fast, slow, i),
                self.position_at(fast, slow, i)
                * (self.closes[i + 1] / self.closes[i] - 1),
            )
            for i in range(first, len(self.closes) - 1)
        ]

    def buy_and_hold(self, *, start: int = 0) -> list[float]:
        return [
            self.closes[i + 1] / self.closes[i] - 1
            for i in range(start, len(self.closes) - 1)
        ]


def _read(path: Path, symbol: str, domain: Domain, periods: int) -> Series:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return Series(
        symbol=symbol,
        dates=[r["date"] for r in rows],
        closes=[float(r["close"]) for r in rows],
        domain=domain,
        periods_per_year=periods,
    )


def load_series(symbol: str, *, root: Optional[Path] = None) -> Series:
    """A Kraken crypto pair — the majors and the memecoins share a format.

    ``root`` defaults to the committed snapshot. The paper runner keeps its own
    freshly-fetched copy elsewhere, so daily operation never edits the dataset
    the examples are documented against.
    """
    domain = Domain.MEME if symbol in MEMECOINS else Domain.CRYPTO
    base = DATA if root is None else Path(root)
    return _read(base / f"kraken_{symbol}_usd_daily.csv", symbol.upper(), domain, 365)


def load_equity(ticker: str) -> Series:
    """A Yahoo ticker, on **adjusted** closes.

    Adjusted matters more than it sounds: a raw close gaps on every dividend
    and split, and a crossover rule will cheerfully "predict" a gap that was
    never tradeable.
    """
    return _read(
        DATA / f"yahoo_{ticker.lower()}_daily.csv", ticker.upper(), Domain.EQUITY, 252
    )


__all__ = [
    "DATA",
    "EQUITIES",
    "MAJORS",
    "MEMECOINS",
    "Series",
    "compounded",
    "load_equity",
    "load_series",
    "sharpe",
]
