"""Fetch the real market data the worked examples run on.

Two public sources, neither needing a key or an account:

* **Kraken** ``/0/public/OHLC`` — 720 daily candles per pair, which is as far
  back as that endpoint goes.
* **Polymarket** Gamma (market metadata and resolutions) plus the CLOB
  ``prices-history`` endpoint, used to recover what a market was *quoting*
  before it resolved. The resolution alone is useless for calibration: a closed
  market reports ``outcomePrices`` of ["1", "0"], which is the answer, not the
  forecast.

Run it to refresh the committed CSVs::

    python -m services.decisions.data.fetch

The output is committed so the examples stay reproducible and offline. Re-run
it and the numbers in the examples will move — that is expected, and the
examples print the date range they actually loaded.

Stdlib only, like everything else here.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

HERE = Path(__file__).parent

KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440"
GAMMA_MARKETS = (
    "https://gamma-api.polymarket.com/markets"
    "?closed=true&limit={limit}&offset={offset}&order=volumeNum&ascending=false"
    "&volume_num_min={vmin}&volume_num_max={vmax}"
)

# Volume bands, sampled separately. Gamma ranks by volume and caps how deep you
# may page, so a single sweep returns only blockbusters — and the claim worth
# testing is that thin markets are less accurate.
VOLUME_BANDS = (
    (1_000, 10_000),
    (10_000, 100_000),
    (100_000, 1_000_000),
    (1_000_000, 10_000_000),
    (10_000_000, 10_000_000_000),
)
CLOB_HISTORY = (
    "https://clob.polymarket.com/prices-history"
    "?market={token}&startTs={start}&endTs={end}&fidelity=180"
)

PAIRS = {"btc": "XBTUSD", "eth": "ETHUSD", "sol": "SOLUSD"}

# Every memecoin Kraken lists against USD. The list is short for a reason worth
# noticing: it is the survivors.
MEME_PAIRS = {
    "doge": "DOGEUSD",
    "shib": "SHIBUSD",
    "pepe": "PEPEUSD",
    "wif": "WIFUSD",
    "bonk": "BONKUSD",
    "floki": "FLOKIUSD",
    "trump": "TRUMPUSD",
    "popcat": "POPCATUSD",
    "mog": "MOGUSD",
    "turbo": "TURBOUSD",
    "pengu": "PENGUUSD",
    "fartcoin": "FARTCOINUSD",
    "meme": "MEMEUSD",
}

# SPY is the benchmark the others have to beat; the rest span sectors so no one
# industry's two years stands in for "equities".
EQUITIES = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "KO", "XOM")

YAHOO_CHART = (
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range={range}&interval=1d&events=div%2Csplit"
)
COINGECKO_MEMES = (
    "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    "&category=meme-token&order=market_cap_desc&per_page={per_page}&page={page}"
)

# Yahoo serves the chart endpoint to browsers; a bare urllib UA gets a 429.
USER_AGENT = "Mozilla/5.0 (compatible; services.decisions/1.0; research)"


def _get(url: str, *, retries: int = 3) -> Any:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"giving up on {url}: {last}")


# ----------------------------------------------------------------- crypto OHLC


def fetch_ohlc(name: str, pair: str) -> Path:
    payload = _get(KRAKEN_OHLC.format(pair=pair))
    if payload.get("error"):
        raise RuntimeError(f"kraken error for {pair}: {payload['error']}")
    result = payload["result"]
    key = next(k for k in result if k != "last")
    rows = result[key]

    path = HERE / f"kraken_{name}_usd_daily.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for ts, o, h, low, c, _vwap, volume, _count in rows:
            date = datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
            writer.writerow([date, o, h, low, c, volume])
    print(f"  {path.name}: {len(rows)} daily bars, {rows[0][0]} → {rows[-1][0]}")
    return path


# -------------------------------------------------------------------- equities


def fetch_equity(symbol: str, *, span: str = "2y") -> Optional[Path]:
    """Daily bars for one ticker, using Yahoo's **adjusted** close.

    Adjusted, not raw: a dividend or a split shows up in a raw close as a gap
    that no strategy could have traded, and a crossover rule will happily
    "predict" it. That is a look-ahead artefact wearing a price's clothes.
    """
    payload = _get(YAHOO_CHART.format(symbol=symbol, range=span))
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        print(f"  {symbol}: no data returned, skipping")
        return None
    series = result[0]
    stamps = series["timestamp"]
    adjusted = series["indicators"]["adjclose"][0]["adjclose"]
    quote = series["indicators"]["quote"][0]

    path = HERE / f"yahoo_{symbol.lower()}_daily.csv"
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for i, ts in enumerate(stamps):
            close = adjusted[i]
            if close is None:
                continue  # a halted or untraded session
            date = datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
            writer.writerow(
                [
                    date,
                    quote["open"][i],
                    quote["high"][i],
                    quote["low"][i],
                    close,
                    quote["volume"][i],
                ]
            )
            written += 1
    print(f"  {path.name}: {written} sessions")
    return path


# ------------------------------------------------------------- meme token universe


def fetch_meme_universe(*, pages: int = 4, per_page: int = 250) -> Path:
    """Every meme token CoinGecko currently lists, with its drawdown from peak.

    This is the closest thing to a survivorship measurement the free data
    allows. It still only contains tokens that are *listed today* — the ones
    that went to zero and were delisted are absent, and no amount of querying
    this endpoint will produce them. That absence is the finding.
    """
    rows: list[dict] = []
    impossible = 0
    for page in range(1, pages + 1):
        try:
            batch = _get(COINGECKO_MEMES.format(per_page=per_page, page=page))
        except RuntimeError as exc:
            print(f"  coingecko stopped at page {page}: {exc}")
            break
        if not isinstance(batch, list) or not batch:
            break
        for coin in batch:
            if coin.get("ath_change_percentage") is None:
                continue
            if float(coin["ath_change_percentage"]) > 0:
                # Nothing can trade above its own all-time high; when the
                # vendor says otherwise the recorded peak is stale, usually
                # after a redenomination. Drop it rather than let one
                # impossible row set the top of the range.
                impossible += 1
                continue
            rows.append(
                {
                    "id": coin["id"],
                    "symbol": coin.get("symbol", ""),
                    "market_cap_usd": coin.get("market_cap") or 0,
                    "price_usd": coin.get("current_price") or 0,
                    "pct_below_ath": round(float(coin["ath_change_percentage"]), 3),
                    "ath_date": (coin.get("ath_date") or "")[:10],
                }
            )
        time.sleep(2.0)  # the free tier is strict

    path = HERE / "coingecko_meme_tokens.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"  {path.name}: {len(rows)} listed meme tokens"
        + (f" ({impossible} dropped for an impossible peak)" if impossible else "")
    )
    return path


# ------------------------------------------------------------------ Polymarket


def _iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def _usable(market: dict) -> bool:
    """A clean binary market with an unambiguous resolution and real volume."""
    if market.get("umaResolutionStatus") != "resolved":
        return False
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        prices = json.loads(market.get("outcomePrices") or "[]")
        tokens = json.loads(market.get("clobTokenIds") or "[]")
    except json.JSONDecodeError:
        return False
    if outcomes != ["Yes", "No"] or len(tokens) != 2:
        return False
    # Resolved cleanly one way or the other — not a refund or a 50/50 split.
    if sorted(prices) != ["0", "1"]:
        return False
    if float(market.get("volumeNum") or 0) < 1_000:
        return False
    return _iso(market.get("endDate") or "") is not None


def _forecast_before_resolution(
    token: str, end: datetime, *, lead: timedelta
) -> Optional[tuple[float, int]]:
    """What the market was quoting ``lead`` before its scheduled end.

    Takes the last print at or before the cutoff, so the number is one a person
    could actually have acted on. Returns ``(price, timestamp)``.
    """
    cutoff = int((end - lead).timestamp())
    window_start = cutoff - int(timedelta(days=14).total_seconds())
    history = _get(
        CLOB_HISTORY.format(token=token, start=window_start, end=cutoff)
    ).get("history", [])
    points = [p for p in history if p["t"] <= cutoff and 0.0 < p["p"] < 1.0]
    if not points:
        return None
    last = points[-1]
    return float(last["p"]), int(last["t"])


def fetch_polymarket(
    *, pages: int = 6, per_page: int = 100, per_band: int = 90, lead_hours: int = 24
) -> Path:
    seen: set[str] = set()
    candidates: list[dict] = []
    for vmin, vmax in VOLUME_BANDS:
        band: list[dict] = []
        for page in range(pages):
            try:
                batch = _get(
                    GAMMA_MARKETS.format(
                        limit=per_page, offset=page * per_page, vmin=vmin, vmax=vmax
                    )
                )
            except RuntimeError as exc:
                print(f"  gamma ${vmin:,}–${vmax:,} stopped at page {page + 1}: {exc}")
                break
            if not batch:
                break
            for market in batch:
                slug = market.get("slug", "")
                if slug in seen or not _usable(market):
                    continue
                seen.add(slug)
                band.append(market)
            time.sleep(0.2)
        # Spread the sample across the band rather than clustering at its top.
        if len(band) > per_band:
            stride = len(band) / per_band
            band = [band[int(i * stride)] for i in range(per_band)]
        print(f"  gamma ${vmin:,}–${vmax:,}: {len(band)} markets")
        candidates.extend(band)

    lead = timedelta(hours=lead_hours)
    rows: list[dict] = []
    for market in candidates:
        end = _iso(market["endDate"])
        yes_token = json.loads(market["clobTokenIds"])[0]
        try:
            quote = _forecast_before_resolution(yes_token, end, lead=lead)
        except RuntimeError:
            continue
        if quote is None:
            continue
        price, quoted_at = quote
        resolved_yes = json.loads(market["outcomePrices"])[0] == "1"
        rows.append(
            {
                "slug": market.get("slug", ""),
                "end_date": end.date().isoformat(),
                "volume_usd": round(float(market["volumeNum"]), 2),
                "quoted_at": datetime.fromtimestamp(
                    quoted_at, timezone.utc
                ).isoformat(timespec="seconds"),
                "hours_before_end": round(
                    (end.timestamp() - quoted_at) / 3600.0, 2
                ),
                "p_yes": price,
                "resolved_yes": int(resolved_yes),
            }
        )
        if len(rows) % 50 == 0:
            print(f"  price history: {len(rows)}/{len(candidates)}")
        time.sleep(0.05)

    path = HERE / "polymarket_resolved.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}: {len(rows)} resolved binary markets")
    return path


def main() -> None:
    print("Kraken daily OHLC — majors:")
    for name, pair in PAIRS.items():
        fetch_ohlc(name, pair)

    print("\nKraken daily OHLC — memecoins:")
    for name, pair in MEME_PAIRS.items():
        try:
            fetch_ohlc(name, pair)
        except RuntimeError as exc:
            print(f"  {name}: {exc}")
        time.sleep(0.3)

    print("\nYahoo daily adjusted closes — equities:")
    for symbol in EQUITIES:
        fetch_equity(symbol)
        time.sleep(0.5)

    print("\nCoinGecko meme token universe:")
    fetch_meme_universe()

    print("\nPolymarket resolved binary markets:")
    fetch_polymarket()
    print("\nDone. Commit the CSVs so the examples stay reproducible.")


if __name__ == "__main__":  # pragma: no cover
    main()
