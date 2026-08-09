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

USER_AGENT = "services.decisions/1.0 (research; stdlib urllib)"


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
    print("Kraken daily OHLC:")
    for name, pair in PAIRS.items():
        fetch_ohlc(name, pair)
    print("\nPolymarket resolved binary markets:")
    fetch_polymarket()
    print("\nDone. Commit the CSVs so the examples stay reproducible.")


if __name__ == "__main__":  # pragma: no cover
    main()
