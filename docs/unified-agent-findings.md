# Unified multi-market agent — what the evidence supports

Findings from an adversarially verified research pass (9 Aug 2026) on the
question: *can crypto, memecoins, equities and Polymarket live inside one
self-improving trading system?*

Method: five search angles, 23 sources fetched, 112 claims extracted, the top 25
put to three independent verifiers each with instructions to refute. Two
refutations killed a claim. **13 survived, 12 were killed.**

Recorded here rather than only in chat because the build order below is
load-bearing, and because the killed claims are the sort of thing that gets
quietly reintroduced later.

---

## Headline

**Build four per-domain pipelines with a thin aggregator on top. Do not build
one agent that reasons across all four markets.**

The unification premise did not survive. Every claim asserting a usable
crypto→equity linkage was refuted 0–3.

## What survived

| Finding | Vote | Consequence |
|---|---|---|
| No demonstrated cross-market signal transfer | 3–0 | Unify only at the portfolio/alerting layer |
| Bigger, more general models do not buy edge | 3–0 | Keep a dumb baseline in every comparison |
| The self-improving loop is the biggest hazard | 3–0 | Count trials; deflate for the search |
| Polymarket: read it, don't trade it | 2–1 | Use odds as an input feature, not an alert-and-cross-the-spread venue |
| Memecoin data can't be backtested as collected | 3–0 | Defer the memecoin domain |
| Session/holiday alignment is solved and free | 3–0 | Use `pandas_market_calendars`; don't write it |

### The cross-market finding, in detail

The one peer-reviewed paper offering concrete numbers (Moro-Visconti, *Finance
Research Open*, 2025) prints its own figure sources as *"Source: Simulated data
from the BTC/USD series"* and *"Source: Author's simulation using geometric
Brownian motion."* It also contradicts itself: §4.6 reports a strategy at Sharpe
1.12 and 68.4% annualised; footnote 14 reports the same strategy at Sharpe 0.34,
CAGR 5.39%, drawdown −59%.

This is **absence of evidence, not proof of absence** — no verified source
tested cross-market transfer properly and found nothing. But the linkage cannot
be assumed. Any cross-market feature must earn its place in its own
out-of-sample test.

### The overfitting finding, in detail

Bailey & López de Prado, *The Deflated Sharpe Ratio* (JPM, 2014), verified
verbatim from the primary PDF:

- Holdout and k-fold CV **do not control backtest overfitting**. ~20 holdout
  applications at 95% make false positives the expected outcome.
- Expected maximum Sharpe rises with trial count **at exactly zero true skill**.
- *"A backtest where the researcher has not controlled for the extent of the
  search involved in his or her finding is worthless, regardless of how
  excellent the reported performance might be."*

Implemented in [`services/decisions/trials.py`](../services/decisions/trials.py).

### Polymarket, in detail

Well calibrated in aggregate (30-day Brier 0.045 across 113,135 resolved
markets; calibration slopes 0.952–0.981) — and brutal as a venue. Four
independent studies put the profitable share at ~30% and falling. One study of
588M trades found 1.7M users down ~$650m, with the top 1% capturing 76.5% of
profits.

Decisive detail: **winners are limit-order liquidity providers; losers are
market-order takers.** A system that scans many small markets and crosses the
spread on ranked alerts is the losing cohort by construction. Accuracy also
degrades sharply in thin markets — ~61% correct under $10k volume versus ~84%
above $100k.

## What was killed

Almost every claim a build case would want to lean on:

| Claim | Vote |
|---|---|
| BTC/S&P correlation flips by regime (0.25 / 0.67 / 0.39) | 0–3 |
| Bitcoin shocks explain 2.3% of daily S&P variance | 0–3 |
| Sentiment adds nothing to a linear volatility model | 0–3 |
| Sentiment only helps with a nonlinear learner | 0–3 |
| Sentiment improved 54.17% of ML cases | 1–2 |
| Directional accuracy ~50% across 918 experiments | 0–3 |
| Overfitted strategies systematically *lose* out of sample | 0–3 |
| Polymarket mispricing concentrates early and near resolution | 0–3 |
| Bitquery Pump.fun starts at $49/month | 0–3 |
| Look-ahead bias is the dominant cause of inflated backtests | 0–3 |

**Do not reintroduce any of these as support.**

## Where the research came up empty

Roughly a third of the brief produced nothing verified. These are gaps in that
round, not findings:

- All costs. No verified monthly figure for any data feed.
- Exchange rate limits; Helius/Birdeye/PumpPortal pricing; equity data tiers;
  Polymarket CLOB and Gamma mechanics.
- Event bus / feature store architecture, and how to timestamp a prediction
  market that only has a resolution date.
- Legal and platform rules — Polymarket US-person restrictions, exchange API
  terms, pattern-day-trader thresholds, tax record-keeping.
- Pump.fun participant loss rates.
- **LLM trading agents specifically** (TradingAgents, FinMem, FinAgent,
  TradingGPT).

That last one needs care: the foundation-model evidence is about **time-series
models, not LLM agents**. It is not evidence about the TradingAgents framework
in either direction. That framework is currently *unevidenced*, which is
different from disproven.

## Build order

1. **Split decision from execution.** — *implemented:
   [`services/decisions/record.py`](../services/decisions/record.py),
   [`sinks.py`](../services/decisions/sinks.py)*
2. **Log every decision with a trial counter.** — *implemented:
   [`journal.py`](../services/decisions/journal.py),
   [`trials.py`](../services/decisions/trials.py)*
3. **Pick one market and prove an edge exists.** Paper-traded, scored against
   reality for months. If you cannot beat a dumb baseline in one market, four
   will not rescue it.
4. **Add markets one at a time, pipelines separate.** Thin aggregator for
   ranking, sizing and correlation-aware exposure caps.
5. **Only then the learning loop, with brakes.** Trial count, minimum sample,
   deflated threshold.
6. **Defer indefinitely: memecoins and auto-execution.** Memecoins because you
   cannot backtest what you collect. Auto-execution because nothing has yet
   established an edge worth automating — revisit after step 3 produces a real
   number.

## How much to trust this

| Finding | Source | Weight |
|---|---|---|
| Deflated Sharpe | Peer-reviewed, verified verbatim from primary PDF | Strongest in the set |
| Market calendars | Primary docs + hands-on reproduction | Verified directly |
| Bitquery pipelines | Vendor docs, 8 Aug 2026 | Can change without notice |
| Foundation models | arXiv preprint | 5 mega-cap US equities, daily, one window |
| Polymarket stats | SSRN working papers, wallet-level not person-level | Figures move fast |
| Cross-market transfer | Absence of surviving evidence | Not proof of absence |

Worth sitting with: the only *peer-reviewed* source in the set is the one caught
publishing simulated data as evidence. Peer review was not the filter here —
reading the figure captions was.

## On "quantum"

The original request asked for a "quantum auto learn and adjust super agent."
Read as colloquial for "very advanced". Quantum hardware has no practical retail
trading application today and was not researched.

---

*This reviews evidence about system design. It is not financial advice, and
nothing here suggests any of these markets can be traded profitably.*
