# Decision records and the brake on the learning loop

Three things, built in the order the evidence says to build them. **No
dependencies — stdlib only.** You should be able to use this before you have
installed anything, spent anything, or decided what you're building.

This implements steps 1 and 2 of the build order in the
[evidence review](../../docs/unified-agent-findings.md). It is deliberately not
a trading system: it records what a system decided, scores whether it was right,
and refuses to let it reweight itself on noise.

## 1. A decision is separate from its execution

```python
from services.decisions import Decision, Domain, Side

Decision(
    domain=Domain.CRYPTO,
    instrument="SOL-USD",
    side=Side.LONG,
    confidence=0.72,
    rationale="ETF inflow spike plus funding reset",
    strategy_id="momentum",
    sources=("news", "funding"),
)
```

`Decision` carries no venue, order type, account or credentials — and a test
asserts it never will. It states intent; something else acts.

```python
from services.decisions import DecisionJournal, alerting_stack

journal = DecisionJournal("data/decisions.jsonl")
sink = alerting_stack(journal, deliver=send_to_telegram)
sink.emit(decision)
```

Today `deliver` is a Telegram message. The day it becomes a broker call you
write one new sink and change one wiring line — the reasoning code is untouched.
That is the entire reason to bother now: it costs nothing today and is expensive
to retrofit.

`ExecutionSink` exists but **refuses to construct**. Automated trading is a
decision a person makes deliberately after seeing a measured edge, not something
that quietly becomes possible because a class was importable.

## 2. Everything is written down before anything acts on it

The journal is append-only JSONL, and outcomes arrive **later as their own
entries**:

```python
from services.decisions import Realisation

journal.record_outcome(
    Realisation(decision_id=decision.decision_id, realised_return=0.031)
)
```

A decision row structurally cannot contain what happened next, because at the
moment it was written nobody knew. That is the guarantee against look-ahead —
a property of the file format, not a convention to remember.

`alerting_stack` puts the journal first on purpose: if the alert fails, the
record survives.

## 3. Scoring — and the number that actually matters

```python
from services.decisions import overall, by_source, calibration, summary

pairs = journal.pairs()
print(summary(pairs))
```

Hit rate and per-source attribution are the obvious outputs. **Calibration is
the important one.** A system claiming 0.8 confidence should be right 80% of the
time; `calibration(pairs).overconfidence` is positive when it isn't.

Watch that number over weeks. If it grows, the system is learning to sound
certain rather than to be right — which is the exact failure this design exists
to catch. The Polymarket research is the cautionary case: prices there are
well calibrated in aggregate and most participants still lose money.

## 4. The brake

```python
from services.decisions import OverfittingGuard, TrialRegister

register = TrialRegister()
register.register("momentum", "20/50 crossover")   # every variant you try

guard = OverfittingGuard(register)
verdict = guard.evaluate(observed_sharpe, n_observations=len(pairs))
print(verdict.report())
```

```
[FAIL] Sharpe 0.104 over 40 obs, 1 trial(s)
  no-skill benchmark (best of 1): 0.000
  deflated Sharpe (P skill is real):            0.740
  observations needed:                          255
  deflated Sharpe 0.740 < 0.95 — not enough evidence for an edge this size:
  40 observations, roughly 255 needed. Keep logging.
```

**Nothing should reweight a signal, promote a strategy or move a threshold
without passing through this.**

### Why it exists

Bailey & López de Prado, *The Deflated Sharpe Ratio* (Journal of Portfolio
Management, 2014), establish three things that between them demolish the usual
retail approach:

- Holdout testing and k-fold cross-validation **do not control backtest
  overfitting.** Apply a holdout about twenty times at 95% confidence and false
  positives stop being unlikely — they become expected.
- The expected *best* Sharpe across N attempts rises with N **even when true
  skill is exactly zero.** Search widely enough and something looks brilliant by
  construction. `expected_max_sharpe(1000) > 3.0`.
- Therefore: *"a backtest where the researcher has not controlled for the extent
  of the search involved in his or her finding is worthless, regardless of how
  excellent the reported performance might be."*

An auto-adapting system is a trial-generating machine. Every reweighting cycle
is another attempt. Without a trial counter it does not learn — it searches,
finds noise, and acts on it with your money.

The guard enforces two independent conditions, because either alone is easy to
fool: **enough evidence** (`min_observations`, and the track-record length the
effect size actually requires), and **enough performance after correcting for
the search** (`min_dsr`, default 0.95).

It also distinguishes the two ways you can fail, because they need opposite
responses:

| verdict says | what it means | what to do |
|---|---|---|
| *not enough evidence* | real but small edge, thin sample | keep logging |
| *the search itself is the problem* | never cleared the no-skill bar | stop searching; more data won't help |

### The argument in one test

```python
honest   = guard.evaluate(1.6, n_observations=800, n_trials=1)      # passes
searched = guard.evaluate(1.6, n_observations=800, n_trials=1000)   # fails
```

Identical Sharpe, identical sample, opposite verdict — decided purely by how
many attempts it took to find. That is the whole point.

## 5. Three worked examples

```bash
python -m services.decisions.examples
```

Three situations you will actually be in, run end to end through the journal,
the scorer and the guard. Seeded, so the numbers reproduce exactly.

| | Situation | Verdict |
|---|---|---|
| 1 | One idea, never re-tuned, real Sharpe-0.9 edge, 260 days | **FAIL** — real but unproven; needs ~812 observations |
| 2 | 400-cell grid search over data with *zero* edge | **PASS** if you report only the winner, **FAIL** once the 400 are counted |
| 3 | Prediction-market triage, 65% hit rate, 5:10 payoff | **FAIL** — right more often than not, and losing money |

Example 2 is the one to sit with. The best cell scores 0.172 per observation —
about 2.7 annualised — over 250 days, with a deflated Sharpe of **0.997** if you
present it alone. Disclose the other 399 attempts and the same numbers give
**0.420**, below a no-skill benchmark of 0.185. The true edge is exactly zero by
construction.

Example 3 is the one people find least intuitive: a 65% hit rate against a
break-even of 66.7% is a losing system, and the hit rate is the number that gets
reported.

## 6. The same three, on real market data

```bash
python -m services.decisions.real_examples     # committed data, offline
python -m services.decisions.data.fetch        # refresh it
```

Nothing simulated. Kraken daily closes for BTC/ETH/SOL (721 bars, the full
depth of the public endpoint) and 354 settled Polymarket binaries, each paired
with what the market was **quoting about a day before it ended** — taken from
the CLOB price history, because a closed market's `outcomePrices` is the answer,
not the forecast. Neither source needs a key.

**BTC/USD 20/50 crossover**, 520 daily decisions, Mar 2025 → Aug 2026. The
textbook parameters, fixed in advance:

```
hit rate            50.6%
per-obs Sharpe      +0.0071  (≈+0.14 annualised)
compounded          -4.4%    (BTC itself: -25.3%)
observations needed 53,147
```

It beat holding, which is worth something and is not an edge.

**Every crossover cell on the same series** — 299 of them:

```
best cell           SMA 5/40, Sharpe +0.0633 (≈+1.21 annualised)
median cell         -0.0045

reporting only the winner:   deflated 0.925   FAIL (just)
counting all 299 attempts:   deflated 0.533   FAIL
                             no-skill benchmark for 299 tries: +0.060
```

The winner's 0.0633 sits a hair above what the best of 299 coin flips produces
on this data. The holdout is the interesting part: **ETH +0.0647, SOL +0.0208**,
neither fitted — and neither clears the bar either.

**Backing the favourite on 354 real resolved markets:**

```
favourite was right   84.2%
P&L on $1 per market  -14.77
at zero trading cost  -12.19
```

84% accurate and losing money — and not because of fees. The odds on a
favourite already *are* the odds. This is example 3's lesson with no simulation
anywhere in it.

One real finding fell out of this. The research brief cited ~61% accuracy under
$10k against ~84% over $100k. On this sample: 81.7% / 86.1% / 87.0% / 78.8% /
88.0% across five volume bands — **the gap does not reproduce**, and the
finding has been demoted in [the evidence
review](../../docs/unified-agent-findings.md) accordingly.

## Caveats worth carrying

- **Sharpe here is per-observation and not annualised.** Annualising a short,
  noisy sample is how a mediocre strategy starts looking impressive.
- **`sr_std_across_trials` sets the scale of the whole correction, and the
  default is not yours.** 1.0 is the paper's convention for *annualised*
  Sharpes. Per-observation Sharpes from a few hundred daily returns are
  dispersed nearer `1/√n` — 0.06 in example 2 — and leaving the default in
  place rejects everything indiscriminately. Run the search, keep the Sharpe of
  every cell including the bad ones, and pass them to
  `measure_trial_dispersion`.
- **The correction assumes roughly independent trials.** Twenty variants of one
  idea have a smaller effective N than twenty unrelated ideas, so the guard is
  optimistic when your search is correlated. Bailey's paper discusses estimating
  the effective count.
- **Disclosing N is necessary, not sufficient.** The paper concedes this openly:
  *"even this will not remove the danger completely."*
- **A genuinely single-use, pre-registered holdout is not attacked** by this
  argument. Decide the test before you look, run it once, and it means what it
  says.

## Tests

```bash
pytest services/decisions/tests -q
```

82 tests, no network, no credentials, no dependencies — the real-data examples
read committed CSVs, so the suite never touches the internet. Several assert
architectural properties rather than behaviour — that `Decision` has no
execution fields, that outcomes never appear in a decision row, that swapping a
sink cannot change a decision. The three worked examples are tested too, because
their conclusions are documentation: if a change to the guard flips one of those
verdicts, the prose describing it has silently become wrong.
