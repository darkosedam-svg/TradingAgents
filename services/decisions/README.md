# Decision records and the brake on the learning loop

Four things, built in the order the evidence says to build them. **No
dependencies — stdlib only.** You should be able to use this before you have
installed anything, spent anything, or decided what you're building.

This implements steps 1–3 of the build order in the
[evidence review](../../docs/unified-agent-findings.md). It is deliberately not
a trading system: it records what a system decided, scores whether it was right,
refuses to let it reweight itself on noise, and runs the daily loop that turns
all of that into an actual track record.

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

## 6. Five examples on real market data — one per domain

```bash
python -m services.decisions.real_examples     # committed data, offline
python -m services.decisions.data.fetch        # refresh it
```

Nothing simulated, no API keys anywhere:

| Source | What |
|---|---|
| Kraken | daily closes, BTC/ETH/SOL + all 13 memecoins listed against USD, 721 bars |
| Yahoo | daily **adjusted** closes, 8 US tickers, 501 sessions |
| Polymarket | 354 settled binaries, each with what it was quoting ~24h before it ended |
| CoinGecko | 999 currently-listed meme tokens and their distance from peak |

Polymarket quotes come from the CLOB price history, not the settled
`outcomePrices` — a closed market reports the answer, not the forecast. Equity
closes are adjusted because a raw close gaps on every dividend and split, and a
crossover rule will happily "predict" a gap nobody could trade.

Each domain fails in a different way, which is the reason to run all five.

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

### Stocks — zero is the wrong bar

Crypto went nowhere over this window, so reading a Sharpe against zero did
little harm. Equities drifted up hard, and a rule that is long most of the time
inherits that drift and reports it as skill.

```
SPY buy-and-hold        Sharpe +0.1270  (≈+2.02 annualised, +33.4%)

2,392 cells (299 pairs × 8 tickers)
best cell               SPY SMA 30/190, Sharpe +0.1192, +31.0%

judged against zero     deflated 0.383   FAIL
judged against SPY      Sharpe -0.0512   deflated 0.001   FAIL
```

The winner of 2,392 attempts is *worse than owning the ticker it was fitted
to*. A Sharpe of 1.89 annualised looks like a result right up until you ask
"compared to what". **In a rising market, the benchmark is the index.**

### Memecoins — the sample is the problem

All 13 memecoins Kraken lists against USD, from each one's first bar to today:

```
DOGE  -31.0%   SHIB  -65.5%   PEPE  -63.0%   WIF   -90.1%
BONK  -84.7%   FLOKI -81.8%   TURBO -79.6%   MEME  -94.9%
POPCAT-95.3%   MOG   -94.6%   TRUMP -94.9%   FART  -91.1%   PENGU -59.7%
```

**13 of 13 lost money** — and these are the ones that made it onto a major
exchange and stayed there. Then search them the way an adaptive system would,
3,887 cells across all 13:

```
best cell    TRUMP SMA 70/80, Sharpe +0.1077, compounded +281.4% over 368 days

reporting only the winner:  deflated 0.980   PASS
counting all 3,887:         deflated 0.332   FAIL
                            no-skill benchmark: +0.131 vs the winner's +0.108
```

A strategy that nearly quadrupled money in a year, correctly rejected. This is
the one case in the set where the naive verdict says **PASS** — which is exactly
how a system without a trial counter ends up trading it.

And the sample. Of 999 meme tokens CoinGecko lists right now, every one a
survivor: **median 95.9% below its peak; 66.9% down more than 90%; 21.8% down
more than 99%.** The ones that went to zero and got delisted are in none of
those numbers and no free source will produce them, so the true distribution is
worse by an amount the data cannot measure. That is why the evidence review
defers this domain rather than modelling it more carefully.

## 7. The daily loop — where a track record actually comes from

Everything above is measurement. This is the part that produces something to
measure. Step 3 of the build order is *pick one market and prove an edge
exists, paper-traded, scored against reality for months* — not a thing more
code can do, but code can make the daily act cost nothing so it actually
happens.

```bash
python -m services.decisions run       # fetch, resolve what's due, emit today
python -m services.decisions status    # where the track record stands
```

### Scheduling it

The job is set up as a GitHub Action:
[`.github/workflows/paper-trading.yml`](../../.github/workflows/paper-trading.yml),
daily at 06:15 UTC, committing the journal to [`paper/`](../../paper/).

**It does nothing until this branch is merged to `main`** — GitHub only runs
scheduled workflows from the default branch. After merging, trigger it once by
hand from the Actions tab to confirm it works rather than waiting for tomorrow.

Running it there rather than on a machine you own is the point: a track record
that lives on one laptop ends the first time that laptop is rebuilt, and
committing every entry makes the append-only property real rather than
aspirational. An entry cannot be quietly revised without the revision itself
being a commit — which is the whole defence against a record that improves in
hindsight.

Two things to know about GitHub's scheduler: it delays jobs under load and
occasionally drops one entirely. That costs nothing here — the loop is
idempotent and resolves whatever it missed next time. It also disables
scheduled workflows after 60 days of repository inactivity, so a long quiet
spell needs a click to re-enable.

If you would rather run it on a box you control — cron on a VPS:

```
15 6 * * *  cd /path/to/repo && /usr/bin/python3 -m services.decisions run --home paper >> paper.log 2>&1
```

systemd, if you want the run to survive a machine that was asleep at 06:15
(`Persistent=true` catches up on boot, which plain cron does not):

```ini
# ~/.config/systemd/user/paper.service
[Service]
Type=oneshot
WorkingDirectory=/path/to/repo
ExecStart=/usr/bin/python3 -m services.decisions run --home paper

# ~/.config/systemd/user/paper.timer
[Timer]
OnCalendar=*-*-* 06:15:00 UTC
Persistent=true
[Install]
WantedBy=timers.target
```

```bash
systemctl --user enable --now paper.timer
```

Windows Task Scheduler:

```powershell
schtasks /create /tn "paper trading" /sc daily /st 08:15 `
  /tr "cmd /c cd /d C:\path\to\repo && python -m services.decisions run --home paper >> paper.log 2>&1"
```

Whichever you pick, run exactly one of them. Two schedulers writing the same
journal will each see an unemitted day and each emit it.

To rehearse the job without waiting a day, or to catch up a run the scheduler
dropped:

```bash
python -m services.decisions run --home paper --today 2026-08-12
```

`--today` moves the line between a closed bar and a live one, so a future date
is refused: it would write decisions against prices nobody could have traded,
and the record has no way to tell that apart afterwards.

One run does two things **in this order**: resolve every decision whose horizon
has elapsed, then emit today's. A decision written today cannot see an outcome
recorded in the same pass — structural, not careful.

```
1071 scored decisions, 9 pending
registered trials: 3

strategy                   n     hit   sharpe     total     DSR
baseline-coinflip        357  47.1%  -0.0034   -11.9%   0.180  (baseline)
baseline-hold            357  48.5%  -0.0237   -25.7%   0.097  (baseline)
sma-20-50                357  52.9%  +0.0047    -5.7%   0.223

Against the baselines:
  beats every baseline on Sharpe: sma-20-50
  — necessary, not sufficient. The DSR column is the one that decides.

Guard:
  nothing clears the bar yet.
```

Note the third row: it beats both baselines and still lost 5.7%. Winning a
comparison is not the same as making money, and the table shows both so neither
can hide behind the other.

### What the loop refuses to do

- **Trade an unfinished bar.** Today's daily candle is a live price, not a
  close. Decisions are written against the last *closed* bar.
- **Emit twice in a day.** Running the job twice would double-count every
  decision, flattering or damning a strategy purely on how often cron fired.
- **Count a rerun as a new trial.** A trial is a distinct *attempt*, not a
  distinct run. `TrialRegister.register_once` and a persisted `trials.jsonl`
  keep the count honest across restarts — a register that lives only in memory
  reports one trial forever, which is the exact failure the guard exists to
  prevent.
- **Guess at a missing bar.** No exit price yet means the decision stays
  pending. Pending is an honest state; scoring against whatever bar happened to
  be next is not. A Friday call on an equity is judged on Monday, and that date
  is recorded.
- **Score against a bar that has not closed either.** The mirror of the first
  rule, and the one that actually bit during setup: resolving yesterday's call
  against today's live price books a partial day as a full day, biased in
  whichever direction the market happens to be moving when cron fires.
- **Run without baselines.** `AlwaysLong` and a date-seeded `Coinflip` ship in
  the defaults. The coin flip cannot be re-rolled, because re-rolling a
  baseline until it looks bad is the same cheat as re-rolling a strategy until
  it looks good.
- **Execute anything.** Decisions land in the journal. What reads the journal
  is somebody else's problem, and today that is a person.

### Backfill, and why it is fenced off

```bash
python -m services.decisions backfill --days 120
```

This replays history through the same loop, so you can see the wiring work
without waiting months. **It is a backtest wearing the runner's clothes.** Every
decision it writes is tagged `replayed`, and it raises rather than write into a
journal that already holds live entries — months of hindsight sitting in the
same file as a forward record would be indistinguishable to the guard.

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
- **Pooling instruments overstates your evidence.** Trading BTC, ETH and SOL
  on the same days gives you three rows per day, not three independent
  observations — they move together. `status` says so; the DSR is optimistic by
  an amount it does not measure.
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

125 tests, no network, no credentials, no dependencies — the real-data examples
read committed CSVs, so the suite never touches the internet. Several assert
architectural properties rather than behaviour — that `Decision` has no
execution fields, that outcomes never appear in a decision row, that swapping a
sink cannot change a decision. The three worked examples are tested too, because
their conclusions are documentation: if a change to the guard flips one of those
verdicts, the prose describing it has silently become wrong.
