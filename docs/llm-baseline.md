# Phase 0 — hosted-LLM baseline

**Status: not measured.** This is the worksheet, not the answer. Nothing past
Phase 1 should be built until the tables below are filled in from seven
consecutive days of real traffic.

Two days of work, and it can legitimately cancel the project — which is the
cheapest possible outcome if the numbers say the hosted bill was never the
problem.

## The off-ramp

> If frontier spend is under **~$15/month** and latency is acceptable, **stop
> here.** The layer is not worth the ops burden yet.

**The threshold moved down (2026-08-08), and the reason matters.** It was ~$30/mo
when the plan assumed a resident GPU — a card that must stay powered, a container
that breaks on driver upgrades, and a nightly eval job are heavy fixed costs, and
they need a bill that actually hurts to justify them.

Serving from a cheap hosted model removes almost all of that. The fixed cost is
zero, there is nothing to keep resident, and the remaining ops burden is a prompt
registry and a nightly eval run. A lower bar clears it. What is left to justify
is the *complexity* of a two-tier routing system, not the cost of hardware.

## Model it before you measure it

The full seven-day log is still the right way to *confirm* this, but you can
find the shape of the answer in two minutes:

```bash
python -m services.llm.breakeven --candidates 200 --frontier-share 1.0 \
    --escalation-rate 0.25 --triage-price 0.30 1.20 --frontier-price 3.00 15.00
```

It reduces to one inequality:

    triage_cost_per_call / frontier_cost_per_call  <  frontier_share_today

Screening every candidate cheaply only saves money if the cheap screen costs
less than the frontier calls it removes. Two runs bracket the whole decision:

| scenario | share reaching frontier today | saving | verdict |
|---|---:|---:|---|
| every candidate gets an expensive call | 100% | ~$93/mo | build it |
| a filter already sits in front | 20% | ~$4/mo | stop |

Same volume, same prices — a 25x swing in the answer. **`frontier_share_today`
is the number that decides this project**, and it is the one worth measuring
first if you measure nothing else. Fill in your own prices; the ones above are
placeholders, and model pricing moves faster than committed code.

Two further conditions worth checking before committing, both of which can still
end the project early:

- **Is the spend concentrated in triage or in decisions?** If most of the money
  goes to a small number of high-value frontier calls rather than to hundreds of
  cheap candidate screens, the escalation router has nothing to route and the
  payoff disappears. This is now the *main* way the project dies — it is a much
  sharper test than the dollar threshold.
- **Is latency actually a problem?** A cheap hosted model is not obviously faster
  than an expensive one; small models decode quicker but you are still paying a
  network round trip. If the case ever rested on latency, it no longer does. The
  case is cost, and only cost.

## 1. Volume and cost — 7 consecutive days

| day | calls | prompt tok | completion tok | $ | notes |
|---|---:|---:|---:|---:|---|
| | | | | | |

| | value |
|---|---:|
| calls/day (mean) | |
| tokens/day (mean) | |
| **$/day (mean)** | |
| **$/month (projected)** | |
| over the $15/mo off-ramp? | |

## 2. Latency

| | p50 | p95 |
|---|---:|---:|
| all calls | | |
| sentiment | | |
| triage / classification | | |
| decision-layer calls | | |

Note the decision window each call sits inside. A p95 that is fine in absolute
terms can still be a problem if the window is tighter.

## 3. Call sites

Where the calls come from, and which are candidates for the local layer. Anything
in an execution lane is out of scope by decision, not by measurement — see
`llm-layer-decisions.md`.

| call site | calls/day | $/day | task type | local candidate? |
|---|---:|---:|---|---|
| | | | | |

**Candidates-vs-finalists ratio.** For the triage path specifically: how many
candidates are screened per item that gets a frontier call today? This number is
the ceiling on what Gate C's ≥60% hosted-call reduction can deliver — if the
ratio is near 1:1, the reduction target is unreachable and the router is not
worth building.

| | value |
|---|---:|
| candidates screened/day | |
| items reaching a frontier call/day | |
| ratio | |

## 4. Triage model

**Settled 2026-08-08: there is no GPU, so this is a hosted model.** See the
amendment in `llm-layer-decisions.md`. What used to be a VRAM inventory is now a
model-and-price choice.

| | value |
|---|---|
| provider | |
| triage model slug | |
| $/1M prompt tokens | |
| $/1M completion tokens | |
| structured output supported? | |

Put the two prices into `Pricing` on the `BudgetLedger` so `spent_usd()` reports
real numbers rather than zero. Check them against the provider's page rather
than trusting any default in this repo — model pricing moves faster than
committed code.

**Projected triage cost.** Multiply the candidate volume from section 3 by the
per-call token estimate. This is the number that has to stay well under the
hosted-decision spend it displaces, or the router is moving money rather than
saving it:

| | value |
|---|---:|
| candidates screened/day | |
| avg tokens per triage call | |
| **projected triage $/day** | |
| frontier $/day displaced | |
| **net saving** | |

## 5. Decision

| | |
|---|---|
| date | |
| decided by | |
| **proceed / stop** | |
| reasoning | |

Record a *stop* here as firmly as a *proceed*. A cancelled project with a
written reason is a result; an abandoned one is a thing someone re-proposes in
six months.
