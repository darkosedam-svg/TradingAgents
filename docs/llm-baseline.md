# Phase 0 — hosted-LLM baseline

**Status: not measured.** This is the worksheet, not the answer. Nothing past
Phase 1 should be built until the tables below are filled in from seven
consecutive days of real traffic.

Two days of work, and it can legitimately cancel the project — which is the
cheapest possible outcome if the numbers say the hosted bill was never the
problem.

## The off-ramp

> If hosted spend is under **~$30/month** and latency is acceptable, **stop
> here.** The layer is not worth the ops burden yet.

This is a real off-ramp, not a formality. A GPU that must stay resident, a
container that breaks on driver upgrades, and a nightly eval job are all
permanent costs. They need to be paid for by a bill that actually hurts.

Two further conditions worth checking before committing, both of which can also
end the project early:

- **Is the spend concentrated in triage or in decisions?** If most of the money
  is going to a small number of high-value frontier calls rather than to
  hundreds of cheap candidate screens, the escalation router has nothing to
  route and the payoff disappears.
- **Is latency actually a problem?** If p95 on the hosted path is comfortably
  inside every decision window, local inference buys nothing on that axis, and
  the case rests entirely on cost.

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
| over the $30/mo off-ramp? | |

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

## 4. Hardware

| | value |
|---|---|
| GPU | |
| VRAM | |
| shared with backtests/training? | |

Pick the serving row from the VRAM:

| VRAM | model | quant | role |
|---|---|---|---|
| ≥ 24 GB | Qwen2.5-14B-Instruct | AWQ-INT4 | primary reasoning/triage |
| 12–24 GB | Qwen2.5-7B-Instruct | AWQ-INT4 | primary |
| any | Qwen2.5-3B-Instruct | AWQ-INT4 | high-volume classify only, never tool-calling |

If the GPU is shared with backtests or training, decide now whether to reserve
it or pin backtests off-hours. Contention shows up as a p95 blowout, and the
circuit breaker will paper over it by silently routing to hosted — which means
the bill quietly comes back and the dashboard is the only place you'd notice.

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
