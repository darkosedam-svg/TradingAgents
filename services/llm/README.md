# Local quantized inference layer

Warm-path agent inference on a local vLLM endpoint, with a hosted API as the
tested fallback. Serves sentiment, news/text triage, structured extraction, and
escalation routing.

It is deliberately decoupled: everything upstream talks to it over HTTP, so it
carries no dependency on the system it serves and can be lifted into another one
unchanged. Dependencies are `httpx` and `pydantic`.

The framing decisions behind it are in [`docs/llm-layer-decisions.md`](../../docs/llm-layer-decisions.md),
and the Phase 0 measurement that gates the whole thing is in
[`docs/llm-baseline.md`](../../docs/llm-baseline.md).

## Layout

```
services/llm/
├─ docker/          vLLM container: compose file + .env.example
├─ client/          client, circuit breaker, router, budgets, metrics
│  ├─ client.py     async schema-constrained calls, 1 retry, hard timeout
│  ├─ breaker.py    trips to hosted after N consecutive failures
│  ├─ router.py     fallback policy + the escalation policy
│  ├─ budget.py     per-task token/latency budgets
│  ├─ advisory.py   off-entry-path enrichment with an enforced deadline
│  ├─ shadow.py     paired-vote logging for Gate B and the Phase 6 spot check
│  └─ observability.py
├─ schemas/         Pydantic contracts, one per task
├─ prompts/         versioned, content-hashed, one file per task
├─ eval/            golden sets, harness, metrics, report, gates, nightly drift
└─ tests/
```

## Running the server

```bash
cd services/llm/docker
cp .env.example .env        # pick the model row that matches your VRAM
docker compose up -d
curl localhost:8000/v1/models
```

Needs WSL2, an NVIDIA driver, and the NVIDIA Container Toolkit inside the WSL
distro. First boot pulls the checkpoint; weights persist in a named volume, so
subsequent restarts are seconds rather than minutes.

Then check the substrate actually performs — this is the Phase 1 acceptance
criterion, and worth re-running after any image or driver change:

```bash
python -m services.llm.smoke              # 100 calls, 20 concurrent
# [PASS] p95 < 2.5s
# [PASS] 100% parse rate
```

## Calling it

```python
from services.llm import LLMClient, LLMSettings, Router
from services.llm.schemas import SentimentVote

async with LLMClient(LLMSettings()) as client:
    await client.warm()                     # never let a cold start land mid-decision
    router = Router(client, hosted=my_hosted_backend)

    outcome = await router.run("sentiment", SentimentVote, headline)
    if outcome.abstained:
        log.info("no vote: %s (%s)", outcome.reason.value, outcome.detail)
    else:
        cast_vote(outcome.value.sentiment, outcome.value.confidence)
```

Every call is schema-constrained through vLLM's `guided_json`, so responses are
guaranteed to parse. Validators guarantee the values are in range and
self-consistent. Neither guarantees they are *right* — that is what `eval/` is
for.

There is no path that returns a default on failure. An unparseable response, a
value that fails a validator, a confidence below the task floor, and an
unreachable endpoint with no fallback all produce an abstaining `Outcome`:

| reason | cause |
|---|---|
| `INSUFFICIENT_DATA` | the input does not support an answer |
| `LOW_CONFIDENCE` | validated, but under the task's floor |
| `SCHEMA_FAIL` | did not parse, or failed a validator |

## Escalation routing

The commercial payoff. Triage every candidate locally, spend frontier tokens
only on what survives.

```python
from services.llm.client import EscalationPolicy
from services.llm.schemas import NewsTriage

policy = EscalationPolicy()          # tuned for recall: a false escalate costs
outcome = await router.run(          # cents, a false skip costs an opportunity
    "news_triage", NewsTriage, rendered_item
)
decision = policy.decide(outcome)
if decision.escalate:
    await frontier_analyst(item, priority=decision.priority)
```

Abstentions escalate by default. The moment the local model cannot read an item
is the worst possible moment to drop it silently.

Watch for drift with `EscalationRateMonitor` — a router that quietly stops
escalating looks exactly like a quiet news week until the P&L disagrees.

## Off the entry path

`advisory.py` is the only sanctioned way to call this layer from anywhere near
an execution decision, and it cannot block past its deadline:

```python
enricher = AdvisoryEnricher(deadline_s=3.0)
enricher.start(mint, lambda: classify_token(mint))
...
candidate.narrative = enricher.peek(mint)   # never awaits, never raises
```

Gate D is the test: replay launches with the feature removed entirely and entry
timing must be byte-identical.

## Evaluating

```bash
# reference and candidate, same golden set
python -m services.llm.eval.harness --task sentiment --label reference-fp16 \
    --base-url http://localhost:8001/v1 --model Qwen/Qwen2.5-14B-Instruct
python -m services.llm.eval.harness --task sentiment --label candidate-awq

# diff table + Gate A verdict; exits non-zero on failure
python -m services.llm.eval.report \
    eval-results/sentiment.reference-fp16.json \
    eval-results/sentiment.candidate-awq.json

# nightly drift against a stored baseline
python -m services.llm.eval.nightly --baseline eval-results/baseline --write-baseline
python -m services.llm.eval.nightly --baseline eval-results/baseline
```

The golden sets in `eval/golden/` are **seeds** — 10–12 items each, enough to
establish the format and cover the failure modes worth designing around. The
real sets are 150–300 hand-labelled items from your own traffic, and `gate_a`
fails any run with fewer than 150. See `eval/golden/README.md` for how to build
them without accidentally building a set that flatters the model.

Metrics: `valid@1`, per-field accuracy/F1, numeric exactness (zero tolerance on
prices), abstain rate, latency percentiles, and **false-confidence rate** — the
one that actually matters.

## Gates

All four are executable, in `eval/gates.py`. `gate_b`, `gate_c` and `gate_d`
take measurements you supply from shadow runs, labelled sets, and replays; they
exist so "we passed Gate B" is a thing you run rather than a thing you remember
deciding.

| gate | phase | the number that blocks |
|---|---|---|
| A | quant vs reference | field accuracy within 2%, `valid@1` ≥ 99%, false-confidence ≤ reference |
| B | sentiment shadow | ≥ 90% agreement over ≥ 500 paired decisions, zero malformed votes |
| C | escalation router | ≥ 95% recall, ≥ 60% hosted-call reduction |
| D | Pump.fun advisory | entry timing byte-identical with the feature removed |

## Fallback drill

Kill the container on purpose and confirm the breaker routes to hosted and
nothing else notices. Do it once deliberately, before it happens by accident:

```bash
docker compose -f services/llm/docker/docker-compose.yml stop vllm
# ... run a decision, confirm it completes via the hosted path ...
docker compose -f services/llm/docker/docker-compose.yml start vllm
```

## Tests

```bash
pip install -r services/llm/requirements-dev.txt
pytest services/llm/tests -q
```

No GPU and no network needed — the client is exercised against an `httpx`
mock transport, and the breaker and budget use injected clocks.
