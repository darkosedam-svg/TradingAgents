# Triage inference layer

Warm-path agent inference on a small, cheap model, reserving frontier calls for
what survives triage. Serves sentiment, news/text triage, structured extraction,
and escalation routing.

It is deliberately decoupled: everything upstream talks to it over HTTP against
an OpenAI-compatible endpoint, so it carries no dependency on the system it
serves and can be lifted into another one unchanged. Runtime dependencies are
`httpx` and `pydantic`.

> **This layer was originally specified to run a quantized model locally under
> vLLM.** There is no GPU available, so it runs a cheap hosted model instead —
> same architecture, same economics, less to break. The reasoning is recorded in
> [`docs/llm-layer-decisions.md`](../../docs/llm-layer-decisions.md); the Phase 0
> measurement that gates the whole thing is in
> [`docs/llm-baseline.md`](../../docs/llm-baseline.md) and is **still unmeasured**.

## Layout

```
services/llm/
├─ client/          client, circuit breaker, router, budgets, metrics
│  ├─ client.py         async schema-constrained calls, 1 retry, hard timeout
│  ├─ schema_tools.py   Pydantic model → provider-acceptable JSON Schema
│  ├─ breaker.py        trips to the fallback provider after N failures
│  ├─ router.py         fallback policy + the escalation policy
│  ├─ budget.py         per-task token, latency and dollar budgets
│  ├─ advisory.py       off-entry-path enrichment with an enforced deadline
│  ├─ shadow.py         paired-vote logging for Gate B and the Phase 6 spot check
│  └─ observability.py
├─ schemas/         Pydantic contracts, one per task
├─ prompts/         versioned, content-hashed, one file per task
├─ eval/            golden sets, harness, metrics, report, gates, nightly drift
├─ smoke.py         the Phase 1 acceptance check
└─ tests/
```

## Configuring

Everything comes from the environment, with working defaults:

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen-2.5-7b-instruct   # confirm the slug and price with your provider
LLM_API_KEY=...                        # falls back to OPENROUTER_API_KEY, then OPENAI_API_KEY
LLM_STRUCTURED_MODE=json_schema        # json_schema | guided_json | prompt
```

### Structured-output modes

Providers disagree on how to constrain output, so it's a setting rather than a
hardcoded field:

| mode | sends | use when |
|---|---|---|
| `json_schema` | `response_format: {"type": "json_schema", strict}` | default — OpenAI, OpenRouter, most gateways |
| `guided_json` | vLLM's `guided_json` extension | self-hosting behind vLLM |
| `prompt` | nothing; the schema goes into the system prompt | the model has no structured-output support |

`prompt` mode parses defensively — markdown fences and "Here you go:" preambles
are recovered — but expect a higher `SCHEMA_FAIL` abstention rate. That's the
correct failure, not a silent one.

Validation keywords (`minimum`, `maxLength`, …) are stripped from the schema
sent upstream, because strict mode rejects them. Nothing is lost: the provider
guarantees the response *parses into the right shape*, and the Pydantic
validators — which still run on every response — guarantee the values are *in
range and self-consistent*.

## Calling it

```python
from services.llm import LLMClient, LLMSettings, Router
from services.llm.schemas import SentimentVote

async with LLMClient(LLMSettings()) as client:
    await client.warm()                     # fails fast on a bad key or dead slug
    router = Router(client, fallback=second_provider)

    outcome = await router.run("sentiment", SentimentVote, headline)
    if outcome.abstained:
        log.info("no vote: %s (%s)", outcome.reason.value, outcome.detail)
    else:
        cast_vote(outcome.value.sentiment, outcome.value.confidence)
```

There is no path that returns a default on failure. An unparseable response, a
value that fails a validator, a confidence below the task floor, and an
unreachable endpoint with no fallback all produce an abstaining `Outcome`:

| reason | cause |
|---|---|
| `INSUFFICIENT_DATA` | the input does not support an answer |
| `LOW_CONFIDENCE` | validated, but under the task's floor |
| `SCHEMA_FAIL` | did not parse, or failed a validator |

## Escalation routing

The commercial payoff. Triage every candidate cheaply, spend frontier tokens
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

Abstentions escalate by default. The moment the triage model cannot read an item
is the worst possible moment to drop it silently.

Watch for drift with `EscalationRateMonitor` — a router that quietly stops
escalating looks exactly like a quiet news week until the P&L disagrees.

## Budgets

Serving is metered now, so the ledger tracks dollars alongside tokens:

```python
from services.llm.client import BudgetLedger, Pricing

ledger = BudgetLedger(pricing=Pricing(prompt_per_1m=0.30, completion_per_1m=1.20))
router = Router(client, fallback, ledger=ledger)
...
ledger.spent_usd()          # today, across all tasks
```

Put your provider's real prices in — the defaults are zero, which reports zero
spend and is worse than no number at all. A task that exhausts
`max_usd_per_day` routes to the fallback rather than retrying.

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
# check the endpoint works and the structured mode is wired right
python -m services.llm.smoke

# reference (frontier) and candidate (cheap), same golden set
python -m services.llm.eval.harness --task sentiment --label reference-frontier \
    --model anthropic/claude-sonnet-4.5
python -m services.llm.eval.harness --task sentiment --label candidate-cheap

# diff table + Gate A verdict; exits non-zero on failure
python -m services.llm.eval.report \
    eval-results/sentiment.reference-frontier.json \
    eval-results/sentiment.candidate-cheap.json

# nightly drift against a stored baseline
python -m services.llm.eval.nightly --baseline eval-results/baseline --write-baseline
python -m services.llm.eval.nightly --baseline eval-results/baseline
```

These cost real money. A golden-set run is one call per item; on a cheap model
that's fractions of a cent, on a frontier reference it is not. Check `--n` and
the model before pointing anything at a frontier slug.

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

| gate | question | the number that blocks |
|---|---|---|
| A | is the cheap model good enough? | field accuracy within 2% of the frontier reference, `valid@1` ≥ 99%, false-confidence ≤ reference |
| B | can it vote on sentiment? | ≥ 90% agreement over ≥ 500 paired decisions, zero malformed votes |
| C | does the router pay for itself? | ≥ 95% recall, ≥ 60% frontier-call reduction |
| D | is the advisory feature off the entry path? | entry timing byte-identical with the feature removed |

## Fallback drill

Revoke the API key, or point `LLM_BASE_URL` at a dead host, and confirm the
breaker routes to the fallback provider and nothing else notices. Do it once
deliberately, before it happens by accident.

## Tests

```bash
pip install -r services/llm/requirements-dev.txt
pytest services/llm/tests -q
```

No network and no credentials needed — the client is exercised against an
`httpx` mock transport, and the breaker and budgets use injected clocks.
