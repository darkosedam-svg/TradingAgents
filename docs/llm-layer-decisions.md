# Local inference layer — framing decisions

Written so they don't get relitigated. Each of these was decided, not left open.
If one of them is revisited, amend this file with the date and the reason rather
than quietly changing the code.

## 1. AWQ-INT4 checkpoints from Hugging Face, served by vLLM. Not OmniQuant.

OmniQuant is a 2023 research artifact whose only turnkey runtime is MLC-LLM. It
does not export into vLLM, SGLang, or llama.cpp. Building on it means owning a
dead toolchain — every upgrade becomes our problem, and none of the ecosystem's
kernel work reaches us.

So: pre-quantized AWQ-INT4 checkpoints pulled from HF, served by vLLM. **Zero
quantization work of our own in v1** — no AutoAWQ runs, no calibration
pipelines. If Gate A fails, the answer is to step *up* in precision (INT8, or
FP8 on Ada/Hopper) or to use a smaller model at higher precision. Never down to
3-bit.

`awq_marlin`, not plain `awq`: same weights, materially faster kernel.

## 2. vLLM in Docker under WSL2, spoken to over HTTP.

The deployment target is Windows and vLLM is Linux-first. The container runs
under WSL2 with GPU passthrough and exposes an OpenAI-compatible endpoint on
`127.0.0.1:8000`. Everything upstream talks to it over HTTP and neither knows
nor cares that it is local.

The payoff is the fallback path: swapping back to a hosted provider is a
base-URL change, which is why the circuit breaker in `client/breaker.py` can be
this simple. Loopback-only binding is deliberate — the endpoint has no auth.

The image tag is pinned. A vLLM or kernel upgrade is a change to the serving
substrate and goes through the golden sets like any other change.

## 3. Warm path only.

The layer serves sentiment, news/text triage, structured extraction, and
escalation routing. It never enters:

- Pump.fun entry racing — PumpPortal WS → buy is a slot race.
- The latency-arb Phase 1 / 2A maker lane — the window is 30–90s but entry is
  sub-second.
- Any sizing, PnL, liquidation, or Kelly math.

Escalation routing is where the money is: it converts a hosted bill that scales
with *candidates* into one that scales with *finalists*.

This is enforced structurally, not by convention. `client/advisory.py` is the
only sanctioned way to call the layer from anywhere near an entry path, and it
cannot block past its deadline. Gate D is the test: remove the feature entirely
and entry timing must be byte-identical.

## 4. Abstention extends to quantization.

The existing `INSUFFICIENT_DATA` guardrail gains two siblings, both resolving to
abstain:

| reason | meaning |
|---|---|
| `INSUFFICIENT_DATA` | the input does not support an answer |
| `LOW_CONFIDENCE` | validated, but below the task's confidence floor |
| `SCHEMA_FAIL` | did not parse, or failed a validator |

Quantized models degrade on instruction-following before they degrade on
fluency. The failure mode to design against is a *confidently malformed* answer,
and the whole point of this machinery is to convert that into a non-vote rather
than a default. `Outcome` enforces the invariant at the type level: exactly one
of value or reason, never both, never neither.

`false_confidence_rate` — confident and wrong — is the metric Gate A is really
written against. Everything else in the eval harness is diagnostic.

## Decisions taken while implementing

**Raw HTTP rather than the `openai` SDK.** The layer needs a hard timeout,
exactly one retry, and a circuit breaker wrapped tightly around the transport.
All three are easier to be sure of when the request is a single `httpx` call,
and it keeps the package's dependencies to `httpx` + `pydantic`. The wire format
is unchanged, so this is invisible to the server.

**`extra="forbid"` on every schema.** Guided decoding keeps responses parseable;
forbidding extra keys means a model that invents a plausible field surfaces as
`SCHEMA_FAIL` instead of being silently dropped.

**Price geometry is a validator, not a metric.** A long stop above its entry is
rejected at parse time. Models that hallucinate prices get the *ordering* wrong
before they get the magnitude wrong, so this is a cheap and sharp detector — and
it means a bad signal abstains rather than arriving with one wrong field.

**Confidence floors are per task** (`client/config.py`). `news_triage` sits low
(0.40) on purpose: the router escalates on uncertainty, so a high floor there
would suppress exactly the items most worth a second look. `signal_parse` sits
high (0.70) because its errors are prices.

**Gate A's false-confidence check is strict by default.** `≤ reference`, no
slack, as specified. Note the cost honestly: on a 200-item set one extra
confident error moves the rate by 0.005, so a genuinely equivalent candidate can
fail on sampling noise. `false_confidence_slack` exists for that, and using it
is a decision to accept a measured increase in confident errors — record the
number here when you do.

## Non-goals (v1)

- Quantizing anything ourselves.
- Fine-tuning — revisit only if Gate A/B fails specifically on domain vocabulary.
- Replacing the frontier decision layer. The local model *feeds* it, never *is* it.
- An LLM anywhere in the latency-arb execution lane.
- Multi-GPU / tensor parallel — single card until throughput is genuinely the bottleneck.
- The ML Model specialist — different problem, don't conflate.
