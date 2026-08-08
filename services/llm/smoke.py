"""Phase 1 acceptance test against a live endpoint.

    python -m services.llm.smoke                      # 100 calls, 20 concurrent
    python -m services.llm.smoke --n 200 --concurrency 40

Asserts the two numbers Phase 1 is accepted on:

* p95 under the latency ceiling at 20 concurrent
* 100% JSON parse rate

Exits non-zero if either fails, so it drops into CI or a post-provider-change
check. Note the second one is about *parsing*, not about being right: a run can
be 100% parseable and still fail Gate A. This test says the endpoint works and
the structured-output mode is wired correctly, nothing more.

It costs real money now — 100 calls against a cheap model is fractions of a
cent, but check --n before pointing it at a frontier model.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Optional, Sequence

from .client.client import LLMClient, UpstreamUnavailable
from .client.config import LLMSettings
from .client.observability import InMemoryMetrics, percentile
from .schemas.base import AbstainReason
from .schemas.sentiment import SentimentVote

# Varied so the run exercises real decoding rather than becoming a provider-side
# cache hit on an identical prompt, which would flatter the latency numbers.
SAMPLES = [
    "Spot ETH ETF inflows hit $310m on Tuesday, the largest single day since launch.",
    "Binance will delist four spot pairs on March 14, citing low liquidity.",
    "The Dencun upgrade activated on mainnet without incident.",
    "A wallet linked to the Mt. Gox trustee moved 47,229 BTC this morning.",
    "Tether's Q4 attestation shows $5.2b of excess reserves, up from $3.3b.",
]


async def _run(n: int, concurrency: int, settings: LLMSettings) -> tuple[list[float], int, int, int]:
    metrics = InMemoryMetrics()
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    parsed = errors = abstained = 0

    async with LLMClient(settings, metrics=metrics) as client:
        print("warming...", flush=True)
        if not await client.warm():
            raise SystemExit(
                "warm() failed — check the API key, the base URL, and that the "
                f"model slug {settings.model!r} still exists"
            )

        async def one(index: int) -> None:
            nonlocal parsed, errors, abstained
            async with semaphore:
                try:
                    outcome = await client.complete(
                        "sentiment", SentimentVote, SAMPLES[index % len(SAMPLES)]
                    )
                except UpstreamUnavailable as exc:
                    errors += 1
                    print(f"  call {index}: {exc}")
                    return

            latencies.append(outcome.latency_s)
            if outcome.reason is AbstainReason.SCHEMA_FAIL:
                print(f"  call {index}: parse failure — {outcome.detail}")
            else:
                parsed += 1
            if outcome.abstained:
                abstained += 1

        started = time.monotonic()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.monotonic() - started

    print(f"\n{n} calls at {concurrency} concurrent in {wall:.1f}s ({n / wall:.1f}/s)")
    return latencies, parsed, errors, abstained


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-p95", type=float, default=8.0)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    settings = LLMSettings()
    if args.base_url:
        settings.base_url = args.base_url
    if args.model:
        settings.model = args.model

    latencies, parsed, errors, abstained = asyncio.run(
        _run(args.n, args.concurrency, settings)
    )

    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    parse_rate = parsed / args.n if args.n else 0.0

    print(f"p50 {p50:.3f}s   p95 {p95:.3f}s")
    print(f"parse rate {parse_rate:.4f}   abstentions {abstained}   transport errors {errors}")

    checks = [
        (f"p95 < {args.max_p95}s", p95 < args.max_p95 and errors == 0),
        ("100% parse rate", parse_rate >= 1.0),
    ]
    print()
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
