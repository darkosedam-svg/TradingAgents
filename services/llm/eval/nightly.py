"""Nightly drift check: re-run the golden sets against the live model.

    python -m services.llm.eval.nightly --baseline eval-results/baseline
    python -m services.llm.eval.nightly --baseline eval-results/baseline --write-baseline

Cheap to run and the only thing that catches a silent break from a model,
kernel, or vLLM upgrade before the P&L does. Exits non-zero on a regression
beyond the budget, so cron/CI alerts without extra glue.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional, Sequence

from ..client.config import LLMSettings
from .. import prompts
from .gates import GateResult, drift_check, summarize
from .harness import TASK_SCHEMAS, evaluate
from .report import load_metrics


async def _run_all(
    tasks: Sequence[str], settings: LLMSettings, baseline_dir: Path, out_dir: Path
) -> tuple[list[GateResult], list[str]]:
    results: list[GateResult] = []
    notes: list[str] = []

    for task in tasks:
        run = await evaluate(task, label="nightly", settings=settings)
        run.save(out_dir / f"{task}.nightly.json")

        baseline_path = baseline_dir / f"{task}.baseline.json"
        if not baseline_path.exists():
            notes.append(f"{task}: no baseline at {baseline_path}, skipped")
            continue

        _, baseline = load_metrics(baseline_path)
        results.append(drift_check(baseline, run.metrics))

    return results, notes


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("eval-results/baseline"))
    parser.add_argument("--out", type=Path, default=Path("eval-results/nightly"))
    parser.add_argument("--tasks", nargs="*", default=sorted(TASK_SCHEMAS))
    parser.add_argument("--max-regression", type=float, default=0.02)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Store this run as the new baseline instead of comparing against it.",
    )
    args = parser.parse_args(argv)

    settings = LLMSettings()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.write_baseline:
        args.baseline.mkdir(parents=True, exist_ok=True)
        for task in args.tasks:
            run = asyncio.run(evaluate(task, label="baseline", settings=settings))
            path = run.save(args.baseline / f"{task}.baseline.json")
            print(f"baseline written: {path}")
        return 0

    results, notes = asyncio.run(
        _run_all(args.tasks, settings, args.baseline, args.out)
    )

    # Log the prompt identity alongside the verdict: a regression is only
    # traceable if you know which prompt produced it.
    print(json.dumps({t: p.ref for t, p in prompts.registry().items()}, indent=2))
    for note in notes:
        print(note)
    if not results:
        print("no baselines to compare against — run with --write-baseline first")
        return 0

    print(summarize(results, title=f"\nDrift vs baseline ({settings.model})\n"))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
