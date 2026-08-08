"""Runs a model over the golden sets and emits scored, comparable results.

This is the phase most likely to get skipped and the one that makes the rest
safe, so it is built to be runnable before any integration exists: point it at
an endpoint, give it a task, get a JSON result file that ``report.py`` and
``gates.py`` consume.

    python -m services.llm.eval.harness --task sentiment --label candidate-awq
    python -m services.llm.eval.harness --task sentiment --label reference-fp16 \\
        --base-url http://localhost:8001/v1 --model Qwen/Qwen2.5-14B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Type

from ..client.client import LLMClient, LocalUnavailable
from ..client.config import LLMSettings
from ..schemas.base import TaskOutput
from ..schemas.news import NewsTriage
from ..schemas.sentiment import SentimentVote
from ..schemas.signal import ParsedSignal
from ..schemas.token_meta import TokenNarrativeFlags
from .metrics import TASK_SPECS, RunItem, TaskMetrics, score_run

GOLDEN_DIR = Path(__file__).parent / "golden"

TASK_SCHEMAS: dict[str, Type[TaskOutput]] = {
    "sentiment": SentimentVote,
    "news_triage": NewsTriage,
    "signal_parse": ParsedSignal,
    "token_meta": TokenNarrativeFlags,
}


@dataclass(frozen=True)
class GoldenItem:
    item_id: str
    payload: Any
    expected: dict[str, Any]
    notes: str = ""


def load_golden(task: str, path: Optional[Path] = None) -> list[GoldenItem]:
    """Read ``golden/<task>.jsonl``. One JSON object per line, blanks skipped."""
    target = path or GOLDEN_DIR / f"{task}.jsonl"
    if not target.exists():
        raise FileNotFoundError(f"no golden set at {target}")

    items: list[GoldenItem] = []
    for lineno, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{lineno} is not valid JSON: {exc}") from exc
        items.append(
            GoldenItem(
                item_id=str(record.get("id", lineno)),
                payload=record["input"],
                expected=record["expected"],
                notes=record.get("notes", ""),
            )
        )
    return items


def render_input(task: str, payload: Any) -> str:
    """Turn a golden item's ``input`` into the user message the model sees.

    Kept here rather than in the golden files so the framing can be changed in
    one place and re-measured, and so the labels stay pure data.
    """
    if isinstance(payload, str):
        return payload

    if task == "news_triage":
        markets = payload.get("markets", [])
        rendered = "\n".join(
            f"- {m['id']}: {m.get('question', '')}" if isinstance(m, dict) else f"- {m}"
            for m in markets
        )
        return (
            f"CANDIDATE MARKETS:\n{rendered or '(none)'}\n\n"
            f"NEWS ITEM:\n{payload.get('text', '')}"
        )

    if task == "token_meta":
        socials = payload.get("socials") or []
        return (
            f"NAME: {payload.get('name', '')}\n"
            f"SYMBOL: {payload.get('symbol', '')}\n"
            f"DESCRIPTION: {payload.get('description', '')}\n"
            f"SOCIALS: {', '.join(socials) if socials else '(none)'}"
        )

    if isinstance(payload, dict) and "text" in payload:
        return str(payload["text"])
    return json.dumps(payload, ensure_ascii=False)


async def run_item(
    client: LLMClient, task: str, item: GoldenItem, *, confidence_floor: float
) -> RunItem:
    schema = TASK_SCHEMAS[task]
    started = time.monotonic()
    try:
        outcome = await client.complete(
            task,
            schema,
            render_input(task, item.payload),
            min_confidence=confidence_floor,
        )
    except LocalUnavailable as exc:
        return RunItem(
            item_id=item.item_id,
            expected=item.expected,
            latency_s=time.monotonic() - started,
            error=str(exc),
        )

    # An abstention still tells us whether the response was *well-formed*;
    # SCHEMA_FAIL is the only reason that means it was not.
    valid = outcome.reason is None or outcome.reason.value != "SCHEMA_FAIL"
    predicted = outcome.value.model_dump() if outcome.value is not None else None
    confidence = outcome.value.confidence if outcome.value is not None else 0.0

    return RunItem(
        item_id=item.item_id,
        expected=item.expected,
        predicted=predicted,
        valid=valid,
        abstained=outcome.abstained,
        abstain_reason=outcome.reason.value if outcome.reason else None,
        confidence=confidence,
        latency_s=outcome.latency_s,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        error=outcome.detail if outcome.abstained else "",
    )


async def run_golden(
    client: LLMClient,
    task: str,
    items: Sequence[GoldenItem],
    *,
    concurrency: int = 8,
    confidence_floor: float = 0.5,
) -> list[RunItem]:
    """Run every item, bounded concurrency, order preserved."""
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(item: GoldenItem) -> RunItem:
        async with semaphore:
            return await run_item(
                client, task, item, confidence_floor=confidence_floor
            )

    return list(await asyncio.gather(*(guarded(item) for item in items)))


@dataclass
class RunResult:
    """A labelled run of one task, ready to diff against another."""

    label: str
    task: str
    model: str
    metrics: TaskMetrics
    items: list[RunItem]

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "task": self.task,
            "model": self.model,
            "metrics": self.metrics.as_dict(),
            "items": [asdict(i) for i in self.items],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path


async def evaluate(
    task: str,
    *,
    label: str,
    settings: Optional[LLMSettings] = None,
    golden_path: Optional[Path] = None,
    concurrency: int = 8,
) -> RunResult:
    settings = settings or LLMSettings()
    items = load_golden(task, golden_path)
    floor = settings.confidence_floor(task)

    async with LLMClient(settings) as client:
        await client.warm(task)
        records = await run_golden(
            client, task, items, concurrency=concurrency, confidence_floor=floor
        )

    metrics = score_run(records, TASK_SPECS[task], confidence_floor=floor)
    return RunResult(
        label=label, task=task, model=settings.model, metrics=metrics, items=records
    )


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SCHEMAS))
    parser.add_argument("--label", required=True, help="e.g. candidate-awq, reference-fp16")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--golden", default=None, type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = LLMSettings()
    if args.base_url:
        settings.base_url = args.base_url
    if args.model:
        settings.model = args.model

    result = asyncio.run(
        evaluate(
            args.task,
            label=args.label,
            settings=settings,
            golden_path=args.golden,
            concurrency=args.concurrency,
        )
    )
    out = args.out or Path("eval-results") / f"{args.task}.{args.label}.json"
    result.save(out)
    print(json.dumps(result.metrics.as_dict(), indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
