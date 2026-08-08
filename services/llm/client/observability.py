"""Per-task instrumentation: latency, parse failures, abstentions, tokens.

Phase 6 wants p50/p95, parse-failure rate, abstain rate, escalation rate and
tokens/day on the existing dashboard. This module is the source of those
numbers. It keeps a bounded in-memory window and exposes a snapshot; wiring it
to a real backend is a matter of implementing :class:`MetricsSink` and passing
it to the client.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol

from ..schemas.base import AbstainReason


@dataclass(frozen=True)
class CallRecord:
    """One local-or-hosted inference call, whatever its outcome."""

    task: str
    model: str
    source: str  # "local" | "hosted"
    latency_s: float
    ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    abstain_reason: Optional[AbstainReason] = None
    prompt_ref: str = ""
    error: str = ""
    escalated: Optional[bool] = None  # only set by triage tasks
    ts: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def parse_failure(self) -> bool:
        """A response that arrived but did not survive the schema."""
        return self.abstain_reason is AbstainReason.SCHEMA_FAIL


class MetricsSink(Protocol):
    def record(self, record: CallRecord) -> None: ...


class NullMetrics:
    """Default sink. Records nothing, costs nothing."""

    def record(self, record: CallRecord) -> None:  # noqa: D102 - protocol impl
        return None


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile. ``q`` in 0..1. Empty input returns 0.0."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return ordered[rank - 1]


@dataclass
class TaskStats:
    calls: int = 0
    errors: int = 0
    parse_failures: int = 0
    abstentions: int = 0
    escalations: int = 0
    escalation_decisions: int = 0
    hosted_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    abstain_by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict[str, float | int | dict[str, int]]:
        calls = self.calls or 1
        return {
            "calls": self.calls,
            "error_rate": self.errors / calls,
            "parse_failure_rate": self.parse_failures / calls,
            "abstain_rate": self.abstentions / calls,
            "abstain_by_reason": dict(self.abstain_by_reason),
            "hosted_share": self.hosted_calls / calls,
            "escalation_rate": (
                self.escalations / self.escalation_decisions
                if self.escalation_decisions
                else 0.0
            ),
            "p50_latency_s": round(percentile(self.latencies, 0.50), 4),
            "p95_latency_s": round(percentile(self.latencies, 0.95), 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class InMemoryMetrics:
    """Bounded, thread-naive metrics store suitable for a single worker."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._by_task: dict[str, TaskStats] = defaultdict(TaskStats)
        self.records: deque[CallRecord] = deque(maxlen=20_000)

    def record(self, record: CallRecord) -> None:
        if record.ts == 0.0:
            record = CallRecord(**{**record.__dict__, "ts": self._clock()})
        self.records.append(record)

        stats = self._by_task[record.task]
        stats.calls += 1
        stats.latencies.append(record.latency_s)
        stats.prompt_tokens += record.prompt_tokens
        stats.completion_tokens += record.completion_tokens
        if record.source == "hosted":
            stats.hosted_calls += 1
        if not record.ok:
            stats.errors += 1
        if record.parse_failure:
            stats.parse_failures += 1
        if record.abstain_reason is not None:
            stats.abstentions += 1
            stats.abstain_by_reason[record.abstain_reason.value] += 1
        if record.escalated is not None:
            stats.escalation_decisions += 1
            stats.escalations += int(record.escalated)

    def snapshot(self) -> dict[str, dict[str, float | int | dict[str, int]]]:
        return {task: stats.as_dict() for task, stats in sorted(self._by_task.items())}

    def escalation_rate(self, task: str = "news_triage") -> float:
        stats = self._by_task.get(task)
        if not stats or not stats.escalation_decisions:
            return 0.0
        return stats.escalations / stats.escalation_decisions


__all__ = [
    "CallRecord",
    "InMemoryMetrics",
    "MetricsSink",
    "NullMetrics",
    "TaskStats",
    "percentile",
]
