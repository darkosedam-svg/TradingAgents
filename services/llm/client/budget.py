"""Per-task token and latency budgets.

Two jobs. First, cap what any one task can spend per day, so a retry storm or a
prompt that suddenly starts emitting 2k tokens shows up as a refusal rather than
as a bill. Second, give each task an explicit latency budget, because the layer
lives on the warm path and "slow" and "wrong" cost the same when a decision
window closes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


class BudgetExceeded(RuntimeError):
    """A task has spent its daily allowance. Resolves to abstain, never to a retry."""


@dataclass(frozen=True)
class TaskBudget:
    """Limits for a single task type."""

    max_tokens_per_call: int = 256
    max_latency_s: float = 8.0
    max_tokens_per_day: Optional[int] = None
    max_calls_per_day: Optional[int] = None


# Sized from the plan's serving config: a 200-token structured response at
# 8192 context. Triage is the high-volume task and gets the largest daily pool;
# signal parsing is rare and gets room for longer messages.
DEFAULT_BUDGETS: dict[str, TaskBudget] = {
    "sentiment": TaskBudget(
        max_tokens_per_call=256, max_latency_s=2.5, max_tokens_per_day=4_000_000
    ),
    "news_triage": TaskBudget(
        max_tokens_per_call=320, max_latency_s=3.0, max_tokens_per_day=12_000_000
    ),
    "signal_parse": TaskBudget(
        max_tokens_per_call=384, max_latency_s=4.0, max_tokens_per_day=2_000_000
    ),
    "token_meta": TaskBudget(
        max_tokens_per_call=256, max_latency_s=5.0, max_tokens_per_day=6_000_000
    ),
}


@dataclass
class _Spend:
    day: int = 0
    tokens: int = 0
    calls: int = 0


@dataclass
class BudgetLedger:
    """Tracks daily spend per task and rolls over at UTC midnight."""

    budgets: dict[str, TaskBudget] = field(
        default_factory=lambda: dict(DEFAULT_BUDGETS)
    )
    default: TaskBudget = TaskBudget()
    clock: Callable[[], float] = time.time
    _spend: dict[str, _Spend] = field(default_factory=dict, repr=False)

    def budget_for(self, task: str) -> TaskBudget:
        return self.budgets.get(task, self.default)

    def _today(self) -> int:
        return int(self.clock() // 86_400)

    def _current(self, task: str) -> _Spend:
        spend = self._spend.setdefault(task, _Spend(day=self._today()))
        today = self._today()
        if spend.day != today:
            spend.day, spend.tokens, spend.calls = today, 0, 0
        return spend

    def check(self, task: str) -> None:
        """Raise :class:`BudgetExceeded` if this task has nothing left today."""
        budget = self.budget_for(task)
        spend = self._current(task)

        if budget.max_calls_per_day is not None and spend.calls >= budget.max_calls_per_day:
            raise BudgetExceeded(
                f"{task}: {spend.calls} calls today, limit {budget.max_calls_per_day}"
            )
        if budget.max_tokens_per_day is not None and spend.tokens >= budget.max_tokens_per_day:
            raise BudgetExceeded(
                f"{task}: {spend.tokens} tokens today, limit {budget.max_tokens_per_day}"
            )

    def charge(self, task: str, tokens: int) -> None:
        spend = self._current(task)
        spend.tokens += max(0, tokens)
        spend.calls += 1

    def spent(self, task: str) -> tuple[int, int]:
        """``(tokens, calls)`` spent by this task today."""
        spend = self._current(task)
        return spend.tokens, spend.calls

    def remaining_tokens(self, task: str) -> Optional[int]:
        budget = self.budget_for(task)
        if budget.max_tokens_per_day is None:
            return None
        return max(0, budget.max_tokens_per_day - self._current(task).tokens)


__all__ = ["BudgetExceeded", "BudgetLedger", "DEFAULT_BUDGETS", "TaskBudget"]
