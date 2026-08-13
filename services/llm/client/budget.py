"""Per-task token, latency, and **cost** budgets.

Cost moved to the centre when serving moved off a card you own. Every triage
call is now metered, so the ledger tracks dollars alongside tokens and can
refuse a task that has spent its daily allowance. A retry storm or a prompt that
starts emitting 2k tokens shows up as a refusal rather than as a surprise
invoice.

The latency budgets are the other half: the layer lives on the warm path, and
"slow" and "wrong" cost the same when a decision window closes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


class BudgetExceeded(RuntimeError):
    """A task has spent its daily allowance. Resolves to abstain, never to a retry."""


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Check your provider — these move."""

    prompt_per_1m: float = 0.0
    completion_per_1m: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_per_1m
            + completion_tokens * self.completion_per_1m
        ) / 1_000_000


@dataclass(frozen=True)
class TaskBudget:
    """Limits for a single task type."""

    max_tokens_per_call: int = 256
    max_latency_s: float = 20.0
    max_tokens_per_day: Optional[int] = None
    max_calls_per_day: Optional[int] = None
    max_usd_per_day: Optional[float] = None


# Sized for a cheap hosted triage model. The latency numbers are warm-path
# ceilings, not targets — measure the real p95 before trusting them.
DEFAULT_BUDGETS: dict[str, TaskBudget] = {
    "sentiment": TaskBudget(
        max_tokens_per_call=256, max_latency_s=8.0, max_usd_per_day=2.00
    ),
    "news_triage": TaskBudget(
        max_tokens_per_call=320, max_latency_s=10.0, max_usd_per_day=5.00
    ),
    "signal_parse": TaskBudget(
        max_tokens_per_call=384, max_latency_s=12.0, max_usd_per_day=1.00
    ),
    "token_meta": TaskBudget(
        max_tokens_per_call=256, max_latency_s=15.0, max_usd_per_day=2.00
    ),
}


@dataclass
class _Spend:
    day: int = 0
    tokens: int = 0
    calls: int = 0
    usd: float = 0.0


@dataclass
class BudgetLedger:
    """Tracks daily spend per task and rolls over at UTC midnight."""

    budgets: dict[str, TaskBudget] = field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    default: TaskBudget = TaskBudget()
    pricing: Optional[Pricing] = None
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
            spend.day, spend.tokens, spend.calls, spend.usd = today, 0, 0, 0.0
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
        if budget.max_usd_per_day is not None and spend.usd >= budget.max_usd_per_day:
            raise BudgetExceeded(
                f"{task}: ${spend.usd:.4f} spent today, limit ${budget.max_usd_per_day:.2f}"
            )

    def charge(
        self, task: str, tokens: int, *, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        """Record one call's spend.

        Pass the prompt/completion split when pricing is configured — the two
        sides are priced differently everywhere, often by 3-5x.
        """
        spend = self._current(task)
        spend.tokens += max(0, tokens)
        spend.calls += 1
        if self.pricing is not None:
            spend.usd += self.pricing.cost(prompt_tokens, completion_tokens)

    def spent(self, task: str) -> tuple[int, int]:
        """``(tokens, calls)`` spent by this task today."""
        spend = self._current(task)
        return spend.tokens, spend.calls

    def spent_usd(self, task: Optional[str] = None) -> float:
        """Dollars spent today, for one task or across all of them."""
        if task is not None:
            return self._current(task).usd
        return sum(self._current(name).usd for name in list(self._spend))

    def remaining_tokens(self, task: str) -> Optional[int]:
        budget = self.budget_for(task)
        if budget.max_tokens_per_day is None:
            return None
        return max(0, budget.max_tokens_per_day - self._current(task).tokens)

    def remaining_usd(self, task: str) -> Optional[float]:
        budget = self.budget_for(task)
        if budget.max_usd_per_day is None:
            return None
        return max(0.0, budget.max_usd_per_day - self._current(task).usd)


__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "DEFAULT_BUDGETS",
    "Pricing",
    "TaskBudget",
]
