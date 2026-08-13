"""Routing: the triage tier first, a fallback on failure, and escalation itself.

Two distinct decisions live here and are deliberately kept apart:

*Fallback* — the primary endpoint is down or over budget, so this call goes to a
second provider instead. An availability decision, and a cheap one.

*Escalation* — the triage model answered fine, and its answer says this
candidate is worth an expensive frontier call. A value decision, and the one the
whole layer is built to make: it converts a bill that scales with candidates
into one that scales with finalists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Type, TypeVar

from ..schemas.base import AbstainReason, Outcome, TaskOutput
from ..schemas.news import NewsTriage
from .budget import BudgetExceeded, BudgetLedger
from .client import LLMClient, UpstreamUnavailable

T = TypeVar("T", bound=TaskOutput)


class FallbackBackend(Protocol):
    """Anything that can answer a task when the primary endpoint cannot."""

    async def complete(
        self, task: str, schema: Type[T], user_content: str, **kwargs: Any
    ) -> Outcome[T]: ...


class Router:
    """Runs a task on the primary endpoint, falling back on unavailability."""

    def __init__(
        self,
        primary: LLMClient,
        fallback: Optional[FallbackBackend] = None,
        *,
        ledger: Optional[BudgetLedger] = None,
        abstain_on_unavailable: bool = True,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.ledger = ledger or BudgetLedger()
        self.abstain_on_unavailable = abstain_on_unavailable

    async def run(
        self, task: str, schema: Type[T], user_content: str, **kwargs: Any
    ) -> Outcome[T]:
        """Answer ``task``, or abstain. Never raises for an ordinary failure."""
        budget = self.ledger.budget_for(task)
        kwargs.setdefault("max_tokens", budget.max_tokens_per_call)

        try:
            self.ledger.check(task)
        except BudgetExceeded as exc:
            return await self._fallback(task, schema, user_content, str(exc), **kwargs)

        try:
            outcome = await self.primary.complete(task, schema, user_content, **kwargs)
        except UpstreamUnavailable as exc:
            return await self._fallback(task, schema, user_content, str(exc), **kwargs)

        self.ledger.charge(
            task,
            outcome.total_tokens,
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
        )
        return outcome

    async def _fallback(
        self, task: str, schema: Type[T], user_content: str, why: str, **kwargs: Any
    ) -> Outcome[T]:
        if self.fallback is None:
            if not self.abstain_on_unavailable:
                raise UpstreamUnavailable(why)
            # No answer and no fallback: there is no data to vote on. Abstaining
            # is the same guardrail INSUFFICIENT_DATA already encodes, so it
            # inherits those semantics rather than inventing new ones for the
            # consumer to handle.
            return Outcome.abstain(
                AbstainReason.INSUFFICIENT_DATA,
                detail=f"upstream unavailable: {why}",
                source="unavailable",
            )

        return await self.fallback.complete(task, schema, user_content, **kwargs)


@dataclass(frozen=True)
class EscalationPolicy:
    """When a triaged item deserves a frontier-model call.

    Tuned for recall over precision: a false escalate costs cents, a false skip
    costs an opportunity. That asymmetry drives all three settings below.
    """

    priority_threshold: float = 0.35
    relevance_threshold: float = 0.30
    # An abstention means the triage model could not read the item. That is the
    # worst possible moment to silently drop it.
    escalate_on_abstain: bool = True
    # A low-confidence but plausibly-relevant item is escalated rather than
    # dropped; the confidence floor for this task is set low for the same reason.
    escalate_on_low_confidence: bool = True

    def decide(self, outcome: Outcome[NewsTriage]) -> "EscalationDecision":
        if outcome.abstained:
            if (
                outcome.reason is AbstainReason.LOW_CONFIDENCE
                and self.escalate_on_low_confidence
            ) or self.escalate_on_abstain:
                return EscalationDecision(
                    escalate=True,
                    priority=1.0,
                    triage=None,
                    why=f"abstained ({outcome.reason.value if outcome.reason else 'unknown'}) — escalating for recall",
                )
            return EscalationDecision(
                escalate=False, priority=0.0, triage=None, why="abstained, policy skips"
            )

        triage = outcome.value
        assert triage is not None  # guaranteed by Outcome's invariant

        if triage.escalate:
            return EscalationDecision(
                escalate=True,
                priority=max(triage.priority, self.priority_threshold),
                triage=triage,
                why="model recommended escalation",
            )
        if triage.priority >= self.priority_threshold:
            return EscalationDecision(
                escalate=True,
                priority=triage.priority,
                triage=triage,
                why=f"priority {triage.priority:.2f} >= {self.priority_threshold:.2f}",
            )
        if triage.relevance >= self.relevance_threshold and triage.market_ids:
            return EscalationDecision(
                escalate=True,
                priority=triage.priority,
                triage=triage,
                why=f"relevant to {len(triage.market_ids)} market(s) despite low priority",
            )
        return EscalationDecision(
            escalate=False,
            priority=triage.priority,
            triage=triage,
            why=f"priority {triage.priority:.2f} < {self.priority_threshold:.2f}, relevance {triage.relevance:.2f}",
        )


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    priority: float
    triage: Optional[NewsTriage]
    why: str


class EscalationRateMonitor:
    """Week-over-week drift alarm on the escalation rate.

    A router that quietly stops escalating looks identical to a quiet news week
    right up until the P&L shows up. Comparing consecutive windows catches a
    silent model swap, a prompt edit, or a provider quietly changing what sits
    behind a slug — all of which shift the distribution without breaking
    anything visibly.
    """

    def __init__(self, drift_threshold: float = 0.20) -> None:
        self.drift_threshold = drift_threshold
        self._windows: list[tuple[int, int]] = []  # (escalated, total)

    def record_window(self, escalated: int, total: int) -> None:
        self._windows.append((escalated, total))

    @property
    def rates(self) -> list[float]:
        return [esc / tot for esc, tot in self._windows if tot]

    def drift(self) -> Optional[float]:
        """Relative change between the last two windows, or ``None``."""
        rates = self.rates
        if len(rates) < 2 or rates[-2] == 0:
            return None
        return (rates[-1] - rates[-2]) / rates[-2]

    def alarm(self) -> Optional[str]:
        drift = self.drift()
        if drift is None or abs(drift) <= self.drift_threshold:
            return None
        direction = "up" if drift > 0 else "down"
        return (
            f"escalation rate drifted {direction} {abs(drift) * 100:.1f}% "
            f"week-over-week ({self.rates[-2]:.3f} -> {self.rates[-1]:.3f}); "
            "check for a model, prompt, or kernel change"
        )


__all__ = [
    "EscalationDecision",
    "EscalationPolicy",
    "EscalationRateMonitor",
    "FallbackBackend",
    "Router",
]
