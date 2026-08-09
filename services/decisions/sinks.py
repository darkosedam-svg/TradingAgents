"""Where a decision goes once it has been made.

This is the seam that makes automated execution an adapter rather than a
rewrite. The pipeline that produces a :class:`Decision` never imports a sink,
never knows whether the destination is a log file, a Telegram message or a
broker — it returns the decision and something else decides what to do.

Today the destination is a human. The day it becomes an exchange, the reasoning
code is untouched: you write one new `Sink` and change one wiring line.

The ordering guarantee in :class:`FanOut` matters more than it looks. The
journal is registered first so that a decision is recorded *before* anyone acts
on it — if the alert fires and the process dies, the journal already has the
row, and the record of what the system believed cannot be lost by a downstream
failure.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

from .journal import DecisionJournal
from .record import Decision, Side

logger = logging.getLogger(__name__)


class Sink(Protocol):
    """Anything that can act on a decision."""

    def emit(self, decision: Decision) -> None: ...


class NullSink:
    """Discards everything. The default, so an unwired pipeline is inert
    rather than accidentally live."""

    def emit(self, decision: Decision) -> None:
        return None


class JournalSink:
    """Writes the decision to the append-only journal. Register this first."""

    def __init__(self, journal: DecisionJournal) -> None:
        self.journal = journal

    def emit(self, decision: Decision) -> None:
        self.journal.append(decision)


def format_alert(decision: Decision) -> str:
    """One decision as a line a human can act on without opening anything."""
    arrow = {Side.LONG: "▲", Side.SHORT: "▼", Side.FLAT: "—"}[decision.side]
    head = (
        f"{arrow} {decision.side.value.upper()} {decision.instrument} "
        f"[{decision.domain.value}]  conf {decision.confidence:.0%}"
    )
    body = [head, decision.rationale]
    if decision.size_fraction:
        body.append(f"suggested size: {decision.size_fraction:.1%} of unit risk")
    if decision.sources:
        body.append("sources: " + ", ".join(decision.sources))
    if decision.horizon_s:
        body.append(f"judge over: {decision.horizon_s}s")
    body.append(f"id: {decision.decision_id[:8]}  strategy: {decision.strategy_id}")
    return "\n".join(body)


class AlertSink:
    """Sends the decision to a human. The default destination.

    ``deliver`` is any callable taking the formatted string — print, a logger,
    a Telegram send. Delivery failures are swallowed and logged: a notification
    that cannot be sent must not take down the pipeline that produced it, and
    the journal already holds the record.
    """

    def __init__(
        self,
        deliver: Callable[[str], None] = print,
        *,
        min_confidence: float = 0.0,
    ) -> None:
        self.deliver = deliver
        self.min_confidence = min_confidence

    def emit(self, decision: Decision) -> None:
        if decision.confidence < self.min_confidence:
            return
        try:
            self.deliver(format_alert(decision))
        except Exception:  # noqa: BLE001 — a failed alert must not stop the system
            logger.warning(
                "alert delivery failed for %s", decision.decision_id, exc_info=True
            )


class ExecutionSink:
    """Placeholder for automated trading. **Refuses to run.**

    Deliberately not implemented. The research this design follows found no
    evidence that any of these markets can be traded profitably by a system
    like this, and the plan is explicit that execution comes only after a
    measured edge exists. Wiring this in is a decision a person makes on
    purpose, having read the numbers — not something that quietly becomes
    possible because the class was there.

    When that day comes, implement `emit` against your broker. Everything
    upstream stays exactly as it is; that is the whole point of the seam.
    """

    def __init__(self, *, i_have_measured_an_edge: bool = False) -> None:
        if not i_have_measured_an_edge:
            raise NotImplementedError(
                "Automated execution is not wired up. Prove an edge first: run "
                "decisions through the journal, score them, and clear "
                "OverfittingGuard with a real trial count. Then implement this "
                "sink against your broker."
            )

    def emit(self, decision: Decision) -> None:  # pragma: no cover
        raise NotImplementedError("Implement against your broker.")


class FanOut:
    """Several sinks, in order, each isolated from the others' failures.

    Order is significant: put the journal first so the record survives a
    downstream crash.
    """

    def __init__(self, *sinks: Sink) -> None:
        self.sinks = sinks

    def emit(self, decision: Decision) -> None:
        for sink in self.sinks:
            try:
                sink.emit(decision)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "sink %s failed on %s",
                    type(sink).__name__,
                    decision.decision_id,
                    exc_info=True,
                )


def alerting_stack(
    journal: DecisionJournal, deliver: Callable[[str], None] = print, **kwargs
) -> FanOut:
    """The recommended wiring: record first, then tell the human."""
    return FanOut(JournalSink(journal), AlertSink(deliver, **kwargs))


__all__ = [
    "AlertSink",
    "ExecutionSink",
    "FanOut",
    "JournalSink",
    "NullSink",
    "Sink",
    "alerting_stack",
    "format_alert",
]
