"""Append-only record of every decision, and the outcomes linked back to them.

Two rules make this useful rather than decorative:

**Append-only.** Decisions are never edited after the fact. A journal you can
rewrite is a journal that will quietly acquire hindsight.

**Outcomes arrive later, in their own entries.** A decision row can never
contain what happened next, because at the moment it was written nobody knew.
That is the structural guarantee against look-ahead — not a convention to
remember, a property of the file format.

One JSONL file, one object per line, each tagged with ``kind``. Ordering is
preserved, so replaying the file reconstructs exactly what was known when.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .record import Decision, Domain, Realisation


@dataclass(frozen=True)
class Pair:
    """A decision and what the market subsequently did."""

    decision: Decision
    realisation: Realisation

    @property
    def correct(self) -> bool:
        return self.realisation.scores(self.decision)

    @property
    def realised_return(self) -> float:
        return self.realisation.realised_return

    @property
    def signed_return(self) -> float:
        """Return as the decision experienced it — negated for a short."""
        from .record import Side

        if self.decision.side is Side.SHORT:
            return -self.realisation.realised_return
        if self.decision.side is Side.FLAT:
            return 0.0
        return self.realisation.realised_return


class DecisionJournal:
    """Append-only JSONL journal. Safe to read while another process writes."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------ write

    def append(self, decision: Decision) -> Decision:
        self._write({"kind": "decision", **decision.to_dict()})
        return decision

    def record_outcome(self, realisation: Realisation) -> Realisation:
        """Log what happened. Does not validate that the decision exists —
        the journal accepts the entry and :meth:`pairs` simply won't match an
        orphan, which keeps writes cheap and non-blocking."""
        self._write({"kind": "outcome", **realisation.to_dict()})
        return realisation

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------- read

    def _entries(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        for lineno, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}:{lineno} is not valid JSON: {exc}") from exc

    def decisions(self) -> list[Decision]:
        return [
            Decision.from_dict(e) for e in self._entries() if e.get("kind") == "decision"
        ]

    def outcomes(self) -> list[Realisation]:
        return [
            Realisation.from_dict(e) for e in self._entries() if e.get("kind") == "outcome"
        ]

    def pairs(self, *, domain: Optional[Domain] = None, strategy_id: Optional[str] = None) -> list[Pair]:
        """Decisions that have a recorded outcome, in decision order.

        A decision with several outcomes keeps the first — a correction should
        be a new decision, not a rewritten verdict on an old one.
        """
        resolved: dict[str, Realisation] = {}
        for outcome in self.outcomes():
            resolved.setdefault(outcome.decision_id, outcome)

        out: list[Pair] = []
        for decision in self.decisions():
            if domain is not None and decision.domain is not domain:
                continue
            if strategy_id is not None and decision.strategy_id != strategy_id:
                continue
            realisation = resolved.get(decision.decision_id)
            if realisation is not None:
                out.append(Pair(decision, realisation))
        return out

    def pending(self) -> list[Decision]:
        """Decisions still waiting on an outcome — the work queue for scoring."""
        resolved = {o.decision_id for o in self.outcomes()}
        return [d for d in self.decisions() if d.decision_id not in resolved]

    def __len__(self) -> int:
        return sum(1 for e in self._entries() if e.get("kind") == "decision")


def replay(journal: DecisionJournal) -> Iterable[Decision]:
    """Decisions in the order they were made — nothing about the future attached."""
    return journal.decisions()


__all__ = ["DecisionJournal", "Pair", "replay"]
