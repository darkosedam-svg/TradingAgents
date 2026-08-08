"""Shadow-mode comparison for Phase 3 and the 1–2% spot check in Phase 6.

Both are the same shape: run two producers over the same input, log both
answers, change nothing about live behaviour, and read a disagreement rate off
the result. The only difference is the sampling rate and which one is primary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..eval.gates import GateResult, gate_b
from ..eval.metrics import agreement
from ..schemas.base import AbstainReason


@dataclass(frozen=True)
class ShadowPair:
    """One decision, seen by both producers."""

    item_id: str
    incumbent: Optional[str]  # the direction voted, or None for an abstention
    candidate: Optional[str]
    candidate_reason: Optional[str] = None
    incumbent_reason: Optional[str] = None
    # True when the candidate response failed its schema *and* the consumer
    # would have voted on it anyway. This must stay at zero: it is the exact
    # failure the abstention machinery exists to prevent.
    malformed_propagated: bool = False
    note: str = ""

    @property
    def agrees(self) -> bool:
        if self.incumbent is None and self.candidate is None:
            return True
        return self.incumbent is not None and self.incumbent == self.candidate


class ShadowLog:
    """Accumulates paired votes and answers "can the candidate go primary yet?"."""

    def __init__(self) -> None:
        self.pairs: list[ShadowPair] = []
        self.reviewed_disagreements: int = 0
        self.systematically_worse: bool = False

    def record(self, pair: ShadowPair) -> None:
        self.pairs.append(pair)

    def record_outcomes(
        self,
        item_id: str,
        incumbent_direction: Optional[str],
        candidate_direction: Optional[str],
        candidate_reason: Optional[AbstainReason] = None,
        malformed_propagated: bool = False,
    ) -> None:
        self.record(
            ShadowPair(
                item_id=item_id,
                incumbent=incumbent_direction,
                candidate=candidate_direction,
                candidate_reason=candidate_reason.value if candidate_reason else None,
                malformed_propagated=malformed_propagated,
            )
        )

    @property
    def agreement_rate(self) -> float:
        return agreement(
            (p.incumbent for p in self.pairs), (p.candidate for p in self.pairs)
        )

    @property
    def disagreements(self) -> list[ShadowPair]:
        """The set a human actually reads before flipping the candidate to primary."""
        return [p for p in self.pairs if not p.agrees]

    @property
    def malformed_propagated(self) -> int:
        return sum(1 for p in self.pairs if p.malformed_propagated)

    def gate(self, **overrides: object) -> GateResult:
        """Evaluate Gate B against everything logged so far."""
        params = {
            "agreement_rate": self.agreement_rate,
            "paired_decisions": len(self.pairs),
            "malformed_propagated": self.malformed_propagated,
            "reviewed_disagreements": self.reviewed_disagreements,
            "systematically_worse": self.systematically_worse,
        }
        params.update(overrides)
        return gate_b(**params)  # type: ignore[arg-type]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for pair in self.pairs:
                handle.write(json.dumps(asdict(pair)) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "ShadowLog":
        log = cls()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                log.record(ShadowPair(**json.loads(line)))
        return log


@dataclass
class SpotCheck:
    """Phase 6's slow-moving quality signal: re-run 1–2% of triage calls on a
    frontier model.

    Not a gate and not an alert on any single sample — the number worth watching
    is the disagreement rate's trend across weeks. A step change in it is the
    cheapest early warning that something upstream moved, and against a hosted
    triage model "upstream" includes changes you were never told about.
    """

    sample_rate: float = 0.015
    pairs: list[ShadowPair] = field(default_factory=list)

    def should_sample(self, position: int) -> bool:
        """Deterministic every-Nth sampling, so a replay reproduces exactly."""
        if self.sample_rate <= 0:
            return False
        stride = max(1, int(round(1 / self.sample_rate)))
        return position % stride == 0

    def record(self, pair: ShadowPair) -> None:
        self.pairs.append(pair)

    @property
    def disagreement_rate(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(1 for p in self.pairs if not p.agrees) / len(self.pairs)


def summarize_disagreements(pairs: Iterable[ShadowPair]) -> dict[str, int]:
    """Counts by ``incumbent -> candidate`` transition, for the manual review pass."""
    counts: dict[str, int] = {}
    for pair in pairs:
        if pair.agrees:
            continue
        key = f"{pair.incumbent or 'abstain'} -> {pair.candidate or 'abstain'}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


__all__ = ["ShadowLog", "ShadowPair", "SpotCheck", "summarize_disagreements"]
