"""What the system decided, and what actually happened.

The load-bearing property of :class:`Decision` is what it does **not** contain:
no order type, no venue, no API credentials, no execution timing. A decision is
a statement of intent — *this instrument, this direction, this confidence, for
this reason*. Something else turns it into an action.

That separation is why adding automated execution later is an adapter rather
than a rewrite. Today the "something else" formats a message for a human. The
day it becomes a broker call, none of the reasoning code changes.

Stdlib only, deliberately: this must be usable before any dependency is
installed and from any process that wants to read the journal.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class Domain(str, Enum):
    """Which market a decision belongs to.

    Kept explicit rather than inferred from the instrument, because the whole
    architecture rests on per-domain pipelines that stay separate — the
    aggregator needs to know which pipeline spoke without parsing a ticker.
    """

    CRYPTO = "crypto"
    EQUITY = "equity"
    PREDICTION = "prediction"
    MEME = "meme"


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"  # an explicit "no position" — different from no decision at all


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Decision:
    """One statement of intent, from one pipeline, at one moment.

    ``confidence`` is the system's own probability that this call is right. It
    is recorded rather than acted on directly, so that calibration can be
    measured later: a system that says 0.8 should be right about 80% of the
    time, and you cannot discover that it isn't unless you wrote the number
    down at the time.
    """

    domain: Domain
    instrument: str
    side: Side
    confidence: float
    rationale: str

    # Which strategy produced this, and which trial of that strategy. The trial
    # id is what makes the overfitting correction in `trials.py` possible — a
    # decision that cannot be traced to a numbered attempt cannot be deflated.
    strategy_id: str = "unassigned"
    trial_id: Optional[str] = None

    # Signal sources that contributed, for per-source attribution later.
    sources: tuple[str, ...] = ()

    # Suggested size as a fraction of whatever the consumer considers its unit
    # of risk. Advisory: the aggregator applies the real limits.
    size_fraction: float = 0.0

    # How long this call is meant to be judged over, in seconds. Without it,
    # "was it right?" has no defined answer.
    horizon_s: Optional[int] = None

    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(default_factory=_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0..1, got {self.confidence}")
        if not 0.0 <= self.size_fraction <= 1.0:
            raise ValueError(f"size_fraction must be 0..1, got {self.size_fraction}")
        if not self.instrument.strip():
            raise ValueError("instrument must not be empty")
        if self.horizon_s is not None and self.horizon_s <= 0:
            raise ValueError(f"horizon_s must be positive, got {self.horizon_s}")
        if self.side is Side.FLAT and self.size_fraction != 0.0:
            raise ValueError("a FLAT decision cannot carry a non-zero size")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["domain"] = self.domain.value
        payload["side"] = self.side.value
        payload["sources"] = list(self.sources)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Decision":
        data = dict(payload)
        data.pop("kind", None)
        data["domain"] = Domain(data["domain"])
        data["side"] = Side(data["side"])
        data["sources"] = tuple(data.get("sources", ()))
        return cls(**data)

    def with_trial(self, trial_id: str) -> "Decision":
        return replace(self, trial_id=trial_id)


@dataclass(frozen=True)
class Realisation:
    """What the market did, linked back to the decision that predicted it.

    Recorded separately and later — never at decision time — because anything
    written at decision time cannot contain the outcome without look-ahead.
    """

    decision_id: str
    realised_return: float
    ts: str = field(default_factory=_now)
    notes: str = ""

    @property
    def direction(self) -> Side:
        """Which way it actually went."""
        if self.realised_return > 0:
            return Side.LONG
        if self.realised_return < 0:
            return Side.SHORT
        return Side.FLAT

    def scores(self, decision: Decision) -> bool:
        """Was the decision right?

        A FLAT call is correct when the move was negligible, which is a
        judgement the caller controls by choosing the flat band; the default
        here is exact zero, so FLAT is almost never scored correct by accident.
        """
        if decision.side is Side.FLAT:
            return self.realised_return == 0.0
        return decision.side is self.direction

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Realisation":
        data = dict(payload)
        data.pop("kind", None)
        return cls(**data)


__all__ = ["Decision", "Domain", "Realisation", "Side"]
