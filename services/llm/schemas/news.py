"""News triage — the input to the escalation router (Phase 4).

``escalate`` is the commercially load-bearing field: it decides which of a few
hundred candidates is worth an expensive hosted call. The router tunes its
threshold for recall, so this schema keeps ``priority`` continuous rather than
letting the model hand back a single opaque boolean.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base import TaskOutput

Direction = Literal["yes", "no", "unclear"]


class NewsTriage(TaskOutput):
    """Whether a piece of news moves any tracked market, and how urgently."""

    relevance: float = Field(
        ..., ge=0.0, le=1.0, description="How strongly this bears on the listed markets."
    )
    market_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="IDs drawn from the candidate list in the prompt. Never invented.",
    )
    direction: Direction = Field(
        ..., description="Which side of the listed markets this favours."
    )
    escalate: bool = Field(
        ..., description="Model's own recommendation that a frontier model look at this."
    )
    priority: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Continuous urgency score; the router thresholds on this, not on `escalate`.",
    )
    rationale: str = Field(default="", max_length=400)

    @field_validator("market_ids")
    @classmethod
    def _dedupe(cls, ids: list[str]) -> list[str]:
        seen: list[str] = []
        for raw in ids:
            market_id = raw.strip()
            if market_id and market_id not in seen:
                seen.append(market_id)
        return seen

    @model_validator(mode="after")
    def _direction_needs_a_market(self) -> "NewsTriage":
        if self.direction != "unclear" and not self.market_ids:
            raise ValueError("a directional call requires at least one market_id")
        return self
