"""Pump.fun narrative enrichment (Phase 5) — **advisory only**.

Nothing in this schema is an execution gate. The deterministic checks (mint and
freeze authority, holder concentration, LP status) remain the sole gate; these
fields attach to the candidate record for scoring and post-hoc analysis, and the
entry decision proceeds unchanged if they never arrive.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .base import TaskOutput

ScamFlag = Literal[
    "impersonation",
    "guaranteed_returns",
    "urgency_pressure",
    "fake_partnership",
    "team_anonymous",
    "recycled_description",
    "contact_bait",
]


class TokenNarrativeFlags(TaskOutput):
    """What a launch is *about*, and how much it smells like a copy or a scam."""

    narrative_cluster: str = Field(
        ...,
        max_length=40,
        description="Short slug for the theme, e.g. 'dog-meme', 'ai-agent', 'politics'.",
    )
    copycat_likelihood: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How derivative this launch looks versus the named references.",
    )
    scam_flags: list[ScamFlag] = Field(default_factory=list, max_length=7)

    @field_validator("narrative_cluster")
    @classmethod
    def _slugify(cls, cluster: str) -> str:
        slug = cluster.strip().lower().replace(" ", "-").replace("_", "-")
        if not slug:
            raise ValueError("narrative_cluster must be non-empty")
        return slug

    @field_validator("scam_flags")
    @classmethod
    def _dedupe(cls, flags: list[str]) -> list[str]:
        seen: list[str] = []
        for flag in flags:
            if flag not in seen:
                seen.append(flag)
        return seen
