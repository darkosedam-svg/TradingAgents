"""Shared primitives for every task schema in the local-inference layer.

The layer's central safety property is *abstention*: a quantized model degrades
on instruction-following before it degrades on fluency, so the dangerous failure
mode is a confidently malformed or confidently wrong answer. Every task result
therefore passes through :class:`Outcome`, which is either a validated value or
an abstention carrying one of three reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class AbstainReason(str, Enum):
    """Why a task produced a non-vote.

    ``INSUFFICIENT_DATA`` is the pre-existing guardrail: the model was asked
    about something the input does not support an answer for. The two siblings
    are what quantization adds:

    ``LOW_CONFIDENCE`` — the response validated but the model's own confidence
    is below the task threshold.

    ``SCHEMA_FAIL`` — the response did not parse, or parsed but failed a
    Pydantic validator (an out-of-range value, an inconsistent price geometry).

    All three resolve to "abstain", never to a default vote.
    """

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    SCHEMA_FAIL = "SCHEMA_FAIL"


class TaskOutput(BaseModel):
    """Base class for the JSON body every task is constrained to emit.

    ``extra="forbid"`` is deliberate. Guided decoding keeps the response
    parseable, but a model that invents a plausible extra key is a model that
    has drifted from the prompt contract, and we want that surfaced as a
    ``SCHEMA_FAIL`` abstention rather than silently dropped.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's own confidence in this answer, 0.0-1.0.",
    )
    insufficient_data: bool = Field(
        default=False,
        description="True when the input does not support an answer at all.",
    )

    def abstain_reason(self, min_confidence: float) -> Optional[AbstainReason]:
        """Return the abstention reason for this output, or ``None`` to vote."""
        if self.insufficient_data:
            return AbstainReason.INSUFFICIENT_DATA
        if self.confidence < min_confidence:
            return AbstainReason.LOW_CONFIDENCE
        return None


@dataclass(frozen=True)
class Outcome(Generic[T]):
    """A task result: either a validated value, or an abstention.

    Never both, never neither. ``value`` is ``None`` exactly when
    ``reason`` is set.
    """

    value: Optional[T]
    reason: Optional[AbstainReason]
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    prompt_sha: str = ""
    source: str = "local"
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise ValueError(
                "Outcome must carry exactly one of value or reason, "
                f"got value={self.value!r} reason={self.reason!r}"
            )

    @property
    def abstained(self) -> bool:
        return self.reason is not None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def voted(cls, value: T, **kwargs: Any) -> "Outcome[T]":
        return cls(value=value, reason=None, **kwargs)

    @classmethod
    def abstain(cls, reason: AbstainReason, **kwargs: Any) -> "Outcome[T]":
        return cls(value=None, reason=reason, **kwargs)
