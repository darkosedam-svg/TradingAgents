"""Structured extraction of trade calls from TG/Discord chatter.

The golden-set metric that governs this schema is *numeric exactness*: a wrong
price is worse than no price. The validators below therefore reject
geometrically impossible signals outright — an inconsistent stop is a
``SCHEMA_FAIL`` abstention, not a signal with one bad field.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from .base import TaskOutput

Side = Literal["long", "short"]


class ParsedSignal(TaskOutput):
    """A trade call extracted from free text, or an explicit "no signal here"."""

    valid: bool = Field(
        ...,
        description="False for hype, commentary, or anything without an actionable call.",
    )
    asset: Optional[str] = Field(default=None, max_length=15)
    side: Optional[Side] = None
    entry: Optional[float] = Field(default=None, gt=0)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: list[float] = Field(default_factory=list, max_length=6)
    leverage: Optional[float] = Field(default=None, gt=0, le=125)

    @model_validator(mode="after")
    def _check_shape(self) -> "ParsedSignal":
        if not self.valid:
            # A negative must be clean: no half-extracted numbers riding along.
            populated = [
                name
                for name in ("asset", "side", "entry", "stop_loss", "leverage")
                if getattr(self, name) is not None
            ]
            if populated or self.take_profit:
                raise ValueError(
                    f"valid=false must carry no trade fields, got {populated + ['take_profit'] * bool(self.take_profit)}"
                )
            return self

        missing = [
            name for name in ("asset", "side") if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"valid=true requires {missing}")

        self.asset = (self.asset or "").strip().lstrip("$").upper()
        if not self.asset:
            raise ValueError("valid=true requires a non-empty asset")

        for price in self.take_profit:
            if price <= 0:
                raise ValueError(f"take_profit must be positive, got {price}")

        return self

    @model_validator(mode="after")
    def _check_geometry(self) -> "ParsedSignal":
        """Stops below entry for longs, above for shorts; targets the reverse.

        Models that hallucinate prices tend to get the *ordering* wrong before
        they get the magnitude wrong, which makes this a cheap and surprisingly
        sharp detector.
        """
        if not self.valid or self.entry is None:
            return self

        long = self.side == "long"

        if self.stop_loss is not None:
            if long and self.stop_loss >= self.entry:
                raise ValueError(
                    f"long stop {self.stop_loss} must sit below entry {self.entry}"
                )
            if not long and self.stop_loss <= self.entry:
                raise ValueError(
                    f"short stop {self.stop_loss} must sit above entry {self.entry}"
                )

        for target in self.take_profit:
            if long and target <= self.entry:
                raise ValueError(
                    f"long target {target} must sit above entry {self.entry}"
                )
            if not long and target >= self.entry:
                raise ValueError(
                    f"short target {target} must sit below entry {self.entry}"
                )

        return self
