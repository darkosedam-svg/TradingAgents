"""Sentiment specialist output schema (Phase 3 consumer, 15% consensus vote)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .base import TaskOutput

Direction = Literal["bullish", "bearish", "neutral"]


class SentimentVote(TaskOutput):
    """A directional read on one headline/article, plus the assets it touches."""

    sentiment: Direction = Field(
        ..., description="Directional read on the assets named in `assets`."
    )
    assets: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Tickers/symbols the text is actually about, uppercase.",
    )
    rationale: str = Field(
        default="",
        max_length=400,
        description="One or two sentences grounded in the text, no speculation.",
    )

    @field_validator("assets")
    @classmethod
    def _normalize_assets(cls, assets: list[str]) -> list[str]:
        """Uppercase, strip a leading ``$``, drop blanks, de-duplicate in order.

        Symbol formatting is the single most common cosmetic disagreement
        between a quantized model and its FP16 reference; normalizing here keeps
        it out of the field-accuracy numbers so real regressions stay visible.
        """
        seen: list[str] = []
        for raw in assets:
            symbol = raw.strip().lstrip("$").upper()
            if symbol and symbol not in seen:
                seen.append(symbol)
        return seen

    @field_validator("assets")
    @classmethod
    def _reject_prose(cls, assets: list[str]) -> list[str]:
        """A ticker is a short token. Anything longer is the model narrating."""
        for symbol in assets:
            if len(symbol) > 15 or " " in symbol:
                raise ValueError(f"not a ticker: {symbol!r}")
        return assets
