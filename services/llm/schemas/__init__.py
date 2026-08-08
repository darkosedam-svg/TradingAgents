"""Pydantic contracts for every task the local model serves.

Guided decoding (XGrammar) guarantees the response *parses*. These validators
guarantee the values are *in range and self-consistent*. Neither guarantees the
values are *right* — that is what ``services/llm/eval`` is for.
"""

from .base import AbstainReason, Outcome, TaskOutput
from .news import NewsTriage
from .sentiment import SentimentVote
from .signal import ParsedSignal
from .token_meta import TokenNarrativeFlags

__all__ = [
    "AbstainReason",
    "NewsTriage",
    "Outcome",
    "ParsedSignal",
    "SentimentVote",
    "TaskOutput",
    "TokenNarrativeFlags",
]
