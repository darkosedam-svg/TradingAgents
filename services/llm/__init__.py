"""Local quantized inference layer for warm-path agents.

Serves sentiment, news/text triage, structured extraction, and escalation
routing from a local vLLM endpoint, with a hosted API as the fallback path.
Never enters an execution lane; see ``README.md`` for the non-goals that make
that a property of the code rather than a promise.
"""

from . import prompts
from .client import (
    AdvisoryEnricher,
    BudgetLedger,
    CircuitBreaker,
    EscalationPolicy,
    InMemoryMetrics,
    LLMClient,
    LLMSettings,
    LocalUnavailable,
    Router,
)
from .schemas import (
    AbstainReason,
    NewsTriage,
    Outcome,
    ParsedSignal,
    SentimentVote,
    TokenNarrativeFlags,
)

__all__ = [
    "AbstainReason",
    "AdvisoryEnricher",
    "BudgetLedger",
    "CircuitBreaker",
    "EscalationPolicy",
    "InMemoryMetrics",
    "LLMClient",
    "LLMSettings",
    "LocalUnavailable",
    "NewsTriage",
    "Outcome",
    "ParsedSignal",
    "Router",
    "SentimentVote",
    "TokenNarrativeFlags",
    "prompts",
]
