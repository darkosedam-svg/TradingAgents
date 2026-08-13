"""Cheap-model triage layer for warm-path agents.

Serves sentiment, news/text triage, structured extraction, and escalation
routing from a small, cheap model over an OpenAI-compatible endpoint, reserving
frontier calls for what survives triage. Never enters an execution lane; see
``README.md`` for the non-goals that make that a property of the code rather
than a promise.
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
    Pricing,
    Router,
    UpstreamUnavailable,
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
    "NewsTriage",
    "Outcome",
    "ParsedSignal",
    "Pricing",
    "Router",
    "SentimentVote",
    "TokenNarrativeFlags",
    "UpstreamUnavailable",
    "prompts",
]
