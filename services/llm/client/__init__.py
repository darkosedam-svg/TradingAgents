"""Client, routing, and budget primitives for the triage inference layer."""

from .advisory import AdvisoryEnricher, with_deadline
from .breaker import BreakerState, CircuitBreaker
from .budget import BudgetExceeded, BudgetLedger, Pricing, TaskBudget
from .client import LLMClient, UpstreamUnavailable, extract_json
from .config import LLMSettings, StructuredMode
from .observability import CallRecord, InMemoryMetrics, MetricsSink, NullMetrics
from .router import (
    EscalationDecision,
    EscalationPolicy,
    EscalationRateMonitor,
    FallbackBackend,
    Router,
)
from .schema_tools import json_schema_for, response_format_for, schema_hint
from .shadow import ShadowLog, ShadowPair, SpotCheck

__all__ = [
    "AdvisoryEnricher",
    "BreakerState",
    "BudgetExceeded",
    "BudgetLedger",
    "CallRecord",
    "CircuitBreaker",
    "EscalationDecision",
    "EscalationPolicy",
    "EscalationRateMonitor",
    "FallbackBackend",
    "InMemoryMetrics",
    "LLMClient",
    "LLMSettings",
    "MetricsSink",
    "NullMetrics",
    "Pricing",
    "Router",
    "ShadowLog",
    "ShadowPair",
    "SpotCheck",
    "StructuredMode",
    "TaskBudget",
    "UpstreamUnavailable",
    "extract_json",
    "json_schema_for",
    "response_format_for",
    "schema_hint",
    "with_deadline",
]
