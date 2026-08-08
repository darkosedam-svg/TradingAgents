"""Client, routing, and budget primitives for the local inference layer."""

from .advisory import AdvisoryEnricher, with_deadline
from .breaker import BreakerState, CircuitBreaker
from .budget import BudgetExceeded, BudgetLedger, TaskBudget
from .client import LLMClient, LocalUnavailable
from .config import LLMSettings
from .observability import CallRecord, InMemoryMetrics, MetricsSink, NullMetrics
from .router import (
    EscalationDecision,
    EscalationPolicy,
    EscalationRateMonitor,
    HostedBackend,
    Router,
)
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
    "HostedBackend",
    "InMemoryMetrics",
    "LLMClient",
    "LLMSettings",
    "LocalUnavailable",
    "MetricsSink",
    "NullMetrics",
    "Router",
    "ShadowLog",
    "ShadowPair",
    "SpotCheck",
    "TaskBudget",
    "with_deadline",
]
