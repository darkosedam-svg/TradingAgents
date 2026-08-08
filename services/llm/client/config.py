"""Environment-driven configuration for the local inference layer.

Every knob has a working default so ``LLMSettings()`` is usable in tests without
touching the environment. Nothing here reads a secret: the local endpoint is
unauthenticated on ``localhost`` and the hosted fallback takes its key from the
caller's existing provider configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Task-level minimum confidence. Below this the layer abstains with
# LOW_CONFIDENCE rather than voting. Tuned per task because the cost of a wrong
# answer differs: a bad sentiment vote is 15% of one consensus, a bad parsed
# signal is a wrong price.
DEFAULT_MIN_CONFIDENCE: dict[str, float] = {
    "sentiment": 0.55,
    "news_triage": 0.40,  # low on purpose — the router wants recall, not certainty
    "signal_parse": 0.70,
    "token_meta": 0.50,
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class LLMSettings:
    """Serving and policy configuration, resolved from the environment."""

    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
    )
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "EMPTY"))

    # Hard timeout. Cold start is 20-40s and must never happen mid-decision, so
    # this is deliberately shorter than a cold start: a timeout here means the
    # container is not warm, and the circuit breaker should route around it.
    timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 8.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 1))

    # Circuit breaker: trip to hosted after N consecutive failures, retry the
    # local endpoint once the cooldown expires.
    breaker_threshold: int = field(
        default_factory=lambda: _env_int("LLM_BREAKER_THRESHOLD", 3)
    )
    breaker_cooldown_s: float = field(
        default_factory=lambda: _env_float("LLM_BREAKER_COOLDOWN_S", 30.0)
    )

    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 256))
    temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0)
    )

    min_confidence: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MIN_CONFIDENCE)
    )

    def confidence_floor(self, task: str) -> float:
        return self.min_confidence.get(task, 0.5)
