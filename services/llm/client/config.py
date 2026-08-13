"""Environment-driven configuration for the triage inference layer.

Every knob has a working default so ``LLMSettings()`` is usable in tests without
touching the environment. The endpoint is an OpenAI-compatible HTTP API — a
cheap hosted model by default, but a self-hosted vLLM/Ollama endpoint works
unchanged by pointing ``LLM_BASE_URL`` at it and switching ``LLM_STRUCTURED_MODE``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# How the provider is asked to constrain its output.
#
#   json_schema  OpenAI-compatible `response_format: {"type": "json_schema"}`.
#                The default: supported by OpenAI, OpenRouter (for models that
#                advertise it), and most compatible gateways.
#   guided_json  vLLM's extension. Use when self-hosting.
#   prompt       No constraint. The schema goes in the system prompt and the
#                response is parsed defensively. Expect a higher SCHEMA_FAIL
#                abstention rate — which is the correct failure, not a silent one.
StructuredMode = Literal["json_schema", "guided_json", "prompt"]

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


def _env(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None and raw.strip() != "" else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _default_api_key() -> str:
    """First key that is actually set, so an existing .env just works.

    A self-hosted endpoint needs no key at all; ``EMPTY`` is the conventional
    placeholder such servers accept.
    """
    for name in ("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return "EMPTY"


@dataclass
class LLMSettings:
    """Endpoint and policy configuration, resolved from the environment."""

    base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )
    # The triage tier: small, cheap, fast. Never the decision tier. Confirm the
    # exact slug and current price on your provider before committing to one —
    # model availability moves faster than this file does.
    model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "qwen/qwen-2.5-7b-instruct")
    )
    api_key: str = field(default_factory=_default_api_key)
    structured_mode: StructuredMode = field(
        default_factory=lambda: _env("LLM_STRUCTURED_MODE", "json_schema")  # type: ignore[return-value]
    )

    # Hard timeout. Generous compared to a loopback endpoint because this now
    # crosses the internet, but still well inside every warm-path window.
    timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 20.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 1))

    # Circuit breaker: trip to the fallback provider after N consecutive
    # failures, probe the primary again once the cooldown expires.
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

    # Optional HTTP headers. OpenRouter uses HTTP-Referer and X-Title for
    # attribution; harmless everywhere else.
    app_url: str = field(default_factory=lambda: _env("LLM_APP_URL", ""))
    app_title: str = field(default_factory=lambda: _env("LLM_APP_TITLE", ""))

    min_confidence: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MIN_CONFIDENCE)
    )

    def confidence_floor(self, task: str) -> float:
        return self.min_confidence.get(task, 0.5)

    def headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers
