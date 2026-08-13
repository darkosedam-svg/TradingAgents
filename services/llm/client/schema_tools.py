"""Turn a Pydantic model into a schema a provider's structured-output mode accepts.

Providers implement a *subset* of JSON Schema for constrained decoding, and the
subsets disagree. OpenAI-style strict mode is the fussiest: every property must
be listed in ``required``, every object must set ``additionalProperties: false``,
and validation keywords like ``minimum`` or ``maxLength`` are rejected outright.

Stripping those keywords is not a loss of safety, because it matches how this
layer already divides the work: the provider guarantees the response *parses
into the right shape*, and the Pydantic validators — which still run on every
response — guarantee the values are *in range and self-consistent*. A confidence
of 1.4 was always going to be caught by the validator, not by the decoder.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

# Validation keywords that constrained-decoding backends commonly reject. Range
# and length checks survive as Pydantic validators, so dropping them here costs
# nothing beyond a slightly wider grammar during decoding.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "format",
        "default",
    }
)


def _strip(node: Any, *, strict: bool) -> Any:
    """Recursively normalize one schema node."""
    if isinstance(node, list):
        return [_strip(item, strict=strict) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned = {
        key: _strip(value, strict=strict)
        for key, value in node.items()
        if key not in _UNSUPPORTED_KEYWORDS
    }

    if cleaned.get("type") == "object" or "properties" in cleaned:
        cleaned["additionalProperties"] = False
        if strict:
            # Strict mode has no notion of an optional field: everything must be
            # required. Pydantic defaults still apply on our side, and a model
            # that must emit every key is a model that cannot quietly omit one.
            properties = cleaned.get("properties") or {}
            cleaned["required"] = list(properties)

    return cleaned


def json_schema_for(model: Type[BaseModel], *, strict: bool = True) -> dict[str, Any]:
    """Provider-ready JSON Schema for ``model``.

    With ``strict`` the result satisfies OpenAI-style strict structured output.
    Without it the shape is preserved but optional fields stay optional, which
    is what vLLM's ``guided_json`` and most grammar backends want.
    """
    return _strip(model.model_json_schema(), strict=strict)


def response_format_for(model: Type[BaseModel], *, strict: bool = True) -> dict[str, Any]:
    """The ``response_format`` block for OpenAI-compatible structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": strict,
            "schema": json_schema_for(model, strict=strict),
        },
    }


def schema_hint(model: Type[BaseModel]) -> str:
    """A compact schema description for providers with no structured-output mode.

    Last resort. Appended to the system prompt so the model at least knows the
    contract; the guarantee then comes entirely from parsing plus validation,
    which is why ``prompt`` mode is expected to abstain more often.
    """
    schema = json_schema_for(model, strict=True)
    properties = schema.get("properties", {})
    lines = []
    for name, spec in properties.items():
        kind = spec.get("type") or ("enum" if "enum" in spec else "any")
        if "enum" in spec:
            kind = " | ".join(repr(v) for v in spec["enum"])
        elif kind == "array":
            items = spec.get("items", {})
            kind = f"array of {items.get('type', 'any')}"
        description = spec.get("description", "")
        lines.append(f"- {name} ({kind}){': ' + description if description else ''}")

    return (
        "Respond with a single JSON object and no other text. No markdown fences, "
        "no commentary. Fields:\n" + "\n".join(lines)
    )


__all__ = ["json_schema_for", "response_format_for", "schema_hint"]
