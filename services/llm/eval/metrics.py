"""Scoring for golden-set runs.

The metric that governs the whole layer is ``false_confidence_rate``: how often
the model votes, above its confidence floor, and is wrong. Everything else here
is diagnostic. A quantized model that abstains more than its reference is
tolerable; one that is confidently wrong more often is not, and Gate A is
written to reject exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence

from ..client.observability import percentile

FieldKind = Literal["categorical", "numeric", "set", "bool", "text"]


@dataclass(frozen=True)
class FieldSpec:
    """How one output key is compared against its label."""

    name: str
    kind: FieldKind
    # Relative tolerance for numeric comparison. 0.0 means bit-for-bit equality,
    # which is what we want for prices copied out of a message.
    tolerance: float = 0.0
    # Only compare this field when the gating field is truthy in the label —
    # e.g. don't score `entry` on messages that carry no signal at all.
    gated_on: Optional[str] = None


@dataclass(frozen=True)
class TaskSpec:
    """Everything scoring needs to know about a task."""

    task: str
    fields: tuple[FieldSpec, ...]
    # The field a "wrong answer" is judged on for false-confidence purposes.
    primary_field: str

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


TASK_SPECS: dict[str, TaskSpec] = {
    "sentiment": TaskSpec(
        task="sentiment",
        primary_field="sentiment",
        fields=(
            FieldSpec("sentiment", "categorical"),
            FieldSpec("assets", "set"),
            FieldSpec("insufficient_data", "bool"),
            FieldSpec("rationale", "text"),
        ),
    ),
    "news_triage": TaskSpec(
        task="news_triage",
        primary_field="escalate",
        fields=(
            FieldSpec("escalate", "bool"),
            FieldSpec("direction", "categorical"),
            FieldSpec("market_ids", "set"),
            FieldSpec("relevance", "numeric", tolerance=0.25),
            FieldSpec("rationale", "text"),
        ),
    ),
    "signal_parse": TaskSpec(
        task="signal_parse",
        primary_field="valid",
        fields=(
            FieldSpec("valid", "bool"),
            FieldSpec("asset", "categorical", gated_on="valid"),
            FieldSpec("side", "categorical", gated_on="valid"),
            # Zero tolerance on purpose: a wrong price is worse than no price.
            FieldSpec("entry", "numeric", gated_on="valid"),
            FieldSpec("stop_loss", "numeric", gated_on="valid"),
            FieldSpec("take_profit", "set", gated_on="valid"),
            FieldSpec("leverage", "numeric", gated_on="valid"),
        ),
    ),
    "token_meta": TaskSpec(
        task="token_meta",
        primary_field="narrative_cluster",
        fields=(
            FieldSpec("narrative_cluster", "categorical"),
            FieldSpec("copycat_likelihood", "numeric", tolerance=0.30),
            FieldSpec("scam_flags", "set"),
        ),
    ),
}


@dataclass
class RunItem:
    """One golden item, run once, scored later."""

    item_id: str
    expected: dict[str, Any]
    predicted: Optional[dict[str, Any]] = None
    valid: bool = False  # parsed AND survived the Pydantic validators
    abstained: bool = False
    abstain_reason: Optional[str] = None
    confidence: float = 0.0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


@dataclass
class FieldScore:
    name: str
    kind: FieldKind
    compared: int = 0
    correct: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # Set when a score is rehydrated from a saved report rather than computed
    # from counts — the tp/fp/fn triple behind an F1 is not recoverable from the
    # F1 alone, so we carry the number instead of inventing counts for it.
    override: Optional[float] = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.compared if self.compared else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / denom if denom else 1.0

    @property
    def score(self) -> float:
        """The single number this field contributes to field accuracy."""
        if self.override is not None:
            return self.override
        return self.f1 if self.kind == "set" else self.accuracy


@dataclass
class TaskMetrics:
    task: str
    n: int = 0
    valid_at_1: float = 0.0
    abstain_rate: float = 0.0
    abstain_by_reason: dict[str, int] = field(default_factory=dict)
    false_confidence_rate: float = 0.0
    numeric_exactness: float = 1.0
    field_scores: dict[str, FieldScore] = field(default_factory=dict)
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    total_tokens: int = 0

    @property
    def field_accuracy(self) -> float:
        """Macro mean over every scorable field. This is Gate A's headline number."""
        scores = [s.score for s in self.field_scores.values() if s.compared]
        return sum(scores) / len(scores) if scores else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "n": self.n,
            "valid_at_1": round(self.valid_at_1, 4),
            "field_accuracy": round(self.field_accuracy, 4),
            "numeric_exactness": round(self.numeric_exactness, 4),
            "abstain_rate": round(self.abstain_rate, 4),
            "abstain_by_reason": self.abstain_by_reason,
            "false_confidence_rate": round(self.false_confidence_rate, 4),
            "p50_latency_s": round(self.p50_latency_s, 4),
            "p95_latency_s": round(self.p95_latency_s, 4),
            "total_tokens": self.total_tokens,
            "fields": {
                name: {
                    "kind": s.kind,
                    "compared": s.compared,
                    "score": round(s.score, 4),
                }
                for name, s in self.field_scores.items()
            },
        }


def _as_set(value: Any) -> set[Any]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {v for v in value}
    return {value}


def _numeric_equal(expected: Any, predicted: Any, tolerance: float) -> bool:
    if expected is None or predicted is None:
        return expected is None and predicted is None
    try:
        exp, pred = float(expected), float(predicted)
    except (TypeError, ValueError):
        return False
    if tolerance <= 0:
        return exp == pred
    return abs(exp - pred) <= tolerance * max(abs(exp), 1e-9)


def _compare(spec: FieldSpec, expected: Any, predicted: Any, score: FieldScore) -> Optional[bool]:
    """Update ``score`` in place; return whether the field matched, or None if skipped."""
    if spec.kind == "text":
        return None  # free text is not scored — that is what the human review is for

    if spec.kind == "set":
        exp_set, pred_set = _as_set(expected), _as_set(predicted)
        tp = len(exp_set & pred_set)
        score.compared += 1
        score.tp += tp
        score.fp += len(pred_set - exp_set)
        score.fn += len(exp_set - pred_set)
        matched = exp_set == pred_set
        score.correct += int(matched)
        return matched

    if spec.kind == "numeric":
        matched = _numeric_equal(expected, predicted, spec.tolerance)
    elif spec.kind == "bool":
        matched = bool(expected) == bool(predicted)
    else:  # categorical
        exp = expected.upper() if isinstance(expected, str) else expected
        pred = predicted.upper() if isinstance(predicted, str) else predicted
        matched = exp == pred

    score.compared += 1
    score.correct += int(matched)
    return matched


def score_run(
    items: Sequence[RunItem],
    spec: TaskSpec,
    *,
    confidence_floor: float = 0.5,
) -> TaskMetrics:
    """Turn a list of run records into the metric set the gates are written against."""
    metrics = TaskMetrics(task=spec.task, n=len(items))
    if not items:
        return metrics

    metrics.field_scores = {f.name: FieldScore(f.name, f.kind) for f in spec.fields}

    abstain_counts: dict[str, int] = {}
    confident_votes = 0
    confident_wrong = 0
    numeric_compared = 0
    numeric_correct = 0

    for item in items:
        if item.abstained:
            reason = item.abstain_reason or "UNKNOWN"
            abstain_counts[reason] = abstain_counts.get(reason, 0) + 1
            continue
        if not item.valid or item.predicted is None:
            continue

        primary_ok: Optional[bool] = None
        for field_spec in spec.fields:
            if field_spec.gated_on and not item.expected.get(field_spec.gated_on):
                continue
            matched = _compare(
                field_spec,
                item.expected.get(field_spec.name),
                item.predicted.get(field_spec.name),
                metrics.field_scores[field_spec.name],
            )
            if matched is None:
                continue
            if field_spec.kind == "numeric":
                numeric_compared += 1
                numeric_correct += int(matched)
            if field_spec.name == spec.primary_field:
                primary_ok = matched

        if item.confidence >= confidence_floor:
            confident_votes += 1
            if primary_ok is False:
                confident_wrong += 1

    metrics.valid_at_1 = sum(1 for i in items if i.valid) / len(items)
    metrics.abstain_rate = sum(1 for i in items if i.abstained) / len(items)
    metrics.abstain_by_reason = abstain_counts
    metrics.false_confidence_rate = (
        confident_wrong / confident_votes if confident_votes else 0.0
    )
    metrics.numeric_exactness = (
        numeric_correct / numeric_compared if numeric_compared else 1.0
    )

    latencies = [i.latency_s for i in items]
    metrics.p50_latency_s = percentile(latencies, 0.50)
    metrics.p95_latency_s = percentile(latencies, 0.95)
    metrics.total_tokens = sum(i.prompt_tokens + i.completion_tokens for i in items)
    return metrics


def agreement(a: Iterable[Optional[str]], b: Iterable[Optional[str]]) -> float:
    """Direction-agreement rate between two vote streams (Gate B).

    A pair where either side abstained counts as a disagreement only if the
    other side voted — two abstentions agree, which is the behaviour we want
    when comparing a cautious local model to a cautious hosted one.
    """
    pairs = list(zip(a, b))
    if not pairs:
        return 0.0
    both_abstained = sum(1 for x, y in pairs if x is None and y is None)
    matched = sum(1 for x, y in pairs if x is not None and x == y)
    return (matched + both_abstained) / len(pairs)


__all__ = [
    "FieldScore",
    "FieldSpec",
    "RunItem",
    "TASK_SPECS",
    "TaskMetrics",
    "TaskSpec",
    "agreement",
    "score_run",
]
