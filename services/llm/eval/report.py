"""FP16-vs-quant diff table.

    python -m services.llm.eval.report \\
        eval-results/sentiment.reference-fp16.json \\
        eval-results/sentiment.candidate-awq.json

Prints a markdown table and the Gate A verdict, and exits non-zero when the gate
fails — so it drops straight into a nightly job without extra glue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from .gates import gate_a
from .metrics import FieldScore, TaskMetrics

# (label, extractor, "higher is better")
ROWS: tuple[tuple[str, str, bool], ...] = (
    ("valid@1", "valid_at_1", True),
    ("field accuracy", "field_accuracy", True),
    ("numeric exactness", "numeric_exactness", True),
    ("abstain rate", "abstain_rate", False),
    ("false-confidence rate", "false_confidence_rate", False),
    ("p50 latency (s)", "p50_latency_s", False),
    ("p95 latency (s)", "p95_latency_s", False),
)


def load_metrics(path: Path) -> tuple[str, TaskMetrics]:
    """Rehydrate a :class:`TaskMetrics` from a harness result file."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    raw: dict[str, Any] = blob["metrics"]

    metrics = TaskMetrics(
        task=raw["task"],
        n=raw["n"],
        valid_at_1=raw["valid_at_1"],
        abstain_rate=raw["abstain_rate"],
        abstain_by_reason=raw.get("abstain_by_reason", {}),
        false_confidence_rate=raw["false_confidence_rate"],
        numeric_exactness=raw["numeric_exactness"],
        p50_latency_s=raw["p50_latency_s"],
        p95_latency_s=raw["p95_latency_s"],
        total_tokens=raw.get("total_tokens", 0),
    )
    # field_accuracy is a derived property, so rebuild the scores that produce
    # it rather than storing a number that could drift from its inputs.
    for name, entry in raw.get("fields", {}).items():
        metrics.field_scores[name] = FieldScore(
            name=name,
            kind=entry["kind"],
            compared=entry["compared"],
            override=entry["score"],
        )

    return blob.get("label", path.stem), metrics


def diff_table(
    reference_label: str,
    reference: TaskMetrics,
    candidate_label: str,
    candidate: TaskMetrics,
) -> str:
    lines = [
        f"### `{candidate.task}` — {candidate_label} vs {reference_label}",
        "",
        f"n = {candidate.n} (reference n = {reference.n})",
        "",
        f"| metric | {reference_label} | {candidate_label} | delta | |",
        "|---|---:|---:|---:|:--|",
    ]

    for label, attr, higher_better in ROWS:
        ref_value = getattr(reference, attr)
        cand_value = getattr(candidate, attr)
        delta = cand_value - ref_value
        good = delta >= 0 if higher_better else delta <= 0
        marker = "" if abs(delta) < 1e-9 else ("✓" if good else "▲")
        lines.append(
            f"| {label} | {ref_value:.3f} | {cand_value:.3f} | {delta:+.3f} | {marker} |"
        )

    per_field = sorted(
        set(reference.field_scores) | set(candidate.field_scores)
    )
    if per_field:
        lines.extend(["", "| field | reference | candidate | delta |", "|---|---:|---:|---:|"])
        for name in per_field:
            ref_score = reference.field_scores.get(name)
            cand_score = candidate.field_scores.get(name)
            ref_value = ref_score.score if ref_score else float("nan")
            cand_value = cand_score.score if cand_score else float("nan")
            lines.append(
                f"| {name} | {ref_value:.3f} | {cand_value:.3f} | {cand_value - ref_value:+.3f} |"
            )

    if candidate.abstain_by_reason:
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(candidate.abstain_by_reason.items())
        )
        lines.extend(["", f"Candidate abstentions: {reasons}"])

    return "\n".join(lines)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-delta", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    ref_label, reference = load_metrics(args.reference)
    cand_label, candidate = load_metrics(args.candidate)
    if reference.task != candidate.task:
        parser.error(
            f"task mismatch: reference is {reference.task}, candidate is {candidate.task}"
        )

    gate = gate_a(reference, candidate, max_delta=args.max_delta)
    body = "\n\n".join(
        [
            diff_table(ref_label, reference, cand_label, candidate),
            "```\n" + gate.report() + "\n```",
        ]
    )
    print(body)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body + "\n", encoding="utf-8")
    return 0 if gate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
