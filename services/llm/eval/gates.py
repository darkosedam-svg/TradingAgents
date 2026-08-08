"""The four gates, as executable checks rather than prose.

Each gate is a hard stop. The point of encoding them here is that "we passed
Gate A" becomes a thing you run, not a thing you remember deciding, and the
failure message tells you which specific number blocked you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .metrics import TaskMetrics


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class GateResult:
    gate: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def report(self) -> str:
        header = f"Gate {self.gate}: {'PASS' if self.passed else 'FAIL'}"
        return "\n".join([header, *(f"  {c}" for c in self.checks)])


def gate_a(
    reference: TaskMetrics,
    candidate: TaskMetrics,
    *,
    max_delta: float = 0.02,
    min_valid_at_1: float = 0.99,
    false_confidence_slack: float = 0.0,
) -> GateResult:
    """Quant-vs-reference parity on a golden set.

    Failure means step *up* in precision (INT8/FP8) or drop to a smaller model
    at higher precision. It never means stepping down to 3-bit, and it never
    means widening ``max_delta``.

    ``false_confidence_slack`` defaults to 0.0, which is the criterion as
    written: the candidate may not be confidently wrong more often than the
    reference, full stop. Be aware of what that costs on a 200-item set — one
    extra confident error moves the rate by 0.005, so a candidate that is
    genuinely equivalent can fail on sampling noise. Raising the slack is a
    decision to accept a real, quantified increase in confident errors, so make
    it deliberately and write down the number; do not reach for it because a
    run came back red.
    """
    result = GateResult(gate=f"A ({candidate.task})")

    field_delta = reference.field_accuracy - candidate.field_accuracy
    result.checks.append(
        Check(
            "field accuracy",
            field_delta <= max_delta,
            f"reference {reference.field_accuracy:.4f} vs candidate "
            f"{candidate.field_accuracy:.4f} (delta {field_delta:+.4f}, budget {max_delta:.4f})",
        )
    )

    numeric_delta = reference.numeric_exactness - candidate.numeric_exactness
    result.checks.append(
        Check(
            "numeric exactness",
            numeric_delta <= max_delta,
            f"reference {reference.numeric_exactness:.4f} vs candidate "
            f"{candidate.numeric_exactness:.4f} (delta {numeric_delta:+.4f})",
        )
    )

    result.checks.append(
        Check(
            "valid@1",
            candidate.valid_at_1 >= min_valid_at_1,
            f"{candidate.valid_at_1:.4f} (floor {min_valid_at_1:.2f})",
        )
    )

    result.checks.append(
        Check(
            "false-confidence rate",
            candidate.false_confidence_rate
            <= reference.false_confidence_rate + false_confidence_slack + 1e-9,
            f"candidate {candidate.false_confidence_rate:.4f} vs reference "
            f"{reference.false_confidence_rate:.4f} (slack {false_confidence_slack:.4f})",
        )
    )

    result.checks.append(
        Check(
            "sample size",
            candidate.n >= 150,
            f"{candidate.n} items (golden sets are specified at 150-300)",
        )
    )
    return result


def gate_b(
    *,
    agreement_rate: float,
    paired_decisions: int,
    malformed_propagated: int,
    reviewed_disagreements: int = 0,
    systematically_worse: bool = False,
    min_agreement: float = 0.90,
    min_pairs: int = 500,
) -> GateResult:
    """Shadow-mode parity for the sentiment specialist before it votes."""
    result = GateResult(gate="B (sentiment shadow)")
    result.checks.append(
        Check(
            "direction agreement",
            agreement_rate >= min_agreement,
            f"{agreement_rate:.4f} (floor {min_agreement:.2f})",
        )
    )
    result.checks.append(
        Check(
            "paired decisions",
            paired_decisions >= min_pairs,
            f"{paired_decisions} (floor {min_pairs})",
        )
    )
    result.checks.append(
        Check(
            "no malformed response would have voted",
            malformed_propagated == 0,
            f"{malformed_propagated} case(s) where a malformed response reached a vote",
        )
    )
    result.checks.append(
        Check(
            "disagreements reviewed",
            reviewed_disagreements > 0 or agreement_rate >= 1.0,
            f"{reviewed_disagreements} disagreement(s) manually reviewed",
        )
    )
    result.checks.append(
        Check(
            "local not systematically worse",
            not systematically_worse,
            "manual review verdict on the disagreement set",
        )
    )
    return result


def gate_c(
    *,
    recall: float,
    hosted_call_reduction: float,
    labelled_positives: int = 0,
    min_recall: float = 0.95,
    min_reduction: float = 0.60,
) -> GateResult:
    """The escalation router, against a labelled "should have escalated" set.

    This is the first gate where the layer touches money-adjacent logic. Hold
    the recall number strictly: precision can be bought back later by raising
    the threshold, but a missed escalation is gone.
    """
    result = GateResult(gate="C (escalation router)")
    result.checks.append(
        Check(
            "recall on should-escalate set",
            recall >= min_recall,
            f"{recall:.4f} (floor {min_recall:.2f})",
        )
    )
    result.checks.append(
        Check(
            "hosted call reduction vs Phase 0 baseline",
            hosted_call_reduction >= min_reduction,
            f"{hosted_call_reduction:.4f} (floor {min_reduction:.2f})",
        )
    )
    result.checks.append(
        Check(
            "labelled positives",
            labelled_positives >= 50,
            f"{labelled_positives} labelled should-escalate items "
            "(too few and the recall number is noise)",
        )
    )
    return result


def gate_d(
    *,
    replayed_launches: int,
    timing_identical: bool,
    enforced_in_code: bool,
    min_launches: int = 25,
) -> GateResult:
    """Proof the narrative feature is genuinely off the entry path.

    ``timing_identical`` comes from replaying launches with the LLM feature
    removed entirely and comparing entry timing byte-for-byte. If it differs,
    the wiring is wrong — not the threshold.
    """
    result = GateResult(gate="D (Pump.fun advisory isolation)")
    result.checks.append(
        Check(
            "entry timing unchanged with feature removed",
            timing_identical,
            "byte-identical entry timing across replayed launches"
            if timing_identical
            else "entry timing differs — the advisory feature is on the entry path",
        )
    )
    result.checks.append(
        Check(
            "replay sample",
            replayed_launches >= min_launches,
            f"{replayed_launches} launches replayed (floor {min_launches})",
        )
    )
    result.checks.append(
        Check(
            "deadline enforced in code",
            enforced_in_code,
            "asyncio.wait_for + default, not a convention",
        )
    )
    return result


def drift_check(
    baseline: TaskMetrics,
    current: TaskMetrics,
    *,
    max_regression: float = 0.02,
) -> GateResult:
    """Nightly golden-set re-run against the live model (Phase 6).

    Cheap, and it is the only thing that catches a silent break from a model,
    kernel, or vLLM upgrade before the P&L does.
    """
    result = GateResult(gate=f"drift ({current.task})")
    for name, base, now in (
        ("field accuracy", baseline.field_accuracy, current.field_accuracy),
        ("valid@1", baseline.valid_at_1, current.valid_at_1),
        ("numeric exactness", baseline.numeric_exactness, current.numeric_exactness),
    ):
        regression = base - now
        result.checks.append(
            Check(
                name,
                regression <= max_regression,
                f"{base:.4f} -> {now:.4f} (regression {regression:+.4f}, budget {max_regression:.4f})",
            )
        )

    rise = current.false_confidence_rate - baseline.false_confidence_rate
    result.checks.append(
        Check(
            "false-confidence rate",
            rise <= max_regression,
            f"{baseline.false_confidence_rate:.4f} -> {current.false_confidence_rate:.4f} ({rise:+.4f})",
        )
    )
    return result


def summarize(results: Sequence[GateResult], title: Optional[str] = None) -> str:
    lines = [title] if title else []
    lines.extend(r.report() for r in results)
    overall = all(r.passed for r in results)
    lines.append(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    return "\n".join(lines)


__all__ = [
    "Check",
    "GateResult",
    "drift_check",
    "gate_a",
    "gate_b",
    "gate_c",
    "gate_d",
    "summarize",
]
