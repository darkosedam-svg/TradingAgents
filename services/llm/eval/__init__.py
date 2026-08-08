"""Golden-set evaluation: the thing that makes everything after it safe."""

from .gates import (
    Check,
    GateResult,
    drift_check,
    gate_a,
    gate_b,
    gate_c,
    gate_d,
    summarize,
)
from .harness import GoldenItem, RunResult, evaluate, load_golden, run_golden
from .metrics import (
    TASK_SPECS,
    FieldSpec,
    RunItem,
    TaskMetrics,
    TaskSpec,
    agreement,
    score_run,
)
from .report import diff_table, load_metrics

__all__ = [
    "Check",
    "FieldSpec",
    "GateResult",
    "GoldenItem",
    "RunItem",
    "RunResult",
    "TASK_SPECS",
    "TaskMetrics",
    "TaskSpec",
    "agreement",
    "diff_table",
    "drift_check",
    "evaluate",
    "gate_a",
    "gate_b",
    "gate_c",
    "gate_d",
    "load_golden",
    "load_metrics",
    "run_golden",
    "score_run",
    "summarize",
]
