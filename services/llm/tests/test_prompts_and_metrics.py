import pytest

from services.llm import prompts
from services.llm.client.observability import (
    CallRecord,
    InMemoryMetrics,
    percentile,
)
from services.llm.eval.harness import TASK_SCHEMAS
from services.llm.schemas import AbstainReason


@pytest.mark.parametrize("task", sorted(TASK_SCHEMAS))
def test_every_task_has_a_prompt(task):
    prompt = prompts.load(task)
    assert prompt.text
    assert prompt.version >= 1
    assert prompt.ref.startswith(f"{task}.v{prompt.version}@")


def test_prompt_hash_is_content_addressed():
    first = prompts.load("sentiment")
    second = prompts.load("sentiment")
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert first.sha256 != prompts.load("news_triage").sha256


def test_registry_covers_every_task():
    assert set(prompts.registry()) == set(TASK_SCHEMAS)


def test_missing_prompt_and_missing_version_both_raise():
    with pytest.raises(FileNotFoundError):
        prompts.load("no_such_task")
    with pytest.raises(FileNotFoundError, match="have versions"):
        prompts.load("sentiment", 99)


def test_load_defaults_to_the_highest_version():
    versions = prompts.available_versions("sentiment")
    assert prompts.load("sentiment").version == versions[-1]


def test_percentile_is_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 0.5) == 5
    assert percentile(values, 0.95) == 10
    assert percentile([], 0.5) == 0.0
    assert percentile([7], 0.99) == 7


def record(**overrides) -> CallRecord:
    payload = {
        "task": "news_triage",
        "model": "qwen",
        "source": "local",
        "latency_s": 1.0,
        "ok": True,
    }
    payload.update(overrides)
    return CallRecord(**payload)


def test_metrics_snapshot_reports_every_phase_6_series():
    metrics = InMemoryMetrics()
    metrics.record(record(prompt_tokens=100, completion_tokens=25, escalated=True))
    metrics.record(record(latency_s=3.0, escalated=False))
    metrics.record(record(abstain_reason=AbstainReason.SCHEMA_FAIL, escalated=True))
    metrics.record(record(ok=False, source="hosted"))

    stats = metrics.snapshot()["news_triage"]

    assert stats["calls"] == 4
    assert stats["error_rate"] == 0.25
    assert stats["parse_failure_rate"] == 0.25
    assert stats["abstain_rate"] == 0.25
    assert stats["hosted_share"] == 0.25
    assert stats["escalation_rate"] == pytest.approx(2 / 3)
    assert stats["total_tokens"] == 125
    assert stats["p50_latency_s"] > 0


def test_escalation_rate_ignores_calls_that_made_no_escalation_decision():
    metrics = InMemoryMetrics()
    metrics.record(record())
    metrics.record(record())
    assert metrics.escalation_rate("news_triage") == 0.0


def test_tasks_are_tracked_separately():
    metrics = InMemoryMetrics()
    metrics.record(record(task="sentiment"))
    metrics.record(record(task="signal_parse", ok=False))

    snapshot = metrics.snapshot()
    assert snapshot["sentiment"]["error_rate"] == 0.0
    assert snapshot["signal_parse"]["error_rate"] == 1.0


def test_records_get_a_timestamp():
    metrics = InMemoryMetrics(clock=lambda: 1234.5)
    metrics.record(record())
    assert metrics.records[0].ts == 1234.5
