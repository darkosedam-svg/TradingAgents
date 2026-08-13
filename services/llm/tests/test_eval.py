import json

import pytest

from services.llm.eval import (
    TASK_SPECS,
    RunItem,
    agreement,
    drift_check,
    gate_a,
    gate_b,
    gate_c,
    gate_d,
    load_golden,
    score_run,
)
from services.llm.eval.harness import TASK_SCHEMAS, render_input
from services.llm.eval.report import diff_table


def sentiment_items(n: int, wrong: int = 0, confidence: float = 0.9) -> list[RunItem]:
    items = []
    for i in range(n):
        predicted = "bearish" if i < wrong else "bullish"
        items.append(
            RunItem(
                item_id=str(i),
                expected={"sentiment": "bullish", "assets": ["ETH"], "insufficient_data": False},
                predicted={
                    "sentiment": predicted,
                    "assets": ["ETH"],
                    "insufficient_data": False,
                    "rationale": "x",
                },
                valid=True,
                confidence=confidence,
                latency_s=0.5 + i * 0.01,
                prompt_tokens=100,
                completion_tokens=20,
            )
        )
    return items


def test_score_run_computes_the_headline_numbers():
    metrics = score_run(sentiment_items(10, wrong=2), TASK_SPECS["sentiment"])

    assert metrics.n == 10
    assert metrics.valid_at_1 == 1.0
    assert metrics.abstain_rate == 0.0
    assert metrics.field_scores["sentiment"].accuracy == pytest.approx(0.8)
    assert metrics.field_scores["assets"].f1 == pytest.approx(1.0)
    assert metrics.false_confidence_rate == pytest.approx(0.2)
    assert metrics.total_tokens == 1200


def test_free_text_fields_are_not_scored():
    metrics = score_run(sentiment_items(4), TASK_SPECS["sentiment"])
    assert metrics.field_scores["rationale"].compared == 0


def test_abstentions_are_excluded_from_field_scores_but_counted():
    items = sentiment_items(4)
    items.append(
        RunItem(
            item_id="a",
            expected={"sentiment": "bullish"},
            abstained=True,
            abstain_reason="LOW_CONFIDENCE",
            valid=True,
        )
    )
    metrics = score_run(items, TASK_SPECS["sentiment"])

    assert metrics.abstain_rate == pytest.approx(0.2)
    assert metrics.abstain_by_reason == {"LOW_CONFIDENCE": 1}
    assert metrics.field_scores["sentiment"].compared == 4
    # An abstention is never a confident wrong answer.
    assert metrics.false_confidence_rate == 0.0


def test_a_wrong_answer_below_the_floor_is_not_false_confidence():
    metrics = score_run(
        sentiment_items(10, wrong=10, confidence=0.2), TASK_SPECS["sentiment"]
    )
    assert metrics.field_scores["sentiment"].accuracy == 0.0
    assert metrics.false_confidence_rate == 0.0


def test_numeric_exactness_is_zero_tolerance_for_prices():
    items = [
        RunItem(
            item_id="1",
            expected={"valid": True, "asset": "BTC", "side": "long", "entry": 64200.0,
                      "stop_loss": 62800.0, "take_profit": [66500.0], "leverage": None},
            predicted={"valid": True, "asset": "BTC", "side": "long", "entry": 64200.5,
                       "stop_loss": 62800.0, "take_profit": [66500.0], "leverage": None},
            valid=True,
            confidence=0.9,
        )
    ]
    metrics = score_run(items, TASK_SPECS["signal_parse"])
    # entry off by 0.5 -> wrong; stop, leverage (both None) -> right.
    assert metrics.numeric_exactness == pytest.approx(2 / 3)


def test_gated_fields_are_skipped_on_negatives():
    items = [
        RunItem(
            item_id="1",
            expected={"valid": False, "asset": None, "side": None, "entry": None,
                      "stop_loss": None, "take_profit": [], "leverage": None},
            predicted={"valid": False, "asset": None, "side": None, "entry": None,
                       "stop_loss": None, "take_profit": [], "leverage": None},
            valid=True,
            confidence=0.9,
        )
    ]
    metrics = score_run(items, TASK_SPECS["signal_parse"])
    assert metrics.field_scores["entry"].compared == 0
    assert metrics.field_scores["valid"].accuracy == 1.0


def test_gate_a_passes_a_candidate_that_matches_the_reference():
    reference = score_run(sentiment_items(200, wrong=10), TASK_SPECS["sentiment"])
    candidate = score_run(sentiment_items(200, wrong=10), TASK_SPECS["sentiment"])

    result = gate_a(reference, candidate)
    assert result.passed, result.report()


def test_gate_a_false_confidence_is_strict_by_default():
    """Two extra confident errors in 200 is inside every other budget and still blocks."""
    reference = score_run(sentiment_items(200, wrong=10), TASK_SPECS["sentiment"])
    candidate = score_run(sentiment_items(200, wrong=12), TASK_SPECS["sentiment"])

    assert [c.name for c in gate_a(reference, candidate).failures] == [
        "false-confidence rate"
    ]
    # ...unless slack is granted deliberately, which is a decision to accept a
    # measured increase in confident errors.
    assert gate_a(reference, candidate, false_confidence_slack=0.02).passed


def test_gate_a_fails_on_a_field_accuracy_drop():
    reference = score_run(sentiment_items(200, wrong=0), TASK_SPECS["sentiment"])
    candidate = score_run(sentiment_items(200, wrong=40), TASK_SPECS["sentiment"])

    result = gate_a(reference, candidate)
    assert not result.passed
    assert any("field accuracy" in c.name for c in result.failures)


def test_gate_a_fails_when_the_candidate_is_more_confidently_wrong():
    """Two errors in 200 is inside the accuracy budget but still blocks the gate."""
    reference = score_run(sentiment_items(200, wrong=0), TASK_SPECS["sentiment"])
    candidate = score_run(sentiment_items(200, wrong=2), TASK_SPECS["sentiment"])

    result = gate_a(reference, candidate)
    assert not result.passed
    assert [c.name for c in result.failures] == ["false-confidence rate"]


def test_gate_a_fails_a_small_sample():
    reference = score_run(sentiment_items(20), TASK_SPECS["sentiment"])
    candidate = score_run(sentiment_items(20), TASK_SPECS["sentiment"])
    assert any("sample size" in c.name for c in gate_a(reference, candidate).failures)


def test_gate_b_requires_agreement_volume_and_zero_malformed_votes():
    passing = gate_b(
        agreement_rate=0.93,
        paired_decisions=600,
        malformed_propagated=0,
        reviewed_disagreements=42,
    )
    assert passing.passed, passing.report()

    leaked = gate_b(
        agreement_rate=0.99,
        paired_decisions=600,
        malformed_propagated=1,
        reviewed_disagreements=5,
    )
    assert not leaked.passed


def test_gate_b_fails_on_thin_evidence():
    result = gate_b(agreement_rate=0.99, paired_decisions=40, malformed_propagated=0)
    assert any("paired decisions" in c.name for c in result.failures)


def test_gate_c_holds_recall_strictly():
    assert gate_c(recall=0.96, frontier_call_reduction=0.7, labelled_positives=120).passed
    assert not gate_c(recall=0.94, frontier_call_reduction=0.9, labelled_positives=120).passed
    assert not gate_c(recall=0.99, frontier_call_reduction=0.4, labelled_positives=120).passed


def test_gate_d_fails_when_entry_timing_moves():
    result = gate_d(replayed_launches=40, timing_identical=False, enforced_in_code=True)
    assert not result.passed
    assert "on the entry path" in result.failures[0].detail


def test_drift_check_catches_a_regression():
    baseline = score_run(sentiment_items(200, wrong=4), TASK_SPECS["sentiment"])
    current = score_run(sentiment_items(200, wrong=30), TASK_SPECS["sentiment"])
    assert not drift_check(baseline, current).passed


def test_agreement_counts_mutual_abstentions_as_agreement():
    assert agreement(["bullish", None, "bearish"], ["bullish", None, "bullish"]) == pytest.approx(2 / 3)
    assert agreement([], []) == 0.0
    assert agreement(["bullish", None], [None, "bullish"]) == 0.0


def test_diff_table_renders_both_sections():
    reference = score_run(sentiment_items(200, wrong=4), TASK_SPECS["sentiment"])
    candidate = score_run(sentiment_items(200, wrong=8), TASK_SPECS["sentiment"])
    table = diff_table("reference-fp16", reference, "candidate-awq", candidate)

    assert "valid@1" in table and "false-confidence rate" in table
    assert "| sentiment |" in table


@pytest.mark.parametrize("task", sorted(TASK_SCHEMAS))
def test_seed_golden_sets_are_well_formed(task):
    """The seeds are small, but they must be loadable and label only real fields."""
    items = load_golden(task)
    assert items, task

    schema = TASK_SCHEMAS[task]
    allowed = set(schema.model_fields)
    scorable = {f.name for f in TASK_SPECS[task].fields}

    for item in items:
        assert set(item.expected) <= allowed, (task, item.item_id)
        assert set(item.expected) & scorable, (task, item.item_id)
        rendered = render_input(task, item.payload)
        assert isinstance(rendered, str) and rendered.strip()


def test_render_input_lists_candidate_markets_for_triage():
    rendered = render_input(
        "news_triage",
        {"text": "something happened", "markets": [{"id": "m-1", "question": "will it?"}]},
    )
    assert "m-1: will it?" in rendered
    assert "something happened" in rendered


def test_load_golden_rejects_a_broken_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "1", "input": "x", "expected": {}}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_golden("sentiment", path)


def test_load_golden_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "sparse.jsonl"
    path.write_text(
        "\n// a note\n" + json.dumps({"id": "1", "input": "x", "expected": {"sentiment": "bullish"}}) + "\n",
        encoding="utf-8",
    )
    assert len(load_golden("sentiment", path)) == 1
