import json
from pathlib import Path

import pytest

from services.decisions.journal import DecisionJournal
from services.decisions.record import Decision, Domain, Realisation, Side
from services.decisions.sinks import (
    AlertSink,
    ExecutionSink,
    FanOut,
    JournalSink,
    NullSink,
    alerting_stack,
    format_alert,
)


def decision(**overrides) -> Decision:
    payload = {
        "domain": Domain.CRYPTO,
        "instrument": "SOL-USD",
        "side": Side.LONG,
        "confidence": 0.72,
        "rationale": "ETF inflow spike plus funding reset",
        "strategy_id": "momentum",
        "sources": ("news", "funding"),
        "size_fraction": 0.15,
        "horizon_s": 3600,
    }
    payload.update(overrides)
    return Decision(**payload)


# --------------------------------------------------------------- the record


def test_a_decision_carries_no_execution_detail():
    """The architectural guarantee: swapping alerts for a broker later must not
    require touching the decision type."""
    fields = set(Decision.__dataclass_fields__)
    for leaked in ("order_type", "venue", "account", "api_key", "limit_price", "filled"):
        assert leaked not in fields


def test_confidence_and_size_must_be_fractions():
    with pytest.raises(ValueError, match="confidence"):
        decision(confidence=1.4)
    with pytest.raises(ValueError, match="size_fraction"):
        decision(size_fraction=-0.1)


def test_instrument_must_be_real():
    with pytest.raises(ValueError, match="instrument"):
        decision(instrument="   ")


def test_horizon_must_be_positive():
    with pytest.raises(ValueError, match="horizon_s"):
        decision(horizon_s=0)


def test_flat_cannot_carry_size():
    with pytest.raises(ValueError, match="FLAT"):
        decision(side=Side.FLAT, size_fraction=0.2)
    decision(side=Side.FLAT, size_fraction=0.0)  # fine


def test_decisions_get_distinct_ids_and_a_utc_stamp():
    a, b = decision(), decision()
    assert a.decision_id != b.decision_id
    assert a.ts.endswith("+00:00")


def test_decision_round_trips_through_json():
    original = decision()
    restored = Decision.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_realisation_scores_direction():
    long_call = decision(side=Side.LONG)
    short_call = decision(side=Side.SHORT)

    up = Realisation(decision_id=long_call.decision_id, realised_return=0.04)
    down = Realisation(decision_id=long_call.decision_id, realised_return=-0.04)

    assert up.scores(long_call) is True
    assert down.scores(long_call) is False
    assert down.scores(short_call) is True


def test_flat_is_only_correct_on_no_move():
    flat = decision(side=Side.FLAT, size_fraction=0.0)
    assert Realisation(decision_id=flat.decision_id, realised_return=0.0).scores(flat)
    assert not Realisation(decision_id=flat.decision_id, realised_return=0.01).scores(flat)


# -------------------------------------------------------------- the journal


def test_journal_appends_and_reads_back(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "log" / "decisions.jsonl")
    first, second = decision(), decision(instrument="BTC-USD")

    journal.append(first)
    journal.append(second)

    assert len(journal) == 2
    assert [d.instrument for d in journal.decisions()] == ["SOL-USD", "BTC-USD"]


def test_journal_is_append_only_on_disk(tmp_path: Path):
    path = tmp_path / "decisions.jsonl"
    journal = DecisionJournal(path)
    journal.append(decision())
    first_line = path.read_text(encoding="utf-8")

    journal.append(decision(instrument="ETH-USD"))
    journal.record_outcome(Realisation(decision_id="x", realised_return=0.1))

    assert path.read_text(encoding="utf-8").startswith(first_line)


def test_outcomes_are_separate_entries_never_folded_into_the_decision(tmp_path: Path):
    """Structural guarantee against look-ahead: a decision row cannot contain
    what happened next, because outcomes are written later as their own rows."""
    path = tmp_path / "decisions.jsonl"
    journal = DecisionJournal(path)
    made = journal.append(decision())
    journal.record_outcome(Realisation(decision_id=made.decision_id, realised_return=0.05))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    decision_row = next(r for r in rows if r["kind"] == "decision")

    assert "realised_return" not in decision_row
    assert len(rows) == 2
    assert rows[1]["kind"] == "outcome"


def test_pairs_match_outcomes_to_decisions(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    hit = journal.append(decision())
    miss = journal.append(decision(instrument="BTC-USD"))
    journal.append(decision(instrument="ETH-USD"))  # no outcome yet

    journal.record_outcome(Realisation(decision_id=hit.decision_id, realised_return=0.03))
    journal.record_outcome(Realisation(decision_id=miss.decision_id, realised_return=-0.02))

    pairs = journal.pairs()
    assert len(pairs) == 2
    assert [p.correct for p in pairs] == [True, False]
    assert [d.instrument for d in journal.pending()] == ["ETH-USD"]


def test_signed_return_flips_for_a_short(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    short = journal.append(decision(side=Side.SHORT))
    journal.record_outcome(Realisation(decision_id=short.decision_id, realised_return=-0.05))

    pair = journal.pairs()[0]
    assert pair.realised_return == -0.05
    assert pair.signed_return == 0.05  # the short made money
    assert pair.correct


def test_the_first_outcome_wins(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    made = journal.append(decision())
    journal.record_outcome(Realisation(decision_id=made.decision_id, realised_return=0.05))
    journal.record_outcome(Realisation(decision_id=made.decision_id, realised_return=-0.99))

    assert journal.pairs()[0].realised_return == 0.05


def test_pairs_filter_by_domain_and_strategy(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    for dom, strat in ((Domain.CRYPTO, "momentum"), (Domain.EQUITY, "meanrev")):
        made = journal.append(decision(domain=dom, strategy_id=strat))
        journal.record_outcome(
            Realisation(decision_id=made.decision_id, realised_return=0.01)
        )

    assert len(journal.pairs(domain=Domain.CRYPTO)) == 1
    assert len(journal.pairs(strategy_id="meanrev")) == 1
    assert len(journal.pairs()) == 2


def test_orphan_outcomes_are_ignored_not_fatal(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    journal.record_outcome(Realisation(decision_id="never-existed", realised_return=1.0))
    assert journal.pairs() == []


def test_missing_journal_reads_as_empty(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "nothing-here.jsonl")
    assert journal.decisions() == []
    assert len(journal) == 0


def test_corrupt_line_is_reported_with_its_position(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"kind": "decision"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="d.jsonl:2"):
        DecisionJournal(path).outcomes()


# ---------------------------------------------------------------- the sinks


def test_swapping_the_sink_does_not_change_the_decision(tmp_path: Path):
    """The seam: identical decisions, different destinations."""
    made = decision()
    seen: list[str] = []

    NullSink().emit(made)
    AlertSink(seen.append).emit(made)
    journal = DecisionJournal(tmp_path / "d.jsonl")
    JournalSink(journal).emit(made)

    assert journal.decisions()[0] == made
    assert len(seen) == 1


def test_alert_text_is_actionable_without_opening_anything():
    text = format_alert(decision())
    assert "LONG SOL-USD" in text
    assert "72%" in text
    assert "ETF inflow spike" in text
    assert "news, funding" in text


def test_alert_sink_can_suppress_low_confidence():
    seen: list[str] = []
    sink = AlertSink(seen.append, min_confidence=0.8)
    sink.emit(decision(confidence=0.5))
    sink.emit(decision(confidence=0.9))
    assert len(seen) == 1


def test_a_failing_alert_does_not_take_down_the_pipeline():
    def explode(_: str) -> None:
        raise RuntimeError("telegram is down")

    AlertSink(explode).emit(decision())  # must not raise


def test_fanout_journals_before_alerting_and_survives_a_bad_sink(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "d.jsonl")
    order: list[str] = []

    class Recorder:
        def __init__(self, name): self.name = name
        def emit(self, d): order.append(self.name)

    class Broken:
        def emit(self, d): raise RuntimeError("nope")

    FanOut(JournalSink(journal), Broken(), Recorder("after")).emit(decision())

    assert len(journal) == 1          # recorded despite the failure
    assert order == ["after"]         # later sinks still ran


def test_recommended_stack_records_first(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    journal = DecisionJournal(path)
    seen: list[str] = []

    alerting_stack(journal, seen.append).emit(decision())

    assert len(journal) == 1
    assert len(seen) == 1


def test_execution_sink_refuses_to_be_used_by_accident():
    with pytest.raises(NotImplementedError, match="Prove an edge first"):
        ExecutionSink()
