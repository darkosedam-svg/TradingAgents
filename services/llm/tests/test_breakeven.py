import pathlib

import pytest

from services.llm.breakeven import (
    DAYS_PER_MONTH,
    CallProfile,
    Pricing,
    Scenario,
    _cli,
    report,
)
from services.llm.client.budget import Pricing as BudgetPricing

CHEAP = Pricing(prompt_per_1m=0.30, completion_per_1m=1.20)
FRONTIER = Pricing(prompt_per_1m=3.00, completion_per_1m=15.00)


def test_breakeven_has_no_package_imports():
    """It must run on bare Python — it is what you use to decide whether to
    install anything at all."""
    source = (
        pathlib.Path(__file__).parent.parent / "breakeven.py"
    ).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "services" not in stripped, stripped
            assert not stripped.startswith("from ."), stripped
            for third_party in ("pydantic", "httpx"):
                assert third_party not in stripped, stripped


def test_local_pricing_agrees_with_the_budget_ledger():
    """The duplication is deliberate; this stops it drifting."""
    local = Pricing(prompt_per_1m=0.30, completion_per_1m=1.20)
    canonical = BudgetPricing(prompt_per_1m=0.30, completion_per_1m=1.20)

    for prompt, completion in ((0, 0), (1, 1), (900, 120), (10_000, 2_500)):
        assert local.cost(prompt, completion) == canonical.cost(prompt, completion)

    assert Pricing().cost(1_000_000, 1_000_000) == 0.0  # unpriced means zero


def scenario(**overrides) -> Scenario:
    payload = {
        "candidates_per_day": 400,
        "frontier_share_today": 1.0,
        "escalation_rate": 0.25,
        "triage": CallProfile(900, 120, CHEAP),
        "frontier": CallProfile(3000, 800, FRONTIER),
    }
    payload.update(overrides)
    return Scenario(**payload)


def test_per_call_costs_are_the_pricing_arithmetic():
    s = scenario()
    # 900 @ $0.30/M + 120 @ $1.20/M = 0.00027 + 0.000144
    assert s.triage.cost == pytest.approx(0.000414)
    # 3000 @ $3/M + 800 @ $15/M = 0.009 + 0.012
    assert s.frontier.cost == pytest.approx(0.021)
    assert s.cost_ratio == pytest.approx(0.000414 / 0.021)


def test_saving_when_everything_currently_hits_the_frontier():
    s = scenario()
    assert s.baseline_daily == pytest.approx(400 * 0.021)
    assert s.routed_daily == pytest.approx(400 * (0.000414 + 0.25 * 0.021))
    assert s.saving_daily > 0
    assert s.saving_monthly == pytest.approx(s.saving_daily * DAYS_PER_MONTH)
    assert 0 < s.saving_fraction < 1


def test_breakeven_is_share_minus_cost_ratio():
    s = scenario(frontier_share_today=0.8)
    assert s.breakeven_escalation_rate == pytest.approx(0.8 - s.cost_ratio)

    # At exactly break-even the two spends match.
    at_be = scenario(
        frontier_share_today=0.8, escalation_rate=s.breakeven_escalation_rate
    )
    assert at_be.saving_daily == pytest.approx(0.0, abs=1e-9)


def test_router_is_not_viable_when_a_cheap_filter_already_exists():
    """The failure mode worth naming: little frontier spend left to remove."""
    s = scenario(frontier_share_today=0.01)
    assert not s.viable
    assert s.breakeven_escalation_rate < 0
    assert s.saving_daily < 0
    assert "cannot pay for itself" in report(s)


def test_escalating_more_than_breakeven_costs_money():
    s = scenario(frontier_share_today=0.5, escalation_rate=0.9)
    assert s.saving_daily < 0
    assert "OUTSIDE" in report(s)


def test_gate_c_reduction_is_about_calls_not_dollars():
    s = scenario(frontier_share_today=1.0, escalation_rate=0.25)
    assert s.frontier_call_reduction == pytest.approx(0.75)

    thin = scenario(frontier_share_today=0.5, escalation_rate=0.4)
    assert thin.frontier_call_reduction == pytest.approx(0.2)
    assert "FAIL" in report(thin)


def test_reduction_is_zero_when_the_router_escalates_everything():
    s = scenario(escalation_rate=1.0)
    assert s.frontier_call_reduction == 0.0


def test_off_ramp_verdict_flips_on_the_threshold():
    small = scenario(candidates_per_day=5)
    assert "off-ramp" in report(small, off_ramp_monthly=15.0)
    assert "Phase 0 says stop here" in report(small, off_ramp_monthly=15.0)

    big = scenario(candidates_per_day=5000)
    assert "clearing the" in report(big, off_ramp_monthly=15.0)


def test_zero_volume_is_handled_rather_than_dividing_by_zero():
    s = scenario(candidates_per_day=0)
    assert s.saving_fraction == 0.0
    assert s.baseline_daily == 0.0
    report(s)  # must not raise


def test_rates_must_be_fractions():
    with pytest.raises(ValueError, match="escalation_rate"):
        scenario(escalation_rate=1.5)
    with pytest.raises(ValueError, match="frontier_share_today"):
        scenario(frontier_share_today=-0.1)
    with pytest.raises(ValueError, match="candidates_per_day"):
        scenario(candidates_per_day=-1)


def test_cli_exit_code_gates_on_the_off_ramp(capsys):
    args = [
        "--candidates", "5000",
        "--triage-price", "0.30", "1.20",
        "--frontier-price", "3.00", "15.00",
    ]
    assert _cli(args) == 0
    assert "VERDICT" in capsys.readouterr().out

    lean = [
        "--candidates", "3",
        "--triage-price", "0.30", "1.20",
        "--frontier-price", "3.00", "15.00",
    ]
    assert _cli(lean) == 1
