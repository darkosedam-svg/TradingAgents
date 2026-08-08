import pytest

from services.llm.client.breaker import BreakerState, CircuitBreaker
from services.llm.client.budget import BudgetExceeded, BudgetLedger, TaskBudget


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_breaker_trips_only_after_the_threshold():
    breaker = CircuitBreaker(threshold=3, cooldown_s=10.0, clock=FakeClock())

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.allows_local()

    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allows_local()


def test_success_resets_the_failure_run():
    breaker = CircuitBreaker(threshold=2, clock=FakeClock())
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.allows_local()


def test_breaker_half_opens_after_cooldown_and_closes_on_a_good_probe():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_s=30.0, clock=clock)
    breaker.record_failure()

    clock.advance(29)
    assert breaker.state is BreakerState.OPEN

    clock.advance(2)
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allows_local()

    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_failed_probe_reopens_and_restarts_the_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_s=30.0, clock=clock)
    breaker.record_failure()
    clock.advance(31)
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN

    clock.advance(29)
    assert breaker.state is BreakerState.OPEN
    clock.advance(2)
    assert breaker.state is BreakerState.HALF_OPEN


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        CircuitBreaker(threshold=0)


def test_budget_blocks_once_the_daily_token_pool_is_spent():
    clock = FakeClock(now=1_700_000_000.0)
    ledger = BudgetLedger(
        budgets={"sentiment": TaskBudget(max_tokens_per_day=500)}, clock=clock
    )

    ledger.check("sentiment")
    ledger.charge("sentiment", 400)
    ledger.check("sentiment")
    ledger.charge("sentiment", 200)

    with pytest.raises(BudgetExceeded):
        ledger.check("sentiment")


def test_budget_blocks_on_call_count_too():
    ledger = BudgetLedger(budgets={"t": TaskBudget(max_calls_per_day=2)}, clock=FakeClock())
    for _ in range(2):
        ledger.check("t")
        ledger.charge("t", 1)

    with pytest.raises(BudgetExceeded):
        ledger.check("t")


def test_budget_rolls_over_at_the_day_boundary():
    clock = FakeClock(now=1_700_000_000.0)
    ledger = BudgetLedger(budgets={"t": TaskBudget(max_tokens_per_day=100)}, clock=clock)
    ledger.charge("t", 150)
    with pytest.raises(BudgetExceeded):
        ledger.check("t")

    clock.advance(86_400)
    ledger.check("t")
    assert ledger.spent("t") == (0, 0)


def test_unlimited_budget_never_blocks():
    ledger = BudgetLedger(budgets={"t": TaskBudget()}, clock=FakeClock())
    ledger.charge("t", 10**9)
    ledger.check("t")
    assert ledger.remaining_tokens("t") is None


def test_unknown_task_falls_back_to_the_default_budget():
    ledger = BudgetLedger(clock=FakeClock())
    assert ledger.budget_for("brand-new-task") is ledger.default
