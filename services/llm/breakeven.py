"""Does the escalation router pay for itself? Four numbers and a division.

    python services/llm/breakeven.py --candidates 400 --frontier-share 1.0 \
        --escalation-rate 0.25 \
        --triage-price 0.30 1.20 --frontier-price 3.00 15.00

**This file has no dependencies and no imports from the rest of the package.**
That is deliberate: it is the first thing anyone runs, *before* deciding whether
to build the layer at all, so it must not require installing the layer first.
Plain Python, no venv, no pip. Copy the single file anywhere and it works.

Phase 0 asks whether the layer is worth building. The full worksheet in
`docs/llm-baseline.md` wants seven days of real traffic, and it should still be
filled in — but the *shape* of the answer does not need a week to find, and a
project that cannot pay for itself on optimistic assumptions will not pay for
itself on real ones.

The whole thing reduces to one inequality:

    triage_cost_per_call / frontier_cost_per_call  <  frontier_share_today

Screening every candidate cheaply only saves money if the cheap screen costs
less than the frontier calls it avoids. If a cheap filter already sits in front
of your frontier model — if `frontier_share_today` is already low — there is
little left for the router to remove, and this prints a negative number.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

DAYS_PER_MONTH = 30.4


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens.

    Mirrors ``client.budget.Pricing`` rather than importing it, to keep this
    file free of package imports — importing anything from ``services.llm``
    pulls in pydantic and httpx. ``test_breakeven`` asserts the two stay in
    agreement, so the duplication cannot drift silently.
    """

    prompt_per_1m: float = 0.0
    completion_per_1m: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_per_1m
            + completion_tokens * self.completion_per_1m
        ) / 1_000_000


@dataclass(frozen=True)
class CallProfile:
    """What one call to a tier costs."""

    prompt_tokens: int
    completion_tokens: int
    pricing: Pricing

    @property
    def cost(self) -> float:
        return self.pricing.cost(self.prompt_tokens, self.completion_tokens)


@dataclass(frozen=True)
class Scenario:
    """One what-if. All rates are fractions of the candidate stream."""

    candidates_per_day: float
    # Fraction of candidates that reach a frontier model *today*. 1.0 means
    # everything gets the expensive treatment — the best case for the router.
    frontier_share_today: float
    # Fraction the router would escalate. Gate C tunes this for recall, so it
    # will be higher than the fraction that turns out to matter.
    escalation_rate: float
    triage: CallProfile
    frontier: CallProfile

    def __post_init__(self) -> None:
        for name in ("frontier_share_today", "escalation_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction 0..1, got {value}")
        if self.candidates_per_day < 0:
            raise ValueError("candidates_per_day must not be negative")

    # ------------------------------------------------------------------ costs

    @property
    def baseline_daily(self) -> float:
        """Today: some share of candidates gets a frontier call, nothing else."""
        return self.candidates_per_day * self.frontier_share_today * self.frontier.cost

    @property
    def routed_daily(self) -> float:
        """With the router: *every* candidate is triaged, survivors escalate."""
        return self.candidates_per_day * (
            self.triage.cost + self.escalation_rate * self.frontier.cost
        )

    @property
    def saving_daily(self) -> float:
        return self.baseline_daily - self.routed_daily

    @property
    def saving_monthly(self) -> float:
        return self.saving_daily * DAYS_PER_MONTH

    @property
    def saving_fraction(self) -> float:
        if self.baseline_daily <= 0:
            return 0.0
        return self.saving_daily / self.baseline_daily

    # ------------------------------------------------------------- thresholds

    @property
    def cost_ratio(self) -> float:
        """Triage cost as a fraction of a frontier call. The number that matters."""
        if self.frontier.cost <= 0:
            return float("inf")
        return self.triage.cost / self.frontier.cost

    @property
    def breakeven_escalation_rate(self) -> float:
        """The escalation rate at which the router exactly breaks even.

        Negative means it never does: triage costs more per candidate than the
        frontier calls it removes, no matter how ruthlessly it filters.
        """
        return self.frontier_share_today - self.cost_ratio

    @property
    def viable(self) -> bool:
        """True if *some* escalation rate makes the router cheaper."""
        return self.breakeven_escalation_rate > 0

    @property
    def frontier_call_reduction(self) -> float:
        """What Gate C measures: the drop in frontier calls, not in dollars."""
        if self.frontier_share_today <= 0:
            return 0.0
        return max(
            0.0,
            (self.frontier_share_today - self.escalation_rate) / self.frontier_share_today,
        )


def report(scenario: Scenario, *, off_ramp_monthly: float = 15.0) -> str:
    lines = [
        "Per-call cost",
        f"  triage      ${scenario.triage.cost:.6f}"
        f"  ({scenario.triage.prompt_tokens} in / {scenario.triage.completion_tokens} out)",
        f"  frontier    ${scenario.frontier.cost:.6f}"
        f"  ({scenario.frontier.prompt_tokens} in / {scenario.frontier.completion_tokens} out)",
        f"  ratio       {scenario.cost_ratio:.4f}  (triage / frontier)",
        "",
        f"Daily volume  {scenario.candidates_per_day:,.0f} candidates,"
        f" {scenario.frontier_share_today:.0%} reach a frontier model today",
        "",
        "Spend",
        f"  today       ${scenario.baseline_daily:8.2f}/day"
        f"   ${scenario.baseline_daily * DAYS_PER_MONTH:9.2f}/mo",
        f"  with router ${scenario.routed_daily:8.2f}/day"
        f"   ${scenario.routed_daily * DAYS_PER_MONTH:9.2f}/mo"
        f"   (escalating {scenario.escalation_rate:.0%})",
        f"  saving      ${scenario.saving_daily:8.2f}/day"
        f"   ${scenario.saving_monthly:9.2f}/mo   ({scenario.saving_fraction:+.1%})",
        "",
        f"Gate C frontier-call reduction: {scenario.frontier_call_reduction:.1%}"
        f"  (floor 60%) {'PASS' if scenario.frontier_call_reduction >= 0.60 else 'FAIL'}",
    ]

    lines.append("")
    if not scenario.viable:
        lines.append(
            "VERDICT: the router cannot pay for itself in this scenario.\n"
            f"  A triage call costs {scenario.cost_ratio:.1%} of a frontier call, but only\n"
            f"  {scenario.frontier_share_today:.0%} of candidates reach the frontier model today.\n"
            "  Screening everything cheaply costs more than the calls it removes.\n"
            "  Fix by using a cheaper/smaller triage model, shortening the triage\n"
            "  prompt, or accepting that there is nothing here to save."
        )
    else:
        lines.append(
            f"Break-even escalation rate: {scenario.breakeven_escalation_rate:.1%}\n"
            f"  Escalate less than this and the router saves money; more and it costs money.\n"
            f"  You are at {scenario.escalation_rate:.0%}, which is "
            f"{'inside' if scenario.escalation_rate < scenario.breakeven_escalation_rate else 'OUTSIDE'}"
            " that bound."
        )
        lines.append("")
        if scenario.saving_monthly < off_ramp_monthly:
            lines.append(
                f"VERDICT: saves ${scenario.saving_monthly:.2f}/mo, under the "
                f"${off_ramp_monthly:.0f}/mo off-ramp.\n"
                "  Phase 0 says stop here. The layer is not worth the ops burden yet."
            )
        else:
            lines.append(
                f"VERDICT: saves ${scenario.saving_monthly:.2f}/mo, clearing the "
                f"${off_ramp_monthly:.0f}/mo off-ramp.\n"
                "  Worth building — but confirm the volume and prices against real\n"
                "  traffic before trusting the number."
            )

    return "\n".join(lines)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidates", type=float, required=True, help="Items screened per day."
    )
    parser.add_argument(
        "--frontier-share",
        type=float,
        default=1.0,
        help="Fraction of candidates that reach a frontier model TODAY (default 1.0).",
    )
    parser.add_argument(
        "--escalation-rate",
        type=float,
        default=0.25,
        help="Fraction the router would escalate (default 0.25).",
    )
    parser.add_argument(
        "--triage-tokens", type=int, nargs=2, metavar=("IN", "OUT"), default=(900, 120)
    )
    parser.add_argument(
        "--frontier-tokens", type=int, nargs=2, metavar=("IN", "OUT"), default=(3000, 800)
    )
    parser.add_argument(
        "--triage-price",
        type=float,
        nargs=2,
        metavar=("IN_PER_1M", "OUT_PER_1M"),
        required=True,
        help="Triage model price in USD per 1M tokens. Check your provider.",
    )
    parser.add_argument(
        "--frontier-price",
        type=float,
        nargs=2,
        metavar=("IN_PER_1M", "OUT_PER_1M"),
        required=True,
        help="Frontier model price in USD per 1M tokens. Check your provider.",
    )
    parser.add_argument("--off-ramp", type=float, default=15.0)
    args = parser.parse_args(argv)

    scenario = Scenario(
        candidates_per_day=args.candidates,
        frontier_share_today=args.frontier_share,
        escalation_rate=args.escalation_rate,
        triage=CallProfile(
            *args.triage_tokens, Pricing(*args.triage_price)
        ),
        frontier=CallProfile(
            *args.frontier_tokens, Pricing(*args.frontier_price)
        ),
    )

    print(report(scenario, off_ramp_monthly=args.off_ramp))
    # Non-zero when the numbers say don't build it, so this can gate a script.
    return 0 if scenario.saving_monthly >= args.off_ramp else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
