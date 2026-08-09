"""Counting trials, and deflating performance for the search that found it.

This module exists because of one finding: the standard guardrails do not work.
Bailey & López de Prado, *The Deflated Sharpe Ratio* (Journal of Portfolio
Management, 2014):

    A backtest where the researcher has not controlled for the extent of the
    search involved in his or her finding is worthless, regardless of how
    excellent the reported performance might be.

and:

    If we apply the holdout method enough times (say 20 times for a 95%
    confidence level), false positives are no longer unlikely: They are
    expected.

The consequence for a self-adjusting system is direct. Every strategy variant,
every reweighting cycle, every walk-forward re-optimisation is another trial.
A system that adapts without counting its trials is not learning — it is
searching, and the best result of a wide search looks excellent even when true
skill is exactly zero.

So: register every trial, and judge any observed Sharpe against what the *best
of that many coin flips* would have produced anyway.

Stdlib only — `statistics.NormalDist` supplies the normal CDF and its inverse.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Optional

# Euler-Mascheroni constant, as used in the paper's Snippet 1.
EULER_MASCHERONI = 0.5772156649015329

_NORM = NormalDist()


@dataclass(frozen=True)
class Trial:
    """One attempt at a strategy. The unit the correction counts."""

    trial_id: str
    strategy_id: str
    description: str
    ts: str


class TrialRegister:
    """Append-only count of every strategy variant attempted.

    Deliberately hard to decrement. The temptation when a result looks good is
    to forget the nineteen variants that came before it, and that forgetting is
    precisely what makes the twentieth look significant.
    """

    def __init__(self) -> None:
        self._trials: list[Trial] = []

    def register(self, strategy_id: str, description: str = "") -> Trial:
        trial = Trial(
            trial_id=uuid.uuid4().hex,
            strategy_id=strategy_id,
            description=description,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._trials.append(trial)
        return trial

    @property
    def count(self) -> int:
        """Total attempts. This is the N that goes into the correction."""
        return len(self._trials)

    def count_for(self, strategy_id: str) -> int:
        return sum(1 for t in self._trials if t.strategy_id == strategy_id)

    @property
    def trials(self) -> tuple[Trial, ...]:
        return tuple(self._trials)


def expected_max_sharpe(
    n_trials: int, *, mean_sr: float = 0.0, std_sr: float = 1.0
) -> float:
    """Expected *best* Sharpe across ``n_trials`` independent attempts.

    This is the null to beat. At ``mean_sr=0`` — no skill whatsoever — the
    expected maximum still rises with the number of attempts, which is the
    whole point: a good-looking backtest is the default outcome of a wide
    enough search.

    Implements Bailey & López de Prado's approximation:

        E[max SR] ≈ mean + std · ((1−γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − 1/(N·e)])
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if std_sr < 0:
        raise ValueError(f"std_sr must not be negative, got {std_sr}")
    if n_trials == 1:
        # The expected maximum of a single draw is just its mean; the formula
        # divides by zero here.
        return mean_sr

    gamma = EULER_MASCHERONI
    max_z = (1 - gamma) * _NORM.inv_cdf(1 - 1 / n_trials) + gamma * _NORM.inv_cdf(
        1 - 1 / (n_trials * math.e)
    )
    return mean_sr + std_sr * max_z


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_std_across_trials: float = 1.0,
) -> float:
    """Probability that ``observed_sr`` reflects real skill rather than search.

    Returns a value in [0, 1]. Above ~0.95 is the conventional bar. The
    benchmark it is tested against is not zero — it is
    :func:`expected_max_sharpe`, i.e. what the best of ``n_trials`` no-skill
    attempts would have produced.

    ``skew`` and ``kurtosis`` describe the returns the Sharpe was computed
    from; the defaults (0, 3) are the normal case. Negative skew and fat tails
    both make a given Sharpe *less* impressive, which the formula accounts for.
    """
    if n_observations < 2:
        raise ValueError(f"n_observations must be >= 2, got {n_observations}")

    benchmark = expected_max_sharpe(n_trials, std_sr=sr_std_across_trials)

    # Standard error of the Sharpe estimator, adjusted for non-normality.
    variance_term = (
        1.0
        - skew * observed_sr
        + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    )
    if variance_term <= 0:
        # Degenerate higher moments; refuse to report a confident number rather
        # than emit a nonsense one.
        return 0.0

    z = (observed_sr - benchmark) * math.sqrt(n_observations - 1) / math.sqrt(
        variance_term
    )
    return _NORM.cdf(z)


def min_track_record_length(
    observed_sr: float,
    *,
    target_sr: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """How many observations you need before an observed Sharpe means anything.

    Answers the question the adaptive loop keeps asking too early: *do I have
    enough evidence to move this weight yet?* Returns the required number of
    observations; ``inf`` when the observed Sharpe does not exceed the target
    at all, because no amount of data rescues a strategy that is not ahead.
    """
    if observed_sr <= target_sr:
        return math.inf

    variance_term = (
        1.0
        - skew * observed_sr
        + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    )
    if variance_term <= 0:
        return math.inf

    z = _NORM.inv_cdf(confidence)
    return 1.0 + variance_term * (z / (observed_sr - target_sr)) ** 2


@dataclass
class Verdict:
    """Whether a result has cleared the search-corrected bar."""

    passed: bool
    observed_sr: float
    benchmark_sr: float
    dsr: float
    n_trials: int
    n_observations: int
    min_observations: float
    reason: str

    def report(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        return (
            f"[{head}] Sharpe {self.observed_sr:.3f} over {self.n_observations} obs, "
            f"{self.n_trials} trial(s)\n"
            f"  no-skill benchmark (best of {self.n_trials}): {self.benchmark_sr:.3f}\n"
            f"  deflated Sharpe (P skill is real):            {self.dsr:.3f}\n"
            f"  observations needed:                          "
            f"{self.min_observations:.0f}\n"
            f"  {self.reason}"
        )


class OverfittingGuard:
    """The brake on the adaptive loop.

    Nothing should reweight a signal, promote a strategy, or change a
    threshold without passing through this. It enforces two independent
    conditions, because either alone is easy to fool: enough evidence
    (``min_observations``), and enough performance *after* correcting for how
    many attempts it took to find (``min_dsr``).
    """

    def __init__(
        self,
        register: Optional[TrialRegister] = None,
        *,
        min_dsr: float = 0.95,
        min_observations: int = 30,
    ) -> None:
        self.register = register or TrialRegister()
        self.min_dsr = min_dsr
        self.min_observations = min_observations

    def evaluate(
        self,
        observed_sr: float,
        *,
        n_observations: int,
        n_trials: Optional[int] = None,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> Verdict:
        trials = self.register.count if n_trials is None else n_trials
        trials = max(1, trials)

        benchmark = expected_max_sharpe(trials)
        dsr = deflated_sharpe_ratio(
            observed_sr,
            n_trials=trials,
            n_observations=n_observations,
            skew=skew,
            kurtosis=kurtosis,
        )
        needed = min_track_record_length(observed_sr, skew=skew, kurtosis=kurtosis)

        if n_observations < self.min_observations:
            return Verdict(
                False, observed_sr, benchmark, dsr, trials, n_observations, needed,
                f"too few observations: {n_observations} < {self.min_observations}. "
                "Collect more before moving any weight.",
            )
        if dsr < self.min_dsr:
            # Two quite different failures share one number, and telling the
            # operator which one they are looking at is the difference between
            # "collect more data" and "stop searching".
            if trials > 1 and observed_sr <= benchmark:
                why = (
                    f"the result ({observed_sr:.3f}) does not even clear what the best "
                    f"of {trials} no-skill attempts would produce ({benchmark:.3f}). "
                    "More data will not fix this; the search itself is the problem."
                )
            elif needed > n_observations:
                why = (
                    f"not enough evidence for an edge this size: {n_observations} "
                    f"observations, roughly {needed:.0f} needed. Keep logging."
                )
            else:
                why = (
                    f"after correcting for {trials} trial(s), this is not yet "
                    "distinguishable from a lucky search."
                )
            return Verdict(
                False, observed_sr, benchmark, dsr, trials, n_observations, needed,
                f"deflated Sharpe {dsr:.3f} < {self.min_dsr:.2f} — {why}",
            )
        return Verdict(
            True, observed_sr, benchmark, dsr, trials, n_observations, needed,
            f"clears the bar after correcting for {trials} trial(s).",
        )


__all__ = [
    "EULER_MASCHERONI",
    "OverfittingGuard",
    "Trial",
    "TrialRegister",
    "Verdict",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
]
