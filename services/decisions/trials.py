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

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, stdev
from typing import Optional, Sequence

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

    Pass ``path`` to persist. A register that lives only in memory resets to
    zero every time the process restarts, so a strategy run daily from cron
    would report one trial forever no matter how many variants had been tried
    across the months — the exact failure this class exists to prevent.
    """

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._trials: list[Trial] = []
        if self.path is not None and self.path.exists():
            self._trials = [
                Trial(**json.loads(line))
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def register(self, strategy_id: str, description: str = "") -> Trial:
        trial = Trial(
            trial_id=uuid.uuid4().hex,
            strategy_id=strategy_id,
            description=description,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._trials.append(trial)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(trial)) + "\n")
        return trial

    def register_once(self, strategy_id: str, description: str = "") -> Trial:
        """Register unless this exact variant is already on the books.

        A trial is a distinct *attempt*, not a distinct *run*. Running the same
        strategy again tomorrow is not a new roll of the dice, and counting it
        as one would inflate the correction until nothing could ever pass. The
        pair ``(strategy_id, description)`` is what identifies an attempt, so
        describe your variants precisely enough to tell them apart.
        """
        for trial in self._trials:
            if trial.strategy_id == strategy_id and trial.description == description:
                return trial
        return self.register(strategy_id, description)

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


def measure_trial_dispersion(trial_sharpes: Sequence[float]) -> float:
    """Spread of Sharpe ratios across the attempts you actually made.

    This is the scale factor the whole correction hangs on, and guessing it is
    the easiest way to get a wrong answer in either direction. The paper's
    worked examples use annualised Sharpes, where a dispersion near 1.0 is
    reasonable; a per-observation Sharpe from a few hundred daily returns has a
    dispersion nearer ``1/sqrt(n_observations)``. Feed the wrong one in and the
    guard either rejects everything or waves everything through.

    So: run the search, collect the Sharpe of every cell you tried — including
    the bad ones, especially the bad ones — and hand the list to this.
    """
    if len(trial_sharpes) < 2:
        raise ValueError(
            "dispersion needs at least 2 trials; with one attempt there is no "
            "search to correct for and the benchmark is simply zero"
        )
    return stdev(trial_sharpes)


def no_skill_dispersion(n_observations: int) -> float:
    """Spread of per-observation Sharpe estimates under no skill: ``1/√n``.

    Use this when you cannot hand :func:`measure_trial_dispersion` the actual
    per-cell results — a live track record, for instance, where the "trials"
    are a handful of strategies rather than a grid you can enumerate. It is the
    sampling standard deviation of the Sharpe estimator when true skill is
    exactly zero, which is precisely the scale the correction needs.

    It is an estimate of the null, not of your search. A grid whose cells
    genuinely differ will disperse wider than this, and passing the measured
    value is always better when you have it.
    """
    if n_observations < 2:
        raise ValueError(f"n_observations must be >= 2, got {n_observations}")
    return 1.0 / math.sqrt(n_observations)


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
        needed = (
            "never — not ahead of the bar"
            if math.isinf(self.min_observations)
            else f"{self.min_observations:,.0f}"
        )
        return (
            f"[{head}] Sharpe {self.observed_sr:.3f} over {self.n_observations} obs, "
            f"{self.n_trials} trial(s)\n"
            f"  no-skill benchmark (best of {self.n_trials}): {self.benchmark_sr:.3f}\n"
            f"  deflated Sharpe (P skill is real):            {self.dsr:.3f}\n"
            f"  observations needed:                          "
            f"{needed}\n"
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
        sr_std_across_trials: float = 1.0,
    ) -> None:
        self.register = register or TrialRegister()
        self.min_dsr = min_dsr
        self.min_observations = min_observations
        self.sr_std_across_trials = sr_std_across_trials

    def evaluate(
        self,
        observed_sr: float,
        *,
        n_observations: int,
        n_trials: Optional[int] = None,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        sr_std_across_trials: Optional[float] = None,
    ) -> Verdict:
        """Judge one result.

        ``sr_std_across_trials`` is the spread of Sharpe ratios *across the
        attempts you made*, and it sets the scale of the whole correction. The
        default of 1.0 comes from the paper, where Sharpes are annualised; if
        you are feeding this per-observation Sharpes — as
        :mod:`services.decisions.scoring` produces — 1.0 is wildly too large and
        the guard will reject everything. Measure it from your own trial results
        and pass it in. :func:`measure_trial_dispersion` does that.
        """
        trials = self.register.count if n_trials is None else n_trials
        trials = max(1, trials)
        std = (
            self.sr_std_across_trials
            if sr_std_across_trials is None
            else sr_std_across_trials
        )

        benchmark = expected_max_sharpe(trials, std_sr=std)
        dsr = deflated_sharpe_ratio(
            observed_sr,
            n_trials=trials,
            n_observations=n_observations,
            skew=skew,
            kurtosis=kurtosis,
            sr_std_across_trials=std,
        )
        # The bar a track record has to clear is the no-skill benchmark, not
        # zero. With one trial those coincide; with many they do not, and
        # measuring against zero would quietly promise that enough data can
        # prove an edge the search already inflated.
        needed = min_track_record_length(
            observed_sr, target_sr=benchmark, skew=skew, kurtosis=kurtosis
        )

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
            elif observed_sr <= 0:
                # Without this branch the next one tells someone whose strategy
                # is losing money to "keep logging", which is the opposite of
                # the right advice.
                why = (
                    f"the result is not positive ({observed_sr:+.3f}). No sample "
                    "size rescues a negative expectancy — there is nothing here "
                    "to collect more evidence about."
                )
            elif needed > n_observations:
                against = (
                    ""
                    if trials == 1
                    else f" to prove it beats the best of {trials} no-skill attempts"
                )
                why = (
                    f"not enough evidence for an edge this size: {n_observations} "
                    f"observations, roughly {needed:,.0f} needed{against}. "
                    "Keep logging."
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
    "measure_trial_dispersion",
    "min_track_record_length",
]
