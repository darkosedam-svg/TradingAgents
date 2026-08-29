"""Scoring the journal: was it right, which sources helped, is it calibrated.

Three questions, in increasing order of usefulness:

*Was it right?* Hit rate. Easy, and the least informative — a system that only
calls the obvious ones scores well and earns nothing.

*Which sources helped?* Per-source attribution. This is what the adaptive loop
consumes, and it is the reason `Decision.sources` exists.

*Does it know what it knows?* Calibration. A system that says 0.8 should be
right 80% of the time. Miscalibration is the failure this whole design is built
against: the Polymarket research found prices well-calibrated in aggregate while
most participants still lost, so being right is not the same as being usefully
confident. Brier score measures both together.

Stdlib only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean, stdev
from typing import Optional, Sequence

from .journal import Pair


@dataclass
class Score:
    """Performance of one slice of the journal.

    ``total_return`` is the **sum** of the signed returns, not a compounded
    one. That is the right number when each decision is an independent unit
    stake — $1 on each of 354 prediction markets, say — and the wrong number
    for a portfolio held over time. Compounding this list would be wrong in a
    second way whenever several instruments trade on the same day, since it
    would chain parallel positions as if they had run one after another; see
    :func:`services.decisions.paper.portfolio_return`.
    """

    label: str
    n: int = 0
    hits: int = 0
    total_return: float = 0.0
    returns: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def mean_return(self) -> float:
        return fmean(self.returns) if self.returns else 0.0

    @property
    def sharpe(self) -> float:
        """Per-observation Sharpe. Not annualised — annualising a short,
        noisy sample is how a mediocre strategy starts looking impressive."""
        if len(self.returns) < 2:
            return 0.0
        spread = stdev(self.returns)
        return self.mean_return / spread if spread > 0 else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "n": self.n,
            "hit_rate": round(self.hit_rate, 4),
            "mean_return": round(self.mean_return, 6),
            "sharpe": round(self.sharpe, 4),
            "total_return": round(self.total_return, 6),
        }


def _score(label: str, pairs: Sequence[Pair]) -> Score:
    score = Score(label=label, n=len(pairs))
    for pair in pairs:
        score.hits += int(pair.correct)
        score.returns.append(pair.signed_return)
        score.total_return += pair.signed_return
    return score


def overall(pairs: Sequence[Pair]) -> Score:
    return _score("overall", pairs)


def by_strategy(pairs: Sequence[Pair]) -> dict[str, Score]:
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.decision.strategy_id].append(pair)
    return {name: _score(name, group) for name, group in sorted(grouped.items())}


def by_domain(pairs: Sequence[Pair]) -> dict[str, Score]:
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.decision.domain.value].append(pair)
    return {name: _score(name, group) for name, group in sorted(grouped.items())}


def by_source(pairs: Sequence[Pair]) -> dict[str, Score]:
    """Attribution per contributing signal source.

    A decision with three sources counts toward all three — these are
    overlapping slices, not a partition, so the totals will exceed the number
    of decisions. That is correct: the question is "when this source was
    involved, how did things go", not "who gets the credit".
    """
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        for source in pair.decision.sources:
            grouped[source].append(pair)
    return {name: _score(name, group) for name, group in sorted(grouped.items())}


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    n: int = 0
    hits: int = 0
    confidence_sum: float = 0.0

    @property
    def observed(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def stated(self) -> float:
        return self.confidence_sum / self.n if self.n else 0.0

    @property
    def gap(self) -> float:
        """Stated minus observed. Positive means overconfident."""
        return self.stated - self.observed


@dataclass
class Calibration:
    bins: list[CalibrationBin]
    brier: float
    n: int

    @property
    def overconfidence(self) -> float:
        """Mean stated confidence minus mean realised accuracy.

        The single number worth watching. Positive and growing means the
        system is learning to sound sure rather than to be right.
        """
        scored = [b for b in self.bins if b.n]
        if not scored:
            return 0.0
        total = sum(b.n for b in scored)
        return sum(b.gap * b.n for b in scored) / total

    def table(self) -> str:
        rows = [f"{'confidence':>12}  {'n':>5}  {'stated':>7}  {'actual':>7}  {'gap':>7}"]
        for b in self.bins:
            if not b.n:
                continue
            rows.append(
                f"{b.lower:.2f}–{b.upper:.2f}  {b.n:>5}  "
                f"{b.stated:>7.3f}  {b.observed:>7.3f}  {b.gap:>+7.3f}"
            )
        rows.append(f"\nBrier {self.brier:.4f}   overconfidence {self.overconfidence:+.3f}")
        return "\n".join(rows)


def calibration(pairs: Sequence[Pair], *, bins: int = 5) -> Calibration:
    """Reliability table plus Brier score.

    Brier is the mean squared error between stated confidence and the binary
    outcome. Lower is better; 0.25 is what you get by always saying 0.5.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    width = 1.0 / bins
    buckets = [CalibrationBin(i * width, (i + 1) * width) for i in range(bins)]

    squared_error = 0.0
    for pair in pairs:
        confidence = pair.decision.confidence
        correct = pair.correct
        index = min(int(confidence / width), bins - 1)
        bucket = buckets[index]
        bucket.n += 1
        bucket.hits += int(correct)
        bucket.confidence_sum += confidence
        squared_error += (confidence - float(correct)) ** 2

    brier = squared_error / len(pairs) if pairs else 0.0
    return Calibration(bins=buckets, brier=brier, n=len(pairs))


def summary(pairs: Sequence[Pair], *, bins: int = 5) -> str:
    """Everything at once, for a nightly log line."""
    if not pairs:
        return "No scored decisions yet."

    total = overall(pairs)
    lines = [
        f"{total.n} scored decisions   hit rate {total.hit_rate:.1%}   "
        f"per-obs Sharpe {total.sharpe:.3f}",
        "",
        "By strategy:",
    ]
    for score in by_strategy(pairs).values():
        lines.append(
            f"  {score.label:<24} n={score.n:<5} hit={score.hit_rate:.1%}  "
            f"sharpe={score.sharpe:+.3f}"
        )

    sources = by_source(pairs)
    if sources:
        lines.extend(["", "By source:"])
        for score in sources.values():
            lines.append(
                f"  {score.label:<24} n={score.n:<5} hit={score.hit_rate:.1%}  "
                f"sharpe={score.sharpe:+.3f}"
            )

    lines.extend(["", calibration(pairs, bins=bins).table()])
    lines.extend(
        [
            "",
            "Sharpe here is per-observation and uncorrected. Before acting on any "
            "of it,\nput it through OverfittingGuard with the trial count.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "Calibration",
    "CalibrationBin",
    "Score",
    "by_domain",
    "by_source",
    "by_strategy",
    "calibration",
    "overall",
    "summary",
]
