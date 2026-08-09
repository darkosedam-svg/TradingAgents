"""Decision records, the journal that scores them, and the brake on the loop.

Three things, in the order the evidence says to build them:

1. **A decision is separate from its execution.** :class:`Decision` states
   intent and carries no venue, order type or credentials. Today a
   :class:`~services.decisions.sinks.AlertSink` tells a human; later a broker
   sink acts. The reasoning code never changes.

2. **Every decision is written down before anything acts on it**, and outcomes
   arrive later as their own entries. A decision row structurally cannot
   contain what happened next.

3. **Nothing reweights itself without counting trials.**
   :class:`~services.decisions.trials.OverfittingGuard` judges a result against
   what the best of that many no-skill attempts would have produced anyway.

Stdlib only. No dependencies, by design — this is the layer you want working
before you have installed anything or spent anything.
"""

from .journal import DecisionJournal, Pair
from .record import Decision, Domain, Realisation, Side
from .scoring import by_domain, by_source, by_strategy, calibration, overall, summary
from .sinks import AlertSink, FanOut, JournalSink, NullSink, Sink, alerting_stack
from .trials import (
    OverfittingGuard,
    TrialRegister,
    Verdict,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    measure_trial_dispersion,
    min_track_record_length,
)

__all__ = [
    "AlertSink",
    "Decision",
    "DecisionJournal",
    "Domain",
    "FanOut",
    "JournalSink",
    "NullSink",
    "OverfittingGuard",
    "Pair",
    "Realisation",
    "Side",
    "Sink",
    "TrialRegister",
    "Verdict",
    "alerting_stack",
    "by_domain",
    "by_source",
    "by_strategy",
    "calibration",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "measure_trial_dispersion",
    "min_track_record_length",
    "overall",
    "summary",
]
