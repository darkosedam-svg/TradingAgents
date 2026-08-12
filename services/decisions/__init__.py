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

4. **A track record accumulates whether or not anyone is watching.**
   :mod:`services.decisions.paper` runs the loop daily — resolve what is due,
   emit today, score nothing prematurely — against two deliberately stupid
   baselines::

       python -m services.decisions run
       python -m services.decisions status

Stdlib only. No dependencies, by design — this is the layer you want working
before you have installed anything or spent anything.
"""

from .journal import DecisionJournal, Pair
from .paper import (
    AlwaysLong,
    Coinflip,
    Crossover,
    RunReport,
    Signal,
    Strategy,
    backfill,
    default_strategies,
    run_once,
    status,
)
from .prices import Series, compounded, load_equity, load_series, sharpe
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
    no_skill_dispersion,
)

__all__ = [
    "AlertSink",
    "AlwaysLong",
    "Coinflip",
    "Crossover",
    "Decision",
    "DecisionJournal",
    "Domain",
    "FanOut",
    "JournalSink",
    "NullSink",
    "OverfittingGuard",
    "Pair",
    "Realisation",
    "RunReport",
    "Series",
    "Side",
    "Signal",
    "Strategy",
    "Sink",
    "TrialRegister",
    "Verdict",
    "alerting_stack",
    "backfill",
    "by_domain",
    "by_source",
    "by_strategy",
    "calibration",
    "deflated_sharpe_ratio",
    "compounded",
    "default_strategies",
    "expected_max_sharpe",
    "load_equity",
    "load_series",
    "measure_trial_dispersion",
    "min_track_record_length",
    "no_skill_dispersion",
    "overall",
    "run_once",
    "sharpe",
    "status",
    "summary",
]
