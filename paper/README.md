# The paper-trading record

This directory is written by the scheduled job in
[`.github/workflows/paper-trading.yml`](../.github/workflows/paper-trading.yml),
which runs `python -m services.decisions run --home paper` once a day and
commits whatever changed.

| file | what |
|---|---|
| `decisions.jsonl` | every decision, and every outcome, in the order they were known |
| `trials.jsonl` | every distinct strategy variant attempted, ever |
| `prices/` | the runner's own price cache — refetched daily, not committed |

**Do not edit these by hand.** The journal is append-only and the git history
is what makes that property real rather than aspirational: an entry cannot be
quietly revised without the revision itself being a commit. That is the entire
defence against a track record improving in hindsight.

`trials.jsonl` matters more than it looks. It is the denominator in the
overfitting correction, and it has to survive restarts — a register that lives
only in memory reports one trial forever no matter how many variants were
actually tried.

To see where things stand:

```bash
python -m services.decisions status --home paper
```

Experimenting locally? Use the default home (`data/paper/`, gitignored) so the
canonical record stays clean:

```bash
python -m services.decisions run          # writes to data/paper/
python -m services.decisions backfill     # replays history, refuses a live journal
```

Nothing in here has been traded. See the
[caveats](../services/decisions/README.md#caveats-worth-carrying) before
reading anything into it — in particular, three instruments on the same days
are not three independent observations.
