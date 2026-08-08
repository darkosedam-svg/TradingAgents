# Golden sets

**These files are seeds, not the golden sets.** Each holds 10–12 items that
establish the format and cover the failure modes worth designing around. Gate A
will not pass on them, and it should not: `gate_a` fails any run with fewer than
150 items.

The real sets are 150–300 hand-labelled items each, drawn from your own traffic.
Nothing else works — a set of items you wrote is a set of items shaped like your
prompt, and it will report parity that a production stream does not.

## Format

One JSON object per line:

```json
{"id": "snt-001", "input": <string|object>, "expected": {...}, "notes": "optional"}
```

`input` is rendered into the user message by `harness.render_input`, so the
framing lives in code and the labels stay pure data. `expected` holds only the
fields listed for that task in `metrics.TASK_SPECS` — `rationale` and other free
text is not scored.

## Building the real sets

1. **Sample from production, not from memory.** Pull a week of real traffic per
   task and take a random sample. Do not filter for interesting items first;
   that is how a set ends up with no easy cases and a misleading accuracy floor.
2. **Label before you look at any model output.** Labelling after seeing a
   prediction anchors you to it, and the resulting set will flatter whatever
   model produced it.
3. **Include negatives deliberately, at their real rate.** For `signal_parse`
   most chat messages are not signals; a set that is 90% valid signals will not
   catch a model that extracts a trade from every message. The seed file's
   `sig-004` and `sig-008` are the shape to aim for.
4. **Write the ambiguous ones down as ambiguous.** An item where a careful human
   would score 0.5 confidence belongs in the set with that expectation, not
   excluded for being unclear. Abstention behaviour is a thing being measured.
5. **Freeze it.** Once a set is used for a gate, changes to it are changes to the
   gate. Version the file and note the change in `docs/llm-layer-decisions.md`.

## Per-task notes

| file | items | must include |
|---|---|---|
| `sentiment.jsonl` | 150–300 | multi-asset headlines, already-priced-in events, non-crypto noise, truncated text |
| `news_triage.jsonl` | 150–300 | labelled **should-escalate** positives (Gate C reads recall off these), relevant-but-not-news items, unreadable stubs |
| `signal_parse.jsonl` | 150–300 | negatives at the production rate, percentage stops, range entries, position closes, ambiguous numbers |
| `token_meta.jsonl` | 150–300 | near-identical copycats, shared-theme-but-original launches, scam language without copycatting, empty metadata |

For `news_triage`, keep the "should have escalated" labels as a separate
reviewable column in your notes — Gate C's recall number is only as good as the
honesty of that label, and it is the one number in the whole plan that is worth
re-reviewing by hand every quarter.
