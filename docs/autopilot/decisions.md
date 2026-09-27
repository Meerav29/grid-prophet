# Decisions made without you — grid-prophet

Every entry here is a design question the v2 spec did not settle, which a build
routine resolved on its own so the slice could ship. Appended by the routines;
newest last.

Per the design's ambiguity policy, a routine picks the most defensible option,
records it here and in the PR body, and proceeds. It does not stop, and it does
not guess silently.

**Read this before reviewing the `auto/queue` diff.** A decision you disagree
with is cheaper to catch here than in the code.

Format:

```
## <date> — slice-<n>, PR #<n>
Question:    <what the spec left open>
Chosen:      <what was done>
Rejected:    <the alternative, and why not>
Blast radius: <files or functions affected, and how reversible>
```

---

## 2026-09-19 — slice-1, PR #5
Question:    §6 calls the post-quali target "a couple of minutes", relaxed for "roughly the first ~6 rounds", but names no numbers and does not say what the clock measures.
Chosen:      120s warm-start target, 300s relaxed allowance, early window = rounds 1–6 inclusive. Both measured as total wall clock from the start of conditioning (warm-start attempt *plus* fallback), because that is when results actually land for the user.
Rejected:    Timing the fallback refit alone — a round that burned 119s failing a warm start would be scored as if that time were free, hiding exactly the slow rounds §6 wants counted. Also rejected: leaving the numbers unnamed, which makes "missed its target" untestable.
Blast radius: Three constants at the top of `src/model/conditioning.py`. Changing them changes which rounds are *recorded* as fallbacks or regressions, never a forecast.

## 2026-09-19 — slice-1, PR #5
Question:    §3.3 says post-quali mode uses "the real grid" but not where that grid is read from.
Chosen:      `grid_position` on the round's race rows (FastF1's `GridPosition`, penalties already applied). 0 (pit-lane start) and missing values rank to the back; ranks are dense 1..n; a round with no usable grid raises.
Rejected:    Quali classification order — it ignores grid penalties, so it is not the grid the race started from. Also rejected: silently simulating a grid when the real one is missing, which would label a pre-weekend forecast as post-quali.
Blast radius: `_real_grid_for_round` in `src/cli.py`, reached only by `--mode post-quali`.

## 2026-09-19 — slice-1, PR #5
Question:    §6 says quali "is added as an observation" but does not say what happens to the round's race row during conditioning.
Chosen:      Hold out the round's race observations, keep its quali, and keep the round on the random-walk axis (`build_pace_data(hold_out_race_round=...)`).
Rejected:    Fitting through round N−1 and treating quali as an extra round — that drops the round's own `car[team, round]` node, so the forecast would come off a stale round. Letting the race rows in is leakage and was never an option.
Blast radius: One new optional argument to `build_pace_data`; every existing caller's behaviour is unchanged.

## 2026-09-27 — slice-2, PR #6
Question:    §3.4 says the sprint has its "own points table" but `data/points_tables.csv` held a single unlabelled block, and the sprint table is not one table — 2021 awarded 3/2/1 to the top three, 2022 onward 8..1 to the top eight.
Chosen:      Widened the CSV to `event,season_from,season_to,position,points` and keyed lookups on both. `load_points_table(event="grand_prix", season=None)` keeps the old call site working and defaults to the most recent block, which is what a current-season forecast wants.
Rejected:    A second file (`sprint_points.csv`) — §5 calls these "the points tables", one hand-maintained thing, and two files drift apart. Also rejected: a single season-blind sprint table, which would quietly misprice every 2021 sprint by a factor of ~2.7 in the backtest and give no error while doing it.
Blast radius: `data/points_tables.csv` and `load_points_table` in `src/sim/race.py`. Both existing callers (`cli.forecast_weekend`, `backtest.forecast_round`) pass no arguments and get exactly the table they got before.

## 2026-09-27 — slice-2, PR #6
Question:    §3.4 says a sprint is "an extra shorter race event". It does not say what "shorter" changes in the resolver.
Chosen:      One constant, `SPRINT_DISTANCE_RATIO = 100/305`, scaling exactly two things: `dnf_prob` (a third of the running time is a third of the exposure to a failure) and `overtaking_difficulty` (which makes the grid-lock penalty per slot 1/ratio larger — the same circuit is harder to pass on with a third of the laps). Race noise is left alone.
Rejected:    Scaling race noise too. Per-lap noise would average down over distance and event-level shocks (a safety car, a bad start) would not; §3.4 does not say which dominates, so scaling it either way is a modelling claim the data has not been asked about. Also rejected: a free-standing set of sprint-only parameters, which is three more numbers nobody has fitted.
Blast radius: `SPRINT_DISTANCE_RATIO` and `sprint_inputs_from` in `src/sim/race.py`. Sprint-only: nothing reads either on a non-sprint round, so no Grand Prix forecast can move.

## 2026-09-27 — slice-2, PR #6
Question:    Both events are resolved from one RNG. The spec does not say in which order.
Chosen:      The Grand Prix draws first, then the sprint — the reverse of the real weekend. Nothing carries between the two events, so the order is free, and taking Sunday's draws first means adding sprint support cannot move any existing Grand Prix forecast by a single trial.
Rejected:    Sprint first, matching the calendar. It reads better and buys nothing: it would shift every sprint round's Grand Prix numbers off their pre-slice values for no modelling reason, and make "rounds without a sprint are unaffected" the weaker claim that only *non-sprint* rounds are unaffected.
Blast radius: Two statements in `simulate_weekend`. Verified end to end: the round-2 Grand Prix forecast is byte-identical with the sprint rows present and with them removed.

## 2026-09-27 — slice-2, PR #6
Question:    §6 fixes `race_forecast.csv`'s columns. It does not say where a sprint forecast goes.
Chosen:      A sibling `sprint_forecast_<season>_<round>.csv`, same columns, written only on sprint rounds. `race_forecast.csv` keeps exactly the §6 shape.
Rejected:    Sprint columns bolted onto `race_forecast.csv` — it breaks §6's stated column list for one round type in three, and leaves `exp_points` ambiguous between "Sunday" and "the weekend". Also rejected: one long file with an `event` column, which would change the row count of an existing output.
Blast radius: One extra `to_csv` in `cmd_race`. Readers of `race_forecast.csv` see no change.

## 2026-09-27 — slice-2, PR #6
Question:    Should sprint rows (`session_type == "S"`) also feed the pace model as extra race observations?
Chosen:      No. `build_pace_data` still reads `R` and `Q` only; sprint rows are used for the sprint calendar and the sprint grid, not as pace data.
Rejected:    Adding them as observations. A sprint's clean-air pace is a different quantity (no pit stop, one tyre compound, a third of the fuel burn), so it needs its own offset and noise scale in the likelihood, which is a pace-model change nothing in slice-2's criteria asks for, and it would change every Grand Prix forecast on every sprint round. Worth doing deliberately, with the backtest available to say whether it helped.
Blast radius: None — this is the existing behaviour, recorded because a reviewer will otherwise wonder whether it was considered.
