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

## 2026-09-27 — slice-2, PR #7
Question:    §3.4 says the sprint has its "own points table" but `data/points_tables.csv` held a single unlabelled block, and the sprint table is not one table — 2021 awarded 3/2/1 to the top three, 2022 onward 8..1 to the top eight.
Chosen:      Widened the CSV to `event,season_from,season_to,position,points` and keyed lookups on both. `load_points_table(event="grand_prix", season=None)` keeps the old call site working and defaults to the most recent block, which is what a current-season forecast wants.
Rejected:    A second file (`sprint_points.csv`) — §5 calls these "the points tables", one hand-maintained thing, and two files drift apart. Also rejected: a single season-blind sprint table, which would quietly misprice every 2021 sprint by a factor of ~2.7 in the backtest and give no error while doing it.
Blast radius: `data/points_tables.csv` and `load_points_table` in `src/sim/race.py`. Both existing callers (`cli.forecast_weekend`, `backtest.forecast_round`) pass no arguments and get exactly the table they got before.

## 2026-09-27 — slice-2, PR #7
Question:    §3.4 says a sprint is "an extra shorter race event". It does not say what "shorter" changes in the resolver.
Chosen:      One constant, `SPRINT_DISTANCE_RATIO = 100/305`, scaling exactly two things: `dnf_prob` (a third of the running time is a third of the exposure to a failure) and `overtaking_difficulty` (which makes the grid-lock penalty per slot 1/ratio larger — the same circuit is harder to pass on with a third of the laps). Race noise is left alone.
Rejected:    Scaling race noise too. Per-lap noise would average down over distance and event-level shocks (a safety car, a bad start) would not; §3.4 does not say which dominates, so scaling it either way is a modelling claim the data has not been asked about. Also rejected: a free-standing set of sprint-only parameters, which is three more numbers nobody has fitted.
Blast radius: `SPRINT_DISTANCE_RATIO` and `sprint_inputs_from` in `src/sim/race.py`. Sprint-only: nothing reads either on a non-sprint round, so no Grand Prix forecast can move.

## 2026-09-27 — slice-2, PR #7
Question:    Both events are resolved from one RNG. The spec does not say in which order.
Chosen:      The Grand Prix draws first, then the sprint — the reverse of the real weekend. Nothing carries between the two events, so the order is free, and taking Sunday's draws first means adding sprint support cannot move any existing Grand Prix forecast by a single trial.
Rejected:    Sprint first, matching the calendar. It reads better and buys nothing: it would shift every sprint round's Grand Prix numbers off their pre-slice values for no modelling reason, and make "rounds without a sprint are unaffected" the weaker claim that only *non-sprint* rounds are unaffected.
Blast radius: Two statements in `simulate_weekend`. Verified end to end: the round-2 Grand Prix forecast is byte-identical with the sprint rows present and with them removed.

## 2026-09-27 — slice-2, PR #7
Question:    §6 fixes `race_forecast.csv`'s columns. It does not say where a sprint forecast goes.
Chosen:      A sibling `sprint_forecast_<season>_<round>.csv`, same columns, written only on sprint rounds. `race_forecast.csv` keeps exactly the §6 shape.
Rejected:    Sprint columns bolted onto `race_forecast.csv` — it breaks §6's stated column list for one round type in three, and leaves `exp_points` ambiguous between "Sunday" and "the weekend". Also rejected: one long file with an `event` column, which would change the row count of an existing output.
Blast radius: One extra `to_csv` in `cmd_race`. Readers of `race_forecast.csv` see no change.

## 2026-09-27 — slice-2, PR #7
Question:    Should sprint rows (`session_type == "S"`) also feed the pace model as extra race observations?
Chosen:      No. `build_pace_data` still reads `R` and `Q` only; sprint rows are used for the sprint calendar and the sprint grid, not as pace data.
Rejected:    Adding them as observations. A sprint's clean-air pace is a different quantity (no pit stop, one tyre compound, a third of the fuel burn), so it needs its own offset and noise scale in the likelihood, which is a pace-model change nothing in slice-2's criteria asks for, and it would change every Grand Prix forecast on every sprint round. Worth doing deliberately, with the backtest available to say whether it helped.
Blast radius: None — this is the existing behaviour, recorded because a reviewer will otherwise wonder whether it was considered.

## 2026-09-29 — slice-3, PR #8
Question:    §3.2 widens the mechanical prior in rule-change years "and shrinking as the season runs", without saying what shrinks.
Chosen:      The *widening* shrinks. Mean and concentration decay linearly from the rule-change prior to the stable one over the first 8 completed rounds of that season (`RULE_CHANGE_WIDENING_DECAY_ROUNDS`), using the round count in the fit window.
Rejected:    Reading it as the posterior naturally tightening as data accumulates — true of every Beta-Binomial ever written, so the clause would say nothing, and it ignores "elevated **early-season** failures", the only words in the sentence carrying information.
Blast radius: `_mechanical_prior` and one constant in `src/model/reliability.py`. Changes how much a rule-change prior inflates a late-season team estimate; never a label, a count, or a data file.

## 2026-09-29 — slice-3, PR #8
Question:    §3.2 asserts "midfield starts crash more than front-row starts" and stops: no functional form, and no statement of which grid a pre-weekend forecast is supposed to read.
Chosen:      Three bands (1–4 / 5–14 / 15+) whose prior multipliers encode that ordering, shrunk toward pooled data by 150 pseudo-starts per band and normalised to a start-weighted mean of 1.0 so the bands redistribute the pooled incident rate rather than inflating it. The slot is the real grid post-quali and the ranking of the posterior-mean quali pace pre-weekend.
Rejected:    A continuous curve (implies a shape nothing has fitted); a fixed multiplier that never learns (an assertion, not an estimate); and moving the DNF draw inside `sim.race` so each trial's simulated grid feeds its own hazard — correct in principle, but it reworks the resolver for a second-order effect on a multiplier.
Blast radius: `GRID_BANDS`, `PRIOR_GRID_MULTIPLIER`, `_grid_multipliers` and `cli._hazard_grid`. Reversible to a constant multiplier by deleting `_grid_multipliers`.

## 2026-09-29 — slice-3, PR #8
Question:    §3.2 says post-2023 rows are "a mixture rather than a clean label" but gives no weights, and says "hazard per driver" without a season index.
Chosen:      A `needs_review` status that still names a system (Suspension, Puncture) leans 0.7 toward its mapped category; a causeless one ("Retired") splits on the observable era's own clean mechanical:incident ratio, or 0.5/0.5 when the window holds no clean labels. Driver incident counts pool across the whole fit window.
Rejected:    Dropping flagged rows — it discards most of 2023+ and biases the hazard down exactly where cars are least reliable. Hard 0/1 labels from the map's category — that is manufacturing labels, which §3.2 forbids. Per-driver-season pooling — ~24 starts is far too thin for a crash propensity.
Blast radius: `FLAGGED_LABEL_LEAN`, `channel_weights` and the `driver_stats` groupby. The weights are two lines; changing them changes counts, not structure.

## 2026-10-02 — slice-4, PR #10
Question:    §3.3 asks for a "track-specific overtaking difficulty parameter" and §5 calls the `data/circuits.csv` column a hand rating. Neither says what observable the fitted version is fitted to, nor on what scale the result lives.
Chosen:      Fit by inverting the resolver's own rule. `sim.race` reduces the parameter to `penalty_per_slot = GRID_LOCK_BASE_PENALTY_S / d` and orders cars by `race_pace + penalty_per_slot * (grid - 1)`, so the fit scores each candidate `d` by how many finishing-order *pairs* that ordering gets right, using the observed `gap_to_winner_median_clean_air_s` as pace and the observed `grid_position`. The argmax over a 60-point log-spaced grid on [0.5, 10] is the estimate.
Rejected:    Mapping a summary statistic (mean |finish − grid|) onto the 1–5 hand-rating range by linear rescaling. It needs an arbitrary calibration constant, it has no inverse in the resolver, and a planted parameter could not be recovered from it — the criterion asking for recovery within a tolerance is unanswerable under that design. Also rejected: a simulation-based method of simulated moments, which needs an assumed pace spread per circuit when the pace is sitting right there in `driver_rounds.csv` as an observed column.
Blast radius: `src/model/overtaking.py` (new), plus the `overtaking_data` argument threaded through `cli._circuit_info` / `cli.forecast_weekend` / `backtest.forecast_race_challenger`. Reversible by passing `overtaking_data=None`, which restores the hand rating exactly.

## 2026-10-02 — slice-4, PR #10
Question:    §3.3 does not say how much data a circuit needs before its fit is trusted over the hand rating, nor what happens at a circuit where the data cannot distinguish candidates.
Chosen:      Shrink on the difficulty scale by `SHRINKAGE_PRIOR_ROUNDS = 4` pseudo-rounds of the hand rating: `(n_rounds * raw + 4 * hand) / (n_rounds + 4)`. A round counts only with at least 6 classified finishers carrying a grid slot and a clean-air pace. Where several candidates tie exactly — real at a locked circuit, where everything below about `d = 1.5` reproduces the finishing order equally well — the tie breaks toward the hand rating. Both `raw_difficulty` and `shrunk_difficulty` are kept on the fit so a thin estimate is visible rather than laundered.
Rejected:    Shrinking on the penalty scale (`1/d`), which is the scale the resolver actually consumes but not the scale the hand rating is written on, so mixing the two there would mean the rating no longer means what §5 says it means. Also rejected: a hard minimum-rounds threshold with no shrinkage, which makes the parameter jump discontinuously the round a circuit crosses it.
Blast radius: Three constants at the top of `src/model/overtaking.py` and `fit_circuit`. Changing them changes how fast a circuit's fit overrides its rating, never the estimate itself.

## 2026-10-02 — slice-4, PR #10
Question:    Nothing in §3.3 says when the overtaking fit is computed, and the repo persists `reliability_data` in the fit artifact while this parameter has no such home.
Chosen:      Built inside `forecast_weekend` (and `forecast_race_challenger`) from rounds strictly *before* the round being forecast, via `build_overtaking_data(..., before_season=, before_round=)`. Callers may pass a prebuilt `OvertakingData` to override.
Rejected:    Persisting it alongside `reliability_data` in the fit directory. It would be the tidier home, but `gp fit --through S:R` and `gp race --round R` are separate invocations with separate windows, so a persisted fit would silently include the forecast round whenever a fit ran through it — exactly the leak that makes a backtest score itself. Also rejected: fitting over the whole file regardless of round, which is the same leak with no window at all.
Blast radius: `cli.forecast_weekend`, `backtest.forecast_race_challenger`. The cost is one grid search per forecast call (~190 pairs × 60 candidates per round, milliseconds); the benefit is that the backtest cannot see the future.

## 2026-10-03 — slice-5, PR #12
Question:    §5 says "wet races get their own noise scale" and stops. It does not say whether the wet scale is estimated independently of the dry one or tied to it, nor what prior it gets.
Chosen:      A *ratio*: `race_sigma_wet = race_sigma * race_sigma_wet_ratio`, with `race_sigma_wet_ratio ~ LogNormal(log 1.5, 0.4)`. Positive by construction, pooled toward the dry scale, centred above 1 because that is the direction §5 is pointing, and loose enough that about 15% of prior mass sits below 1.0 (its 5th percentile is about 0.78) — a window whose wet rounds were tidier than its dry ones can still say so.
Rejected:    An independent `HalfNormal("race_sigma_wet", 0.5)`. Wet rounds are a small minority of any fit window, a small minority, so an independent scale is estimated off a handful of races with nothing pulling it toward the bulk of the data. Also rejected: a fixed multiplier, which is an assertion rather than an estimate.
Blast radius: `WET_SIGMA_RATIO_PRIOR` and `WET_SIGMA_RATIO_SD` at the top of `src/model/pace.py`, plus one `if` block in `build_model`. A point mass at 1.0 restores the pre-slice likelihood (though not the graph, which still carries the extra parameter). The prior is a choice, not a fit — the backtest has not been asked whether it helps.

## 2026-10-03 — slice-5, PR #12
Question:    What a dry-only fit window should build, given the slice criterion asks for pre-slice behaviour "exactly".
Chosen:      Conditional graph construction (`_has_wet_races`). No wet race in the window means no `race_sigma_wet_ratio` and the graph *is* the pre-slice graph — same free variables, same initial point, logp equal to the last bit against `auto/queue`'s own `pace.py` (checked by hand on the toy fixture, see the PR body; the committed tests assert the free-variable set).
Rejected:    Always adding the parameter and letting it sit at its prior. Simpler code, wrong in a way that matters here: an unobserved parameter still occupies the sampler's state, changes the posterior's dimension and consumes random numbers, so every dry-only backtest round would shift for no modelling reason. "Exactly" would have to be softened to "statistically indistinguishable", which is not what the criterion says and not something a two-minute unit suite can check.
Blast radius: `_has_wet_races` and one `if` in `build_model`. The cost worth naming: a window's parameter space now depends on its weather, so a backtest crossing from a dry prefix into a wet one changes dimension at that round. Nothing warm-starts across that boundary today (`model.conditioning` warm-starts within a round, not across rounds).

## 2026-10-03 — slice-5, PR #12
Question:    §5 says "wet **races**". The pace model has two likelihoods and `is_wet` is recorded per session, so a wet-quali/dry-race weekend is a real and distinguishable thing in the data. Does quali get a wet scale too?
Chosen:      Race only. The flag is read off the race rows specifically, so a wet quali on a dry-race weekend correctly leaves the race likelihood alone, and a dry quali on a wet-race weekend correctly does not stop the race rows being wet.
Rejected:    A parallel wet scale on the quali likelihood. About eight more lines and it would plausibly help — a wet Q3 is at least as noisy as a wet race. But §5's sentence says races, it is a second modelling claim nothing has measured, and it would mean "dry-only reproduces pre-slice exactly" has to hold across two likelihoods instead of one. Worth doing deliberately with the backtest available to say whether it helped.
Blast radius: None beyond the slice — this is a scope boundary, recorded because a reviewer will otherwise wonder whether quali was considered. Left as a follow-up in the PR body.

## 2026-10-03 — slice-5, PR #12
Question:    The slice criterion says "Mark them unknown and treat as dry", which is two different representations of the same round, and does not say where the line between them falls.
Chosen:      The line is `wet_flags`. The CSV, `wet_state` and the validator's coverage report keep unknown distinct from dry; `wet_flags` — and only `wet_flags`, which is what `build_pace_data` calls — collapses it. So `collect_v2._weather_summary` now writes a blank rather than `False` for a session FastF1 has no weather for.
Rejected:    Treating unknown as dry at collection time, which is the pre-slice behaviour and makes the gap unmeasurable after the fact — you cannot tell a dry 2018 round from one FastF1 had no weather for. Also rejected: a third likelihood branch for unknown rounds, which is a noise scale fitted to "we don't know".
Blast radius: `_weather_summary` in `src/collect_v2.py`; `wet_state` / `wet_flags` / `weather_coverage` in the new `src/model/weather.py`. Existing `driver_rounds.csv` files are unaffected — they were written with `False` and still read as dry, which means the coverage report reads `not recorded 0` on any pre-existing file, truthfully, because that file genuinely cannot distinguish the two. Only data collected after this change carries the distinction.

## 2026-10-03 — slice-6, PR #TBD
Question:    §3.2 says post-2023 generic rows are "a mixture rather than a clean label", and the slice asks for the ratio to be clamped away from 0 and 1. Neither says how far away.
Chosen:      A flat floor, `CAUSELESS_RATIO_FLOOR = 0.05`, applied only at the ends: `clip(clean_mech / clean_total, 0.05, 0.95)`. The empirical ratio is untouched everywhere short of a 19:1 clean split, so the clamp changes a window's weights only where the old code produced a hard label. The all-flagged case still returns 0.5 and is not the floor applied twice.
Rejected:    Laplace smoothing the ratio (`(m+1)/(m+i+2)`), which is the more principled fix — it can never reach 0 or 1 and converges on the data — but it moves *every* window, not just the degenerate ones: it turns slice-3's documented 3:1 → 0.75 split into 0.667 and so changes hazard estimates on the full 2018–2026 window for a bug that only bites on short backtest prefixes. Also rejected: clamping to `[1 - FLAGGED_LABEL_LEAN, FLAGGED_LABEL_LEAN]` = [0.3, 0.7], on the appealing argument that a label naming nothing should never be more informative than one naming a system — it would cap a genuinely 9:1 era at 0.7, discarding real signal, and it contradicts slice-3's own 0.75 criterion.
Blast radius: One constant and three lines in `channel_weights` (`src/model/reliability.py`). Setting the floor to 0.0 restores pre-slice-6 weights exactly. Changes fractional counts in `team_stats` / `driver_stats` on one-sided windows, never a structure or a data file.

## 2026-10-03 — slice-6, PR #TBD
Question:    §3.2 says the lap-of-retirement distribution "matters for whether they'd already scored in a sprint" and stops. It does not say how a sprint's incident probability is derived from it, and §3.4's sprint paragraph says nothing about reliability at all.
Chosen:      A sprint carries the incident mass of the *opening* `distance_ratio` of a Grand Prix's laps: `sprint_incident_exposure` sums `lap_of_retirement_pmf("incident", 57)` over the first `round(0.3279 × 57) = 19` laps, giving 30/68 = 0.4412 against a distance share of 0.3279. The mechanical channel still scales with distance, because that channel is flat in lap number. The spike therefore has exactly one definition in the codebase, and `distance_ratio` stays the single distance knob — the incident factor is derived from it, not configured beside it.
Rejected:    A second constant (`SPRINT_INCIDENT_RATIO = 0.44`) set by hand. It duplicates the spike: `FIRST_LAP_HAZARD_MULTIPLIER` could then be changed without the sprint noticing, which is the drift that leaves a parameter inert in the first place. Also rejected: simulating laps so the spike resolves per-lap rather than as a share. That is the structurally correct answer and is what §3.2's distribution is really for, but the resolver draws one Bernoulli per driver per event with no lap-level simulation (slice-3's own note says so), and adding one is a resolver rewrite nothing in this slice's criteria asks for.
Blast radius: `sprint_incident_exposure` in `src/model/reliability.py` and the `if has_channels` branch in `sim.race.sprint_inputs_from`. Sprint DNF probabilities only — about +18% on a 0.06 mechanical / 0.05 incident driver. Returning `distance_ratio` from the new function restores the old numbers.

## 2026-10-03 — slice-6, PR #TBD
Question:    `RaceTrialInputs.dnf_prob` is the fold of both channels, so the resolver had no way to shorten them separately. Nothing in the spec says which representation the resolver should hold.
Chosen:      Carry the channels as two *optional* fields beside `dnf_prob`, which stays what `simulate_positions` draws against. A caller holding only a combined probability leaves them None and gets the pre-slice-6 flat scaling rather than an invented split. `cli.forecast_weekend` fills them; `backtest.forecast_race_challenger` does not, because it resolves the Grand Prix alone through `simulate_positions` and has no sprint to price.
Rejected:    Making the channels the primary fields and deriving `dnf_prob` as a computed property. Tidier, and it removes the chance of the fold disagreeing with its parts — but a dataclass property cannot be passed to the constructor, so every existing `RaceTrialInputs(dnf_prob=...)` call site and four test modules would have to be rewritten, for no change in behaviour. Also rejected: a single `incident_share` scalar, which assumes one split for the whole field when the mechanical channel is per team and the incident channel per driver.
Blast radius: Two fields on `RaceTrialInputs` and two keyword arguments in `cli.forecast_weekend`. The risk worth naming: `dnf_prob` and the channels are now two statements of the same quantity and nothing enforces that they agree — a caller could set channels inconsistent with the fold and the Grand Prix would use one while the sprint used the other. `tests/test_cli.py::test_both_channels_reach_the_resolver_and_fold_to_dnf_prob` asserts the production path keeps them consistent; the dataclass itself does not.
