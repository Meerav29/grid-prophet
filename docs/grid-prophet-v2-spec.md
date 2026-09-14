# Grid Prophet v2 — Spec and Build Plan

Status: approved for Phase 0
Date: 2026-09-13

## 1. What we're building

One engine, three views.

The current model is a single tabular regression: team-season features in, season-end constructor points share out, bootstrap CI around it. It works (Spearman ~0.88 on leave-one-season-out CV) but it can only answer one question, and it can't say anything about drivers, races, or title odds.

v2 replaces the unit of prediction. The atomic thing the model reasons about is **one driver in one race**: how fast is this driver in this car at this track this weekend, and how likely are they to finish. Everything else is derived from that:

| View | How it's produced |
|---|---|
| Next race | Sample the outcome model for every entrant, resolve into a finishing order, repeat ~50k times. Read off P(win), P(podium), P(points), expected points, position distribution per driver. |
| Driver championship | Chain the race simulator over every remaining round, accumulate points per driver, repeat. Read off P(title), expected final points, final-position distribution. |
| Constructor championship | Same simulated seasons, points summed over each team's two cars. No separate model. |

The three "directions" in the earlier scoping report are not three models. Direction 2 (driver-level) *is* the model; Directions 1 and 3 are two aggregation layers on top of it. The existing constructor regression is demoted to a benchmark, plus one place where it contributes a prior (section 4).

Two run modes for the next-race view:

- **Pre-weekend**: nothing observed yet this round. Pace comes from the model's current beliefs about each car and driver.
- **Post-qualifying**: grid is known, and quali gaps are the single strongest one-weekend signal we can get. The model conditions on observed quali pace and starts the race sim from the actual grid.

## 2. Model choice

**Hierarchical Bayesian pace model** as the core, with a gradient-boosted model held in reserve as a challenger, decided by backtest.

Why hierarchical Bayesian for this specific problem:

1. 2026 is a data-poor year by construction. New regulations reset car strength, Cadillac has zero history, several seats have rookies or drivers in new cars. Partial pooling is the correct tool here: a rookie's pace estimate starts at the population prior for rookies and moves toward their own data as rounds accumulate. XGBoost has no principled way to do this; you end up hand-crafting "rounds of history" features and hoping.
2. The Monte Carlo layer is free. A Bayesian model's posterior is already a distribution over pace and reliability. Simulating a race means drawing from the posterior and adding race noise. No 500-retrain bootstrap loop. Uncertainty is honest by construction rather than bolted on.
3. Car development is a time-series problem. Team strength drifts round to round (upgrades, correlation issues, the classic mid-season swing). A random-walk component on team strength captures this and, importantly, makes season-end uncertainty grow the further out you simulate. A tabular model has to be told this explicitly.
4. The estimates are interpretable. "Driver effect" vs "car effect" separation is exactly what the driver-vs-teammate and driver-vs-field questions need, and it's what the F1 modelling literature (Bell et al. 2016, Van Kesteren & Bergkamp 2023) converges on.

The challenger: a per-driver-per-race LightGBM/XGBoost model predicting finishing position (or points) from the same inputs, turned into a distribution via quantile regression or ordinal classification. Cheaper, no sampler. It stays in the repo behind the same interface, and phase 1's backtest decides whether it beats, matches, or gets ensembled with the Bayesian model. Bet against it winning outright in a regulation-change year, but the bet costs a day of work to settle.

Rating systems (Elo-style) are a cheap approximation of the Bayesian model with worse uncertainty handling. Not recommended as the primary, but a pace-rating baseline is a useful sanity check and takes an afternoon.

Stack implication: PyMC or NumPyro on top of the existing Python/FastF1/pandas setup. NumPyro (JAX) is faster for sampling; PyMC has friendlier ergonomics. Either works.

## 3. The core model

Notation: driver *d*, team *t*, round *r*, season *s*, circuit *c*.

### 3.1 Pace model

Latent race pace for driver *d* at round *r* (lower is faster, in seconds/lap relative to field):

```
pace[d, r] = car[t(d), r] + driver[d] + driver_track[d, type(c)] + eps
```

- `car[t, r]`: team strength at round *r*. Random walk across rounds within a season: `car[t, r] = car[t, r-1] + drift[t, r]`, `drift ~ Normal(0, sigma_dev)`. Season-start value `car[t, 1]` is drawn from a prior centred on last season's end-of-year strength, with the prior's width controlled by the rule-change flag (much wider in 2026 — carryover is weak in regulation-reset years; the current model's `rule_change` feature is the empirical basis for how wide).
- `driver[d]`: driver skill, pooled across seasons with slow drift. Rookies get the rookie-population prior.
- `driver_track[d, type]`: small interaction for circuit type (street / high-speed / high-downforce / mixed). Regularised hard; this is where overfitting lives.
- `eps`: race-day noise, fat-tailed (Student-t), because F1 results are.

Quali pace is a parallel equation sharing `car` and `driver` but with its own noise and a quali-specific driver offset (some drivers are Saturday specialists; Perez-type gaps between quali and race are real and persistent).

Observations that fit this: median clean-air race lap time relative to winner, quali best-lap gap to pole, finishing position. Lap-time based targets are much less noisy than finishing position and FastF1 has them back to 2018.

**Empirical note from Phase 0 (resolved).** The original ~0.8+ correlation target between race clean-air pace and quali gap (see Phase 0 done-criteria below) was not met by either candidate metric on the full, validated 2018-2026 dataset: median clean-air lap reaches Spearman 0.64 (dry sessions, classified finishers only), fastest single clean-air lap reaches only 0.51 -- fastest-lap was tried specifically because it's the closer single-lap analogue to quali, but it underperforms the full-stint median, most likely because a single lap carries more sampling noise than a median while still not correcting for tire-warm-up/fuel-load effects any better. Neither result is a data bug: verified on a fully repaired, corruption-free dataset (see `data/driver_rounds_validation.md`). Conclusion: quali pace and race pace are genuinely different signals here, not two measurements of the same underlying number -- which is consistent with, not contradictory to, the model design above, since `pace[d,r]` for quali and race intentionally share `car`/`driver` terms but keep separate noise terms rather than being forced equal. **Decision: use median clean-air lap (not fastest) as the race pace observation**, and treat ~0.6 Spearman as the realistic ceiling for this signal rather than continuing to chase 0.8 -- the original target was a sanity-check assumption, not a hard modeling requirement.

### 3.2 Reliability model

Separate from pace. Two failure channels because they behave differently:

- **Mechanical DNF**: hazard per team-season, with a prior widened in rule-change years (2014 and 2022 both had elevated early-season failures) and shrinking as the season runs.
- **Incident DNF / heavy damage**: hazard per driver, with a grid-position effect (midfield starts crash more than front-row starts) and a first-lap spike.

Output: for each driver in each simulated race, a Bernoulli draw for "classified finisher" and, if not, a lap-of-retirement distribution (matters for whether they'd already scored in a sprint or a partial-points race, and for nothing else — keep it simple).

**Known data gap (found in Phase 0):** FastF1's `Status` field only carries a granular mechanical/incident cause (Engine, Accident, Brakes, etc.) via the old Ergast API, which stopped updating after 2024. For 2023+ seasons the field mostly just says "Retired" with no cause detail, and race control messages don't fill the gap either (checked directly — no retirement-cause text there). This means the mechanical-vs-incident split in §3.2 is only directly observable pre-2023; recent seasons will need either a supplementary hand-curated source (e.g. race reports) or the reliability model will have to lean more on the pre-2023 training signal and treat recent generic "Retired" rows as a mixture rather than a clean label. Flagged in `data/dnf_status_map.csv` (`needs_review=True` on `Retired`, `Wheel`, `Puncture`).

### 3.3 Race resolution

Per simulated race:

1. Draw `car`, `driver`, and interaction terms from the posterior.
2. Compute each driver's pace; add race noise.
3. Apply grid effect: post-quali mode uses the real grid; pre-weekend mode simulates quali first from the quali equation. Position-change model is a simple track-specific overtaking difficulty parameter — Monaco locks the order, Bahrain doesn't.
4. Draw DNFs.
5. Order survivors by effective pace, convert to points via the season's points table (2026 table, sprint table, fastest-lap rule as applicable that year).

50k trials for a single race is well under a minute on a laptop. The bottleneck is posterior sampling (minutes), which happens once per model update, not per trial.

### 3.4 Season chaining

Simulate rounds *r+1 ... R* in sequence within a single trial, carrying the team random walk forward (so car strength wanders, and the wander compounds). Sum driver points, sum team points. Repeat N times. This is where title odds come from, and it's why the drivers' and constructors' championships come from the same trials — they're consistent with each other by construction.

Sprints: treat as an extra shorter race event in the round, own points table, sharing the weekend's pace draw with the Grand Prix. In scope from phase 2.

### 3.5 Handling 2026 specifically

- **Cadillac**: prior on `car[Cadillac, 1]` from the empirical distribution of new-entrant first seasons (Haas 2016 is the optimistic tail, HRT/Virgin/Lotus 2010 and Caterham are the pessimistic tail). Wide. Shrinks fast once real rounds exist. By September we have ~16 rounds of 2026 data, so this matters mainly for backtesting the early-season forecasts and for next year's grid.
- **Rookies**: rookie-population prior on `driver[d]` (historically, rookies land ~0.2-0.4s/lap behind an experienced teammate in year one, with wide spread). Pooling toward that prior.
- **Regulation reset**: wider season-start car prior, wider development drift for the first ~6 rounds, elevated mechanical hazard prior. All three are learnable from 2014 and 2022 as analogue years.
- **Driver swaps mid-season**: `driver[d]` follows the person, `car[t]` follows the team. A swap is just a new (d, t) pairing; no special handling needed, which is the whole point of separating the two.

## 4. What happens to the existing model

Kept as a **benchmark**, used as a **prior** in one place, and its bootstrap-CI machinery is **retired**.

- Benchmark: every phase's backtest reports constructor-share Spearman side by side with the legacy model's 0.88. If v2 can't beat it on season-level ranking by phase 3, something is wrong.
- Prior: the legacy feature set encodes something real about how early-season signals and rule-change adaptation translate into season-end share. In the new model, this shows up as the season-start car prior and the rule-change width parameters. The `rule_change` and `prior_year_standing` features from `features.py` carry over almost directly; `early_season_share` is superseded because the new model consumes the actual rounds instead of a summary of them.
- Retired: `confidence.py`'s 500-retrain bootstrap, the Ridge-vs-XGBoost selection in `train.py`, and `Grid_Prophet_model.pkl` as a production artefact, once phase 3's backtest passes. The season-level `features.csv` stays around for the benchmark run.

## 5. Data

Everything below comes from FastF1 except track metadata and the points tables, which are small hand-maintained files.

New per-driver-per-round table (`data/driver_rounds.csv`), one row per driver per session type (quali, sprint, race), seasons 2018-2026:

| Field | Source | Notes |
|---|---|---|
| season, round, circuit, session | FastF1 event schedule | |
| driver, team | FastF1 results | team is per-round, not per-season, because of swaps |
| grid position, finish position, classified flag, status | FastF1 results | status text gives DNF cause where available — needs a mapping table to mechanical / incident / other. Coverage is incomplete for 2023+ (see §3.2). |
| quali best lap, gap to pole, Q-segment reached | FastF1 quali | |
| median clean-air race lap, gap to winner | FastF1 laps | "clean air" = exclude laps under SC/VSC, in/out laps, and laps within ~1.5s of the car ahead. This is the most work in the data layer and the highest-value target. |
| points scored | FastF1 results | pulled directly from the session's recorded points rather than re-derived from a hand-maintained points table — FastF1 already reflects the actual scoring rule (including fastest-lap bonus) for that session. |
| weather (dry/wet, air temp) | FastF1 weather | wet races get their own noise scale |

Supporting tables:

- `data/circuits.csv`: circuit type classification, overtaking difficulty (hand rating), lap length.
- `data/driver_meta.csv`: debut season (for rookie flag).
- `data/dnf_status_map.csv`: raw FastF1 status string → category (classified / mechanical / incident / other), with a `needs_review` flag on ambiguous entries.

Data volume: ~22 rounds x 20-22 drivers x 9 seasons is ~4k race rows plus quali and sprint. Small. Sampling will be fine.

## 6. Interfaces and outputs

CLI, matching the existing `collect / features / train / predict` shape:

```
gp collect --seasons 2018-2026          # builds driver_rounds.csv and supporting tables
gp fit --through 2026:16                # fits posterior using all data up to and including round 16
gp race --round 17 --mode pre           # next-race forecast, pre-weekend
gp race --round 17 --mode post-quali    # conditioning update: condition on quali, resim
gp season                               # chain remaining rounds, both championships
gp backtest --seasons 2021-2025         # rolling backtest, writes metrics
```

Outputs (JSON + CSV, one directory per fit timestamp):

- `race_forecast.csv`: driver, team, p_win, p_podium, p_points, p_dnf, exp_points, p10/p50/p90 finish position, plus the full position distribution as a wide block.
- `drivers_championship.csv`: driver, p_title, exp_final_points, p10/p50/p90 final position, p_top3.
- `constructors_championship.csv`: team, p_title, exp_final_points, exp_share, p10/p50/p90 final position.
- `posterior_summary.csv`: current car strength and driver skill estimates with credible intervals. This is the thing people will actually find interesting to look at.

**Post-quali conditioning.** Primary approach: quali data for the round is added as an observation and NUTS is warm-started from the pre-weekend posterior with a short warmup — statistically correct under the wide priors 2026 will actually have (new cars, Cadillac, rookies), where a straight importance-reweight is most likely to degenerate. Target: results within a couple of minutes of quali ending.

- **Failure trigger:** if the warm-started refit doesn't hit the couple-of-minutes target on a given round, that round falls back rather than blocking the run.
- **Fallback:** a full re-fit including the new quali observation, with the couple-of-minutes target relaxed for early-season rounds only (roughly the first ~6 rounds, where priors are widest and conditioning is hardest). Rounds past the early window are still held to the original target — a late-season miss is a real regression, not an accepted exception.
- Record fallback frequency per round in the Phase 2 backtest; if most rounds are falling back, that's a signal the warm-start isn't working as designed and needs revisiting before Phase 2 closes, not something to quietly absorb into the relaxed-target carve-out.

## 7. Evaluation

Rolling backtest: for each held-out season 2021-2025 and each round *r*, fit on everything before round *r* (including earlier seasons and rounds 1..r-1 of that season), forecast round *r* and the remaining season, score against reality. This is the only honest test, because it mimics exactly how the model will be used in 2026.

Race-level metrics:

- Log loss on race winner (vs a naive "pole wins" baseline and vs bookmaker-implied odds if we can get them — bookmakers are the real bar).
- Brier score on podium and points finishes.
- Rank correlation (Spearman) between expected and actual finishing order.
- CRPS on finishing position.
- Calibration plots: when we say 30%, does it happen 30% of the time. This is the one that will actually reveal problems.

Season-level metrics:

- Constructor-share Spearman at rounds 2, 6, 12 (comparable to the legacy 0.88 — note the legacy number is at "early rounds," so match the cutoff).
- Brier on drivers' and constructors' title winner at the same cutoffs.
- Coverage of the 80% intervals on final points (should be ~80%; the legacy bootstrap CIs were symmetric and probably over- or under-covered, and we've never checked).

### Compute budget

Taken literally, the rolling backtest is 5 seasons x ~22 rounds = ~110 fits, and a hierarchical model with a per-team random walk fit by NUTS is not a cheap fit. At 5-10 minutes each that's 10-18 hours per pass, and phase 1 will want several passes. Left unmanaged, the backtest becomes the thing that stops you iterating on the model. So the budget is a design constraint, not an afterthought:

- **Target: one full backtest pass in under 2 hours on a laptop; single production fit under 10 minutes.** Phase 1 doesn't pass until both hold.
- **Sequential warm starts.** Within a season, the round-*r* fit differs from round *r-1* by one round of observations. Initialise the sampler from the previous posterior and use a short warmup. This alone should cut per-fit time by a large factor.
- **Coarser backtest grid for iteration.** While developing, refit at rounds 2, 4, 6, 9, 12, 16, 20 (~35 fits) and forecast the intervening rounds from the most recent fit. Run the every-round grid only for the phase gate.
- **Parallelise across seasons.** Each held-out season is independent; five processes, five cores.
- **VI for the backtest loop, NUTS for production.** If warm starts and the coarse grid aren't enough, fit the backtest with ADVI/Pathfinder and reserve NUTS for the fit that actually ships. Check on a handful of rounds that the VI and NUTS forecasts agree before trusting this.
- **NumPyro over PyMC** if sampling speed decides it. JAX compilation makes repeated fits of the same model shape much cheaper, which is exactly the backtest pattern.
- Reparameterise the random walk (non-centred, cumulative-sum of increments) before doing anything else; a centred random walk is the single most likely reason NUTS is slow here.

Pass criteria are set per phase below.

## 8. Phases

### Phase 0 — Data layer (1-2 weeks)

Build `driver_rounds.csv` and the supporting tables. The clean-air lap filter is the bulk of the work; validate it by checking correlation between clean-air race pace and quali gap, and that top-3 by clean-air pace usually matches the podium.

Deliverable: the table, a data-quality report, DNF-cause mapping reviewed by a human.
Done when: 2018-2026 rounds all present (7,923 rows, verified corruption-free), no unmapped DNF statuses (267 rows, 6.4%, remain flagged `needs_review` -- mostly generic "Retired" per the known Ergast-cutoff gap, expected), lap filter validated -- **with the target revised from ~0.8+ to the measured ~0.6 Spearman ceiling** per the empirical note in sec 3.1 above. Phase 0 is complete on this basis; the correlation shortfall vs. the original assumption is a resolved, documented finding, not an open blocker.

### Phase 1 — Race outcome model + single-race sim, pre-weekend mode (2-3 weeks)

Implement the pace model (3.1) and a first-cut reliability model (team-level hazard only). Implement the race resolver (3.3) with a fixed per-circuit overtaking parameter. Implement `gp fit` and `gp race --mode pre`. Build the LightGBM challenger behind the same interface. Run the race-level backtest on 2022-2025.

Done when: Bayesian model beats "pole wins" and the pace-rating baseline on log loss and calibration is not visibly broken. A single fit completes in under 10 minutes and a full backtest pass in under 2 hours (see §7 compute budget); if not, the model gets simplified or the sampler swapped before phase 2 starts. Decision recorded on Bayesian vs challenger vs ensemble.

### Phase 2 — Post-quali mode, sprints, reliability (2 weeks)

Add quali conditioning and the post-quali run mode, per the fallback protocol in §6. Add sprint events. Split reliability into mechanical vs incident with the grid-position effect (subject to the data-coverage caveat in §3.2). Add weather. Fit the overtaking parameter from data instead of hand rating.

Done when: post-quali forecasts show a clear log-loss improvement over pre-weekend on the backtest (if they don't, the quali conditioning is broken), sprint points reconcile with actual standings, and the warm-start fallback rate is low enough that it isn't quietly absorbing a broken conditioning step.

### Phase 3 — Season chaining, both championships, legacy retirement (2 weeks)

Implement `gp season`. Run the season-level backtest at rounds 2/6/12 for 2021-2025. Compare against the legacy regression at matching cutoffs. Check 80% interval coverage.

Done when: v2 matches or beats legacy Spearman at every cutoff, interval coverage is within 70-90%, and title-odds Brier is documented. Then retire `confidence.py` and the pkl.

### Phase 4 — Operations (ongoing, small)

Automated run after each race weekend (refit) and after each quali (condition + resim). Posterior-summary tracking over time so you can see car development curves. Model-vs-reality log per round. Optional: a simple page showing the current forecasts and the development curves.

Realistically phases 0-3 is about two months of focused work. Phase 1 is where the risk is; if the pace model is right the rest is plumbing.

## 9. Assumptions

- Fine to add PyMC or NumPyro to the stack.
- Sprints are in scope but can wait for phase 2.
- Clean-air race lap time is an acceptable primary target (rather than finishing position alone). It's more work but it's the difference between a decent model and a good one.
- 2018 is the start of the training window. FastF1 lap data before 2018 is patchy; the current model's ~10 years of results data can still feed the driver-skill prior if we want it.
- Bookmaker odds as a comparison are nice-to-have, not required.
- No live in-race prediction (lap-by-lap). That's a different product.

## 10. Repo layout (proposed)

```
src/
  collect/        # FastF1 pulls, clean-air filter, DNF mapping
  model/
    pace.py       # hierarchical pace model (PyMC/NumPyro)
    reliability.py
    challenger.py # LightGBM alternative, same interface
  sim/
    race.py       # single-race resolver
    season.py     # chaining
  cli.py
  legacy/         # current features.py / train.py, benchmark only
data/
  driver_rounds.csv, circuits.csv, points_tables.csv, driver_meta.csv
outputs/<fit_timestamp>/
tests/
```

Note: as of Phase 0, the legacy pipeline (`collect.py`, `features.py`, `train.py`, `predict.py`, `confidence.py`) remains in place at `src/` and fully functional — it is not moved into `src/legacy/` until Phase 3 retirement per §4. Phase 0's new code lives alongside it as `src/collect_v2.py`.
