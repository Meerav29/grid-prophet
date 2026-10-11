# Phase 2 status — Grid Prophet v2

Scope: spec section "Phase 2 — Post-quali mode, sprints, reliability"
(`docs/grid-prophet-v2-spec.md` §8). Written 2026-10-04, after the first full
backtest of the Phase 2 code.

## Bottom line

**Phase 2's backtest gate is cleared on the evidence the spec asks for, with
the caveats below.** On a rolling 2022-2025 backtest (88 rounds, every-round
grid, NUTS throughout), conditioning on quali improves race-winner log loss from
1.801 to 1.258, the warm-start fallback fired in 0 of 88 rounds, and no round
missed its time target.

What this document does *not* claim: that each Phase 2 slice improved the
forecast on its own. The backtest measures post-quali against pre-weekend. It
does not ablate sprints, the reliability split, weather or the overtaking fit.

## What was built

Slices 1-5 were built by the autopilot routines (`docs/autopilot/queue.md`) and
merged to `main` via PR #13. Slice 6 was merged separately as PR #14.

| Spec item | Slice | Status |
|---|---|---|
| Quali conditioning, `gp race --mode post-quali`, §6 fallback protocol | 1 (PR #5) | Merged |
| Sprint events sharing the weekend pace draw | 2 (PR #7) | Merged |
| Reliability split: mechanical vs incident, grid-position effect | 3 (PR #8) | Merged |
| Overtaking parameter fitted from data | 4 (PR #10) | Merged |
| Weather: wet races get their own noise scale | 5 (PR #12) | Merged |
| Slice-3 carry-forwards (channel-ratio clamp, first-lap spike in sprints) | 6 (PR #14) | Merged 2026-10-03 |

**The backtest ran on code without slice 6.** The worktree it ran from was cut
at `ccc20ad`, before PR #14 merged. Slice 6 touches `channel_weights` and the
sprint incident path, so it should barely move race outcomes, but the numbers
below are for slices 1-5.

The backtest harness gained a post-quali arm for this run (`gp backtest
--post-quali`, commit `44579b1`): for each scored round it calls
`condition_on_quali`, forecasts from the real grid, scores it with the same
metrics, writes every round's conditioning record to `conditioning_log.csv`, and
adds the paired comparison and fallback counts to `backtest_summary.json`.

## Data

`data/driver_rounds.csv` is gitignored and was regenerated for this run:
8,189 rows, 2018-2026 (2026 through round 16), 44 drivers in
`driver_meta.csv`. The clean-air filter reproduces Phase 0: median clean-air lap
against quali gap gives Spearman 0.630 on dry, classified rows (Phase 0: ~0.64).

One round needed a manual repair. **2025 R6 (Miami)** has no quali times
upstream: FastF1's results rows carry positions but empty Q1/Q2/Q3. Its quali
`best_lap_s` was filled from each driver's fastest lap in the session laps, and
its quali, sprint and race rows were appended by hand. Lap-derived times order
close pairs slightly differently from the official classification (NOR, PIA and
ANT sit within 0.002s). `collect_v2` was not changed, so a clean re-collect will
hit the same gap.

Collection also hit FastF1's 500-calls/hour limit repeatedly. A resume loop
(re-run, prune rounds with under 50% clean-air coverage unless flagged wet, wait
20 minutes) completed it. 2021 R12 (Belgium) legitimately has low clean-air
coverage — a one-lap rain race — which is why wet rounds are exempt from the
prune.

## Backtest results (rolling, 2022-2025)

Run: `gp backtest --seasons 2022-2025 --rounds all --method nuts --post-quali`,
saved to `outputs/phase2_2022_2025/`. 88 rounds scored, 88 fits plus 88
conditioning refits, 9,605 seconds (2.7 hours) on one laptop, sequential.

| Metric | Pre-weekend | Post-quali |
|---|---|---|
| Mean log loss (race winner) | 1.801 | **1.258** (delta -0.543) |
| Mean Brier (podium) | 0.0886 | **0.0661** |
| Mean Brier (points finish) | 0.1814 | **0.1494** |
| Mean Spearman (expected vs actual order) | 0.599 | **0.664** |

Baselines on the same rounds, pre-weekend: pole-wins 3.219, pace-rating 2.099,
XGBoost challenger 2.064 (log loss). Pre-weekend still beats all three.

By season (log loss, pre → post; Spearman, pre → post):

| Season | Rounds | Log loss | Spearman |
|---|---|---|---|
| 2022 | 21 | 1.615 → 1.431 | 0.572 → 0.585 |
| 2023 | 21 | 1.569 → 0.868 | 0.615 → 0.664 |
| 2024 | 23 | 2.070 → 1.506 | 0.649 → 0.738 |
| 2025 | 23 | 1.913 → 1.208 | 0.559 → 0.658 |

Post-quali beat pre-weekend on log loss in 75 of 88 rounds. Both log loss and
Spearman improved in every season; 2022 is the weakest year, and its Spearman
gain is small.

### Warm-start fallback (spec §6)

| | Result |
|---|---|
| Rounds conditioned | 88 |
| Fell back to a full refit | **0** |
| Regressions (missed the applicable target) | **0** |
| Warm-start time, mean / max | 72.8s / 108.7s (target 120s) |

By season, mean warm-start time was 49s (2022), 79s (2023), 93s (2024) and 69s
(2025). Time rises with the amount of data in the fit window, not with
early-versus-late season: rounds 1-6 averaged 62s and later rounds 76s. Nine
rounds ran over 100s, so the 120s target has about 10% headroom at the slow end
on this machine.

§6 says a high fallback rate means the warm start needs revisiting. A rate of
zero means that test cannot fail on this evidence — but it is also an untested
branch. The fallback path has unit tests (`tests/test_conditioning.py`) and has
never run in anger.

## Caveats

1. **Post-quali is a high-variance forecast.** The per-round standard deviation
   of winner log loss is 0.77 post-quali against 0.42 pre-weekend. It is much
   better on average and worse in 13 rounds, some badly: 2022 R13 (1.51 → 3.90),
   2022 R14 (1.57 → 3.81), 2024 R21 (2.18 → 3.75). Those are rounds where the
   real grid pointed confidently at someone who then did not win. The mean hides
   this. Calibration was not re-checked for post-quali, and Phase 1's
   overconfidence at the back of the field may be larger here.
2. **The size of the gain is mostly the grid.** Knowing the grid is a strong
   signal on its own; pole-wins is a weak baseline only because it is scored as
   a hard 0/1. Do not read 0.54 nats as the value of the warm-started refit
   versus a plain quali-conditioned forecast. The backtest cannot separate those.
3. **No leakage found, with two small advantages.** The forecast round's race
   rows are held out of the conditioning fit (checked: race observation counts
   are identical between the pre and post fits for 2025 R10 and 2022 R5; quali
   rows grow by 20). The pace fit reads only Q and R sessions, and the forecast
   never reads the round's wet flag. Two things give post-quali a slight edge
   without being a leak: the grid is read from the race rows' `grid_position`,
   which includes post-quali penalties and pit-lane starts, so it is known a
   little after quali ends; and the entrant list is built from the round's Q and
   R rows, so a driver who qualified but did not start is included.
4. **Pre-weekend did not regress.** 1.801 against Phase 1's 1.844, Spearman
   0.599 against 0.597. That is "no regression", not "improvement": the sampler
   settings and data window differ from Phase 1's run.
5. **The 2021 and earlier format.** Not exercised; the backtest starts in 2022.
6. **Single run, single seed per round.** No confidence interval on the 0.54
   delta. The paired per-round differences are available in
   `backtest_metrics.csv` for anyone who wants one.

## What is still open

- A pre-weekend-only vs. slice-by-slice ablation, if it matters which slices are
  earning their keep.
- Post-quali calibration, and whether the confident misses above share a cause.
- Re-run with slice 6 included, to confirm it leaves race outcomes unchanged.
- `collect_v2` should fall back to session laps when quali results carry no
  times (Miami 2025), so the repair does not have to be redone by hand.
- Slice-4 carry-forwards from `docs/autopilot/queue.md`: a test guarding the two
  overtaking-fit call sites, and fitting only the forecast circuit rather than
  every circuit on each forecast.
- Phase 3: `gp season`, both championships, and the comparison against the
  legacy 0.88.
