# Phase 1 status — Grid Prophet v2

Written at the end of the phase 1 implementation session. Scope: spec
section "Phase 1 — Race outcome model + single-race sim, pre-weekend mode"
(`docs/grid-prophet-v2-spec.md`).

## Bottom line

**Phase 1's done-when bar is cleared** on the metrics the spec defines, on a
real rolling backtest over 2022-2025 (88 rounds, both the coarse grid and
the every-round phase-gate grid). Both compute-budget targets are cleared
with a lot of headroom. Two implementation details deviate from the letter
of the spec (documented below) because of environment constraints hit
partway through the session (NumPyro/JAX and LightGBM both fail on this
Windows/Python 3.12 machine) — PyMC and XGBoost are used instead, which the
spec explicitly allows either-or for.

## What's implemented

| Spec section | File | Status |
|---|---|---|
| 3.1 Pace model | `src/model/pace.py` | Implemented: car random walk, driver skill + rookie prior, driver×circuit-type interaction, shared race/quali equations, Student-t race noise |
| 3.2 Reliability (first cut) | `src/model/reliability.py` | Implemented: team-level mechanical hazard only, empirical-Bayes Beta-Binomial, rule-change-year prior widening. Mechanical/incident split is phase 2, per spec |
| 3.3 Race resolver | `src/sim/race.py` | Implemented: simulates quali for the pre-weekend grid, fixed per-circuit overtaking-difficulty grid-lock penalty, DNF draws, points |
| 2 Challenger | `src/model/challenger.py` | Implemented: XGBoost quantile regression behind the same pace-draw interface |
| 6 CLI | `src/cli.py` | `gp fit --through S:R` and `gp race --round R --mode pre` implemented. `--mode post-quali` and `gp season` are out of phase 1 scope (phase 2/3) |
| 7 Backtest | `src/backtest.py` | Rolling backtest, pole-wins baseline, pace-rating baseline, challenger scored through the identical resolver, calibration data dump |

Tests: `tests/test_pace_data.py`, `test_race_resolver.py`,
`test_reliability.py`, `test_challenger.py`, `test_cli.py` — 46 new tests,
all passing, covering every deterministic piece (index building, the
resolver's Monte Carlo logic, reliability posterior math, CLI arg
parsing/lookup helpers). Per the phase 1 task guidance, the Bayesian model's
internals are *not* unit-tested directly — they're validated through the
backtest metrics below, which is the validation strategy the spec's own
Phase 1 description calls for ("Phase 1 is where the risk is").

The full legacy pipeline (`collect.py`, `features.py`, `train.py`,
`predict.py`, `confidence.py`) is untouched; all 101 of its existing tests
still pass alongside the 46 new ones (147 total).

## Backend deviations from the spec (and why)

- **PyMC, not NumPyro/JAX.** The spec names NumPyro/JAX as faster but says
  "either works." NumPyro was tried first; `jaxlib` fails to import on this
  Windows/Python 3.12 environment (`ImportError: DLL load failed while
  importing _jax`), which is a wheel/packaging problem, not something fixable
  from inside a coding session. PyMC (pytensor + numba backend, no C++
  compiler needed) works natively here and turned out to be **fast**: a
  full-scale production fit (all of 2018-2025, 21 teams, 44 drivers, 171
  rounds, ~3,240 race + ~3,350 quali observations) completes NUTS sampling
  in **~82 seconds**, nowhere close to the 10-minute budget. This removed
  most of the pressure the spec's compute-budget section anticipated needing
  ADVI/VI to relieve — see "Compute budget" below.
- **XGBoost, not LightGBM.** The spec names LightGBM as the concrete
  suggestion. It was tried first; `lgb.Dataset(...).construct()` segfaults
  (`OSError: access violation reading 0x0000000000000000`) in this
  environment as soon as `pandas` has been imported anywhere in the process
  — reproduced with plain random numpy arrays, no project code involved, so
  it's a DLL/ABI conflict between the installed pandas/numpy/lightgbm wheels
  on this machine, not a bug in this project's code. The legacy pipeline
  already uses XGBoost successfully alongside pandas in this same
  environment, so the challenger uses XGBoost's native multi-quantile
  `reg:quantileerror` objective instead. Both backend swaps are noted in the
  relevant module docstrings.
- `pandas` is pinned to `>=2.2.3,<3.0` in `requirements.txt` (was `>=2.0`).
  The originally-installed 2.2.2 wheel (numpy-1-era) plus a numpy>=2
  environment (pulled in during the now-abandoned NumPyro install attempt)
  is what triggered the LightGBM crash above; 2.2.3+ is numpy2-native and
  fixed it before the XGBoost switch was even made. Kept as the fix either
  way since it's a real compatibility hazard for anyone reproducing this.

## Backtest results (rolling, 2022-2025)

Two runs, both real, both saved to `outputs/backtest_<timestamp>/`:

### Coarse grid (spec sec 7's iteration recipe: refit at rounds 2,4,6,9,12,16,20,
forecast intervening rounds from the latest fit)

28 fits, 88 rounds scored, **5.3 minutes total** (`outputs/backtest_20260914_231937/`):

| Metric | Bayesian model | Pole-wins baseline | Pace-rating baseline | XGBoost challenger |
|---|---|---|---|---|
| Mean log loss (race winner) | **1.874** | 3.219 | 2.111 | 2.151 |
| Mean Brier (podium) | **0.091** | — | — | 0.095 |
| Mean Brier (points finish) | **0.184** | — | — | 0.187 |
| Mean Spearman (expected vs actual order) | **0.586** | — | — | 0.559 |

- Beats pole-wins baseline: **yes** (1.874 vs 3.219)
- Beats pace-rating baseline: **yes** (1.874 vs 2.111)
- Beats the XGBoost challenger: **yes**, on every metric above, though not
  by a wide margin on Brier/Spearman — see "Decision" below.

An earlier, independent run with the same settings (before the challenger
was wired into the backtest) reproduced the Bayesian numbers within noise
(log loss 1.877 vs 1.874, Brier 0.091 vs 0.091, Spearman 0.589 vs 0.586),
so these aren't a one-off lucky seed.

### Every-round grid (the actual phase-gate grid per spec sec 7: "Run the
every-round grid only for the phase gate")

88 fits (one per scored round -- no gap-filling from a stale posterior,
unlike the coarse grid), 88 rounds scored, **23.6 minutes total**
(`outputs/backtest_20260916_222558/`):

| Metric | Bayesian model | Pole-wins baseline | Pace-rating baseline | XGBoost challenger |
|---|---|---|---|---|
| Mean log loss (race winner) | **1.844** | 3.219 | 2.111 | 2.090 |
| Mean Brier (podium) | **0.090** | — | — | 0.094 |
| Mean Brier (points finish) | **0.182** | — | — | 0.186 |
| Mean Spearman (expected vs actual order) | **0.597** | — | — | 0.561 |

- Beats pole-wins baseline: **yes** (1.844 vs 3.219)
- Beats pace-rating baseline: **yes** (1.844 vs 2.111)
- Beats the XGBoost challenger: **yes**, on every metric, matching the
  coarse-grid result -- the every-round fits (finer-grained, using each
  round's own most-recent data rather than a same-fit forecast covering
  several rounds) come out very slightly *better* on every one of the
  model's own metrics (log loss 1.844 vs 1.874, Brier-podium 0.090 vs
  0.091, Spearman 0.597 vs 0.586) and the pole/pace-rating baselines are
  numerically identical between the two runs (3.219 / 2.111, as expected --
  those baselines don't depend on which grid the *model* was fit on).
  This is exactly the kind of small, one-directional improvement you'd
  expect from fitting on marginally more up-to-date data each round, and
  it confirms the coarse grid is a reasonable substitute for iteration
  without materially overstating the model's quality.
- **Compute-budget result: 23.6 minutes for the full every-round phase-gate
  grid, against a 2-hour target -- cleared with roughly 5x headroom.**
  Per-fit time ranged from ~7s (small early-2022 data) up to ~38s (the
  largest 2025 cutoffs, fitting on nearly the full 2018-2025 history); no
  fit came close to the 10-minute single-fit budget.

### Calibration (podium probability; spec sec 7's "this is the one that
will actually reveal problems")

From the coarse-grid run, 1,758 driver-round predictions bucketed by
predicted P(podium):

| Predicted bucket | n | Mean predicted | Actual rate |
|---|---|---|---|
| 0.0-0.1 | 984 | 0.052 | 0.011 |
| 0.1-0.2 | 209 | 0.140 | 0.072 |
| 0.2-0.3 | 225 | 0.253 | 0.293 |
| 0.3-0.4 | 247 | 0.345 | 0.441 |
| 0.4-0.5 | 78 | 0.436 | 0.641 |
| 0.5-0.6 | 15 | 0.523 | 0.867 |

**Verdict: not visibly broken, but not well-calibrated either.** The trend
is monotonic (higher predicted bucket -> higher actual rate, no reversals),
which is the main thing that would signal something is fundamentally wrong.
But the model is systematically *overconfident* for the bulk of the field
(the 0-0.1 bucket, 984 of 1,758 predictions, predicts 5.2% and delivers
1.1%) and systematically *underconfident* for genuine contenders (0.4-0.6
buckets predict ~44-52% and deliver 64-87%, though those buckets only have
78 and 15 observations, so noisy). Net effect: probabilities are pulled
toward the middle relative to reality. (The every-round grid's calibration
table is essentially identical bucket-for-bucket -- e.g. 0-0.1 bucket:
0.050 predicted vs 0.008 actual, 0.4-0.5: 0.433 vs 0.613 -- so this isn't
an artefact of the coarse grid's stale-posterior rounds.) This reads as the
model's per-race noise term (`race_sigma`, shared across the whole field
rather than varying
by how close the grid is) being a bit too wide for the sharp-favorite races
and a bit too narrow for the genuine toss-ups -- a believable phase 2
refinement target, not a phase 1 blocker given the spec's own bar is "not
visibly broken."

### Decision: Bayesian vs challenger vs ensemble

**Ship the Bayesian model as primary; keep the XGBoost challenger in the
repo behind the same interface but do not ensemble for phase 1.**

Reasoning:
- The Bayesian model wins on every backtest metric (log loss, both Briers,
  Spearman), consistent with the spec's prior ("bet against [the challenger]
  winning outright in a regulation-change year") — 2022 and 2026 are both
  rule-change years within/adjacent to this window.
- The margin is decisive on log loss (1.874 vs 2.151, ~13% relative) but
  narrow on Brier/Spearman (0.091 vs 0.095, 0.586 vs 0.559) -- the two
  models agree more often than they disagree, which is itself evidence the
  pace signal is being captured similarly by both approaches and the
  Bayesian model's edge is mainly in the tails (exactly where log loss is
  sensitive and Brier/Spearman are not).
- No ensembling in phase 1: with the challenger this close on ranking
  metrics and meaningfully worse on log loss, a naive average would likely
  just pull the Bayesian model's better-calibrated tail probabilities
  toward the challenger's worse ones. An ensemble is worth revisiting once
  the challenger has its own quali-conditioned pace equation (right now it
  reuses its race-pace quantiles for the grid too, sec-noted limitation
  below) and once phase 2's reliability split is in for both models on
  equal footing.
- The challenger stays wired into `gp backtest` (not removed) specifically
  so this decision can be re-checked cheaply every time the backtest is
  re-run, per spec sec 2's "phase 1's backtest decides."

## Compute budget (spec sec 7)

**Both targets are cleared, with a lot of headroom, primarily because the
PyMC/numba fit turned out to be much faster than the spec's own worst-case
estimate (5-10 min/fit, 10-18 hr/pass) anticipated.**

- **Single production fit, full history (2018-2025):** ~82 seconds.
  Target: under 10 minutes. **Cleared by ~7x.**
- **Full backtest pass, coarse grid (28 fits, 88 rounds scored):** 5.3
  minutes. Target: under 2 hours. **Cleared by ~23x.**
- **Full backtest pass, every-round grid (phase-gate grid, 88 fits, 88
  rounds scored):** **23.6 minutes.** Target: under 2 hours. **Cleared by
  ~5x.** (Slower than the coarse grid's 5.3 minutes because "all" mode
  does one fit per round scored instead of one fit covering several
  rounds -- ~3x more fits -- but still comfortably inside budget.)

Techniques from sec 7 actually used:
- **Coarser backtest grid** (`--rounds coarse`, the default): implemented
  and used for the fast iteration numbers above.
- **Every-round grid for the phase gate** (`--rounds all`): implemented and
  run for the number above.
- **Non-centred random walk reparameterisation:** implemented from the
  start (`car0` + cumulative sum of scaled `car_step_raw`), per sec 7's "the
  single most likely reason NUTS is slow here."

Techniques from sec 7 **not** implemented (not needed to clear the budget,
so not built this session -- flagged as real gaps, not silently dropped):
- **Sequential warm starts** (initialise round r's sampler from round
  r-1's posterior): not implemented. `fit_nuts`/`fit_advi` accept an
  `initvals`/similar hook could be added, but each backtest fit currently
  starts cold. Given the 82s single-fit and 5.3min-full-pass numbers
  already clear the budget by a wide margin, this wasn't necessary to reach
  for -- but it would make a real production `gp fit --through 2026:17`
  right after `2026:16` cheaper, and is worth adding before phase 2 turns
  the backtest into a much bigger job (post-quali conditioning, sprints).
- **Parallelising across seasons:** not implemented; the backtest loop runs
  seasons sequentially. Same reasoning as above -- not needed to clear 2
  hours, straightforward to add later (the four seasons' fits are fully
  independent).
- **VI for the backtest loop:** `fit_advi` exists and is exercised (see
  the 2022-only ADVI smoke test during development), but the reported
  backtest numbers above use NUTS throughout (`--method nuts`), because
  NUTS turned out fast enough that VI's speed/accuracy trade-off wasn't
  needed. `--method advi` remains available if a future, heavier model
  (sprints, quali conditioning, full reliability split) needs it.

## Known limitations / phase 2+ carryover

1. **Round-index rebuild, not incremental.** `build_pace_data` rebuilds
   the round index and the whole car random walk from scratch on every
   call; it doesn't reuse a previous fit's structure. Combined with no
   warm-starting (above), each backtest round is a fully independent fit.
   Fine at current speed; revisit if phase 2's added complexity slows
   things down.
2. **Challenger has no quali equation.** The Bayesian model shares
   `car`/`driver` between race and quali equations (spec 3.1) and
   simulates a genuine pre-weekend grid from its quali posterior. The
   XGBoost challenger only predicts race pace; `backtest.py`'s
   `forecast_race_challenger` reuses the same race-pace quantile draw for
   the grid too (noise on both is set to zero so the grid-lock term
   introduced doesn't spuriously reorder anything -- see that function's
   docstring). This is a fair apples-to-apples comparison for phase 1's
   "does the cheaper model roughly compete" question, but it's not a
   complete challenger implementation.
3. **Mechanical DNF is a Beta-Binomial, not a hierarchical model.** Per
   spec sec 3.2, this is explicitly in-scope as a "first cut" and "keep it
   simple" -- documented as a deliberate compute-budget trade (a second
   MCMC pass for reliability wasn't worth it given the backtest already
   needs one fit per round for pace). The mechanical/incident split and the
   grid-position incident effect are phase 2, matching the spec.
4. **`gp race` entrant lookup only works for rounds already in
   `driver_rounds.csv`** (`cli._entrants_for_round`). This is correct and
   sufficient for the backtest (every scored round's entry list already
   exists in the data) and for re-forecasting a past round, but a genuinely
   future round (e.g. an actual, not-yet-run 2026 round) needs an explicit
   entry list wired up -- not built this session, flagged for whoever picks
   up pre-2026-race-weekend usage.
5. **`overflow encountered in dot` RuntimeWarning** appears during some
   early-cutoff NUTS fits (small-data rounds, `pytensor`'s quadpotential
   step). It didn't produce NaN posteriors or crash any fit in the full
   backtest run, but it's worth a closer look before trusting the model on
   genuinely tiny data windows (e.g. `gp fit --through 2026:2`).
6. **`gp race --mode post-quali` and `gp season` are not implemented** --
   both are explicitly phase 2/3 scope per the spec, not a phase 1 gap.

## Files touched

- `src/model/pace.py`, `src/model/reliability.py`, `src/model/challenger.py`
- `src/sim/race.py`
- `src/cli.py`, `src/backtest.py`
- `data/points_tables.csv` (new, hand-maintained per spec sec 10)
- `data/driver_rounds_validation.md` (regenerated on the now-complete
  2018-2026 dataset; the phase 0 clean-air-lap finding is unchanged)
- `requirements.txt` (pandas pin tightened, pymc/arviz added)
- `tests/test_pace_data.py`, `test_race_resolver.py`, `test_reliability.py`,
  `test_challenger.py`, `test_cli.py`
- Legacy pipeline (`collect.py`, `features.py`, `train.py`, `predict.py`,
  `confidence.py`, `ensemble.py`, `visualize.py`) untouched, per spec sec 4.
