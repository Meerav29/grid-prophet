# Autopilot queue — grid-prophet

Design: [2026-09-19-autopilot-design.md](../superpowers/specs/2026-09-19-autopilot-design.md)
Rotation: Sun / Tue / Thu, plus Fri / Sat for the Oct 2-4 race weekend (merge by Sat 20:00 ET). Base branch: `auto/queue`. Window: 2026-09-20 → 09-25.

These slices decompose **Phase 2** of [the v2 spec](../grid-prophet-v2-spec.md) §8.
Phase 2 is scoped at two weeks and roughly six PRs; slices 1–3 were built in the original window. Slices 4–6 below were queued
2026-10-01 after the queue ran dry, ordered by how much they move the next
race forecast: overtaking fit, weather, then the slice-3 carry-forwards. The Phase 2 backtest stays out of scope
for routines.

**Phase 2's real "done when" bar requires the full backtest**, which takes up to
two hours (§7) and is not in CI. Nothing here closes Phase 2. These slices are
built and merged to `auto/queue` on unit tests and review; the backtest is run
with the owner present, afterwards.

Status values: `todo` → `in-progress` → `in-review` → `merged`.
Terminal: `held` (owner vetoed), `blocked` (review rejected).

---

## slice-1 — Quali conditioning and `gp race --mode post-quali`

Status: merged — PR #5 merged by the owner 2026-09-27 (`abd045a`).

Criterion 1 was unblocked by option 1: `data/driver_rounds_fixture.csv`, 80 real
rows (2024 R1–R2, Q + R). Copy it to `data/driver_rounds.csv` to run the
criterion. It was checked twice against PR #5's head `b19c544` — by the owner
locally, and independently in a cloud sandbox — with matching forecasts
(VER p_win 0.569 vs 0.5662). Evidence is in the comments on #5.

Two things carried forward from this slice:

- **`gp fit` fails on Python 3.12, which is what CI runs.** `requirements.txt`
  declares no netCDF backend and relied on arviz 0.x pulling `h5netcdf` in;
  arviz 1.x dropped that, so `idata.to_netcdf` raises. Sampling succeeds, the
  write fails. CI is green only because no test exercises `cmd_fit`.
  **Fixed 2026-09-27** by declaring `h5netcdf[h5py]>=1.0` in `requirements.txt`.
  Plain `h5netcdf` is not enough: 1.8+ no longer installs `h5py`. Verified in a
  clean venv (arviz 1.3.0, no netCDF4), where `idata.nc` wrote and read back.
- The fixture has no sprint weekend, so slice-2 may need true sprint values
  hard-coded in its test.
Spec: `docs/grid-prophet-v2-spec.md` §3.3 (step 3), §6 (post-quali conditioning
and the fallback protocol), §8 Phase 2

Acceptance:

- [ ] `gp race --round N --mode post-quali` runs end to end on a completed round
      and writes `race_forecast.csv` with the columns §6 lists.
- [ ] Post-quali uses the **real** grid; pre-weekend still simulates quali from
      the quali equation. A test shows the two modes produce different finishing
      distributions when the real grid differs from the simulated one.
- [ ] Conditioning is a warm-started NUTS refit from the pre-weekend posterior,
      per §6's stated primary approach — not an importance reweight.
- [ ] The §6 fallback exists: when the warm start misses its time target the
      round falls back to a full refit **rather than blocking the run**.
- [ ] Fallback is **recorded per round** and countable, not silent. §6 is
      explicit that a high fallback rate is a signal, not something to absorb.
- [ ] The relaxed time target applies only to early-season rounds (roughly the
      first six). A late-season miss is recorded as a regression. Covered by a
      test that distinguishes the two cases.
- [ ] Existing tests still pass; new behavior has tests that fail without it.

Out of scope: running the backtest, the log-loss comparison against pre-weekend
mode (that is Phase 2's gate, slice 6), weather, sprints.

---

## slice-2 — Sprint events

Status: merged — PR #7 merged by the review routine 2026-09-27 (`2edcf87`).
Spec: `docs/grid-prophet-v2-spec.md` §3.4 (sprints as an extra event), §3.3
(step 5, points tables), §8 Phase 2

Acceptance:

- [ ] A sprint is modeled as an extra, shorter event within the round, with its
      own points table from `data/points_tables.csv`.
- [ ] The sprint **shares the weekend's pace draw** with the Grand Prix rather
      than drawing independently (§3.4). A test asserts the shared draw — this
      is the part that is easy to get silently wrong.
- [ ] Sprint points reconcile with actual final standings for at least one real
      sprint weekend, as a test with the true values.
- [ ] Rounds without a sprint are unaffected; a test covers a non-sprint round.
- [ ] Existing tests still pass; new behavior has tests that fail without it.

Out of scope: season chaining (Phase 3), the fastest-lap rule for years where it
does not apply.

---

## slice-3 — Reliability split: mechanical vs incident

Status: merged — PR #8 merged by the review routine 2026-09-29 (`99f0151`).

All five criteria were verified against the diff rather than against the PR
body, and the PR's mutation claims were re-derived from the test source rather
than taken on trust. CI (`test`) was green on `d1e79f2`; the diff is 598 lines,
inside the 600-line review cap. The data gap is honored on the production path
and not only in tests: `src/validate_driver_rounds.py` confirms the real
`driver_rounds.csv` schema carries the `status` column that `channel_weights`
needs, so the flagged-row logic does not silently no-op outside the fixtures.

Two things carried forward, neither blocking:

- **The causeless-mixture split can still hand a flagged row a hard label.**
  `channel_weights` splits a causeless `needs_review` row ("Retired") on the
  observable era's clean mechanical:incident ratio. When a fit window's clean
  labels all sit in one channel that ratio is 0.0 or 1.0, so the flagged row
  books weight 1.0 to a single channel — a clean label in all but the
  bookkeeping. This is live in the PR's own
  `test_build_makes_the_mixture_countable`, where the single "Retired" row lands
  at `w_mechanical = 1.0`. Harmless on the full 2018–2026 window, which has
  hundreds of clean labels of both kinds; reachable through the backtest's early
  `through_season` / `through_round` cutoffs. Clamping the ratio away from 0 and
  1 is a one-line fix. The all-flagged case is already handled correctly (0.5).
- **The first-lap spike is inert.** It lives in `lap_of_retirement_pmf`, which
  no code path consumes. §3.2 names sprint and partial-points scoring as its
  only consumers and neither reads it yet. Accepted because the resolver draws
  one Bernoulli per driver per race with no lap-level simulation, so the lap
  distribution is the only coherent home for a spike. Relatedly,
  `sprint_inputs_from` scales the whole of `dnf_prob` by
  `SPRINT_DISTANCE_RATIO`, so sprint incident risk is understated by roughly the
  spike's share of the incident channel — still an improvement on the
  pre-slice-3 sprint path, which omitted the incident channel altogether. The
  ~45-line fix was removed to stay under the 600-line cap and is flagged in the
  PR body for a follow-up slice.
Spec: `docs/grid-prophet-v2-spec.md` §3.2 including its **Known data gap**
paragraph, §8 Phase 2

Replaces the Phase 1 first cut, which is a single team-level Beta-Binomial
(see `docs/phase1-status.md`, caveat 3).

Acceptance:

- [ ] Mechanical DNF: hazard per team-season, prior widened in rule-change years
      (2014, 2022), shrinking as the season runs.
- [ ] Incident DNF: hazard per driver, with a grid-position effect and a
      first-lap spike.
- [ ] Lap-of-retirement distribution exists. §3.2 says keep it simple — it
      matters for sprint and partial-points scoring and nothing else.
- [ ] **The data gap is honored.** FastF1's cause detail stops after 2024; 2023+
      rows mostly say generic "Retired". Those rows are treated as a *mixture*,
      not as clean mechanical or clean incident labels. A test asserts that a
      `needs_review=True` row from `data/dnf_status_map.csv` is not consumed as a
      clean label.
- [ ] Existing tests still pass; new behavior has tests that fail without it.

If the split cannot be identified from the available labels, say so in the PR
body and narrow the slice. Do not manufacture labels for post-2023 rows.

Out of scope: hand-curating a supplementary retirement-cause source. That is the
owner's call, not a routine's — flag it in the PR body if the data gap blocks
the split.

---

## slice-4 — Fit the overtaking parameter from data

Status: merged — PR #10 merged by the review routine 2026-10-02 (`44938bb`).

All four criteria were verified against the diff rather than against the PR
body, and every number the PR claims was re-derived independently in a clean
venv rather than taken on trust. CI (`test`) was green on `b9e23d3`; the diff is
491 lines, inside the 600-line review cap. All five of the PR's mutation results
reproduced exactly as stated (7 / 3 / 2 / 1 / 9 failures), as did the review
pack's fixture output (Bahrain raw 2.294 / shrunk 4.059, Jeddah 3.110 / 3.422,
and the 2024:2 forecast falling back to the 3.5 hand rating on a leak-safe
window) and all eight planted-recovery rows (worst error 0.190 against the
stated ±0.45 tolerance; Monaco-like 1.018 against Bahrain-like 5.168). Nothing
in the body was found overstated.

The backtest has still not been run against this slice. The fit recovers a planted
parameter and breaks no existing test, but whether it *improves* the forecast is
unmeasured — that is Phase 2's gate, with the owner present.

The fit is live on the production path and not only in fixtures:
`src/validate_driver_rounds.py` confirms the real `driver_rounds.csv` schema
carries `gap_to_winner_median_clean_air_s`, `classified`, `grid_position` and
`finish_position`, so `_usable_race_rows` does not silently return empty outside
the test data. Both `_circuit_info` call sites are wired, and `src/sim/race.py`'s
diff is comment-only.

Two things carried forward, neither blocking:

- **Neither production call site is test-protected.** The two wirings that make
  this slice do anything in a real run — `build_overtaking_data` inside
  `cli.forecast_weekend` and inside `backtest.forecast_race_challenger` — can
  each be deleted with all 182 tests still passing. Criterion 2's "a test covers
  both paths" is satisfied at the `_circuit_info` / `circuit_difficulty` seam,
  where the test hands the fit in; nothing asserts that a forecast builds one
  for itself off the pre-round window. So a later refactor can drop every
  forecast back to the hand rating silently, and the leak-safety this design
  turns on is guarded only by a unit test of `build_overtaking_data`'s own
  window argument, not by anything at the call site. Merged because the code is
  correct as written and all five mutations confirm the new behavior does have
  failing-without tests — the gap is regression cover on three lines, not a
  missing behavior. A test asserting `forecast_weekend` returns a different
  forecast with and without a fitted circuit would close it.
- **The fit is rebuilt in full on every forecast call.**
  `build_overtaking_data` fits *every* circuit in the window when only the
  forecast round's circuit is consumed. Milliseconds at the current data size,
  but a full backtest pays it once per round, so it scales with
  rounds × circuits rather than with rounds.

Note for the owner: two agents wrote this status within minutes of each other on
2026-10-02 — `a4fa8d4` recorded the merge as the owner's. It was the review
routine's. Worth a look at whether the review cron fired twice.

Spec: `docs/grid-prophet-v2-spec.md` §3.3 (step 3), §5 (`data/circuits.csv`
overtaking difficulty hand rating), §8 Phase 2

Acceptance:

- [ ] A per-circuit overtaking parameter is estimated from observed
      grid-to-finish position changes in `driver_rounds.csv`, shrunk toward the
      hand rating in `data/circuits.csv` when a circuit has few observations.
- [ ] The resolver uses the fitted value, falling back to the hand rating when
      no fit exists; a test covers both paths.
- [ ] A test on synthetic data with a known planted parameter recovers it
      within a stated tolerance. A locked-order circuit (Monaco-like) gets a
      visibly lower value than an open one (Bahrain-like).
- [ ] Existing tests still pass; new behavior has tests that fail without it.

Out of scope: the Phase 2 backtest and its log-loss comparison (owner present).

---

## slice-5 — Weather: wet races get their own noise scale

Status: merged — PR #12 merged by the review routine 2026-10-03 (`28c2199`).
Spec: `docs/grid-prophet-v2-spec.md` §5 (weather row: "wet races get their own
noise scale"), §8 Phase 2

Acceptance:

- [ ] The `driver_rounds.csv` schema (and `src/validate_driver_rounds.py`)
      carries a wet/dry flag per round sourced from FastF1 weather; absence in
      older fixtures degrades to dry, not an error.
- [ ] The pace likelihood uses a separate noise scale for wet rounds. A test
      shows wet and dry rounds get different scales and that dry-only data
      reproduces pre-slice behavior exactly.
- [ ] Wet is a recorded per-round input, not guessed; do not fabricate weather
      for rounds with no data. Mark them unknown and treat as dry.
- [ ] Existing tests still pass; new behavior has tests that fail without it.

Out of scope: air-temperature effects, in-race weather changes, running the
backtest.

---

## slice-6 — Slice-3 carry-forwards: clamp the channel ratio, route the first-lap spike into sprints

Status: in-review — PR #14, opened 2026-10-03.

Both halves shipped; the 600-line fallback in "Out of scope" below was not
needed — the diff stays well inside the review cap. `CAUSELESS_RATIO_FLOOR = 0.05` clamps the
causeless split; `sprint_incident_exposure` sums the incident lap pmf over the
sprint's lap window (0.4412 against a distance share of 0.3279), which is the
first-lap spike's first real consumer. Three decisions recorded in
`docs/autopilot/decisions.md`. The backtest has still not been run against any
of Phase 2 — sprint DNF probability moves ~18% on a plausible driver and
nothing says that is closer to the truth.
Spec: `docs/grid-prophet-v2-spec.md` §3.2, §3.4; follows slice-3 (PR #8)

Both items were flagged by the review of PR #8 as non-blocking. Read the
"carried forward" notes under slice-3 above for the exact locations
(`channel_weights`, `lap_of_retirement_pmf`, `sprint_inputs_from`,
`SPRINT_DISTANCE_RATIO`).

Acceptance:

- [ ] `channel_weights` clamps the clean mechanical:incident ratio away from 0
      and 1, so a flagged (`needs_review`) row can never book weight 1.0 to one
      channel. A test with a fit window whose clean labels are all in one
      channel proves it; it fails without the clamp. The all-flagged case still
      returns 0.5.
- [ ] The first-lap spike in `lap_of_retirement_pmf` has a real consumer: sprint
      incident risk uses the lap distribution instead of scaling all of
      `dnf_prob` by `SPRINT_DISTANCE_RATIO`. A test shows sprint incident risk
      differs from the old flat scaling when the spike is non-zero.
- [ ] Race (non-sprint) outputs are unchanged; a test covers it.
- [ ] Existing tests still pass; new behavior has tests that fail without it.

Out of scope: any new data source for post-2023 retirement causes (owner's call).
If the sprint rewrite would exceed the 600-line cap, ship the clamp alone and
leave the sprint half as a `todo` follow-up in this file.
