# Autopilot queue — grid-prophet

Design: [2026-09-19-autopilot-design.md](../superpowers/specs/2026-09-19-autopilot-design.md)
Rotation: Sun / Tue / Thu. Base branch: `auto/queue`. Window: 2026-09-20 → 09-25.

These slices decompose **Phase 2** of [the v2 spec](../grid-prophet-v2-spec.md) §8.
Phase 2 is scoped at two weeks and roughly six PRs; slices 4–6 (weather, fitting
the overtaking parameter from data, and the Phase 2 backtest) are out of scope
for this window.

**Phase 2's real "done when" bar requires the full backtest**, which takes up to
two hours (§7) and is not in CI. Nothing here closes Phase 2. These slices are
built and merged to `auto/queue` on unit tests and review; the backtest is run
with the owner present, afterwards.

Status values: `todo` → `in-progress` → `in-review` → `merged`.
Terminal: `held` (owner vetoed), `blocked` (review rejected).

---

## slice-1 — Quali conditioning and `gp race --mode post-quali`

Status: in-review
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

Status: todo
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

Status: todo
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
