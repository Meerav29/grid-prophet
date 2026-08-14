# Expanded Early-Season Window & Feature/Model Complexity — Design

## Context

Grid Prophet currently predicts the 2026 constructor championship using only
rounds 1–2 as the "early-season signal" window, for both historical training
seasons and the live 2026 row. With the 2026 season now at the summer break,
significantly more of the season has actually run. The model is also
considered too simple: it uses a small, season-level feature set and a
winner-take-all Ridge-vs-XGBoost selection with an optional two-way ensemble.

This spec covers two coupled changes:

1. Widen the early-season window from a hardcoded 2 rounds to the number of
   rounds actually completed so far in the current season, applied
   consistently to historical training rows and the live prediction row.
2. Add richer season-level features derived from data already collected, and
   strengthen the model/ensemble to handle the larger feature set without
   overfitting the ~120-row dataset.

## Locked decisions carried over (not renegotiated here)

- Season-level unit of analysis (not race-level). See `Project.md`.
- Points-share target variable.
- Stats-based constructor identity (no categorical team encoding).
- Constructor lineage / rebrand dampening logic in `features.py`.

## 1. Early-season window

### Current behavior
`EARLY_ROUNDS = 2` is a module-level constant in `src/features.py`, used
identically to compute `early_points_share` / `early_avg_finish` for every
historical season and for the 2026 row.

### New behavior
- `build_features()` gains an `early_rounds: int | None = None` parameter.
  When `None`, it falls back to the existing `EARLY_ROUNDS` constant (keeps
  default CLI/test behavior stable).
- A new helper, `latest_completed_round(year: int) -> int`, auto-detects the
  most recent round with results present in `race_results.csv` for a given
  season (reuses the detection logic already used by the `update` CLI
  command in `src/__main__.py` — extracted into a shared function rather than
  duplicated).
- The `update` and `run` CLI paths compute `N = latest_completed_round(2026)`
  and pass `early_rounds=N` into `build_features()`. This means:
  - The 2026 row's early-season features are computed over all rounds
    completed so far.
  - **Every historical season's early-season features are recomputed over
    the same N-round window**, so training and prediction never see a
    train/predict window mismatch (avoids leakage/skew).
- Historical seasons with fewer than N rounds total (shouldn't occur for
  full seasons, but guarded) use all available rounds for that season.
- `predict.py --rounds N` continues to work as an explicit override for
  ad-hoc what-if predictions (e.g. re-running with a smaller window), but the
  default `update`/`run` path uses auto-detection.

### Explicit hypothesis shift
This changes "early-season signal" from strictly "first 2 races" to "races
completed so far." That's an intentional, requested change — flagged here so
it's visible in one place rather than buried in a diff.

## 2. New features (no new data collection)

All new features are season-level aggregates computed from columns already
present in `data/race_results.csv` (`grid_position`, `finish_position`,
`classification`). No changes to `collect.py` are required.

Added to `_early_season_features` (computed over the same early-round window
as `early_points_share`/`early_avg_finish`, so they stay leak-free at
prediction time):

| Feature | Definition |
|---|---|
| `constructor_dnf_rate` | Fraction of the team's car-entries in the window with `classification == "DNF"`. |
| `avg_grid_to_finish_delta` | Mean of `grid_position - finish_position` across the team's entries in the window (positive = net overtaking gain). Entries with missing grid or finish position (DNS/DNF without a finish rank) are excluded from the mean. |
| `teammate_head_to_head` | Fraction of races in the window where the team's better-classified driver (lower finish position among classified finishers) is driver A vs driver B — expressed as driver-agnostic "share of races won by whichever teammate is nominally 'first' by alphabetical driver code," to keep the feature symmetric and not driver-identity-leaky. Races where both DNF are excluded from the denominator. |
| `development_trend` | Slope (via `numpy.polyfit` degree 1) of the team's per-round average finish position across the rounds in the window, against round number. Negative slope = improving (finishing position numbers going down). Requires at least 2 rounds in the window; 0.0 if window has only 1 round. |

These are added as new columns in `data/features.csv`; existing columns are
unchanged.

## 3. Model complexity: feature selection + ensemble

### Feature selection
Before the existing Ridge-vs-XGBoost comparison in `train.py`:
- Add an L1-regularized selection step (`sklearn.linear_model.LassoCV` on
  standardized features) run once on the full training set to rank
  features by whether they survive shrinkage to a nonzero coefficient.
- Features with a zero coefficient across the LassoCV fit are dropped before
  the Ridge/XGBoost comparison and CV. This is a fixed pre-processing step,
  not re-tuned per CV fold (keeps it simple and avoids a second nested CV
  loop on top of the existing leave-one-season-out CV).
- Log which features were dropped, for visibility.
- This step is applied whenever the new richer feature set is present
  (i.e., always, going forward) — it isn't conditional on a flag.

### Ensemble
No new model family. The existing `ensemble.py` blend (full-data model +
rule-change-years-only model, LOOCV-tuned alpha) is kept as-is, but now
trains on the post-selection feature set automatically (it already consumes
whatever `feature_cols` the base bundle defines, so this requires no
structural change to `ensemble.py` — only verifying the feature list flows
through correctly, which the existing test suite covers).

## Data flow changes

```
collect.py        (unchanged)
        |
        v
features.py  --early_rounds=N (auto-detected)--> features.csv
        |         (adds 4 new columns)
        v
train.py     --LassoCV feature selection--> Ridge/XGBoost CV --> model.pkl
        |
        v
ensemble.py  (unchanged internally, consumes selected feature_cols)
        |
        v
predict.py   (unchanged interface; --rounds still available as override)
```

## Testing

- `tests/test_train.py`: add cases for the new features being present and
  for LassoCV dropping a synthetic all-zero/noise feature.
- New `tests/test_features.py` (currently no dedicated feature test file;
  features are exercised indirectly) covering: `constructor_dnf_rate`,
  `avg_grid_to_finish_delta`, `teammate_head_to_head`, `development_trend`
  on small synthetic race-result fixtures, plus `latest_completed_round`.
- `tests/test_main.py`: verify `update`/`run` compute and pass
  `early_rounds` from `latest_completed_round(2026)`.
- Existing `tests/test_ensemble.py`, `tests/test_predict.py`,
  `tests/test_confidence.py`, `tests/test_visualize.py` should continue to
  pass unmodified (interfaces unchanged); re-run as regression checks.

## Out of scope (explicitly deferred)

- Qualifying-session pace deltas (would require new FastF1 `'Q'` session
  collection — separate future work).
- Race-level granularity (locked decision, not revisited here).
- New model families (e.g. neural nets, stacking beyond the existing
  two-way ensemble).

## Rollout

- Retrain and regenerate `data/features.csv`, `models/Grid_Prophet_model.pkl`,
  `models/Grid_Prophet_ensemble.pkl`, and `data/predictions_2026.csv` /
  plots as part of implementation, so the README's predictions table reflects
  the new pipeline.
