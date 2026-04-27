# Grid Prophet Stage 5 — Polish & Iteration Design

**Date:** 2026-04-26  
**Status:** Approved  
**Approach:** Option B — new dedicated modules alongside existing `src/` scripts

---

## Overview

Stage 5 adds five capabilities to the completed Stages 1–4 pipeline:

1. Makefile + Python module CLI entry point
2. Confidence intervals via bootstrap
3. Ensemble model (full-data + rule-change-only blend)
4. Mid-season update capability
5. Visualizations (PNGs + notebook)

All new capabilities are implemented as focused new files. Existing scripts (`collect.py`, `features.py`, `train.py`, `predict.py`) are extended minimally — only to wire in new flags or call new modules.

---

## Section 1: CLI Entry Point & Makefile

### `src/__main__.py`

Enables `python -m grid_prophet <command>`. Sub-commands:

| Command | Action |
|---|---|
| `collect` | Runs `collect.py main()` |
| `features` | Runs `features.py main()` |
| `train` | Runs `train.py main()` |
| `predict` | Runs `predict.py main()` |
| `run` | Runs all four in sequence |
| `update` | Auto-detects latest completed round, re-predicts |
| `plots` | Generates all four charts via `visualize.py` |

Each sub-command accepts the same flags as the underlying script's `argparse`. The `update` command queries FastF1 for the latest completed round of `PREDICT_YEAR` and passes `--rounds N` through to `predict`.

### `Makefile`

Thin wrappers over the CLI:

```
make collect      → python -m grid_prophet collect
make features     → python -m grid_prophet features
make train        → python -m grid_prophet train
make predict      → python -m grid_prophet predict
make run          → python -m grid_prophet run
make update       → python -m grid_prophet update
make plots        → python -m grid_prophet plots
make clean        → remove derived data files (features.csv, cv_results.csv, predictions_*.csv) and models/*.pkl; does NOT delete race_results.csv or constructor_standings.csv
```

---

## Section 2: Confidence Intervals (`src/confidence.py`)

### Method

Bootstrap resampling: N=500 iterations. Each iteration:
1. Resample training rows with replacement
2. Retrain the winning model pipeline (from the saved bundle) on the bootstrap sample
3. Predict 2026 using the pre-built `features_2026` DataFrame
4. Collect predicted share per constructor

From the 500 predictions per constructor, compute:
- `ci_low` = 10th percentile
- `ci_high` = 90th percentile  
- `ci_half` = (ci_high - ci_low) / 2  (half-width of 80% interval)

### Interface

```python
def bootstrap_confidence_intervals(
    bundle: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features_2026: pd.DataFrame,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Returns DataFrame with columns: constructor, ci_low, ci_high, ci_half."""
```

### Output

- Terminal table updated to show `± ci_half` column
- `predictions_2026.csv` gains columns: `ci_low`, `ci_high`, `ci_half`
- `predict.py main()` calls `bootstrap_confidence_intervals()` after point prediction; `--no-ci` flag skips it for speed

### Terminal Format

```
  Rank  Constructor              Predicted Share     80% CI
  ──────────────────────────────────────────────────────────
  1     Mercedes                 30.4%           ± 4.2%
  2     Ferrari                  22.5%           ± 3.8%
```

---

## Section 3: Ensemble Model (`src/ensemble.py`)

### Two Sub-Models

1. **Full-data model** — existing winner from `train.py` (XGBoost or Ridge, 2014–2025)
2. **Rule-change-only model** — same architecture, trained on rule-change years + following years: {2014, 2015, 2022, 2023}

### Blend Weight Tuning

Leave-one-season-out CV on rule-change seasons only (2014, 2022). Candidate blend ratios for α (full-data weight): `[0.3, 0.4, 0.5, 0.6, 0.7]`. Select α that maximizes Spearman correlation on held-out rule-change seasons.

Final prediction:
```
blended_share = α × full_model_pred + (1 - α) × rc_model_pred
```

### Saved Artifact

`models/Grid_Prophet_ensemble.pkl` — bundle containing:
- `full_model`: the base winning pipeline
- `rc_model`: rule-change-only pipeline
- `alpha`: tuned blend weight
- `feature_cols`: same as base model
- `winner_name`: e.g. `"Ensemble(XGBoost, α=0.6)"`
- `rule_change_weight`: inherited from base model

### CLI Integration

`predict.py` gains `--ensemble` flag: loads `Grid_Prophet_ensemble.pkl` instead of base model. Bootstrap CI in `confidence.py` works on whichever bundle is passed — no changes needed.

`train.py` gains `--ensemble` flag that, after saving the base model, calls `ensemble.py` to build and save the ensemble bundle.

---

## Section 4: Mid-Season Updates

### `--rounds N` flag in `predict.py`

Controls how many rounds of early-season data to use (default: `EARLY_ROUNDS = 2`). When N > 2, `build_2026_features()` uses rounds 1–N for `early_points_share` and `early_avg_finish`. All other features (historical momentum, driver quality) are unchanged.

### `update` Command in `__main__.py`

1. Queries FastF1 event schedule for `PREDICT_YEAR` to find latest completed round N
2. Calls `python -m grid_prophet predict --rounds N`
3. Saves to `data/predictions_2026_r{N}.csv` (round-stamped snapshot) in addition to overwriting `data/predictions_2026.csv`

This preserves a full history of predictions as the season progresses.

### `make update`

Delegates to `python -m grid_prophet update`.

---

## Section 5: Visualizations (`src/visualize.py`)

Four charts. All saved as PNGs to `plots/`. Also callable from `notebooks/exploration.ipynb` inline.

### Chart 1: CV Accuracy (Predicted vs Actual Rank)

- One subplot per held-out season (2014–2025), arranged in a grid
- Scatter: x = actual constructor rank, y = predicted rank
- Identity line (perfect prediction)
- Points colored by constructor
- Title shows per-season Spearman ρ

### Chart 2: Spearman Correlation by Season

- Bar chart, one bar per season
- Bars for rule-change years (2014, 2022) highlighted in a distinct color
- Horizontal dashed line at mean Spearman across all seasons
- Data source: `data/cv_results.csv`

### Chart 3: 2026 Prediction with Confidence Intervals

- Horizontal bar chart, constructors ranked 1–11 top-to-bottom
- Bar length = predicted points share
- Error bars = 80% CI (`ci_low` to `ci_high`)
- Data source: `data/predictions_2026.csv` (requires CI columns)

### Chart 4: Feature Importance / Coefficients

- Horizontal bar chart, features sorted by importance descending
- XGBoost: `feature_importances_` values
- Ridge: absolute coefficient values, labeled as coefficients
- Data source: model bundle loaded from `models/Grid_Prophet_model.pkl`

### Interface

```python
def plot_cv_accuracy(cv_df, features_df, out_dir="plots/") -> None
def plot_spearman_by_season(cv_df, out_dir="plots/") -> None
def plot_2026_predictions(predictions_df, out_dir="plots/") -> None
def plot_feature_importance(bundle, out_dir="plots/") -> None
def plot_all(out_dir="plots/") -> None  # loads CSVs and calls all four
```

`make plots` calls `python -m grid_prophet plots` which calls `plot_all()`.

### Notebook

`notebooks/exploration.ipynb` gets a "Visualizations" section that imports `visualize` and calls each function inline (renders in-cell).

---

## File Changes Summary

| File | Change |
|---|---|
| `src/__main__.py` | **New** — CLI entry point |
| `src/confidence.py` | **New** — bootstrap CI |
| `src/ensemble.py` | **New** — rule-change blend model |
| `src/visualize.py` | **New** — all chart generation |
| `Makefile` | **New** — pipeline shortcuts |
| `src/predict.py` | Add `--rounds`, `--ensemble`, `--no-ci` flags; call `confidence.py` |
| `src/train.py` | Add `--ensemble` flag; call `ensemble.py` after base model saved |
| `notebooks/exploration.ipynb` | Add "Visualizations" section |
| `plots/` | **New directory** — PNG outputs |

---

## Testing Notes

- `confidence.py`: test with N=10 bootstrap iterations for speed; assert output shape and column names
- `ensemble.py`: test blend weight tuning returns α in [0.3, 0.7] and ensemble predictions differ from base
- `visualize.py`: test that `plot_all()` creates the expected PNG files in `plots/`
- `__main__.py`: test each sub-command routes to the correct function

---

## Open Items for User Review

- Chart aesthetics (colors, fonts, layout) — user will review generated charts after implementation and provide feedback
