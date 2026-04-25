# Stage 3 — Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/train.py` to train and compare XGBoost vs Ridge regression with leave-one-season-out CV, tune rule-change sample weights, select the best model, and save it to `models/Grid_Prophet_model.pkl`.

**Architecture:** Load `data/features.csv`, run leave-one-season-out CV twice (once to tune sample weight, once to compare XGBoost vs Ridge), select the winner, retrain on all data, and persist the model + CV results. The script is a standalone CLI — no shared state with other modules.

**Tech Stack:** scikit-learn (Ridge, cross-val utilities), xgboost, scipy (spearmanr), pandas, numpy, joblib (model persistence)

---

## Pre-Flight: Data Gap

The current `data/` files only cover 2014–2022. Before training, re-run data collection to extend through 2025, then re-run feature engineering. These are prerequisites — do them manually before starting Task 1:

```bash
python src/collect.py --resume --end-year 2025
python src/features.py
```

After this, `data/features.csv` should have rows for 2014–2025 with `season_points_share` populated for all years (and NaN only for any 2026 rows if added later).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/train.py` | **Create** | All training logic — CV, weight tuning, model comparison, persistence |
| `data/cv_results.csv` | **Generated** | Per-fold CV results (season, model, spearman, mae) |
| `models/Grid_Prophet_model.pkl` | **Generated** | Serialized winning model + metadata dict |
| `tests/test_train.py` | **Create** | Unit tests for CV helpers and weight tuning logic |

---

## Task 1: Scaffold `train.py` with data loading and feature/target split

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import load_data, FEATURE_COLS

def test_load_data_returns_train_split(tmp_path):
    # Minimal fake features.csv — 3 seasons, 2 constructors each
    rows = []
    for year in [2014, 2015, 2016]:
        for ctor in ["TeamA", "TeamB"]:
            rows.append({
                "year": year, "constructor": ctor,
                "early_points_share": 0.5, "early_avg_finish": 5.0,
                "is_rule_change_year": 0, "rule_change_adaptation_score": 0.0,
                "prev_year_points_share": 0.3, "prev_year_standing": 3,
                "constructor_win_rate_5yr": 0.4,
                "avg_driver_career_points_per_race": 2.0,
                "season_points_share": 0.5,
            })
    # Add a row with NaN target (simulates 2026)
    rows.append({
        "year": 2026, "constructor": "TeamA",
        "early_points_share": 0.4, "early_avg_finish": 4.0,
        "is_rule_change_year": 1, "rule_change_adaptation_score": -0.5,
        "prev_year_points_share": 0.3, "prev_year_standing": 2,
        "constructor_win_rate_5yr": 0.6,
        "avg_driver_career_points_per_race": 3.0,
        "season_points_share": float("nan"),
    })
    df = pd.DataFrame(rows)
    csv = tmp_path / "features.csv"
    df.to_csv(csv, index=False)

    X, y, meta = load_data(str(csv))

    assert len(X) == 6  # only complete seasons
    assert len(y) == 6
    assert list(X.columns) == FEATURE_COLS
    assert meta["year"].tolist() == [2014, 2014, 2015, 2015, 2016, 2016]
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd c:/Users/meera/Github-Projects/grid-prophet
python -m pytest tests/test_train.py::test_load_data_returns_train_split -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `train` doesn't exist yet.

- [ ] **Step 3: Implement data loading in `src/train.py`**

```python
"""Train XGBoost model on historical seasons with rule-change-year weighting."""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [
    "early_points_share",
    "early_avg_finish",
    "is_rule_change_year",
    "rule_change_adaptation_score",
    "prev_year_points_share",
    "prev_year_standing",
    "constructor_win_rate_5yr",
    "avg_driver_career_points_per_race",
]

RULE_CHANGE_WEIGHT_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_data(csv_path: str):
    """
    Load features.csv. Returns (X, y, meta) where meta is a DataFrame with
    year and constructor columns aligned with X and y.
    Only rows with non-NaN season_points_share are included (training rows).
    """
    df = pd.read_csv(csv_path)
    train = df[df["season_points_share"].notna()].copy().reset_index(drop=True)
    X = train[FEATURE_COLS].copy()
    y = train["season_points_share"].copy()
    meta = train[["year", "constructor"]].copy()
    return X, y, meta
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_train.py::test_load_data_returns_train_split -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: scaffold train.py with data loading and FEATURE_COLS"
```

---

## Task 2: Implement leave-one-season-out CV helper

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
from train import leave_one_season_out_cv
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def _make_fake_data():
    rows = []
    for year in [2014, 2015, 2016, 2017]:
        for ctor in ["TeamA", "TeamB"]:
            rows.append({
                "year": year, "constructor": ctor,
                "early_points_share": 0.5 + (year - 2014) * 0.05,
                "early_avg_finish": 5.0,
                "is_rule_change_year": int(year == 2014),
                "rule_change_adaptation_score": 0.0,
                "prev_year_points_share": 0.3,
                "prev_year_standing": 3,
                "constructor_win_rate_5yr": 0.4,
                "avg_driver_career_points_per_race": 2.0,
                "season_points_share": 0.5 + (year - 2014) * 0.05,
            })
    df = pd.DataFrame(rows)
    train = df[df["season_points_share"].notna()].reset_index(drop=True)
    X = train[FEATURE_COLS]
    y = train["season_points_share"]
    meta = train[["year", "constructor"]]
    return X, y, meta

def test_loocv_returns_one_fold_per_season():
    X, y, meta = _make_fake_data()
    model = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    results = leave_one_season_out_cv(model, X, y, meta, rule_change_weight=1.0)
    assert len(results) == 4  # 4 seasons
    for fold in results:
        assert "season" in fold
        assert "spearman" in fold
        assert "mae" in fold

def test_loocv_spearman_is_float():
    X, y, meta = _make_fake_data()
    model = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    results = leave_one_season_out_cv(model, X, y, meta, rule_change_weight=1.0)
    for fold in results:
        assert isinstance(fold["spearman"], float)
        assert isinstance(fold["mae"], float)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_train.py::test_loocv_returns_one_fold_per_season tests/test_train.py::test_loocv_spearman_is_float -v
```

Expected: `ImportError: cannot import name 'leave_one_season_out_cv'`

- [ ] **Step 3: Implement `leave_one_season_out_cv` in `src/train.py`**

Add after the `load_data` function:

```python
def leave_one_season_out_cv(model, X: pd.DataFrame, y: pd.Series,
                             meta: pd.DataFrame, rule_change_weight: float) -> list[dict]:
    """
    Leave-one-season-out CV. Returns a list of dicts, one per held-out season:
      {"season": int, "spearman": float, "mae": float}
    rule_change_weight is applied to rows where is_rule_change_year == 1.
    """
    seasons = sorted(meta["year"].unique())
    results = []

    for held_out in seasons:
        train_mask = meta["year"] != held_out
        test_mask = meta["year"] == held_out

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_test, y_test = X[test_mask].copy(), y[test_mask].copy()

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Sample weights
        rc_col = X_train["is_rule_change_year"]
        sample_weights = np.where(rc_col == 1, rule_change_weight, 1.0)

        # Fit — pass sample_weight only to the final estimator step
        try:
            last_step_name = model.steps[-1][0]
            fit_params = {f"{last_step_name}__sample_weight": sample_weights}
            model.fit(X_train, y_train, **fit_params)
        except TypeError:
            # Fallback: model doesn't accept sample_weight
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        if len(preds) < 2:
            spearman = float("nan")
        else:
            corr, _ = spearmanr(preds, y_test)
            spearman = float(corr)

        mae = float(np.mean(np.abs(preds - y_test.values)))
        results.append({"season": held_out, "spearman": spearman, "mae": mae})

    return results
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_train.py::test_loocv_returns_one_fold_per_season tests/test_train.py::test_loocv_spearman_is_float -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: implement leave-one-season-out CV helper"
```

---

## Task 3: Implement rule-change weight tuning

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
from train import tune_rule_change_weight

def test_tune_weight_returns_best_from_candidates():
    X, y, meta = _make_fake_data()
    model = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    best_weight, weight_scores = tune_rule_change_weight(model, X, y, meta)
    assert best_weight in [1.0, 1.5, 2.0, 2.5, 3.0]
    assert set(weight_scores.keys()) == {1.0, 1.5, 2.0, 2.5, 3.0}
    for v in weight_scores.values():
        assert isinstance(v, float)

def test_tune_weight_picks_highest_spearman():
    X, y, meta = _make_fake_data()
    model = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    best_weight, weight_scores = tune_rule_change_weight(model, X, y, meta)
    best_score = weight_scores[best_weight]
    for score in weight_scores.values():
        assert best_score >= score
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_train.py::test_tune_weight_returns_best_from_candidates tests/test_train.py::test_tune_weight_picks_highest_spearman -v
```

Expected: `ImportError: cannot import name 'tune_rule_change_weight'`

- [ ] **Step 3: Implement `tune_rule_change_weight` in `src/train.py`**

Add after `leave_one_season_out_cv`:

```python
def tune_rule_change_weight(model, X: pd.DataFrame, y: pd.Series,
                             meta: pd.DataFrame) -> tuple[float, dict]:
    """
    Try each weight in RULE_CHANGE_WEIGHT_CANDIDATES via leave-one-season-out CV.
    Returns (best_weight, {weight: avg_spearman}).
    NaN folds are excluded from the average.
    """
    import copy
    weight_scores = {}

    for w in RULE_CHANGE_WEIGHT_CANDIDATES:
        m = copy.deepcopy(model)
        folds = leave_one_season_out_cv(m, X, y, meta, rule_change_weight=w)
        spearmans = [f["spearman"] for f in folds if not np.isnan(f["spearman"])]
        avg = float(np.mean(spearmans)) if spearmans else float("nan")
        weight_scores[w] = avg
        log.info("  Weight %.1f → avg Spearman: %.4f", w, avg)

    best_weight = max(weight_scores, key=lambda w: weight_scores[w])
    return best_weight, weight_scores
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_train.py::test_tune_weight_returns_best_from_candidates tests/test_train.py::test_tune_weight_picks_highest_spearman -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: implement rule-change weight tuning via CV"
```

---

## Task 4: Implement model comparison (XGBoost vs Ridge)

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
from train import compare_models

def test_compare_models_returns_winner_and_results():
    X, y, meta = _make_fake_data()
    best_weight = 1.0
    winner_name, winner_model, cv_df = compare_models(X, y, meta, best_weight)
    assert winner_name in ("XGBoost", "Ridge")
    assert hasattr(winner_model, "predict")
    assert set(cv_df.columns) >= {"model", "season", "spearman", "mae"}

def test_compare_models_winner_has_higher_spearman():
    X, y, meta = _make_fake_data()
    winner_name, winner_model, cv_df = compare_models(X, y, meta, rule_change_weight=1.0)
    xgb_avg = cv_df[cv_df["model"] == "XGBoost"]["spearman"].mean()
    ridge_avg = cv_df[cv_df["model"] == "Ridge"]["spearman"].mean()
    if winner_name == "XGBoost":
        assert xgb_avg >= ridge_avg
    else:
        assert ridge_avg >= xgb_avg
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_train.py::test_compare_models_returns_winner_and_results tests/test_train.py::test_compare_models_winner_has_higher_spearman -v
```

Expected: `ImportError: cannot import name 'compare_models'`

- [ ] **Step 3: Implement `compare_models` in `src/train.py`**

Add after `tune_rule_change_weight`:

```python
def compare_models(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                   rule_change_weight: float) -> tuple[str, object, pd.DataFrame]:
    """
    Run leave-one-season-out CV for XGBoost and Ridge. Returns:
      (winner_name, winner_pipeline, cv_results_dataframe)
    winner is whichever has higher average Spearman across folds.
    """
    import copy

    candidates = {
        "XGBoost": Pipeline([
            ("imp", SimpleImputer()),
            ("reg", XGBRegressor(
                max_depth=4, n_estimators=200, learning_rate=0.05,
                random_state=42, verbosity=0,
            )),
        ]),
        "Ridge": Pipeline([
            ("imp", SimpleImputer()),
            ("reg", Ridge(alpha=1.0)),
        ]),
    }

    all_rows = []
    avg_spearmans = {}

    for name, pipeline in candidates.items():
        log.info("Running CV for %s ...", name)
        folds = leave_one_season_out_cv(
            copy.deepcopy(pipeline), X, y, meta, rule_change_weight
        )
        for fold in folds:
            all_rows.append({"model": name, **fold})
        spearmans = [f["spearman"] for f in folds if not np.isnan(f["spearman"])]
        avg = float(np.mean(spearmans)) if spearmans else float("nan")
        avg_spearmans[name] = avg
        log.info("  %s avg Spearman: %.4f", name, avg)

    cv_df = pd.DataFrame(all_rows)
    winner_name = max(avg_spearmans, key=lambda n: avg_spearmans[n])
    winner_pipeline = candidates[winner_name]
    return winner_name, winner_pipeline, cv_df
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_train.py::test_compare_models_returns_winner_and_results tests/test_train.py::test_compare_models_winner_has_higher_spearman -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: implement model comparison via CV (XGBoost vs Ridge)"
```

---

## Task 5: Implement final retraining, output printing, and persistence

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
from train import retrain_and_save

def test_retrain_and_save_creates_pkl(tmp_path):
    X, y, meta = _make_fake_data()
    pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    model_path = str(tmp_path / "model.pkl")
    retrain_and_save(
        pipeline, X, y, meta,
        winner_name="Ridge",
        rule_change_weight=1.5,
        model_path=model_path,
    )
    assert os.path.exists(model_path)
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    assert "model" in bundle
    assert "feature_cols" in bundle
    assert "winner_name" in bundle
    assert "rule_change_weight" in bundle
    assert bundle["feature_cols"] == FEATURE_COLS
    assert bundle["winner_name"] == "Ridge"
    assert bundle["rule_change_weight"] == 1.5
    # Saved model should be able to predict
    preds = bundle["model"].predict(X)
    assert len(preds) == len(X)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_train.py::test_retrain_and_save_creates_pkl -v
```

Expected: `ImportError: cannot import name 'retrain_and_save'`

- [ ] **Step 3: Implement `retrain_and_save` and `main` in `src/train.py`**

Add after `compare_models`:

```python
def retrain_and_save(pipeline, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                     winner_name: str, rule_change_weight: float, model_path: str):
    """Retrain pipeline on all data with best weight, then save a bundle to model_path."""
    rc_col = X["is_rule_change_year"]
    sample_weights = np.where(rc_col == 1, rule_change_weight, 1.0)

    last_step_name = pipeline.steps[-1][0]
    try:
        pipeline.fit(X, y, **{f"{last_step_name}__sample_weight": sample_weights})
    except TypeError:
        pipeline.fit(X, y)

    bundle = {
        "model": pipeline,
        "feature_cols": FEATURE_COLS,
        "winner_name": winner_name,
        "rule_change_weight": rule_change_weight,
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved model bundle → %s", model_path)


def _print_summary(winner_name: str, best_weight: float, weight_scores: dict,
                   cv_df: pd.DataFrame, pipeline):
    print("\n" + "=" * 60)
    print("GRID PROPHET — TRAINING SUMMARY")
    print("=" * 60)

    print("\n-- Rule-change weight tuning --")
    for w, s in sorted(weight_scores.items()):
        marker = " ◄ BEST" if w == best_weight else ""
        print(f"  weight {w:.1f} → avg Spearman {s:.4f}{marker}")

    print("\n-- Model comparison (leave-one-season-out CV) --")
    summary = cv_df.groupby("model")[["spearman", "mae"]].mean()
    for model_name, row in summary.iterrows():
        marker = " ◄ WINNER" if model_name == winner_name else ""
        print(f"  {model_name}: avg Spearman {row['spearman']:.4f}, avg MAE {row['mae']:.4f}{marker}")

    print(f"\n-- Winner: {winner_name} (weight={best_weight:.1f}) --")

    # Feature importances (XGBoost only; Ridge uses coefficients)
    last_estimator = pipeline.steps[-1][1]
    if hasattr(last_estimator, "feature_importances_"):
        print("\n-- Feature importances (XGBoost) --")
        importances = last_estimator.feature_importances_
        for col, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
            print(f"  {col:<42} {imp:.4f}")
    elif hasattr(last_estimator, "coef_"):
        print("\n-- Ridge coefficients --")
        for col, coef in sorted(zip(FEATURE_COLS, last_estimator.coef_), key=lambda x: -abs(x[1])):
            print(f"  {col:<42} {coef:+.4f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Grid Prophet model.")
    parser.add_argument(
        "--features", default=os.path.join(DATA_DIR, "features.csv"),
        help="Path to features.csv"
    )
    parser.add_argument(
        "--model-out", default=os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl"),
        help="Output path for saved model"
    )
    parser.add_argument(
        "--cv-out", default=os.path.join(DATA_DIR, "cv_results.csv"),
        help="Output path for CV results CSV"
    )
    args = parser.parse_args()

    log.info("Loading data from %s ...", args.features)
    X, y, meta = load_data(args.features)
    log.info("Training set: %d rows, %d seasons", len(X), meta["year"].nunique())

    # Build a Ridge pipeline to tune weights (cheap model for tuning)
    import copy
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge

    tune_pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge(alpha=1.0))])

    log.info("Tuning rule-change sample weight ...")
    best_weight, weight_scores = tune_rule_change_weight(tune_pipeline, X, y, meta)
    log.info("Best rule-change weight: %.1f", best_weight)

    log.info("Comparing XGBoost vs Ridge ...")
    winner_name, winner_pipeline, cv_df = compare_models(X, y, meta, best_weight)

    cv_df.to_csv(args.cv_out, index=False)
    log.info("CV results saved → %s", args.cv_out)

    _print_summary(winner_name, best_weight, weight_scores, cv_df, winner_pipeline)

    log.info("Retraining %s on all data ...", winner_name)
    retrain_and_save(
        winner_pipeline, X, y, meta,
        winner_name=winner_name,
        rule_change_weight=best_weight,
        model_path=args.model_out,
    )

    log.info("Done. Model saved to %s", args.model_out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_train.py -v
```

Expected: all tests `PASSED` (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: implement retrain_and_save, summary printing, and main CLI"
```

---

## Task 6: Run end-to-end and verify outputs

This task has no unit tests — it's a manual smoke test of the full pipeline.

- [ ] **Step 1: Ensure data is up to date (prerequisite)**

Check that `data/features.csv` covers 2014–2025:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/features.csv'); print(sorted(df['year'].unique()))"
```

Expected: `[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]`

If the list stops at 2022, run the data pipeline first:
```bash
python src/collect.py --resume --end-year 2025
python src/features.py
```

- [ ] **Step 2: Run training**

```bash
python src/train.py
```

Expected output (values will vary):
```
GRID PROPHET — TRAINING SUMMARY
================================
-- Rule-change weight tuning --
  weight 1.0 → avg Spearman 0.XXXX
  ...
  weight X.X → avg Spearman 0.XXXX ◄ BEST

-- Model comparison (leave-one-season-out CV) --
  XGBoost: avg Spearman 0.XXXX, avg MAE 0.XXXX
  Ridge:   avg Spearman 0.XXXX, avg MAE 0.XXXX  ◄ WINNER

-- Feature importances / Ridge coefficients --
  ...
```

- [ ] **Step 3: Verify output files exist**

```bash
python -c "
import os, pickle
assert os.path.exists('models/Grid_Prophet_model.pkl'), 'model missing'
assert os.path.exists('data/cv_results.csv'), 'cv_results missing'
with open('models/Grid_Prophet_model.pkl', 'rb') as f:
    b = pickle.load(f)
print('Model bundle keys:', list(b.keys()))
print('Winner:', b['winner_name'])
print('Weight:', b['rule_change_weight'])
print('Can predict:', b['model'].predict([[0.3, 5.0, 1, -0.5, 0.3, 3, 0.4, 2.0]]))
"
```

Expected: no assertion errors, model bundle prints correctly, prediction is a float array.

- [ ] **Step 4: Commit**

```bash
git add data/cv_results.csv
git commit -m "feat: stage 3 complete — trained Grid Prophet model saved"
```

---

## Self-Review

**Spec coverage:**
- [x] Load `data/features.csv` → `load_data`
- [x] Drop `year`/`constructor` from features → `FEATURE_COLS` list excludes them
- [x] Tune rule-change weight via LOOCV across [1.0, 1.5, 2.0, 2.5, 3.0] → `tune_rule_change_weight`
- [x] XGBoost (max_depth=4, n_estimators=200, lr=0.05) → `compare_models`
- [x] Ridge (alpha=1.0) → `compare_models`
- [x] LOOCV for both, compute Spearman + MAE per fold → `leave_one_season_out_cv`
- [x] Select model with higher avg Spearman → `compare_models` winner logic
- [x] Print: best weight, comparison results, feature importances → `_print_summary`
- [x] Retrain winner on all data with best weight → `retrain_and_save`
- [x] Save to `models/Grid_Prophet_model.pkl` → `retrain_and_save`
- [x] Save CV results to `data/cv_results.csv` → `main`

**Data gap:** Raw data currently covers only 2014–2022. The pre-flight step addresses this.

**Placeholder scan:** No TBDs, no "implement later", all code blocks complete.

**Type consistency:** `leave_one_season_out_cv` returns `list[dict]` with keys `season/spearman/mae` — consumed correctly in `tune_rule_change_weight` and `compare_models`.
