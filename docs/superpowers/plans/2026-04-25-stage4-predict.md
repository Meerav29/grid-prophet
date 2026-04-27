# Stage 4 — 2026 Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `predict.py` to collect 2026 rounds 1–2 from FastF1, build 2026 feature rows, run the trained model, and print + save ranked predictions.

**Architecture:** `predict.py` imports reusable helpers directly from `collect.py` and `features.py` (no code duplication). It collects 2026 rounds 1–2 via FastF1, appends them to the existing race results, rebuilds the full standings/features pipeline for 2026 only, loads the saved model bundle, predicts points share per constructor, and prints a ranked table. Cadillac is a new 2026 entrant with no historical data — it is handled gracefully (NaN momentum features passed through `SimpleImputer` in the trained pipeline).

**Tech Stack:** fastf1, pandas, numpy, scikit-learn, xgboost, pickle — all already installed.

---

## Pre-Flight Notes

- `data/race_results.csv` currently covers 2014–2025 (5,065 rows). 2026 rounds 1–2 must be collected and appended.
- `data/features.csv` has NO 2026 rows yet. Task 2 builds them.
- `models/Grid_Prophet_model.pkl` exists and is a valid bundle with keys: `model`, `feature_cols`, `winner_name`, `rule_change_weight`.
- `features.py` already has `2026` in `RULE_CHANGE_YEARS`, so `is_rule_change_year=1` for all 2026 rows is correct.
- Cadillac is a brand-new 2026 constructor. `_canonical("Cadillac")` returns `"Cadillac"` (unknown → pass-through). It will have NaN for all momentum/history features — that is correct; the imputer handles it.
- 2026 rounds 1–2 FastF1 data is confirmed available (Australian GP and Chinese GP results verified).
- Windows terminal uses cp1252 — use only ASCII characters in all print statements (no →, ◄, —, etc.).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/predict.py` | Implement (currently a docstring stub) | Full prediction pipeline |
| `tests/test_predict.py` | Create | Unit tests for predict.py functions |
| `tests/conftest.py` | No change | sys.path already set up |

---

### Task 1: Collect 2026 early-season data

**Context:** `collect.py` already has `collect_race_results(start_year, end_year)` which fetches FastF1 data and returns a DataFrame. We'll call it directly to get 2026 rounds 1–2, then append to the existing CSV.

**Files:**
- Modify: `src/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_predict.py`:

```python
import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import collect_2026_early_rounds


def test_collect_2026_early_rounds_returns_dataframe(tmp_path):
    """collect_2026_early_rounds returns a DataFrame with expected columns."""
    # We can't hit FastF1 in a unit test, so we mock collect_race_results
    import unittest.mock as mock
    fake_rows = pd.DataFrame({
        "year": [2026, 2026],
        "round": [1, 1],
        "event_name": ["Fake GP", "Fake GP"],
        "driver": ["Driver A", "Driver B"],
        "abbreviation": ["DRA", "DRB"],
        "constructor": ["Mercedes", "Ferrari"],
        "grid_position": [1, 2],
        "finish_position": [1, 2],
        "classification": ["Finished", "Finished"],
        "points": [25.0, 18.0],
    })
    with mock.patch("predict.collect_race_results", return_value=fake_rows):
        result = collect_2026_early_rounds()
    assert isinstance(result, pd.DataFrame)
    expected_cols = {"year", "round", "constructor", "driver", "points", "finish_position"}
    assert expected_cols.issubset(set(result.columns))
    assert (result["year"] == 2026).all()


def test_collect_2026_early_rounds_filters_to_early_rounds(tmp_path):
    """collect_2026_early_rounds only returns rounds <= EARLY_ROUNDS."""
    import unittest.mock as mock
    fake_rows = pd.DataFrame({
        "year": [2026, 2026, 2026],
        "round": [1, 2, 3],
        "event_name": ["GP1", "GP2", "GP3"],
        "driver": ["D1", "D2", "D3"],
        "abbreviation": ["D1", "D2", "D3"],
        "constructor": ["Mercedes", "Ferrari", "McLaren"],
        "grid_position": [1, 2, 3],
        "finish_position": [1, 2, 3],
        "classification": ["Finished", "Finished", "Finished"],
        "points": [25.0, 18.0, 15.0],
    })
    with mock.patch("predict.collect_race_results", return_value=fake_rows):
        result = collect_2026_early_rounds()
    assert result["round"].max() <= 2
```

- [ ] **Step 2: Run the test to confirm it fails**

```
cd c:\Users\meera\Github-Projects\grid-prophet
pytest tests/test_predict.py -v
```

Expected: `ImportError: cannot import name 'collect_2026_early_rounds' from 'predict'`

- [ ] **Step 3: Implement `collect_2026_early_rounds` in `src/predict.py`**

Replace the docstring stub in `src/predict.py` with:

```python
"""Generate 2026 constructor championship predictions from early-season data."""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from collect import collect_race_results
from features import (
    EARLY_ROUNDS,
    RULE_CHANGE_YEARS,
    CONSTRUCTOR_LINEAGE,
    _canonical,
    _early_season_features,
    _rule_change_features,
    _momentum_features,
    _driver_quality_feature,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

PREDICT_YEAR = 2026


def collect_2026_early_rounds() -> pd.DataFrame:
    """Fetch rounds 1..EARLY_ROUNDS of PREDICT_YEAR from FastF1."""
    log.info("Collecting %d rounds 1-%d from FastF1 ...", PREDICT_YEAR, EARLY_ROUNDS)
    raw = collect_race_results(PREDICT_YEAR, PREDICT_YEAR)
    return raw[raw["round"] <= EARLY_ROUNDS].copy()
```

- [ ] **Step 4: Run the tests to confirm they pass**

```
pytest tests/test_predict.py::test_collect_2026_early_rounds_returns_dataframe tests/test_predict.py::test_collect_2026_early_rounds_filters_to_early_rounds -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: predict.py scaffold with collect_2026_early_rounds"
```

---

### Task 2: Build 2026 feature rows

**Context:** We need to construct the 2026 feature rows by combining the freshly-collected 2026 early-race data with the existing 2014–2025 race history. We reuse the same feature-builder functions from `features.py`.

The pipeline:
1. Load `data/race_results.csv` (2014–2025)
2. Append the 2026 rows from `collect_2026_early_rounds()`
3. Canonicalize constructor names in both `results` and the existing `constructor_standings.csv`
4. Re-aggregate standings (canonical merge) and compute points_share
5. Call `_early_season_features`, `_rule_change_features`, `_momentum_features`, `_driver_quality_feature` on the combined data
6. Filter result to `year == 2026` only

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_predict.py`:

```python
from predict import build_2026_features


def _make_minimal_race_history():
    """3 seasons of fake race data for 2 constructors."""
    rows = []
    for year in [2024, 2025]:
        for rnd in [1, 2]:
            for driver, team, pts, pos in [
                ("Driver A", "Mercedes", 25.0 if rnd == 1 else 18.0, 1),
                ("Driver B", "Ferrari", 18.0 if rnd == 1 else 25.0, 2),
            ]:
                rows.append({
                    "year": year, "round": rnd, "event_name": "GP",
                    "driver": driver, "abbreviation": driver[:3].upper(),
                    "constructor": team, "grid_position": pos,
                    "finish_position": pos, "classification": "Finished",
                    "points": pts,
                })
    return pd.DataFrame(rows)


def test_build_2026_features_returns_2026_rows():
    """build_2026_features returns only 2026 rows, one per constructor."""
    import unittest.mock as mock
    history = _make_minimal_race_history()
    early_2026 = pd.DataFrame({
        "year": [2026, 2026], "round": [1, 1], "event_name": ["GP", "GP"],
        "driver": ["Driver A", "Driver B"], "abbreviation": ["DRA", "DRB"],
        "constructor": ["Mercedes", "Ferrari"],
        "grid_position": [1, 2], "finish_position": [1, 2],
        "classification": ["Finished", "Finished"], "points": [25.0, 18.0],
    })
    standings_hist = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "constructor": ["Mercedes", "Ferrari", "Mercedes", "Ferrari"],
        "total_points": [200.0, 150.0, 180.0, 160.0],
        "standing": [1, 2, 1, 2],
    })
    with mock.patch("predict.collect_2026_early_rounds", return_value=early_2026), \
         mock.patch("predict.pd.read_csv", side_effect=[history, standings_hist]):
        result = build_2026_features()
    assert isinstance(result, pd.DataFrame)
    assert (result["year"] == 2026).all()
    assert set(result["constructor"]).issubset({"Mercedes", "Ferrari"})


def test_build_2026_features_has_required_columns():
    """build_2026_features output has all model feature columns."""
    import unittest.mock as mock
    from train import FEATURE_COLS
    history = _make_minimal_race_history()
    early_2026 = pd.DataFrame({
        "year": [2026, 2026], "round": [1, 1], "event_name": ["GP", "GP"],
        "driver": ["Driver A", "Driver B"], "abbreviation": ["DRA", "DRB"],
        "constructor": ["Mercedes", "Ferrari"],
        "grid_position": [1, 2], "finish_position": [1, 2],
        "classification": ["Finished", "Finished"], "points": [25.0, 18.0],
    })
    standings_hist = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "constructor": ["Mercedes", "Ferrari", "Mercedes", "Ferrari"],
        "total_points": [200.0, 150.0, 180.0, 160.0],
        "standing": [1, 2, 1, 2],
    })
    with mock.patch("predict.collect_2026_early_rounds", return_value=early_2026), \
         mock.patch("predict.pd.read_csv", side_effect=[history, standings_hist]):
        result = build_2026_features()
    for col in FEATURE_COLS:
        assert col in result.columns, f"Missing feature column: {col}"
```

- [ ] **Step 2: Run the test to confirm it fails**

```
pytest tests/test_predict.py::test_build_2026_features_returns_2026_rows -v
```

Expected: `ImportError: cannot import name 'build_2026_features' from 'predict'`

- [ ] **Step 3: Implement `build_2026_features` in `src/predict.py`**

Append after `collect_2026_early_rounds`:

```python
def build_2026_features() -> pd.DataFrame:
    """Build the 2026 feature rows by combining historical data with 2026 early rounds."""
    race_csv = os.path.join(DATA_DIR, "race_results.csv")
    standings_csv = os.path.join(DATA_DIR, "constructor_standings.csv")

    log.info("Loading historical race results ...")
    history = pd.read_csv(race_csv)

    log.info("Loading historical constructor standings ...")
    standings_raw = pd.read_csv(standings_csv)

    log.info("Collecting 2026 early rounds ...")
    early_2026 = collect_2026_early_rounds()

    # Combine history with 2026 early data
    results = pd.concat([history, early_2026], ignore_index=True)
    results["constructor_canonical"] = results["constructor"].apply(_canonical)

    # Canonicalize and re-aggregate standings from historical data only (2026 has no full-season standings yet)
    standings_raw["constructor_canonical"] = standings_raw["constructor"].apply(_canonical)
    standings = (
        standings_raw.groupby(["year", "constructor_canonical"])["total_points"]
        .sum()
        .reset_index()
    )
    # Re-rank after canonical merge
    ranked_rows = []
    for year, grp in standings.groupby("year"):
        grp = grp.sort_values("total_points", ascending=False).reset_index(drop=True)
        grp["standing"] = grp.index + 1
        ranked_rows.append(grp)
    standings = pd.concat(ranked_rows, ignore_index=True)

    # Add points_share for momentum features
    season_total = standings.groupby("year")["total_points"].sum().rename("season_total")
    standings = standings.merge(season_total, on="year")
    standings["points_share"] = standings["total_points"] / standings["season_total"]

    # Add 2026 constructors to standings (no standing yet — just so feature builders see them)
    constructors_2026 = results[results["year"] == PREDICT_YEAR]["constructor_canonical"].unique()
    stub_rows = pd.DataFrame({
        "year": PREDICT_YEAR,
        "constructor_canonical": constructors_2026,
        "total_points": np.nan,
        "standing": np.nan,
        "season_total": np.nan,
        "points_share": np.nan,
    })
    standings = pd.concat([standings, stub_rows], ignore_index=True)

    log.info("Computing features for 2026 ...")
    early = _early_season_features(results)
    rc = _rule_change_features(standings)
    momentum = _momentum_features(standings)
    driver_q = _driver_quality_feature(results)

    # Build base from 2026 constructors only
    base = pd.DataFrame({
        "year": PREDICT_YEAR,
        "constructor_canonical": constructors_2026,
    })
    features = base.copy()
    for df in [early, rc, momentum, driver_q]:
        features = features.merge(df, on=["year", "constructor_canonical"], how="left")

    features = features.rename(columns={"constructor_canonical": "constructor"})
    features["season_points_share"] = np.nan  # target unknown for 2026
    return features
```

- [ ] **Step 4: Run the tests to confirm they pass**

```
pytest tests/test_predict.py::test_build_2026_features_returns_2026_rows tests/test_predict.py::test_build_2026_features_has_required_columns -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: build_2026_features constructs 2026 feature rows"
```

---

### Task 3: Load model and generate predictions

**Context:** Load the model bundle from `models/Grid_Prophet_model.pkl`, extract feature columns, run `.predict()`, and return a ranked DataFrame with `constructor` and `predicted_points_share` columns.

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_predict.py`:

```python
from predict import load_model_bundle, predict_standings
import pickle
import tempfile
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer


def _make_fake_bundle(tmp_path):
    """Create a minimal model bundle pickle for testing."""
    from train import FEATURE_COLS
    pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    # Fit on tiny data so the pipeline is trained
    X = pd.DataFrame({col: [0.1, 0.2] for col in FEATURE_COLS})
    y = pd.Series([0.3, 0.4])
    pipeline.fit(X, y)
    bundle = {
        "model": pipeline,
        "feature_cols": FEATURE_COLS,
        "winner_name": "Ridge",
        "rule_change_weight": 1.5,
    }
    path = tmp_path / "test_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return str(path), bundle


def test_load_model_bundle_returns_bundle(tmp_path):
    """load_model_bundle loads and returns a dict with expected keys."""
    path, _ = _make_fake_bundle(tmp_path)
    bundle = load_model_bundle(path)
    assert isinstance(bundle, dict)
    for key in ("model", "feature_cols", "winner_name", "rule_change_weight"):
        assert key in bundle


def test_predict_standings_returns_ranked_df(tmp_path):
    """predict_standings returns a DataFrame ranked by predicted_points_share descending."""
    from train import FEATURE_COLS
    _, bundle = _make_fake_bundle(tmp_path)
    features_2026 = pd.DataFrame({
        "year": [2026, 2026],
        "constructor": ["Mercedes", "Ferrari"],
        **{col: [0.5, 0.3] for col in FEATURE_COLS},
        "season_points_share": [np.nan, np.nan],
    })
    result = predict_standings(bundle, features_2026)
    assert isinstance(result, pd.DataFrame)
    assert "constructor" in result.columns
    assert "predicted_points_share" in result.columns
    assert "rank" in result.columns
    # Ranked 1 first (descending by predicted share)
    assert result.iloc[0]["rank"] == 1
    assert result["predicted_points_share"].iloc[0] >= result["predicted_points_share"].iloc[1]
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
pytest tests/test_predict.py::test_load_model_bundle_returns_bundle tests/test_predict.py::test_predict_standings_returns_ranked_df -v
```

Expected: `ImportError: cannot import name 'load_model_bundle' from 'predict'`

- [ ] **Step 3: Implement `load_model_bundle` and `predict_standings` in `src/predict.py`**

Append after `build_2026_features`:

```python
def load_model_bundle(model_path: str) -> dict:
    """Load and return the pickled model bundle."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_standings(bundle: dict, features_2026: pd.DataFrame) -> pd.DataFrame:
    """Run prediction and return a DataFrame ranked by predicted_points_share."""
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X = features_2026[feature_cols].copy()
    preds = model.predict(X)

    result = features_2026[["constructor"]].copy()
    result["predicted_points_share"] = preds
    result = result.sort_values("predicted_points_share", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result
```

- [ ] **Step 4: Run the tests to confirm they pass**

```
pytest tests/test_predict.py::test_load_model_bundle_returns_bundle tests/test_predict.py::test_predict_standings_returns_ranked_df -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: load_model_bundle and predict_standings"
```

---

### Task 4: Print formatted table, save CSV, and wire CLI

**Context:** Add `_print_predictions` to display a formatted ASCII standings table, then wire up `main()` with argparse. All print output must use ASCII only (no Unicode arrows or dashes — Windows cp1252 encoding).

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_predict.py`:

```python
import io
from unittest.mock import patch
from predict import _print_predictions


def test_print_predictions_outputs_all_constructors(capsys):
    """_print_predictions prints one line per constructor."""
    predictions = pd.DataFrame({
        "rank": [1, 2, 3],
        "constructor": ["Mercedes", "Ferrari", "McLaren"],
        "predicted_points_share": [0.35, 0.28, 0.20],
    })
    _print_predictions(predictions, winner_name="XGBoost", rule_change_weight=3.0)
    captured = capsys.readouterr()
    assert "Mercedes" in captured.out
    assert "Ferrari" in captured.out
    assert "McLaren" in captured.out


def test_print_predictions_ranked_order(capsys):
    """_print_predictions prints constructors in rank order (rank 1 first)."""
    predictions = pd.DataFrame({
        "rank": [1, 2],
        "constructor": ["Mercedes", "Ferrari"],
        "predicted_points_share": [0.35, 0.28],
    })
    _print_predictions(predictions, winner_name="XGBoost", rule_change_weight=3.0)
    captured = capsys.readouterr()
    assert captured.out.index("Mercedes") < captured.out.index("Ferrari")
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
pytest tests/test_predict.py::test_print_predictions_outputs_all_constructors tests/test_predict.py::test_print_predictions_ranked_order -v
```

Expected: `ImportError: cannot import name '_print_predictions' from 'predict'`

- [ ] **Step 3: Implement `_print_predictions` and `main` in `src/predict.py`**

Append after `predict_standings`:

```python
def _print_predictions(predictions: pd.DataFrame, winner_name: str, rule_change_weight: float):
    """Print a formatted 2026 constructor standings prediction table."""
    print("\n" + "=" * 60)
    print("GRID PROPHET - 2026 CONSTRUCTOR CHAMPIONSHIP PREDICTION")
    print("=" * 60)
    print(f"Model: {winner_name}  |  Rule-change weight: {rule_change_weight:.1f}")
    print(f"Based on rounds 1-{EARLY_ROUNDS} early-season data")
    print()
    print(f"  {'Rank':<6} {'Constructor':<24} {'Predicted Share':>15}")
    print("  " + "-" * 48)
    for _, row in predictions.iterrows():
        print(f"  {int(row['rank']):<6} {row['constructor']:<24} {row['predicted_points_share']:>14.1%}")
    print("=" * 60 + "\n")


def main():
    import argparse
    import fastf1

    parser = argparse.ArgumentParser(description="Generate 2026 Grid Prophet predictions.")
    parser.add_argument(
        "--model", default=os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl"),
    )
    parser.add_argument(
        "--out", default=os.path.join(DATA_DIR, "predictions_2026.csv"),
    )
    args = parser.parse_args()

    cache_dir = os.path.join(DATA_DIR, "fastf1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    log.info("Loading model from %s ...", args.model)
    bundle = load_model_bundle(args.model)
    log.info("Model: %s  |  rule_change_weight: %.1f", bundle["winner_name"], bundle["rule_change_weight"])

    log.info("Building 2026 features ...")
    features_2026 = build_2026_features()
    log.info("2026 constructors: %s", sorted(features_2026["constructor"].tolist()))

    log.info("Running predictions ...")
    predictions = predict_standings(bundle, features_2026)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    predictions.to_csv(args.out, index=False)
    log.info("Predictions saved -> %s", args.out)

    _print_predictions(predictions, bundle["winner_name"], bundle["rule_change_weight"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all predict tests**

```
pytest tests/test_predict.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Run full test suite to confirm no regressions**

```
pytest tests/ -v
```

Expected: All tests PASSED (9 train tests + predict tests)

- [ ] **Step 6: Smoke test the full predict pipeline end-to-end**

```
cd c:\Users\meera\Github-Projects\grid-prophet
python src/predict.py
```

Expected output: A ranked prediction table printed to console, `data/predictions_2026.csv` written. Verify with:

```
python -c "import pandas as pd; print(pd.read_csv('data/predictions_2026.csv').to_string())"
```

- [ ] **Step 7: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: complete predict.py with CLI, print table, and save CSV"
```

---

## Self-Review

**1. Spec coverage (Planning.md Stage 4):**

| Requirement | Covered by |
|-------------|-----------|
| Load model from pkl | Task 3: `load_model_bundle` |
| Load features.csv / filter to 2026 rows | Task 2: `build_2026_features` |
| If 2026 rows don't exist, construct from 2025 lineup + early rounds | Task 1+2: collect FastF1 2026 data then build features |
| Run prediction → predicted points_share per constructor | Task 3: `predict_standings` |
| Rank and print formatted standings table | Task 4: `_print_predictions` |
| Save to data/predictions_2026.csv | Task 4: `main()` |

All requirements covered. ✅

**2. Placeholder scan:** No TBDs, TODOs, or vague instructions. All code blocks complete. ✅

**3. Type consistency:**
- `collect_2026_early_rounds()` returns `pd.DataFrame` — used in `build_2026_features` ✅
- `build_2026_features()` returns `pd.DataFrame` — used in `predict_standings` ✅
- `load_model_bundle(path)` returns `dict` — used in `predict_standings` and `_print_predictions` ✅
- `predict_standings(bundle, features_2026)` returns `pd.DataFrame` with columns `constructor`, `predicted_points_share`, `rank` — used in `_print_predictions` and `.to_csv()` ✅
- `_print_predictions(predictions, winner_name, rule_change_weight)` — called in `main()` with correct args ✅
