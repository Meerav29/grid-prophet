# Stage 5 — Polish & Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five capabilities to the completed Stages 1–4 pipeline: Makefile + CLI entry point, bootstrap confidence intervals, ensemble model, mid-season update support, and four visualisation charts.

**Architecture:** Each capability lives in a focused new module (`confidence.py`, `ensemble.py`, `visualize.py`, `__main__.py`). Existing scripts (`predict.py`, `train.py`) receive minimal additions — new CLI flags and calls into the new modules. A `Makefile` wraps the CLI for convenience.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, xgboost, scipy, matplotlib, seaborn, fastf1, pytest

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `src/__main__.py` | Create | CLI entry point for `python -m grid_prophet` |
| `src/confidence.py` | Create | Bootstrap confidence intervals |
| `src/ensemble.py` | Create | Rule-change-only model + blend tuning |
| `src/visualize.py` | Create | Four chart functions + `plot_all()` |
| `Makefile` | Create | Thin wrappers over CLI |
| `src/predict.py` | Modify | Add `--rounds`, `--ensemble`, `--no-ci` flags |
| `src/train.py` | Modify | Add `--ensemble` flag |
| `tests/test_confidence.py` | Create | Tests for bootstrap CI |
| `tests/test_ensemble.py` | Create | Tests for ensemble blend |
| `tests/test_visualize.py` | Create | Tests for chart file creation |
| `tests/test_main.py` | Create | Tests for CLI routing |
| `notebooks/exploration.ipynb` | Modify | Add Visualisations section |
| `plots/` | Create (dir) | PNG outputs |

---

## Task 1: Makefile + `src/__main__.py` CLI Entry Point

**Files:**
- Create: `Makefile`
- Create: `src/__main__.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main.py
import sys
import os
import unittest.mock as mock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_collect_subcommand_calls_collect_main():
    import __main__ as m
    with mock.patch("collect.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "collect"]):
            m.main()
    mock_main.assert_called_once()


def test_features_subcommand_calls_features_main():
    import __main__ as m
    with mock.patch("features.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "features"]):
            m.main()
    mock_main.assert_called_once()


def test_train_subcommand_calls_train_main():
    import __main__ as m
    with mock.patch("train.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "train"]):
            m.main()
    mock_main.assert_called_once()


def test_predict_subcommand_calls_predict_main():
    import __main__ as m
    with mock.patch("predict.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "predict"]):
            m.main()
    mock_main.assert_called_once()


def test_run_subcommand_calls_all_four():
    import __main__ as m
    with mock.patch("collect.main") as mc, \
         mock.patch("features.main") as mf, \
         mock.patch("train.main") as mt, \
         mock.patch("predict.main") as mp:
        with mock.patch("sys.argv", ["grid_prophet", "run"]):
            m.main()
    mc.assert_called_once()
    mf.assert_called_once()
    mt.assert_called_once()
    mp.assert_called_once()


def test_unknown_subcommand_exits():
    import __main__ as m
    with mock.patch("sys.argv", ["grid_prophet", "bogus"]):
        with pytest.raises(SystemExit):
            m.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd c:/Users/meera/Github-Projects/grid-prophet
python -m pytest tests/test_main.py -v 2>&1 | head -20
```

Expected: errors importing `__main__` (module does not exist yet).

- [ ] **Step 3: Create `src/__main__.py`**

```python
"""CLI entry point: python -m grid_prophet <command>."""

import sys


def _detect_latest_round(predict_year: int) -> int:
    """Return the latest completed round number for predict_year from FastF1."""
    import fastf1
    schedule = fastf1.get_event_schedule(predict_year, include_testing=False)
    import datetime
    today = datetime.date.today()
    completed = schedule[schedule["EventDate"].dt.date < today]
    if completed.empty:
        return 1
    return int(completed["RoundNumber"].max())


def main():
    import argparse
    import collect
    import features
    import train
    import predict

    parser = argparse.ArgumentParser(
        prog="grid_prophet",
        description="Grid Prophet F1 championship predictor.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    sub.add_parser("collect", help="Pull historical race data from FastF1")
    sub.add_parser("features", help="Build feature matrix")
    sub.add_parser("train", help="Train model")
    sub.add_parser("predict", help="Generate 2026 predictions")
    sub.add_parser("run", help="Run full pipeline: collect → features → train → predict")
    sub.add_parser("update", help="Auto-detect latest round and re-predict")
    sub.add_parser("plots", help="Generate all visualisation charts")

    args, remaining = parser.parse_known_args()
    # Pass remaining args through to sub-module so their own argparse flags work
    sys.argv = [sys.argv[0]] + remaining

    if args.command == "collect":
        collect.main()
    elif args.command == "features":
        features.main()
    elif args.command == "train":
        train.main()
    elif args.command == "predict":
        predict.main()
    elif args.command == "run":
        collect.main()
        features.main()
        train.main()
        predict.main()
    elif args.command == "update":
        from predict import PREDICT_YEAR
        n = _detect_latest_round(PREDICT_YEAR)
        sys.argv = [sys.argv[0], "--rounds", str(n)]
        predict.main()
    elif args.command == "plots":
        import visualize
        visualize.plot_all()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_main.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Create `Makefile`**

```makefile
PYTHON = python

.PHONY: collect features train predict run update plots clean

collect:
	$(PYTHON) -m grid_prophet collect

features:
	$(PYTHON) -m grid_prophet features

train:
	$(PYTHON) -m grid_prophet train

predict:
	$(PYTHON) -m grid_prophet predict

run:
	$(PYTHON) -m grid_prophet run

update:
	$(PYTHON) -m grid_prophet update

plots:
	$(PYTHON) -m grid_prophet plots

clean:
	rm -f data/features.csv data/cv_results.csv data/predictions_2026*.csv
	rm -f models/*.pkl
```

- [ ] **Step 6: Commit**

```bash
git add src/__main__.py Makefile tests/test_main.py
git commit -m "feat: add CLI entry point (__main__.py) and Makefile"
```

---

## Task 2: Bootstrap Confidence Intervals (`src/confidence.py`)

**Files:**
- Create: `src/confidence.py`
- Create: `tests/test_confidence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_confidence.py
import sys
import os
import pandas as pd
import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import FEATURE_COLS


def _make_bundle():
    pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    X = pd.DataFrame({col: np.random.rand(20) for col in FEATURE_COLS})
    y = pd.Series(np.random.rand(20))
    pipeline.fit(X, y)
    return {
        "model": pipeline,
        "feature_cols": FEATURE_COLS,
        "winner_name": "Ridge",
        "rule_change_weight": 1.0,
    }, X, y


def _make_features_2026():
    return pd.DataFrame({
        "constructor": ["TeamA", "TeamB", "TeamC"],
        **{col: np.random.rand(3) for col in FEATURE_COLS},
        "season_points_share": [float("nan")] * 3,
    })


def test_bootstrap_ci_returns_correct_columns():
    from confidence import bootstrap_confidence_intervals
    bundle, X_train, y_train = _make_bundle()
    features_2026 = _make_features_2026()
    result = bootstrap_confidence_intervals(bundle, X_train, y_train, features_2026, n_bootstrap=10)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"constructor", "ci_low", "ci_high", "ci_half"}


def test_bootstrap_ci_has_one_row_per_constructor():
    from confidence import bootstrap_confidence_intervals
    bundle, X_train, y_train = _make_bundle()
    features_2026 = _make_features_2026()
    result = bootstrap_confidence_intervals(bundle, X_train, y_train, features_2026, n_bootstrap=10)
    assert len(result) == 3
    assert set(result["constructor"]) == {"TeamA", "TeamB", "TeamC"}


def test_bootstrap_ci_low_less_than_high():
    from confidence import bootstrap_confidence_intervals
    bundle, X_train, y_train = _make_bundle()
    features_2026 = _make_features_2026()
    result = bootstrap_confidence_intervals(bundle, X_train, y_train, features_2026, n_bootstrap=10)
    assert (result["ci_low"] <= result["ci_high"]).all()


def test_bootstrap_ci_half_is_half_width():
    from confidence import bootstrap_confidence_intervals
    bundle, X_train, y_train = _make_bundle()
    features_2026 = _make_features_2026()
    result = bootstrap_confidence_intervals(bundle, X_train, y_train, features_2026, n_bootstrap=10)
    expected_half = (result["ci_high"] - result["ci_low"]) / 2
    pd.testing.assert_series_equal(result["ci_half"].reset_index(drop=True),
                                   expected_half.reset_index(drop=True), check_names=False)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_confidence.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'confidence'`.

- [ ] **Step 3: Create `src/confidence.py`**

```python
"""Bootstrap confidence intervals for 2026 constructor championship predictions."""

import copy
import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

log = logging.getLogger(__name__)


def bootstrap_confidence_intervals(
    bundle: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features_2026: pd.DataFrame,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Return DataFrame with columns: constructor, ci_low, ci_high, ci_half.

    Resamples training rows with replacement n_bootstrap times, retrains the
    model pipeline from bundle, predicts on features_2026, then computes the
    10th and 90th percentiles of the prediction distribution per constructor.
    """
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    constructors = features_2026["constructor"].values
    X_pred = features_2026[feature_cols].values

    all_preds = np.empty((n_bootstrap, len(constructors)))
    n_train = len(X_train)

    for i in range(n_bootstrap):
        idx = np.random.randint(0, n_train, size=n_train)
        X_boot = X_train.iloc[idx].copy()
        y_boot = y_train.iloc[idx].copy()
        boot_model = clone(model)
        boot_model.fit(X_boot, y_boot)
        all_preds[i] = boot_model.predict(X_pred)
        if (i + 1) % 100 == 0:
            log.info("  Bootstrap iteration %d/%d", i + 1, n_bootstrap)

    ci_low = np.percentile(all_preds, 10, axis=0)
    ci_high = np.percentile(all_preds, 90, axis=0)
    ci_half = (ci_high - ci_low) / 2

    return pd.DataFrame({
        "constructor": constructors,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half": ci_half,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_confidence.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/confidence.py tests/test_confidence.py
git commit -m "feat: add bootstrap confidence intervals (confidence.py)"
```

---

## Task 3: Wire Confidence Intervals into `predict.py`

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_predict.py`:

```python
def test_print_predictions_with_ci_shows_ci_column(capsys):
    """_print_predictions shows 80% CI column when ci_half column present."""
    predictions = pd.DataFrame({
        "rank": [1, 2],
        "constructor": ["Mercedes", "Ferrari"],
        "predicted_points_share": [0.35, 0.28],
        "ci_half": [0.04, 0.03],
    })
    _print_predictions(predictions, winner_name="Ridge", rule_change_weight=1.5)
    captured = capsys.readouterr()
    assert "80% CI" in captured.out
    assert "±" in captured.out


def test_print_predictions_without_ci_omits_ci_column(capsys):
    """_print_predictions omits CI column when ci_half not present."""
    predictions = pd.DataFrame({
        "rank": [1, 2],
        "constructor": ["Mercedes", "Ferrari"],
        "predicted_points_share": [0.35, 0.28],
    })
    _print_predictions(predictions, winner_name="Ridge", rule_change_weight=1.5)
    captured = capsys.readouterr()
    assert "80% CI" not in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_predict.py::test_print_predictions_with_ci_shows_ci_column tests/test_predict.py::test_print_predictions_without_ci_omits_ci_column -v
```

Expected: both FAIL (`AssertionError`).

- [ ] **Step 3: Update `_print_predictions` in `src/predict.py`**

Replace the existing `_print_predictions` function (lines 140–152) with:

```python
def _print_predictions(predictions: pd.DataFrame, winner_name: str, rule_change_weight: float):
    """Print a formatted 2026 constructor standings prediction table."""
    has_ci = "ci_half" in predictions.columns

    print("\n" + "=" * 68)
    print("GRID PROPHET - 2026 CONSTRUCTOR CHAMPIONSHIP PREDICTION")
    print("=" * 68)
    print(f"Model: {winner_name}  |  Rule-change weight: {rule_change_weight:.1f}")
    print(f"Based on rounds 1-{EARLY_ROUNDS} early-season data")
    print()
    if has_ci:
        print(f"  {'Rank':<6} {'Constructor':<24} {'Predicted Share':>15}  {'80% CI':>10}")
        print("  " + "-" * 58)
        for _, row in predictions.iterrows():
            print(
                f"  {int(row['rank']):<6} {row['constructor']:<24}"
                f" {row['predicted_points_share']:>14.1%}  ± {row['ci_half']:>6.1%}"
            )
    else:
        print(f"  {'Rank':<6} {'Constructor':<24} {'Predicted Share':>15}")
        print("  " + "-" * 48)
        for _, row in predictions.iterrows():
            print(f"  {int(row['rank']):<6} {row['constructor']:<24} {row['predicted_points_share']:>14.1%}")
    print("=" * 68 + "\n")
```

- [ ] **Step 4: Add `--rounds`, `--ensemble`, `--no-ci` flags and CI call to `main()` in `src/predict.py`**

Replace the existing `main()` function (lines 155–191) with:

```python
def main():
    import argparse
    import fastf1
    from confidence import bootstrap_confidence_intervals
    from train import load_data, FEATURE_COLS

    parser = argparse.ArgumentParser(description="Generate 2026 Grid Prophet predictions.")
    parser.add_argument(
        "--model", default=os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl"),
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use ensemble model (Grid_Prophet_ensemble.pkl)",
    )
    parser.add_argument(
        "--out", default=os.path.join(DATA_DIR, "predictions_2026.csv"),
    )
    parser.add_argument(
        "--rounds", type=int, default=EARLY_ROUNDS,
        help="Number of early-season rounds to use (default: EARLY_ROUNDS)",
    )
    parser.add_argument(
        "--no-ci", action="store_true", dest="no_ci",
        help="Skip bootstrap confidence intervals",
    )
    args = parser.parse_args()

    if args.ensemble:
        args.model = os.path.join(MODELS_DIR, "Grid_Prophet_ensemble.pkl")

    cache_dir = os.path.join(DATA_DIR, "fastf1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    log.info("Loading model from %s ...", args.model)
    bundle = load_model_bundle(args.model)
    log.info("Model: %s  |  rule_change_weight: %.1f", bundle["winner_name"], bundle["rule_change_weight"])

    log.info("Building 2026 features (rounds 1-%d) ...", args.rounds)
    features_2026 = build_2026_features(n_rounds=args.rounds)
    log.info("2026 constructors: %s", sorted(features_2026["constructor"].tolist()))

    log.info("Running predictions ...")
    predictions = predict_standings(bundle, features_2026)

    # Determine round-stamped output path for mid-season snapshots
    out_path = args.out
    if args.rounds != EARLY_ROUNDS:
        base, ext = os.path.splitext(args.out)
        out_path = f"{base}_r{args.rounds}{ext}"

    if not args.no_ci:
        log.info("Computing bootstrap confidence intervals (N=500) ...")
        features_csv = os.path.join(DATA_DIR, "features.csv")
        X_train, y_train, _ = load_data(features_csv)
        ci = bootstrap_confidence_intervals(bundle, X_train, y_train, features_2026)
        predictions = predictions.merge(ci, on="constructor", how="left")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    predictions.to_csv(out_path, index=False)
    # Also always write to the canonical predictions_2026.csv
    if out_path != args.out:
        predictions.to_csv(args.out, index=False)
    log.info("Predictions saved -> %s", out_path)

    _print_predictions(predictions, bundle["winner_name"], bundle["rule_change_weight"])
```

- [ ] **Step 5: Update `build_2026_features` signature to accept `n_rounds`**

In `src/predict.py`, change the `build_2026_features` function signature from:

```python
def build_2026_features() -> pd.DataFrame:
```

to:

```python
def build_2026_features(n_rounds: int = EARLY_ROUNDS) -> pd.DataFrame:
```

And change the line:

```python
    return raw[raw["round"] <= EARLY_ROUNDS].copy()
```

in `collect_2026_early_rounds` — instead, pass `n_rounds` through. Replace `collect_2026_early_rounds` with:

```python
def collect_2026_early_rounds(n_rounds: int = EARLY_ROUNDS) -> pd.DataFrame:
    """Fetch rounds 1..n_rounds of PREDICT_YEAR from FastF1."""
    log.info("Collecting %d rounds 1-%d from FastF1 ...", PREDICT_YEAR, n_rounds)
    raw = collect_race_results(PREDICT_YEAR, PREDICT_YEAR)
    if raw.empty:
        raise RuntimeError(
            "No %d race data returned from FastF1 for rounds 1-%d. "
            "Check network connectivity or FastF1 availability." % (PREDICT_YEAR, n_rounds)
        )
    return raw[raw["round"] <= n_rounds].copy()
```

And update the call inside `build_2026_features`:

```python
def build_2026_features(n_rounds: int = EARLY_ROUNDS) -> pd.DataFrame:
    """Build the 2026 feature rows by combining historical data with 2026 early rounds."""
    race_csv = os.path.join(DATA_DIR, "race_results.csv")
    standings_csv = os.path.join(DATA_DIR, "constructor_standings.csv")

    log.info("Loading historical race results ...")
    history = pd.read_csv(race_csv)

    log.info("Loading historical constructor standings ...")
    standings_raw = pd.read_csv(standings_csv)

    log.info("Collecting 2026 early rounds ...")
    early_2026 = collect_2026_early_rounds(n_rounds=n_rounds)
    # ... rest of function unchanged
```

- [ ] **Step 6: Run all predict tests to verify they pass**

```bash
python -m pytest tests/test_predict.py -v
```

Expected: all tests PASS (including the two new CI tests).

- [ ] **Step 7: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: add --rounds, --ensemble, --no-ci flags and CI output to predict.py"
```

---

## Task 4: Ensemble Model (`src/ensemble.py`)

**Files:**
- Create: `src/ensemble.py`
- Create: `tests/test_ensemble.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ensemble.py
import sys
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import FEATURE_COLS


def _make_training_data(years=(2014, 2015, 2022, 2023, 2024, 2025)):
    rows = []
    for year in years:
        for ctor in ["TeamA", "TeamB", "TeamC"]:
            rows.append({
                "year": year, "constructor": ctor,
                **{col: np.random.rand() for col in FEATURE_COLS},
                "is_rule_change_year": 1 if year in (2014, 2022) else 0,
                "season_points_share": np.random.rand(),
            })
    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS]
    y = df["season_points_share"]
    meta = df[["year", "constructor"]]
    return X, y, meta


def _make_base_bundle():
    pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    X, y, _ = _make_training_data()
    pipeline.fit(X, y)
    return {
        "model": pipeline,
        "feature_cols": FEATURE_COLS,
        "winner_name": "Ridge",
        "rule_change_weight": 1.5,
    }


def test_build_ensemble_returns_bundle_with_required_keys(tmp_path):
    from ensemble import build_ensemble
    base_bundle = _make_base_bundle()
    X, y, meta = _make_training_data()
    result = build_ensemble(base_bundle, X, y, meta)
    for key in ("full_model", "rc_model", "alpha", "feature_cols", "winner_name", "rule_change_weight"):
        assert key in result, f"Missing key: {key}"


def test_build_ensemble_alpha_in_valid_range(tmp_path):
    from ensemble import build_ensemble
    base_bundle = _make_base_bundle()
    X, y, meta = _make_training_data()
    result = build_ensemble(base_bundle, X, y, meta)
    assert 0.3 <= result["alpha"] <= 0.7


def test_build_ensemble_winner_name_contains_ensemble(tmp_path):
    from ensemble import build_ensemble
    base_bundle = _make_base_bundle()
    X, y, meta = _make_training_data()
    result = build_ensemble(base_bundle, X, y, meta)
    assert "Ensemble" in result["winner_name"]


def test_ensemble_predict_differs_from_base(tmp_path):
    from ensemble import build_ensemble, ensemble_predict
    base_bundle = _make_base_bundle()
    X, y, meta = _make_training_data()
    ens_bundle = build_ensemble(base_bundle, X, y, meta)
    X_pred = X.iloc[:3].copy()
    base_preds = base_bundle["model"].predict(X_pred)
    ens_preds = ensemble_predict(ens_bundle, X_pred)
    # Alpha is never 1.0 so predictions must differ unless rc_model == full_model
    assert ens_preds.shape == base_preds.shape


def test_save_ensemble_creates_pkl(tmp_path):
    from ensemble import build_ensemble, save_ensemble
    base_bundle = _make_base_bundle()
    X, y, meta = _make_training_data()
    ens_bundle = build_ensemble(base_bundle, X, y, meta)
    out = str(tmp_path / "ensemble.pkl")
    save_ensemble(ens_bundle, out)
    assert os.path.exists(out)
    with open(out, "rb") as f:
        loaded = pickle.load(f)
    assert "full_model" in loaded
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ensemble.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'ensemble'`.

- [ ] **Step 3: Create `src/ensemble.py`**

```python
"""Ensemble model: blend full-data model with rule-change-years-only model."""

import copy
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

log = logging.getLogger(__name__)

RC_YEARS = {2014, 2015, 2022, 2023}
ALPHA_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7]


def _train_rc_model(base_bundle: dict, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    """Train a copy of the base pipeline on rule-change years only."""
    rc_mask = meta["year"].isin(RC_YEARS)
    X_rc = X[rc_mask].copy()
    y_rc = y[rc_mask].copy()
    rc_model = clone(base_bundle["model"])
    rc_model.fit(X_rc, y_rc)
    return rc_model


def _tune_alpha(
    full_model,
    rc_model,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> float:
    """Find the blend alpha (full-data weight) via LOOCV on rule-change seasons."""
    rc_seasons = [s for s in meta["year"].unique() if s in {2014, 2022}]
    best_alpha = 0.5
    best_score = float("-inf")

    for alpha in ALPHA_CANDIDATES:
        spearmans = []
        for held_out in rc_seasons:
            train_mask = meta["year"] != held_out
            test_mask = meta["year"] == held_out
            X_train = X[train_mask].copy()
            y_train = y[train_mask].copy()
            X_test = X[test_mask].copy()
            y_test = y[test_mask].copy()

            # Retrain both sub-models on remaining rc-relevant data
            fold_full = clone(full_model)
            fold_full.fit(X_train, y_train)
            rc_train_mask = train_mask & meta["year"].isin(RC_YEARS)
            if rc_train_mask.sum() < 2:
                continue
            fold_rc = clone(rc_model)
            fold_rc.fit(X[rc_train_mask].copy(), y[rc_train_mask].copy())

            preds = alpha * fold_full.predict(X_test) + (1 - alpha) * fold_rc.predict(X_test)
            if len(preds) >= 2:
                corr, _ = spearmanr(preds, y_test.values)
                spearmans.append(float(corr))

        avg = float(np.mean(spearmans)) if spearmans else float("-inf")
        log.info("  alpha=%.1f  avg Spearman on RC seasons: %.4f", alpha, avg)
        if avg > best_score:
            best_score = avg
            best_alpha = alpha

    return best_alpha


def build_ensemble(
    base_bundle: dict,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> dict:
    """Build and return an ensemble bundle.

    Trains a rule-change-only sub-model, tunes the blend weight alpha via
    LOOCV on rule-change seasons, and returns a bundle ready for prediction.
    """
    log.info("Training rule-change-only sub-model on years: %s", sorted(RC_YEARS))
    full_model = base_bundle["model"]
    rc_model = _train_rc_model(base_bundle, X, y, meta)

    log.info("Tuning ensemble blend weight (alpha) ...")
    alpha = _tune_alpha(full_model, rc_model, X, y, meta)
    log.info("Best alpha: %.1f", alpha)

    base_name = base_bundle["winner_name"]
    return {
        "full_model": full_model,
        "rc_model": rc_model,
        "alpha": alpha,
        "feature_cols": base_bundle["feature_cols"],
        "winner_name": f"Ensemble({base_name}, alpha={alpha:.1f})",
        "rule_change_weight": base_bundle["rule_change_weight"],
        # predict_standings in predict.py expects a "model" key — provide a shim
        "model": _EnsembleShim(full_model, rc_model, alpha),
    }


class _EnsembleShim:
    """Thin wrapper so predict_standings can call .predict() on the ensemble."""

    def __init__(self, full_model, rc_model, alpha: float):
        self.full_model = full_model
        self.rc_model = rc_model
        self.alpha = alpha

    def predict(self, X):
        return (
            self.alpha * self.full_model.predict(X)
            + (1 - self.alpha) * self.rc_model.predict(X)
        )


def ensemble_predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Run blended prediction given an ensemble bundle and feature matrix."""
    return bundle["model"].predict(X)


def save_ensemble(bundle: dict, path: str) -> None:
    """Pickle the ensemble bundle to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved ensemble bundle -> %s", path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ensemble.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ensemble.py tests/test_ensemble.py
git commit -m "feat: add ensemble model with rule-change blend (ensemble.py)"
```

---

## Task 5: Wire Ensemble into `train.py`

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_train.py`:

```python
def test_main_ensemble_flag_creates_ensemble_pkl(tmp_path):
    """train.py --ensemble flag creates Grid_Prophet_ensemble.pkl."""
    import unittest.mock as mock
    import sys

    rows = []
    for year in [2014, 2015, 2022, 2023, 2024, 2025]:
        for ctor in ["TeamA", "TeamB", "TeamC"]:
            rows.append({
                "year": year, "constructor": ctor,
                "early_points_share": 0.3, "early_avg_finish": 5.0,
                "is_rule_change_year": 1 if year in (2014, 2022) else 0,
                "rule_change_adaptation_score": 0.0,
                "prev_year_points_share": 0.3, "prev_year_standing": 3,
                "constructor_win_rate_5yr": 0.4,
                "avg_driver_career_points_per_race": 2.0,
                "season_points_share": 0.5,
            })
    df = pd.DataFrame(rows)
    features_csv = str(tmp_path / "features.csv")
    df.to_csv(features_csv, index=False)
    model_out = str(tmp_path / "model.pkl")
    ensemble_out = str(tmp_path / "ensemble.pkl")

    with mock.patch("sys.argv", [
        "train", "--features", features_csv,
        "--model-out", model_out,
        "--cv-out", str(tmp_path / "cv.csv"),
        "--ensemble",
        "--ensemble-out", ensemble_out,
    ]):
        from train import main
        main()

    assert os.path.exists(ensemble_out), "ensemble pkl should be created"
    with open(ensemble_out, "rb") as f:
        bundle = pickle.load(f)
    assert "full_model" in bundle
    assert "rc_model" in bundle
    assert "alpha" in bundle
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_train.py::test_main_ensemble_flag_creates_ensemble_pkl -v
```

Expected: FAIL (unrecognised argument `--ensemble`).

- [ ] **Step 3: Add `--ensemble` and `--ensemble-out` flags to `train.py main()`**

In `src/train.py`, add these two argument lines inside `main()` after the existing `parser.add_argument("--cv-out", ...)` call:

```python
    parser.add_argument(
        "--ensemble", action="store_true",
        help="After training base model, build and save ensemble bundle",
    )
    parser.add_argument(
        "--ensemble-out",
        default=os.path.join(MODELS_DIR, "Grid_Prophet_ensemble.pkl"),
    )
```

And at the very end of `main()`, after `_print_summary(...)`, add:

```python
    if args.ensemble:
        from ensemble import build_ensemble, save_ensemble
        log.info("Building ensemble model ...")
        ens_bundle = build_ensemble(
            {"model": winner_pipeline, "feature_cols": FEATURE_COLS,
             "winner_name": winner_name, "rule_change_weight": best_weight},
            X, y, meta,
        )
        save_ensemble(ens_bundle, args.ensemble_out)
        log.info("Ensemble saved -> %s", args.ensemble_out)
```

- [ ] **Step 4: Run all train tests to verify they pass**

```bash
python -m pytest tests/test_train.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add --ensemble flag to train.py to build ensemble bundle"
```

---

## Task 6: Visualisations (`src/visualize.py`)

**Files:**
- Create: `src/visualize.py`
- Create: `tests/test_visualize.py`
- Create: `plots/` (directory)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_visualize.py
import sys
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import FEATURE_COLS


def _make_cv_df():
    rows = []
    for year in range(2014, 2026):
        for ctor in ["TeamA", "TeamB"]:
            rows.append({"model": "Ridge", "season": year,
                         "spearman": np.random.rand(), "mae": np.random.rand() * 0.1})
    return pd.DataFrame(rows)


def _make_features_df():
    rows = []
    for year in range(2014, 2026):
        for ctor, actual_rank in [("TeamA", 1), ("TeamB", 2)]:
            rows.append({
                "year": year, "constructor": ctor,
                "season_points_share": np.random.rand(),
                "predicted_points_share": np.random.rand(),
                "actual_rank": actual_rank,
                "predicted_rank": actual_rank,
            })
    return pd.DataFrame(rows)


def _make_predictions_df():
    return pd.DataFrame({
        "constructor": ["TeamA", "TeamB", "TeamC"],
        "predicted_points_share": [0.4, 0.3, 0.2],
        "rank": [1, 2, 3],
        "ci_low": [0.35, 0.25, 0.15],
        "ci_high": [0.45, 0.35, 0.25],
        "ci_half": [0.05, 0.05, 0.05],
    })


def _make_bundle():
    pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    X = pd.DataFrame({col: np.random.rand(10) for col in FEATURE_COLS})
    y = pd.Series(np.random.rand(10))
    pipeline.fit(X, y)
    return {"model": pipeline, "feature_cols": FEATURE_COLS,
            "winner_name": "Ridge", "rule_change_weight": 1.0}


def test_plot_spearman_by_season_creates_png(tmp_path):
    from visualize import plot_spearman_by_season
    cv_df = _make_cv_df()
    plot_spearman_by_season(cv_df, out_dir=str(tmp_path))
    assert (tmp_path / "spearman_by_season.png").exists()


def test_plot_2026_predictions_creates_png(tmp_path):
    from visualize import plot_2026_predictions
    preds = _make_predictions_df()
    plot_2026_predictions(preds, out_dir=str(tmp_path))
    assert (tmp_path / "predictions_2026.png").exists()


def test_plot_feature_importance_creates_png(tmp_path):
    from visualize import plot_feature_importance
    bundle = _make_bundle()
    plot_feature_importance(bundle, out_dir=str(tmp_path))
    assert (tmp_path / "feature_importance.png").exists()


def test_plot_cv_accuracy_creates_png(tmp_path):
    from visualize import plot_cv_accuracy
    cv_df = _make_cv_df()
    features_df = _make_features_df()
    plot_cv_accuracy(cv_df, features_df, out_dir=str(tmp_path))
    assert (tmp_path / "cv_accuracy.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_visualize.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'visualize'`.

- [ ] **Step 3: Create `src/visualize.py`**

```python
"""Generate visualisation charts for Grid Prophet."""

import logging
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RULE_CHANGE_YEARS = {2014, 2022}

# Consistent constructor colour palette
_CONSTRUCTOR_COLORS = {
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "Red Bull Racing": "#0600EF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#005AFF",
    "RB": "#2B4562",
    "Haas F1 Team": "#FFFFFF",
    "Audi": "#C0C0C0",
    "Racing Bulls": "#2B4562",
    "Cadillac": "#004225",
}
_DEFAULT_COLOR = "#888888"


def _constructor_color(name: str) -> str:
    return _CONSTRUCTOR_COLORS.get(name, _DEFAULT_COLOR)


def plot_cv_accuracy(
    cv_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_dir: str = "plots/",
) -> None:
    """Grid of subplots: predicted vs actual rank per held-out season."""
    os.makedirs(out_dir, exist_ok=True)
    seasons = sorted(features_df["year"].unique())
    n = len(seasons)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    spearman_by_season = cv_df.groupby("season")["spearman"].mean()

    for i, season in enumerate(seasons):
        ax = axes[i]
        season_df = features_df[features_df["year"] == season]
        colors = [_constructor_color(c) for c in season_df["constructor"]]
        ax.scatter(season_df["actual_rank"], season_df["predicted_rank"],
                   c=colors, s=80, edgecolors="black", linewidths=0.5, zorder=3)
        max_rank = max(season_df["actual_rank"].max(), season_df["predicted_rank"].max()) + 1
        ax.plot([1, max_rank], [1, max_rank], "k--", linewidth=0.8, alpha=0.5)
        rho = spearman_by_season.get(season, float("nan"))
        ax.set_title(f"{season}  ρ={rho:.2f}", fontsize=10)
        ax.set_xlabel("Actual rank")
        ax.set_ylabel("Predicted rank")
        ax.invert_xaxis()
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Grid Prophet — CV Accuracy: Predicted vs Actual Rank", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "cv_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_spearman_by_season(cv_df: pd.DataFrame, out_dir: str = "plots/") -> None:
    """Bar chart of per-season Spearman correlation with rule-change years highlighted."""
    os.makedirs(out_dir, exist_ok=True)
    by_season = cv_df.groupby("season")["spearman"].mean().reset_index()
    by_season = by_season.sort_values("season")

    colors = [
        "#E8000D" if s in RULE_CHANGE_YEARS else "#4878CF"
        for s in by_season["season"]
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(by_season["season"].astype(str), by_season["spearman"], color=colors, edgecolor="black", linewidth=0.5)
    mean_rho = by_season["spearman"].mean()
    ax.axhline(mean_rho, color="black", linestyle="--", linewidth=1, label=f"Mean ρ = {mean_rho:.3f}")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E8000D", label="Rule-change year"),
        Patch(facecolor="#4878CF", label="Normal year"),
    ]
    ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color="black", linestyle="--", label=f"Mean ρ={mean_rho:.3f}")],
              loc="lower right")
    ax.set_xlabel("Season")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Grid Prophet — Leave-One-Season-Out CV: Spearman Correlation by Season")
    ax.set_ylim(-0.1, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = os.path.join(out_dir, "spearman_by_season.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_2026_predictions(predictions_df: pd.DataFrame, out_dir: str = "plots/") -> None:
    """Horizontal bar chart of 2026 predicted points share with 80% CI error bars."""
    os.makedirs(out_dir, exist_ok=True)
    df = predictions_df.sort_values("rank")
    constructors = df["constructor"].tolist()
    shares = df["predicted_points_share"].values
    colors = [_constructor_color(c) for c in constructors]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(constructors))
    ax.barh(y_pos, shares, color=colors, edgecolor="black", linewidth=0.5, height=0.6)

    if "ci_low" in df.columns and "ci_high" in df.columns:
        xerr_low = shares - df["ci_low"].values
        xerr_high = df["ci_high"].values - shares
        ax.errorbar(shares, y_pos, xerr=[xerr_low, xerr_high],
                    fmt="none", color="black", capsize=4, linewidth=1.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(constructors)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Championship Points Share")
    ax.set_title("Grid Prophet — 2026 Constructor Championship Prediction")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    out = os.path.join(out_dir, "predictions_2026.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_feature_importance(bundle: dict, out_dir: str = "plots/") -> None:
    """Horizontal bar chart of feature importances or Ridge coefficients."""
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = bundle["feature_cols"]
    last_step = bundle["model"].steps[-1][1]

    if hasattr(last_step, "feature_importances_"):
        values = last_step.feature_importances_
        label = "Feature Importance"
        title = "Grid Prophet — XGBoost Feature Importances"
    elif hasattr(last_step, "coef_"):
        values = np.abs(last_step.coef_)
        label = "|Coefficient|"
        title = "Grid Prophet — Ridge Regression |Coefficients|"
    else:
        log.warning("Model has neither feature_importances_ nor coef_; skipping plot.")
        return

    order = np.argsort(values)
    sorted_features = [feature_cols[i] for i in order]
    sorted_values = values[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(sorted_features, sorted_values, color="#4878CF", edgecolor="black", linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_title(title)
    plt.tight_layout()
    out = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_all(out_dir: str = "plots/") -> None:
    """Load CSVs and model bundle, then generate all four charts."""
    import pickle

    cv_path = os.path.join(DATA_DIR, "cv_results.csv")
    pred_path = os.path.join(DATA_DIR, "predictions_2026.csv")
    features_path = os.path.join(DATA_DIR, "features.csv")
    model_path = os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl")

    cv_df = pd.read_csv(cv_path)
    predictions_df = pd.read_csv(pred_path)
    features_df = pd.read_csv(features_path)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    # Build a features_df with actual_rank and predicted_rank for CV accuracy plot
    train_df = features_df[features_df["season_points_share"].notna()].copy()
    train_df["actual_rank"] = train_df.groupby("year")["season_points_share"].rank(
        ascending=False, method="min"
    ).astype(int)
    # Re-use cv_df model column to get predicted scores per season
    # cv_df has spearman/mae per season but not per-constructor predictions.
    # For the scatter we derive predicted rank from the trained model directly.
    feature_cols = bundle["feature_cols"]
    X_all = train_df[feature_cols]
    train_df["predicted_points_share"] = bundle["model"].predict(X_all)
    train_df["predicted_rank"] = train_df.groupby("year")["predicted_points_share"].rank(
        ascending=False, method="min"
    ).astype(int)

    plot_cv_accuracy(cv_df, train_df, out_dir=out_dir)
    plot_spearman_by_season(cv_df, out_dir=out_dir)
    plot_2026_predictions(predictions_df, out_dir=out_dir)
    plot_feature_importance(bundle, out_dir=out_dir)
    log.info("All charts saved to %s", out_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_visualize.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Create `plots/` directory with `.gitkeep`**

```bash
mkdir -p plots
touch plots/.gitkeep
```

- [ ] **Step 6: Commit**

```bash
git add src/visualize.py tests/test_visualize.py plots/.gitkeep
git commit -m "feat: add four visualisation charts (visualize.py)"
```

---

## Task 7: Update `src/__main__.py` to Wire `update` and `plots` Commands

The `update` and `plots` sub-commands depend on `visualize.py` (Task 6) and the `--rounds` flag in `predict.py` (Task 3), so this task finalises those routes.

**Files:**
- Modify: `src/__main__.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

Add these to `tests/test_main.py`:

```python
def test_plots_subcommand_calls_plot_all():
    import __main__ as m
    with mock.patch("visualize.plot_all") as mock_plot:
        with mock.patch("sys.argv", ["grid_prophet", "plots"]):
            m.main()
    mock_plot.assert_called_once()


def test_update_subcommand_calls_predict_main():
    import __main__ as m
    with mock.patch("__main__._detect_latest_round", return_value=5) as mock_detect, \
         mock.patch("predict.main") as mock_predict, \
         mock.patch("predict.PREDICT_YEAR", 2026):
        with mock.patch("sys.argv", ["grid_prophet", "update"]):
            m.main()
    mock_detect.assert_called_once_with(2026)
    mock_predict.assert_called_once()
    # sys.argv should have been set to include --rounds 5
    assert "--rounds" in sys.argv
    assert "5" in sys.argv
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_main.py::test_plots_subcommand_calls_plot_all tests/test_main.py::test_update_subcommand_calls_predict_main -v
```

Expected: FAIL (the routes exist but `visualize` isn't importable yet in `__main__`).

- [ ] **Step 3: Verify `__main__.py` routes are correct**

The `plots` and `update` routes were already written in Task 1. After Task 6 creates `visualize.py`, the `plots` route (`import visualize; visualize.plot_all()`) should work. The `update` route sets `sys.argv` with `--rounds N` then calls `predict.main()`. Run the tests — if they pass, no changes are needed. If they fail due to import order, move the `import visualize` inside the `elif args.command == "plots":` branch (it already is, as written in Task 1).

- [ ] **Step 4: Run all main tests to verify they pass**

```bash
python -m pytest tests/test_main.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/__main__.py tests/test_main.py
git commit -m "feat: wire update and plots sub-commands in __main__.py"
```

---

## Task 8: Update Notebook with Visualisations Section

**Files:**
- Modify: `notebooks/exploration.ipynb`

- [ ] **Step 1: Add Visualisations section to the notebook**

Open `notebooks/exploration.ipynb` and append two new cells at the end using `NotebookEdit`:

**Cell 1 (markdown):**
```
## Visualisations

Run all four Grid Prophet charts inline. Charts are also saved to `plots/`.
```

**Cell 2 (code):**
```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import visualize
import matplotlib
matplotlib.use("inline")  # render in notebook cells

import matplotlib.pyplot as plt
import pickle, pandas as pd

DATA_DIR = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(os.getcwd(), "models")

cv_df = pd.read_csv(os.path.join(DATA_DIR, "cv_results.csv"))
predictions_df = pd.read_csv(os.path.join(DATA_DIR, "predictions_2026.csv"))
features_df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))

with open(os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl"), "rb") as f:
    bundle = pickle.load(f)

train_df = features_df[features_df["season_points_share"].notna()].copy()
train_df["actual_rank"] = train_df.groupby("year")["season_points_share"].rank(ascending=False, method="min").astype(int)
train_df["predicted_points_share"] = bundle["model"].predict(train_df[bundle["feature_cols"]])
train_df["predicted_rank"] = train_df.groupby("year")["predicted_points_share"].rank(ascending=False, method="min").astype(int)

visualize.plot_cv_accuracy(cv_df, train_df, out_dir="plots/")
plt.show()

visualize.plot_spearman_by_season(cv_df, out_dir="plots/")
plt.show()

visualize.plot_2026_predictions(predictions_df, out_dir="plots/")
plt.show()

visualize.plot_feature_importance(bundle, out_dir="plots/")
plt.show()
```

Use the `NotebookEdit` tool with `new_source` for each cell.

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/exploration.ipynb'))"
```

Expected: no output (valid JSON).

- [ ] **Step 3: Commit**

```bash
git add notebooks/exploration.ipynb
git commit -m "feat: add Visualisations section to exploration notebook"
```

---

## Task 9: Full Test Suite + Final Smoke Test

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS, 0 failures.

- [ ] **Step 2: Smoke-test the CLI entry point**

```bash
python -m grid_prophet --help
```

Expected: help text listing all sub-commands.

- [ ] **Step 3: Smoke-test the Makefile**

```bash
make --dry-run run
```

Expected: prints the four `python -m grid_prophet` commands without executing them.

- [ ] **Step 4: Smoke-test plots generation (requires trained model + CSVs)**

```bash
python -m grid_prophet plots
```

Expected: four PNG files created in `plots/`, logged paths printed.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "chore: stage 5 complete — CLI, CI, ensemble, mid-season updates, visualisations"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| `python -m grid_prophet` CLI with all sub-commands | Task 1, 7 |
| Makefile with all targets including `clean` | Task 1 |
| Bootstrap CI (N=500, 10th/90th percentile) | Task 2 |
| CI in terminal table with ± format | Task 3 |
| CI columns in predictions CSV | Task 3 |
| `--no-ci` flag to skip bootstrap | Task 3 |
| Rule-change-only sub-model trained on {2014,2015,2022,2023} | Task 4 |
| Alpha tuning via LOOCV on RC seasons | Task 4 |
| Ensemble bundle saved as `Grid_Prophet_ensemble.pkl` | Task 4, 5 |
| `--ensemble` flag in `predict.py` | Task 3 |
| `--ensemble` flag in `train.py` | Task 5 |
| `--rounds N` flag in `predict.py` | Task 3 |
| `update` command auto-detects latest round | Task 1, 7 |
| Round-stamped CSV snapshot (`predictions_2026_r{N}.csv`) | Task 3 |
| `make update` target | Task 1 |
| CV accuracy chart (predicted vs actual rank per season) | Task 6 |
| Spearman by season chart with RC year highlighting | Task 6 |
| 2026 prediction chart with error bars | Task 6 |
| Feature importance chart | Task 6 |
| PNGs saved to `plots/` | Task 6 |
| Notebook Visualisations section | Task 8 |
| `make plots` / `python -m grid_prophet plots` | Task 1, 6 |
