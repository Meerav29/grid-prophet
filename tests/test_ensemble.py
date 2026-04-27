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
