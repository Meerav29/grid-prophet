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
