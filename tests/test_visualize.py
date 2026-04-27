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
