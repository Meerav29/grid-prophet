# tests/test_train.py
import sys
import os
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from train import load_data, FEATURE_COLS, leave_one_season_out_cv, tune_rule_change_weight

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
    assert sorted(meta["year"].tolist()) == [2014, 2014, 2015, 2015, 2016, 2016]
    assert 2026 not in meta["year"].values


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

def test_loocv_accepts_nonidentity_weight():
    X, y, meta = _make_fake_data()
    model = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
    # weight=1.0 is identity; weight=2.5 should run without error
    results_weighted = leave_one_season_out_cv(model, X, y, meta, rule_change_weight=2.5)
    assert len(results_weighted) == 4
    for fold in results_weighted:
        assert isinstance(fold["spearman"], float)

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
        if np.isnan(best_score) and np.isnan(score):
            continue
        elif np.isnan(best_score):
            assert False, "best_score is NaN but other scores are not"
        else:
            assert best_score >= score
