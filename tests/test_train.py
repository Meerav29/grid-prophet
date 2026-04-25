# tests/test_train.py
import pandas as pd
import numpy as np
import pytest
import sys
import os

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
    assert sorted(meta["year"].tolist()) == [2014, 2014, 2015, 2015, 2016, 2016]
    assert 2026 not in meta["year"].values
