import unittest.mock as mock
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from features import _early_season_features, build_features, latest_completed_round, EARLY_ROUNDS


def _make_results():
    return pd.DataFrame([
        {"year": 2024, "round": 1, "constructor_canonical": "TeamA", "driver": "Alice",
         "points": 25.0, "grid_position": 2, "finish_position": 1, "classification": "Finished"},
        {"year": 2024, "round": 1, "constructor_canonical": "TeamA", "driver": "Bob",
         "points": 0.0, "grid_position": 5, "finish_position": np.nan, "classification": "DNF"},
        {"year": 2024, "round": 1, "constructor_canonical": "TeamB", "driver": "Carl",
         "points": 18.0, "grid_position": 1, "finish_position": 2, "classification": "Finished"},
        {"year": 2024, "round": 1, "constructor_canonical": "TeamB", "driver": "Dana",
         "points": 15.0, "grid_position": 3, "finish_position": 3, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamA", "driver": "Alice",
         "points": 18.0, "grid_position": 1, "finish_position": 2, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamA", "driver": "Bob",
         "points": 25.0, "grid_position": 3, "finish_position": 1, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamB", "driver": "Carl",
         "points": 10.0, "grid_position": 2, "finish_position": 4, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamB", "driver": "Dana",
         "points": 12.0, "grid_position": 4, "finish_position": 3, "classification": "Finished"},
        {"year": 2024, "round": 3, "constructor_canonical": "TeamA", "driver": "Alice",
         "points": 15.0, "grid_position": 1, "finish_position": 3, "classification": "Finished"},
        {"year": 2024, "round": 3, "constructor_canonical": "TeamB", "driver": "Carl",
         "points": 18.0, "grid_position": 2, "finish_position": 2, "classification": "Finished"},
    ])


def test_early_season_features_default_window_uses_EARLY_ROUNDS():
    result = _early_season_features(_make_results())
    teamA = result[result["constructor_canonical"] == "TeamA"].iloc[0]
    # window = rounds 1-2: TeamA total = 25+0+18+25 = 68; all-team window total = 123
    assert teamA["early_points_share"] == pytest.approx(68 / 123)


def test_early_season_features_custom_window_includes_more_rounds():
    result = _early_season_features(_make_results(), early_rounds=3)
    teamA = result[result["constructor_canonical"] == "TeamA"].iloc[0]
    # window = rounds 1-3: TeamA total = 68+15 = 83; all-team window total = 123+33 = 156
    assert teamA["early_points_share"] == pytest.approx(83 / 156)


def test_build_features_accepts_early_rounds_override():
    results = pd.DataFrame(
        [
            {"year": 2024, "round": r, "constructor": "TeamA", "driver": "Alice",
             "points": p, "grid_position": 1, "finish_position": pos, "classification": "Finished"}
            for r, p, pos in [(1, 25.0, 1), (2, 18.0, 2), (3, 15.0, 3)]
        ] + [
            {"year": 2024, "round": 3, "constructor": "TeamB", "driver": "Carl",
             "points": 10.0, "grid_position": 4, "finish_position": 4, "classification": "Finished"},
        ]
    )
    standings = pd.DataFrame({
        "year": [2024, 2024],
        "constructor": ["TeamA", "TeamB"],
        "total_points": [58.0, 10.0],
    })

    with mock.patch("features.pd.read_csv", side_effect=[results, standings]):
        feats_default = build_features()
    with mock.patch("features.pd.read_csv", side_effect=[results, standings]):
        feats_wide = build_features(early_rounds=3)

    row_default = feats_default[feats_default["constructor"] == "TeamA"].iloc[0]
    row_wide = feats_wide[feats_wide["constructor"] == "TeamA"].iloc[0]
    # rounds 1-2: TeamB hasn't scored yet, so TeamA has 100% of the window's points
    assert row_default["early_points_share"] == pytest.approx(1.0)
    # rounds 1-3: TeamB's round-3 points dilute TeamA's share
    assert row_wide["early_points_share"] == pytest.approx((25 + 18 + 15) / (25 + 18 + 15 + 10))
    assert row_default["early_points_share"] != row_wide["early_points_share"]


def test_latest_completed_round_returns_max_completed_round():
    schedule = pd.DataFrame({
        "RoundNumber": [1, 2, 3, 4],
        "EventDate": pd.to_datetime(["2026-03-01", "2026-03-15", "2026-08-01", "2026-08-20"]),
    })
    fake_today = dt.date(2026, 8, 13)
    with mock.patch("fastf1.get_event_schedule", return_value=schedule), \
         mock.patch("datetime.date") as mock_date:
        mock_date.today.return_value = fake_today
        result = latest_completed_round(2026)
    assert result == 3


def test_latest_completed_round_returns_1_when_none_completed():
    schedule = pd.DataFrame({
        "RoundNumber": [1, 2],
        "EventDate": pd.to_datetime(["2026-03-01", "2026-03-15"]),
    })
    fake_today = dt.date(2026, 1, 1)
    with mock.patch("fastf1.get_event_schedule", return_value=schedule), \
         mock.patch("datetime.date") as mock_date:
        mock_date.today.return_value = fake_today
        result = latest_completed_round(2026)
    assert result == 1
