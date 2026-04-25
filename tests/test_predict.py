import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import collect_2026_early_rounds, build_2026_features
from train import FEATURE_COLS


def test_collect_2026_early_rounds_returns_dataframe():
    """collect_2026_early_rounds returns a DataFrame with expected columns."""
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


def test_collect_2026_early_rounds_filters_to_early_rounds():
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
