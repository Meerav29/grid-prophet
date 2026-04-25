import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import collect_2026_early_rounds


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
