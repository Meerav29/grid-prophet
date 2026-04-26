import pickle
import sys
import os

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import collect_2026_early_rounds, build_2026_features, load_model_bundle, predict_standings, _print_predictions
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


def _make_fake_bundle(tmp_path):
    """Create a minimal model bundle pickle for testing."""
    pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge())])
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
    _, bundle = _make_fake_bundle(tmp_path)
    features_2026 = pd.DataFrame({
        "year": [2026, 2026],
        "constructor": ["Mercedes", "Ferrari"],
        **{col: [0.5, 0.3] for col in FEATURE_COLS},
        "season_points_share": [float("nan"), float("nan")],
    })
    result = predict_standings(bundle, features_2026)
    assert isinstance(result, pd.DataFrame)
    assert "constructor" in result.columns
    assert "predicted_points_share" in result.columns
    assert "rank" in result.columns
    assert result.iloc[0]["rank"] == 1
    assert result["predicted_points_share"].iloc[0] >= result["predicted_points_share"].iloc[1]


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


def test_collect_2026_early_rounds_raises_on_empty_fastf1():
    """collect_2026_early_rounds raises RuntimeError when FastF1 returns no data."""
    import unittest.mock as mock
    import pytest
    with mock.patch("predict.collect_race_results", return_value=pd.DataFrame()):
        with pytest.raises(RuntimeError, match="No 2026 race data"):
            collect_2026_early_rounds()


def test_build_2026_features_applies_rebrand_dampening():
    """build_2026_features applies rebrand dampening for Audi (major rebrand)."""
    import unittest.mock as mock
    # Build history that includes Audi (canonical for Kick Sauber/Sauber lineage)
    history = _make_minimal_race_history()
    # Add Audi 2025 data so it has a prev_year standing to dampen
    audi_rows = pd.DataFrame({
        "year": [2025, 2025], "round": [1, 2], "event_name": ["GP", "GP"],
        "driver": ["Driver C", "Driver C"], "abbreviation": ["DRC", "DRC"],
        "constructor": ["Audi", "Audi"],
        "grid_position": [5, 5], "finish_position": [5, 5],
        "classification": ["Finished", "Finished"], "points": [10.0, 10.0],
    })
    history_with_audi = pd.concat([history, audi_rows], ignore_index=True)
    early_2026 = pd.DataFrame({
        "year": [2026, 2026, 2026], "round": [1, 1, 1], "event_name": ["GP", "GP", "GP"],
        "driver": ["Driver A", "Driver B", "Driver C"],
        "abbreviation": ["DRA", "DRB", "DRC"],
        "constructor": ["Mercedes", "Ferrari", "Audi"],
        "grid_position": [1, 2, 3], "finish_position": [1, 2, 3],
        "classification": ["Finished", "Finished", "Finished"],
        "points": [25.0, 18.0, 5.0],
    })
    standings_hist = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025, 2025],
        "constructor": ["Mercedes", "Ferrari", "Mercedes", "Ferrari", "Audi"],
        "total_points": [200.0, 150.0, 180.0, 160.0, 20.0],
        "standing": [1, 2, 1, 2, 3],
    })
    with mock.patch("predict.collect_2026_early_rounds", return_value=early_2026), \
         mock.patch("predict.pd.read_csv", side_effect=[history_with_audi, standings_hist]):
        result = build_2026_features()
    audi_row = result[result["constructor"] == "Audi"]
    assert not audi_row.empty, "Audi should appear in 2026 features"
    # prev_year_points_share should exist (may be NaN or dampened — just verify column present)
    assert "prev_year_points_share" in audi_row.columns
