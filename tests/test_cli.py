"""Tests for the deterministic CLI plumbing in src/cli.py (spec sec 6):
argument parsing and the lookup helpers that go from (season, round) to
entrants / circuit metadata. The `fit`/`race`/`backtest` command bodies
themselves invoke the probabilistic model and are covered by the backtest
run, not by unit tests here (see model/pace.py module docstring)."""

import pandas as pd
import pytest

from cli import build_parser, _parse_through, _entrants_for_round, _circuit_info


class TestParseThrough:
    def test_parses_season_colon_round(self):
        assert _parse_through("2026:16") == (2026, 16)

    def test_parses_single_digit_round(self):
        assert _parse_through("2022:1") == (2022, 1)


class TestArgParsing:
    def test_fit_requires_through(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["fit"])

    def test_fit_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["fit", "--through", "2026:16"])
        assert args.through == "2026:16"
        assert args.method == "nuts"

    def test_race_requires_round(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["race", "--mode", "pre"])

    def test_race_defaults_to_pre_mode(self):
        parser = build_parser()
        args = parser.parse_args(["race", "--round", "17"])
        assert args.mode == "pre"
        assert args.season == 2026

    def test_backtest_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["backtest"])
        assert args.seasons == "2022-2025"
        assert args.rounds == "coarse"


def _toy_driver_rounds():
    rows = []
    for season_round, event, ctype in [((2023, 1), "Bahrain Grand Prix", "mixed")]:
        season, rnd = season_round
        for team, drv in [("Team A", "d1"), ("Team A", "d2"), ("Team B", "d3")]:
            rows.append(dict(season=season, round=rnd, event_name=event, circuit_type=ctype,
                              session_type="R", abbreviation=drv, team=team, grid_position=1))
    return pd.DataFrame(rows)


class TestEntrantsAndCircuitLookup:
    def test_entrants_for_round(self):
        df = _toy_driver_rounds()
        entrants = _entrants_for_round(df, 2023, 1)
        assert set(entrants["abbreviation"]) == {"d1", "d2", "d3"}

    def test_entrants_missing_round_raises(self):
        df = _toy_driver_rounds()
        with pytest.raises(ValueError):
            _entrants_for_round(df, 2023, 99)

    def test_circuit_info_looks_up_overtaking_difficulty(self):
        df = _toy_driver_rounds()
        circuits = pd.DataFrame({
            "event_name": ["Bahrain Grand Prix"], "circuit_type": ["mixed"], "overtaking_difficulty": [4.5],
        })
        circuit_type, overtaking = _circuit_info(circuits, df, 2023, 1)
        assert circuit_type == "mixed"
        assert overtaking == 4.5

    def test_circuit_info_defaults_when_unmatched(self):
        df = _toy_driver_rounds()
        circuits = pd.DataFrame({"event_name": [], "circuit_type": [], "overtaking_difficulty": []})
        circuit_type, overtaking = _circuit_info(circuits, df, 2023, 1)
        assert circuit_type == "mixed"
        assert overtaking == 3.0
