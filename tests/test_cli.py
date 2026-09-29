"""Tests for the deterministic CLI plumbing in src/cli.py (spec sec 6):
argument parsing and the lookup helpers that go from (season, round) to
entrants / circuit metadata. The `fit`/`race`/`backtest` command bodies
themselves invoke the probabilistic model and are covered by the backtest
run, not by unit tests here (see model/pace.py module docstring)."""

import numpy as np
import pandas as pd
import pytest

from cli import (
    build_parser, _parse_through, _entrants_for_round, _circuit_info, _hazard_grid,
    _real_grid_for_round, _round_has_sprint,
)


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


def _toy_driver_rounds(grid_positions=(1, 2, 3)):
    rows = []
    for season_round, event, ctype in [((2023, 1), "Bahrain Grand Prix", "mixed")]:
        season, rnd = season_round
        for (team, drv), gp in zip([("Team A", "d1"), ("Team A", "d2"), ("Team B", "d3")],
                                    grid_positions):
            rows.append(dict(season=season, round=rnd, event_name=event, circuit_type=ctype,
                              session_type="R", abbreviation=drv, team=team, grid_position=gp))
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


class TestRealGridLookup:
    """Spec sec 3.3 step 3 / sec 6: post-quali mode reads the round's real
    starting grid rather than simulating one."""

    def test_grid_is_aligned_to_the_requested_driver_order(self):
        df = _toy_driver_rounds(grid_positions=(3, 1, 2))
        grid = _real_grid_for_round(df, 2023, 1, ["d1", "d2", "d3"])
        assert list(grid) == [3, 1, 2]
        # a different entrant ordering gets the same grid, permuted with it
        grid = _real_grid_for_round(df, 2023, 1, ["d3", "d1", "d2"])
        assert list(grid) == [2, 3, 1]

    def test_non_contiguous_grid_numbers_are_densely_ranked(self):
        """Penalties and non-starters leave gaps; positions still have to come
        out as a clean 1..n permutation for the resolver."""
        df = _toy_driver_rounds(grid_positions=(5, 12, 9))
        assert list(_real_grid_for_round(df, 2023, 1, ["d1", "d2", "d3"])) == [1, 3, 2]

    @pytest.mark.parametrize("bad", [0, np.nan])  # FastF1 writes 0 for a pit-lane start
    def test_pit_lane_and_missing_starts_rank_last_not_to_pole(self, bad):
        df = _toy_driver_rounds(grid_positions=(bad, 1, 2))
        assert _real_grid_for_round(df, 2023, 1, ["d1", "d2", "d3"])[0] == 3

    def test_round_without_a_usable_grid_raises_rather_than_silently_simulating(self):
        for df, rnd in [(_toy_driver_rounds(grid_positions=(np.nan,) * 3), 1),
                         (_toy_driver_rounds(), 99)]:
            with pytest.raises(ValueError):
                _real_grid_for_round(df, 2023, rnd, ["d1", "d2", "d3"])


def _toy_sprint_weekend(race_grid=(1, 2, 3), sprint_grid=(3, 2, 1)):
    """A sprint round as collect_v2 writes it: Q rows, an S block, an R
    block, with the two races starting from their own grids (which from 2023
    are set by separate sessions and routinely differ)."""
    rows = []
    for session, grids in [("S", sprint_grid), ("R", race_grid)]:
        for (team, drv), gp in zip([("Team A", "d1"), ("Team A", "d2"), ("Team B", "d3")], grids):
            rows.append(dict(season=2023, round=4, event_name="Azerbaijan Grand Prix",
                              circuit_type="street", session_type=session,
                              abbreviation=drv, team=team, grid_position=gp))
    return pd.DataFrame(rows)


class TestSprintRoundDetection:
    """Spec sec 3.4: the sprint is an extra event inside the round, and
    whether a round has one is read off the collected data."""

    def test_round_with_sprint_rows_is_a_sprint_weekend(self):
        assert _round_has_sprint(_toy_sprint_weekend(), 2023, 4) is True

    def test_round_without_sprint_rows_is_not(self):
        assert _round_has_sprint(_toy_driver_rounds(), 2023, 1) is False

    def test_missing_round_is_not_a_sprint_weekend(self):
        assert _round_has_sprint(_toy_sprint_weekend(), 2023, 99) is False


class TestSprintGridLookup:
    def test_sprint_grid_is_read_from_the_sprint_rows_not_the_race_rows(self):
        df = _toy_sprint_weekend(race_grid=(1, 2, 3), sprint_grid=(3, 2, 1))
        drivers = ["d1", "d2", "d3"]
        assert list(_real_grid_for_round(df, 2023, 4, drivers, session_type="S")) == [3, 2, 1]
        assert list(_real_grid_for_round(df, 2023, 4, drivers)) == [1, 2, 3]

    def test_non_sprint_round_has_no_sprint_grid_to_read(self):
        with pytest.raises(ValueError):
            _real_grid_for_round(_toy_driver_rounds(), 2023, 1, ["d1", "d2", "d3"], session_type="S")


class TestHazardGrid:
    """The grid slot sec 3.2's incident hazard is conditioned on. Pre-weekend
    there is no grid yet, so the posterior-mean quali pace has to stand in --
    and the rank direction is the easy thing to invert (quali pace is a gap to
    the session best, so *lower* is further forward)."""

    def test_real_grid_is_used_verbatim_when_given(self):
        assert list(_hazard_grid(np.array([3, 1, 2]), np.zeros((5, 3)))) == [3.0, 1.0, 2.0]

    def test_pre_weekend_grid_ranks_the_fastest_car_to_pole(self):
        quali_pace = np.array([[0.9, 0.0, 0.4], [1.1, 0.0, 0.6]])
        assert list(_hazard_grid(None, quali_pace)) == [3.0, 1.0, 2.0]
        assert sorted(_hazard_grid(None, np.array([[0.3, 0.1, 0.2, 0.4]]))) == [1.0, 2.0, 3.0, 4.0]
