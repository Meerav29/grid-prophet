"""Unit tests for the deterministic data-prep in model.pace (spec sec 3.1)."""

import numpy as np
import pandas as pd
import pytest

from model.pace import (
    build_pace_data, _build_round_index, _step_sigma_scale, _rookie_mask,
    RULE_CHANGE_SEASONS, SEASON_BOUNDARY_SCALE, RULE_CHANGE_BOUNDARY_SCALE,
    RULE_CHANGE_EARLY_SCALE, RULE_CHANGE_EARLY_ROUNDS,
)


def _toy_driver_rounds():
    rows = []
    for season in [2021, 2022, 2023]:
        for rnd in range(1, 4):
            for team, drivers in [("Team A", ["d1", "d2"]), ("Team B", ["d3", "d4"])]:
                for drv in drivers:
                    rows.append(dict(
                        season=season, round=rnd, team=team, abbreviation=drv,
                        circuit_type="mixed", session_type="R",
                        median_clean_air_lap_s=90.0, gap_to_winner_median_clean_air_s=0.5,
                    ))
                    rows.append(dict(
                        season=season, round=rnd, team=team, abbreviation=drv,
                        circuit_type="mixed", session_type="Q",
                        median_clean_air_lap_s=np.nan, gap_to_winner_median_clean_air_s=np.nan,
                        gap_to_best_s=0.3,
                    ))
    return pd.DataFrame(rows)


class TestRoundIndex:
    def test_chronological_ordering(self):
        df = _toy_driver_rounds()
        round_index, round_to_idx = _build_round_index(df)
        # sorted by (season, round)
        assert list(round_index["season"]) == sorted(round_index["season"])
        assert round_to_idx[(2021, 1)] == 0
        assert round_to_idx[(2023, 3)] == len(round_index) - 1

    def test_round_idx_is_contiguous(self):
        df = _toy_driver_rounds()
        round_index, _ = _build_round_index(df)
        assert list(round_index["round_idx"]) == list(range(len(round_index)))


class TestStepSigmaScale:
    def test_first_round_unused_default_one(self):
        df = _toy_driver_rounds()
        round_index, _ = _build_round_index(df)
        scale = _step_sigma_scale(round_index)
        assert scale[0] == 1.0

    def test_normal_season_boundary_scaled_up(self):
        # build toy data spanning a non-rule-change boundary (2018 -> 2019)
        rows = []
        for season in [2018, 2019]:
            for rnd in range(1, 4):
                rows.append(dict(season=season, round=rnd, team="Team A", abbreviation="d1"))
        df = pd.DataFrame(rows)
        round_index, round_to_idx = _build_round_index(df)
        scale = _step_sigma_scale(round_index)
        assert 2019 not in RULE_CHANGE_SEASONS
        boundary_idx = round_to_idx[(2019, 1)]
        assert scale[boundary_idx] == SEASON_BOUNDARY_SCALE
        # within-season steps stay at baseline
        mid_season_idx = round_to_idx[(2019, 2)]
        assert scale[mid_season_idx] == 1.0

    def test_rule_change_boundary_scaled_up_more(self):
        df = _toy_driver_rounds()
        round_index, round_to_idx = _build_round_index(df)
        scale = _step_sigma_scale(round_index)
        boundary_idx = round_to_idx[(2023, 1)] if 2023 in RULE_CHANGE_SEASONS else None
        # 2023 isn't a rule-change season in this repo's constant, so synthesize one
        assert 2022 in RULE_CHANGE_SEASONS
        idx_2022 = round_to_idx[(2022, 1)]
        assert scale[idx_2022] == RULE_CHANGE_BOUNDARY_SCALE
        assert RULE_CHANGE_BOUNDARY_SCALE > SEASON_BOUNDARY_SCALE

    def test_early_rounds_of_rule_change_season_elevated(self):
        df = _toy_driver_rounds()
        round_index, round_to_idx = _build_round_index(df)
        scale = _step_sigma_scale(round_index)
        # round 2 of 2022 (a rule-change season) should be elevated above baseline
        idx = round_to_idx[(2022, 2)]
        assert scale[idx] == RULE_CHANGE_EARLY_SCALE


class TestRookieMask:
    def test_debut_at_or_after_cutoff_is_rookie(self):
        driver_names = ["veteran", "rookie2023", "unknown"]
        meta = pd.DataFrame({"abbreviation": ["veteran", "rookie2023"], "debut_season": [2015, 2023]})
        mask = _rookie_mask(driver_names, meta, through_season=2023)
        assert mask[0] == False   # noqa: E712 -- debuted long before cutoff
        assert mask[1] == True    # noqa: E712 -- debut season == cutoff season
        assert mask[2] == True    # noqa: E712 -- unknown driver treated conservatively as rookie

    def test_no_meta_defaults_to_no_rookies(self):
        mask = _rookie_mask(["a", "b"], None, through_season=2023)
        assert not mask.any()


class TestBuildPaceData:
    def test_shapes_and_vocab(self):
        df = _toy_driver_rounds()
        data = build_pace_data(df, through_season=2023, through_round=3)
        assert data.n_teams == 2
        assert data.n_drivers == 4
        assert len(data.race_y) == len(data.race_team_idx) == len(data.race_driver_idx)
        assert len(data.quali_y) == len(data.quali_team_idx)
        # 3 seasons x 3 rounds x 2 teams x 2 drivers = 36 race rows
        assert len(data.race_y) == 36

    def test_cutoff_restricts_observations_but_not_full_history(self):
        df = _toy_driver_rounds()
        data_early = build_pace_data(df, through_season=2021, through_round=2)
        data_late = build_pace_data(df, through_season=2023, through_round=3)
        assert len(data_early.race_y) < len(data_late.race_y)
        # team/driver vocab stable regardless of cutoff
        assert data_early.team_names == data_late.team_names
        assert data_early.driver_names == data_late.driver_names

    def test_round_axis_trimmed_to_cutoff(self):
        """The random-walk round axis shouldn't pay for rounds beyond the
        cutoff even though they exist elsewhere in driver_rounds.csv (compute
        budget, see model.pace.build_pace_data docstring)."""
        df = _toy_driver_rounds()
        data_early = build_pace_data(df, through_season=2021, through_round=2)
        data_full = build_pace_data(df, through_season=None)
        assert data_early.n_rounds < data_full.n_rounds
        assert data_early.n_rounds == 2  # 2021 round 1, round 2

    def test_drops_rows_missing_required_fields(self):
        df = _toy_driver_rounds()
        df.loc[df.index[0], "gap_to_winner_median_clean_air_s"] = np.nan
        data = build_pace_data(df, through_season=2023, through_round=3)
        assert len(data.race_y) == 35

    def test_no_cutoff_includes_everything(self):
        df = _toy_driver_rounds()
        data = build_pace_data(df)
        assert len(data.race_y) == 36
