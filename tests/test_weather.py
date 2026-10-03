"""Wet/dry weather flag and the wet race-noise scale (spec sec 5)."""

import numpy as np
import pandas as pd
import pytest

from model.pace import (
    WET_SIGMA_RATIO_PRIOR, WET_SIGMA_RATIO_SD, _has_wet_races, build_model, build_pace_data,
)
from model.weather import (
    observation_sigma_scale, weather_coverage, wet_flags, wet_state,
)

# The free variables the pace model had before this slice. A dry-only fit
# window must still build exactly these and nothing else.
PRE_SLICE_FREE_RVS = {
    "car0", "car_step_raw", "driver_skill", "quali_offset", "driver_track",
    "race_nu", "race_sigma", "quali_sigma",
}


def _driver_rounds(wet_rounds=(), with_column=True, seasons=(2023,), rounds=(1, 2, 3)):
    """Toy frame in the shape of driver_rounds.csv, one wet flag per round."""
    rows = []
    for season in seasons:
        for rnd in rounds:
            is_wet = (season, rnd) in wet_rounds
            for team, drivers in [("Team A", ["d1", "d2"]), ("Team B", ["d3", "d4"])]:
                for drv in drivers:
                    base = dict(
                        season=season, round=rnd, team=team, abbreviation=drv,
                        circuit_type="mixed",
                    )
                    race = dict(base, session_type="R", median_clean_air_lap_s=90.0,
                                gap_to_winner_median_clean_air_s=0.5)
                    quali = dict(base, session_type="Q", median_clean_air_lap_s=np.nan,
                                 gap_to_winner_median_clean_air_s=np.nan, gap_to_best_s=0.3)
                    if with_column:
                        race["is_wet"] = is_wet
                        quali["is_wet"] = is_wet
                    rows.append(race)
                    rows.append(quali)
    return pd.DataFrame(rows)


class TestWetState:
    def test_missing_column_is_unknown_not_an_error(self):
        frame = _driver_rounds(with_column=False)
        assert list(wet_state(frame).unique()) == [None]

    def test_missing_column_degrades_to_dry_for_the_model(self):
        frame = _driver_rounds(with_column=False)
        assert not wet_flags(frame).any()
        assert wet_flags(frame).shape == (len(frame),)

    def test_blank_cell_is_unknown_and_treated_as_dry(self):
        frame = pd.DataFrame({"is_wet": [True, False, np.nan, None]})
        assert list(wet_state(frame)) == [True, False, None, None]
        assert list(wet_flags(frame)) == [True, False, False, False]

    @pytest.mark.parametrize("raw,expected", [
        ("True", True), ("true", True), ("WET", True), (1, True), (1.0, True),
        ("False", False), ("no", False), (0, False),
        ("", None), ("probably", None),
    ])
    def test_token_coercion(self, raw, expected):
        assert wet_state(pd.DataFrame({"is_wet": [raw]})).iloc[0] is expected

    def test_coverage_counts_the_three_states_separately(self):
        frame = pd.DataFrame({"is_wet": [True, True, False, np.nan]})
        assert weather_coverage(frame) == {"wet": 2, "dry": 1, "unknown": 1}


class TestObservationSigmaScale:
    def test_dry_rows_keep_the_dry_scale_and_wet_rows_do_not(self):
        scale = observation_sigma_scale([True, False, True], 2.5)
        assert list(scale) == [2.5, 1.0, 2.5]

    def test_ratio_of_one_is_the_pre_slice_single_scale(self):
        scale = observation_sigma_scale([True, False], 1.0)
        assert list(scale) == [1.0, 1.0]

    def test_empty_is_empty(self):
        assert observation_sigma_scale([], 2.0).shape == (0,)


class TestPaceDataCarriesTheFlag:
    def test_flag_is_aligned_with_the_surviving_race_rows(self):
        frame = _driver_rounds(wet_rounds={(2023, 2)})
        data = build_pace_data(frame)
        assert data.race_is_wet.shape == data.race_y.shape
        # 3 rounds x 4 drivers; round 2's four rows are the wet ones.
        assert data.race_is_wet.sum() == 4
        wet_round_idx = data.round_to_idx[(2023, 2)]
        assert set(data.race_round_idx[data.race_is_wet]) == {wet_round_idx}

    def test_flag_survives_the_dropna_that_filters_race_rows(self):
        """The flag is read off the *filtered* frame, so a dropped row must
        not shift every later row's weather by one."""
        frame = _driver_rounds(wet_rounds={(2023, 3)})
        race_mask = frame["session_type"] == "R"
        first_race = frame[race_mask].index[0]
        frame.loc[first_race, "gap_to_winner_median_clean_air_s"] = np.nan
        data = build_pace_data(frame)
        assert len(data.race_y) == 11            # 12 race rows less the dropped one
        assert data.race_is_wet.sum() == 4       # still exactly round 3's four rows
        assert set(data.race_round_idx[data.race_is_wet]) == {data.round_to_idx[(2023, 3)]}

    def test_frame_without_the_column_is_all_dry(self):
        data = build_pace_data(_driver_rounds(with_column=False))
        assert data.race_is_wet.shape == data.race_y.shape
        assert not data.race_is_wet.any()


class TestHasWetRaces:
    def test_none_flag_from_a_pre_slice_pacedata_is_not_wet(self):
        data = build_pace_data(_driver_rounds())
        data.race_is_wet = None
        assert _has_wet_races(data) is False

    def test_all_dry_window_is_not_wet(self):
        assert _has_wet_races(build_pace_data(_driver_rounds())) is False

    def test_one_wet_round_is_enough(self):
        assert _has_wet_races(build_pace_data(_driver_rounds(wet_rounds={(2023, 1)}))) is True


class TestWetNoiseScaleInTheModel:
    def test_dry_only_window_builds_the_pre_slice_graph_exactly(self):
        pytest.importorskip("pymc")
        model = build_model(build_pace_data(_driver_rounds()))
        assert {rv.name for rv in model.free_RVs} == PRE_SLICE_FREE_RVS
        assert "race_sigma_wet" not in {v.name for v in model.deterministics}

    def test_window_without_the_column_also_builds_the_pre_slice_graph(self):
        pytest.importorskip("pymc")
        model = build_model(build_pace_data(_driver_rounds(with_column=False)))
        assert {rv.name for rv in model.free_RVs} == PRE_SLICE_FREE_RVS

    def test_wet_window_adds_a_second_scale(self):
        pytest.importorskip("pymc")
        model = build_model(build_pace_data(_driver_rounds(wet_rounds={(2023, 2)})))
        assert {rv.name for rv in model.free_RVs} == PRE_SLICE_FREE_RVS | {"race_sigma_wet_ratio"}
        assert "race_sigma_wet" in {v.name for v in model.deterministics}

    def test_the_two_scales_are_different_numbers(self):
        """`race_sigma_wet` is the dry scale times the ratio, so the wet scale
        tracks the ratio and is not pinned to the dry one."""
        pytest.importorskip("pymc")
        import pytensor

        model = build_model(build_pace_data(_driver_rounds(wet_rounds={(2023, 2)})))
        outs = model.replace_rvs_by_values([model["race_sigma"], model["race_sigma_wet"]])
        by_name = {v.name: v for v in model.value_vars}
        fn = pytensor.function(
            [by_name["race_sigma_log__"], by_name["race_sigma_wet_ratio_log__"]], outs,
        )
        dry_val, wet_val = fn(np.log(0.4), np.log(2.5))

        assert float(dry_val) == pytest.approx(0.4)
        assert float(wet_val) == pytest.approx(0.4 * 2.5)
        assert float(wet_val) != float(dry_val)

    def test_only_wet_observations_move_when_the_wet_ratio_moves(self):
        """The sharp version of "wet rounds get their own scale": changing the
        wet ratio must change the density of the wet race rows and leave every
        dry row's density untouched."""
        pytest.importorskip("pymc")

        data = build_pace_data(_driver_rounds(wet_rounds={(2023, 2)}))
        model = build_model(data)
        elemwise = model.compile_logp(vars=[model["race_obs"]], sum=False)
        point = model.initial_point()

        point["race_sigma_wet_ratio_log__"] = np.array(np.log(1.0))
        at_one = np.asarray(elemwise(point)[0]).copy()
        point["race_sigma_wet_ratio_log__"] = np.array(np.log(3.0))
        at_three = np.asarray(elemwise(point)[0]).copy()

        wet = data.race_is_wet
        assert wet.any() and (~wet).any()
        assert np.allclose(at_one[~wet], at_three[~wet])
        assert not np.any(np.isclose(at_one[wet], at_three[wet]))

    def test_ratio_prior_is_centred_above_one_but_admits_a_tidier_wet_race(self):
        assert WET_SIGMA_RATIO_PRIOR > 1.0
        # 5th percentile of the LogNormal ratio prior sits below 1.0, so the
        # prior leans wet-is-noisier without asserting it.
        assert np.exp(np.log(WET_SIGMA_RATIO_PRIOR) - 1.645 * WET_SIGMA_RATIO_SD) < 1.0


class _StubSession:
    """Stands in for a FastF1 session in `_weather_summary`."""

    def __init__(self, weather_data):
        self._weather_data = weather_data

    @property
    def weather_data(self):
        if isinstance(self._weather_data, Exception):
            raise self._weather_data
        return self._weather_data


class TestCollectorRecordsUnknownWeather:
    """Criterion 3: do not fabricate weather for rounds with no data."""

    @staticmethod
    def _summary(weather_data):
        from collect_v2 import _weather_summary
        return _weather_summary(_StubSession(weather_data))

    def test_rainfall_anywhere_in_the_session_is_wet(self):
        wx = pd.DataFrame({"Rainfall": [False, True, False], "AirTemp": [20.0, 20.0, 20.0]})
        assert self._summary(wx) == (True, 20.0)

    def test_no_rainfall_is_dry(self):
        wx = pd.DataFrame({"Rainfall": [False, False], "AirTemp": [22.0, 24.0]})
        assert self._summary(wx) == (False, 23.0)

    def test_empty_weather_frame_is_unknown_not_dry(self):
        assert self._summary(pd.DataFrame()) == (None, None)

    def test_missing_weather_is_unknown_not_dry(self):
        assert self._summary(None) == (None, None)

    def test_all_nan_rainfall_is_unknown_but_keeps_the_temperature(self):
        wx = pd.DataFrame({"Rainfall": [np.nan, np.nan], "AirTemp": [19.0, 21.0]})
        is_wet, air_temp = self._summary(wx)
        assert is_wet is None
        assert air_temp == pytest.approx(20.0)

    def test_a_failing_session_is_unknown_not_dry(self):
        assert self._summary(RuntimeError("no weather for this session")) == (None, None)


class TestValidatorWeatherSection:
    """Criterion 1: the validator carries the flag, and says when it is absent."""

    @staticmethod
    def _section(frame):
        from validate_driver_rounds import weather_section
        return "\n".join(weather_section(frame))

    def test_counts_wet_dry_and_unrecorded_rows(self):
        frame = _driver_rounds(wet_rounds={(2023, 1)})
        frame["is_wet"] = frame["is_wet"].astype(object)
        race_rows = frame.index[frame["session_type"] == "R"]
        frame.loc[race_rows[:2], "is_wet"] = np.nan   # two rounds-1 race rows unmeasured
        text = self._section(frame)
        assert "## Weather coverage (spec sec 5)" in text
        assert "wet 2, dry 8, not recorded 2" in text
        assert "Rounds with at least one wet race/sprint session: 1" in text
        assert "2023:1" in text

    def test_a_file_without_the_column_reports_it_instead_of_raising(self):
        text = self._section(_driver_rounds(with_column=False))
        assert "not recorded 12" in text          # 3 rounds x 4 drivers of race rows
        assert "Rounds with at least one wet race/sprint session: 0" in text
        assert "(none)" in text


class TestWeatherSummaryEdges:
    @staticmethod
    def _summary(weather_data):
        from collect_v2 import _weather_summary
        return _weather_summary(_StubSession(weather_data))

    def test_rainfall_without_a_temperature_column(self):
        assert self._summary(pd.DataFrame({"Rainfall": [False, True]})) == (True, None)

    def test_temperature_without_a_rainfall_column_is_unknown(self):
        assert self._summary(pd.DataFrame({"AirTemp": [20.0, 22.0]})) == (None, 21.0)

    def test_all_nan_temperature_is_none_not_nan(self):
        wx = pd.DataFrame({"Rainfall": [False, False], "AirTemp": [np.nan, np.nan]})
        assert self._summary(wx) == (False, None)

    def test_partly_measured_rainfall_uses_the_measured_values(self):
        wx = pd.DataFrame({"Rainfall": [np.nan, True], "AirTemp": [20.0, 20.0]})
        assert self._summary(wx)[0] is True

    def test_a_failing_session_logs_a_warning(self, caplog):
        with caplog.at_level("WARNING"):
            self._summary(RuntimeError("boom"))
        assert "weather unavailable" in caplog.text


class TestUnknownIsVisible:
    def test_csv_round_trip_keeps_unknown_distinct_from_dry(self, tmp_path):
        frame = pd.DataFrame({"is_wet": [True, None, False]})
        path = tmp_path / "rounds.csv"
        frame.to_csv(path, index=False)
        assert list(wet_state(pd.read_csv(path))) == [True, None, False]

    def test_unrecognised_values_warn_but_blanks_do_not(self, caplog):
        with caplog.at_level("WARNING"):
            wet_state(pd.DataFrame({"is_wet": [True, np.nan, False]}))
        assert caplog.text == ""
        with caplog.at_level("WARNING"):
            states = wet_state(pd.DataFrame({"is_wet": ["rain", "True"]}))
        assert list(states) == [None, True]
        assert "unrecognised" in caplog.text

    def test_unrecorded_race_weather_warns_at_fit_time(self, caplog):
        frame = _driver_rounds(wet_rounds={(2023, 1)})
        frame["is_wet"] = frame["is_wet"].astype(object)
        frame.loc[frame.index[frame["session_type"] == "R"][:1], "is_wet"] = np.nan
        with caplog.at_level("WARNING"):
            build_pace_data(frame)
        assert "weather not recorded" in caplog.text

    def test_mismatched_wet_flags_raise_instead_of_going_dry(self):
        data = build_pace_data(_driver_rounds(wet_rounds={(2023, 1)}))
        data.race_is_wet = data.race_is_wet[:-1]
        with pytest.raises(ValueError, match="race_is_wet"):
            _has_wet_races(data)
