"""Unit tests for post-quali conditioning (spec sec 6).

The sampling itself is not exercised here -- `run_conditioning` takes the
warm-started fit, the fallback fit and the clock as arguments precisely so
the fallback protocol can be tested deterministically, without PyMC. Same
strategy the rest of the v2 model uses: deterministic pieces unit-tested,
the sampler validated through the backtest (see model/pace.py's docstring).
"""

import types

import numpy as np
import pandas as pd
import pytest

from model.conditioning import (
    EARLY_SEASON_ROUNDS, FALLBACK_FULL_REFIT, RELAXED_TARGET_S, WARM_START,
    WARM_START_TARGET_S, append_record, fallback_rate, read_log, regressions,
    run_conditioning, warm_start_initvals,
)
from model.pace import CIRCUIT_TYPES, build_pace_data


def _FakeClock(ticks):
    """A clock returning the times it is given, one per call, so a fit's
    'duration' is whatever the test says it is."""
    remaining = iter(list(ticks))
    last = [0.0]

    def tick():
        last[0] = next(remaining, last[0])
        return last[0]

    return tick


def _run(round_, ticks):
    """Run the protocol with fits that record which of them was called."""
    calls = []
    idata, record = run_conditioning(
        2026, round_,
        lambda: (calls.append("warm"), "warm-idata")[1],
        lambda: (calls.append("full"), "full-idata")[1],
        clock=_FakeClock(ticks))
    return idata, record, calls


FAST = [0.0, 10.0]                      # warm start well inside the 120s target
SLOW = [0.0, 150.0, 150.0, 280.0]       # warm 150s over; 280s total, inside 300s but not 120s
VERY_SLOW = [0.0, 150.0, 150.0, 900.0]  # 900s total, past even the relaxed allowance


class TestConditioningProtocol:
    def test_warm_start_within_target_does_not_refit(self):
        idata, record, calls = _run(12, FAST)
        assert calls == ["warm"]  # the fallback fit never ran
        assert idata == "warm-idata"
        assert record.path == WARM_START
        assert record.fell_back is False
        assert record.regression is False
        assert record.total_s == pytest.approx(10.0)

    def test_slow_warm_start_falls_back_to_the_full_refit(self):
        """Sec 6: a slow round "falls back rather than blocking the run", and
        the fallback -- refit on the same quali observation -- is the posterior
        that then gets used."""
        idata, record, calls = _run(12, SLOW)
        assert calls == ["warm", "full"]
        assert record.path == FALLBACK_FULL_REFIT
        assert record.fell_back is True
        assert idata == "full-idata"


class TestRelaxedTargetAppliesEarlySeasonOnly:
    """Sec 6: the relaxed target is an early-season carve-out, so the same slow
    round is an accepted exception at round 2 and a regression at round 18."""

    @pytest.mark.parametrize("round_, ticks, target, regression", [
        (2, SLOW, RELAXED_TARGET_S, False),      # early season, inside the relaxed allowance
        (18, SLOW, WARM_START_TARGET_S, True),   # same timing, late season: a real regression
        (2, VERY_SLOW, RELAXED_TARGET_S, True),  # early season, past even the relaxed allowance
    ])
    def test_regression_depends_on_where_in_the_season_the_round_is(
            self, round_, ticks, target, regression):
        _, record, _ = _run(round_, ticks)
        assert record.early_season is (round_ <= EARLY_SEASON_ROUNDS)
        assert record.fell_back is True
        assert record.applied_target_s == target
        assert record.regression is regression


class TestConditioningLog:
    """Sec 6: fallbacks are "recorded per round" and countable, because a high
    fallback rate is a signal, not something to absorb silently."""

    def _log(self, tmp_path):
        """Three rounds written one at a time -- each `gp race --mode
        post-quali` is a separate process, so the log has to accumulate rather
        than overwrite the previous round."""
        path = str(tmp_path / "conditioning_log.csv")
        for rnd, ticks in [(1, FAST), (2, SLOW), (18, SLOW)]:
            append_record(path, _run(rnd, ticks)[1])
        return read_log(path)

    def test_records_counts_and_isolates_regressions(self, tmp_path):
        log = self._log(tmp_path)
        assert list(log["round"]) == [1, 2, 18]
        assert list(log["fell_back"]) == [False, True, True]
        assert fallback_rate(log) == pytest.approx(2 / 3)
        assert list(regressions(log)["round"]) == [18]
        assert fallback_rate(log.iloc[:0]) == 0.0


def _fake_data(n_rounds=5):
    return types.SimpleNamespace(n_teams=2, n_drivers=4, n_rounds=n_rounds,
                                  n_circuit_types=len(CIRCUIT_TYPES))


def _fake_idata(n_rounds=5, chains=2, draws=3):
    import arviz as az

    rng = np.random.default_rng(0)
    shapes = {"car0": (2,), "car_step_raw": (2, n_rounds - 1), "driver_skill": (4,),
               "quali_offset": (4,), "driver_track": (4, len(CIRCUIT_TYPES)),
               "race_nu": (), "race_sigma": (), "quali_sigma": ()}
    # positive draws throughout: race_nu and the sigmas are constrained
    return az.from_dict(posterior={name: np.abs(rng.normal(size=(chains, draws) + shape)) + 0.5
                                    for name, shape in shapes.items()})


class TestWarmStartInitvals:
    def test_shapes_and_values_come_from_the_prior_posterior(self):
        idata = _fake_idata()
        initvals = warm_start_initvals(idata, _fake_data())
        assert initvals["car_step_raw"].shape == (2, 4)
        assert initvals["driver_track"].shape == (4, len(CIRCUIT_TYPES))
        assert np.isscalar(initvals["race_sigma"])
        expected = idata.posterior["driver_skill"].mean(dim=("chain", "draw")).to_numpy()
        assert np.allclose(initvals["driver_skill"], expected)

    def test_new_round_pads_the_random_walk_with_a_zero_step(self):
        """Conditioning on a round the pre-weekend fit had never seen grows the
        random walk by one step; the new step warm-starts at zero."""
        idata = _fake_idata(n_rounds=5)
        steps = warm_start_initvals(idata, _fake_data(n_rounds=6))["car_step_raw"]
        prior_mean = idata.posterior["car_step_raw"].mean(dim=("chain", "draw")).to_numpy()
        assert steps.shape == (2, 5)
        assert np.allclose(steps[:, -1], 0.0)
        assert np.allclose(steps[:, :4], prior_mean)


def _toy_driver_rounds():
    rows = []
    for rnd in range(1, 4):
        for team, drv in [("Team A", "d1"), ("Team A", "d2"), ("Team B", "d3"), ("Team B", "d4")]:
            common = dict(season=2026, round=rnd, team=team, abbreviation=drv, circuit_type="mixed")
            rows.append(dict(common, session_type="R", median_clean_air_lap_s=90.0,
                              gap_to_winner_median_clean_air_s=0.5))
            rows.append(dict(common, session_type="Q", median_clean_air_lap_s=np.nan,
                              gap_to_winner_median_clean_air_s=np.nan, gap_to_best_s=0.3))
    return pd.DataFrame(rows)


class TestHoldOutRaceRound:
    """Post-quali conditioning adds the round's quali as an observation while
    the race it is forecasting stays out of the fit."""

    def test_quali_kept_and_race_dropped_for_the_held_out_round(self):
        data = build_pace_data(_toy_driver_rounds(), through_season=2026, through_round=3,
                                hold_out_race_round=(2026, 3))
        target = data.round_to_idx[(2026, 3)]
        assert (data.quali_round_idx == target).sum() == 4   # all four cars' quali
        assert (data.race_round_idx == target).sum() == 0    # no race rows
        assert (data.race_round_idx == data.round_to_idx[(2026, 2)]).sum() == 4
        # and the round keeps its place on the random-walk axis, so there is
        # still a `car[team, round_idx]` to forecast from
        assert target == data.n_rounds - 1

    def test_without_hold_out_the_race_rows_are_included(self):
        data = build_pace_data(_toy_driver_rounds(), through_season=2026, through_round=3)
        assert (data.race_round_idx == data.round_to_idx[(2026, 3)]).sum() == 4
