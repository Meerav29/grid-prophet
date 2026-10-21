"""Season chaining and `gp season` (spec sec 3.4, sec 6; autopilot slice-8)."""

import numpy as np
import pandas as pd
import pytest

import sim.season
from sim.season import (
    ScheduleUnavailable, SeasonRound, SeasonTrialInputs, final_positions, remaining_calendar,
    simulate_season,
)

GP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def _inputs(n_trials=2000, n_teams=5, dnf=0.0, seed=0):
    """Synthetic posterior: two drivers per team, every trial's car identical
    at the start, so any spread in `car_path` is the random walk's own."""
    rng = np.random.default_rng(seed)
    n_drivers = 2 * n_teams
    team_names = [f"T{j}" for j in range(n_teams)]
    shape = (n_trials, n_drivers)
    return SeasonTrialInputs(
        driver_ids=[f"D{i}" for i in range(n_drivers)],
        teams=[team_names[i // 2] for i in range(n_drivers)],
        team_names=team_names,
        car=np.tile(np.linspace(0.0, 1.0, n_teams), (n_trials, 1)),
        driver_skill=rng.normal(0.0, 0.1, size=shape),
        driver_track=np.zeros((*shape, 4)),
        quali_offset=np.zeros(shape),
        race_noise_nu=np.full(n_trials, 5.0), race_noise_sigma=np.full(n_trials, 0.3),
        quali_noise_sigma=np.full(n_trials, 0.2),
        mechanical_dnf_prob=np.full(shape, dnf), incident_dnf_prob=np.full(shape, dnf),
    )


def _rounds(n, sprint_rounds=(), step=0.08):
    return [SeasonRound(round=k, has_sprint=k in sprint_rounds, circuit_type_idx=0,
                        overtaking_difficulty=3.0, step_sigma=step) for k in range(1, n + 1)]


class TestRandomWalkCarriesForward:
    def test_late_round_car_draw_is_wider_than_early(self):
        out = simulate_season(_inputs(), _rounds(12), GP, SPRINT, np.random.default_rng(1))
        early = out.car_path[:, 0].std(axis=0)
        late = out.car_path[:, -1].std(axis=0)
        # Compounding drift: sd after k steps is step * sqrt(k). Re-drawing each
        # round's step from the fitted value would leave every round at one step.
        assert np.all(late > 2.5 * early)
        np.testing.assert_allclose(late, 0.08 * np.sqrt(12), rtol=0.1)

    def test_rounds_draw_pace_from_the_wandered_car(self):
        # A huge step makes the order random by the last round even though T0's
        # fitted car starts a full second clear: proof the resolver reads the
        # running car, not `inputs.car`.
        inputs = _inputs(n_trials=500)
        calm = simulate_season(inputs, _rounds(3, step=0.0), GP, SPRINT, np.random.default_rng(2))
        wild = simulate_season(inputs, _rounds(3, step=5.0), GP, SPRINT, np.random.default_rng(2))
        assert calm.team_points[:, 0].mean() > wild.team_points[:, 0].mean() + 20


class TestPoints:
    def test_sprint_rounds_score_their_own_table(self):
        out = simulate_season(_inputs(n_trials=50), _rounds(3, sprint_rounds=(2,)), GP, SPRINT,
                              np.random.default_rng(3))
        # No DNFs: every trial hands out three GP tables and one sprint table.
        np.testing.assert_allclose(out.driver_points.sum(axis=1), 3 * 101 + 36)

    def test_sprint_goes_through_simulate_weekend(self, monkeypatch):
        calls = []
        real = sim.season.simulate_weekend

        def spy(inputs, rng, has_sprint=False, **kw):
            calls.append(has_sprint)
            return real(inputs, rng, has_sprint=has_sprint, **kw)

        monkeypatch.setattr(sim.season, "simulate_weekend", spy)
        simulate_season(_inputs(n_trials=10), _rounds(3, sprint_rounds=(3,)), GP, SPRINT,
                        np.random.default_rng(0))
        assert calls == [False, False, True]

    def test_constructor_points_are_the_sum_of_their_drivers_per_trial(self):
        inputs = _inputs(n_trials=300, dnf=0.1)
        out = simulate_season(inputs, _rounds(4, sprint_rounds=(2,)), GP, SPRINT,
                              np.random.default_rng(4))
        for j, team in enumerate(inputs.team_names):
            cols = [i for i, t in enumerate(inputs.teams) if t == team]
            np.testing.assert_array_equal(out.team_points[:, j], out.driver_points[:, cols].sum(axis=1))
            np.testing.assert_array_equal(out.team_wins[:, j], out.driver_wins[:, cols].sum(axis=1))


def test_final_positions_break_points_ties_on_wins():
    points = np.array([[100.0, 100.0, 50.0]])
    wins = np.array([[1, 3, 0]])
    np.testing.assert_array_equal(final_positions(points, wins), [[2, 1, 3]])


def _schedule(rounds=6, sprint=(3,)):
    return pd.DataFrame({
        "RoundNumber": list(range(0, rounds + 1)),
        "EventName": ["Pre-Season Testing"] + [f"Event {r}" for r in range(1, rounds + 1)],
        "EventFormat": ["testing"] + ["sprint_qualifying" if r in sprint else "conventional"
                                      for r in range(1, rounds + 1)],
    })


class TestRemainingCalendar:
    def test_rounds_after_cutoff_with_sprint_flags(self):
        cal = remaining_calendar(2026, 2, loader=lambda s: _schedule())
        assert cal["round"].tolist() == [3, 4, 5, 6]
        assert cal["has_sprint"].tolist() == [True, False, False, False]

    def test_unreadable_schedule_fails_rather_than_inventing_one(self):
        def broken(season):
            raise ConnectionError("no network")

        with pytest.raises(ScheduleUnavailable, match="will not guess"):
            remaining_calendar(2026, 2, loader=broken)


# ---------------------------------------------------------------------------
# gp season end to end, on test_cli's fake posterior
# ---------------------------------------------------------------------------

def _fit_inputs():
    from model.reliability import build_reliability_data
    from test_cli import _fake_fit, _sprint_weekend_rounds

    driver_rounds = _sprint_weekend_rounds()
    driver_rounds["points"] = 0.0
    is_race = driver_rounds["session_type"] == "R"
    driver_rounds.loc[is_race, "points"] = driver_rounds.loc[is_race, "finish_position"].map(GP).fillna(0.0)
    idata, data = _fake_fit(driver_rounds)
    reliability = build_reliability_data(driver_rounds, through_season=2024, through_round=2)
    circuits = pd.DataFrame({"event_name": ["Bahrain Grand Prix"], "circuit_type": ["mixed"],
                             "overtaking_difficulty": [4.5]})
    return idata, data, reliability, driver_rounds, circuits


def test_forecast_season_tables(monkeypatch):
    from cli import forecast_season

    idata, data, reliability, driver_rounds, circuits = _fit_inputs()
    drivers, constructors = forecast_season(idata, data, reliability, driver_rounds, circuits,
                                            n_trials=400, seed=0,
                                            schedule_loader=lambda s: _schedule(rounds=5))
    for col in ["driver", "p_title", "exp_final_points", "p10_final_position",
                "p50_final_position", "p90_final_position", "p_top3"]:
        assert col in drivers.columns
    for col in ["team", "p_title", "exp_final_points", "exp_share", "p10_final_position",
                "p50_final_position", "p90_final_position"]:
        assert col in constructors.columns

    # current points: d1 won both rounds already run
    current = drivers.set_index("driver")["current_points"]
    assert current["d1"] == 50.0 and current["d2"] == 36.0
    assert (drivers["exp_final_points"] >= drivers["current_points"]).all()
    assert drivers["p_title"].sum() == pytest.approx(1.0)
    assert constructors["exp_share"].sum() == pytest.approx(1.0)
    # same trials: constructor expectations are their drivers' expectations summed
    by_team = drivers.groupby("team")["exp_final_points"].sum()
    for team, exp in constructors.set_index("team")["exp_final_points"].items():
        assert exp == pytest.approx(by_team[team])


def test_season_finished_means_standings_are_final():
    from cli import forecast_season

    idata, data, reliability, driver_rounds, circuits = _fit_inputs()
    drivers, _ = forecast_season(idata, data, reliability, driver_rounds, circuits, n_trials=50,
                                 schedule_loader=lambda s: _schedule(rounds=2))
    leader = drivers.iloc[0]
    assert leader["driver"] == "d1" and leader["p_title"] == 1.0
    np.testing.assert_allclose(drivers["exp_final_points"], drivers["current_points"])


class TestCmdSeason:
    def _args(self, tmp_path):
        import argparse

        return argparse.Namespace(fit_dir=str(tmp_path), out_dir=str(tmp_path), n_trials=50, seed=0)

    def _patch(self, monkeypatch, schedule):
        import cli

        idata, data, reliability, driver_rounds, circuits = _fit_inputs()
        monkeypatch.setattr(cli, "load_fit", lambda d: (idata, data, reliability, None))
        monkeypatch.setattr(cli, "_load_driver_rounds", lambda: driver_rounds)
        monkeypatch.setattr(cli, "_load_circuits", lambda: circuits)
        monkeypatch.setattr(sim.season, "_fastf1_schedule", schedule)

    def test_writes_both_championship_csvs(self, monkeypatch, tmp_path):
        import cli

        self._patch(monkeypatch, lambda s: _schedule(rounds=4))
        cli.cmd_season(self._args(tmp_path))
        assert pd.read_csv(tmp_path / "drivers_championship.csv").shape[0] == 4
        assert pd.read_csv(tmp_path / "constructors_championship.csv").shape[0] == 2

    def test_unreadable_schedule_exits_with_a_message(self, monkeypatch, tmp_path, capsys):
        import cli

        def broken(season):
            raise ConnectionError("rate limited")

        self._patch(monkeypatch, broken)
        with pytest.raises(SystemExit):
            cli.cmd_season(self._args(tmp_path))
        assert "rate limited" in capsys.readouterr().err
        assert not (tmp_path / "drivers_championship.csv").exists()

