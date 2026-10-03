"""Tests for the deterministic CLI plumbing in src/cli.py (spec sec 6):
argument parsing and the lookup helpers that go from (season, round) to
entrants / circuit metadata. The `fit`/`race`/`backtest` command bodies
themselves invoke the probabilistic model and are covered by the backtest
run, not by unit tests here (see model/pace.py module docstring)."""

import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cli import (
    build_parser, _parse_through, _entrants_for_round, _circuit_info, _hazard_grid,
    _real_grid_for_round, _round_has_sprint, forecast_weekend,
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


# --- slice-6: the sprint path is wired, not just implemented ---------------
#
# `sim.race.sprint_inputs_from` can only price sec 3.2's first-lap spike if
# `forecast_weekend` hands it both reliability channels and not just their
# fold. That wiring is two keyword arguments deep inside a function the rest of
# this file does not exercise, so without the tests below it could be deleted
# with the whole suite still green -- which is exactly the regression gap PR
# #10's review flagged on slice-4's own wiring. These are the cheapest fixtures
# that run `forecast_weekend` end to end.

DRIVERS = ["d1", "d2", "d3", "d4"]
TEAMS = ["Alpha", "Alpha", "Beta", "Beta"]


def _sprint_weekend_rounds():
    """Two rounds at one circuit; the second is a sprint weekend. Carries every
    column `build_reliability_data` and `build_overtaking_data` read."""
    rows = []
    for rnd in (1, 2):
        for session in ("Q", "R", "S") if rnd == 2 else ("Q", "R"):
            for i, (drv, team) in enumerate(zip(DRIVERS, TEAMS)):
                rows.append(dict(
                    season=2024, round=rnd, event_name="Bahrain Grand Prix",
                    circuit_type="mixed", session_type=session, abbreviation=drv, team=team,
                    grid_position=float(i + 1), finish_position=float(i + 1), classified=True,
                    status="Accident" if (rnd == 1 and session == "R" and i == 3) else "Finished",
                    dnf_cause="incident" if (rnd == 1 and session == "R" and i == 3) else None,
                    gap_to_winner_median_clean_air_s=0.25 * i, is_wet=False,
                ))
    return pd.DataFrame(rows)


def _fake_fit(driver_rounds, chains=2, draws=4):
    """(idata, data) standing in for a fitted pre-weekend posterior. A plain
    xarray Dataset for the same reason tests/test_conditioning.py uses one: the
    `arviz.from_dict` signature moved between the 0.x and 1.x lines, and
    xarray is all `posterior_pace_draws` touches."""
    from model.pace import CIRCUIT_TYPES

    teams = sorted(set(TEAMS))
    rounds = sorted({(int(s), int(r)) for s, r in
                      zip(driver_rounds["season"], driver_rounds["round"])})
    rng = np.random.default_rng(0)

    def arr(*extra):
        return np.abs(rng.normal(size=(chains, draws, *extra))) + 0.5

    posterior = xr.Dataset(
        {
            "car": (("chain", "draw", "team", "round"), arr(len(teams), len(rounds))),
            "car0": (("chain", "draw", "team"), arr(len(teams))),
            "driver_skill": (("chain", "draw", "driver"), arr(len(DRIVERS))),
            "quali_offset": (("chain", "draw", "driver"), arr(len(DRIVERS))),
            "driver_track": (("chain", "draw", "driver", "circuit_type"),
                              arr(len(DRIVERS), len(CIRCUIT_TYPES))),
            "race_nu": (("chain", "draw"), arr()),
            "race_sigma": (("chain", "draw"), arr()),
            "quali_sigma": (("chain", "draw"), arr()),
        },
        coords={"team": teams, "driver": DRIVERS, "circuit_type": list(CIRCUIT_TYPES),
                 "round": list(range(len(rounds)))},
    )
    data = types.SimpleNamespace(
        n_teams=len(teams), n_drivers=len(DRIVERS), n_rounds=len(rounds),
        team_to_idx={t: i for i, t in enumerate(teams)},
        driver_to_idx={d: i for i, d in enumerate(DRIVERS)},
        round_to_idx={sr: i for i, sr in enumerate(rounds)},
    )
    return types.SimpleNamespace(posterior=posterior), data


class TestForecastWeekendWiresBothReliabilityChannels:
    """slice-6: `forecast_weekend` must pass sec 3.2's two channels through to
    the resolver, or a sprint has nothing to price the first-lap spike from and
    silently falls back to scaling the combined probability flat."""

    CIRCUITS = pd.DataFrame({
        "event_name": ["Bahrain Grand Prix"], "circuit_type": ["mixed"],
        "overtaking_difficulty": [4.5],
    })

    def _run(self, monkeypatch):
        """Forecast the sprint round, capturing the `RaceTrialInputs` the
        resolver was handed."""
        import sim.race
        from model.reliability import build_reliability_data

        driver_rounds = _sprint_weekend_rounds()
        idata, data = _fake_fit(driver_rounds)
        captured = {}
        real = sim.race.sprint_inputs_from

        def spy(inputs, *a, **kw):
            captured["gp"] = inputs
            captured["sprint"] = real(inputs, *a, **kw)
            return captured["sprint"]

        monkeypatch.setattr(sim.race, "sprint_inputs_from", spy)
        gp, sprint = forecast_weekend(
            idata, data, build_reliability_data(driver_rounds), driver_rounds,
            self.CIRCUITS, season=2024, round_=2, n_trials=200, seed=0)
        return captured, gp, sprint

    def test_the_sprint_round_really_is_resolved_as_one(self, monkeypatch):
        captured, gp, sprint = self._run(monkeypatch)
        assert sprint is not None and len(sprint) == len(DRIVERS)
        assert len(gp) == len(DRIVERS)
        assert captured, "sprint_inputs_from was never called -- no sprint was resolved"

    def test_both_channels_reach_the_resolver_and_fold_to_dnf_prob(self, monkeypatch):
        from model.reliability import combined_dnf_prob

        captured, _gp, _sprint = self._run(monkeypatch)
        inputs = captured["gp"]
        assert inputs.mechanical_dnf_prob is not None, "mechanical channel not wired"
        assert inputs.incident_dnf_prob is not None, "incident channel not wired"
        assert inputs.dnf_prob == pytest.approx(
            combined_dnf_prob(inputs.mechanical_dnf_prob, inputs.incident_dnf_prob))

    def test_the_sprint_prices_the_spike_rather_than_scaling_flat(self, monkeypatch):
        """The behavioural end of the wiring: the sprint's DNF probability is
        strictly above the flat distance scaling this path used before slice-6,
        for every driver."""
        from sim.race import SPRINT_DISTANCE_RATIO

        captured, _gp, _sprint = self._run(monkeypatch)
        flat = captured["gp"].dnf_prob * SPRINT_DISTANCE_RATIO
        assert (captured["sprint"].dnf_prob > flat).all()
