"""End-to-end check of `gp race --mode post-quali` on a real, completed round
-- slice-1's first acceptance criterion (spec §3.3 step 3, §6).

This is the criterion CI could not previously check. `data/driver_rounds.csv`
is gitignored, so no cloud checkout had a real round to run against and the
only execution evidence was synthetic. `tests/fixtures/driver_rounds_sample.csv`
is four completed 2024 rounds straight out of `collect_v2` (regenerate with
`tests/fixtures/make_driver_rounds_fixture.py`), which closes that gap.

These tests really do fit the model and really do condition on quali, so
`test_post_quali_runs_end_to_end` is by far the slowest test in the suite
(roughly a minute). That is the cost of the criterion being real.

The CLI reads its paths from module-level constants, so pointing it at the
fixture needs no production code change -- monkeypatching those is enough.
"""

import os

import pandas as pd
import pytest

import cli

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "driver_rounds_sample.csv")

# rounds 1-3 are fit; round 4 is held out and conditioned on
FIT_THROUGH = "2024:3"
TARGET_SEASON, TARGET_ROUND = 2024, 4

# spec §6's listed columns, before the wide position-distribution block
SPEC_COLUMNS = [
    "driver", "team", "p_win", "p_podium", "p_points", "p_dnf", "exp_points",
    "p10_finish", "p50_finish", "p90_finish",
]


@pytest.fixture(scope="module")
def driver_rounds():
    return pd.read_csv(FIXTURE)


class TestFixtureIsUsableRealData:
    """Cheap assertions about the real-data contract that the synthetic run
    could not establish: that these are real completed rounds and that
    `grid_position` is actually populated on them."""

    def test_fixture_covers_the_fit_and_target_rounds(self, driver_rounds):
        rounds = set(driver_rounds["round"].unique())
        assert {1, 2, 3, 4}.issubset(rounds)
        assert set(driver_rounds["session_type"].unique()) == {"Q", "R"}

    def test_real_race_rows_carry_a_usable_grid_position(self, driver_rounds):
        race = driver_rounds[(driver_rounds["season"] == TARGET_SEASON)
                              & (driver_rounds["round"] == TARGET_ROUND)
                              & (driver_rounds["session_type"] == "R")]
        assert len(race) > 10
        # post-quali mode is worthless if the real grid is not actually there
        assert (race["grid_position"].fillna(0) > 0).sum() > 10

    def test_real_grid_resolves_to_a_clean_permutation(self, driver_rounds):
        entrants = cli._entrants_for_round(driver_rounds, TARGET_SEASON, TARGET_ROUND)
        drivers = entrants["abbreviation"].tolist()
        grid = cli._real_grid_for_round(driver_rounds, TARGET_SEASON, TARGET_ROUND, drivers)
        assert sorted(grid) == list(range(1, len(drivers) + 1))


def test_post_quali_runs_end_to_end(tmp_path, monkeypatch):
    """`gp fit` then `gp race --mode post-quali` against a completed round,
    writing a forecast with the columns §6 lists."""
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(cli, "DRIVER_ROUNDS_CSV", FIXTURE)
    monkeypatch.setattr(cli, "DRIVER_META_CSV", str(tmp_path / "absent.csv"))
    monkeypatch.setattr(cli, "OUTPUTS_DIR", str(outputs))
    monkeypatch.setattr(cli, "LATEST_POINTER", str(outputs / "latest.txt"))

    fit_dir = cli.main([
        "fit", "--through", FIT_THROUGH, "--draws", "100", "--tune", "100",
        "--chains", "2", "--skip-challenger",
    ])
    assert os.path.exists(os.path.join(fit_dir, "idata.nc"))

    out_dir = tmp_path / "race"
    cli.main([
        "race", "--round", str(TARGET_ROUND), "--season", str(TARGET_SEASON),
        "--mode", "post-quali", "--n-trials", "2000", "--chains", "2",
        "--fit-dir", fit_dir, "--out-dir", str(out_dir),
    ])

    # NB: spec §6 and the acceptance criterion both name `race_forecast.csv`;
    # the CLI has written `race_forecast_{season}_{round}.csv` since phase 1.
    # Asserting what it actually does rather than papering over the mismatch.
    forecast_path = out_dir / f"race_forecast_{TARGET_SEASON}_{TARGET_ROUND}.csv"
    assert forecast_path.exists()

    forecast = pd.read_csv(forecast_path)
    assert SPEC_COLUMNS == list(forecast.columns)[:len(SPEC_COLUMNS)]
    n = len(forecast)
    assert [f"p_pos_{p}" for p in range(1, n + 1)] == list(forecast.columns)[len(SPEC_COLUMNS):]

    assert forecast["p_win"].sum() == pytest.approx(1.0, abs=1e-6)
    assert ((forecast["p_dnf"] >= 0) & (forecast["p_dnf"] <= 1)).all()

    log = pd.read_csv(out_dir / "conditioning_log.csv")
    assert len(log) == 1
    assert log.iloc[0]["season"] == TARGET_SEASON and log.iloc[0]["round"] == TARGET_ROUND
    assert log.iloc[0]["path"] in {"warm_start", "fallback_full_refit"}
