"""Season-level backtest harness (spec sec 7 season-level metrics, sec 8
Phase 3; autopilot slice-9). One end-to-end run on the fixture with a real
ADVI fit at tiny step counts; everything else is pure or stubbed."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import backtest
import cli
import model.pace
from backtest import (
    backtest_schedule, legacy_constructor_spearman, run_season_backtest, score_season_cutoff,
    season_level_summary,
)
from sim.season import constructors_championship, drivers_championship, remaining_calendar
from train import FEATURE_COLS

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "data", "driver_rounds_fixture.csv")


def _fixture():
    return pd.read_csv(FIXTURE)


def _legacy_features(teams=("A", "B", "C", "D", "E"), years=(2022, 2023, 2024), seed=0):
    """Synthetic `features.build_features` output: the target is a monotone
    function of `early_points_share`, so a fitted Ridge ranks it perfectly."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        share = rng.dirichlet(np.ones(len(teams)))
        for t, s in zip(teams, share):
            row = {c: 0.0 for c in FEATURE_COLS}
            row.update(year=y, constructor=t, early_points_share=s, season_points_share=s)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# End to end on the fixture: real ADVI fit, real chain, real scoring
# ---------------------------------------------------------------------------

def test_end_to_end_on_the_fixture_writes_a_season_level_block(tmp_path):
    # a race-level summary already in the out dir is kept, not overwritten
    (tmp_path / "backtest_summary.json").write_text(json.dumps({"n_rounds_scored": 7}))
    rows = run_season_backtest([2024], cutoffs=[1], n_trials=200, advi_steps=200, advi_draws=20,
                               out_dir=str(tmp_path), driver_rounds=_fixture(),
                               legacy_features=lambda c: _legacy_features())

    assert rows["status"].tolist() == ["ok"]
    summary = json.loads((tmp_path / "backtest_summary.json").read_text())
    assert summary["n_rounds_scored"] == 7
    block = summary["season_level"]
    assert block["gate_valid"] is True and block["gate_invalid_reasons"] == []
    assert isinstance(block["v2_beats_legacy_spearman"], bool)
    assert block["per_cutoff"]["1"]["n_seasons"] == 1
    assert block["per_cutoff"]["1"]["mean_constructor_spearman_legacy"] == pytest.approx(1.0)
    r = rows.iloc[0]
    assert -1.0 <= r["constructor_spearman_v2"] <= 1.0
    assert 0.0 <= r["brier_drivers_title"] <= 2.0 and 0.0 <= r["brier_constructors_title"] <= 2.0
    assert 0.0 <= r["coverage80_drivers"] <= 1.0
    for f in ("season_backtest_metrics.csv", "season_2024_r1_drivers_championship.csv",
              "season_2024_r1_constructors_championship.csv"):
        assert (tmp_path / f).exists()


# ---------------------------------------------------------------------------
# The loop with the fit and chain stubbed: leak safety and failure statuses
# ---------------------------------------------------------------------------

def _tables(teams=("Red Bull Racing", "Ferrari", "McLaren", "Mercedes")):
    drivers = pd.DataFrame({"driver": ["VER"], "p_title": [1.0], "p10_final_points": [0.0],
                            "p90_final_points": [100.0]})
    constructors = pd.DataFrame({"team": list(teams), "p_title": [1.0] + [0.0] * (len(teams) - 1),
                                 "exp_share": np.linspace(0.4, 0.1, len(teams)),
                                 "p10_final_points": 0.0, "p90_final_points": 100.0})
    return drivers, constructors


class TestLoop:
    def _run(self, monkeypatch, tmp_path, chain=None, fit=None, cutoffs=(1,)):
        seen = {}

        def fake_chain(idata, data, reliability, driver_rounds, circuits, **kw):
            seen["rounds"] = driver_rounds
            seen["calendar"] = remaining_calendar(2024, 1, loader=kw["schedule_loader"])
            return (chain or _tables)()

        monkeypatch.setattr(model.pace, "fit_advi", fit or (lambda *a, **kw: "IDATA"))
        monkeypatch.setattr(cli, "forecast_season", fake_chain)
        rows = run_season_backtest([2024], cutoffs=list(cutoffs), n_trials=10, out_dir=str(tmp_path),
                                   driver_rounds=_fixture(),
                                   legacy_features=lambda c: _legacy_features())
        return rows, seen

    def test_the_chain_sees_nothing_after_the_cutoff_but_the_real_calendar(self, monkeypatch, tmp_path):
        rows, seen = self._run(monkeypatch, tmp_path)
        assert rows["status"].tolist() == ["ok"]
        assert seen["rounds"]["round"].max() == 1
        assert seen["calendar"]["event_name"].tolist() == ["Saudi Arabian Grand Prix"]

    def test_a_failed_fit_is_recorded_and_voids_the_verdict(self, monkeypatch, tmp_path):
        def boom(*a, **kw):
            raise RuntimeError("diverged")
        rows, _ = self._run(monkeypatch, tmp_path, fit=boom)
        assert rows["status"].tolist() == ["fit_failed"]
        block = json.loads((tmp_path / "backtest_summary.json").read_text())["season_level"]
        assert block["failed_cutoffs"] == ["2024:1 (fit_failed)"]
        assert block["v2_beats_legacy_spearman"] is None

    def test_a_failed_chain_is_recorded(self, monkeypatch, tmp_path):
        def boom():
            raise RuntimeError("no schedule")
        rows, _ = self._run(monkeypatch, tmp_path, chain=boom)
        assert rows["status"].tolist() == ["forecast_failed"]

    def test_a_cutoff_with_nothing_left_to_chain_is_a_failure_not_a_skip(self, monkeypatch, tmp_path):
        rows, _ = self._run(monkeypatch, tmp_path, cutoffs=(1, 2, 6))
        assert rows["status"].tolist() == ["ok", "no_rounds_after_cutoff", "no_rounds_after_cutoff"]
        block = json.loads((tmp_path / "backtest_summary.json").read_text())["season_level"]
        assert block["n_cutoffs_expected"] == 3 and block["n_cutoffs_ok"] == 1
        assert block["gate_valid"] is False and block["v2_beats_legacy_spearman"] is None


# ---------------------------------------------------------------------------
# Summary: gate_valid / null-verdict discipline
# ---------------------------------------------------------------------------

def _rows(v2, legacy, statuses=None, cutoffs=None):
    n = len(v2)
    return pd.DataFrame({
        "season": [2022 + i // 3 for i in range(n)], "cutoff": cutoffs or [(2, 6, 12)[i % 3] for i in range(n)],
        "status": statuses or ["ok"] * n, "constructor_spearman_v2": v2,
        "constructor_spearman_legacy": legacy, "brier_drivers_title": [0.5] * n,
        "brier_constructors_title": [0.4] * n, "coverage80_drivers": [0.8] * n,
        "coverage80_constructors": [0.7] * n,
    })


def test_complete_run_reports_a_real_verdict_per_cutoff():
    out = season_level_summary(_rows([0.9, 0.8, 0.95], [0.85, 0.85, 0.85]), [2022], [2, 6, 12])
    assert out["gate_valid"] is True
    assert out["v2_beats_legacy_spearman"] is True
    assert out["per_cutoff"]["6"]["mean_constructor_spearman_v2"] == pytest.approx(0.8)
    assert out["per_cutoff"]["6"]["mean_constructor_spearman_legacy"] == pytest.approx(0.85)
    assert out["legacy_reference_spearman"] == 0.88


def test_v2_losing_to_legacy_is_false_not_none():
    out = season_level_summary(_rows([0.5, 0.5, 0.5], [0.9, 0.9, 0.9]), [2022], [2, 6, 12])
    assert out["v2_beats_legacy_spearman"] is False


def test_a_partial_run_cannot_report_a_verdict():
    # season 2023 was requested but never attempted
    out = season_level_summary(_rows([0.9, 0.9, 0.9], [0.1, 0.1, 0.1]), [2022, 2023], [2, 6, 12])
    assert out["v2_beats_legacy_spearman"] is None
    assert "2023:2 (not attempted)" in out["failed_cutoffs"]
    assert out["n_cutoffs_expected"] == 6 and out["n_cutoffs_ok"] == 3


def test_a_failed_cutoff_cannot_report_a_verdict_and_is_named():
    rows = _rows([0.9, np.nan, 0.9], [0.1, np.nan, 0.1], statuses=["ok", "fit_failed", "ok"])
    out = season_level_summary(rows, [2022], [2, 6, 12])
    assert out["gate_valid"] is False and out["v2_beats_legacy_spearman"] is None
    assert out["failed_cutoffs"] == ["2022:6 (fit_failed)"]


def test_a_missing_legacy_number_cannot_report_a_verdict():
    out = season_level_summary(_rows([0.9, 0.9, 0.9], [0.1, np.nan, 0.1]), [2022], [2, 6, 12])
    assert out["v2_beats_legacy_spearman"] is None
    assert any("legacy" in r for r in out["gate_invalid_reasons"])


# ---------------------------------------------------------------------------
# Scoring against the real final standings
# ---------------------------------------------------------------------------

def test_scores_against_the_whole_season_not_the_cutoff():
    dr = _fixture()
    final = dr[dr["session_type"] == "R"].groupby("abbreviation")["points"].sum()
    leader = final.idxmax()  # VER, both rounds
    drivers = pd.DataFrame({"driver": [leader, "PER"], "p_title": [0.75, 0.25],
                            "p10_final_points": [final[leader], 0.0],
                            "p90_final_points": [final[leader], 1.0]})
    _, constructors = _tables()
    out = score_season_cutoff(drivers, constructors, dr, 2024, lineup=[leader, "PER"])
    assert out["brier_drivers_title"] == pytest.approx(0.25 ** 2 + 0.25 ** 2)
    # leader's interval is exactly the true total (covered); PER's misses
    assert out["coverage80_drivers"] == 0.5
    assert out["brier_constructors_title"] == 0.0


def test_coverage_ignores_drivers_the_chain_did_not_simulate():
    dr = _fixture()
    drivers = pd.DataFrame({"driver": ["VER", "XXX"], "p_title": [1.0, 0.0],
                            "p10_final_points": [0.0, 5.0], "p90_final_points": [100.0, 10.0]})
    out = score_season_cutoff(drivers, _tables()[1], dr, 2024, lineup=["VER"])
    assert out["coverage80_drivers"] == 1.0


def test_championship_tables_carry_the_80pct_points_interval():
    pts = np.tile(np.arange(100.0), (2, 1)).T  # 100 trials, two entries
    d = drivers_championship(pts, np.zeros_like(pts), ["A", "B"], ["T", "T"], [0.0, 0.0])
    c = constructors_championship(pts, np.zeros_like(pts), ["T", "U"], [0.0, 0.0])
    for table in (d, c):
        assert table["p10_final_points"].tolist() == pytest.approx([9.9, 9.9])
        assert table["p90_final_points"].tolist() == pytest.approx([89.1, 89.1])


def test_schedule_is_read_off_the_rounds_that_ran_with_sprints_flagged():
    dr = _fixture()
    sprint = dr[(dr["round"] == 2) & (dr["session_type"] == "R")].assign(session_type="S")
    cal = remaining_calendar(2024, 0, loader=backtest_schedule(pd.concat([dr, sprint])))
    assert cal["round"].tolist() == [1, 2]
    assert cal["has_sprint"].tolist() == [False, True]


# ---------------------------------------------------------------------------
# Legacy at the matching cutoff
# ---------------------------------------------------------------------------

def test_legacy_trains_only_on_earlier_seasons():
    feats = _legacy_features(years=(2021, 2022, 2023, 2024))
    assert legacy_constructor_spearman(feats, 2023) == pytest.approx(1.0)
    # poisoning a later season's target cannot move an earlier season's number
    poisoned = feats.copy()
    later = poisoned["year"] == 2024
    poisoned.loc[later, "season_points_share"] = -poisoned.loc[later, "early_points_share"] * 50
    assert legacy_constructor_spearman(poisoned, 2023) == legacy_constructor_spearman(feats, 2023)


def test_legacy_without_earlier_seasons_has_no_number():
    assert np.isnan(legacy_constructor_spearman(_legacy_features(years=(2024,)), 2024))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_routes_season_level_to_the_season_harness(monkeypatch):
    calls = {}
    monkeypatch.setattr(backtest, "run_season_backtest", lambda **kw: calls.update(kw))
    monkeypatch.setattr(backtest, "run_backtest", lambda **kw: pytest.fail("race-level ran"))
    args = cli.build_parser().parse_args(["backtest", "--season-level", "--seasons", "2022-2023"])
    args.func(args)
    assert calls["seasons"] == [2022, 2023] and calls["cutoffs"] == [2, 6, 12]
    assert cli.build_parser().parse_args(["backtest"]).season_level is False
