"""Post-quali calibration (autopilot slice-10; docs/phase2-status.md caveat 1):
a podium calibration file per arm, a binned table and an ECE for both arms in
the summary, and the rounds where post-quali did much worse than pre-weekend."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import backtest
import model.pace
from backtest import (
    CALIBRATION_COLUMNS, calibration_summary, post_quali_summary, postquali_worse_rounds,
    score_round,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "data", "driver_rounds_fixture.csv")


def _calib(pairs):
    return pd.DataFrame([(p, h, 2024, 1) for p, h in pairs], columns=CALIBRATION_COLUMNS)


# --- calibration_summary ----------------------------------------------------

def test_binned_table_and_ece_match_a_hand_computation():
    # bin [0.0, 0.1): preds 0.05, 0.05, hits 0/2 -> |0.05 - 0| = 0.05, weight 2/5
    # bin [0.5, 0.6): preds 0.55 x3, hits 3/3 -> 0.45, weight 3/5
    out = calibration_summary(_calib([(0.05, False), (0.05, False),
                                      (0.55, True), (0.55, True), (0.55, True)]))
    assert out["n"] == 5
    assert [(b["bin_lo"], b["n"]) for b in out["bins"]] == [(0.0, 2), (0.5, 3)]
    assert out["bins"][1]["mean_pred"] == pytest.approx(0.55)
    assert out["bins"][1]["observed_rate"] == 1.0
    assert out["ece"] == pytest.approx(2 / 5 * 0.05 + 3 / 5 * 0.45)


def test_a_perfectly_calibrated_arm_has_zero_ece():
    pairs = [(0.25, True)] + [(0.25, False)] * 3 + [(0.75, True)] * 3 + [(0.75, False)]
    assert calibration_summary(_calib(pairs))["ece"] == pytest.approx(0.0)


def test_p_of_one_falls_in_the_top_bin_and_empty_bins_are_omitted():
    out = calibration_summary(_calib([(1.0, True), (0.0, False)]))
    assert [(b["bin_lo"], b["bin_hi"]) for b in out["bins"]] == [(0.0, 0.1), (0.9, 1.0)]


def test_an_empty_arm_reports_no_ece_rather_than_zero():
    assert calibration_summary(_calib([])) == {"n": 0, "ece": None, "bins": []}


# --- the confident-miss listing --------------------------------------------

def _metrics(pre, post):
    n = len(pre)
    return pd.DataFrame({
        "season": [2022] * n, "round": list(range(1, n + 1)),
        "log_loss_winner_model": pre, "log_loss_winner_postquali": post,
        "grid_pole_driver": [f"P{i}" for i in range(n)],
        "actual_winner": [f"W{i}" for i in range(n)],
        "brier_podium_model": 0.1, "brier_points_model": 0.2, "spearman_model": 0.5,
        "brier_podium_postquali": 0.1, "brier_points_postquali": 0.2, "spearman_postquali": 0.5,
    })


def test_only_rounds_worse_by_more_than_the_threshold_are_listed_worst_first():
    # deltas: +2.4, +1.0 (at threshold: not listed), -0.5, +1.5, NaN (no post score)
    m = _metrics([1.5, 1.0, 1.0, 1.0, 1.0], [3.9, 2.0, 0.5, 2.5, float("nan")])
    out = postquali_worse_rounds(m, threshold=1.0)
    assert [(r["round"], r["grid_pole_driver"], r["actual_winner"]) for r in out] == [
        (1, "P0", "W0"), (4, "P3", "W3")]
    assert out[0]["delta"] == pytest.approx(2.4)
    assert out[0]["log_loss_pre"] == 1.5 and out[0]["log_loss_postquali"] == 3.9


def test_post_quali_summary_carries_the_listing_and_its_threshold():
    out = post_quali_summary(_metrics([1.0, 1.0], [3.0, 0.5]),
                             pd.DataFrame({"fell_back": [False, False], "regression": [False, False]}))
    assert out["postquali_worse_threshold_nats"] == backtest.POSTQUALI_WORSE_THRESHOLD_NATS
    assert [r["round"] for r in out["postquali_worse_rounds"]] == [1]
    json.dumps(out)  # the listing must survive into backtest_summary.json


def test_score_round_names_the_pole_sitter_and_the_winner():
    dr = pd.read_csv(FIXTURE)
    race = dr[(dr["season"] == 2024) & (dr["round"] == 1) & (dr["session_type"] == "R")]
    forecast = pd.DataFrame({"driver": race["abbreviation"], "p_win": 0.05, "p_podium": 0.15,
                             "p_points": 0.5, "exp_points": np.arange(len(race), 0, -1)})
    out = score_round(forecast, dr, 2024, 1)
    assert out["grid_pole_driver"] == race.loc[race["grid_position"] == 1, "abbreviation"].iloc[0]
    assert out["actual_winner"] == race.loc[race["finish_position"] == 1, "abbreviation"].iloc[0]


# --- the arm's calibration rows ----------------------------------------------

def test_the_post_quali_arm_adds_one_calibration_row_per_scored_driver(monkeypatch, tmp_path):
    import cli
    import model.conditioning as conditioning
    from model.conditioning import ConditioningRecord

    dr = pd.read_csv(FIXTURE)
    race = dr[(dr["season"] == 2024) & (dr["round"] == 2) & (dr["session_type"] == "R")]
    forecast = pd.DataFrame({"driver": race["abbreviation"], "p_win": 0.05, "p_podium": 0.2,
                             "p_points": 0.5, "exp_points": np.arange(len(race), 0, -1)})
    record = ConditioningRecord(season=2024, round=2, path="warm_start", fell_back=False,
                                early_season=True, warm_start_s=1.0, fallback_s=0.0, total_s=1.0,
                                warm_target_s=120.0, applied_target_s=300.0, regression=False)
    monkeypatch.setattr(conditioning, "condition_on_quali", lambda *a, **kw: ("I", "D", record))
    monkeypatch.setattr(cli, "forecast_race", lambda *a, **kw: forecast)
    rows = []
    out = backtest._post_quali_arm("PRIOR", dr, None, None, "REL", 2024, 2, 100, 1, [],
                                   str(tmp_path), calibration_rows=rows)
    assert out["postquali_status"] == "ok"
    assert len(rows) == len(race)
    assert sum(hit for _, hit, _, _ in rows) == 3
    assert {(s, r) for _, _, s, r in rows} == {(2024, 2)}


# --- end to end on the fixture ---------------------------------------------

def test_end_to_end_writes_both_calibration_files_and_both_summaries(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(FIXTURE, data_dir / "driver_rounds.csv")
    monkeypatch.setattr(backtest, "DATA_DIR", str(data_dir))
    # conditioning is NUTS; swap in a tiny ADVI fit so this runs in seconds
    real_advi = model.pace.fit_advi
    monkeypatch.setattr(model.pace, "fit_nuts",
                        lambda data, **kw: real_advi(data, n=200, draws=20))
    out_dir = tmp_path / "out"
    backtest.run_backtest([2024], n_trials=200, method="advi", rounds="all", advi_steps=200,
                          out_dir=str(out_dir), post_quali=True, post_quali_chains=1)

    pre = pd.read_csv(out_dir / "calibration_podium.csv")
    post = pd.read_csv(out_dir / "calibration_podium_postquali.csv")
    assert list(pre.columns) == list(post.columns) == CALIBRATION_COLUMNS
    assert len(post) == len(pre) > 0
    summary = json.loads((out_dir / "backtest_summary.json").read_text())
    calib = summary["calibration_podium"]
    assert set(calib) == {"pre", "pre_paired", "postquali"}
    for arm in calib.values():
        assert arm["n"] == len(pre) and 0.0 <= arm["ece"] <= 1.0
        assert sum(b["n"] for b in arm["bins"]) == arm["n"]
    assert isinstance(summary["postquali_worse_rounds"], list)
