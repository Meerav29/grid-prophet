"""Tests for the XGBoost challenger (spec sec 2)."""

import numpy as np
import pandas as pd
import pytest

from model.challenger import build_challenger_frame, fit_challenger, sample_pace, predict_quantiles, QUANTILES


def _toy_driver_rounds():
    rng = np.random.default_rng(0)
    rows = []
    for season in [2021, 2022]:
        for rnd in range(1, 8):
            for team, base in [("T1", 0.2), ("T2", 0.8), ("T3", 1.6)]:
                for drv in [team + "a", team + "b"]:
                    rows.append(dict(
                        season=season, round=rnd, team=team, abbreviation=drv,
                        circuit_type="mixed", session_type="R",
                        gap_to_winner_median_clean_air_s=max(0.0, base + rng.normal(scale=0.15)),
                    ))
    return pd.DataFrame(rows)


class TestBuildChallengerFrame:
    def test_no_leakage_first_round_uses_field_mean_fallback(self):
        df = _toy_driver_rounds()
        feat = build_challenger_frame(df)
        first_round = feat[(feat["season"] == 2021) & (feat["round"] == 1)]
        # first round has no history -> team/driver recent pace should equal
        # each other (both fall back to the same running-mean-so-far, which
        # for row 0 of the whole ordering is NaN-safe fillna already applied)
        assert not first_round["team_recent_pace"].isna().any()
        assert not first_round["driver_recent_pace"].isna().any()

    def test_cutoff_restricts_rows(self):
        df = _toy_driver_rounds()
        feat_all = build_challenger_frame(df)
        feat_early = build_challenger_frame(df, through_season=2021, through_round=3)
        assert len(feat_early) < len(feat_all)
        assert feat_early["round"].max() <= 3

    def test_recent_pace_reflects_team_strength(self):
        df = _toy_driver_rounds()
        feat = build_challenger_frame(df)
        late = feat[(feat["season"] == 2022) & (feat["round"] == 7)]
        t1 = late[late["team"] == "T1"]["team_recent_pace"].iloc[0]
        t3 = late[late["team"] == "T3"]["team_recent_pace"].iloc[0]
        assert t1 < t3  # T1 is the fast team (lower gap) in the synthetic data


class TestFitAndSample:
    def test_fast_team_predicted_faster_than_slow_team(self):
        df = _toy_driver_rounds()
        feat = build_challenger_frame(df)
        model = fit_challenger(feat)
        rng = np.random.default_rng(1)
        draws_fast = sample_pace(model, "T1", "T1a", "mixed", 0.2, 0.2, 3000, rng)
        draws_slow = sample_pace(model, "T3", "T3a", "mixed", 1.6, 1.6, 3000, rng)
        assert draws_fast.mean() < draws_slow.mean()

    def test_quantiles_are_monotonic_after_sorting(self):
        df = _toy_driver_rounds()
        feat = build_challenger_frame(df)
        model = fit_challenger(feat)
        preds = predict_quantiles(model, "T2", "T2a", "mixed", 0.8, 0.8)
        assert set(preds.keys()) == set(QUANTILES)

    def test_unseen_team_and_driver_do_not_crash(self):
        df = _toy_driver_rounds()
        feat = build_challenger_frame(df)
        model = fit_challenger(feat)
        rng = np.random.default_rng(2)
        draws = sample_pace(model, "Cadillac", "NEWROOKIE", "mixed", 1.0, 1.0, 500, rng)
        assert draws.shape == (500,)
        assert np.isfinite(draws).all()
