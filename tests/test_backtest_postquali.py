"""Phase 2 backtest arm: the pure summary and the CLI flag. The sampling path
is exercised by the real backtest run, not here."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import post_quali_summary  # noqa: E402
from cli import build_parser  # noqa: E402


def _metrics(pre, post):
    return pd.DataFrame({
        "log_loss_winner_model": pre, "brier_podium_model": [0.1] * len(pre),
        "brier_points_model": [0.2] * len(pre), "spearman_model": [0.5] * len(pre),
        "log_loss_winner_postquali": post, "brier_podium_postquali": [0.09] * len(pre),
        "brier_points_postquali": [0.19] * len(pre), "spearman_postquali": [0.6] * len(pre),
    })


def _log(fell_back, regression=None):
    regression = regression or [False] * len(fell_back)
    return pd.DataFrame({"fell_back": fell_back, "regression": regression})


def test_delta_is_post_minus_pre_over_paired_rounds():
    out = post_quali_summary(_metrics([2.0, 2.0], [1.5, 1.0]), _log([False, False]))
    assert out["postquali_log_loss_delta"] == -0.75
    assert out["postquali_beats_pre_log_loss"] is True or out["postquali_beats_pre_log_loss"] == True  # noqa: E712


def test_rounds_without_a_postquali_score_do_not_flatter_either_side():
    # round 2 failed to condition (NaN): pre mean must be taken over round 1 only
    out = post_quali_summary(_metrics([1.0, 9.0], [1.2, float("nan")]), _log([False]))
    assert out["n_rounds_scored_postquali"] == 1
    assert out["mean_log_loss_winner_pre_paired"] == 1.0
    assert not out["postquali_beats_pre_log_loss"]


def test_fallback_rate_and_regressions_are_counted():
    out = post_quali_summary(_metrics([1.0] * 4, [1.0] * 4),
                              _log([True, False, True, False], [True, False, False, False]))
    assert out["fallback_rate"] == 0.5
    assert out["n_fallbacks"] == 2
    assert out["n_regressions"] == 1


def test_no_postquali_scores_yields_only_the_zero_count():
    m = _metrics([1.0], [float("nan")])
    assert post_quali_summary(m, _log([]))["n_rounds_scored_postquali"] == 0


def test_cli_flag_parses():
    args = build_parser().parse_args(["backtest", "--post-quali", "--rounds", "all"])
    assert args.post_quali is True and args.post_quali_chains == 2
    assert build_parser().parse_args(["backtest"]).post_quali is False
