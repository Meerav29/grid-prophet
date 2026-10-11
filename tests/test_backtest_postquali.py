"""Phase 2 backtest arm: the pure summary and the CLI flag. The sampling path
is exercised by the real backtest run, not here."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import backtest  # noqa: E402
from backtest import post_quali_summary  # noqa: E402
from cli import build_parser  # noqa: E402
from model.conditioning import ConditioningRecord  # noqa: E402


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
    assert out["postquali_beats_pre_log_loss"] is True
    assert out["gate_valid"] is True and out["gate_invalid_reasons"] == []


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


# --- gate validity and failure accounting ----------------------------------

def _with_status(df, statuses, rounds=None):
    df = df.copy()
    df["season"] = 2024
    df["round"] = rounds or list(range(2, 2 + len(df)))
    df["postquali_status"] = statuses
    return df


def test_failed_rounds_are_counted_named_and_invalidate_the_gate():
    m = _with_status(_metrics([1.0, 1.0, 1.0], [0.5, float("nan"), 0.5]),
                     ["ok", "conditioning_failed", "ok"])
    out = post_quali_summary(m, _log([False, False]))
    assert out["n_postquali_attempted"] == 3
    assert out["n_postquali_failed"] == 1
    assert out["postquali_failed_rounds"] == ["2024:3 (conditioning_failed)"]
    assert out["gate_valid"] is False
    assert out["postquali_beats_pre_log_loss"] is None


def test_coarse_rounds_invalidate_the_gate_even_with_no_failures():
    m = _with_status(_metrics([1.0, 1.0], [0.5, 0.5]), ["ok", "ok"])
    out = post_quali_summary(m, _log([False, False]), rounds_mode="coarse")
    assert out["gate_valid"] is False
    assert "coarse" in out["gate_invalid_reasons"][0]
    assert out["postquali_beats_pre_log_loss"] is None
    assert out["postquali_log_loss_delta"] == -0.5  # the number is still reported


def test_all_ok_in_all_mode_keeps_a_real_boolean_verdict():
    m = _with_status(_metrics([1.0, 1.0], [1.5, 1.5]), ["ok", "ok"])
    out = post_quali_summary(m, _log([False, False]), rounds_mode="all")
    assert out["gate_valid"] is True
    assert out["postquali_beats_pre_log_loss"] is False


# --- the arm itself, with the sampler and the simulator patched out --------

class TestPostQualiArm:
    """`_post_quali_arm` is the wiring that makes the arm do anything. Patching
    its collaborators lets these assert what it hands them, which is where a
    deleted holdout, a swapped posterior or a dropped log write would hide."""

    @staticmethod
    def _record(**over):
        base = dict(season=2024, round=5, path="warm_start", fell_back=False, early_season=True,
                    warm_start_s=10.0, fallback_s=0.0, total_s=10.0, warm_target_s=120.0,
                    applied_target_s=120.0, regression=False)
        base.update(over)
        return ConditioningRecord(**base)

    def _patch(self, monkeypatch, record=None, condition_raises=None, forecast_raises=None,
               scored="metrics"):
        import cli
        import model.conditioning as conditioning

        seen = {"forecast": None}

        def fake_condition(driver_rounds, meta, season, rnd, prior, chains, seed):
            if condition_raises:
                raise condition_raises
            seen["condition"] = dict(prior=prior, chains=chains, season=season, round=rnd)
            return "CONDITIONED_IDATA", "CONDITIONED_DATA", record or self._record()

        def fake_forecast(idata, data, rel, dr, circuits, season, rnd, **kw):
            if forecast_raises:
                raise forecast_raises
            seen["forecast"] = dict(idata=idata, data=data, **kw)
            return "FORECAST"

        monkeypatch.setattr(conditioning, "condition_on_quali", fake_condition)
        monkeypatch.setattr(cli, "forecast_race", fake_forecast)
        if scored == "metrics":
            scored = {f"{m}_model": 0.5 for m in backtest.POST_QUALI_METRICS}
        monkeypatch.setattr(backtest, "score_round", lambda *a, **kw: scored)
        return seen

    def _run(self, tmp_path, records):
        return backtest._post_quali_arm("PRIOR_IDATA", None, None, None, "REL", 2024, 5, 100, 3,
                                        records, str(tmp_path))

    def test_conditions_from_the_prior_then_forecasts_from_the_conditioned_posterior(
            self, monkeypatch, tmp_path):
        seen = self._patch(monkeypatch)
        out = self._run(tmp_path, [])
        assert seen["condition"] == dict(prior="PRIOR_IDATA", chains=3, season=2024, round=5)
        assert seen["forecast"]["idata"] == "CONDITIONED_IDATA"
        assert seen["forecast"]["data"] == "CONDITIONED_DATA"
        assert seen["forecast"]["mode"] == "post-quali"
        assert out["postquali_status"] == "ok"
        assert set(out) >= {f"{m}_postquali" for m in backtest.POST_QUALI_METRICS}

    def test_the_conditioning_record_is_kept_in_memory_and_in_the_log(self, monkeypatch, tmp_path):
        from model.conditioning import read_log

        self._patch(monkeypatch, record=self._record(fell_back=True, regression=True))
        records = []
        self._run(tmp_path, records)
        assert len(records) == 1 and records[0]["fell_back"] is True
        log = read_log(str(tmp_path / "conditioning_log.csv"))
        assert len(log) == 1 and bool(log["regression"].iloc[0]) is True

    def test_a_conditioning_failure_scores_nothing_and_records_nothing(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, condition_raises=RuntimeError("sampler blew up"))
        records = []
        assert self._run(tmp_path, records) == {"postquali_status": "conditioning_failed"}
        assert records == [] and not (tmp_path / "conditioning_log.csv").exists()

    def test_a_forecast_failure_is_flagged_but_keeps_the_conditioning_record(
            self, monkeypatch, tmp_path):
        self._patch(monkeypatch, forecast_raises=ValueError("no grid"))
        records = []
        assert self._run(tmp_path, records) == {"postquali_status": "forecast_failed"}
        assert len(records) == 1

    def test_an_unscorable_round_is_flagged_not_zeroed(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, scored=None)
        assert self._run(tmp_path, []) == {"postquali_status": "unscored"}


class TestBacktestWiring:
    def test_cmd_backtest_hands_the_post_quali_flags_to_run_backtest(self, monkeypatch):
        import cli

        seen = {}
        monkeypatch.setattr(backtest, "run_backtest", lambda **kw: seen.update(kw))
        args = build_parser().parse_args(
            ["backtest", "--seasons", "2024", "--post-quali", "--post-quali-chains", "4",
             "--rounds", "all"])
        cli.cmd_backtest(args)
        assert seen["post_quali"] is True
        assert seen["post_quali_chains"] == 4
        assert seen["rounds"] == "all"
