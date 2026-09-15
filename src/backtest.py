"""Rolling race-level backtest (spec sec 7), phase 1 scope.

For each held-out season and round r: fit on everything strictly before
round r (earlier seasons in full, plus rounds 1..r-1 of season r), forecast
round r, score against what actually happened.

Baselines (spec sec 7):
  - "pole wins": p_win = 1 for whoever actually started on pole, 0 else.
    A zero-model baseline -- it needs no fit, just the round's own grid.
  - "pace rating": a simple Elo-style sanity check (spec sec 2: "a
    pace-rating baseline is a useful sanity check and takes an afternoon").
    Implemented here as a trailing-average-pace rating per driver
    (identical feature to the challenger's `driver_recent_pace`) converted
    to a win probability via a softmax over the round's actual entrants.

Compute-budget techniques actually used (spec sec 7):
  - ADVI by default for the backtest loop (`--method advi`), NUTS reserved
    for the production `gp fit`.
  - A coarse round grid by default (`--rounds coarse`): refit at rounds
    2, 4, 6, 9, 12, 16, 20 and score every round using the most recently
    fitted posterior (this is explicitly the sec 7 "coarser backtest grid
    for iteration" recipe). `--rounds all` runs the full per-round grid
    for the actual phase gate.
  - Reduced Monte Carlo trial count by default (5,000, vs 50k for a real
    forecast) -- ranking/log-loss metrics converge well before 50k trials
    and the backtest runs this dozens of times.

Sequential warm starts (fit round r from round r-1's posterior) and
parallelising across seasons are NOT yet implemented -- flagged as
remaining work in docs/phase1-status.md; each season's fits currently run
independently and from a fresh init, which is the main reason the full-grid
backtest is slower than the sec 7 target in this environment.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "..", "data")
OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")

COARSE_ROUNDS = [2, 4, 6, 9, 12, 16, 20]
EPS = 1e-3


def _log_loss_winner(p_win: dict, actual_winner: str) -> float:
    p = p_win.get(actual_winner, 0.0)
    p = min(max(p, EPS), 1 - EPS)
    return -np.log(p)


def _pole_wins_baseline(entrants: pd.DataFrame) -> dict:
    """p_win = 1 for the pole-sitter (grid_position == 1), 0 for everyone else."""
    p = {d: 0.0 for d in entrants["abbreviation"]}
    pole = entrants[entrants["grid_position"] == 1]
    if not pole.empty:
        p[pole.iloc[0]["abbreviation"]] = 1.0
    else:
        # no recorded grid (shouldn't happen for a real race row) -- fall back to uniform
        p = {d: 1.0 / len(p) for d in p}
    return p


def _pace_rating_baseline(driver_rounds: pd.DataFrame, entrants: pd.DataFrame,
                           season: int, round_: int, temperature: float = 0.4) -> dict:
    """Softmax over each entrant's trailing-mean race pace (lower gap = faster)."""
    hist = driver_rounds[(driver_rounds["session_type"] == "R") &
                          ((driver_rounds["season"] < season) |
                           ((driver_rounds["season"] == season) & (driver_rounds["round"] < round_)))]
    hist = hist.dropna(subset=["gap_to_winner_median_clean_air_s"])
    recent = hist.groupby("abbreviation")["gap_to_winner_median_clean_air_s"].apply(
        lambda s: s.tail(5).mean()
    )
    field_mean = hist["gap_to_winner_median_clean_air_s"].mean() if len(hist) else 1.0

    ratings = {}
    for d in entrants["abbreviation"]:
        ratings[d] = -recent.get(d, field_mean) / temperature
    m = max(ratings.values())
    exps = {d: np.exp(v - m) for d, v in ratings.items()}
    z = sum(exps.values())
    return {d: v / z for d, v in exps.items()}


def _actual_outcomes(driver_rounds: pd.DataFrame, season: int, round_: int) -> pd.DataFrame:
    rows = driver_rounds[(driver_rounds["season"] == season) & (driver_rounds["round"] == round_)
                          & (driver_rounds["session_type"] == "R")]
    return rows.dropna(subset=["abbreviation"])


def _brier(predicted_prob: dict, actual_hits: set) -> float:
    errs = [(predicted_prob.get(d, 0.0) - (1.0 if d in actual_hits else 0.0)) ** 2 for d in predicted_prob]
    return float(np.mean(errs)) if errs else float("nan")


def _spearman_vs_actual(forecast: pd.DataFrame, actual: pd.DataFrame) -> float:
    common = [d for d in forecast["driver"] if d in set(actual["abbreviation"])]
    model_rank = {d: i for i, d in enumerate(forecast.sort_values("exp_points", ascending=False)["driver"])
                  if d in common}
    actual_rank = dict(zip(actual["abbreviation"], actual["finish_position"]))
    if len(common) < 3:
        return float("nan")
    spear, _ = spearmanr([model_rank[d] for d in common], [actual_rank[d] for d in common])
    return spear


def score_round(forecast: pd.DataFrame, driver_rounds: pd.DataFrame, season: int, round_: int,
                 challenger_forecast: pd.DataFrame | None = None) -> dict | None:
    """Score the Bayesian model's forecast against reality and against the
    pole-wins / pace-rating baselines (spec sec 7). If `challenger_forecast`
    is given, also scores the XGBoost challenger through the identical
    metrics -- both models go through the same `sim.race` resolver and the
    same reliability draws, so this is an apples-to-apples comparison (spec
    sec 2: "phase 1's backtest decides whether it beats, matches, or gets
    ensembled with the Bayesian model")."""
    actual = _actual_outcomes(driver_rounds, season, round_)
    if actual.empty:
        return None

    actual_winner_rows = actual[actual["finish_position"] == 1]
    if actual_winner_rows.empty:
        return None
    actual_winner = actual_winner_rows.iloc[0]["abbreviation"]
    podium = set(actual[actual["finish_position"] <= 3]["abbreviation"])
    points_scorers = set(actual[actual["points"] > 0]["abbreviation"])

    p_win_model = dict(zip(forecast["driver"], forecast["p_win"]))
    p_podium_model = dict(zip(forecast["driver"], forecast["p_podium"]))
    p_points_model = dict(zip(forecast["driver"], forecast["p_points"]))

    p_win_pole = _pole_wins_baseline(actual)
    p_win_pace = _pace_rating_baseline(driver_rounds, actual, season, round_)

    result = {
        "season": season, "round": round_, "n_drivers": len(actual),
        "log_loss_winner_model": _log_loss_winner(p_win_model, actual_winner),
        "log_loss_winner_pole": _log_loss_winner(p_win_pole, actual_winner),
        "log_loss_winner_pace_rating": _log_loss_winner(p_win_pace, actual_winner),
        "brier_podium_model": _brier(p_podium_model, podium),
        "brier_points_model": _brier(p_points_model, points_scorers),
        "spearman_model": _spearman_vs_actual(forecast, actual),
    }

    if challenger_forecast is not None:
        p_win_ch = dict(zip(challenger_forecast["driver"], challenger_forecast["p_win"]))
        p_podium_ch = dict(zip(challenger_forecast["driver"], challenger_forecast["p_podium"]))
        p_points_ch = dict(zip(challenger_forecast["driver"], challenger_forecast["p_points"]))
        result.update({
            "log_loss_winner_challenger": _log_loss_winner(p_win_ch, actual_winner),
            "brier_podium_challenger": _brier(p_podium_ch, podium),
            "brier_points_challenger": _brier(p_points_ch, points_scorers),
            "spearman_challenger": _spearman_vs_actual(challenger_forecast, actual),
        })

    return result


def forecast_race_challenger(challenger_model, reliability_data, driver_rounds: pd.DataFrame,
                              circuits: pd.DataFrame, season: int, round_: int,
                              n_trials: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Same race resolver, same reliability draws, pace sourced from the
    XGBoost challenger's quantile function instead of the pace-model
    posterior -- an apples-to-apples comparison against `cli.forecast_race`.

    Simplification: the challenger has no separate quali equation (only the
    Bayesian model does, per spec sec 3.1), so its own race-pace quantile
    draws stand in for both race pace and grid here (quali/race noise set to
    0 -- the quantile draw already carries the challenger's full predictive
    uncertainty, so adding resolver noise on top would double-count it).
    Since grid and race pace share the same per-trial draw, the grid-lock
    term this induces doesn't reorder anything, so it is harmless."""
    from model.challenger import sample_pace
    from model.reliability import sample_mechanical_dnf_prob
    from sim.race import RaceTrialInputs, simulate_positions, summarize_trials, load_points_table
    from cli import _entrants_for_round, _circuit_info

    rng = np.random.default_rng(seed)
    entrants = _entrants_for_round(driver_rounds, season, round_)
    circuit_type, overtaking_difficulty = _circuit_info(circuits, driver_rounds, season, round_)

    hist = driver_rounds[(driver_rounds["session_type"] == "R") &
                          ((driver_rounds["season"] < season) |
                           ((driver_rounds["season"] == season) & (driver_rounds["round"] < round_)))]
    hist = hist.dropna(subset=["gap_to_winner_median_clean_air_s"])
    team_recent = hist.groupby("team")["gap_to_winner_median_clean_air_s"].apply(lambda s: s.tail(5).mean())
    driver_recent = hist.groupby("abbreviation")["gap_to_winner_median_clean_air_s"].apply(lambda s: s.tail(5).mean())
    field_mean = float(hist["gap_to_winner_median_clean_air_s"].mean()) if len(hist) else 1.0

    driver_ids = entrants["abbreviation"].tolist()
    teams = entrants["team"].tolist()
    n_drivers = len(driver_ids)

    race_pace = np.zeros((n_trials, n_drivers))
    dnf_prob = np.zeros((n_trials, n_drivers))
    team_dnf_cache = {team: sample_mechanical_dnf_prob(reliability_data, team, season, n_trials, rng)
                       for team in set(teams)}

    for i, (driver, team) in enumerate(zip(driver_ids, teams)):
        trp = float(team_recent.get(team, field_mean))
        drp = float(driver_recent.get(driver, field_mean))
        race_pace[:, i] = sample_pace(challenger_model, team, driver, circuit_type, trp, drp, n_trials, rng)
        dnf_prob[:, i] = team_dnf_cache[team]

    inputs = RaceTrialInputs(
        driver_ids=driver_ids, race_pace=race_pace,
        race_noise_nu=np.full(n_trials, 8.0), race_noise_sigma=np.zeros(n_trials),
        quali_pace=race_pace.copy(), quali_noise_sigma=np.zeros(n_trials),
        dnf_prob=dnf_prob, overtaking_difficulty=overtaking_difficulty,
    )
    positions, dnf = simulate_positions(inputs, rng)
    points_table = load_points_table()
    return summarize_trials(positions, dnf, driver_ids, teams, points_table)


def _calibration_bins(forecast: pd.DataFrame, actual: pd.DataFrame, col: str, threshold_fn) -> list[tuple[float, bool]]:
    hits = threshold_fn(actual)
    out = []
    for _, row in forecast.iterrows():
        d = row["driver"]
        if d in set(actual["abbreviation"]):
            out.append((row[col], d in hits))
    return out


def run_backtest(seasons: list[int], n_trials: int = 5000, method: str = "advi",
                  rounds: str = "coarse", advi_steps: int = 4000, out_dir: str | None = None) -> pd.DataFrame:
    from model.pace import build_pace_data, fit_nuts, fit_advi
    from model.reliability import build_reliability_data
    from model.challenger import build_challenger_frame, fit_challenger
    from cli import forecast_race, _load_circuits

    driver_rounds = pd.read_csv(os.path.join(DATA_DIR, "driver_rounds.csv"))
    driver_meta_path = os.path.join(DATA_DIR, "driver_meta.csv")
    driver_meta = pd.read_csv(driver_meta_path) if os.path.exists(driver_meta_path) else None
    circuits = _load_circuits()

    results = []
    calibration_rows = []
    t_start = time.time()

    for season in seasons:
        season_rounds = sorted(driver_rounds[(driver_rounds["season"] == season) &
                                              (driver_rounds["session_type"] == "R")]["round"].unique())
        if not season_rounds:
            continue
        fit_rounds = [r for r in season_rounds if r >= 2]
        if rounds == "coarse":
            fit_rounds = [r for r in COARSE_ROUNDS if r in season_rounds]
        if not fit_rounds:
            continue

        cached_idata, cached_data = None, None
        cached_through = None

        for i, r in enumerate(fit_rounds):
            t0 = time.time()
            through_round = r - 1
            print(f"[backtest] season={season} round={r}: fitting through {season}:{through_round} "
                  f"({method}) ...")
            data = build_pace_data(driver_rounds, driver_meta, through_season=season, through_round=through_round)
            if method == "advi":
                idata = fit_advi(data, n=advi_steps, draws=300)
            else:
                idata = fit_nuts(data, draws=250, tune=250, chains=1)
            reliability_data = build_reliability_data(driver_rounds, through_season=season, through_round=through_round)

            challenger_model = None
            try:
                challenger_frame = build_challenger_frame(driver_rounds, through_season=season,
                                                            through_round=through_round)
                if len(challenger_frame) >= 30:
                    challenger_model = fit_challenger(challenger_frame)
            except Exception as exc:
                print(f"    WARNING: challenger fit failed: {exc}")

            fit_elapsed = time.time() - t0
            print(f"    fit done in {fit_elapsed:.1f}s (challenger={'yes' if challenger_model else 'no'})")

            # score round r, and (coarse mode) every round strictly between this
            # fit and the next fit-round using this same posterior, per sec 7's
            # "forecast the intervening rounds from the most recent fit".
            next_fit_round = fit_rounds[i + 1] if i + 1 < len(fit_rounds) else (max(season_rounds) + 1)
            score_rounds = [rr for rr in season_rounds if r <= rr < next_fit_round]

            for rr in score_rounds:
                try:
                    forecast = forecast_race(idata, data, reliability_data, driver_rounds, circuits,
                                              season, rr, n_trials=n_trials, seed=rr)
                except Exception as exc:
                    print(f"    WARNING: forecast failed for {season}:{rr}: {exc}")
                    continue

                challenger_forecast = None
                if challenger_model is not None:
                    try:
                        challenger_forecast = forecast_race_challenger(
                            challenger_model, reliability_data, driver_rounds, circuits,
                            season, rr, n_trials=n_trials, seed=rr,
                        )
                    except Exception as exc:
                        print(f"    WARNING: challenger forecast failed for {season}:{rr}: {exc}")

                metrics = score_round(forecast, driver_rounds, season, rr, challenger_forecast=challenger_forecast)
                if metrics is not None:
                    metrics["fit_elapsed_s"] = fit_elapsed
                    results.append(metrics)
                    actual = _actual_outcomes(driver_rounds, season, rr)
                    calibration_rows.extend(
                        (p, hit) for p, hit in _calibration_bins(
                            forecast, actual, "p_podium", lambda a: set(a[a["finish_position"] <= 3]["abbreviation"])
                        )
                    )

    total_elapsed = time.time() - t_start
    print(f"[backtest] total elapsed: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    metrics_df = pd.DataFrame(results)
    calib_df = pd.DataFrame(calibration_rows, columns=["p_podium_pred", "podium_actual"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = out_dir or os.path.join(OUTPUTS_DIR, f"backtest_{ts}")
    os.makedirs(outdir, exist_ok=True)
    metrics_df.to_csv(os.path.join(outdir, "backtest_metrics.csv"), index=False)
    calib_df.to_csv(os.path.join(outdir, "calibration_podium.csv"), index=False)

    summary = {
        "seasons": seasons, "n_rounds_scored": len(metrics_df), "rounds_mode": rounds,
        "method": method, "n_trials": n_trials, "total_elapsed_s": total_elapsed,
    }
    if not metrics_df.empty:
        summary.update({
            "mean_log_loss_winner_model": float(metrics_df["log_loss_winner_model"].mean()),
            "mean_log_loss_winner_pole": float(metrics_df["log_loss_winner_pole"].mean()),
            "mean_log_loss_winner_pace_rating": float(metrics_df["log_loss_winner_pace_rating"].mean()),
            "mean_brier_podium_model": float(metrics_df["brier_podium_model"].mean()),
            "mean_brier_points_model": float(metrics_df["brier_points_model"].mean()),
            "mean_spearman_model": float(metrics_df["spearman_model"].mean()),
            "beats_pole_baseline": bool(metrics_df["log_loss_winner_model"].mean() <
                                         metrics_df["log_loss_winner_pole"].mean()),
            "beats_pace_rating_baseline": bool(metrics_df["log_loss_winner_model"].mean() <
                                                metrics_df["log_loss_winner_pace_rating"].mean()),
        })
        if "log_loss_winner_challenger" in metrics_df.columns and metrics_df["log_loss_winner_challenger"].notna().any():
            ch = metrics_df.dropna(subset=["log_loss_winner_challenger"])
            summary.update({
                "n_rounds_scored_challenger": len(ch),
                "mean_log_loss_winner_challenger": float(ch["log_loss_winner_challenger"].mean()),
                "mean_brier_podium_challenger": float(ch["brier_podium_challenger"].mean()),
                "mean_brier_points_challenger": float(ch["brier_points_challenger"].mean()),
                "mean_spearman_challenger": float(ch["spearman_challenger"].mean()),
                "bayesian_beats_challenger_log_loss": bool(
                    ch["log_loss_winner_model"].mean() < ch["log_loss_winner_challenger"].mean()
                ),
            })
    with open(os.path.join(outdir, "backtest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved -> {outdir}")
    return metrics_df
