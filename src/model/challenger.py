"""XGBoost challenger (spec sec 2, "challenger" paragraph): a per-driver-
per-race model predicting finishing position from the same inputs as the
Bayesian pace model, behind the same race-forecast interface.

Team/driver/circuit-type are integer-coded categoricals, plus a handful of
recent-form features built directly from `driver_rounds.csv` (a driver's own
trailing pace average, their team's trailing pace average). Turned into a
probability distribution over finishing position via quantile regression on
`gap_to_winner_median_clean_air_s` (`reg:quantileerror`, one model per
quantile array via XGBoost's native multi-quantile support), which is then
converted into simulated race outcomes the same way the Bayesian pace draws
are: sample a quantile-implied pace value per driver per trial and feed it
through `sim.race.simulate_positions`.

This is intentionally the cheaper, non-Bayesian alternative sec 2 asks the
phase 1 backtest to score against the Bayesian model ("cheaper, no sampler
... phase 1's backtest decides whether it beats, matches, or gets ensembled
with the Bayesian model").

Backend note: the spec allows LightGBM or XGBoost. LightGBM was tried first,
but `lgb.Dataset(...).construct()` segfaults (`access violation reading
0x0`) in this environment as soon as `pandas` has been imported in the
process -- reproduced with plain random data, independent of this project's
pipeline, and present even feeding LightGBM pure numpy arrays with no
pandas objects in sight (a DLL/ABI conflict between the installed
pandas/numpy/lightgbm wheels on this Windows/Python 3.12 setup). Since the
legacy pipeline (`train.py`) already uses XGBoost successfully alongside
pandas in this exact environment, XGBoost is used here instead -- a
packaging problem, not a modelling one; see docs/phase1-status.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
FEATURE_COLS = ["team", "driver", "circuit_type", "team_recent_pace", "driver_recent_pace"]


def _trailing_mean(df: pd.DataFrame, group_col: str, value_col: str, window: int = 5) -> pd.Series:
    """Trailing mean of `value_col` within `group_col`, computed strictly
    before each row's own round (no leakage) -- ordered by (season, round)."""
    df = df.sort_values(["season", "round"])
    return (
        df.groupby(group_col)[value_col]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


def build_challenger_frame(driver_rounds: pd.DataFrame, through_season: int | None = None,
                            through_round: int | None = None) -> pd.DataFrame:
    df = driver_rounds[driver_rounds["session_type"] == "R"].copy()
    df = df.dropna(subset=["gap_to_winner_median_clean_air_s", "team", "abbreviation"])
    df["circuit_type"] = df["circuit_type"].fillna("mixed")
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    df["team_recent_pace"] = _trailing_mean(df, "team", "gap_to_winner_median_clean_air_s")
    df["driver_recent_pace"] = _trailing_mean(df, "abbreviation", "gap_to_winner_median_clean_air_s")
    # fall back to the field-wide mean-so-far for cold-start rows (new team/driver)
    running_mean = df["gap_to_winner_median_clean_air_s"].expanding().mean().shift(1)
    df["team_recent_pace"] = df["team_recent_pace"].fillna(running_mean)
    df["driver_recent_pace"] = df["driver_recent_pace"].fillna(running_mean)
    df["team_recent_pace"] = df["team_recent_pace"].fillna(df["gap_to_winner_median_clean_air_s"].mean())
    df["driver_recent_pace"] = df["driver_recent_pace"].fillna(df["gap_to_winner_median_clean_air_s"].mean())

    if through_season is not None:
        cutoff = (through_season, through_round if through_round is not None else 10_000)
        keep = [(s, r) <= cutoff for s, r in zip(df["season"], df["round"])]
        df = df[keep]

    return df


@dataclass
class ChallengerModel:
    booster: object        # xgboost.Booster, trained with quantile_alpha=QUANTILES (multi-target)
    team_categories: list
    driver_categories: list
    circuit_categories: list


def _category_index(categories: list, value) -> int:
    """Index of `value` in `categories`, or len(categories) (an extra
    "unknown" bucket the model sees as just another integer level) if unseen."""
    try:
        return categories.index(value)
    except ValueError:
        return len(categories)


def _encode_rows(team: pd.Series, driver: pd.Series, circuit: pd.Series,
                  team_categories: list, driver_categories: list, circuit_categories: list) -> np.ndarray:
    t = np.array([_category_index(team_categories, v) for v in team], dtype=np.float64)
    d = np.array([_category_index(driver_categories, v) for v in driver], dtype=np.float64)
    c = np.array([_category_index(circuit_categories, v) for v in circuit], dtype=np.float64)
    return np.column_stack([t, d, c])


def fit_challenger(train_df: pd.DataFrame, seed: int = 0) -> ChallengerModel:
    import xgboost as xgb

    df = train_df.copy()
    team_categories = sorted(df["team"].unique())
    driver_categories = sorted(df["abbreviation"].unique())
    circuit_categories = sorted(df["circuit_type"].unique())

    cat_cols = _encode_rows(df["team"], df["abbreviation"], df["circuit_type"],
                             team_categories, driver_categories, circuit_categories)
    num_cols = df[["team_recent_pace", "driver_recent_pace"]].to_numpy(dtype=np.float64)
    X = np.column_stack([cat_cols, num_cols])
    y = df["gap_to_winner_median_clean_air_s"].to_numpy(dtype=np.float64)

    # Multi-quantile regression in one model: XGBoost's reg:quantileerror
    # accepts a list of alphas and predicts all of them at once (one column
    # per quantile), which is both simpler and cheaper than one booster per
    # quantile.
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES),
        n_estimators=150, max_depth=4, learning_rate=0.1,
        min_child_weight=5, random_state=seed, verbosity=0,
    )
    model.fit(X, y)

    return ChallengerModel(booster=model, team_categories=team_categories,
                            driver_categories=driver_categories, circuit_categories=circuit_categories)


def predict_quantiles(model: ChallengerModel, team: str, driver: str, circuit_type: str,
                       team_recent_pace: float, driver_recent_pace: float) -> dict:
    """Predict the quantiles of gap-to-field-leader pace for one entrant.
    Unseen team/driver/circuit values get their own extra integer code
    (see `_category_index`) -- a wide "unknown car" fallback driven by
    whatever the trees learned for that unseen bucket at fit time."""
    row = _encode_rows(pd.Series([team]), pd.Series([driver]), pd.Series([circuit_type]),
                        model.team_categories, model.driver_categories, model.circuit_categories)
    X = np.column_stack([row, np.array([[team_recent_pace, driver_recent_pace]])])
    preds = model.booster.predict(X)[0]  # shape (n_quantiles,)
    return {q: float(v) for q, v in zip(QUANTILES, np.atleast_1d(preds))}


def sample_pace(model: ChallengerModel, team: str, driver: str, circuit_type: str,
                 team_recent_pace: float, driver_recent_pace: float, n_draws: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Sample pace draws by linearly interpolating the predicted quantile
    function -- the cheap non-Bayesian analogue of a posterior draw, used so
    the challenger can be run through the same `sim.race` resolver."""
    q_preds = predict_quantiles(model, team, driver, circuit_type, team_recent_pace, driver_recent_pace)
    qs = np.array(sorted(q_preds.keys()))
    vals = np.array([q_preds[q] for q in qs])
    vals = np.sort(vals)  # guard against small quantile-crossing from independent quantile fits
    u = rng.uniform(qs.min(), qs.max(), size=n_draws)
    return np.interp(u, qs, vals)
