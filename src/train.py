"""Train XGBoost model on historical seasons with rule-change-year weighting."""

import argparse
import copy
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [
    "early_points_share",
    "early_avg_finish",
    "is_rule_change_year",
    "rule_change_adaptation_score",
    "prev_year_points_share",
    "prev_year_standing",
    "constructor_win_rate_5yr",
    "avg_driver_career_points_per_race",
]

RULE_CHANGE_WEIGHT_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_data(csv_path: str):
    """Load features CSV; returns (X, y, meta) for rows with a known target."""
    df = pd.read_csv(csv_path)
    train = df[df["season_points_share"].notna()].copy().reset_index(drop=True)
    X = train[FEATURE_COLS].copy()
    y = train["season_points_share"].copy()
    meta = train[["year", "constructor"]].copy()
    return X, y, meta


def leave_one_season_out_cv(model, X: pd.DataFrame, y: pd.Series,
                             meta: pd.DataFrame, rule_change_weight: float) -> list[dict]:
    """Leave-one-season-out CV. Returns list of {season, spearman, mae} dicts."""
    seasons = sorted(meta["year"].unique())
    results = []

    for held_out in seasons:
        fold_model = clone(model)

        train_mask = meta["year"] != held_out
        test_mask = meta["year"] == held_out

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_test, y_test = X[test_mask].copy(), y[test_mask].copy()

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        rc_col = X_train["is_rule_change_year"]
        sample_weights = np.where(rc_col == 1, rule_change_weight, 1.0)

        if isinstance(fold_model, Pipeline):
            last_step_name = fold_model.steps[-1][0]
            fold_model.fit(X_train, y_train, **{f"{last_step_name}__sample_weight": sample_weights})
        else:
            fold_model.fit(X_train, y_train)

        preds = fold_model.predict(X_test)

        if len(preds) < 2:
            spearman = float("nan")
        else:
            corr, _ = spearmanr(preds, y_test)
            spearman = float(corr)

        mae = float(np.mean(np.abs(preds - y_test.values)))
        results.append({"season": held_out, "spearman": spearman, "mae": mae})

    return results


def tune_rule_change_weight(model, X: pd.DataFrame, y: pd.Series,
                             meta: pd.DataFrame) -> tuple[float, dict]:
    """Try each candidate weight via LOOCV. Returns (best_weight, {weight: avg_spearman})."""
    weight_scores = {}

    for w in RULE_CHANGE_WEIGHT_CANDIDATES:
        m = copy.deepcopy(model)
        folds = leave_one_season_out_cv(m, X, y, meta, rule_change_weight=w)
        spearmans = [f["spearman"] for f in folds if not np.isnan(f["spearman"])]
        avg = float(np.mean(spearmans)) if spearmans else float("nan")
        weight_scores[w] = avg
        log.info("  Weight %.1f → avg Spearman: %.4f", w, avg)

    best_weight = max(weight_scores, key=lambda w: (not np.isnan(weight_scores[w]), weight_scores[w]))
    return best_weight, weight_scores


def compare_models(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                   rule_change_weight: float) -> tuple[str, object, pd.DataFrame]:
    """Run LOOCV for XGBoost and Ridge; return (winner_name, winner_pipeline, cv_df)."""
    candidates = {
        "XGBoost": Pipeline([
            ("imp", SimpleImputer()),
            ("reg", XGBRegressor(
                max_depth=4, n_estimators=200, learning_rate=0.05,
                random_state=42, verbosity=0,
            )),
        ]),
        "Ridge": Pipeline([
            ("imp", SimpleImputer()),
            ("reg", Ridge(alpha=1.0)),
        ]),
    }

    all_rows = []
    avg_spearmans = {}

    for name, pipeline in candidates.items():
        log.info("Running CV for %s ...", name)
        folds = leave_one_season_out_cv(
            copy.deepcopy(pipeline), X, y, meta, rule_change_weight
        )
        for fold in folds:
            all_rows.append({"model": name, **fold})
        spearmans = [f["spearman"] for f in folds if not np.isnan(f["spearman"])]
        avg = float(np.mean(spearmans)) if spearmans else float("nan")
        avg_spearmans[name] = avg
        log.info("  %s avg Spearman: %.4f", name, avg)

    cv_df = pd.DataFrame(all_rows)
    winner_name = max(avg_spearmans, key=lambda n: avg_spearmans[n])
    winner_pipeline = copy.deepcopy(candidates[winner_name])
    return winner_name, winner_pipeline, cv_df
