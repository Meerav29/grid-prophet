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


def retrain_and_save(pipeline, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                     winner_name: str, rule_change_weight: float, model_path: str):
    """Retrain pipeline on all data with best weight and save a bundle to model_path."""
    rc_col = X["is_rule_change_year"]
    sample_weights = np.where(rc_col == 1, rule_change_weight, 1.0)

    if isinstance(pipeline, Pipeline):
        last_step_name = pipeline.steps[-1][0]
        pipeline.fit(X, y, **{f"{last_step_name}__sample_weight": sample_weights})
    else:
        pipeline.fit(X, y)

    bundle = {
        "model": pipeline,
        "feature_cols": FEATURE_COLS,
        "winner_name": winner_name,
        "rule_change_weight": rule_change_weight,
    }
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved model bundle → %s", model_path)


def _print_summary(winner_name: str, best_weight: float, weight_scores: dict,
                   cv_df: pd.DataFrame, pipeline):
    """Print training summary: weight tuning results, model comparison, feature importances."""
    print("\n" + "=" * 60)
    print("GRID PROPHET — TRAINING SUMMARY")
    print("=" * 60)

    print("\n-- Rule-change weight tuning --")
    for w, s in sorted(weight_scores.items()):
        marker = " ◄ BEST" if w == best_weight else ""
        print(f"  weight {w:.1f} → avg Spearman {s:.4f}{marker}")

    print("\n-- Model comparison (leave-one-season-out CV) --")
    summary = cv_df.groupby("model")[["spearman", "mae"]].mean()
    for model_name, row in summary.iterrows():
        marker = " ◄ WINNER" if model_name == winner_name else ""
        print(f"  {model_name}: avg Spearman {row['spearman']:.4f}, avg MAE {row['mae']:.4f}{marker}")

    print(f"\n-- Winner: {winner_name} (weight={best_weight:.1f}) --")

    last_estimator = pipeline.steps[-1][1]
    if hasattr(last_estimator, "feature_importances_"):
        print("\n-- Feature importances (XGBoost) --")
        for col, imp in sorted(zip(FEATURE_COLS, last_estimator.feature_importances_), key=lambda x: -x[1]):
            print(f"  {col:<42} {imp:.4f}")
    elif hasattr(last_estimator, "coef_"):
        print("\n-- Ridge coefficients --")
        for col, coef in sorted(zip(FEATURE_COLS, last_estimator.coef_), key=lambda x: -abs(x[1])):
            print(f"  {col:<42} {coef:+.4f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Grid Prophet model.")
    parser.add_argument(
        "--features", default=os.path.join(DATA_DIR, "features.csv"),
    )
    parser.add_argument(
        "--model-out", default=os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl"),
    )
    parser.add_argument(
        "--cv-out", default=os.path.join(DATA_DIR, "cv_results.csv"),
    )
    args = parser.parse_args()

    log.info("Loading data from %s ...", args.features)
    X, y, meta = load_data(args.features)
    log.info("Training set: %d rows, %d seasons", len(X), meta["year"].nunique())

    tune_pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge(alpha=1.0))])
    log.info("Tuning rule-change sample weight ...")
    best_weight, weight_scores = tune_rule_change_weight(tune_pipeline, X, y, meta)
    log.info("Best rule-change weight: %.1f", best_weight)

    log.info("Comparing XGBoost vs Ridge ...")
    winner_name, winner_pipeline, cv_df = compare_models(X, y, meta, best_weight)

    cv_df.to_csv(args.cv_out, index=False)
    log.info("CV results saved → %s", args.cv_out)

    _print_summary(winner_name, best_weight, weight_scores, cv_df, winner_pipeline)

    log.info("Retraining %s on all data ...", winner_name)
    retrain_and_save(
        winner_pipeline, X, y, meta,
        winner_name=winner_name,
        rule_change_weight=best_weight,
        model_path=args.model_out,
    )
    log.info("Done. Model saved to %s", args.model_out)


if __name__ == "__main__":
    main()
