"""Generate 2026 constructor championship predictions from early-season data."""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from collect import collect_race_results
from features import (
    EARLY_ROUNDS,
    RULE_CHANGE_YEARS,
    CONSTRUCTOR_LINEAGE,
    _canonical,
    _apply_rebrand_dampening,
    _early_season_features,
    _rule_change_features,
    _momentum_features,
    _driver_quality_feature,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

PREDICT_YEAR = 2026


def collect_2026_early_rounds() -> pd.DataFrame:
    """Fetch rounds 1..EARLY_ROUNDS of PREDICT_YEAR from FastF1."""
    log.info("Collecting %d rounds 1-%d from FastF1 ...", PREDICT_YEAR, EARLY_ROUNDS)
    raw = collect_race_results(PREDICT_YEAR, PREDICT_YEAR)
    if raw.empty:
        raise RuntimeError(
            "No %d race data returned from FastF1 for rounds 1-%d. "
            "Check network connectivity or FastF1 availability." % (PREDICT_YEAR, EARLY_ROUNDS)
        )
    return raw[raw["round"] <= EARLY_ROUNDS].copy()


def build_2026_features() -> pd.DataFrame:
    """Build the 2026 feature rows by combining historical data with 2026 early rounds."""
    race_csv = os.path.join(DATA_DIR, "race_results.csv")
    standings_csv = os.path.join(DATA_DIR, "constructor_standings.csv")

    log.info("Loading historical race results ...")
    history = pd.read_csv(race_csv)

    log.info("Loading historical constructor standings ...")
    standings_raw = pd.read_csv(standings_csv)

    log.info("Collecting 2026 early rounds ...")
    early_2026 = collect_2026_early_rounds()

    # Combine history with 2026 early data
    results = pd.concat([history, early_2026], ignore_index=True)
    results["constructor_canonical"] = results["constructor"].apply(_canonical)

    # Canonicalize historical standings
    standings_raw["constructor_canonical"] = standings_raw["constructor"].apply(_canonical)
    standings = (
        standings_raw.groupby(["year", "constructor_canonical"])["total_points"]
        .sum()
        .reset_index()
    )
    # Re-rank after canonical merge
    ranked_rows = []
    for year, grp in standings.groupby("year"):
        grp = grp.sort_values("total_points", ascending=False).reset_index(drop=True)
        grp["standing"] = grp.index + 1
        ranked_rows.append(grp)
    standings = pd.concat(ranked_rows, ignore_index=True)

    # Add points_share for momentum features
    season_total = standings.groupby("year")["total_points"].sum().rename("season_total")
    standings = standings.merge(season_total, on="year")
    standings["points_share"] = standings["total_points"] / standings["season_total"]

    # Add 2026 constructors as stub rows (no standing yet — so momentum features can look them up)
    constructors_2026 = results[results["year"] == PREDICT_YEAR]["constructor_canonical"].unique()
    stub_rows = pd.DataFrame({
        "year": PREDICT_YEAR,
        "constructor_canonical": constructors_2026,
        "total_points": np.nan,
        "standing": np.nan,
        "season_total": np.nan,
        "points_share": np.nan,
    })
    standings = pd.concat([standings, stub_rows], ignore_index=True)

    log.info("Computing features for %d ...", PREDICT_YEAR)
    early = _early_season_features(results)
    rc = _rule_change_features(standings)
    momentum = _momentum_features(standings)
    driver_q = _driver_quality_feature(results)

    # Build output for 2026 constructors only
    features = pd.DataFrame({
        "year": PREDICT_YEAR,
        "constructor_canonical": constructors_2026,
    })
    for df in [early, rc, momentum, driver_q]:
        features = features.merge(df, on=["year", "constructor_canonical"], how="left")

    # Apply rebrand dampening to match training-time feature values
    features = _apply_rebrand_dampening(features)
    features = features.rename(columns={"constructor_canonical": "constructor"})
    features["season_points_share"] = np.nan
    return features[features["year"] == PREDICT_YEAR].reset_index(drop=True)


def load_model_bundle(model_path: str) -> dict:
    """Load and return the pickled model bundle."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_standings(bundle: dict, features_2026: pd.DataFrame) -> pd.DataFrame:
    """Run prediction and return a DataFrame ranked by predicted_points_share descending."""
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X = features_2026[feature_cols].copy()
    preds = model.predict(X)

    result = features_2026[["constructor"]].copy()
    result["predicted_points_share"] = preds
    result = result.sort_values("predicted_points_share", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def _print_predictions(predictions: pd.DataFrame, winner_name: str, rule_change_weight: float):
    """Print a formatted 2026 constructor standings prediction table."""
    print("\n" + "=" * 60)
    print("GRID PROPHET - 2026 CONSTRUCTOR CHAMPIONSHIP PREDICTION")
    print("=" * 60)
    print(f"Model: {winner_name}  |  Rule-change weight: {rule_change_weight:.1f}")
    print(f"Based on rounds 1-{EARLY_ROUNDS} early-season data")
    print()
    print(f"  {'Rank':<6} {'Constructor':<24} {'Predicted Share':>15}")
    print("  " + "-" * 48)
    for _, row in predictions.iterrows():
        print(f"  {int(row['rank']):<6} {row['constructor']:<24} {row['predicted_points_share']:>14.1%}")
    print("=" * 60 + "\n")


def main():
    import argparse
    import fastf1

    parser = argparse.ArgumentParser(description="Generate 2026 Grid Prophet predictions.")
    parser.add_argument(
        "--model", default=os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl"),
    )
    parser.add_argument(
        "--out", default=os.path.join(DATA_DIR, "predictions_2026.csv"),
    )
    args = parser.parse_args()

    cache_dir = os.path.join(DATA_DIR, "fastf1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    log.info("Loading model from %s ...", args.model)
    bundle = load_model_bundle(args.model)
    log.info("Model: %s  |  rule_change_weight: %.1f", bundle["winner_name"], bundle["rule_change_weight"])

    log.info("Building 2026 features ...")
    features_2026 = build_2026_features()
    log.info("2026 constructors: %s", sorted(features_2026["constructor"].tolist()))

    log.info("Running predictions ...")
    predictions = predict_standings(bundle, features_2026)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    predictions.to_csv(args.out, index=False)
    log.info("Predictions saved -> %s", args.out)

    _print_predictions(predictions, bundle["winner_name"], bundle["rule_change_weight"])


if __name__ == "__main__":
    main()
