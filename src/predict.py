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

    features = features.rename(columns={"constructor_canonical": "constructor"})
    features["season_points_share"] = np.nan
    return features[features["year"] == PREDICT_YEAR].reset_index(drop=True)
