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
