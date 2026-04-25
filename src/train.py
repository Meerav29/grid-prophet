"""Train XGBoost model on historical seasons with rule-change-year weighting."""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
