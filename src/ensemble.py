"""Ensemble model: blend full-data model with rule-change-years-only model."""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

log = logging.getLogger(__name__)

RC_YEARS = {2014, 2015, 2022, 2023}
ALPHA_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7]


def _train_rc_model(base_bundle: dict, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    """Train a copy of the base pipeline on rule-change years only."""
    rc_mask = meta["year"].isin(RC_YEARS)
    X_rc = X[rc_mask].copy()
    y_rc = y[rc_mask].copy()
    rc_model = clone(base_bundle["model"])
    rc_model.fit(X_rc, y_rc)
    return rc_model


def _tune_alpha(
    full_model,
    rc_model,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> float:
    """Find the blend alpha (full-data weight) via LOOCV on rule-change seasons."""
    rc_seasons = [s for s in meta["year"].unique() if s in {2014, 2022}]
    best_alpha = 0.5
    best_score = float("-inf")

    for alpha in ALPHA_CANDIDATES:
        spearmans = []
        for held_out in rc_seasons:
            train_mask = meta["year"] != held_out
            test_mask = meta["year"] == held_out
            X_train = X[train_mask].copy()
            y_train = y[train_mask].copy()
            X_test = X[test_mask].copy()
            y_test = y[test_mask].copy()

            fold_full = clone(full_model)
            fold_full.fit(X_train, y_train)
            rc_train_mask = train_mask & meta["year"].isin(RC_YEARS)
            if rc_train_mask.sum() < 2:
                continue
            fold_rc = clone(rc_model)
            fold_rc.fit(X[rc_train_mask].copy(), y[rc_train_mask].copy())

            preds = alpha * fold_full.predict(X_test) + (1 - alpha) * fold_rc.predict(X_test)
            if len(preds) >= 2:
                corr, _ = spearmanr(preds, y_test.values)
                spearmans.append(float(corr))

        avg = float(np.mean(spearmans)) if spearmans else float("-inf")
        log.info("  alpha=%.1f  avg Spearman on RC seasons: %.4f", alpha, avg)
        if avg > best_score:
            best_score = avg
            best_alpha = alpha

    return best_alpha


def build_ensemble(
    base_bundle: dict,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> dict:
    """Build and return an ensemble bundle.

    Trains a rule-change-only sub-model, tunes the blend weight alpha via
    LOOCV on rule-change seasons, and returns a bundle ready for prediction.
    """
    log.info("Training rule-change-only sub-model on years: %s", sorted(RC_YEARS))
    full_model = base_bundle["model"]
    rc_model = _train_rc_model(base_bundle, X, y, meta)

    log.info("Tuning ensemble blend weight (alpha) ...")
    alpha = _tune_alpha(full_model, rc_model, X, y, meta)
    log.info("Best alpha: %.1f", alpha)

    base_name = base_bundle["winner_name"]
    return {
        "full_model": full_model,
        "rc_model": rc_model,
        "alpha": alpha,
        "feature_cols": base_bundle["feature_cols"],
        "winner_name": f"Ensemble({base_name}, alpha={alpha:.1f})",
        "rule_change_weight": base_bundle["rule_change_weight"],
        "model": _EnsembleShim(full_model, rc_model, alpha),
    }


class _EnsembleShim:
    """Thin wrapper so predict_standings can call .predict() on the ensemble."""

    def __init__(self, full_model, rc_model, alpha: float):
        self.full_model = full_model
        self.rc_model = rc_model
        self.alpha = alpha

    def predict(self, X):
        return (
            self.alpha * self.full_model.predict(X)
            + (1 - self.alpha) * self.rc_model.predict(X)
        )


def ensemble_predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Run blended prediction given an ensemble bundle and feature matrix."""
    return bundle["model"].predict(X)


def save_ensemble(bundle: dict, path: str) -> None:
    """Pickle the ensemble bundle to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved ensemble bundle -> %s", path)
