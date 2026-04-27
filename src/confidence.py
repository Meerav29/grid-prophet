"""Bootstrap confidence intervals for 2026 constructor championship predictions."""

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

log = logging.getLogger(__name__)


def bootstrap_confidence_intervals(
    bundle: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features_2026: pd.DataFrame,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Return DataFrame with columns: constructor, ci_low, ci_high, ci_half.

    Resamples training rows with replacement n_bootstrap times, retrains the
    model pipeline from bundle, predicts on features_2026, then computes the
    10th and 90th percentiles of the prediction distribution per constructor.
    """
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    constructors = features_2026["constructor"].values
    X_pred = features_2026[feature_cols].values

    all_preds = np.empty((n_bootstrap, len(constructors)))
    n_train = len(X_train)

    for i in range(n_bootstrap):
        idx = np.random.randint(0, n_train, size=n_train)
        X_boot = X_train.iloc[idx].copy()
        y_boot = y_train.iloc[idx].copy()
        boot_model = clone(model)
        boot_model.fit(X_boot, y_boot)
        all_preds[i] = boot_model.predict(X_pred)
        if (i + 1) % 100 == 0:
            log.info("  Bootstrap iteration %d/%d", i + 1, n_bootstrap)

    ci_low = np.percentile(all_preds, 10, axis=0)
    ci_high = np.percentile(all_preds, 90, axis=0)
    ci_half = (ci_high - ci_low) / 2

    return pd.DataFrame({
        "constructor": constructors,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half": ci_half,
    })
