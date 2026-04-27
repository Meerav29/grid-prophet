"""Generate visualisation charts for Grid Prophet."""

import logging
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RULE_CHANGE_YEARS = {2014, 2022}

_CONSTRUCTOR_COLORS = {
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "Red Bull Racing": "#0600EF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#005AFF",
    "RB": "#2B4562",
    "Haas F1 Team": "#FFFFFF",
    "Audi": "#C0C0C0",
    "Racing Bulls": "#2B4562",
    "Cadillac": "#004225",
}
_DEFAULT_COLOR = "#888888"


def _constructor_color(name: str) -> str:
    return _CONSTRUCTOR_COLORS.get(name, _DEFAULT_COLOR)


def plot_cv_accuracy(
    cv_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_dir: str = "plots/",
) -> None:
    """Grid of subplots: predicted vs actual rank per held-out season."""
    os.makedirs(out_dir, exist_ok=True)
    seasons = sorted(features_df["year"].unique())
    n = len(seasons)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    spearman_by_season = cv_df.groupby("season")["spearman"].mean()

    for i, season in enumerate(seasons):
        ax = axes[i]
        season_df = features_df[features_df["year"] == season]
        colors = [_constructor_color(c) for c in season_df["constructor"]]
        ax.scatter(season_df["actual_rank"], season_df["predicted_rank"],
                   c=colors, s=80, edgecolors="black", linewidths=0.5, zorder=3)
        max_rank = max(season_df["actual_rank"].max(), season_df["predicted_rank"].max()) + 1
        ax.plot([1, max_rank], [1, max_rank], "k--", linewidth=0.8, alpha=0.5)
        rho = spearman_by_season.get(season, float("nan"))
        ax.set_title(f"{season}  ρ={rho:.2f}", fontsize=10)
        ax.set_xlabel("Actual rank")
        ax.set_ylabel("Predicted rank")
        ax.invert_xaxis()
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Grid Prophet — CV Accuracy: Predicted vs Actual Rank", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "cv_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_spearman_by_season(cv_df: pd.DataFrame, out_dir: str = "plots/") -> None:
    """Bar chart of per-season Spearman correlation with rule-change years highlighted."""
    os.makedirs(out_dir, exist_ok=True)
    by_season = cv_df.groupby("season")["spearman"].mean().reset_index()
    by_season = by_season.sort_values("season")

    colors = [
        "#E8000D" if s in RULE_CHANGE_YEARS else "#4878CF"
        for s in by_season["season"]
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(by_season["season"].astype(str), by_season["spearman"], color=colors, edgecolor="black", linewidth=0.5)
    mean_rho = by_season["spearman"].mean()
    ax.axhline(mean_rho, color="black", linestyle="--", linewidth=1, label=f"Mean ρ = {mean_rho:.3f}")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E8000D", label="Rule-change year"),
        Patch(facecolor="#4878CF", label="Normal year"),
    ]
    ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color="black", linestyle="--", label=f"Mean ρ={mean_rho:.3f}")],
              loc="lower right")
    ax.set_xlabel("Season")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Grid Prophet — Leave-One-Season-Out CV: Spearman Correlation by Season")
    ax.set_ylim(-0.1, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = os.path.join(out_dir, "spearman_by_season.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_2026_predictions(predictions_df: pd.DataFrame, out_dir: str = "plots/") -> None:
    """Horizontal bar chart of 2026 predicted points share with 80% CI error bars."""
    os.makedirs(out_dir, exist_ok=True)
    df = predictions_df.sort_values("rank")
    constructors = df["constructor"].tolist()
    shares = df["predicted_points_share"].values
    colors = [_constructor_color(c) for c in constructors]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(constructors))
    ax.barh(y_pos, shares, color=colors, edgecolor="black", linewidth=0.5, height=0.6)

    if "ci_low" in df.columns and "ci_high" in df.columns:
        xerr_low = shares - df["ci_low"].values
        xerr_high = df["ci_high"].values - shares
        ax.errorbar(shares, y_pos, xerr=[xerr_low, xerr_high],
                    fmt="none", color="black", capsize=4, linewidth=1.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(constructors)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Championship Points Share")
    ax.set_title("Grid Prophet — 2026 Constructor Championship Prediction")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    out = os.path.join(out_dir, "predictions_2026.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_feature_importance(bundle: dict, out_dir: str = "plots/") -> None:
    """Horizontal bar chart of feature importances or Ridge coefficients."""
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = bundle["feature_cols"]
    last_step = bundle["model"].steps[-1][1]

    if hasattr(last_step, "feature_importances_"):
        values = last_step.feature_importances_
        label = "Feature Importance"
        title = "Grid Prophet — XGBoost Feature Importances"
    elif hasattr(last_step, "coef_"):
        values = np.abs(last_step.coef_)
        label = "|Coefficient|"
        title = "Grid Prophet — Ridge Regression |Coefficients|"
    else:
        log.warning("Model has neither feature_importances_ nor coef_; skipping plot.")
        return

    order = np.argsort(values)
    sorted_features = [feature_cols[i] for i in order]
    sorted_values = values[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(sorted_features, sorted_values, color="#4878CF", edgecolor="black", linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_title(title)
    plt.tight_layout()
    out = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_all(out_dir: str = "plots/") -> None:
    """Load CSVs and model bundle, then generate all four charts."""
    cv_path = os.path.join(DATA_DIR, "cv_results.csv")
    pred_path = os.path.join(DATA_DIR, "predictions_2026.csv")
    features_path = os.path.join(DATA_DIR, "features.csv")
    model_path = os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl")

    cv_df = pd.read_csv(cv_path)
    predictions_df = pd.read_csv(pred_path)
    features_df = pd.read_csv(features_path)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    train_df = features_df[features_df["season_points_share"].notna()].copy()
    train_df["actual_rank"] = train_df.groupby("year")["season_points_share"].rank(
        ascending=False, method="min"
    ).astype(int)
    feature_cols = bundle["feature_cols"]
    X_all = train_df[feature_cols]
    train_df["predicted_points_share"] = bundle["model"].predict(X_all)
    train_df["predicted_rank"] = train_df.groupby("year")["predicted_points_share"].rank(
        ascending=False, method="min"
    ).astype(int)

    plot_cv_accuracy(cv_df, train_df, out_dir=out_dir)
    plot_spearman_by_season(cv_df, out_dir=out_dir)
    plot_2026_predictions(predictions_df, out_dir=out_dir)
    plot_feature_importance(bundle, out_dir=out_dir)
    log.info("All charts saved to %s", out_dir)
