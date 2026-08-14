"""Build per-constructor, per-season feature matrix from raw race results."""

import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARLY_ROUNDS = 2

RULE_CHANGE_YEARS = [2009, 2014, 2017, 2022, 2026]

# New name → old lineage root where historical features should be dampened 50%
# on the FIRST season under the new identity.
MAJOR_REBRANDS = {
    "Audi": "Sauber",
}

# Maps every current/recent team name to the full set of names it has raced
# under (including itself). Used to unify stats across rebrands.
CONSTRUCTOR_LINEAGE = {
    "Alpine": ["Alpine", "Renault", "Lotus F1", "Lotus"],
    "RB": ["RB", "AlphaTauri", "Toro Rosso"],
    "Aston Martin": ["Aston Martin", "Racing Point", "Force India"],
    "Audi": ["Audi", "Kick Sauber", "Alfa Romeo", "Sauber"],
    # Stable identities — listed explicitly so normalisation works uniformly
    "Mercedes": ["Mercedes"],
    "Ferrari": ["Ferrari"],
    "Red Bull Racing": ["Red Bull Racing", "Red Bull"],
    "McLaren": ["McLaren"],
    "Williams": ["Williams"],
    "Haas F1 Team": ["Haas F1 Team", "Haas"],
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(name: str) -> str:
    """Return the canonical (current) constructor name for any historical alias."""
    for canonical, aliases in CONSTRUCTOR_LINEAGE.items():
        if name in aliases:
            return canonical
    return name  # unknown teams stay as-is


def _first_rebrand_season(canonical: str, results: pd.DataFrame) -> int | None:
    """
    Return the first year that `canonical` appears in the data under its
    current name, if it is a major rebrand. Otherwise return None.
    """
    if canonical not in MAJOR_REBRANDS:
        return None
    aliases = CONSTRUCTOR_LINEAGE.get(canonical, [canonical])
    years_as_current = results[results["constructor_canonical"] == canonical]["year"]
    if years_as_current.empty:
        return None
    return int(years_as_current.min())


def latest_completed_round(year: int) -> int:
    """Return the latest completed round number for `year` from FastF1.
    Falls back to 1 if no rounds of `year` have completed yet."""
    import fastf1
    import datetime

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    today = datetime.date.today()
    completed = schedule[schedule["EventDate"].dt.date < today]
    if completed.empty:
        return 1
    return int(completed["RoundNumber"].max())


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _teammate_h2h_rate(group: pd.DataFrame) -> float:
    """Fraction of races (within the window) where the alphabetically-first
    driver on the team beat their teammate(s) on finish position. Races
    where the team fielded fewer than 2 classified drivers are excluded
    from the denominator. NaN if no races qualify."""
    wins = 0
    total = 0
    for _, race in group.groupby("round"):
        drivers = sorted(race["driver"].unique())
        if len(drivers) < 2:
            continue
        first_driver = drivers[0]
        first_finish = race.loc[race["driver"] == first_driver, "finish_position"].min()
        best_finish = race["finish_position"].min()
        total += 1
        if first_finish == best_finish:
            wins += 1
    return wins / total if total else np.nan


def _development_trend(group: pd.DataFrame) -> float:
    """Slope of avg finish position per round vs round number, over the
    window. Negative = improving (finish position numbers decreasing).
    0.0 if the window contains fewer than 2 rounds with data."""
    per_round = group.groupby("round")["finish_position"].mean().dropna()
    if len(per_round) < 2:
        return 0.0
    x = per_round.index.values.astype(float)
    y = per_round.values.astype(float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _early_season_features(results: pd.DataFrame, early_rounds: int = EARLY_ROUNDS) -> pd.DataFrame:
    """
    early_points_share            — team's share of all points from rounds 1..early_rounds
    early_avg_finish               — average finishing position in those rounds
    constructor_dnf_rate           — fraction of the team's entries classified DNF
    avg_grid_to_finish_delta       — mean(grid_position - finish_position), classified entries only
    teammate_head_to_head          — fraction of races where the alphabetically-first-named
                                      teammate beat the other teammate(s) on finish position
    development_trend              — slope of avg finish position per round (negative = improving)
    """
    early = results[results["round"] <= early_rounds].copy()

    total_early_points = early.groupby("year")["points"].sum().rename("total_early_pts")

    team_early = (
        early.groupby(["year", "constructor_canonical"])
        .agg(team_early_pts=("points", "sum"), team_early_finish=("finish_position", "mean"))
        .reset_index()
    )
    team_early = team_early.merge(total_early_points, on="year")
    team_early["early_points_share"] = (
        team_early["team_early_pts"] / team_early["total_early_pts"]
    )
    team_early["early_avg_finish"] = team_early["team_early_finish"]

    dnf_rate = (
        early.groupby(["year", "constructor_canonical"])["classification"]
        .apply(lambda s: (s == "DNF").mean())
        .rename("constructor_dnf_rate")
        .reset_index()
    )

    classified = early.dropna(subset=["grid_position", "finish_position"]).copy()
    classified["grid_to_finish_delta"] = classified["grid_position"] - classified["finish_position"]
    grid_delta = (
        classified.groupby(["year", "constructor_canonical"])["grid_to_finish_delta"]
        .mean()
        .rename("avg_grid_to_finish_delta")
        .reset_index()
    )

    h2h = (
        classified.groupby(["year", "constructor_canonical"])
        .apply(_teammate_h2h_rate, include_groups=False)
        .rename("teammate_head_to_head")
        .reset_index()
    )

    trend = (
        early.groupby(["year", "constructor_canonical"])
        .apply(_development_trend, include_groups=False)
        .rename("development_trend")
        .reset_index()
    )

    team_early = team_early.merge(dnf_rate, on=["year", "constructor_canonical"], how="left")
    team_early = team_early.merge(grid_delta, on=["year", "constructor_canonical"], how="left")
    team_early = team_early.merge(h2h, on=["year", "constructor_canonical"], how="left")
    team_early = team_early.merge(trend, on=["year", "constructor_canonical"], how="left")

    return team_early[[
        "year", "constructor_canonical",
        "early_points_share", "early_avg_finish",
        "constructor_dnf_rate", "avg_grid_to_finish_delta",
        "teammate_head_to_head", "development_trend",
    ]]


def _rule_change_features(standings: pd.DataFrame) -> pd.DataFrame:
    """
    is_rule_change_year          — binary flag
    rule_change_adaptation_score — avg standing shift in prior rule-change years
                                   (negative = improved). 0 if no history.
    """
    rows = []
    constructors = standings["constructor_canonical"].unique()
    years = sorted(standings["year"].unique())

    for constructor in constructors:
        cs = standings[standings["constructor_canonical"] == constructor].set_index("year")

        for year in years:
            is_rc = int(year in RULE_CHANGE_YEARS)

            # Prior rule-change years that exist in our data
            prior_rc_years = [y for y in RULE_CHANGE_YEARS if y < year]
            shifts = []
            for rc_year in prior_rc_years:
                if rc_year in cs.index and (rc_year - 1) in cs.index:
                    shift = cs.loc[rc_year, "standing"] - cs.loc[rc_year - 1, "standing"]
                    shifts.append(shift)

            adaptation_score = float(np.mean(shifts)) if shifts else 0.0

            rows.append(
                {
                    "year": year,
                    "constructor_canonical": constructor,
                    "is_rule_change_year": is_rc,
                    "rule_change_adaptation_score": adaptation_score,
                }
            )

    return pd.DataFrame(rows)


def _momentum_features(standings: pd.DataFrame) -> pd.DataFrame:
    """
    prev_year_points_share  — points share from previous season
    prev_year_standing      — championship position from previous season
    constructor_win_rate_5yr — fraction of last 5 seasons with top-3 finish
    """
    rows = []
    constructors = standings["constructor_canonical"].unique()
    years = sorted(standings["year"].unique())

    for constructor in constructors:
        cs = standings[standings["constructor_canonical"] == constructor].set_index("year")

        for year in years:
            prev_year = year - 1
            prev_pts_share = cs.loc[prev_year, "points_share"] if prev_year in cs.index else np.nan
            prev_standing = cs.loc[prev_year, "standing"] if prev_year in cs.index else np.nan

            lookback = [y for y in range(year - 5, year) if y in cs.index]
            if lookback:
                top3_count = sum(1 for y in lookback if cs.loc[y, "standing"] <= 3)
                win_rate_5yr = top3_count / len(lookback)
            else:
                win_rate_5yr = np.nan

            rows.append(
                {
                    "year": year,
                    "constructor_canonical": constructor,
                    "prev_year_points_share": prev_pts_share,
                    "prev_year_standing": prev_standing,
                    "constructor_win_rate_5yr": win_rate_5yr,
                }
            )

    return pd.DataFrame(rows)


def _driver_quality_feature(results: pd.DataFrame) -> pd.DataFrame:
    """
    avg_driver_career_points_per_race — average career PPR for the team's
    drivers in a given season, computed only from PRIOR seasons (no leakage).
    """
    rows = []
    years = sorted(results["year"].unique())

    for year in years:
        # Career stats use only data strictly before this season
        prior = results[results["year"] < year]
        if prior.empty:
            driver_ppr = pd.Series(dtype=float)
        else:
            driver_stats = (
                prior.groupby("driver")
                .agg(total_pts=("points", "sum"), races=("round", "count"))
                .reset_index()
            )
            driver_stats["ppr"] = driver_stats["total_pts"] / driver_stats["races"]
            driver_ppr = driver_stats.set_index("driver")["ppr"]

        # Current season drivers per constructor
        season = results[results["year"] == year]
        for constructor, grp in season.groupby("constructor_canonical"):
            drivers = grp["driver"].unique()
            pprs = [driver_ppr.get(d, np.nan) for d in drivers]
            valid = [p for p in pprs if not np.isnan(p)]
            avg_ppr = float(np.mean(valid)) if valid else np.nan

            rows.append(
                {
                    "year": year,
                    "constructor_canonical": constructor,
                    "avg_driver_career_points_per_race": avg_ppr,
                }
            )

    return pd.DataFrame(rows)


def _season_points_share(standings: pd.DataFrame) -> pd.DataFrame:
    """Target variable: each constructor's share of all points that season."""
    df = standings.copy()
    df["season_points_share"] = df["points_share"]
    return df[["year", "constructor_canonical", "season_points_share"]]


def _apply_rebrand_dampening(features: pd.DataFrame) -> pd.DataFrame:
    """Multiply historically-derived features by 0.5 for major rebrands in
    their first season under the new identity."""
    historical_cols = [
        "rule_change_adaptation_score",
        "prev_year_points_share",
        "prev_year_standing",
        "constructor_win_rate_5yr",
    ]
    for new_name in MAJOR_REBRANDS:
        first_season = features.loc[
            features["constructor_canonical"] == new_name, "year"
        ].min()
        if pd.isna(first_season):
            continue
        mask = (features["constructor_canonical"] == new_name) & (
            features["year"] == first_season
        )
        for col in historical_cols:
            if col in features.columns:
                features.loc[mask, col] = features.loc[mask, col] * 0.5
        log.info(
            "Applied 50%% rebrand dampening to %s in %d", new_name, int(first_season)
        )
    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_features(early_rounds: int | None = None) -> pd.DataFrame:
    effective_early_rounds = early_rounds if early_rounds is not None else EARLY_ROUNDS

    race_csv = os.path.join(DATA_DIR, "race_results.csv")
    standings_csv = os.path.join(DATA_DIR, "constructor_standings.csv")

    log.info("Loading %s ...", race_csv)
    results = pd.read_csv(race_csv)

    log.info("Loading %s ...", standings_csv)
    standings_raw = pd.read_csv(standings_csv)

    # Normalise constructor names to canonical lineage names
    results["constructor_canonical"] = results["constructor"].apply(_canonical)
    standings_raw["constructor_canonical"] = standings_raw["constructor"].apply(_canonical)

    # Re-aggregate standings using canonical names (merges e.g. Toro Rosso + AlphaTauri)
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

    # Compute season total for points_share used in momentum features
    season_total = standings.groupby("year")["total_points"].sum().rename("season_total")
    standings = standings.merge(season_total, on="year")
    standings["points_share"] = standings["total_points"] / standings["season_total"]

    log.info("Computing early-season features (window: rounds 1-%d) ...", effective_early_rounds)
    early = _early_season_features(results, early_rounds=effective_early_rounds)

    log.info("Computing rule-change features ...")
    rc = _rule_change_features(standings)

    log.info("Computing momentum features ...")
    momentum = _momentum_features(standings)

    log.info("Computing driver quality feature ...")
    driver_q = _driver_quality_feature(results)

    log.info("Computing target variable ...")
    target = _season_points_share(standings)

    # Merge everything
    base = standings[["year", "constructor_canonical"]].drop_duplicates()
    features = base.copy()
    for df in [early, rc, momentum, driver_q, target]:
        features = features.merge(df, on=["year", "constructor_canonical"], how="left")

    # Apply rebrand dampening
    features = _apply_rebrand_dampening(features)

    # Rename for clarity
    features = features.rename(columns={"constructor_canonical": "constructor"})

    log.info("Feature matrix shape: %s", features.shape)
    return features


def main():
    parser = argparse.ArgumentParser(description="Build Grid Prophet feature matrix.")
    parser.add_argument(
        "--early-rounds", type=int, default=None, dest="early_rounds",
        help="Override the early-season round window (default: EARLY_ROUNDS constant)",
    )
    args = parser.parse_args()

    features = build_features(early_rounds=args.early_rounds)
    out_path = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    log.info("Saved features → %s", out_path)
    log.info("\n%s", features.head(20).to_string())


if __name__ == "__main__":
    main()
