"""
Validation tests for Grid Prophet data pipeline.
Run from the repo root:  python test.py
All checks print PASS or FAIL with a short description.
"""

import sys
import pandas as pd

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = 0


def check(label: str, condition: bool, detail: str = ""):
    global failures
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}]  {label}{suffix}")
    if not condition:
        failures += 1


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("\n--- Loading CSVs ---")
try:
    race = pd.read_csv("data/race_results.csv")
    print(f"  race_results.csv       {len(race):,} rows")
except FileNotFoundError:
    print(f"  [{FAIL}]  data/race_results.csv not found — run collect.py first")
    sys.exit(1)

try:
    standings = pd.read_csv("data/constructor_standings.csv")
    print(f"  constructor_standings.csv  {len(standings):,} rows")
except FileNotFoundError:
    print(f"  [{FAIL}]  data/constructor_standings.csv not found — run collect.py first")
    sys.exit(1)

try:
    f = pd.read_csv("data/features.csv")
    print(f"  features.csv           {len(f):,} rows, {len(f.columns)} cols")
except FileNotFoundError:
    print(f"  [{FAIL}]  data/features.csv not found — run features.py first")
    sys.exit(1)

# ---------------------------------------------------------------------------
# race_results checks
# ---------------------------------------------------------------------------

print("\n--- race_results.csv ---")

expected_years = list(range(2014, 2026))
actual_years = sorted(race["year"].unique())
check(
    "Covers 2014–2025",
    all(y in actual_years for y in expected_years),
    f"found years {actual_years[0]}–{actual_years[-1]}",
)

required_cols = ["year", "round", "driver", "constructor", "grid_position",
                 "finish_position", "classification", "points"]
missing = [c for c in required_cols if c not in race.columns]
check("Required columns present", len(missing) == 0, f"missing: {missing}" if missing else "")

check(
    "No null constructors",
    race["constructor"].isnull().sum() == 0,
    f"{race['constructor'].isnull().sum()} nulls",
)

check(
    "classification values are valid",
    race["classification"].isin(["Finished", "DNF", "DNS"]).all(),
    str(race[~race["classification"].isin(["Finished", "DNF", "DNS"])]["classification"].unique()),
)

check(
    "points are non-negative",
    (race["points"].fillna(0) >= 0).all(),
    f"{(race['points'] < 0).sum()} negative, {race['points'].isnull().sum()} NaN rows",
)

# ---------------------------------------------------------------------------
# constructor_standings checks
# ---------------------------------------------------------------------------

print("\n--- constructor_standings.csv ---")

check(
    "Covers 2014–2025",
    all(y in standings["year"].unique() for y in expected_years),
    f"found {sorted(standings['year'].unique())}",
)

# Each year should have exactly one rank-1 constructor
rank1_per_year = standings[standings["standing"] == 1].groupby("year").size()
check(
    "Exactly one champion per year",
    (rank1_per_year == 1).all(),
    str(rank1_per_year[rank1_per_year != 1].to_dict()),
)

# ---------------------------------------------------------------------------
# features.csv checks
# ---------------------------------------------------------------------------

print("\n--- features.csv ---")

check(
    "Covers 2014–2025",
    all(y in f["year"].unique() for y in expected_years),
    f"found {sorted(f['year'].unique())}",
)

feature_cols = [
    "early_points_share", "early_avg_finish",
    "is_rule_change_year", "rule_change_adaptation_score",
    "prev_year_points_share", "prev_year_standing", "constructor_win_rate_5yr",
    "avg_driver_career_points_per_race", "season_points_share",
]
missing_feat = [c for c in feature_cols if c not in f.columns]
check("All feature columns present", len(missing_feat) == 0, f"missing: {missing_feat}")

# Points shares must sum to ~1.0 per season
share_sums = f.groupby("year")["season_points_share"].sum()
bad_years = share_sums[~share_sums.between(0.99, 1.01)]
check(
    "season_points_share sums to ~1.0 per year",
    len(bad_years) == 0,
    f"bad years: {bad_years.to_dict()}" if len(bad_years) else "",
)

# early_points_share must sum to ~1.0 per season
early_sums = f.groupby("year")["early_points_share"].sum()
bad_early = early_sums[~early_sums.between(0.99, 1.01)]
check(
    "early_points_share sums to ~1.0 per year",
    len(bad_early) == 0,
    f"bad years: {bad_early.to_dict()}" if len(bad_early) else "",
)

# 2014 is the first year — prev_year features must all be NaN
prev_cols = ["prev_year_points_share", "prev_year_standing"]
for col in prev_cols:
    val_2014 = f[f["year"] == 2014][col]
    check(f"{col} is NaN for 2014", val_2014.isnull().all())

# Rule-change years flagged correctly within training range
rc_years_in_data = set(f[f["is_rule_change_year"] == 1]["year"].unique())
expected_rc = {y for y in [2014, 2017, 2022] if y in f["year"].values}
check(
    "Rule-change years flagged correctly",
    rc_years_in_data == expected_rc,
    f"expected {sorted(expected_rc)}, got {sorted(rc_years_in_data)}",
)

# Lineage continuity — RB should have data back to 2014 (as Toro Rosso)
rb_years = set(f[f["constructor"] == "RB"]["year"].unique())
check(
    "RB lineage goes back to 2014",
    2014 in rb_years,
    f"earliest RB year: {min(rb_years) if rb_years else 'N/A'}",
)

# Mercedes should lead points_share in 2014–2016
for yr in [2014, 2015, 2016]:
    if yr not in f["year"].values:
        continue
    yr_data = f[f["year"] == yr].sort_values("season_points_share", ascending=False)
    top = yr_data.iloc[0]["constructor"] if len(yr_data) else None
    check(f"Mercedes leads season_points_share in {yr}", top == "Mercedes", f"top: {top}")

# No NaNs in early-season features (every constructor raced rounds 1-2)
null_early = f[["early_points_share", "early_avg_finish"]].isnull().sum()
check(
    "No NaNs in early_points_share / early_avg_finish",
    null_early.sum() == 0,
    str(null_early[null_early > 0].to_dict()),
)

# avg_driver_career_points_per_race must be NaN in 2014 (no prior data)
val_2014_drv = f[f["year"] == 2014]["avg_driver_career_points_per_race"]
check("avg_driver_career_points_per_race is NaN for 2014", val_2014_drv.isnull().all())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if failures == 0:
    print(f"All checks passed.\n")
else:
    print(f"{failures} check(s) FAILED.\n")
    sys.exit(1)
