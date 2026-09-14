"""Phase 0 data-quality validation for data/driver_rounds.csv.

Checks called out in the v2 spec (docs/grid-prophet-v2-spec.md sec 8, Phase 0):
  1. Coverage: all expected (season, round) pairs present, no unmapped DNF statuses.
  2. Clean-air lap filter sanity: median clean-air race pace should correlate
     strongly (~0.8+) with qualifying pace, and top-3 by clean-air pace should
     usually match the actual podium.

Writes a plain-text report to data/driver_rounds_validation.md.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "driver_rounds.csv"))
    lines = []

    lines.append("# driver_rounds.csv validation report\n")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Seasons covered: {sorted(df['season'].unique())}")
    lines.append(f"Session type counts:\n{df['session_type'].value_counts().to_string()}\n")

    # -- Coverage: rounds per season --
    rounds_per_season = df.groupby("season")["round"].nunique().sort_index()
    lines.append("## Rounds collected per season\n")
    lines.append(rounds_per_season.to_string())
    lines.append("")

    # -- DNF status mapping coverage --
    race_rows = df[df["session_type"].isin(["R", "S"])]
    unmapped = race_rows[(race_rows["classified"] == False) & (race_rows["dnf_cause"].isna())]  # noqa: E712
    lines.append("## DNF status mapping\n")
    lines.append(f"Unclassified rows with no dnf_cause (should be 0 or explainable): {len(unmapped)}")
    lines.append(f"dnf_cause distribution:\n{race_rows['dnf_cause'].value_counts(dropna=False).to_string()}\n")

    review_map = pd.read_csv(os.path.join(DATA_DIR, "dnf_status_map.csv"))
    flagged = review_map[review_map["needs_review"] == True]  # noqa: E712
    flagged_statuses = set(flagged["status"])
    flagged_rows = race_rows[race_rows["status"].isin(flagged_statuses)]
    lines.append(f"Rows whose status is flagged needs_review in dnf_status_map.csv: {len(flagged_rows)} "
                 f"({len(flagged_rows) / max(len(race_rows), 1):.1%} of race/sprint rows)")
    lines.append(flagged_rows["status"].value_counts().to_string() if not flagged_rows.empty else "(none)")
    lines.append("")

    # -- Clean-air lap filter validation --
    lines.append("## Clean-air lap filter validation\n")

    q = df[df["session_type"] == "Q"][["season", "round", "abbreviation", "gap_to_best_s"]].rename(
        columns={"gap_to_best_s": "quali_gap_s"}
    )
    r = df[df["session_type"] == "R"][
        ["season", "round", "abbreviation", "gap_to_winner_best_clean_air_s",
         "gap_to_winner_median_clean_air_s", "finish_position", "classified", "is_wet"]
    ]
    merged = q.merge(r, on=["season", "round", "abbreviation"], how="inner")

    for label, col in [("fastest clean-air lap (primary)", "gap_to_winner_best_clean_air_s"),
                        ("median clean-air lap (diagnostic)", "gap_to_winner_median_clean_air_s")]:
        pairs = merged.dropna(subset=["quali_gap_s", col])
        if len(pairs) <= 10:
            lines.append(f"## {label}: not enough paired rows\n")
            continue
        pear_r, pear_p = pearsonr(pairs["quali_gap_s"], pairs[col])
        spear_r, spear_p = spearmanr(pairs["quali_gap_s"], pairs[col])
        lines.append(f"### {label} -- all sessions (n={len(pairs)})")
        lines.append(f"Pearson r = {pear_r:.3f} (p={pear_p:.2e}), Spearman r = {spear_r:.3f} (p={spear_p:.2e})")

        dry = pairs[(~pairs["is_wet"]) & (pairs["classified"])]
        if len(dry) > 10:
            pear_r2, _ = pearsonr(dry["quali_gap_s"], dry[col])
            spear_r2, _ = spearmanr(dry["quali_gap_s"], dry[col])
            lines.append(f"Dry + classified only (n={len(dry)}): Pearson r = {pear_r2:.3f}, Spearman r = {spear_r2:.3f}")
            lines.append(f"Target from spec: ~0.8+ -- {'PASS' if spear_r2 >= 0.75 else 'BELOW TARGET'}\n")
        else:
            lines.append("")

    # -- Top-3 clean-air pace vs actual podium --
    matches, total = 0, 0
    for (season, rnd), g in df[df["session_type"] == "R"].groupby(["season", "round"]):
        g = g.dropna(subset=["best_clean_air_lap_s", "finish_position"])
        if len(g) < 5:
            continue
        pace_top3 = set(g.nsmallest(3, "best_clean_air_lap_s")["abbreviation"])
        actual_podium = set(g[g["finish_position"] <= 3]["abbreviation"])
        overlap = len(pace_top3 & actual_podium)
        matches += overlap
        total += 3

    if total > 0:
        lines.append(f"Top-3 clean-air pace vs actual podium overlap: {matches}/{total} driver-slots "
                     f"({matches / total:.1%})\n")
    else:
        lines.append("Not enough race rows to compute podium overlap.\n")

    # -- Circuit type coverage --
    lines.append("## Circuit type coverage\n")
    lines.append(df.groupby("circuit_type")["event_name"].nunique().to_string())
    lines.append("")

    report = "\n".join(str(x) for x in lines)
    out_path = os.path.join(DATA_DIR, "driver_rounds_validation.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
