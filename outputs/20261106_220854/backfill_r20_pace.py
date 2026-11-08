"""Fill 2026 R20 Grand Prix race pace in data/driver_rounds.csv from jolpica lap times.

The R20 race rows were built before jolpica published laps, so the four
clean-air pace columns are blank. Run from the repo root:

    python -P outputs/20261106_220854/backfill_r20_pace.py

Exit code 0 = pace written; 3 = jolpica has no laps yet (nothing changed);
1 = something unexpected (nothing changed).
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
import collect_v2 as cv  # noqa: E402
from jolpica_laps import jolpica_clean_air  # noqa: E402

S, R = 2026, 20
DRIVER_ID_TO_CODE = {"rafael_camara": "CAM"}  # jolpica has no code for Camara


def main() -> int:
    dr = pd.read_csv(cv.DRIVER_ROUNDS_CSV)
    mask = (dr.season == S) & (dr["round"] == R) & (dr.session_type == "R")
    if mask.sum() != 22:
        print(f"expected 22 R20 race rows, found {mask.sum()}")
        return 1
    if dr.loc[mask, "gap_to_winner_median_clean_air_s"].notna().any():
        print("R20 race pace already filled; nothing to do")
        return 0

    pace = jolpica_clean_air(S, R)
    pace = {DRIVER_ID_TO_CODE.get(k, k): v for k, v in pace.items()}
    medians = [v[0] for v in pace.values() if v[0] is not None]
    bests = [v[1] for v in pace.values() if v[1] is not None]
    if not medians:
        print("jolpica has no R20 lap times yet")
        return 3
    unknown = sorted(set(pace) - set(dr.loc[mask, "abbreviation"]))
    if unknown:
        print(f"jolpica codes not in the R20 rows: {unknown}")
        return 1

    best_med, best_best = min(medians), min(bests)
    for i in dr.index[mask]:
        med, best = pace.get(dr.at[i, "abbreviation"], (None, None))
        dr.at[i, "median_clean_air_lap_s"] = med
        dr.at[i, "best_clean_air_lap_s"] = best
        dr.at[i, "gap_to_winner_median_clean_air_s"] = None if med is None else med - best_med
        dr.at[i, "gap_to_winner_best_clean_air_s"] = None if best is None else best - best_best
    dr.to_csv(cv.DRIVER_ROUNDS_CSV, index=False)
    filled = dr.loc[mask, ["abbreviation", "finish_position", "gap_to_winner_median_clean_air_s"]]
    print(filled.sort_values("gap_to_winner_median_clean_air_s").to_string(index=False))
    print(f"pace filled for {len(medians)} of 22 drivers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
