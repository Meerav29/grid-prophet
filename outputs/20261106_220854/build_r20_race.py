"""Add 2026 R20 Grand Prix rows to driver_rounds.csv from jolpica, matching collect_v2._build_race_rows.

FastF1 3.5.3 cannot parse this jolpica response (no `grid` field, no code for
Camara), so the rows are built here with the collector's own helpers. The grid
comes from the official formula1.com starting grid file.
"""
import json
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, "src")
import collect_v2 as cv  # noqa: E402
from jolpica_laps import jolpica_clean_air  # noqa: E402

S, R = 2026, 20
GRID = "outputs/20261106_220854/grid_penalties/grid_2026_20.csv"

url = f"https://api.jolpi.ca/ergast/f1/{S}/{R}/results.json?limit=100"
race = json.load(urllib.request.urlopen(url))["MRData"]["RaceTable"]["Races"][0]

dr = pd.read_csv(cv.DRIVER_ROUNDS_CSV)
q = dr[(dr.season == S) & (dr["round"] == R) & (dr.session_type == "Q")]
assert len(q) == 22 and not ((dr.season == S) & (dr["round"] == R) & (dr.session_type == "R")).any()
meta = q.set_index("abbreviation")
by_driver_id = {"rafael_camara": "CAM"}
grid = dict(pd.read_csv(GRID).values)

dnf_map = cv._load_dnf_map()
pace = jolpica_clean_air(S, R)
pace = {by_driver_id.get(k, k): v for k, v in pace.items()}
medians = [v[0] for v in pace.values() if v[0] is not None]
bests = [v[1] for v in pace.values() if v[1] is not None]
best_med = min(medians) if medians else None   # jolpica laps not published yet -> blank pace
best_best = min(bests) if bests else None

rows = []
for x in race["Results"]:
    code = x["Driver"].get("code") or by_driver_id[x["Driver"]["driverId"]]
    status = x["status"]
    category = cv._categorize_status(status, dnf_map)
    med, best = pace.get(code, (None, None))
    m = meta.loc[code]
    rows.append({
        "season": S, "round": R, "event_name": m["event_name"], "circuit_type": m["circuit_type"],
        "session_type": "R", "driver": m["driver"], "abbreviation": code, "team": m["team"],
        "grid_position": float(grid[code]), "finish_position": float(x["position"]),
        "classified": status.upper() not in ("DNS", "DID NOT START", "DID NOT QUALIFY"),
        "status": status, "dnf_cause": None if category == "classified" else category,
        "q_segment": None, "best_lap_s": None, "gap_to_best_s": None,
        "best_clean_air_lap_s": best,
        "gap_to_winner_best_clean_air_s": None if best is None else best - best_best,
        "median_clean_air_lap_s": med,
        "gap_to_winner_median_clean_air_s": None if med is None else med - best_med,
        "points": float(x["points"]), "is_wet": None, "air_temp_c": None,
    })

new = pd.DataFrame(rows, columns=cv.COLUMNS)
assert len(new) == 22 and set(new.abbreviation) == set(q.abbreviation)
pd.concat([dr, new], ignore_index=True).to_csv(cv.DRIVER_ROUNDS_CSV, index=False)
print(new[["abbreviation", "grid_position", "finish_position", "status", "dnf_cause", "points",
           "gap_to_winner_median_clean_air_s"]].to_string(index=False))
