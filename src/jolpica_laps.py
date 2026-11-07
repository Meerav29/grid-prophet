"""Clean-air race pace from jolpica (Ergast-compatible) lap times.

The fallback for rounds where FastF1 has results but no laps -- since late
2026 F1 live timing returns 401 without an F1TV subscription token, and
FastF1 then loads results from Ergast and nothing else.

jolpica carries per-lap times and positions and the pit-stop laps, but not
FastF1's TrackStatus or IsAccurate. They are approximated:

- Safety car / VSC: a lap whose field-median time is more than
  `SLOW_FIELD_RATIO` x the race's median lap is neutralised; it and the lap
  after it (the restart) are dropped for everyone.
- Inaccurate / damaged laps: a driver's lap more than `SLOW_DRIVER_RATIO` x
  that lap's field median is dropped.

Against FastF1 on 2026 R13-R17, the median clean-air lap (what the pace
fit reads) is within 0.001-0.12 s per round (median absolute difference)
and the median gap to the best correlates 0.91-1.00. The fastest clean-air
lap is a weaker match (up to 1.9 s off for single drivers), because one
lap is sensitive to which laps the proxies drop. Weather is not
available from jolpica, so a fallback round's `is_wet` stays unknown.
"""

import json
import time
import urllib.request

import numpy as np
import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100
CLEAN_AIR_GAP_THRESHOLD_S = 1.5   # same as collect_v2
SLOW_FIELD_RATIO = 1.07
SLOW_DRIVER_RATIO = 1.10


def _http_get(path: str, offset: int) -> dict:
    url = f"{BASE_URL}/{path}/?limit={PAGE_SIZE}&offset={offset}"
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as exc:   # rate limit (429) or transient network error
            last = exc
            time.sleep(2 + 2 * attempt)
    raise OSError(f"jolpica request failed after retries: {url}: {last}")


def _pages(path: str, get_json) -> list[dict]:
    out, offset = [], 0
    while True:
        page = get_json(path, offset)["MRData"]
        out.append(page)
        offset += PAGE_SIZE
        if offset >= int(page["total"]):
            return out


def _seconds(t: str) -> float:
    if ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    return float(t)


def race_laps(season: int, rnd: int, get_json=_http_get) -> tuple[pd.DataFrame, set]:
    """(laps[lap, code, pos, t], pit stops {(code, lap)}) for a Grand Prix."""
    codes = {}
    for page in _pages(f"{season}/{rnd}/drivers", get_json):
        for d in page["DriverTable"]["Drivers"]:
            codes[d["driverId"]] = d.get("code") or d["driverId"]

    rows = []
    for page in _pages(f"{season}/{rnd}/laps", get_json):
        for race in page["RaceTable"]["Races"]:
            for lap in race["Laps"]:
                for t in lap["Timings"]:
                    rows.append((int(lap["number"]), codes.get(t["driverId"], t["driverId"]),
                                 int(t["position"]), _seconds(t["time"])))
    laps = pd.DataFrame(rows, columns=["lap", "code", "pos", "t"]).drop_duplicates(["lap", "code"])

    pits = set()
    for page in _pages(f"{season}/{rnd}/pitstops", get_json):
        for race in page["RaceTable"]["Races"]:
            for p in race.get("PitStops", []):
                pits.add((codes.get(p["driverId"], p["driverId"]), int(p["lap"])))
    return laps, pits


def clean_air_pace(laps: pd.DataFrame, pits: set,
                   gap_threshold: float = CLEAN_AIR_GAP_THRESHOLD_S) -> dict:
    """{code: (median, fastest)} clean-air lap time in seconds, (None, None) if none."""
    if laps.empty:
        return {}
    laps = laps.sort_values(["code", "lap"]).copy()
    laps["cum"] = laps.groupby("code")["t"].cumsum()
    ahead = laps[["lap", "pos", "cum"]].rename(columns={"cum": "cum_ahead"})
    ahead["pos"] += 1
    laps = laps.merge(ahead, on=["lap", "pos"], how="left")
    laps["gap"] = laps["cum"] - laps["cum_ahead"]   # NaN for the leader

    field_med = laps.groupby("lap")["t"].median()
    slow = set(field_med[field_med > SLOW_FIELD_RATIO * field_med.median()].index)
    slow |= {lap + 1 for lap in slow}
    pit_laps = pits | {(c, lap + 1) for c, lap in pits}
    is_pit = np.array([(c, lap) in pit_laps for c, lap in zip(laps["code"], laps["lap"])], dtype=bool)

    mask = (
        (laps["lap"] > 1)
        & ~laps["lap"].isin(slow)
        & ~is_pit
        & (laps["t"] <= SLOW_DRIVER_RATIO * laps["lap"].map(field_med))
        & (laps["gap"].isna() | (laps["gap"] > gap_threshold))
    )
    clean = laps[mask].groupby("code")["t"]
    med, best = clean.median(), clean.min()
    return {c: ((float(med[c]), float(best[c])) if c in med.index else (None, None))
            for c in laps["code"].unique()}


def jolpica_clean_air(season: int, rnd: int, get_json=_http_get) -> dict:
    laps, pits = race_laps(season, rnd, get_json=get_json)
    return clean_air_pace(laps, pits)
