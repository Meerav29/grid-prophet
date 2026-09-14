"""Phase 0 data layer: per-driver-per-round-per-session table for Grid Prophet v2.

Builds data/driver_rounds.csv -- one row per driver per session type (Q / S / R)
per round, seasons 2018-2026. See docs/grid-prophet-v2-spec.md section 5.
"""

import argparse
import logging
import os
import sys

import fastf1
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR = os.path.join(DATA_DIR, "fastf1_cache")
DRIVER_ROUNDS_CSV = os.path.join(DATA_DIR, "driver_rounds.csv")
CIRCUITS_CSV = os.path.join(DATA_DIR, "circuits.csv")
DNF_MAP_CSV = os.path.join(DATA_DIR, "dnf_status_map.csv")

CLEAN_AIR_GAP_THRESHOLD_S = 1.5

COLUMNS = [
    "season", "round", "event_name", "circuit_type", "session_type",
    "driver", "abbreviation", "team",
    "grid_position", "finish_position", "classified", "status", "dnf_cause",
    "q_segment", "best_lap_s", "gap_to_best_s",
    "best_clean_air_lap_s", "gap_to_winner_best_clean_air_s",
    "median_clean_air_lap_s", "gap_to_winner_median_clean_air_s",
    "points", "is_wet", "air_temp_c",
]


def _load_dnf_map() -> dict:
    df = pd.read_csv(DNF_MAP_CSV)
    return dict(zip(df["status"], df["category"]))


def _categorize_status(status: str, dnf_map: dict) -> str:
    if status in dnf_map:
        return dnf_map[status]
    if status.startswith("+") and "Lap" in status:
        return "classified"
    log.warning("  Unmapped status %r -- categorizing as 'other', add to dnf_status_map.csv", status)
    return "other"


def _add_gap_to_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    """Add a GapToAhead column: seconds to the car directly ahead on the same lap."""
    laps = laps.copy()
    laps["GapToAhead"] = np.nan
    for _, group in laps.groupby("LapNumber"):
        g = group.dropna(subset=["Position", "Time"]).sort_values("Position")
        if len(g) < 2:
            continue
        times = g["Time"].values
        idx = g.index.values
        for i in range(1, len(g)):
            gap_s = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
            laps.loc[idx[i], "GapToAhead"] = gap_s
    return laps


def _clean_air_pace(laps: pd.DataFrame, driver_code: str) -> tuple[float | None, float | None]:
    """(median, fastest) lap time in seconds for laps run in clean air by this driver.

    Median blends fuel loads and tire states across the whole stint; fastest is
    the single-lap analogue of a quali time and is what's actually used as the
    quali-comparable pace signal (see docs/grid-prophet-v2-spec.md sec 3.1 and
    data/driver_rounds_validation.md) -- median is kept only as a diagnostic.
    """
    d = laps[laps["Driver"] == driver_code]
    if d.empty:
        return None, None

    track_clear = d["TrackStatus"].fillna("1").apply(lambda s: set(str(s)) <= {"1"})
    no_pit = d["PitInTime"].isna() & d["PitOutTime"].isna()
    accurate = d["IsAccurate"].fillna(False)
    clean_gap = d["GapToAhead"].isna() | (d["GapToAhead"] > CLEAN_AIR_GAP_THRESHOLD_S)

    mask = track_clear & no_pit & accurate & clean_gap
    clean_laps = d.loc[mask, "LapTime"].dropna()
    if clean_laps.empty:
        return None, None
    return (
        clean_laps.median() / np.timedelta64(1, "s"),
        clean_laps.min() / np.timedelta64(1, "s"),
    )


def _circuit_lookup(event_name: str, circuits: pd.DataFrame) -> str:
    match = circuits[circuits["event_name"] == event_name]
    if match.empty:
        log.warning("  No circuit_type entry for %r -- defaulting to 'mixed'", event_name)
        return "mixed"
    return match.iloc[0]["circuit_type"]


def _weather_summary(session) -> tuple[bool, float | None]:
    try:
        wx = session.weather_data
        if wx is None or wx.empty:
            return False, None
        is_wet = bool((wx["Rainfall"] == True).any())  # noqa: E712
        air_temp = float(wx["AirTemp"].mean())
        return is_wet, air_temp
    except Exception:
        return False, None


def _build_quali_rows(year: int, rnd: int, event_name: str, circuit_type: str) -> list[dict]:
    rows = []
    try:
        session = fastf1.get_session(year, rnd, "Q")
        session.load(telemetry=False, weather=True, messages=False)
    except Exception as exc:
        log.warning("  No qualifying session for %d Round %d: %s", year, rnd, exc)
        return rows

    results = session.results
    if results is None or results.empty:
        return rows

    is_wet, air_temp = _weather_summary(session)
    pole_time = None
    for q_col in ("Q3", "Q2", "Q1"):
        col = results[q_col].dropna()
        if not col.empty:
            pole_time = col.min() / np.timedelta64(1, "s")
            break

    for _, r in results.iterrows():
        best = None
        q_segment = None
        for q_col in ("Q3", "Q2", "Q1"):
            t = r.get(q_col)
            if pd.notna(t):
                best = t / np.timedelta64(1, "s")
                q_segment = q_col
                break
        rows.append({
            "season": year, "round": rnd, "event_name": event_name, "circuit_type": circuit_type,
            "session_type": "Q",
            "driver": r.get("FullName", ""), "abbreviation": r.get("Abbreviation", ""),
            "team": r.get("TeamName", ""),
            "grid_position": None, "finish_position": r.get("Position"),
            "classified": best is not None, "status": None, "dnf_cause": None,
            "q_segment": q_segment, "best_lap_s": best,
            "gap_to_best_s": (best - pole_time) if (best is not None and pole_time is not None) else None,
            "best_clean_air_lap_s": None, "gap_to_winner_best_clean_air_s": None,
            "median_clean_air_lap_s": None, "gap_to_winner_median_clean_air_s": None,
            "points": 0.0, "is_wet": is_wet, "air_temp_c": air_temp,
        })
    return rows


def _build_race_rows(year: int, rnd: int, event_name: str, circuit_type: str,
                      session_code: str, dnf_map: dict) -> list[dict]:
    rows = []
    try:
        session = fastf1.get_session(year, rnd, session_code)
        session.load(telemetry=False, weather=True, messages=False)
    except Exception as exc:
        log.warning("  No %s session for %d Round %d: %s", session_code, year, rnd, exc)
        return rows

    results = session.results
    if results is None or results.empty:
        return rows

    # Defensive check: in a long-running process, a session occasionally comes back
    # with results populated but Status blank across the board (seen during a many-
    # hour backfill; a fresh short-lived process re-fetching the same cache entry
    # gets correct data, so this reflects process-level state drift, not bad source
    # data or a bad cache entry). Treat as a failed fetch rather than saving garbage.
    if results["Status"].isna().mean() > 0.5:
        log.warning("  %d Round %d %s: Status mostly blank (%d/%d) -- treating as failed fetch, will retry",
                    year, rnd, session_code, results["Status"].isna().sum(), len(results))
        return rows

    is_wet, air_temp = _weather_summary(session)

    try:
        laps = session.laps
        have_laps = laps is not None and not laps.empty
    except Exception as exc:
        log.warning("  No lap data available for %d Round %d %s: %s", year, rnd, session_code, exc)
        laps, have_laps = None, False
    if have_laps:
        laps = _add_gap_to_ahead(laps)

    clean_air = {}
    if have_laps:
        for code in results["Abbreviation"].dropna().unique():
            clean_air[code] = _clean_air_pace(laps, code)
        medians = [v[0] for v in clean_air.values() if v[0] is not None]
        bests = [v[1] for v in clean_air.values() if v[1] is not None]
        best_median_clean_air = min(medians) if medians else None
        best_of_best_clean_air = min(bests) if bests else None
    else:
        best_median_clean_air = None
        best_of_best_clean_air = None

    for _, r in results.iterrows():
        status = str(r.get("Status", ""))
        finish_pos = r.get("Position")
        classified = pd.notna(finish_pos) and status.upper() not in ("DNS", "DID NOT START", "DID NOT QUALIFY")
        category = _categorize_status(status, dnf_map)
        cause = None if category == "classified" else category
        code = r.get("Abbreviation", "")
        my_median, my_best = clean_air.get(code, (None, None))
        rows.append({
            "season": year, "round": rnd, "event_name": event_name, "circuit_type": circuit_type,
            "session_type": session_code,
            "driver": r.get("FullName", ""), "abbreviation": code,
            "team": r.get("TeamName", ""),
            "grid_position": r.get("GridPosition"), "finish_position": finish_pos,
            "classified": classified, "status": status, "dnf_cause": cause,
            "q_segment": None, "best_lap_s": None, "gap_to_best_s": None,
            "best_clean_air_lap_s": my_best,
            "gap_to_winner_best_clean_air_s": (my_best - best_of_best_clean_air) if (my_best is not None and best_of_best_clean_air is not None) else None,
            "median_clean_air_lap_s": my_median,
            "gap_to_winner_median_clean_air_s": (my_median - best_median_clean_air) if (my_median is not None and best_median_clean_air is not None) else None,
            "points": float(r.get("Points") or 0.0), "is_wet": is_wet, "air_temp_c": air_temp,
        })
    return rows


def build_driver_rounds(start_year: int, end_year: int, existing: pd.DataFrame) -> pd.DataFrame:
    """Collect rounds one at a time, saving to disk after each so a crash or
    rate-limit error partway through doesn't lose already-fetched rounds."""
    circuits = pd.read_csv(CIRCUITS_CSV)
    dnf_map = _load_dnf_map()
    combined = existing

    # A round only counts as "done" once it has both Q and R rows -- a round that
    # picked up Race rows from an older cache but never got Qualifying (e.g. after
    # a rate-limit interruption) must still be revisited on resume.
    done_rounds = set()
    if not existing.empty:
        have_q = set(zip(existing.loc[existing["session_type"] == "Q", "season"],
                          existing.loc[existing["session_type"] == "Q", "round"]))
        have_r = set(zip(existing.loc[existing["session_type"] == "R", "season"],
                          existing.loc[existing["session_type"] == "R", "round"]))
        done_rounds = have_q & have_r

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            log.error("Failed to get schedule for %d: %s", year, exc)
            continue

        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            if rnd == 0:
                continue
            if (year, rnd) in done_rounds:
                continue

            event_name = event.get("EventName", f"Round {rnd}")
            event_format = str(event.get("EventFormat", "")).lower()
            circuit_type = _circuit_lookup(event_name, circuits)

            log.info("Collecting %d Round %d — %s ...", year, rnd, event_name)

            try:
                round_rows = []
                round_rows += _build_quali_rows(year, rnd, event_name, circuit_type)
                if "sprint" in event_format:
                    round_rows += _build_race_rows(year, rnd, event_name, circuit_type, "S", dnf_map)
                round_rows += _build_race_rows(year, rnd, event_name, circuit_type, "R", dnf_map)
            except Exception as exc:
                log.error("  Failed on %d Round %d (%s): %s -- skipping round, progress so far is saved",
                          year, rnd, event_name, exc)
                continue

            if not round_rows:
                continue

            # Drop any partial rows already collected for this round (e.g. Race-only
            # rows left over from a rate-limit interruption) before re-adding it whole.
            if not combined.empty:
                combined = combined[~((combined["season"] == year) & (combined["round"] == rnd))]

            new_df = pd.DataFrame(round_rows, columns=COLUMNS)
            combined = pd.concat([combined, new_df], ignore_index=True) if not combined.empty else new_df
            combined.to_csv(DRIVER_ROUNDS_CSV, index=False)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Collect per-driver-per-round data via FastF1 (Grid Prophet v2).")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume", default=True,
        help="Re-fetch rounds already present in driver_rounds.csv.",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    existing = pd.DataFrame()
    if args.resume and os.path.exists(DRIVER_ROUNDS_CSV):
        existing = pd.read_csv(DRIVER_ROUNDS_CSV)
        log.info("Resuming: %d existing rows covering seasons %s", len(existing),
                  sorted(existing["season"].unique()) if not existing.empty else [])

    combined = build_driver_rounds(args.start_year, args.end_year, existing)

    if combined.empty:
        log.error("No rows collected. Exiting.")
        sys.exit(1)

    combined.to_csv(DRIVER_ROUNDS_CSV, index=False)
    log.info("Saved %d rows → %s", len(combined), DRIVER_ROUNDS_CSV)


if __name__ == "__main__":
    main()
