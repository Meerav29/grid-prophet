"""Pull historical race results from the FastF1 API and save to CSV."""

import argparse
import logging
import os
import sys

import fastf1
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR = os.path.join(DATA_DIR, "fastf1_cache")


def collect_race_results(start_year: int, end_year: int) -> pd.DataFrame:
    """Return a DataFrame of per-driver race results for start_year..end_year."""
    rows = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            log.error("Failed to get schedule for %d: %s", year, exc)
            continue

        race_rounds = schedule[schedule["EventFormat"].str.lower() != "testing"]

        for _, event in race_rounds.iterrows():
            round_num = int(event["RoundNumber"])
            event_name = event.get("EventName", f"Round {round_num}")

            log.info("Collecting %d Round %d — %s ...", year, round_num, event_name)

            try:
                session = fastf1.get_session(year, round_num, "R")
                session.load(telemetry=False, weather=False, messages=False)
            except Exception as exc:
                log.warning(
                    "  Skipping %d Round %d (%s): %s", year, round_num, event_name, exc
                )
                continue

            results = session.results
            if results is None or results.empty:
                log.warning("  No results for %d Round %d — skipping.", year, round_num)
                continue

            for _, driver in results.iterrows():
                status = str(driver.get("Status", ""))
                if status.upper().startswith("DNS"):
                    classification = "DNS"
                elif status.upper() == "FINISHED" or str(driver.get("Position", "")).isdigit():
                    classification = "Finished"
                else:
                    classification = "DNF"

                rows.append(
                    {
                        "year": year,
                        "round": round_num,
                        "event_name": event_name,
                        "driver": driver.get("FullName", ""),
                        "abbreviation": driver.get("Abbreviation", ""),
                        "constructor": driver.get("TeamName", ""),
                        "grid_position": driver.get("GridPosition", None),
                        "finish_position": driver.get("Position", None),
                        "classification": classification,
                        "points": driver.get("Points", 0.0),
                    }
                )

    return pd.DataFrame(rows)


def derive_constructor_standings(race_results: pd.DataFrame) -> pd.DataFrame:
    """Derive final constructor standings by summing points per season."""
    if race_results.empty:
        return pd.DataFrame(
            columns=["year", "constructor", "total_points", "standing"]
        )

    season_points = (
        race_results.groupby(["year", "constructor"])["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "total_points"})
    )

    standings_rows = []
    for year, group in season_points.groupby("year"):
        ranked = group.sort_values("total_points", ascending=False).reset_index(drop=True)
        ranked["standing"] = ranked.index + 1
        ranked["year"] = year
        standings_rows.append(ranked)

    return pd.concat(standings_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Collect F1 race results via FastF1.")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    log.info(
        "Collecting F1 race results %d–%d ...", args.start_year, args.end_year
    )
    race_results = collect_race_results(args.start_year, args.end_year)

    if race_results.empty:
        log.error("No race results collected. Exiting.")
        sys.exit(1)

    race_csv = os.path.join(DATA_DIR, "race_results.csv")
    race_results.to_csv(race_csv, index=False)
    log.info("Saved %d rows → %s", len(race_results), race_csv)

    standings = derive_constructor_standings(race_results)
    standings_csv = os.path.join(DATA_DIR, "constructor_standings.csv")
    standings.to_csv(standings_csv, index=False)
    log.info(
        "Saved %d constructor-season standings → %s", len(standings), standings_csv
    )


if __name__ == "__main__":
    main()
