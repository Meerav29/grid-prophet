"""Build data/driver_meta.csv: debut season per driver, for the rookie prior (spec sec 3.5)."""

import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    driver_rounds = pd.read_csv(os.path.join(DATA_DIR, "driver_rounds.csv"))
    legacy_path = os.path.join(DATA_DIR, "race_results.csv")

    frames = [driver_rounds[["season", "driver", "abbreviation"]].rename(columns={"season": "year"})]
    if os.path.exists(legacy_path):
        legacy = pd.read_csv(legacy_path)
        frames.append(legacy[["year", "driver", "abbreviation"]])

    combined = pd.concat(frames, ignore_index=True).dropna(subset=["abbreviation"])
    combined = combined[combined["abbreviation"] != ""]

    debut = (
        combined.groupby("abbreviation")
        .agg(driver=("driver", "first"), debut_season=("year", "min"))
        .reset_index()
        .sort_values("debut_season")
    )

    out_path = os.path.join(DATA_DIR, "driver_meta.csv")
    debut.to_csv(out_path, index=False)
    log.info("Saved %d drivers -> %s", len(debut), out_path)


if __name__ == "__main__":
    main()
