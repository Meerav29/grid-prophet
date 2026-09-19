"""Regenerate the small real `driver_rounds` fixture from a full collection.

The fixture exists so slice acceptance criteria of the form "runs end to end
on a completed round" are checkable in CI and by the autopilot routines.
`data/driver_rounds.csv` is gitignored (it is large and reproducible via
`collect_v2`), so without a committed sample no cloud checkout has a real
round to run against.

Run from the repo root on a machine that has the full table:

    python tests/fixtures/make_driver_rounds_fixture.py

It slices a few complete, conventional (non-sprint) rounds out of
`data/driver_rounds.csv` and writes them here. Nothing is synthesised: every
row is exactly what `collect_v2` produced from FastF1.

Kept deliberately small — a handful of rounds is enough to fit the pace model
and condition on one held-out round, and the file has to stay reviewable in a
diff.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "..", "..", "data", "driver_rounds.csv")
TARGET = os.path.join(HERE, "driver_rounds_sample.csv")

# Four conventional 2024 rounds: Bahrain, Saudi Arabia, Australia, Japan.
# 2024's first sprint was round 5 (China), so this window has no sprint rows
# and stays slice-1-shaped. Rounds 1-3 fit the model; round 4 is held out and
# conditioned on, which is what the post-quali path needs.
SEASON = 2024
ROUNDS = [1, 2, 3, 4]


def main() -> int:
    if not os.path.exists(SOURCE):
        print(f"No {SOURCE}. Run `python -m src.collect_v2` first.", file=sys.stderr)
        return 1

    df = pd.read_csv(SOURCE)
    sample = df[(df["season"] == SEASON) & (df["round"].isin(ROUNDS))
                & (df["session_type"].isin(["Q", "R"]))].copy()

    if sample.empty:
        print(f"No {SEASON} rounds {ROUNDS} in {SOURCE}.", file=sys.stderr)
        return 1

    missing = sorted(set(ROUNDS) - set(sample["round"].unique()))
    if missing:
        print(f"Rounds missing from the source table: {missing}", file=sys.stderr)
        return 1

    sprint = df[(df["season"] == SEASON) & (df["round"].isin(ROUNDS))
                & (df["session_type"] == "S")]
    if not sprint.empty:
        print(f"Unexpected sprint rows in {SEASON} rounds {ROUNDS}; pick a different "
              "window or teach the fixture about sprints.", file=sys.stderr)
        return 1

    sample = sample.sort_values(["season", "round", "session_type", "abbreviation"])
    sample.to_csv(TARGET, index=False)

    print(f"wrote {len(sample)} rows -> {TARGET}")
    print(sample.groupby(["round", "session_type"]).size().to_string())

    race = sample[sample["session_type"] == "R"]
    print(f"\nrace rows with a usable grid_position: "
          f"{int((race['grid_position'].fillna(0) > 0).sum())}/{len(race)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
