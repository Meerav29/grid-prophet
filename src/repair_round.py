"""One-off repair for a single (season, round, session_type) in driver_rounds.csv.

Used when a round's Race/Sprint rows came back with Status blank across the
board from a long-running collect_v2 process (see docs/grid-prophet-v2-spec.md
and data/driver_rounds_validation.md) -- a fresh, short-lived process re-fetching
the same cached session has consistently produced correct data, so this script
is invoked once per broken round, each as its own process.
"""

import argparse
import sys

import fastf1
import pandas as pd

import collect_v2 as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    parser.add_argument("session_type", choices=["Q", "S", "R"])
    args = parser.parse_args()

    fastf1.Cache.enable_cache(c.CACHE_DIR)
    circuits = pd.read_csv(c.CIRCUITS_CSV)
    dnf_map = c._load_dnf_map()

    schedule = fastf1.get_event_schedule(args.season, include_testing=False)
    event = schedule[schedule["RoundNumber"] == args.round].iloc[0]
    event_name = event.get("EventName", f"Round {args.round}")
    circuit_type = c._circuit_lookup(event_name, circuits)

    if args.session_type == "Q":
        rows = c._build_quali_rows(args.season, args.round, event_name, circuit_type)
    else:
        rows = c._build_race_rows(args.season, args.round, event_name, circuit_type, args.session_type, dnf_map)

    if not rows:
        print(f"FAILED: no rows returned for {args.season} R{args.round} {args.session_type}", file=sys.stderr)
        sys.exit(1)

    bad = sum(1 for r in rows if r.get("status") is None and args.session_type != "Q")
    if args.session_type != "Q" and bad > len(rows) * 0.5:
        print(f"STILL BROKEN: {args.season} R{args.round} {args.session_type} -- {bad}/{len(rows)} blank status",
              file=sys.stderr)
        sys.exit(1)

    existing = pd.read_csv(c.DRIVER_ROUNDS_CSV)
    existing = existing[~((existing["season"] == args.season) & (existing["round"] == args.round) &
                           (existing["session_type"] == args.session_type))]
    new_df = pd.DataFrame(rows, columns=c.COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(c.DRIVER_ROUNDS_CSV, index=False)
    print(f"OK: repaired {args.season} R{args.round} {args.session_type} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
