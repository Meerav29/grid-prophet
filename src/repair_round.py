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


def merge_repaired_rows(existing: pd.DataFrame, new_df: pd.DataFrame, season: int, round_: int,
                        session_type: str) -> pd.DataFrame:
    """Swap one session's rows in `existing` for `new_df`.

    The new rows' team names go through `collect_v2.canonical_team_names`, the
    same mapping collect_v2 applies: a round re-fetched through the jolpica
    fallback otherwise comes back with Ergast names ("RB F1 Team") next to the
    season's FastF1 ones ("Racing Bulls"). `check_team_names` then refuses to
    write a file the fit would refuse to load."""
    from team_names import check_team_names

    keep = existing[~((existing["season"] == season) & (existing["round"] == round_) &
                      (existing["session_type"] == session_type))]
    new_df = new_df.copy()
    new_df["team"] = c.canonical_team_names(new_df["team"], season, keep)
    return check_team_names(pd.concat([keep, new_df], ignore_index=True))


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
    combined = merge_repaired_rows(existing, pd.DataFrame(rows, columns=c.COLUMNS),
                                   args.season, args.round, args.session_type)
    combined.to_csv(c.DRIVER_ROUNDS_CSV, index=False)
    print(f"OK: repaired {args.season} R{args.round} {args.session_type} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
