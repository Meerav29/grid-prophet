"""Ergast vs FastF1 team names, and a guard against a season that uses both.

Kept apart from `collect_v2` (which imports FastF1) so the fit and backtest
paths can run the guard cheaply on every load."""

import pandas as pd

# Ergast constructor name -> the FastF1 team names it can stand for. When F1
# live timing is unavailable FastF1 loads results from Ergast, whose names
# differ from the ones every earlier row uses, and the pace model would then
# see a new team. A candidate already used in that season wins; else the first.
ERGAST_TEAM_ALIASES = {
    "Red Bull": ["Red Bull Racing"],
    "Alpine F1 Team": ["Alpine"],
    "Cadillac F1 Team": ["Cadillac"],
    "RB F1 Team": ["Racing Bulls", "RB"],
    "Sauber": ["Kick Sauber", "Sauber"],
    "Alfa Romeo": ["Alfa Romeo", "Alfa Romeo Racing"],
}


def split_team_names(driver_rounds: pd.DataFrame) -> list[tuple[int, str, str]]:
    """(season, ergast_name, fastf1_name) for every season that carries one
    team under both names. Empty when every season is consistent."""
    found = []
    for season, teams in driver_rounds.groupby("season")["team"]:
        names = set(teams.dropna())
        for ergast, candidates in ERGAST_TEAM_ALIASES.items():
            if ergast not in names:
                continue
            found += [(int(season), ergast, c) for c in candidates if c != ergast and c in names]
    return sorted(found)


def check_team_names(driver_rounds: pd.DataFrame) -> pd.DataFrame:
    """Raise if any season carries one team under two names; else return the frame."""
    splits = split_team_names(driver_rounds)
    if splits:
        listed = "; ".join(f"{s}: {e!r} and {f!r}" for s, e, f in splits)
        raise ValueError(
            f"driver_rounds.csv has teams under two names in the same season ({listed}). "
            "The pace model would treat each pair as two teams. Map the Ergast names "
            "onto the season's FastF1 names with collect_v2.canonical_team_names on "
            "those rows, then refit.")
    return driver_rounds
