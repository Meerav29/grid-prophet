import pandas as pd

from repair_round import merge_repaired_rows


def _rows(season, round_, session, teams):
    return pd.DataFrame({"season": season, "round": round_, "session_type": session,
                         "abbreviation": [f"D{i}" for i in range(len(teams))], "team": teams})


def test_repaired_rows_take_the_seasons_fastf1_team_names():
    existing = pd.concat([_rows(2026, 10, "R", ["Racing Bulls", "Red Bull Racing"]),
                          _rows(2026, 11, "R", ["Racing Bulls", "Red Bull Racing"])])
    repaired = _rows(2026, 11, "R", ["RB F1 Team", "Red Bull"])
    out = merge_repaired_rows(existing, repaired, 2026, 11, "R")
    assert sorted(out.loc[out["round"] == 11, "team"]) == ["Racing Bulls", "Red Bull Racing"]
    assert len(out) == 4


def test_only_the_repaired_session_is_replaced():
    existing = pd.concat([_rows(2026, 11, "Q", ["Alpine"]), _rows(2026, 11, "R", ["Alpine"])])
    repaired = _rows(2026, 11, "R", ["Alpine F1 Team"])
    out = merge_repaired_rows(existing, repaired, 2026, 11, "R")
    assert sorted(out["session_type"]) == ["Q", "R"]
    assert set(out["team"]) == {"Alpine"}
