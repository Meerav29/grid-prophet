"""A season that carries one team under two names must stop a fit, not split the team.

The Nov 26 championship run found 2026 R18-21 under Ergast names ("Red Bull")
while R1-17 used FastF1's ("Red Bull Racing"), so the pace model saw a new team
from R18. canonical_team_names stops that for newly collected rounds; this
guard catches a data file built before it."""

import pandas as pd
import pytest

from team_names import ERGAST_TEAM_ALIASES, check_team_names, split_team_names


def _dr(rows):
    return pd.DataFrame(rows, columns=["season", "round", "team"])


def test_a_season_with_both_names_is_reported():
    dr = _dr([(2026, 17, "Red Bull Racing"), (2026, 18, "Red Bull"), (2026, 18, "Ferrari")])
    assert split_team_names(dr) == [(2026, "Red Bull", "Red Bull Racing")]


def test_check_raises_and_names_the_season_and_both_names():
    dr = _dr([(2026, 1, "Alpine"), (2026, 18, "Alpine F1 Team")])
    with pytest.raises(ValueError) as e:
        check_team_names(dr)
    msg = str(e.value)
    assert "2026" in msg and "'Alpine F1 Team'" in msg and "'Alpine'" in msg
    assert "canonical_team_names" in msg


def test_an_ergast_name_alone_in_its_season_is_fine():
    # A season loaded wholly from Ergast is consistent: only a mix splits a team.
    dr = _dr([(2026, 1, "Red Bull"), (2026, 2, "Red Bull"), (2025, 1, "Red Bull Racing")])
    assert split_team_names(dr) == []
    check_team_names(dr)


def test_names_in_different_seasons_do_not_clash():
    dr = _dr([(2024, 1, "RB"), (2025, 1, "RB F1 Team")])
    assert split_team_names(dr) == []


def test_alias_and_candidate_with_the_same_name_is_not_a_split():
    # "Sauber" maps to ["Kick Sauber", "Sauber"]; "Sauber" alone is one name.
    dr = _dr([(2023, 1, "Sauber"), (2023, 2, "Sauber")])
    assert split_team_names(dr) == []


def test_the_collector_uses_the_same_alias_table():
    import collect_v2
    assert collect_v2.ERGAST_TEAM_ALIASES is ERGAST_TEAM_ALIASES


def test_the_cli_loader_refuses_a_split_file(tmp_path, monkeypatch):
    import cli
    p = tmp_path / "driver_rounds.csv"
    _dr([(2026, 17, "Red Bull Racing"), (2026, 18, "Red Bull")]).to_csv(p, index=False)
    monkeypatch.setattr(cli, "DRIVER_ROUNDS_CSV", str(p))
    with pytest.raises(ValueError, match="Red Bull"):
        cli._load_driver_rounds()
