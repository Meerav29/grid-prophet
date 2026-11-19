"""Ergast constructor names mapped onto the FastF1 team names driver_rounds.csv uses."""

import logging

import pandas as pd

import collect_v2
from collect_v2 import canonical_team_names


def _existing(season, teams):
    return pd.DataFrame({"season": season, "team": teams})


def test_ergast_names_map_to_the_seasons_fastf1_names():
    existing = _existing(2026, ["Red Bull Racing", "Alpine", "Cadillac", "Racing Bulls", "Ferrari"])
    got = canonical_team_names(
        pd.Series(["Red Bull", "Alpine F1 Team", "Cadillac F1 Team", "RB F1 Team", "Ferrari"]),
        2026, existing)
    assert got.tolist() == ["Red Bull Racing", "Alpine", "Cadillac", "Racing Bulls", "Ferrari"]


def test_season_dependent_names_follow_what_that_season_already_uses():
    existing = pd.concat([_existing(2024, ["RB", "Kick Sauber"]), _existing(2025, ["Racing Bulls"])])
    assert canonical_team_names(pd.Series(["RB F1 Team", "Sauber"]), 2024, existing).tolist() \
        == ["RB", "Kick Sauber"]
    assert canonical_team_names(pd.Series(["RB F1 Team"]), 2025, existing).tolist() == ["Racing Bulls"]


def test_fastf1_names_pass_through_untouched():
    existing = _existing(2026, ["Red Bull Racing", "Haas F1 Team"])
    names = pd.Series(["Red Bull Racing", "Haas F1 Team"])
    assert canonical_team_names(names, 2026, existing).tolist() == names.tolist()


def test_new_season_with_no_rows_uses_the_default_alias():
    got = canonical_team_names(pd.Series(["Red Bull", "RB F1 Team"]), 2027, pd.DataFrame())
    assert got.tolist() == ["Red Bull Racing", "Racing Bulls"]


def test_a_name_the_season_has_never_seen_is_kept_and_warned(caplog):
    existing = _existing(2026, ["Ferrari"])
    with caplog.at_level(logging.WARNING, logger=collect_v2.log.name):
        got = canonical_team_names(pd.Series(["Brand New F1 Team"]), 2026, existing)
    assert got.tolist() == ["Brand New F1 Team"]
    assert "Brand New F1 Team" in caplog.text


def test_collected_rounds_are_written_with_canonical_names(monkeypatch, tmp_path):
    existing = pd.DataFrame([{c: None for c in collect_v2.COLUMNS}] * 2)
    existing["season"], existing["round"] = 2026, 18
    existing["session_type"] = ["Q", "R"]
    existing["team"] = "Red Bull Racing"

    class Schedule:
        def iterrows(self):
            yield 0, {"RoundNumber": 19, "EventName": "Mexico City Grand Prix", "EventFormat": "conventional"}

    def row(session):
        return {**{c: None for c in collect_v2.COLUMNS}, "season": 2026, "round": 19,
                "session_type": session, "abbreviation": "VER", "team": "Red Bull"}

    monkeypatch.setattr(collect_v2, "DRIVER_ROUNDS_CSV", str(tmp_path / "dr.csv"))
    monkeypatch.setattr(collect_v2.fastf1, "get_event_schedule", lambda *a, **k: Schedule())
    monkeypatch.setattr(collect_v2, "_circuit_lookup", lambda *a: "mixed")
    monkeypatch.setattr(collect_v2, "_build_quali_rows", lambda *a: [row("Q")])
    monkeypatch.setattr(collect_v2, "_build_race_rows", lambda *a: [row("R")])

    out = collect_v2.build_driver_rounds(2026, 2026, existing)
    assert set(out.loc[out["round"] == 19, "team"]) == {"Red Bull Racing"}
