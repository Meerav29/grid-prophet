"""Clean-air race pace from jolpica lap times, the fallback for when FastF1 has no laps."""

import pandas as pd
import pytest

import collect_v2
import jolpica_laps
from jolpica_laps import clean_air_pace, race_laps

BASE_LAP = 90.0


def _laps(times: dict[str, list[float]]) -> pd.DataFrame:
    """Lap frame from per-driver lap times; positions follow cumulative time."""
    rows = []
    n = len(next(iter(times.values())))
    cum = {c: 0.0 for c in times}
    for lap in range(1, n + 1):
        for c in times:
            cum[c] += times[c][lap - 1]
        order = sorted(times, key=lambda c: cum[c])
        for pos, c in enumerate(order, start=1):
            rows.append({"lap": lap, "code": c, "pos": pos, "t": times[c][lap - 1]})
    return pd.DataFrame(rows)


def test_leader_in_clean_air_and_follower_in_traffic():
    # B sits 0.5 s behind A all race; C is 10 s back, alone.
    times = {
        "AAA": [100.0] + [BASE_LAP] * 7,
        "BBB": [100.5] + [BASE_LAP] * 7,
        "CCC": [110.0] + [BASE_LAP + 1.0] * 7,
    }
    pace = clean_air_pace(_laps(times), pits=set())
    assert pace["AAA"] == (BASE_LAP, BASE_LAP)
    assert pace["CCC"] == (BASE_LAP + 1.0, BASE_LAP + 1.0)
    assert pace["BBB"] == (None, None)


def test_lap_one_pit_laps_and_safety_car_laps_are_dropped():
    a = [80.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0]
    a[4] = 70.0      # in-lap of a pit stop on lap 5 (implausibly fast, so a leak shows)
    a[5] = 71.0      # out-lap
    times = {"AAA": a, "CCC": [x + 20.0 for x in a]}
    # Laps 8 for everyone run under a safety car; lap 9 is the restart.
    for c in times:
        times[c][7] = 140.0
    pace = clean_air_pace(_laps(times), pits={("AAA", 5)})
    kept = [91.0, 92.0, 93.0, 96.0, 99.0]   # laps 2-4, 7, 10: lap 1, pit 5-6, SC 8, restart 9 dropped
    med, best = pace["AAA"]
    assert best == 91.0
    assert med == pytest.approx(pd.Series(kept).median())


def test_a_driver_limping_far_off_the_field_does_not_set_their_median():
    # D has damage from lap 4 on and runs 25% off the pace in clean air.
    times = {
        "AAA": [100.0] + [BASE_LAP] * 9,
        "CCC": [105.0] + [BASE_LAP] * 9,
        "EEE": [110.0] + [BASE_LAP] * 9,
        "DDD": [120.0, BASE_LAP, BASE_LAP] + [BASE_LAP * 1.25] * 7,
    }
    med, best = clean_air_pace(_laps(times), pits=set())["DDD"]
    assert (med, best) == (BASE_LAP, BASE_LAP)


def _page(total, offset, races):
    return {"MRData": {"total": str(total), "offset": str(offset), "RaceTable": {"Races": races},
                       "DriverTable": {"Drivers": races}}}


def test_race_laps_paginates_and_maps_driver_ids_to_codes():
    lap_pages = {
        0: [{"Laps": [{"number": "1", "Timings": [
            {"driverId": "alpha", "position": "1", "time": "1:30.500"},
            {"driverId": "bravo", "position": "2", "time": "1:31.000"}]}]}],
        100: [{"Laps": [{"number": "2", "Timings": [
            {"driverId": "alpha", "position": "1", "time": "1:29.000"},
            {"driverId": "bravo", "position": "2", "time": "59.900"}]}]}],
    }
    calls = []

    def fake_get(path, offset):
        calls.append((path, offset))
        if path.endswith("/laps"):
            return _page(150, offset, lap_pages[offset])
        if path.endswith("/pitstops"):
            return _page(1, 0, [{"PitStops": [{"driverId": "bravo", "lap": "1"}]}])
        if path.endswith("/drivers"):
            return _page(2, 0, [{"driverId": "alpha", "code": "ALP"},
                                {"driverId": "bravo", "code": "BRV"}])
        raise AssertionError(path)

    laps, pits = race_laps(2026, 19, get_json=fake_get)
    assert ("2026/19/laps", 100) in calls
    assert sorted(laps["code"].unique()) == ["ALP", "BRV"]
    assert laps.loc[(laps.code == "ALP") & (laps.lap == 1), "t"].item() == pytest.approx(90.5)
    assert laps.loc[(laps.code == "BRV") & (laps.lap == 2), "t"].item() == pytest.approx(59.9)
    assert pits == {("BRV", 1)}


# --- wiring into collect_v2 -------------------------------------------------

class _NoLapsSession:
    """A FastF1 session that loaded results from Ergast but has no laps or weather."""

    def __init__(self):
        self.results = pd.DataFrame({
            "Abbreviation": ["AAA", "BBB", "CCC"], "FullName": ["A", "B", "C"],
            "TeamName": ["T1", "T1", "T2"], "GridPosition": [1, 2, 3],
            "Position": [1, 2, 3], "Status": ["Finished"] * 3, "Points": [25, 18, 15],
        })

    def load(self, **kwargs):
        pass

    @property
    def laps(self):
        raise RuntimeError("The data you are trying to access has not been loaded yet")

    @property
    def weather_data(self):
        raise RuntimeError("not loaded")


def _build(monkeypatch, session_code, fallback):
    monkeypatch.setattr(collect_v2.fastf1, "get_session", lambda *a: _NoLapsSession())
    monkeypatch.setattr(collect_v2, "jolpica_clean_air", fallback)
    rows = collect_v2._build_race_rows(2026, 19, "Test GP", "mixed", session_code, {})
    return {r["abbreviation"]: r for r in rows}


def test_race_without_fastf1_laps_takes_pace_from_jolpica(monkeypatch):
    seen = []

    def fallback(season, rnd):
        seen.append((season, rnd))
        return {"AAA": (91.0, 90.0), "BBB": (None, None), "CCC": (91.5, 90.2)}

    rows = _build(monkeypatch, "R", fallback)
    assert seen == [(2026, 19)]
    assert rows["AAA"]["median_clean_air_lap_s"] == 91.0
    assert rows["CCC"]["gap_to_winner_median_clean_air_s"] == pytest.approx(0.5)
    assert rows["CCC"]["gap_to_winner_best_clean_air_s"] == pytest.approx(0.2)
    assert rows["BBB"]["median_clean_air_lap_s"] is None
    assert rows["AAA"]["is_wet"] is None   # weather is not part of the fallback


def test_sprint_does_not_use_the_race_fallback(monkeypatch):
    def fallback(season, rnd):
        raise AssertionError("jolpica has no sprint laps; must not be called for S")

    rows = _build(monkeypatch, "S", fallback)
    assert rows["AAA"]["median_clean_air_lap_s"] is None


def test_fallback_failure_leaves_pace_blank_not_the_round(monkeypatch):
    def fallback(season, rnd):
        raise OSError("jolpica down")

    rows = _build(monkeypatch, "R", fallback)
    assert len(rows) == 3
    assert all(r["median_clean_air_lap_s"] is None for r in rows.values())
    assert rows["AAA"]["finish_position"] == 1
