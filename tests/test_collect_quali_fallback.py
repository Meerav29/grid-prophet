"""`collect_v2` quali rows when the results carry positions but no Q1/Q2/Q3
times -- the 2025 R6 Miami gap, repaired by hand for the Phase 2 backtest
(docs/phase2-status.md, "Data"). Autopilot slice-7.

FastF1 is mocked throughout: no network, no cache.
"""

import logging

import numpy as np
import pandas as pd
import pytest

import collect_v2
from collect_v2 import LAPS_FALLBACK_SEGMENT, _build_quali_rows


def _td(seconds):
    return pd.Timedelta(seconds=seconds) if seconds is not None else pd.NaT


class _StubQuali:
    def __init__(self, results, laps):
        self.results = results
        self.laps = laps
        self.weather_data = pd.DataFrame({"AirTemp": [25.0], "Rainfall": [False]})
        self.event, self.name = "Miami Grand Prix", "Qualifying"

    def load(self, **kwargs):
        pass


def _results(q_times):
    """q_times: abbreviation -> (Q1, Q2, Q3) in seconds or None, in position order."""
    rows = []
    for pos, (abbr, (q1, q2, q3)) in enumerate(q_times.items(), start=1):
        rows.append({"Abbreviation": abbr, "FullName": abbr, "TeamName": "T",
                     "Position": float(pos), "Q1": _td(q1), "Q2": _td(q2), "Q3": _td(q3)})
    return pd.DataFrame(rows)


def _laps(rows):
    """rows: (driver, lap_time_s, deleted)."""
    return pd.DataFrame([{"Driver": d, "LapTime": _td(t), "Deleted": deleted}
                         for d, t, deleted in rows])


MIAMI_LAPS = _laps([
    ("NOR", 86.400, False), ("NOR", 86.269, False),
    ("PIA", 86.271, False), ("PIA", 85.900, True),    # struck off: must not count
    ("ANT", 86.271, False), ("ANT", None, False),
])


def _rows(monkeypatch, results, laps):
    monkeypatch.setattr(collect_v2.fastf1, "get_session", lambda *a: _StubQuali(results, laps))
    return {r["abbreviation"]: r for r in _build_quali_rows(2025, 6, "Miami Grand Prix", "mixed")}


class TestNoQTimesFallsBackToSessionLaps:
    RESULTS = _results({"NOR": (None, None, None), "PIA": (None, None, None),
                        "ANT": (None, None, None)})

    def test_best_lap_is_the_fastest_non_deleted_session_lap(self, monkeypatch):
        rows = _rows(monkeypatch, self.RESULTS, MIAMI_LAPS)
        assert rows["NOR"]["best_lap_s"] == pytest.approx(86.269)
        assert rows["PIA"]["best_lap_s"] == pytest.approx(86.271)
        assert rows["ANT"]["best_lap_s"] == pytest.approx(86.271)

    def test_fallback_rows_are_flagged_and_classified(self, monkeypatch):
        rows = _rows(monkeypatch, self.RESULTS, MIAMI_LAPS)
        assert {r["q_segment"] for r in rows.values()} == {LAPS_FALLBACK_SEGMENT}
        assert all(r["classified"] for r in rows.values())

    def test_gaps_are_to_the_fastest_lap_derived_time(self, monkeypatch):
        rows = _rows(monkeypatch, self.RESULTS, MIAMI_LAPS)
        assert rows["NOR"]["gap_to_best_s"] == pytest.approx(0.0)
        assert rows["PIA"]["gap_to_best_s"] == pytest.approx(0.002)

    def test_positions_are_kept_from_the_results(self, monkeypatch):
        rows = _rows(monkeypatch, self.RESULTS, MIAMI_LAPS)
        assert [rows[d]["finish_position"] for d in ("NOR", "PIA", "ANT")] == [1.0, 2.0, 3.0]

    def test_the_fallback_is_logged_with_the_drivers(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger=collect_v2.log.name):
            _rows(monkeypatch, self.RESULTS, MIAMI_LAPS)
        assert any("fastest session lap" in m and "NOR" in m for m in caplog.messages)

    def test_a_driver_with_only_a_deleted_lap_stays_untimed(self, monkeypatch):
        laps = _laps([("NOR", 86.3, False), ("PIA", 85.9, True), ("ANT", 86.4, False)])
        rows = _rows(monkeypatch, self.RESULTS, laps)
        assert rows["PIA"]["best_lap_s"] is None
        assert rows["PIA"]["q_segment"] is None and not rows["PIA"]["classified"]


class TestTimedSessionsAreUnchanged:
    def test_official_times_win_and_laps_are_not_read(self, monkeypatch):
        results = _results({"VER": (87.0, 86.5, 86.1), "NOR": (87.1, 86.6, 86.2)})

        class NoLaps(_StubQuali):
            @property
            def laps(self):
                raise AssertionError("laps read for a session with Q times")

            @laps.setter
            def laps(self, value):
                pass

        monkeypatch.setattr(collect_v2.fastf1, "get_session", lambda *a: NoLaps(results, None))
        rows = {r["abbreviation"]: r for r in _build_quali_rows(2024, 1, "X", "mixed")}
        assert rows["VER"]["best_lap_s"] == pytest.approx(86.1)
        assert rows["NOR"]["q_segment"] == "Q3"
        assert rows["NOR"]["gap_to_best_s"] == pytest.approx(0.1)

    def test_partial_gap_fills_only_the_missing_row_and_keeps_official_pole(self, monkeypatch):
        results = _results({"VER": (87.0, 86.5, 86.1), "NOR": (None, None, None)})
        laps = _laps([("VER", 85.0, False), ("NOR", 86.3, False)])
        rows = _rows(monkeypatch, results, laps)
        assert rows["VER"]["best_lap_s"] == pytest.approx(86.1)
        assert rows["VER"]["q_segment"] == "Q3"
        assert rows["NOR"]["q_segment"] == LAPS_FALLBACK_SEGMENT
        assert rows["NOR"]["gap_to_best_s"] == pytest.approx(0.2)

    def test_an_unclassified_row_without_times_is_left_alone(self, monkeypatch):
        results = _results({"VER": (87.0, 86.5, 86.1)})
        results = pd.concat([results, pd.DataFrame([{
            "Abbreviation": "SAR", "FullName": "SAR", "TeamName": "T", "Position": np.nan,
            "Q1": pd.NaT, "Q2": pd.NaT, "Q3": pd.NaT}])], ignore_index=True)
        laps = _laps([("SAR", 88.0, False)])
        rows = _rows(monkeypatch, results, laps)
        assert rows["SAR"]["best_lap_s"] is None and rows["SAR"]["q_segment"] is None
