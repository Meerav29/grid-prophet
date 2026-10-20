"""The fitted overtaking parameter reaches both production forecasts, and each
forecast fits only its own circuit (autopilot slice-7, slice-4 carry-forwards).

PR #10's review found that `build_overtaking_data` could be deleted from
either call site (`cli.forecast_weekend`, `backtest.forecast_race_challenger`)
with the whole suite green: the tests handed the fit in at the `_circuit_info`
seam, so nothing checked that a forecast builds one for itself. The fixture
below makes the fit depart from the hand rating, so a forecast that silently
falls back to `data/circuits.csv` is caught.
"""

import numpy as np
import pandas as pd
import pytest

from cli import forecast_weekend
from model.overtaking import build_overtaking_data
from test_cli import DRIVERS, _fake_fit, _sprint_weekend_rounds

HAND_RATING = 4.5
HISTORY_EVENTS = ["Bahrain Grand Prix", "Monaco Grand Prix", "Italian Grand Prix",
                   "Japanese Grand Prix", "Brazilian Grand Prix"]
CIRCUITS = pd.DataFrame({
    "event_name": HISTORY_EVENTS, "circuit_type": ["mixed"] * len(HISTORY_EVENTS),
    "overtaking_difficulty": [HAND_RATING] * len(HISTORY_EVENTS),
})


def _locked_history(events=HISTORY_EVENTS, rounds_per_event=6, n_drivers=8):
    """2023 races where nobody passes: the fastest car starts last and
    finishes last. Only a very low difficulty explains that, so every
    circuit's fit lands well below the 4.5 hand rating."""
    rows = []
    rnd = 0
    for event in events:
        for _ in range(rounds_per_event):
            rnd += 1
            for i in range(n_drivers):
                grid = float(n_drivers - i)
                rows.append(dict(
                    season=2023, round=rnd, event_name=event, circuit_type="mixed",
                    session_type="R", abbreviation=f"H{i}", team="Alpha" if i % 2 else "Beta",
                    grid_position=grid, finish_position=grid, classified=True,
                    status="Finished", dnf_cause=None,
                    gap_to_winner_median_clean_air_s=1.0 * i, is_wet=False,
                ))
    return pd.DataFrame(rows)


def _driver_rounds():
    return pd.concat([_locked_history(), _sprint_weekend_rounds()], ignore_index=True)


def _fitted_bahrain(driver_rounds):
    fit = build_overtaking_data(driver_rounds, CIRCUITS, before_season=2024,
                                before_round=2).fit_for("Bahrain Grand Prix")
    assert fit is not None
    # The fixture is only useful if the fit really departs from the rating.
    assert abs(fit.shrunk_difficulty - HAND_RATING) > 1.0
    return fit.shrunk_difficulty


def _count_fits(monkeypatch):
    import model.overtaking

    fitted = []
    real = model.overtaking.fit_circuit

    def counting(rows, event_name, hand_rating):
        fitted.append(event_name)
        return real(rows, event_name, hand_rating)

    monkeypatch.setattr(model.overtaking, "fit_circuit", counting)
    return fitted


def _run_forecast_weekend(monkeypatch, driver_rounds):
    import sim.race
    from model.reliability import build_reliability_data

    captured = {}
    real = sim.race.simulate_weekend

    def spy(inputs, *a, **kw):
        captured["inputs"] = inputs
        return real(inputs, *a, **kw)

    monkeypatch.setattr(sim.race, "simulate_weekend", spy)
    idata, data = _fake_fit(driver_rounds)
    forecast_weekend(idata, data, build_reliability_data(driver_rounds), driver_rounds,
                     CIRCUITS, season=2024, round_=2, n_trials=200, seed=0)
    return captured["inputs"]


def _run_challenger(monkeypatch, driver_rounds):
    import model.challenger
    import sim.race
    from backtest import forecast_race_challenger
    from model.reliability import build_reliability_data

    captured = {}
    real = sim.race.simulate_positions

    def spy(inputs, *a, **kw):
        captured["inputs"] = inputs
        return real(inputs, *a, **kw)

    def fake_pace(model, team, driver, circuit_type, trp, drp, n, rng):
        return rng.normal(DRIVERS.index(driver) * 0.3, 0.1, n)

    monkeypatch.setattr(sim.race, "simulate_positions", spy)
    monkeypatch.setattr(model.challenger, "sample_pace", fake_pace)
    forecast_race_challenger(None, build_reliability_data(driver_rounds), driver_rounds,
                             CIRCUITS, season=2024, round_=2, n_trials=200, seed=0)
    return captured["inputs"]


RUNNERS = {"forecast_weekend": _run_forecast_weekend,
           "forecast_race_challenger": _run_challenger}


@pytest.mark.parametrize("call_site", sorted(RUNNERS))
class TestBothCallSitesFitTheirOwnCircuit:
    def test_the_resolver_gets_the_fitted_value_not_the_hand_rating(self, monkeypatch, call_site):
        driver_rounds = _driver_rounds()
        inputs = RUNNERS[call_site](monkeypatch, driver_rounds)
        assert inputs.overtaking_difficulty == pytest.approx(_fitted_bahrain(driver_rounds))
        assert inputs.overtaking_difficulty != pytest.approx(HAND_RATING)

    def test_one_circuit_is_fitted_however_many_the_window_holds(self, monkeypatch, call_site):
        """Five circuits in the window, one fit per forecast. Before slice-7
        this counted every circuit with history (five here)."""
        driver_rounds = _driver_rounds()
        fitted = _count_fits(monkeypatch)
        RUNNERS[call_site](monkeypatch, driver_rounds)
        assert fitted == ["Bahrain Grand Prix"]

    def test_the_fit_count_does_not_grow_with_the_window(self, monkeypatch, call_site):
        small = pd.concat([_locked_history(HISTORY_EVENTS[:1]), _sprint_weekend_rounds()],
                          ignore_index=True)
        counts = []
        for driver_rounds in (small, _driver_rounds()):
            fitted = _count_fits(monkeypatch)
            RUNNERS[call_site](monkeypatch, driver_rounds)
            counts.append(len(fitted))
        assert counts == [1, 1]


class TestSingleCircuitFit:
    def test_restricting_to_one_event_matches_that_event_in_the_full_fit(self):
        driver_rounds = _locked_history()
        full = build_overtaking_data(driver_rounds, CIRCUITS)
        one = build_overtaking_data(driver_rounds, CIRCUITS, event_name="Monaco Grand Prix")
        assert set(one.by_circuit) == {"Monaco Grand Prix"}
        assert one.fit_for("Monaco Grand Prix") == full.fit_for("Monaco Grand Prix")

    def test_an_event_with_no_history_gets_no_fit(self):
        one = build_overtaking_data(_locked_history(), CIRCUITS, event_name="Las Vegas Grand Prix")
        assert one.by_circuit == {} and one.skipped == {}

    def test_no_event_still_fits_the_whole_window(self):
        assert len(build_overtaking_data(_locked_history(), CIRCUITS).by_circuit) == len(HISTORY_EVENTS)
