"""Event names matched to circuits.csv regardless of accents and same-track renames."""

import pandas as pd
import pytest

from event_names import circuit_key
from model.overtaking import build_overtaking_data, hand_ratings
from test_overtaking import planted_rounds


def test_accents_and_spacing_do_not_matter():
    assert circuit_key("São Paulo Grand Prix") == circuit_key("Sao Paulo Grand Prix")
    assert circuit_key("  Sao  Paulo Grand Prix ") == circuit_key("Sao Paulo Grand Prix")


def test_brazil_renamed_event_is_the_same_track():
    assert circuit_key("Brazilian Grand Prix") == circuit_key("São Paulo Grand Prix")


def test_unrelated_events_stay_distinct():
    assert circuit_key("Mexico City Grand Prix") != circuit_key("Sao Paulo Grand Prix")
    assert circuit_key(None) is None


CIRCUITS = pd.DataFrame({
    "event_name": ["Brazilian Grand Prix", "Sao Paulo Grand Prix"],
    "circuit_type": ["mixed", "mixed"], "overtaking_difficulty": [4.0, 4.0],
})


def _rows(event, season, n_rounds):
    return planted_rounds(5.0, n_rounds=n_rounds, event=event, season=season)


def test_an_accented_event_gets_its_hand_rating_and_is_shrunk():
    data = build_overtaking_data(_rows("São Paulo Grand Prix", 2023, 3), CIRCUITS)
    fit = data.fit_for("São Paulo Grand Prix")
    assert fit.hand_rating == 4.0
    assert fit.shrunk_difficulty == pytest.approx((3 * fit.raw_difficulty + 4 * 4.0) / 7)


def test_both_brazil_names_pool_into_one_fit():
    rows = pd.concat([_rows("Brazilian Grand Prix", 2019, 2), _rows("São Paulo Grand Prix", 2023, 3)])
    data = build_overtaking_data(rows, CIRCUITS)
    assert len(data.by_circuit) == 1
    assert data.fit_for("Brazilian Grand Prix") is data.fit_for("São Paulo Grand Prix")
    assert data.fit_for("Sao Paulo Grand Prix").n_rounds == 5


def test_hand_ratings_are_keyed_the_same_way():
    assert hand_ratings(CIRCUITS)[circuit_key("São Paulo Grand Prix")] == 4.0


def test_circuit_info_finds_the_rating_for_an_accented_round():
    from cli import _circuit_info

    rows = _rows("São Paulo Grand Prix", 2026, 1)
    rows["round"] = 20
    assert _circuit_info(CIRCUITS, rows, 2026, 20) == ("mixed", 4.0)


def test_collector_circuit_lookup_matches_the_accented_name():
    import collect_v2

    circuits = pd.DataFrame({"event_name": ["Sao Paulo Grand Prix"], "circuit_type": ["street"]})
    assert collect_v2._circuit_lookup("São Paulo Grand Prix", circuits) == "street"
