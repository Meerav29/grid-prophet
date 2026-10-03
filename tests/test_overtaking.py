"""Unit tests for the fitted per-circuit overtaking parameter (spec sec 3.3
step 3, sec 5; autopilot slice-4).

The planted-parameter tests generate finishing orders with the resolver's own
rule -- order by `race_pace + (GRID_LOCK_BASE_PENALTY_S / d) * (grid - 1)` --
at a known `d`, then check the fitter reads `d` back off nothing but the
observed grid, pace and finish columns. RECOVERY_TOLERANCE is the stated
tolerance sec 3.3 does not give us: see the comment on the constant.
"""

import numpy as np
import pandas as pd
import pytest

from model.overtaking import (
    DEFAULT_DIFFICULTY, DIFFICULTY_MAX, MIN_FINISHERS_PER_ROUND, PACE_COLUMN,
    SHRINKAGE_PRIOR_ROUNDS, build_overtaking_data, circuit_difficulty, fit_circuit,
)
from sim.race import GRID_LOCK_BASE_PENALTY_S

# The search grid is 60 log-spaced points over [0.5, 10], so neighbouring
# candidates sit ~5% apart; 0.45 at d=3 is ~15%, comfortably coarser than the
# grid and tight enough that a locked circuit can never be read as an open one.
RECOVERY_TOLERANCE = 0.45


def planted_rounds(d_true: float, n_rounds: int = 40, n_drivers: int = 20,
                    pace_sd: float = 0.8, noise_sd: float = 0.0, seed: int = 0,
                    event: str = "Planted Grand Prix", season: int = 2020) -> pd.DataFrame:
    """Race rows whose finishing order came from the resolver's rule at `d_true`."""
    rng = np.random.default_rng(seed)
    k = GRID_LOCK_BASE_PENALTY_S / d_true
    rows = []
    for r in range(1, n_rounds + 1):
        pace = np.sort(rng.normal(0.0, pace_sd, n_drivers))   # gap to winner, so sorted
        grid = rng.permutation(n_drivers) + 1
        effective = pace + k * (grid - 1) + rng.normal(0.0, noise_sd, n_drivers)
        finish = np.argsort(np.argsort(effective)) + 1
        for i in range(n_drivers):
            rows.append(dict(season=season, round=r, event_name=event, circuit_type="mixed",
                              session_type="R", classified=True, abbreviation=f"D{i:02d}",
                              grid_position=float(grid[i]), finish_position=float(finish[i]),
                              **{PACE_COLUMN: float(pace[i])}))
    return pd.DataFrame(rows)


class TestPlantedParameterRecovery:
    @pytest.mark.parametrize("d_true", [1.5, 3.0, 5.0])
    def test_recovers_a_planted_parameter(self, d_true):
        """Criterion 3: a known planted parameter comes back within tolerance.

        `hand_rating=None` so nothing anchors the answer -- this is the raw
        fit, not a number nudged toward a prior that already knew it.
        """
        fit = fit_circuit(planted_rounds(d_true), "Planted Grand Prix", hand_rating=None)
        assert fit is not None
        assert abs(fit.raw_difficulty - d_true) < RECOVERY_TOLERANCE

    def test_recovery_survives_noise_the_resolver_would_add(self):
        fit = fit_circuit(planted_rounds(3.0, noise_sd=0.3, seed=7), "Planted Grand Prix",
                           hand_rating=None)
        assert abs(fit.raw_difficulty - 3.0) < RECOVERY_TOLERANCE

    def test_a_wrong_hand_rating_does_not_drag_the_raw_fit(self):
        """The prior may shrink the value the resolver uses; it must not move
        the estimate the data supports."""
        rows = planted_rounds(5.0)
        anchored = fit_circuit(rows, "Planted Grand Prix", hand_rating=1.0)
        free = fit_circuit(rows, "Planted Grand Prix", hand_rating=None)
        assert anchored.raw_difficulty == pytest.approx(free.raw_difficulty)
        assert anchored.shrunk_difficulty < anchored.raw_difficulty  # the prior pulls down

    def test_locked_circuit_scores_visibly_lower_than_an_open_one(self):
        """Criterion 3's second half: Monaco-like < Bahrain-like, by a lot."""
        monaco = fit_circuit(planted_rounds(1.0, n_rounds=10, noise_sd=0.25, seed=1),
                              "Monaco Grand Prix", hand_rating=None)
        bahrain = fit_circuit(planted_rounds(5.0, n_rounds=10, noise_sd=0.25, seed=2),
                               "Bahrain Grand Prix", hand_rating=None)
        assert monaco.raw_difficulty < 2.0 < bahrain.raw_difficulty
        assert bahrain.raw_difficulty > 2 * monaco.raw_difficulty


class TestShrinkage:
    def test_a_thin_circuit_leans_on_the_hand_rating(self):
        """Criterion 1: few observations -> shrunk toward data/circuits.csv."""
        thin = fit_circuit(planted_rounds(6.0, n_rounds=1), "Planted Grand Prix", hand_rating=1.0)
        thick = fit_circuit(planted_rounds(6.0, n_rounds=40), "Planted Grand Prix", hand_rating=1.0)
        assert thin.n_rounds == 1 and thick.n_rounds == 40
        # both fits land near 6.0 raw; only the weight on the rating differs
        assert thin.shrunk_difficulty < thick.shrunk_difficulty
        assert abs(thin.shrunk_difficulty - 1.0) < abs(thick.shrunk_difficulty - 1.0)

    def test_shrinkage_is_the_documented_weighted_mean(self):
        fit = fit_circuit(planted_rounds(5.0, n_rounds=6), "Planted Grand Prix", hand_rating=2.0)
        expected = (6 * fit.raw_difficulty + SHRINKAGE_PRIOR_ROUNDS * 2.0) / (6 + SHRINKAGE_PRIOR_ROUNDS)
        assert fit.shrunk_difficulty == pytest.approx(expected)

    def test_no_hand_rating_means_no_shrinkage(self):
        fit = fit_circuit(planted_rounds(4.0), "Planted Grand Prix", hand_rating=None)
        assert fit.shrunk_difficulty == pytest.approx(fit.raw_difficulty)


class TestUsableRows:
    def test_a_round_with_too_few_finishers_is_not_fitted(self):
        rows = planted_rounds(3.0, n_rounds=2, n_drivers=MIN_FINISHERS_PER_ROUND - 1)
        assert fit_circuit(rows, "Planted Grand Prix", hand_rating=None) is None

    def test_unclassified_and_gridless_rows_are_dropped(self):
        rows = planted_rounds(3.0, n_rounds=3)
        baseline = build_overtaking_data(rows, None).fit_for("Planted Grand Prix")
        polluted = pd.concat([rows, rows.assign(classified=False, finish_position=99.0),
                               rows.assign(grid_position=0.0, finish_position=98.0)])
        after = build_overtaking_data(polluted, None).fit_for("Planted Grand Prix")
        assert after.n_pairs == baseline.n_pairs
        assert after.raw_difficulty == pytest.approx(baseline.raw_difficulty)

    def test_a_frame_without_the_pace_column_yields_no_fits(self):
        rows = planted_rounds(3.0, n_rounds=3).drop(columns=[PACE_COLUMN])
        data = build_overtaking_data(rows, None)
        assert data.by_circuit == {}


class TestWindowing:
    def test_the_forecast_round_is_excluded_from_its_own_fit(self):
        rows = planted_rounds(3.0, n_rounds=5)
        data = build_overtaking_data(rows, None, before_season=2020, before_round=4)
        assert data.fit_for("Planted Grand Prix").n_rounds == 3

    def test_no_prior_rounds_means_no_fit_at_all(self):
        rows = planted_rounds(3.0, n_rounds=5)
        data = build_overtaking_data(rows, None, before_season=2020, before_round=1)
        assert data.fit_for("Planted Grand Prix") is None
        assert data.by_circuit == {}

    def test_a_circuit_whose_rounds_are_all_too_thin_is_recorded_as_skipped(self):
        """Not silently absent: `skipped` says why a circuit has no fit."""
        rows = planted_rounds(3.0, n_rounds=3, n_drivers=MIN_FINISHERS_PER_ROUND - 1)
        data = build_overtaking_data(rows, None)
        assert data.by_circuit == {}
        assert "Planted Grand Prix" in data.skipped


class TestCircuitDifficultyFallback:
    """Criterion 2: the resolver uses the fitted value, falling back to the
    hand rating when no fit exists. Both paths covered."""

    def test_uses_the_fitted_value_when_one_exists(self):
        circuits = pd.DataFrame({"event_name": ["Planted Grand Prix"], "overtaking_difficulty": [1.0]})
        data = build_overtaking_data(planted_rounds(5.0, n_rounds=40), circuits)
        fit = data.fit_for("Planted Grand Prix")
        assert circuit_difficulty(data, "Planted Grand Prix", 1.0) == pytest.approx(fit.shrunk_difficulty)
        assert circuit_difficulty(data, "Planted Grand Prix", 1.0) > 1.0   # not just the rating

    def test_falls_back_to_the_hand_rating_with_no_fit(self):
        data = build_overtaking_data(pd.DataFrame(), None)
        assert circuit_difficulty(data, "Unraced Grand Prix", 4.5) == 4.5
        assert circuit_difficulty(None, "Unraced Grand Prix", 4.5) == 4.5

    def test_falls_back_to_the_default_with_neither(self):
        assert circuit_difficulty(None, "Unraced Grand Prix", None) == DEFAULT_DIFFICULTY
        assert circuit_difficulty(None, "Unraced Grand Prix", float("nan")) == DEFAULT_DIFFICULTY


class TestResolverIntegration:
    def test_circuit_info_prefers_the_fit_over_the_hand_rating(self):
        """The wiring criterion 2 asks for, at the seam `sim.race` reads."""
        from cli import _circuit_info

        rows = planted_rounds(5.0, n_rounds=40, season=2024)
        circuits = pd.DataFrame({"event_name": ["Planted Grand Prix"], "circuit_type": ["mixed"],
                                  "overtaking_difficulty": [1.0]})
        _, hand_only = _circuit_info(circuits, rows, 2024, 1)
        data = build_overtaking_data(rows, circuits)
        _, fitted = _circuit_info(circuits, rows, 2024, 1, data)
        assert hand_only == 1.0                 # unchanged without a fit
        assert fitted > hand_only + 0.5         # the fit moves it, and upward
        assert fitted == pytest.approx(data.fit_for("Planted Grand Prix").shrunk_difficulty)

    def test_a_fitted_value_changes_the_finishing_distribution(self):
        """The parameter is not cosmetic: the same grid and pace resolve
        differently at the fitted value than at the hand rating."""
        from sim.race import RaceTrialInputs, simulate_positions

        pace = np.tile(np.array([0.0, 0.10, 0.20]), (4000, 1))
        base = dict(driver_ids=["A", "B", "C"], race_pace=pace,
                     race_noise_nu=np.full(4000, 8.0), race_noise_sigma=np.full(4000, 0.05),
                     quali_pace=pace, quali_noise_sigma=np.full(4000, 0.05),
                     dnf_prob=np.zeros((4000, 3)), grid=np.array([3, 2, 1]))
        rng = np.random.default_rng(0)
        locked, _ = simulate_positions(RaceTrialInputs(overtaking_difficulty=1.0, **base), rng)
        rng = np.random.default_rng(0)
        open_, _ = simulate_positions(RaceTrialInputs(overtaking_difficulty=8.0, **base), rng)
        # driver A is fastest but starts last: wins far more often when passing is cheap
        assert np.mean(open_[:, 0] == 1) > np.mean(locked[:, 0] == 1) + 0.5


class TestRealCircuitsFile:
    def test_hand_ratings_load_from_the_shipped_csv(self):
        from cli import _load_circuits
        from model.overtaking import hand_ratings

        ratings = hand_ratings(_load_circuits())
        assert ratings["Monaco Grand Prix"] < ratings["Bahrain Grand Prix"]
        assert all(0 < v <= DIFFICULTY_MAX for v in ratings.values())
