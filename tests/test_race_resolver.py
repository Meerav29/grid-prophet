"""Unit tests for the pure-numpy Monte Carlo race resolver (spec sec 3.3)."""

import numpy as np
import pytest

from sim.race import RaceTrialInputs, simulate_positions, summarize_trials, load_points_table, points_for_position


def _make_inputs(race_pace, quali_pace, overtaking_difficulty, n_trials=4000,
                  dnf_prob=None, race_sigma=0.3, quali_sigma=0.2):
    n_drivers = len(race_pace)
    dnf_prob = np.zeros((n_trials, n_drivers)) if dnf_prob is None else dnf_prob
    return RaceTrialInputs(
        driver_ids=[f"d{i}" for i in range(n_drivers)],
        race_pace=np.tile(race_pace, (n_trials, 1)),
        race_noise_nu=np.full(n_trials, 10.0),
        race_noise_sigma=np.full(n_trials, race_sigma),
        quali_pace=np.tile(quali_pace, (n_trials, 1)),
        quali_noise_sigma=np.full(n_trials, quali_sigma),
        dnf_prob=dnf_prob,
        overtaking_difficulty=overtaking_difficulty,
    )


class TestPointsTable:
    def test_loads_standard_top10_table(self):
        table = load_points_table()
        assert table[1] == 25.0
        assert table[10] == 1.0

    def test_points_zero_outside_top10(self):
        table = load_points_table()
        assert points_for_position(11, table) == 0.0
        assert points_for_position(1, table) == 25.0


class TestSimulatePositions:
    def test_faster_driver_wins_more_often_high_overtaking(self):
        rng = np.random.default_rng(0)
        inputs = _make_inputs(race_pace=[0.0, 0.5, 1.0, 1.5], quali_pace=[0.0, 0.5, 1.0, 1.5],
                               overtaking_difficulty=5.0)
        positions, dnf = simulate_positions(inputs, rng)
        win_rate = (positions[:, 0] == 1).mean()
        assert win_rate > 0.6

    def test_grid_lock_at_low_overtaking_difficulty(self):
        """At overtaking_difficulty -> hard (Monaco-like), a big grid
        advantage should dominate a modest pace disadvantage."""
        rng = np.random.default_rng(0)
        # driver 0 starts on pole (quali_pace lowest) despite having worse race pace
        inputs = _make_inputs(race_pace=[0.3, 0.0, 0.6, 0.9], quali_pace=[0.0, 1.0, 2.0, 3.0],
                               overtaking_difficulty=1.0, race_sigma=0.05)
        positions, dnf = simulate_positions(inputs, rng)
        win_rate_pole = (positions[:, 0] == 1).mean()
        assert win_rate_pole > 0.5

    def test_high_overtaking_lets_pace_dominate_grid(self):
        rng = np.random.default_rng(0)
        # driver 1 has much better race pace despite starting further back
        inputs = _make_inputs(race_pace=[1.0, 0.0, 1.0, 1.0], quali_pace=[0.0, 3.0, 4.0, 5.0],
                               overtaking_difficulty=8.0, race_sigma=0.05)
        positions, dnf = simulate_positions(inputs, rng)
        win_rate_fast = (positions[:, 1] == 1).mean()
        assert win_rate_fast > 0.5

    def test_dnf_drivers_never_score_top_position(self):
        rng = np.random.default_rng(0)
        n_trials = 2000
        dnf_prob = np.zeros((n_trials, 2))
        dnf_prob[:, 0] = 1.0  # driver 0 always DNFs
        inputs = _make_inputs(race_pace=[0.0, 1.0], quali_pace=[0.0, 1.0],
                               overtaking_difficulty=3.0, n_trials=n_trials, dnf_prob=dnf_prob)
        positions, dnf = simulate_positions(inputs, rng)
        assert dnf[:, 0].all()
        assert (positions[:, 0] == 2).all()  # always classified last (behind the finisher)

    def test_positions_are_a_permutation_of_1_to_n(self):
        rng = np.random.default_rng(1)
        inputs = _make_inputs(race_pace=[0.0, 0.2, 0.4, 0.6, 0.8], quali_pace=[0.0, 0.2, 0.4, 0.6, 0.8],
                               overtaking_difficulty=3.0, n_trials=50)
        positions, _ = simulate_positions(inputs, rng)
        for row in positions:
            assert sorted(row) == list(range(1, 6))


class TestSummarizeTrials:
    def test_probabilities_sum_sensibly(self):
        rng = np.random.default_rng(0)
        inputs = _make_inputs(race_pace=[0.0, 0.3, 0.6], quali_pace=[0.0, 0.3, 0.6], overtaking_difficulty=3.0)
        positions, dnf = simulate_positions(inputs, rng)
        table = load_points_table()
        df = summarize_trials(positions, dnf, inputs.driver_ids, ["T1", "T2", "T3"], table)
        assert abs(df["p_win"].sum() - 1.0) < 1e-9
        assert (df["p_podium"] <= 1.0).all() and (df["p_podium"] >= 0.0).all()
        assert (df["exp_points"] >= 0).all()

    def test_faster_driver_has_higher_expected_points(self):
        rng = np.random.default_rng(0)
        inputs = _make_inputs(race_pace=[0.0, 1.0, 2.0], quali_pace=[0.0, 1.0, 2.0], overtaking_difficulty=4.0)
        positions, dnf = simulate_positions(inputs, rng)
        table = load_points_table()
        df = summarize_trials(positions, dnf, inputs.driver_ids, ["T1", "T2", "T3"], table)
        df = df.set_index("driver")
        assert df.loc["d0", "exp_points"] > df.loc["d1", "exp_points"] > df.loc["d2", "exp_points"]
