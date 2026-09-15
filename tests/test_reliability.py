"""Unit tests for the empirical-Bayes team mechanical-hazard model (spec sec 3.2)."""

import numpy as np
import pandas as pd
import pytest

from model.reliability import (
    build_reliability_data, team_mechanical_posterior, sample_mechanical_dnf_prob,
    _mechanical_prior, RULE_CHANGE_SEASONS, PRIOR_MEAN_STABLE, PRIOR_MEAN_RULE_CHANGE,
    PRIOR_CONCENTRATION_STABLE, PRIOR_CONCENTRATION_RULE_CHANGE,
)


def _toy_driver_rounds():
    rows = []
    # Team A: 2 mechanical DNFs out of 10 starts in 2022 (a rule-change season)
    for i in range(10):
        rows.append(dict(season=2022, round=i + 1, team="Team A", grid_position=i % 5 + 1,
                          session_type="R", dnf_cause="mechanical" if i < 2 else None))
    # Team B: 0 mechanical DNFs out of 10 starts in 2021 (stable season)
    for i in range(10):
        rows.append(dict(season=2021, round=i + 1, team="Team B", grid_position=i % 5 + 1,
                          session_type="R", dnf_cause=None))
    # a DNS row (no grid position) shouldn't count as exposure
    rows.append(dict(season=2022, round=11, team="Team A", grid_position=np.nan,
                      session_type="R", dnf_cause="mechanical"))
    return pd.DataFrame(rows)


class TestMechanicalPrior:
    def test_rule_change_prior_is_higher_mean_lower_concentration(self):
        a_rc, b_rc = _mechanical_prior(2022)
        a_stable, b_stable = _mechanical_prior(2021)
        assert 2022 in RULE_CHANGE_SEASONS
        assert a_rc / (a_rc + b_rc) > a_stable / (a_stable + b_stable)
        assert (a_rc + b_rc) < (a_stable + b_stable)


class TestBuildReliabilityData:
    def test_counts_only_started_races(self):
        df = _toy_driver_rounds()
        data = build_reliability_data(df)
        row = data.team_stats[(data.team_stats["team"] == "Team A") & (data.team_stats["season"] == 2022)]
        assert row.iloc[0]["starts"] == 10  # the DNS-like row (no grid) excluded
        assert row.iloc[0]["mechanical_dnfs"] == 2

    def test_cutoff_restricts_counted_rounds(self):
        df = _toy_driver_rounds()
        data = build_reliability_data(df, through_season=2022, through_round=5)
        row = data.team_stats[(data.team_stats["team"] == "Team A") & (data.team_stats["season"] == 2022)]
        assert row.iloc[0]["starts"] == 5


class TestPosterior:
    def test_team_with_more_dnfs_has_higher_posterior_mean(self):
        df = _toy_driver_rounds()
        data = build_reliability_data(df)
        a_a, b_a = team_mechanical_posterior(data, "Team A", 2022)
        a_b, b_b = team_mechanical_posterior(data, "Team B", 2021)
        mean_a = a_a / (a_a + b_a)
        mean_b = a_b / (a_b + b_b)
        assert mean_a > mean_b

    def test_unknown_team_falls_back_to_population_prior(self):
        df = _toy_driver_rounds()
        data = build_reliability_data(df)
        a, b = team_mechanical_posterior(data, "Cadillac", 2026)
        expected_a, expected_b = _mechanical_prior(2026)
        assert (a, b) == (expected_a, expected_b)


class TestSampling:
    def test_draws_are_valid_probabilities(self):
        df = _toy_driver_rounds()
        data = build_reliability_data(df)
        rng = np.random.default_rng(0)
        draws = sample_mechanical_dnf_prob(data, "Team A", 2022, 1000, rng)
        assert draws.shape == (1000,)
        assert (draws >= 0).all() and (draws <= 1).all()

    def test_more_reliable_team_draws_lower_on_average(self):
        df = _toy_driver_rounds()
        data = build_reliability_data(df)
        rng = np.random.default_rng(0)
        draws_a = sample_mechanical_dnf_prob(data, "Team A", 2022, 5000, rng)
        draws_b = sample_mechanical_dnf_prob(data, "Team B", 2021, 5000, rng)
        assert draws_a.mean() > draws_b.mean()
