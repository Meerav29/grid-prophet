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
        rows.append(dict(season=2022, round=i + 1, team="Team A", abbreviation="AAA",
                          grid_position=i % 5 + 1,
                          session_type="R", dnf_cause="mechanical" if i < 2 else None))
    # Team B: 0 mechanical DNFs out of 10 starts in 2021 (stable season)
    for i in range(10):
        rows.append(dict(season=2021, round=i + 1, team="Team B", abbreviation="BBB",
                          grid_position=i % 5 + 1,
                          session_type="R", dnf_cause=None))
    # a DNS row (no grid position) shouldn't count as exposure
    rows.append(dict(season=2022, round=11, team="Team A", abbreviation="AAA",
                      grid_position=np.nan, session_type="R", dnf_cause="mechanical"))
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

# --- slice-3: the mechanical/incident split (spec sec 3.2) ----------------
import os

from model.reliability import (
    CAUSELESS_RATIO_FLOOR, DNF_STATUS_MAP_CSV, FLAGGED_LABEL_LEAN,
    FIRST_LAP_HAZARD_MULTIPLIER, NOMINAL_RACE_LAPS, NOMINAL_SPRINT_LAPS,
    PRIOR_MEAN_INCIDENT, RULE_CHANGE_WIDENING_DECAY_ROUNDS, channel_weights,
    combined_dnf_prob, driver_incident_posterior, first_lap_incident_share,
    grid_band, grid_multiplier, lap_of_retirement_pmf, load_label_policy,
    sample_incident_dnf_prob, sprint_incident_exposure,
)

FIXTURE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "driver_rounds_fixture.csv")


def _race_rows(specs):
    """specs: (status, dnf_cause) pairs -> a minimal race-rows frame."""
    return pd.DataFrame([
        dict(season=2024, round=i + 1, team="T", abbreviation="AAA", grid_position=5,
             session_type="R", status=st, dnf_cause=cause)
        for i, (st, cause) in enumerate(specs)
    ])


class TestRuleChangeWideningShrinks:
    """Sec 3.2 widens the prior in rule-change years *and shrinks it as the
    season runs* -- 2014 and 2022 had elevated **early-season** failures."""

    def test_widening_decays_toward_the_stable_prior(self):
        conc = [sum(_mechanical_prior(2022, r)) for r in (0, 4, RULE_CHANGE_WIDENING_DECAY_ROUNDS)]
        assert conc[0] < conc[1] < conc[2] == pytest.approx(PRIOR_CONCENTRATION_STABLE)
        means = [a / (a + b) for a, b in (_mechanical_prior(2022, r) for r in (0, 4, 8))]
        assert means[0] > means[1] > means[2] == pytest.approx(PRIOR_MEAN_STABLE)
        # saturates at the stable prior; a stable season never moves at all
        assert _mechanical_prior(2022, 40) == pytest.approx(_mechanical_prior(2021, 0))
        assert _mechanical_prior(2021, 0) == _mechanical_prior(2021, 12)

    def test_team_posterior_uses_the_round_count_from_its_window(self):
        df = _toy_driver_rounds()
        early = build_reliability_data(df, through_season=2022, through_round=2)
        late = build_reliability_data(df, through_season=2022, through_round=10)
        # Same two DNFs in both windows, but the late window's prior has decayed,
        # so its posterior is tighter (larger a+b) and less inflated.
        a_e, b_e = team_mechanical_posterior(early, "Team A", 2022)
        a_l, b_l = team_mechanical_posterior(late, "Team A", 2022)
        assert (a_l + b_l) > (a_e + b_e)
        assert a_l / (a_l + b_l) < a_e / (a_e + b_e)


class TestDataGapIsHonored:
    """Sec 3.2's "Known data gap": FastF1's cause detail stops after 2024, so
    2023+ rows mostly say a generic "Retired". `data/dnf_status_map.csv` flags
    those `needs_review=True` and they must be consumed as a *mixture*."""

    def test_generic_retired_row_is_not_a_clean_label(self):
        w = channel_weights(_race_rows([("Retired", "other")]))
        assert bool(w["label_flagged"].iloc[0])
        m, i = w["w_mechanical"].iloc[0], w["w_incident"].iloc[0]
        assert 0 < m < 1 and 0 < i < 1 and m + i == pytest.approx(1.0)

    def test_every_needs_review_status_in_the_map_splits(self):
        policy = load_label_policy(DNF_STATUS_MAP_CSV)
        flagged = policy[policy["needs_review"]]
        assert len(flagged) >= 3  # the spec names Retired, Wheel, Puncture at minimum
        w = channel_weights(_race_rows([(r.status, r.category) for r in flagged.itertuples()
                                         if r.category != "classified"]), policy)
        assert w["label_flagged"].all()
        assert (w["w_mechanical"] < 1.0).all() and (w["w_incident"] < 1.0).all()
        assert (w["w_mechanical"] > 0.0).all() and (w["w_incident"] > 0.0).all()

    def test_unflagged_status_is_a_clean_label_and_finishers_carry_none(self):
        w = channel_weights(_race_rows([("Engine", "mechanical"), ("Accident", "incident"),
                                         ("Finished", None), ("Illness", "other")]))
        assert not w["label_flagged"].any()
        assert list(w["w_mechanical"]) == [1.0, 0.0, 0.0, 0.0]
        assert list(w["w_incident"]) == [0.0, 1.0, 0.0, 0.0]

    def test_flagged_row_naming_a_system_leans_but_does_not_commit(self):
        w = channel_weights(_race_rows([("Suspension", "mechanical"), ("Puncture", "incident")]))
        assert w["w_mechanical"].iloc[0] == pytest.approx(FLAGGED_LABEL_LEAN)
        assert w["w_incident"].iloc[1] == pytest.approx(FLAGGED_LABEL_LEAN)

    def test_causeless_generic_split_follows_the_observable_eras_ratio(self):
        # three clean mechanicals to one clean incident -> a causeless "Retired"
        # row books 0.75 mechanical / 0.25 incident.
        w = channel_weights(_race_rows([("Engine", "mechanical"), ("Gearbox", "mechanical"),
                                         ("Brakes", "mechanical"), ("Accident", "incident"),
                                         ("Retired", "other")]))
        assert w["w_mechanical"].iloc[4] == pytest.approx(0.75)
        assert w["w_incident"].iloc[4] == pytest.approx(0.25)

    def test_build_makes_the_mixture_countable(self):
        data = build_reliability_data(
            _race_rows([("Engine", "mechanical"), ("Retired", "other"), ("Finished", None)]))
        assert (data.flagged_rows, data.clean_rows) == (1, 1)

    def test_real_2024_fixture_retirements_are_treated_as_a_mixture(self):
        """The shipped fixture's only two retirements are 2024 "Retired" rows --
        exactly the gap sec 3.2 describes, on real data."""
        data = build_reliability_data(pd.read_csv(FIXTURE_CSV))
        assert (data.flagged_rows, data.clean_rows) == (2, 0)
        # and neither channel got a whole retirement booked to it
        assert 0 < data.team_stats["mechanical_dnfs"].sum() < 2
        assert 0 < data.team_stats["incident_dnfs"].sum() < 2


def _incident_rounds():
    """CCC crashes a lot from the midfield; teammate DDD does not."""
    rows = []
    for i in range(20):
        rows.append(dict(season=2021, round=i + 1, team="Team C", abbreviation="CCC",
                          grid_position=9, session_type="R",
                          status="Accident" if i < 5 else "Finished",
                          dnf_cause="incident" if i < 5 else None))
        rows.append(dict(season=2021, round=i + 1, team="Team C", abbreviation="DDD",
                          grid_position=2, session_type="R", status="Finished", dnf_cause=None))
    return pd.DataFrame(rows)


class TestIncidentChannel:
    def test_driver_with_more_incidents_has_the_higher_posterior(self):
        data = build_reliability_data(_incident_rounds())
        a_c, b_c = driver_incident_posterior(data, "CCC")
        a_d, b_d = driver_incident_posterior(data, "DDD")
        assert a_c / (a_c + b_c) > a_d / (a_d + b_d)

    def test_hazard_is_per_driver_not_per_team(self):
        """Teammates share a mechanical draw; sharing an incident one would
        collapse sec 3.2's per-driver hazard back into the Phase 1 model."""
        data = build_reliability_data(_incident_rounds())
        rng = np.random.default_rng(3)
        assert (sample_incident_dnf_prob(data, "CCC", 9, 4000, rng).mean()
                > 1.5 * sample_incident_dnf_prob(data, "DDD", 9, 4000, rng).mean())

    def test_unknown_driver_falls_back_to_the_prior_and_draws_are_probabilities(self):
        data = build_reliability_data(_incident_rounds())
        a, b = driver_incident_posterior(data, "ZZZ")
        assert a / (a + b) == pytest.approx(PRIOR_MEAN_INCIDENT)
        draws = sample_incident_dnf_prob(data, "ZZZ", 9, 500, np.random.default_rng(0))
        assert draws.shape == (500,) and (draws >= 0).all() and (draws <= 1).all()


class TestGridPositionEffect:
    def test_bands_cover_the_grid_and_send_pit_lane_starts_to_the_back(self):
        assert grid_band(1) == grid_band(4) == "front"
        assert grid_band(5) == grid_band(14) == "midfield"
        assert grid_band(15) == grid_band(20) == "rear"
        assert grid_band(0) == "rear"        # pit-lane start, per slice-1
        assert grid_band(np.nan) == grid_band(None) == "rear"

    def test_midfield_beats_the_front_row_without_inflating_the_pooled_rate(self):
        """The bands redistribute the pooled incident rate across the grid; they
        must not inflate it. Half the toy field starts P9, half P2."""
        data = build_reliability_data(_incident_rounds())
        assert grid_multiplier(data, 9) > grid_multiplier(data, 1)
        mid, front = data.grid_multipliers["midfield"], data.grid_multipliers["front"]
        assert 0.5 * mid + 0.5 * front == pytest.approx(1.0)
        # and with no incidents at all to learn from, the prior still carries
        # sec 3.2's ordering rather than flattening the effect away
        clean = build_reliability_data(_race_rows([("Finished", None)] * 4))
        assert clean.grid_multipliers["front"] < clean.grid_multipliers["midfield"]

    def test_sampled_probability_tracks_the_band(self):
        data = build_reliability_data(_incident_rounds())
        rng = np.random.default_rng(1)
        assert (sample_incident_dnf_prob(data, "CCC", 9, 8000, rng).mean()
                > sample_incident_dnf_prob(data, "CCC", 1, 8000, rng).mean())


class TestCombinedChannels:
    def test_combined_exceeds_either_channel_and_stays_a_probability(self):
        m, i = np.array([0.05, 0.10]), np.array([0.04, 0.20])
        c = combined_dnf_prob(m, i)
        assert (c > m).all() and (c > i).all()
        assert c[0] == pytest.approx(1 - 0.95 * 0.96)
        assert list(combined_dnf_prob(np.array([0.0, 1.0]), np.array([0.0, 1.0]))) == [0.0, 1.0]


class TestLapOfRetirement:
    """Sec 3.2's lap-of-retirement distribution, kept simple."""

    def test_incident_distribution_spikes_on_lap_one(self):
        pmf = lap_of_retirement_pmf("incident", NOMINAL_RACE_LAPS)
        assert pmf[0] == pytest.approx(FIRST_LAP_HAZARD_MULTIPLIER * pmf[1])
        assert pmf[1:] == pytest.approx(pmf[1]) and pmf.sum() == pytest.approx(1.0)
        # the spike is a spike, not just a named constant: lap 1 carries more
        # than any later lap and more than a flat hazard would put there
        assert pmf[0] > pmf[1] and pmf[0] > 1.0 / NOMINAL_RACE_LAPS

    def test_mechanical_distribution_has_no_spike(self):
        flat = lap_of_retirement_pmf("mechanical", NOMINAL_RACE_LAPS)
        assert flat == pytest.approx(1.0 / NOMINAL_RACE_LAPS)

    def test_first_lap_share_grows_as_the_event_shortens(self):
        assert first_lap_incident_share(NOMINAL_SPRINT_LAPS) > first_lap_incident_share(NOMINAL_RACE_LAPS)
        assert first_lap_incident_share(1) == pytest.approx(1.0)

    @pytest.mark.parametrize("channel,n_laps", [("incident", 0), ("weather", 57)])
    def test_bad_arguments_raise(self, channel, n_laps):
        with pytest.raises(ValueError):
            lap_of_retirement_pmf(channel, n_laps)


# --- slice-6: the slice-3 carry-forwards ----------------------------------


class TestCauselessSplitIsNeverAHardLabel:
    """PR #8's first carry-forward. `channel_weights` splits a causeless flagged
    row on the observable era's clean mechanical:incident ratio, and that ratio
    is 0.0 or 1.0 when a fit window's clean labels all sit in one channel -- so
    the flagged row booked weight 1.0 to a single channel, which is a clean
    label in all but the bookkeeping. Sec 3.2's data-gap paragraph forbids
    exactly that. Unreachable on the full 2018-2026 window; reachable through
    the backtest's early `through_season` / `through_round` cutoffs.
    """

    def test_an_all_mechanical_clean_window_still_leaves_a_mixture(self):
        w = channel_weights(_race_rows([("Engine", "mechanical"), ("Gearbox", "mechanical"),
                                         ("Retired", "other")]))
        m, i = w["w_mechanical"].iloc[2], w["w_incident"].iloc[2]
        assert m == pytest.approx(1.0 - CAUSELESS_RATIO_FLOOR)
        assert i == pytest.approx(CAUSELESS_RATIO_FLOOR)
        assert 0.0 < m < 1.0 and 0.0 < i < 1.0 and m + i == pytest.approx(1.0)

    def test_the_clamp_is_symmetric(self):
        """An all-incident window is the same bug the other way up."""
        w = channel_weights(_race_rows([("Accident", "incident"), ("Collision", "incident"),
                                         ("Retired", "other")]))
        assert w["w_incident"].iloc[2] == pytest.approx(1.0 - CAUSELESS_RATIO_FLOOR)
        assert w["w_mechanical"].iloc[2] == pytest.approx(CAUSELESS_RATIO_FLOOR)

    def test_a_lopsided_window_short_of_the_floor_keeps_its_own_ratio(self):
        """The clamp is a floor, not a shrinkage: 18:1 is lopsided and left
        alone, because the floor only binds past 19:1."""
        specs = ([("Engine", "mechanical")] * 18 + [("Accident", "incident"),
                                                     ("Retired", "other")])
        w = channel_weights(_race_rows(specs))
        assert w["w_mechanical"].iloc[-1] == pytest.approx(18 / 19)

    def test_an_all_flagged_window_still_splits_evenly(self):
        """No clean labels at all was already handled correctly, and stays so --
        0.5 is not the floor applied twice."""
        w = channel_weights(_race_rows([("Retired", "other"), ("Retired", "other")]))
        assert list(w["w_mechanical"]) == pytest.approx([0.5, 0.5])
        assert list(w["w_incident"]) == pytest.approx([0.5, 0.5])

    def test_build_never_books_a_whole_retirement_to_one_channel(self):
        """The production path, not just the weighting helper: this is the case
        live in PR #8's own `test_build_makes_the_mixture_countable`, where the
        single "Retired" row landed at `w_mechanical = 1.0`."""
        data = build_reliability_data(
            _race_rows([("Engine", "mechanical"), ("Retired", "other")]))
        assert data.flagged_rows == 1
        assert float(data.team_stats["incident_dnfs"].sum()) > 0.0
        assert float(data.team_stats["mechanical_dnfs"].sum()) < 2.0


class TestSprintIncidentExposure:
    """PR #8's second carry-forward: the first-lap spike had no consumer. A
    sprint is the opening third of a Grand Prix's laps, and lap 1 is the whole
    point of the spike, so a sprint's incident exposure is not its distance
    share."""

    def test_a_sprint_carries_more_incident_risk_than_its_distance_share(self):
        ratio = NOMINAL_SPRINT_LAPS / NOMINAL_RACE_LAPS
        exposure = sprint_incident_exposure(ratio)
        assert exposure > ratio
        # 19 laps of a 57-lap profile whose lap 1 carries 12x the rest:
        # (12 + 18) / (12 + 56).
        assert exposure == pytest.approx(30 / 68)

    def test_exposure_is_the_lap_pmf_summed_over_the_sprint_window(self):
        """One definition of the spike in the codebase, not two."""
        pmf = lap_of_retirement_pmf("incident", NOMINAL_RACE_LAPS)
        assert sprint_incident_exposure(NOMINAL_SPRINT_LAPS / NOMINAL_RACE_LAPS) == pytest.approx(
            pmf[:NOMINAL_SPRINT_LAPS].sum())

    def test_the_real_sprint_distance_rounds_to_the_nominal_sprint_lap_count(self):
        """100 km of 305 over 57 laps is 18.7 laps, which rounds to the 19 of
        NOMINAL_SPRINT_LAPS -- the two constants agree rather than drifting."""
        from sim.race import SPRINT_DISTANCE_RATIO
        assert sprint_incident_exposure(SPRINT_DISTANCE_RATIO) == pytest.approx(
            sprint_incident_exposure(NOMINAL_SPRINT_LAPS / NOMINAL_RACE_LAPS))

    def test_a_full_distance_event_carries_all_of_the_risk(self):
        assert sprint_incident_exposure(1.0) == pytest.approx(1.0)

    def test_exposure_rises_with_distance_and_never_falls_below_lap_one(self):
        shares = [sprint_incident_exposure(r) for r in (0.1, 0.3, 0.6, 1.0)]
        assert shares == sorted(shares)
        assert shares[0] >= first_lap_incident_share(NOMINAL_RACE_LAPS)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_bad_distance_ratio_raises(self, bad):
        with pytest.raises(ValueError):
            sprint_incident_exposure(bad)
