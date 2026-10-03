"""Sprint weekends (spec sec 3.4): "treat as an extra shorter race event in
the round, own points table, sharing the weekend's pace draw with the Grand
Prix".

The shared pace draw is the part that is easy to get silently wrong -- a
sprint that drew its own pace would still look plausible in aggregate and
would only show up as a missing correlation between the two events. It gets
both a structural assertion (one array, not two) and a behavioural one
(same per-trial winner) with an explicit control showing what breaking it
would look like.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model.reliability import combined_dnf_prob, sprint_incident_exposure
from sim.race import (
    SPRINT_DISTANCE_RATIO, RaceTrialInputs, load_points_table, points_for_position,
    simulate_positions, simulate_weekend, sprint_inputs_from, summarize_trials,
)


def _inputs(race_pace, quali_pace=None, n_trials=4000, overtaking_difficulty=3.0,
            race_sigma=0.3, dnf_prob=None, grid=None,
            mechanical_dnf_prob=None, incident_dnf_prob=None):
    """`race_pace` is either a per-driver list (constant across trials) or an
    already-shaped (n_trials, n_drivers) array."""
    race_pace = np.asarray(race_pace, dtype=float)
    if race_pace.ndim == 1:
        race_pace = np.tile(race_pace, (n_trials, 1))
    n_trials, n_drivers = race_pace.shape
    quali_pace = race_pace if quali_pace is None else np.tile(np.asarray(quali_pace, dtype=float),
                                                               (n_trials, 1))
    return RaceTrialInputs(
        driver_ids=[f"d{i}" for i in range(n_drivers)],
        race_pace=race_pace,
        race_noise_nu=np.full(n_trials, 10.0),
        race_noise_sigma=np.full(n_trials, race_sigma),
        quali_pace=quali_pace,
        quali_noise_sigma=np.full(n_trials, 0.2),
        dnf_prob=np.zeros((n_trials, n_drivers)) if dnf_prob is None else dnf_prob,
        overtaking_difficulty=overtaking_difficulty,
        grid=grid,
        mechanical_dnf_prob=mechanical_dnf_prob,
        incident_dnf_prob=incident_dnf_prob,
    )


def _channelled(mechanical, incident, n_trials=4000, n_drivers=2, **kw):
    """Trial inputs carrying both sec 3.2 channels as well as their fold --
    what `cli.forecast_weekend` builds, and the only shape from which a sprint
    can price the first-lap spike."""
    mech = np.full((n_trials, n_drivers), mechanical)
    inc = np.full((n_trials, n_drivers), incident)
    return _inputs(race_pace=np.linspace(0.0, 0.5, n_drivers), n_trials=n_trials,
                    dnf_prob=combined_dnf_prob(mech, inc),
                    mechanical_dnf_prob=mech, incident_dnf_prob=inc, **kw)


class TestSprintPointsTable:
    """Sec 3.4's "own points table", read from data/points_tables.csv."""

    def test_grand_prix_table_is_unchanged(self):
        table = load_points_table()
        assert table[1] == 25.0 and table[10] == 1.0
        assert len(table) == 10

    def test_sprint_table_from_2022_scores_the_top_eight(self):
        table = load_points_table("sprint", 2024)
        assert len(table) == 8
        assert [table[p] for p in range(1, 9)] == [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        assert points_for_position(9, table) == 0.0

    def test_sprint_table_in_2021_scored_only_the_top_three(self):
        """The sprint table is not one table: 2021's first season awarded
        3/2/1, and a season-blind lookup would silently misprice it."""
        table = load_points_table("sprint", 2021)
        assert table == {1: 3.0, 2: 2.0, 3: 1.0}

    def test_season_defaults_to_the_current_rules(self):
        assert load_points_table("sprint") == load_points_table("sprint", 2026)

    def test_unknown_event_or_uncovered_season_raises(self):
        with pytest.raises(ValueError):
            load_points_table("shootout")
        with pytest.raises(ValueError):
            load_points_table("sprint", 2019)   # no sprints before 2021


class TestReconcilesWithRealSprintStandings:
    """Sec 8 Phase 2: "sprint points reconcile with actual standings". The
    fixture has no sprint weekend (see docs/autopilot/queue.md slice-1), so
    these are the true classifications, hard-coded, scored through the same
    `summarize_trials` path a forecast uses.
    """

    @staticmethod
    def _score(order, season):
        """Award points for a known finishing order, via the production
        scoring path: every trial is that order, nobody retires."""
        n = len(order)
        positions = np.tile(np.arange(1, n + 1), (8, 1))
        df = summarize_trials(positions, np.zeros((8, n), dtype=bool), list(order),
                               ["?"] * n, load_points_table("sprint", season))
        return df.set_index("driver")["exp_points"].to_dict()

    def test_2021_british_gp_sprint(self):
        # Silverstone, 17 July 2021 -- the first sprint ever run, on the
        # 3/2/1 table: Verstappen, Hamilton, Bottas, then Leclerc off the
        # points.
        awarded = self._score(["VER", "HAM", "BOT", "LEC"], season=2021)
        assert awarded == {"VER": 3.0, "HAM": 2.0, "BOT": 1.0, "LEC": 0.0}

    def test_2024_chinese_gp_sprint_podium(self):
        # Shanghai, 20 April 2024, on the 8..1 table: Verstappen from
        # Hamilton and Perez. Only the podium is asserted -- positions this
        # test does not claim to know are filled with placeholders.
        order = ["VER", "HAM", "PER"] + [f"p{i}" for i in range(4, 21)]
        awarded = self._score(order, season=2024)
        assert (awarded["VER"], awarded["HAM"], awarded["PER"]) == (8.0, 7.0, 6.0)
        assert awarded["p9"] == 0.0        # ninth is outside the sprint points
        assert sum(awarded.values()) == 36.0   # 8+7+6+5+4+3+2+1


class TestSharesTheWeekendPaceDraw:
    """Sec 3.4: the sprint shares the weekend's pace draw with the Grand Prix
    "rather than drawing independently"."""

    def test_sprint_reuses_the_grand_prix_pace_array_itself(self):
        gp = _inputs(race_pace=[0.0, 0.5, 1.0])
        sprint = sprint_inputs_from(gp)
        assert sprint.race_pace is gp.race_pace
        assert sprint.quali_pace is gp.quali_pace

    def test_per_trial_winner_matches_because_the_pace_is_shared(self):
        """Pace that flips between trials: with no race noise, the two events
        can only agree trial-by-trial if they are reading the same draw."""
        n_trials = 600
        pace = np.tile([0.0, 1.0, 2.0], (n_trials, 1))
        pace[1::2] = [2.0, 1.0, 0.0]        # the fast car alternates
        grid = np.array([1, 2, 3])
        gp = _inputs(pace, race_sigma=0.0, grid=grid)

        outcome = simulate_weekend(gp, np.random.default_rng(0), has_sprint=True, sprint_grid=grid)
        gp_winner = np.argmin(outcome.positions, axis=1)
        sprint_winner = np.argmin(outcome.sprint_positions, axis=1)
        assert (gp_winner == sprint_winner).all()
        assert len(set(gp_winner)) == 2     # the draw really does vary by trial

        # Control: the same resolver on an independently-drawn pace (here,
        # the trials reordered) agrees only by chance. This is the silent bug
        # the assertion above exists to catch.
        independent = simulate_positions(
            sprint_inputs_from(replace(gp, race_pace=gp.race_pace[::-1]), grid=grid),
            np.random.default_rng(0))[0]
        assert (np.argmin(independent, axis=1) == gp_winner).mean() < 0.1

    def test_each_event_still_draws_its_own_noise_and_dnfs(self):
        """Shared pace is not a shared result -- a sprint retirement does not
        put the car out of Sunday's race, and vice versa."""
        n_trials = 4000
        dnf_prob = np.full((n_trials, 3), 0.5)
        gp = _inputs(race_pace=[0.0, 0.5, 1.0], n_trials=n_trials, dnf_prob=dnf_prob)
        outcome = simulate_weekend(gp, np.random.default_rng(0), has_sprint=True)
        agree = (outcome.dnf == outcome.sprint_dnf).mean()
        assert 0.4 < agree < 0.75           # independent draws, not a copy


class TestSprintIsShorter:
    def test_fewer_laps_means_fewer_retirements(self):
        n_trials = 20_000
        dnf_prob = np.full((n_trials, 2), 0.3)
        gp = _inputs(race_pace=[0.0, 0.5], n_trials=n_trials, dnf_prob=dnf_prob)
        outcome = simulate_weekend(gp, np.random.default_rng(0), has_sprint=True)
        assert outcome.sprint_dnf.mean() == pytest.approx(0.3 * SPRINT_DISTANCE_RATIO, abs=0.01)
        assert outcome.dnf.mean() == pytest.approx(0.3, abs=0.01)

    def test_fewer_laps_means_the_grid_matters_more(self):
        """A third of the distance is a third of the chances to pass, so the
        same circuit is more grid-locked on Saturday than on Sunday."""
        # A pace advantage worth two grid slots on Sunday (0.55/5 = 0.11s per
        # slot) but not on Saturday, where a third of the laps makes the same
        # slot cost 0.335s.
        grid = np.array([3, 2, 1])          # the fastest car starts last
        gp = _inputs(race_pace=[0.0, 0.25, 0.5], overtaking_difficulty=5.0,
                      race_sigma=0.05, grid=grid)
        outcome = simulate_weekend(gp, np.random.default_rng(0), has_sprint=True, sprint_grid=grid)
        gp_recovers = (outcome.positions[:, 0] == 1).mean()
        sprint_recovers = (outcome.sprint_positions[:, 0] == 1).mean()
        assert gp_recovers > 0.9
        assert sprint_recovers < 0.1

    def test_distance_ratio_must_be_a_fraction(self):
        gp = _inputs(race_pace=[0.0, 1.0])
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError):
                sprint_inputs_from(gp, distance_ratio=bad)


class TestNonSprintRoundsAreUnaffected:
    def test_weekend_without_a_sprint_is_exactly_the_old_resolver(self):
        gp = _inputs(race_pace=[0.0, 0.4, 0.8], dnf_prob=np.full((4000, 3), 0.1))
        expected, expected_dnf = simulate_positions(gp, np.random.default_rng(7))
        outcome = simulate_weekend(gp, np.random.default_rng(7), has_sprint=False)
        assert not outcome.has_sprint
        assert outcome.sprint_positions is None and outcome.sprint_dnf is None
        assert (outcome.positions == expected).all()
        assert (outcome.dnf == expected_dnf).all()

    def test_adding_a_sprint_does_not_move_the_grand_prix_forecast(self):
        """The Grand Prix takes its draws off the RNG first, so a round's
        Sunday numbers are identical whether or not Saturday is simulated."""
        gp = _inputs(race_pace=[0.0, 0.4, 0.8], dnf_prob=np.full((4000, 3), 0.1))
        without = simulate_weekend(gp, np.random.default_rng(3), has_sprint=False)
        with_ = simulate_weekend(gp, np.random.default_rng(3), has_sprint=True)
        assert (without.positions == with_.positions).all()
        assert (without.dnf == with_.dnf).all()


class TestSprintIncidentRiskReadsTheFirstLapSpike:
    """slice-6, carried forward from PR #8. `sprint_inputs_from` used to scale
    the whole of `dnf_prob` by the distance ratio, which prices a sprint as if
    it had a third of a lap 1. Sec 3.2 puts a spike on lap 1, and a sprint has
    all of it, so the incident channel scales by
    `reliability.sprint_incident_exposure` and only the mechanical channel
    scales with distance.
    """

    MECH, INC = 0.06, 0.05

    def _expected_sprint_dnf(self):
        return combined_dnf_prob(self.MECH * SPRINT_DISTANCE_RATIO,
                                  self.INC * sprint_incident_exposure(SPRINT_DISTANCE_RATIO))

    def test_the_two_channels_shorten_separately(self):
        sprint = sprint_inputs_from(_channelled(self.MECH, self.INC))
        assert sprint.mechanical_dnf_prob == pytest.approx(self.MECH * SPRINT_DISTANCE_RATIO)
        assert sprint.incident_dnf_prob == pytest.approx(
            self.INC * sprint_incident_exposure(SPRINT_DISTANCE_RATIO))
        # the point of the slice: incident risk is NOT the distance share
        assert sprint.incident_dnf_prob.mean() > self.INC * SPRINT_DISTANCE_RATIO
        assert sprint.dnf_prob == pytest.approx(self._expected_sprint_dnf())

    def test_sprint_retirements_exceed_the_old_flat_scaling(self):
        """The behavioural form: more cars retire in the simulated sprint than
        the pre-slice-6 flat scaling of the combined probability produced, by
        about the spike's share of the incident channel."""
        n_trials = 40_000
        gp = _channelled(self.MECH, self.INC, n_trials=n_trials)
        outcome = simulate_weekend(gp, np.random.default_rng(0), has_sprint=True)
        expected = self._expected_sprint_dnf()
        flat = combined_dnf_prob(self.MECH, self.INC) * SPRINT_DISTANCE_RATIO
        assert outcome.sprint_dnf.mean() == pytest.approx(expected, abs=0.004)
        assert expected > flat * 1.1          # ~18% more, not a rounding artefact
        assert outcome.sprint_dnf.mean() > flat

    def test_a_sprint_with_no_incident_risk_is_still_flat_in_distance(self):
        """The change is confined to the incident channel: zero out the incident
        hazard and the sprint is exactly the pre-slice-6 distance scaling."""
        sprint = sprint_inputs_from(_channelled(0.3, 0.0))
        assert sprint.dnf_prob == pytest.approx(0.3 * SPRINT_DISTANCE_RATIO)

    def test_a_caller_holding_only_the_combined_probability_is_unaffected(self):
        """Nothing can price a spike it was not handed the channels for, so that
        caller keeps the flat scaling rather than silently guessing a split."""
        gp = _inputs(race_pace=[0.0, 1.0], dnf_prob=np.full((4000, 2), 0.3))
        sprint = sprint_inputs_from(gp)
        assert sprint.dnf_prob == pytest.approx(0.3 * SPRINT_DISTANCE_RATIO)
        assert sprint.mechanical_dnf_prob is None and sprint.incident_dnf_prob is None

    def test_a_shorter_sprint_still_carries_the_whole_spike(self):
        """Halve the distance and the mechanical channel halves with it; the
        incident channel does not, because lap 1 does not get shorter."""
        half = sprint_inputs_from(_channelled(self.MECH, self.INC), distance_ratio=0.5)
        full = sprint_inputs_from(_channelled(self.MECH, self.INC), distance_ratio=1.0)
        assert half.mechanical_dnf_prob == pytest.approx(0.5 * full.mechanical_dnf_prob)
        assert half.incident_dnf_prob.mean() > 0.5 * full.incident_dnf_prob.mean()
        assert full.incident_dnf_prob == pytest.approx(self.INC)


class TestRaceOutputsAreUnchangedByTheChannels:
    """slice-6 criterion 3. The resolver draws against `dnf_prob`; the channels
    exist so that a *shorter* event can rescale them, and must not touch the
    Grand Prix."""

    def test_carrying_the_channels_does_not_move_the_grand_prix(self):
        n_trials = 20_000
        mech, inc = 0.06, 0.05
        combined = combined_dnf_prob(np.full((n_trials, 2), mech), np.full((n_trials, 2), inc))
        without = _inputs(race_pace=[0.0, 0.5], n_trials=n_trials, dnf_prob=combined)
        with_ = _channelled(mech, inc, n_trials=n_trials)
        assert with_.dnf_prob == pytest.approx(without.dnf_prob)

        a = simulate_weekend(without, np.random.default_rng(5), has_sprint=True)
        b = simulate_weekend(with_, np.random.default_rng(5), has_sprint=True)
        assert (a.positions == b.positions).all()
        assert (a.dnf == b.dnf).all()
        # ...and the sprint is the only thing that moved
        assert b.sprint_dnf.mean() > a.sprint_dnf.mean()

    def test_a_round_without_a_sprint_is_identical_either_way(self):
        mech, inc = 0.06, 0.05
        combined = combined_dnf_prob(np.full((4000, 2), mech), np.full((4000, 2), inc))
        a = simulate_weekend(_inputs(race_pace=[0.0, 0.5], dnf_prob=combined),
                              np.random.default_rng(9), has_sprint=False)
        b = simulate_weekend(_channelled(mech, inc), np.random.default_rng(9), has_sprint=False)
        assert (a.positions == b.positions).all() and (a.dnf == b.dnf).all()
        assert not b.has_sprint
