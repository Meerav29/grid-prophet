"""Season chaining (spec sec 3.4): both championships from one set of trials.

Each trial starts from one posterior draw of `car[t]` at the last fitted round
and, before each future round, steps every team's car by the pace model's
random walk (sec 3.1). The step lands on the running value, not the fitted
one, so the wander compounds over the rest of the year. Each round runs
through `sim.race.simulate_weekend` (a sprint shares its Grand Prix's pace
draw). Team points are the sum of that team's drivers' points in the same
trial, so the two championships are consistent by construction.

Everything else that is a posterior parameter (driver skill, track
interaction, quali offset, noise scales, both DNF hazards) is drawn once per
trial and held for the season: only `car` has a random walk in sec 3.1.

The remaining calendar comes from FastF1 through `remaining_calendar`'s
`loader` seam. If it cannot be read the command fails; it never invents one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sim.race import RaceTrialInputs, points_for_trials, simulate_weekend


class ScheduleUnavailable(RuntimeError):
    """The season's event schedule could not be read. `gp season` stops."""


def _fastf1_schedule(season: int) -> pd.DataFrame:
    import fastf1

    return fastf1.get_event_schedule(season, include_testing=False)


def remaining_calendar(season: int, after_round: int, loader=None) -> pd.DataFrame:
    """Rounds after `after_round` as `round, event_name, has_sprint`, from a
    FastF1-shaped `loader(season)`. Sprints are read off `EventFormat` as in
    `collect_v2`. An unreadable or empty schedule raises; an empty result
    only ever means the season is over."""
    loader = loader or _fastf1_schedule
    try:
        schedule = loader(season)
    except Exception as exc:
        raise ScheduleUnavailable(
            f"Could not read the {season} event schedule from FastF1 ({exc}). "
            "gp season will not guess the remaining calendar.") from exc
    needed = {"RoundNumber", "EventName", "EventFormat"}
    if schedule is None or len(schedule) == 0 or not needed.issubset(schedule.columns):
        raise ScheduleUnavailable(
            f"The {season} event schedule came back empty or without {sorted(needed)}. "
            "gp season will not guess the remaining calendar.")
    round_numbers = schedule["RoundNumber"].astype(int)
    rounds = schedule[round_numbers > max(after_round, 0)].sort_values("RoundNumber")
    return pd.DataFrame({
        "round": rounds["RoundNumber"].astype(int).to_numpy(),
        "event_name": rounds["EventName"].astype(str).to_numpy(),
        "has_sprint": rounds["EventFormat"].astype(str).str.lower().str.contains("sprint").to_numpy(),
    })


def step_sigma_for_round(season: int, round_: int) -> float:
    """Random-walk step into `round_`, as `model.pace._step_sigma_scale` within a season."""
    from model.pace import (
        RULE_CHANGE_EARLY_ROUNDS, RULE_CHANGE_EARLY_SCALE, RULE_CHANGE_SEASONS, SIGMA_DEV,
    )

    early = season in RULE_CHANGE_SEASONS and round_ - 1 < RULE_CHANGE_EARLY_ROUNDS
    return SIGMA_DEV * (RULE_CHANGE_EARLY_SCALE if early else 1.0)


@dataclass
class SeasonRound:
    round: int
    has_sprint: bool
    circuit_type_idx: int            # column of `SeasonTrialInputs.driver_track`
    overtaking_difficulty: float
    step_sigma: float                # sd of the car random-walk step into this round


@dataclass
class SeasonTrialInputs:
    """One posterior draw per trial, held for the whole season except `car`."""
    driver_ids: list
    teams: list                      # per driver, len n_drivers
    team_names: list                 # columns of `car`
    car: np.ndarray                  # (n_trials, n_teams) at the last fitted round
    driver_skill: np.ndarray         # (n_trials, n_drivers)
    driver_track: np.ndarray         # (n_trials, n_drivers, n_circuit_types)
    quali_offset: np.ndarray         # (n_trials, n_drivers)
    race_noise_nu: np.ndarray        # (n_trials,)
    race_noise_sigma: np.ndarray     # (n_trials,)
    quali_noise_sigma: np.ndarray    # (n_trials,)
    mechanical_dnf_prob: np.ndarray  # (n_trials, n_drivers)
    incident_dnf_prob: np.ndarray    # (n_trials, n_drivers)


@dataclass
class SeasonOutcome:
    """Simulated rounds only; `cli` adds the points already scored."""
    driver_points: np.ndarray        # (n_trials, n_drivers)
    driver_wins: np.ndarray          # (n_trials, n_drivers), Grand Prix wins
    team_points: np.ndarray          # (n_trials, n_teams)
    team_wins: np.ndarray            # (n_trials, n_teams)
    car_path: np.ndarray             # (n_trials, n_rounds, n_teams), car at each simulated round


def simulate_season(inputs: SeasonTrialInputs, rounds: list, gp_points: dict, sprint_points: dict,
                     rng: np.random.Generator) -> SeasonOutcome:
    """Chain `rounds` in order within every trial (sec 3.4)."""
    from model.reliability import combined_dnf_prob

    n_trials, n_drivers = inputs.driver_skill.shape
    team_col = np.array([inputs.team_names.index(t) for t in inputs.teams])
    dnf_prob = combined_dnf_prob(inputs.mechanical_dnf_prob, inputs.incident_dnf_prob)

    car = np.array(inputs.car, dtype=float, copy=True)
    car_path = np.zeros((n_trials, len(rounds), car.shape[1]))
    points = np.zeros((n_trials, n_drivers))
    wins = np.zeros((n_trials, n_drivers))
    for k, rnd in enumerate(rounds):
        # The step lands on the running value: this is the carry-forward.
        car = car + rng.normal(0.0, rnd.step_sigma, size=car.shape)
        car_path[:, k] = car
        car_d = car[:, team_col]
        race_inputs = RaceTrialInputs(
            driver_ids=inputs.driver_ids,
            race_pace=car_d + inputs.driver_skill + inputs.driver_track[:, :, rnd.circuit_type_idx],
            race_noise_nu=inputs.race_noise_nu, race_noise_sigma=inputs.race_noise_sigma,
            quali_pace=car_d + inputs.driver_skill + inputs.quali_offset,
            quali_noise_sigma=inputs.quali_noise_sigma,
            dnf_prob=dnf_prob,
            mechanical_dnf_prob=inputs.mechanical_dnf_prob,
            incident_dnf_prob=inputs.incident_dnf_prob,
            overtaking_difficulty=rnd.overtaking_difficulty,
        )
        outcome = simulate_weekend(race_inputs, rng, has_sprint=rnd.has_sprint)
        points += points_for_trials(outcome.positions, outcome.dnf, gp_points)
        wins += (outcome.positions == 1) & ~outcome.dnf
        if outcome.has_sprint:
            points += points_for_trials(outcome.sprint_positions, outcome.sprint_dnf, sprint_points)

    membership = np.zeros((n_drivers, len(inputs.team_names)))
    membership[np.arange(n_drivers), team_col] = 1.0
    return SeasonOutcome(driver_points=points, driver_wins=wins,
                          team_points=points @ membership, team_wins=wins @ membership,
                          car_path=car_path)


def final_positions(points: np.ndarray, wins: np.ndarray) -> np.ndarray:
    """1-indexed championship position per trial; points ties go to more GP wins."""
    # Points come in steps of >= 0.5, wins are < 50 a season, so this key
    # orders by points first and wins second without a per-trial lexsort.
    key = np.asarray(points, dtype=float) * 100.0 + np.asarray(wins, dtype=float)
    order = np.argsort(-key, axis=1, kind="stable")
    return np.argsort(order, axis=1, kind="stable") + 1


def _position_columns(pos: np.ndarray) -> dict:
    return {"p10_final_position": float(np.percentile(pos, 10)),
            "p50_final_position": float(np.percentile(pos, 50)),
            "p90_final_position": float(np.percentile(pos, 90))}


def _points_interval(points: np.ndarray) -> dict:
    """The 80% interval on final points, which the season backtest scores (sec 7)."""
    return {"p10_final_points": float(np.percentile(points, 10)),
            "p90_final_points": float(np.percentile(points, 90))}


def drivers_championship(total_points: np.ndarray, total_wins: np.ndarray, driver_ids: list,
                          teams: list, current_points) -> pd.DataFrame:
    """sec 6 `drivers_championship.csv`, plus `team`, `current_points` and
    the 80% interval on final points."""
    pos = final_positions(total_points, total_wins)
    rows = []
    for i, driver in enumerate(driver_ids):
        rows.append({"driver": driver, "team": teams[i], "current_points": float(current_points[i]),
                     "p_title": float(np.mean(pos[:, i] == 1)),
                     "exp_final_points": float(np.mean(total_points[:, i])),
                     **_position_columns(pos[:, i]),
                     "p_top3": float(np.mean(pos[:, i] <= 3)),
                     **_points_interval(total_points[:, i])})
    return pd.DataFrame(rows).sort_values("exp_final_points", ascending=False).reset_index(drop=True)


def constructors_championship(total_points: np.ndarray, total_wins: np.ndarray, team_names: list,
                               current_points) -> pd.DataFrame:
    """sec 6 `constructors_championship.csv`, plus `current_points` and the
    80% interval on final points.
    `exp_share` is the per-trial share of all constructors' points, averaged."""
    pos = final_positions(total_points, total_wins)
    totals = total_points.sum(axis=1, keepdims=True)
    share = np.divide(total_points, totals, out=np.zeros_like(total_points), where=totals > 0)
    rows = []
    for j, team in enumerate(team_names):
        rows.append({"team": team, "current_points": float(current_points[j]),
                     "p_title": float(np.mean(pos[:, j] == 1)),
                     "exp_final_points": float(np.mean(total_points[:, j])),
                     "exp_share": float(np.mean(share[:, j])),
                     **_position_columns(pos[:, j]),
                     **_points_interval(total_points[:, j])})
    return pd.DataFrame(rows).sort_values("exp_final_points", ascending=False).reset_index(drop=True)
