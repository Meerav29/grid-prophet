"""Race/weekend resolver, pre-weekend and post-quali modes (spec sec 3.3).

Per simulated trial:
  1. Draw car/driver/interaction terms from the posterior (done by the
     caller via `model.pace.posterior_*_draws`; this module consumes the
     resulting arrays).
  2. Compute each driver's pace; add race noise.
  3. Pre-weekend mode: simulate quali first (its own pace + noise) to get a
     grid. Post-quali mode: use the round's real grid instead (`inputs.grid`,
     identical across trials). Either way, apply a track-specific
     overtaking-difficulty parameter to turn (grid, race pace) into a
     finishing order. The value comes from `model.overtaking`, which fits it
     per circuit from observed grid-to-finish changes and shrinks it toward
     the `data/circuits.csv` hand rating; circuits with no usable history
     still get the hand rating itself.
  4. Draw DNFs against `inputs.dnf_prob`, which `model.reliability` folds
     from sec 3.2's mechanical and incident channels.
  5. Order survivors by effective pace, convert to points via the points
     table.

A sprint weekend runs the same resolver twice (`simulate_weekend`, spec
sec 3.4: "an extra shorter race event in the round, own points table,
sharing the weekend's pace draw with the Grand Prix"). The two events share
the trial's pace draw -- the same `race_pace` array object, not a second
draw from the posterior -- and differ only in the things a third of
the race distance actually changes: fewer laps to recover grid positions,
fewer laps over which to break, a first lap that is not a third of anything
(sec 3.2's incident spike, priced by `reliability.sprint_incident_exposure`),
and the sprint points table.

The Monte Carlo core (`simulate_positions`) is pure numpy, takes already-
drawn per-trial pace/noise/dnf arrays, and is deterministic given a seeded
RNG -- this is the part covered by unit tests (tests/test_race_resolver.py).
Posterior sampling happens once per `gp fit`, not per trial, per sec 3.3's
"the bottleneck is posterior sampling, which happens once per model update,
not per trial".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
POINTS_TABLE_CSV = os.path.join(DATA_DIR, "points_tables.csv")

# Seconds of "effective time" penalty per grid slot when overtaking is
# maximally hard (overtaking_difficulty == 1, e.g. Monaco): starting further
# back costs roughly this much per slot, and pace differences smaller than
# that can't claw the place back within the race. Scaled down as
# overtaking_difficulty rises (1=hardest, ~5=easiest). This constant also
# fixes the units `model.overtaking` fits in -- changing it rescales every
# fitted difficulty, so the two move together and neither is free to drift.
GRID_LOCK_BASE_PENALTY_S = 0.55


def load_points_table(event: str = "grand_prix", season: int | None = None,
                       path: str = POINTS_TABLE_CSV) -> dict:
    """{position: points} for one event type in one season.

    `data/points_tables.csv` carries one block per (event, season range):
    the Grand Prix table, and the sprint table sec 3.4 requires -- which is
    not one table but two, 3/2/1 to the top three in 2021 and 8..1 to the
    top eight from 2022. `season=None` takes the most recent block, which is
    what a forecast for the current season wants.
    """
    df = pd.read_csv(path)
    rows = df[df["event"] == event]
    if rows.empty:
        raise ValueError(f"No points table for event={event!r} in {path}")
    if season is None:
        rows = rows[rows["season_from"] == rows["season_from"].max()]
    else:
        rows = rows[(rows["season_from"] <= season) & (season <= rows["season_to"])]
        if rows.empty:
            raise ValueError(f"No {event} points table covering season {season} in {path}")
    return dict(zip(rows["position"].astype(int), rows["points"].astype(float)))


def points_for_position(position: int, table: dict) -> float:
    return table.get(int(position), 0.0)


@dataclass
class RaceTrialInputs:
    """Per-driver arrays of shape (n_trials,) for one simulated race."""
    driver_ids: list                 # length n_drivers, stable ordering
    race_pace: np.ndarray            # (n_trials, n_drivers) -- latent car+driver+interaction
    race_noise_nu: np.ndarray        # (n_trials,) Student-t dof, shared draw per trial
    race_noise_sigma: np.ndarray     # (n_trials,)
    quali_pace: np.ndarray           # (n_trials, n_drivers)
    quali_noise_sigma: np.ndarray    # (n_trials,)
    dnf_prob: np.ndarray             # (n_trials, n_drivers) -- P(retired), both channels
    overtaking_difficulty: float     # circuits.csv value, 1 (hard) .. ~5 (easy)
    grid: np.ndarray = None          # (n_drivers,) 1-indexed real grid, post-quali mode only
    # The two sec 3.2 channels `dnf_prob` was folded from, carried alongside it
    # because they do not shorten alike: a shorter event scales mechanical risk
    # with distance and incident risk with the lap distribution (see
    # `sprint_inputs_from`). `dnf_prob` stays what the resolver draws against,
    # so a caller that has only the combined number leaves these None and gets
    # the pre-slice-6 flat scaling.
    mechanical_dnf_prob: np.ndarray = None   # (n_trials, n_drivers)
    incident_dnf_prob: np.ndarray = None     # (n_trials, n_drivers)


def simulate_positions(inputs: RaceTrialInputs, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Run the Monte Carlo race resolution. Returns (positions, dnf) both
    shaped (n_trials, n_drivers): `positions` is 1-indexed finishing position
    (DNF'd drivers get positions after all classified finishers, order
    among them arbitrary), `dnf` is a boolean DNF mask.
    """
    n_trials, n_drivers = inputs.race_pace.shape

    # --- grid: post-quali mode uses the round's real starting order, identical
    # in every trial (it is an observed fact by then, not a quantity with a
    # posterior); pre-weekend mode simulates quali from the quali equation ---
    if inputs.grid is not None:
        grid = np.tile(np.asarray(inputs.grid, dtype=int), (n_trials, 1))
    else:
        quali_eps = rng.normal(0.0, 1.0, size=(n_trials, n_drivers)) * inputs.quali_noise_sigma[:, None]
        grid = np.argsort(np.argsort(inputs.quali_pace + quali_eps, axis=1), axis=1) + 1

    # --- race: pace + Student-t noise ---
    # Student-t via normal / sqrt(chi2/nu), vectorised per trial (nu, sigma shared within a trial).
    z = rng.standard_normal(size=(n_trials, n_drivers))
    chi2 = rng.chisquare(inputs.race_noise_nu)[:, None]
    t_noise = z / np.sqrt(np.clip(chi2, 1e-6, None) / inputs.race_noise_nu[:, None])
    race_eps = t_noise * inputs.race_noise_sigma[:, None]
    race_time = inputs.race_pace + race_eps

    # --- overtaking-difficulty grid-lock penalty ---
    penalty_per_slot = GRID_LOCK_BASE_PENALTY_S / max(inputs.overtaking_difficulty, 1e-6)
    effective_time = race_time + penalty_per_slot * (grid - 1)

    # --- DNFs ---
    dnf = rng.random(size=(n_trials, n_drivers)) < inputs.dnf_prob

    # DNF'd cars are sorted after all classified cars, regardless of effective_time.
    sort_key = np.where(dnf, effective_time + 1e6, effective_time)
    positions = np.argsort(np.argsort(sort_key, axis=1), axis=1) + 1

    return positions, dnf


# Sprint distance as a fraction of a Grand Prix: ~100 km against ~305 km
# (spec sec 3.4, "an extra shorter race event in the round"). One number
# drives both sprint adjustments below, so there is one place to change if a
# future format moves the distance.
SPRINT_DISTANCE_RATIO = 100.0 / 305.0


@dataclass
class WeekendOutcome:
    """One sprint weekend's trials. `sprint_positions`/`sprint_dnf` are None
    on a round without a sprint, which is how callers tell the two apart."""
    positions: np.ndarray                    # (n_trials, n_drivers), the Grand Prix
    dnf: np.ndarray
    sprint_positions: np.ndarray = None
    sprint_dnf: np.ndarray = None

    @property
    def has_sprint(self) -> bool:
        return self.sprint_positions is not None


def sprint_inputs_from(inputs: RaceTrialInputs, grid: np.ndarray = None,
                        distance_ratio: float = SPRINT_DISTANCE_RATIO) -> RaceTrialInputs:
    """The Grand Prix's trial inputs, re-scaled for the sprint.

    `race_pace` and `quali_pace` are carried over **by reference**: sec 3.4
    is explicit that the sprint shares the weekend's pace draw with the
    Grand Prix rather than drawing independently, and the cheapest way to
    make that true (and to keep it true) is for there to be exactly one
    array. Three things do change over a third of the distance:

    * `dnf_prob` shortens, but the two channels sec 3.2 folded it from do not
      shorten alike. **Mechanical** risk scales with distance: a third of the
      running time is a third of the exposure to a failure that is flat in lap
      number. **Incident** risk does not, because a sprint still has a lap 1
      and that is where sec 3.2 puts its spike -- so it scales by
      `reliability.sprint_incident_exposure`, which reads the share off the
      lap-of-retirement distribution (0.44 at the real sprint distance, against
      a distance share of 0.33). Scaling the combined probability flat, as this
      function did before slice-6, understated sprint retirements by about the
      spike's share of the incident channel. A caller that supplies only
      `dnf_prob` and neither channel still gets that flat scaling -- there is
      nothing to price the spike against.
    * `overtaking_difficulty` scales with distance too, which makes the
      grid-lock penalty per slot 1/ratio times larger: the same circuit is
      harder to pass on when there are a third as many laps to do it in.
    * `grid` is the sprint's own starting order (from sprint quali where we
      have it; None re-simulates it from the shared quali pace, with its own
      noise draw, because it is a separate session).

    Race noise is deliberately left alone. Per-lap noise would average down
    over distance and event-level shocks (a safety car, a bad start) would
    not, and the spec does not say which dominates; scaling it either way
    would be a modelling claim the data has not been asked about yet.
    """
    if not 0 < distance_ratio <= 1:
        raise ValueError(f"distance_ratio must be in (0, 1], got {distance_ratio}")
    # Imported here, not at module scope: every other cross-package reference
    # in this repo (cli, backtest) is a local import, and `sim` has not needed
    # `model` until now. Once per weekend, not once per trial.
    from model.reliability import combined_dnf_prob, sprint_incident_exposure

    has_channels = inputs.mechanical_dnf_prob is not None and inputs.incident_dnf_prob is not None
    if has_channels:
        mechanical = inputs.mechanical_dnf_prob * distance_ratio
        incident = inputs.incident_dnf_prob * sprint_incident_exposure(distance_ratio)
        dnf_prob = combined_dnf_prob(mechanical, incident)
    else:
        mechanical = incident = None
        dnf_prob = inputs.dnf_prob * distance_ratio
    return replace(
        inputs,
        dnf_prob=dnf_prob,
        mechanical_dnf_prob=mechanical,
        incident_dnf_prob=incident,
        overtaking_difficulty=inputs.overtaking_difficulty * distance_ratio,
        grid=grid,
    )


def simulate_weekend(inputs: RaceTrialInputs, rng: np.random.Generator,
                      has_sprint: bool = False, sprint_grid: np.ndarray = None,
                      distance_ratio: float = SPRINT_DISTANCE_RATIO) -> WeekendOutcome:
    """Resolve a round's events off one trial pace draw.

    Each event gets its own noise and its own DNF draw from `rng` -- they are
    separate races, and a car that breaks in the sprint is not thereby out of
    Sunday's. What they share is the latent pace, which is the point of sec
    3.4.

    The Grand Prix is drawn first even though the sprint runs first in the
    real weekend. Nothing carries between the two events, so the order is
    free, and taking the Grand Prix's draws off `rng` first means adding a
    sprint to a round cannot move that round's Grand Prix forecast by a
    single trial: for a given seed the Sunday numbers are identical with the
    sprint and without it.
    """
    positions, dnf = simulate_positions(inputs, rng)
    if not has_sprint:
        return WeekendOutcome(positions=positions, dnf=dnf)

    sprint_inputs = sprint_inputs_from(inputs, grid=sprint_grid, distance_ratio=distance_ratio)
    sprint_positions, sprint_dnf = simulate_positions(sprint_inputs, rng)
    return WeekendOutcome(positions=positions, dnf=dnf,
                           sprint_positions=sprint_positions, sprint_dnf=sprint_dnf)


def summarize_trials(positions: np.ndarray, dnf: np.ndarray, driver_ids: list, teams: list,
                      points_table: dict) -> pd.DataFrame:
    """Turn (n_trials, n_drivers) position/dnf arrays into race_forecast.csv rows."""
    n_trials, n_drivers = positions.shape
    points_by_pos = np.zeros(n_drivers + 1)
    for pos, pts in points_table.items():
        if 1 <= pos <= n_drivers:
            points_by_pos[pos] = pts
    trial_points = points_by_pos[positions]
    trial_points = np.where(dnf, 0.0, trial_points)

    rows = []
    for i, driver in enumerate(driver_ids):
        pos_i = positions[:, i]
        row = {
            "driver": driver,
            "team": teams[i] if teams is not None else None,
            "p_win": float(np.mean(pos_i == 1)),
            "p_podium": float(np.mean(pos_i <= 3)),
            "p_points": float(np.mean(trial_points[:, i] > 0)),
            "p_dnf": float(np.mean(dnf[:, i])),
            "exp_points": float(np.mean(trial_points[:, i])),
            "p10_finish": float(np.percentile(pos_i, 10)),
            "p50_finish": float(np.percentile(pos_i, 50)),
            "p90_finish": float(np.percentile(pos_i, 90)),
        }
        # sec 6: "plus the full position distribution as a wide block"
        for pos in range(1, n_drivers + 1):
            row[f"p_pos_{pos}"] = float(np.mean(pos_i == pos))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("exp_points", ascending=False).reset_index(drop=True)
