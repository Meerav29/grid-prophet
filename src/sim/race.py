"""Single-race resolver, pre-weekend and post-quali modes (spec sec 3.3).

Per simulated trial:
  1. Draw car/driver/interaction terms from the posterior (done by the
     caller via `model.pace.posterior_*_draws`; this module consumes the
     resulting arrays).
  2. Compute each driver's pace; add race noise.
  3. Pre-weekend mode: simulate quali first (its own pace + noise) to get a
     grid. Post-quali mode: use the round's real grid instead (`inputs.grid`,
     identical across trials). Either way, apply a track-specific
     overtaking-difficulty parameter to turn (grid, race pace) into a
     finishing order -- a fixed per-circuit value from `data/circuits.csv`,
     exactly as sec 3.3 specifies for phase 1 ("fixed per-circuit overtaking
     parameter"; fitting it from data is phase 2).
  4. Draw DNFs (team-level mechanical hazard only, sec 3.2 first cut).
  5. Order survivors by effective pace, convert to points via the points
     table.

The Monte Carlo core (`simulate_positions`) is pure numpy, takes already-
drawn per-trial pace/noise/dnf arrays, and is deterministic given a seeded
RNG -- this is the part covered by unit tests (tests/test_race_resolver.py).
Posterior sampling happens once per `gp fit`, not per trial, per sec 3.3's
"the bottleneck is posterior sampling, which happens once per model update,
not per trial".
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
POINTS_TABLE_CSV = os.path.join(DATA_DIR, "points_tables.csv")

# Seconds of "effective time" penalty per grid slot when overtaking is
# maximally hard (overtaking_difficulty == 1, e.g. Monaco): starting further
# back costs roughly this much per slot, and pace differences smaller than
# that can't claw the place back within the race. Scaled down as
# overtaking_difficulty rises (data/circuits.csv, 1=hardest, ~5=easiest).
GRID_LOCK_BASE_PENALTY_S = 0.55


def load_points_table(path: str = POINTS_TABLE_CSV) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["position"].astype(int), df["points"].astype(float)))


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
    dnf_prob: np.ndarray             # (n_trials, n_drivers) -- mechanical DNF probability
    overtaking_difficulty: float     # circuits.csv value, 1 (hard) .. ~5 (easy)
    grid: np.ndarray = None          # (n_drivers,) 1-indexed real grid, post-quali mode only


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
