"""Hierarchical Bayesian pace model (spec sec 3.1).

    pace[d, r] = car[t(d), r] + driver[d] + driver_track[d, type(c)] + eps

`car[t, r]` follows a random walk across a single global, chronologically
ordered round index shared by every team (round 0 = the first collected round
of the earliest season in the fit window). The random-walk step size is
inflated at season boundaries, and inflated further at the season boundaries
of regulation-reset years (2014, 2022, 2026) and for the first few rounds of
those seasons -- this is the "first cut" way of encoding sec 3.1's "car[t, 1]
drawn from a prior centred on last season's end-of-year strength, wider in
rule-change years": a bigger step *is* a wider prior on the new value, and it
keeps the whole history as a single non-centred random walk instead of a
separate per-season prior-lookup mechanism, which is much simpler to get
right and to warm-start between backtest rounds.

The quali equation shares `car` and `driver` with the race equation (sec 3.1,
"Quali pace is a parallel equation sharing car and driver... with its own
noise and a quali-specific driver offset") and is fit jointly.

Deterministic data-prep (`build_pace_data`) is unit-tested directly (see
tests/test_pace_data.py). The PyMC model itself is validated indirectly,
through the backtest metrics the spec specifies (sec 7) -- that is the
intended validation strategy for the probabilistic core, per the phase 1
task description.

Backend note: the spec allows either PyMC or NumPyro. NumPyro/JAX was tried
first (it's the faster option for the repeated-fit backtest pattern per sec
7), but the `jaxlib` wheel available for this Windows/Python 3.12 environment
fails to import (`ImportError: DLL load failed while importing _jax`) --
a packaging problem, not a modelling one, but not fixable from inside this
sandbox. PyMC (pytensor + numba backend, no C++ compiler needed) works
natively on this machine and is used instead; see docs/phase1-status.md for
the resulting compute-time impact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RULE_CHANGE_SEASONS = {2014, 2022, 2026}
RULE_CHANGE_EARLY_ROUNDS = 6            # sec 3.5: wider drift for ~first 6 rounds
SEASON_BOUNDARY_SCALE = 3.0             # step-sigma multiplier at a normal season change
RULE_CHANGE_BOUNDARY_SCALE = 8.0        # step-sigma multiplier at a rule-change season start
RULE_CHANGE_EARLY_SCALE = 2.0           # extra multiplier for the first few rounds after that
CIRCUIT_TYPES = ["mixed", "street", "high_downforce", "high_speed"]


@dataclass
class PaceData:
    """Arrays + index metadata needed to build and predict from the pace model.

    Team/driver/round vocabularies are always built from the *full* dataset
    (all seasons present in `driver_rounds.csv`), not just the rows that pass
    the `through` cutoff -- this keeps index meaning stable across successive
    `gp fit --through ...` calls so a later fit's posterior can warm-start
    from an earlier one (spec sec 7 compute budget: "sequential warm starts").
    """

    # race (session_type == "R") observations
    race_team_idx: np.ndarray
    race_driver_idx: np.ndarray
    race_round_idx: np.ndarray
    race_circuit_idx: np.ndarray
    race_y: np.ndarray

    # quali (session_type == "Q") observations
    quali_team_idx: np.ndarray
    quali_driver_idx: np.ndarray
    quali_round_idx: np.ndarray
    quali_y: np.ndarray

    n_teams: int
    n_drivers: int
    n_rounds: int
    n_circuit_types: int = len(CIRCUIT_TYPES)

    team_names: list = field(default_factory=list)
    driver_names: list = field(default_factory=list)     # by abbreviation
    round_index: pd.DataFrame = None                      # round_idx -> season, round
    rookie_mask: np.ndarray = None                         # (n_drivers,) bool
    step_sigma_scale: np.ndarray = None                    # (n_rounds,) float, step i-1->i

    team_to_idx: dict = field(default_factory=dict)
    driver_to_idx: dict = field(default_factory=dict)
    round_to_idx: dict = field(default_factory=dict)       # (season, round) -> idx


def _build_round_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rounds = (
        df[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )
    rounds["round_idx"] = rounds.index
    round_to_idx = {(int(s), int(r)): int(i) for s, r, i in
                     zip(rounds["season"], rounds["round"], rounds["round_idx"])}
    return rounds, round_to_idx


def _step_sigma_scale(round_index: pd.DataFrame) -> np.ndarray:
    n = len(round_index)
    scale = np.ones(n)
    seasons = round_index["season"].to_numpy()
    for i in range(1, n):
        if seasons[i] != seasons[i - 1]:
            if seasons[i] in RULE_CHANGE_SEASONS:
                scale[i] = RULE_CHANGE_BOUNDARY_SCALE
            else:
                scale[i] = SEASON_BOUNDARY_SCALE
        elif seasons[i] in RULE_CHANGE_SEASONS:
            # position within the season (1-indexed round number, not global idx)
            season_start = np.searchsorted(seasons, seasons[i])
            pos_in_season = i - season_start
            if pos_in_season < RULE_CHANGE_EARLY_ROUNDS:
                scale[i] = RULE_CHANGE_EARLY_SCALE
    return scale


def _rookie_mask(driver_names: list, driver_meta: pd.DataFrame, through_season: int | None) -> np.ndarray:
    """A driver is a rookie w.r.t. the fit window if their debut season is the
    cutoff season (or later, or unknown -- treated conservatively as rookie)."""
    if driver_meta is None or driver_meta.empty:
        return np.zeros(len(driver_names), dtype=bool)
    debut = dict(zip(driver_meta["abbreviation"], driver_meta["debut_season"]))
    cutoff = through_season if through_season is not None else 9999
    mask = np.zeros(len(driver_names), dtype=bool)
    for i, abbr in enumerate(driver_names):
        d = debut.get(abbr)
        mask[i] = d is None or d >= cutoff
    return mask


def build_pace_data(
    driver_rounds: pd.DataFrame,
    driver_meta: pd.DataFrame | None = None,
    through_season: int | None = None,
    through_round: int | None = None,
) -> PaceData:
    """Build index arrays for the pace model.

    `through_season`/`through_round`: inclusive cutoff -- only rows with
    (season, round) <= (through_season, through_round) are included as
    observations. Team/driver vocabularies are built from the full dataframe
    passed in (so callers should pass the *entire* available `driver_rounds`
    table), but the round index/random-walk axis is trimmed to rounds at or
    before the cutoff -- an early backtest cutoff (e.g. through 2022:2) would
    otherwise still pay for a car random-walk over every future round
    already sitting in `driver_rounds.csv` (up to 2025+), which is pure
    waste against the sec 7 compute budget and buys nothing (sequential
    warm-starting across cutoffs, the thing a *stable* round index would be
    for, isn't implemented yet -- see docs/phase1-status.md).
    """
    df = driver_rounds.copy()
    df["circuit_type"] = df["circuit_type"].fillna("mixed")

    team_names = sorted(df["team"].dropna().unique().tolist())
    driver_names = sorted(df["abbreviation"].dropna().unique().tolist())
    team_to_idx = {t: i for i, t in enumerate(team_names)}
    driver_to_idx = {d: i for i, d in enumerate(driver_names)}
    circuit_to_idx = {c: i for i, c in enumerate(CIRCUIT_TYPES)}

    if through_season is not None:
        cutoff = (through_season, through_round if through_round is not None else 10_000)
        df_for_rounds = df[[(int(s), int(r)) <= cutoff for s, r in zip(df["season"], df["round"])]]
    else:
        df_for_rounds = df
    round_index, round_to_idx = _build_round_index(df_for_rounds)
    step_scale = _step_sigma_scale(round_index)
    rookie_mask = _rookie_mask(driver_names, driver_meta, through_season)

    def _cutoff_mask(frame: pd.DataFrame) -> pd.Series:
        if through_season is None:
            return pd.Series(True, index=frame.index)
        key = list(zip(frame["season"], frame["round"]))
        cutoff = (through_season, through_round if through_round is not None else 10_000)
        return pd.Series([k <= cutoff for k in key], index=frame.index)

    race = df[(df["session_type"] == "R")].copy()
    race = race[_cutoff_mask(race)]
    race = race.dropna(subset=["median_clean_air_lap_s", "gap_to_winner_median_clean_air_s", "team", "abbreviation"])
    race = race[race["team"].isin(team_to_idx) & race["abbreviation"].isin(driver_to_idx)]

    quali = df[df["session_type"] == "Q"].copy()
    quali = quali[_cutoff_mask(quali)]
    quali = quali.dropna(subset=["gap_to_best_s", "team", "abbreviation"])
    quali = quali[quali["team"].isin(team_to_idx) & quali["abbreviation"].isin(driver_to_idx)]

    def _round_idx_col(frame: pd.DataFrame) -> np.ndarray:
        return np.array([round_to_idx[(int(s), int(r))] for s, r in zip(frame["season"], frame["round"])])

    return PaceData(
        race_team_idx=race["team"].map(team_to_idx).to_numpy(dtype=int),
        race_driver_idx=race["abbreviation"].map(driver_to_idx).to_numpy(dtype=int),
        race_round_idx=_round_idx_col(race),
        race_circuit_idx=race["circuit_type"].map(circuit_to_idx).fillna(0).to_numpy(dtype=int),
        race_y=race["gap_to_winner_median_clean_air_s"].to_numpy(dtype=float),
        quali_team_idx=quali["team"].map(team_to_idx).to_numpy(dtype=int),
        quali_driver_idx=quali["abbreviation"].map(driver_to_idx).to_numpy(dtype=int),
        quali_round_idx=_round_idx_col(quali),
        quali_y=quali["gap_to_best_s"].to_numpy(dtype=float),
        n_teams=len(team_names),
        n_drivers=len(driver_names),
        n_rounds=len(round_index),
        team_names=team_names,
        driver_names=driver_names,
        round_index=round_index,
        rookie_mask=rookie_mask,
        step_sigma_scale=step_scale,
        team_to_idx=team_to_idx,
        driver_to_idx=driver_to_idx,
        round_to_idx=round_to_idx,
    )


# ---------------------------------------------------------------------------
# PyMC model
# ---------------------------------------------------------------------------

SIGMA_DEV = 0.08        # base random-walk step size, seconds/lap per round
ROOKIE_PENALTY_S = 0.3  # sec 3.5: rookies land ~0.2-0.4s/lap behind in year one


def build_model(data: PaceData, sigma_dev: float = SIGMA_DEV, rookie_penalty_s: float = ROOKIE_PENALTY_S):
    """Build (but do not fit) the PyMC model graph for the pace equations."""
    import pymc as pm
    import pytensor.tensor as pt

    n_teams, n_rounds, n_drivers, n_ct = data.n_teams, data.n_rounds, data.n_drivers, data.n_circuit_types
    step_scale = data.step_sigma_scale  # (n_rounds,), index 0 unused
    rookie = data.rookie_mask
    driver_prior_mean = np.where(rookie, rookie_penalty_s, 0.0)
    driver_prior_sd = np.where(rookie, 0.4, 0.25)

    coords = {
        "team": data.team_names,
        "round": list(range(n_rounds)),
        "driver": data.driver_names,
        "circuit_type": CIRCUIT_TYPES,
    }

    with pm.Model(coords=coords) as model:
        # --- car[team, round]: non-centred random walk per team ---
        car0 = pm.Normal("car0", 0.0, 2.0, dims="team")
        step_raw = pm.Normal("car_step_raw", 0.0, 1.0, shape=(n_teams, n_rounds - 1))
        step_sigma = sigma_dev * step_scale[1:]  # (n_rounds-1,)
        steps = step_raw * step_sigma[None, :]
        car = pm.Deterministic(
            "car",
            pt.concatenate([car0[:, None], car0[:, None] + pt.cumsum(steps, axis=1)], axis=1),
            dims=("team", "round"),
        )

        # --- driver[d]: population skill, rookies pooled toward a wider/shifted prior ---
        driver = pm.Normal("driver_skill", driver_prior_mean, driver_prior_sd, dims="driver")
        quali_offset = pm.Normal("quali_offset", 0.0, 0.15, dims="driver")

        # --- driver x circuit-type interaction, regularised hard ---
        driver_track = pm.Normal("driver_track", 0.0, 0.1, dims=("driver", "circuit_type"))

        # --- race likelihood: Student-t, fat tails ---
        race_mu = (
            car[data.race_team_idx, data.race_round_idx]
            + driver[data.race_driver_idx]
            + driver_track[data.race_driver_idx, data.race_circuit_idx]
        )
        race_nu = pm.Gamma("race_nu", alpha=5.0, beta=0.5)
        race_sigma = pm.HalfNormal("race_sigma", 0.5)
        pm.StudentT("race_obs", nu=race_nu, mu=race_mu, sigma=race_sigma, observed=data.race_y)

        # --- quali likelihood: shares car/driver, own noise + Saturday-specialist offset ---
        quali_mu = (
            car[data.quali_team_idx, data.quali_round_idx]
            + driver[data.quali_driver_idx]
            + quali_offset[data.quali_driver_idx]
        )
        quali_sigma = pm.HalfNormal("quali_sigma", 0.5)
        pm.Normal("quali_obs", mu=quali_mu, sigma=quali_sigma, observed=data.quali_y)

    return model


def fit_nuts(data: PaceData, draws=400, tune=400, chains=2, seed=0, initvals=None):
    """Full NUTS fit -- use for the production fit shipped for `gp race`."""
    import pymc as pm

    model = build_model(data)
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, random_seed=seed,
            initvals=initvals, progressbar=False, compute_convergence_checks=False,
        )
    return idata


def fit_advi(data: PaceData, n=8000, seed=0, draws=500):
    """Fast ADVI fit -- use for the backtest loop (spec sec 7 compute budget:
    "VI for the backtest loop, NUTS for production")."""
    import pymc as pm

    model = build_model(data)
    with model:
        approx = pm.fit(n=n, method="advi", random_seed=seed, progressbar=False)
        idata = approx.sample(draws)
    return idata


def posterior_pace_draws(idata, data: PaceData, team: str, driver: str, circuit_type: str,
                          round_idx: int | None = None) -> np.ndarray:
    """Flattened (chain*draw,) samples of race pace (gap-to-field-leader, seconds)
    for one (team, driver, circuit_type), read off the fitted `round_idx`
    (defaults to the last / most recent round in the fit window)."""
    car = idata.posterior["car"]  # dims: chain, draw, team, round
    r = round_idx if round_idx is not None else data.n_rounds - 1

    if team in data.team_to_idx:
        car_vals = car.isel(round=r).sel(team=team).to_numpy().reshape(-1)
    else:
        # unseen team (e.g. a brand-new entrant): wide fallback = population car0 mean
        car_vals = idata.posterior["car0"].to_numpy().reshape(-1, data.n_teams).mean(axis=1)

    n_draws = car_vals.shape[0]
    if driver in data.driver_to_idx:
        driver_vals = idata.posterior["driver_skill"].sel(driver=driver).to_numpy().reshape(-1)
        ct = circuit_type if circuit_type in CIRCUIT_TYPES else CIRCUIT_TYPES[0]
        track_vals = idata.posterior["driver_track"].sel(driver=driver, circuit_type=ct).to_numpy().reshape(-1)
    else:
        driver_vals = np.zeros(n_draws)
        track_vals = np.zeros(n_draws)

    return car_vals + driver_vals + track_vals


def posterior_quali_draws(idata, data: PaceData, team: str, driver: str,
                           round_idx: int | None = None) -> np.ndarray:
    car = idata.posterior["car"]
    r = round_idx if round_idx is not None else data.n_rounds - 1

    if team in data.team_to_idx:
        car_vals = car.isel(round=r).sel(team=team).to_numpy().reshape(-1)
    else:
        car_vals = idata.posterior["car0"].to_numpy().reshape(-1, data.n_teams).mean(axis=1)

    n_draws = car_vals.shape[0]
    if driver in data.driver_to_idx:
        driver_vals = idata.posterior["driver_skill"].sel(driver=driver).to_numpy().reshape(-1)
        offset_vals = idata.posterior["quali_offset"].sel(driver=driver).to_numpy().reshape(-1)
    else:
        driver_vals = np.zeros(n_draws)
        offset_vals = np.zeros(n_draws)

    return car_vals + driver_vals + offset_vals


def posterior_summary(idata, data: PaceData) -> pd.DataFrame:
    """Current car strength (latest round) and driver skill estimates with
    credible intervals -- feeds `posterior_summary.csv` (spec sec 6)."""
    r = data.n_rounds - 1
    car = idata.posterior["car"].isel(round=r)  # dims: chain, draw, team
    car_flat = car.to_numpy().reshape(-1, data.n_teams)
    rows = []
    for i, team in enumerate(data.team_names):
        vals = car_flat[:, i]
        rows.append({
            "entity": "team", "name": team,
            "estimate": float(np.mean(vals)),
            "p05": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95)),
        })
    driver_flat = idata.posterior["driver_skill"].to_numpy().reshape(-1, data.n_drivers)
    for i, driver in enumerate(data.driver_names):
        vals = driver_flat[:, i]
        rows.append({
            "entity": "driver", "name": driver,
            "estimate": float(np.mean(vals)),
            "p05": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95)),
        })
    return pd.DataFrame(rows)

