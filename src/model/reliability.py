"""Reliability model (spec sec 3.2): mechanical DNF hazard (per team-season)
plus incident DNF hazard (per driver, with a grid-position effect) as two
independent failure channels, per the spec's "two failure channels because
they behave differently".

Both are empirical-Bayes Beta-Binomial rather than another MCMC model --
deliberately cheap, since sec 3.2 says "keep it simple" and sec 7's compute
budget is tight enough that a second sampler pass for reliability isn't
worth it when a closed-form update gives essentially the same qualitative
behaviour (partial pooling toward a population rate, shrinking as more
data accumulates):

- Mechanical: a team's per-round DNF probability is Beta(a, b), with a
  population prior widened (lower a+b, i.e. less confident / more spread
  out) in rule-change years, and each team's posterior obtained by
  conjugate update against that team's own mechanical-DNF count and start
  count within the fit window (reset each season -- car reliability is a
  seasonal property).
- Incident: a driver's DNF probability depends on where they started
  (spec's "midfield starts crash more than front-row starts"). Modelled as
  a three-layer empirical-Bayes chain -- global prior -> grid-bucket
  population posterior (front/mid/back) -> driver posterior (that driver's
  own counts in that bucket, leave-one-out against the bucket total to
  avoid double-counting) -- rather than fitting a driver x bucket rate
  directly, which the data is too sparse to support per driver. Pooled
  across all seasons in the fit window (not reset per season, unlike
  mechanical): a driver's incident propensity is treated as a stable trait
  of their driving style, not a per-season car property. The empirical
  rate already reflects whatever share of incidents happened on lap 1
  historically, so the spec's "first-lap spike" is folded into the rate
  rather than modelled as a separate lap-indexed effect -- the resolver
  has no per-lap dimension in phase 1/2's single-shot race Monte Carlo.

At sim time: mechanical draws p ~ Beta(a_team, b_team) once per trial,
shared across both of that team's cars. Incident draws one Beta sample per
driver per grid bucket; `sim.race.simulate_positions` picks the bucket
matching that trial's *simulated* grid position (grid isn't known ahead of
time in pre-weekend mode) and combines the two as independent hazards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RULE_CHANGE_SEASONS = {2014, 2022, 2026}

# Population prior: mechanical DNF rate is typically ~5-8% per car-race in a
# stable regulation era. a0/b0 chosen so the prior mean is ~6% with modest
# concentration; rule-change seasons use a materially higher mean and a much
# lower concentration (wide/uncertain), per 3.5 "elevated mechanical hazard
# prior" and 3.2 "prior widened in rule-change years".
PRIOR_MEAN_STABLE = 0.06
PRIOR_CONCENTRATION_STABLE = 40.0   # a0 + b0
PRIOR_MEAN_RULE_CHANGE = 0.11
PRIOR_CONCENTRATION_RULE_CHANGE = 8.0

# Grid-position buckets for the incident-hazard effect (spec 3.2: "midfield
# starts crash more than front-row starts"). Three buckets keeps the
# per-bucket population counts large enough to be useful without needing a
# fitted curve -- consistent with 3.2's "keep it simple".
GRID_BUCKETS = (("front", 1, 4), ("mid", 5, 14), ("back", 15, 99))

# Global incident-hazard prior, before any bucket or driver adjustment.
# Incident DNF rate has historically run a bit above the mechanical rate;
# concentration is deliberately low since this is only the outermost layer
# of a three-layer empirical-Bayes chain (global -> bucket -> driver).
PRIOR_MEAN_INCIDENT_GLOBAL = 0.08
PRIOR_CONCENTRATION_INCIDENT_GLOBAL = 20.0


def grid_bucket(grid_position: int) -> str:
    for name, lo, hi in GRID_BUCKETS:
        if lo <= grid_position <= hi:
            return name
    return GRID_BUCKETS[-1][0]


@dataclass
class ReliabilityData:
    team_stats: pd.DataFrame  # team, season, mechanical_dnfs, starts
    global_rate: float
    driver_bucket_stats: pd.DataFrame  # abbreviation, grid_bucket, incident_dnfs, starts
    global_incident_rate: float


def _mechanical_prior(season: int) -> tuple[float, float]:
    if season in RULE_CHANGE_SEASONS:
        mean, conc = PRIOR_MEAN_RULE_CHANGE, PRIOR_CONCENTRATION_RULE_CHANGE
    else:
        mean, conc = PRIOR_MEAN_STABLE, PRIOR_CONCENTRATION_STABLE
    return mean * conc, (1 - mean) * conc


def build_reliability_data(
    driver_rounds: pd.DataFrame,
    through_season: int | None = None,
    through_round: int | None = None,
) -> ReliabilityData:
    """Count mechanical DNFs and starts per (team, season) from race rows."""
    df = driver_rounds[driver_rounds["session_type"] == "R"].copy()
    if through_season is not None:
        cutoff = (through_season, through_round if through_round is not None else 10_000)
        keep = [(s, r) <= cutoff for s, r in zip(df["season"], df["round"])]
        df = df[keep]

    df = df.dropna(subset=["team"])
    df["is_mechanical"] = df["dnf_cause"] == "mechanical"
    # a car that started the race (grid_position not null) counts as a start;
    # this deliberately excludes DNS/DNQ, which are not mechanical-hazard exposure.
    df = df[df["grid_position"].notna()]

    stats = (
        df.groupby(["team", "season"])
        .agg(mechanical_dnfs=("is_mechanical", "sum"), starts=("is_mechanical", "count"))
        .reset_index()
    )
    global_rate = float(df["is_mechanical"].mean()) if len(df) else PRIOR_MEAN_STABLE

    # Incident propensity is pooled across all seasons in the cutoff window
    # (not per-season like mechanical hazard above): a driver's crash rate is
    # more a stable trait of their style than a per-season car property, so
    # there's no reason to reset it each season the way team reliability does.
    df["is_incident"] = df["dnf_cause"] == "incident"
    df["grid_bucket"] = df["grid_position"].apply(lambda g: grid_bucket(int(g)))
    driver_bucket_stats = (
        df.groupby(["abbreviation", "grid_bucket"])
        .agg(incident_dnfs=("is_incident", "sum"), starts=("is_incident", "count"))
        .reset_index()
    )
    global_incident_rate = float(df["is_incident"].mean()) if len(df) else PRIOR_MEAN_INCIDENT_GLOBAL

    return ReliabilityData(
        team_stats=stats, global_rate=global_rate,
        driver_bucket_stats=driver_bucket_stats, global_incident_rate=global_incident_rate,
    )


def team_mechanical_posterior(data: ReliabilityData, team: str, season: int) -> tuple[float, float]:
    """Return (a, b) of the Beta posterior for `team`'s mechanical-DNF rate,
    pooling that team's in-window (season) history against the population prior."""
    a0, b0 = _mechanical_prior(season)
    row = data.team_stats[(data.team_stats["team"] == team) & (data.team_stats["season"] == season)]
    if row.empty:
        return a0, b0
    dnfs = float(row["mechanical_dnfs"].iloc[0])
    starts = float(row["starts"].iloc[0])
    return a0 + dnfs, b0 + (starts - dnfs)


def bucket_incident_posterior(data: ReliabilityData, bucket: str, exclude_driver: str | None = None) -> tuple[float, float]:
    """Beta posterior for `bucket`'s population incident rate: the global
    prior updated with every driver's counts in that bucket, pooled across
    the whole cutoff window. This is the middle layer of the three-layer
    chain (global -> bucket -> driver) and doubles as the prior for
    `driver_incident_posterior` below -- which passes `exclude_driver` so a
    driver's own counts aren't folded into their own prior (leave-one-out),
    avoiding double-counting them against the driver-level update on top."""
    a0 = PRIOR_MEAN_INCIDENT_GLOBAL * PRIOR_CONCENTRATION_INCIDENT_GLOBAL
    b0 = (1 - PRIOR_MEAN_INCIDENT_GLOBAL) * PRIOR_CONCENTRATION_INCIDENT_GLOBAL
    rows = data.driver_bucket_stats[data.driver_bucket_stats["grid_bucket"] == bucket]
    if exclude_driver is not None:
        rows = rows[rows["abbreviation"] != exclude_driver]
    if rows.empty:
        return a0, b0
    dnfs = float(rows["incident_dnfs"].sum())
    starts = float(rows["starts"].sum())
    return a0 + dnfs, b0 + (starts - dnfs)


def driver_incident_posterior(data: ReliabilityData, driver: str, grid_position: int) -> tuple[float, float]:
    """Beta posterior for `driver`'s incident-DNF rate starting from
    `grid_position`: the bucket posterior (population rate for that part of
    the grid) further updated with this driver's own counts in that bucket.
    Sparse per-driver data degrades gracefully to the bucket rate, which is
    the point of chaining empirical-Bayes layers rather than fitting a
    driver x bucket rate directly."""
    bucket = grid_bucket(grid_position)
    a0, b0 = bucket_incident_posterior(data, bucket, exclude_driver=driver)
    row = data.driver_bucket_stats[
        (data.driver_bucket_stats["abbreviation"] == driver)
        & (data.driver_bucket_stats["grid_bucket"] == bucket)
    ]
    if row.empty:
        return a0, b0
    dnfs = float(row["incident_dnfs"].iloc[0])
    starts = float(row["starts"].iloc[0])
    return a0 + dnfs, b0 + (starts - dnfs)


def sample_mechanical_dnf_prob(data: ReliabilityData, team: str, season: int,
                                n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_draws` samples of the mechanical-DNF probability for one
    (team, season) trial -- one draw per simulated race/season trial, shared
    across both cars on that team for that trial (a mechanical issue with one
    team's build is correlated across its two cars in reality more than the
    naive per-driver-independent model would suggest, but a shared per-trial
    team draw at least captures the team-level hazard level varying trial to
    trial, which is the sec 3.2 "hazard per team-season" ask)."""
    a, b = team_mechanical_posterior(data, team, season)
    return rng.beta(a, b, size=n_draws)


def sample_incident_dnf_prob(data: ReliabilityData, driver: str, grid_position: int,
                              n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_draws` samples of `driver`'s incident-DNF probability starting
    from `grid_position`, one draw per simulated race -- an independent
    per-driver draw (unlike the shared per-team mechanical draw above),
    since incidents are not correlated across teammates the way a mechanical
    build issue is. The empirically-estimated rate already reflects however
    much of it happened on lap 1 historically, so no separate first-lap term
    is needed (spec 3.2's "first-lap spike" is folded into the rate rather
    than modelled as its own lap-indexed effect, since the resolver has no
    per-lap dimension -- see reliability model design notes)."""
    a, b = driver_incident_posterior(data, driver, grid_position)
    return rng.beta(a, b, size=n_draws)
