"""First-cut reliability model (spec sec 3.2, Phase 1 scope): team-level
mechanical hazard only. The mechanical-vs-incident split is Phase 2.

Implemented as an empirical-Bayes Beta-Binomial rather than another MCMC
model: a team's per-round mechanical-DNF probability is Beta(a, b), with a
population prior widened (lower a+b, i.e. less confident / more spread out)
in rule-change years, and each team's posterior obtained by conjugate update
against that team's mechanical-DNF count and start count within the fit
window. This is deliberately cheap -- sec 3.2 says "keep it simple", and
sec 7's compute budget is tight enough that a second sampler pass for
reliability is not worth it when a closed-form update is available and gives
essentially the same qualitative behaviour (partial pooling toward a
population rate, shrinking as more of the team's own season accumulates).

At sim time, draw p ~ Beta(a_team, b_team) once per trial and then a
Bernoulli(p) per driver-race for "retired with a mechanical failure".
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


@dataclass
class ReliabilityData:
    team_stats: pd.DataFrame  # team, season, mechanical_dnfs, starts
    global_rate: float


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
    return ReliabilityData(team_stats=stats, global_rate=global_rate)


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
