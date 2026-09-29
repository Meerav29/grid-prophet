"""Reliability model (spec sec 3.2): two failure channels, not one.

Phase 1's first cut was a single team-level mechanical hazard (see
`docs/phase1-status.md`, caveat 3). Sec 3.2 asks for two:

* **Mechanical** -- hazard per (team, season). The population prior is widened
  in rule-change years, and that widening shrinks as the season runs: sec 3.2's
  parenthetical is about *early-season* failures in 2014 and 2022.
* **Incident / heavy damage** -- hazard per driver, scaled by a grid-position
  effect (midfield starts crash more than front-row starts), with sec 3.2's
  first-lap spike carried by the lap-of-retirement distribution.

Both are empirical-Bayes Beta-Binomials, not MCMC models: sec 3.2 says "keep it
simple" and sec 7's budget has no room for a second sampler pass when a
conjugate update gives the same partial pooling. A team's mechanical
probability is drawn once per trial and shared by its two cars; a driver's
incident probability per driver per trial; `combined_dnf_prob` folds them into
the one probability the resolver consumes.

**The data gap (sec 3.2, "Known data gap") is load-bearing here.** FastF1's
cause detail stops after 2024; from 2023 most retirements just read "Retired",
and `data/dnf_status_map.csv` flags those statuses `needs_review=True`. Nothing
here consumes a flagged row as a clean label: they contribute *fractional*
weight to both channels (`channel_weights`), so a 2024 "Retired" row nudges
both hazards instead of being booked to one. Channel counts are therefore
floats, and `clean_rows` / `flagged_rows` make the mixture countable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DNF_STATUS_MAP_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "dnf_status_map.csv",
)

RULE_CHANGE_SEASONS = {2014, 2022, 2026}

# --- mechanical channel ---------------------------------------------------
# ~5-8% per car-race in a stable regulation era; a rule-change season starts
# from a higher mean and a much lower concentration (wide/uncertain), per 3.5
# "elevated mechanical hazard prior" and 3.2 "prior widened in rule-change".
PRIOR_MEAN_STABLE = 0.06
PRIOR_CONCENTRATION_STABLE = 40.0   # a0 + b0
PRIOR_MEAN_RULE_CHANGE = 0.11
PRIOR_CONCENTRATION_RULE_CHANGE = 8.0
# "...and shrinking as the season runs": linear decay back to the stable prior.
RULE_CHANGE_WIDENING_DECAY_ROUNDS = 8

# --- incident channel -----------------------------------------------------
# A little under the mechanical rate, and a driver's record is thinner than a
# team's, so this prior is no tighter.
PRIOR_MEAN_INCIDENT = 0.05
PRIOR_CONCENTRATION_INCIDENT = 30.0

# Grid-position effect (sec 3.2: "midfield starts crash more than front-row
# starts"). Bands, not a curve: three numbers is what a one-line claim supports.
# The priors encode the spec's ordering; pooled data moves them, shrunk by
# GRID_MULTIPLIER_PRIOR_STARTS pseudo-starts per band.
GRID_BANDS = (("front", 1, 4), ("midfield", 5, 14), ("rear", 15, 10_000))
PRIOR_GRID_MULTIPLIER = {"front": 0.70, "midfield": 1.25, "rear": 1.05}
GRID_MULTIPLIER_PRIOR_STARTS = 150.0

# --- lap of retirement ----------------------------------------------------
# `data/circuits.csv` carries lap *length*, not a lap count, and adding one is
# data curation, not modelling -- callers that know better pass n_laps.
NOMINAL_RACE_LAPS = 57
NOMINAL_SPRINT_LAPS = 19
# Sec 3.2's "first-lap spike": lap 1 carries this many times the incident
# weight of any other lap. At 57 laps that is ~18% of incident retirements.
FIRST_LAP_HAZARD_MULTIPLIER = 12.0

# --- label mixture (the data gap) -----------------------------------------
# A flagged status that still names a system (Suspension, Puncture) is evidence
# but not proof, so it lands mostly on that channel. One that names nothing
# ("Retired") splits on the observable era's own mechanical:incident ratio.
FLAGGED_LABEL_LEAN = 0.7


@dataclass
class ReliabilityData:
    team_stats: pd.DataFrame    # team, season, mechanical_dnfs, incident_dnfs, starts
    global_rate: float          # pooled mechanical rate over the fit window
    driver_stats: pd.DataFrame = field(default_factory=pd.DataFrame)  # abbreviation, ...
    grid_multipliers: dict = field(default_factory=dict)              # band -> multiplier
    rounds_by_season: dict = field(default_factory=dict)              # season -> rounds in window
    clean_rows: int = 0         # DNF rows carrying a usable cause label
    flagged_rows: int = 0       # DNF rows folded in as a mixture instead
    #    clean_rows / flagged_rows are the countable form of the data gap: the
    #    flagged share is large for 2023+ and is a signal about the fit.


def load_label_policy(path: str = DNF_STATUS_MAP_CSV) -> pd.DataFrame:
    """`data/dnf_status_map.csv` as status / category / needs_review."""
    df = pd.read_csv(path)
    df["needs_review"] = df["needs_review"].astype(bool)
    return df[["status", "category", "needs_review"]]


def channel_weights(rows: pd.DataFrame, policy: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `w_mechanical`, `w_incident` and `label_flagged` to race rows.

    A row that retired contributes total weight 1.0 across the two channels; a
    finisher, and a non-car withdrawal (illness, DSQ, DNS), contributes 0 to
    both. The middle case is the point: a row whose `status` is flagged
    `needs_review` splits across *both* channels and so is never a clean label,
    however confident its mapped category looks. Rows with no `status` column
    (unit fixtures, callers holding only `dnf_cause`) are taken at their mapped
    category -- there is no flag to honour there.
    """
    policy = load_label_policy() if policy is None else policy
    flagged = set(policy.loc[policy["needs_review"], "status"])
    cause = rows["dnf_cause"].where(rows["dnf_cause"].notna(), None).to_numpy(dtype=object)
    is_dnf = np.array([c is not None for c in cause], dtype=bool)
    if "status" in rows.columns:
        is_flagged = rows["status"].isin(flagged).to_numpy() & is_dnf
    else:
        is_flagged = np.zeros(len(rows), dtype=bool)

    clean_mech = int(((cause == "mechanical") & is_dnf & ~is_flagged).sum())
    clean_inc = int(((cause == "incident") & is_dnf & ~is_flagged).sum())
    # No observable era in the window -> an even split is the only defensible
    # mixture: it keeps the exposure in both channels without claiming a way.
    ratio = clean_mech / (clean_mech + clean_inc) if (clean_mech + clean_inc) else 0.5

    lean = np.where(cause == "mechanical", FLAGGED_LABEL_LEAN,
                    np.where(cause == "incident", 1.0 - FLAGGED_LABEL_LEAN, ratio))
    out = rows.copy()
    out["w_mechanical"] = np.where(is_flagged, lean,
                                   np.where(is_dnf & (cause == "mechanical"), 1.0, 0.0))
    out["w_incident"] = np.where(is_flagged, 1.0 - lean,
                                  np.where(is_dnf & (cause == "incident"), 1.0, 0.0))
    out["label_flagged"] = is_flagged
    return out


def grid_band(grid_position) -> str:
    """Band for a 1-indexed grid slot. A pit-lane start (0) or a missing slot
    bands as `rear`, matching how the resolver ranks them (slice-1)."""
    try:
        pos = float(grid_position)
    except (TypeError, ValueError):
        return "rear"
    if not np.isfinite(pos) or pos < 1:
        return "rear"
    for name, lo, hi in GRID_BANDS:
        if lo <= pos <= hi:
            return name
    return "rear"


def _mechanical_prior(season: int, rounds_completed: int = 0) -> tuple[float, float]:
    """(a0, b0) of the population prior for a season's mechanical hazard. A
    rule-change season starts wider and higher and decays linearly back to the
    stable prior over RULE_CHANGE_WIDENING_DECAY_ROUNDS completed rounds."""
    if season not in RULE_CHANGE_SEASONS:
        mean, conc = PRIOR_MEAN_STABLE, PRIOR_CONCENTRATION_STABLE
    else:
        frac = min(max(rounds_completed, 0) / RULE_CHANGE_WIDENING_DECAY_ROUNDS, 1.0)
        mean = PRIOR_MEAN_RULE_CHANGE + (PRIOR_MEAN_STABLE - PRIOR_MEAN_RULE_CHANGE) * frac
        conc = (PRIOR_CONCENTRATION_RULE_CHANGE
                + (PRIOR_CONCENTRATION_STABLE - PRIOR_CONCENTRATION_RULE_CHANGE) * frac)
    return mean * conc, (1 - mean) * conc


def _grid_multipliers(df: pd.DataFrame) -> dict:
    """Pooled incident multiplier per band, shrunk toward PRIOR_GRID_MULTIPLIER
    and normalised so the start-weighted mean is 1.0: the bands redistribute the
    pooled incident rate across the grid, they do not inflate it."""
    n = len(df)
    overall = float(df["w_incident"].sum()) / n if n else 0.0
    mult = {}
    for name, _lo, _hi in GRID_BANDS:
        if overall <= 0:
            mult[name] = PRIOR_GRID_MULTIPLIER[name]
            continue
        sub = df[df["grid_band"] == name]
        w = GRID_MULTIPLIER_PRIOR_STARTS
        rate = (PRIOR_GRID_MULTIPLIER[name] * overall * w + float(sub["w_incident"].sum())) / (w + len(sub))
        mult[name] = rate / overall
    if n:
        shares = {name: float((df["grid_band"] == name).sum()) / n for name, _l, _h in GRID_BANDS}
        mean = sum(shares[k] * mult[k] for k in mult)
        if mean > 0:
            mult = {k: v / mean for k, v in mult.items()}
    return mult


def build_reliability_data(
    driver_rounds: pd.DataFrame,
    through_season: int | None = None,
    through_round: int | None = None,
    policy: pd.DataFrame | None = None,
) -> ReliabilityData:
    """Accumulate both channels' exposure and (fractional) failure counts."""
    df = driver_rounds[driver_rounds["session_type"] == "R"].copy()
    if through_season is not None:
        cutoff = (through_season, through_round if through_round is not None else 10_000)
        keep = [(s, r) <= cutoff for s, r in zip(df["season"], df["round"])]
        df = df[keep]

    df = df.dropna(subset=["team"])
    # a car that started the race (grid_position not null) counts as a start;
    # this deliberately excludes DNS/DNQ, which are not hazard exposure.
    df = df[df["grid_position"].notna()]
    df = channel_weights(df, policy)
    df["grid_band"] = [grid_band(g) for g in df["grid_position"]]

    team_stats = (
        df.groupby(["team", "season"])
        .agg(mechanical_dnfs=("w_mechanical", "sum"), incident_dnfs=("w_incident", "sum"),
             starts=("w_mechanical", "count"))
        .reset_index()
    )
    # Sec 3.2 says "hazard per driver", no season index: one season is ~24
    # starts, too thin for a crash propensity. See docs/autopilot/decisions.md.
    driver_stats = (
        df.groupby("abbreviation")
        .agg(incident_dnfs=("w_incident", "sum"), mechanical_dnfs=("w_mechanical", "sum"),
             starts=("w_incident", "count"))
        .reset_index()
    )
    n = len(df)
    return ReliabilityData(
        team_stats=team_stats,
        global_rate=float(df["w_mechanical"].mean()) if n else PRIOR_MEAN_STABLE,
        driver_stats=driver_stats,
        grid_multipliers=_grid_multipliers(df),
        rounds_by_season=df.groupby("season")["round"].nunique().to_dict() if n else {},
        clean_rows=int(((df["w_mechanical"] + df["w_incident"]) > 0).sum() - df["label_flagged"].sum()),
        flagged_rows=int(df["label_flagged"].sum()),
    )


def team_mechanical_posterior(data: ReliabilityData, team: str, season: int) -> tuple[float, float]:
    """(a, b) of the Beta posterior for `team`'s mechanical-DNF rate, pooling
    that team's in-window (season) history against the population prior."""
    a0, b0 = _mechanical_prior(season, data.rounds_by_season.get(season, 0))
    row = data.team_stats[(data.team_stats["team"] == team) & (data.team_stats["season"] == season)]
    if row.empty:
        return a0, b0
    dnfs = float(row["mechanical_dnfs"].iloc[0])
    starts = float(row["starts"].iloc[0])
    return a0 + dnfs, b0 + max(starts - dnfs, 0.0)


def driver_incident_posterior(data: ReliabilityData, driver: str) -> tuple[float, float]:
    """(a, b) of the Beta posterior for `driver`'s incident-DNF rate, before
    the grid-position multiplier is applied."""
    a0 = PRIOR_MEAN_INCIDENT * PRIOR_CONCENTRATION_INCIDENT
    b0 = PRIOR_CONCENTRATION_INCIDENT - a0
    if data.driver_stats.empty:
        return a0, b0
    row = data.driver_stats[data.driver_stats["abbreviation"] == driver]
    if row.empty:
        return a0, b0
    dnfs = float(row["incident_dnfs"].iloc[0])
    starts = float(row["starts"].iloc[0])
    return a0 + dnfs, b0 + max(starts - dnfs, 0.0)


def grid_multiplier(data: ReliabilityData, grid_position) -> float:
    band = grid_band(grid_position)
    return float(data.grid_multipliers.get(band, PRIOR_GRID_MULTIPLIER[band]))


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


def sample_incident_dnf_prob(data: ReliabilityData, driver: str, grid_position,
                              n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_draws` samples of one driver's incident-DNF probability from a
    given grid slot. Unlike the mechanical channel this is per driver, not
    shared across a team: sec 3.2 puts the incident hazard on the driver."""
    a, b = driver_incident_posterior(data, driver)
    return np.clip(rng.beta(a, b, size=n_draws) * grid_multiplier(data, grid_position), 0.0, 1.0)


def combined_dnf_prob(mechanical, incident):
    """P(retired) from two channels treated as independent within a trial.
    Not an innocent choice -- a car nursing damage is likelier to break -- but
    sec 3.2 gives no coupling and asks for simple, so: product of complements.
    """
    m = np.asarray(mechanical, dtype=float)
    i = np.asarray(incident, dtype=float)
    return 1.0 - (1.0 - m) * (1.0 - i)


def lap_of_retirement_pmf(channel: str, n_laps: int = NOMINAL_RACE_LAPS) -> np.ndarray:
    """Distribution over the lap a retirement happens on, laps 1..n_laps.

    Sec 3.2 says keep it simple -- it matters for sprint and partial-points
    scoring and nothing else. Mechanical failures are flat in lap number;
    incidents carry the first-lap spike.
    """
    if n_laps < 1:
        raise ValueError(f"n_laps must be >= 1, got {n_laps}")
    if channel not in ("mechanical", "incident"):
        raise ValueError(f"channel must be 'mechanical' or 'incident', got {channel!r}")
    w = np.ones(int(n_laps), dtype=float)
    if channel == "incident":
        w[0] = FIRST_LAP_HAZARD_MULTIPLIER
    return w / w.sum()


def first_lap_incident_share(n_laps: int = NOMINAL_RACE_LAPS) -> float:
    """Share of a driver's incident retirements that land on lap 1."""
    return float(lap_of_retirement_pmf("incident", n_laps)[0])
