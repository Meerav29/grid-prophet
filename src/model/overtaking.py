"""Per-circuit overtaking parameter, fitted from data (spec sec 3.3, sec 5).

Phase 1 read `overtaking_difficulty` straight off `data/circuits.csv` -- a
hand rating, 1 (Monaco, order locked) to ~5 (Bahrain, passes are cheap). Sec
3.3 step 3 wants the resolver's "simple track-specific overtaking difficulty
parameter"; sec 5 calls the CSV column a *hand rating*. This module estimates
the same parameter from observed grid-to-finish position changes and shrinks
it toward that hand rating where a circuit is thin on observations.

**The estimate is on the resolver's own scale, by construction.** `sim.race`
turns the parameter into one number and only one:

    penalty_per_slot = GRID_LOCK_BASE_PENALTY_S / overtaking_difficulty

and then orders cars by `race_pace + penalty_per_slot * (grid - 1)`. So the
fit inverts exactly that rule. For a candidate difficulty `d` we form each
car's predicted effective time from its *observed* race pace
(`gap_to_winner_median_clean_air_s`, already traffic-filtered per sec 5) and
its *observed* grid slot, and score the candidate by how many finishing-order
pairs that ordering gets right. The candidate with the most concordant pairs
is the one whose grid-lock penalty best explains the positions actually
swapped at that circuit. Nothing has to be mapped from one scale to another,
because the fitted number means the same thing the hand rating was guessing.

Degenerate ends are real and handled rather than hidden. At a locked circuit
every candidate below roughly `d = 1.5` reproduces the finishing order
equally well -- the data cannot tell a very hard circuit from an extremely
hard one -- so the search plateaus. Ties break toward the hand rating: where
the observations are indifferent, the human's number is the better answer.

Thin circuits shrink: `SHRINKAGE_PRIOR_ROUNDS` pseudo-rounds of the hand
rating are mixed with the fit, so a circuit on the calendar once leans on the
rating and one with nine seasons of races mostly does not. A circuit with no
usable round gets no fit at all and the caller falls back (`circuit_difficulty`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sim.race import GRID_LOCK_BASE_PENALTY_S

# Sec 5's hand rating runs 1..5; the search runs wider so a fit is never
# pinned by the prior's range, and log-spaced because the parameter enters
# the resolver as 1/d -- equal steps in d are not equal steps in penalty.
DIFFICULTY_MIN = 0.5
DIFFICULTY_MAX = 10.0
DIFFICULTY_GRID_POINTS = 60

# Used when a circuit has no fit and no hand rating -- the same 3.0
# `cli._circuit_info` already fell back to, kept so this module cannot
# change a forecast for a circuit it knows nothing about.
DEFAULT_DIFFICULTY = 3.0

# Pseudo-rounds of the hand rating mixed into every fit.
SHRINKAGE_PRIOR_ROUNDS = 4.0

# A round contributes only if this many cars were classified with both a grid
# slot and a clean-air pace: fewer than six finishers is a race whose order
# says more about attrition than about overtaking.
MIN_FINISHERS_PER_ROUND = 6

# Candidates scoring within this of the best are treated as indistinguishable.
PLATEAU_TOL = 1e-9

PACE_COLUMN = "gap_to_winner_median_clean_air_s"


@dataclass
class CircuitOvertakingFit:
    """One circuit's estimate. `raw` is the data alone, `shrunk` is what the
    resolver consumes; keeping both is what makes a thin fit visible."""
    event_name: str
    raw_difficulty: float
    shrunk_difficulty: float
    hand_rating: float | None
    n_rounds: int
    n_pairs: int
    concordance: float


@dataclass
class OvertakingData:
    by_circuit: dict = field(default_factory=dict)   # event_name -> CircuitOvertakingFit
    skipped: dict = field(default_factory=dict)      # event_name -> why no fit

    def fit_for(self, event_name) -> CircuitOvertakingFit | None:
        return self.by_circuit.get(event_name)


def difficulty_grid() -> np.ndarray:
    return np.geomspace(DIFFICULTY_MIN, DIFFICULTY_MAX, DIFFICULTY_GRID_POINTS)


def hand_ratings(circuits: pd.DataFrame) -> dict:
    """`data/circuits.csv` as event_name -> overtaking_difficulty."""
    if circuits is None or circuits.empty or "overtaking_difficulty" not in circuits.columns:
        return {}
    rows = circuits.dropna(subset=["event_name", "overtaking_difficulty"])
    return {str(e): float(v) for e, v in zip(rows["event_name"], rows["overtaking_difficulty"])}


def _round_pair_counts(pace: np.ndarray, grid: np.ndarray, finish: np.ndarray,
                        candidates: np.ndarray) -> tuple[np.ndarray, int]:
    """(concordant pairs per candidate, comparable pairs) for one round.

    A pair is concordant when the candidate's effective-time ordering puts the
    two cars the same way round as the finishing order did.
    """
    k = GRID_LOCK_BASE_PENALTY_S / candidates                      # (C,)
    t = pace[None, :] + k[:, None] * (grid[None, :] - 1.0)         # (C, n)
    dt = np.sign(t[:, :, None] - t[:, None, :])                    # (C, n, n)
    df = np.sign(finish[:, None] - finish[None, :])                # (n, n)
    upper = np.triu(np.ones_like(df, dtype=bool), k=1) & (df != 0)
    total = int(upper.sum())
    concordant = ((dt == df[None, :, :]) & upper[None, :, :]).sum(axis=(1, 2))
    return concordant.astype(float), total


def _usable_race_rows(driver_rounds: pd.DataFrame) -> pd.DataFrame:
    """Classified race finishers carrying a grid slot and a clean-air pace."""
    needed = {"session_type", "event_name", "season", "round", "grid_position", "finish_position"}
    if driver_rounds is None or driver_rounds.empty or not needed.issubset(driver_rounds.columns):
        return driver_rounds.iloc[0:0] if driver_rounds is not None else pd.DataFrame()
    if PACE_COLUMN not in driver_rounds.columns:
        return driver_rounds.iloc[0:0]
    df = driver_rounds[driver_rounds["session_type"] == "R"].copy()
    if "classified" in df.columns:
        df = df[df["classified"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    df = df.dropna(subset=["grid_position", "finish_position", PACE_COLUMN, "event_name"])
    return df[df["grid_position"] > 0]


def fit_circuit(rows: pd.DataFrame, event_name: str, hand_rating: float | None) -> CircuitOvertakingFit | None:
    """Fit one circuit from its usable race rows, or None if there are none.

    Ties break toward `hand_rating` (see the module docstring): across the
    plateau at a locked circuit the observations are literally indifferent,
    so the hand rating is the only information left.
    """
    candidates = difficulty_grid()
    concordant = np.zeros_like(candidates)
    pairs = 0
    n_rounds = 0
    for _, group in rows.groupby(["season", "round"], sort=True):
        if len(group) < MIN_FINISHERS_PER_ROUND:
            continue
        c, total = _round_pair_counts(
            group[PACE_COLUMN].to_numpy(dtype=float),
            group["grid_position"].to_numpy(dtype=float),
            group["finish_position"].to_numpy(dtype=float),
            candidates,
        )
        if total == 0:
            continue
        concordant += c
        pairs += total
        n_rounds += 1
    if n_rounds == 0 or pairs == 0:
        return None

    score = concordant / pairs
    best = score.max()
    tied = candidates[score >= best - PLATEAU_TOL]
    anchor = hand_rating if hand_rating is not None else DEFAULT_DIFFICULTY
    raw = float(tied[np.argmin(np.abs(tied - anchor))])

    if hand_rating is None:
        shrunk = raw
    else:
        shrunk = (n_rounds * raw + SHRINKAGE_PRIOR_ROUNDS * hand_rating) / (n_rounds + SHRINKAGE_PRIOR_ROUNDS)
    return CircuitOvertakingFit(
        event_name=event_name, raw_difficulty=raw, shrunk_difficulty=float(shrunk),
        hand_rating=hand_rating, n_rounds=n_rounds, n_pairs=pairs, concordance=float(best),
    )


def build_overtaking_data(driver_rounds: pd.DataFrame, circuits: pd.DataFrame | None = None,
                           before_season: int | None = None,
                           before_round: int | None = None) -> OvertakingData:
    """Fit every circuit with enough history in the window.

    `before_season`/`before_round` cut the window to rounds strictly *before*
    that round, which is what keeps a backtest honest: the round being
    forecast must not be in its own overtaking fit.
    """
    ratings = hand_ratings(circuits)
    df = _usable_race_rows(driver_rounds)
    if before_season is not None:
        cutoff = (before_season, before_round if before_round is not None else 0)
        keep = [(s, r) < cutoff for s, r in zip(df["season"], df["round"])]
        df = df[keep]

    data = OvertakingData()
    if df.empty:
        return data
    for event_name, rows in df.groupby("event_name", sort=True):
        fit = fit_circuit(rows, str(event_name), ratings.get(str(event_name)))
        if fit is None:
            data.skipped[str(event_name)] = f"no round with >= {MIN_FINISHERS_PER_ROUND} usable finishers"
        else:
            data.by_circuit[str(event_name)] = fit
    return data


def circuit_difficulty(data: OvertakingData | None, event_name, hand_rating: float | None) -> float:
    """The parameter the resolver should use: the fitted value where one
    exists, the hand rating where it does not, DEFAULT_DIFFICULTY where
    neither does."""
    if data is not None:
        fit = data.fit_for(str(event_name) if event_name is not None else None)
        if fit is not None:
            return float(fit.shrunk_difficulty)
    if hand_rating is not None and np.isfinite(hand_rating):
        return float(hand_rating)
    return DEFAULT_DIFFICULTY
