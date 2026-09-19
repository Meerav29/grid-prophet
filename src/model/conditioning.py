"""Post-quali conditioning (spec sec 6, "Post-quali conditioning").

Once quali has run, the round's grid is a fact and its quali times are a fresh
observation. Sec 6's *primary* approach is to add that observation and
warm-start NUTS from the pre-weekend posterior with a short warmup -- not an
importance reweight, which sec 6 expects to degenerate under the wide priors
2026 actually has (new cars, Cadillac, rookies).

Sec 6 also specifies a failure protocol, which is most of this module: a warm
start that misses its couple-of-minutes target falls back to a full refit
*including the new quali observation* rather than blocking the run; the target
is relaxed for roughly the first six rounds only, so a late-season miss is a
real regression rather than an accepted exception; and fallbacks are recorded
per round so the rate can be read off, because sec 6 says a high rate means
the warm start needs revisiting, "not something to quietly absorb into the
relaxed-target carve-out". `run_conditioning` takes the two fits as callables
and a clock, so unit tests exercise the protocol without sampling anything.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# Sec 6 says "a couple of minutes" and "relaxed ... for early-season rounds"
# without naming numbers; these are this repo's reading of both. See
# docs/autopilot/decisions.md.
WARM_START_TARGET_S = 120.0      # "results within a couple of minutes of quali ending"
RELAXED_TARGET_S = 300.0         # early-season fallback allowance
EARLY_SEASON_ROUNDS = 6          # sec 6: "roughly the first ~6 rounds"

WARM_START = "warm_start"
FALLBACK_FULL_REFIT = "fallback_full_refit"


def time_target_s(round_: int, warm_target_s: float = WARM_START_TARGET_S,
                   relaxed_target_s: float = RELAXED_TARGET_S,
                   early_rounds: int = EARLY_SEASON_ROUNDS) -> float:
    """The wall-clock target a round that fell back is held to: relaxed for
    rounds 1..early_rounds, the original target from there on (sec 6)."""
    return relaxed_target_s if int(round_) <= early_rounds else warm_target_s


@dataclass
class ConditioningRecord:
    """One round's conditioning outcome -- the per-round record sec 6 requires."""
    season: int
    round: int
    path: str                    # WARM_START | FALLBACK_FULL_REFIT
    fell_back: bool
    early_season: bool
    warm_start_s: float
    fallback_s: float            # 0.0 when the warm start hit its target
    total_s: float
    warm_target_s: float
    applied_target_s: float      # relaxed one for early rounds, original otherwise
    regression: bool             # missed the target that actually applies to it


def run_conditioning(season: int, round_: int, warm_fit, full_fit,
                      clock=time.perf_counter,
                      warm_target_s: float = WARM_START_TARGET_S,
                      relaxed_target_s: float = RELAXED_TARGET_S,
                      early_rounds: int = EARLY_SEASON_ROUNDS):
    """Run sec 6's conditioning protocol. Returns (idata, ConditioningRecord).

    `warm_fit`/`full_fit` are zero-argument callables returning an
    InferenceData. `full_fit` runs only on a missed target, and its result is
    then the one returned -- it refits on the same quali observation, so it is
    the better posterior, not just the slower one. A missed target never
    raises: sec 6 says a slow round "falls back rather than blocking the run"."""
    t0 = clock()
    idata = warm_fit()
    warm_s = clock() - t0

    fell_back = warm_s > warm_target_s
    fallback_s = 0.0
    if fell_back:
        t1 = clock()
        idata = full_fit()
        fallback_s = clock() - t1

    total_s = warm_s + fallback_s
    # a warm start inside its target is on time by definition; only a round
    # that fell back is held to a target at all
    applied = (time_target_s(round_, warm_target_s, relaxed_target_s, early_rounds)
                if fell_back else warm_target_s)

    return idata, ConditioningRecord(
        season=int(season), round=int(round_),
        path=FALLBACK_FULL_REFIT if fell_back else WARM_START,
        fell_back=fell_back, early_season=int(round_) <= early_rounds,
        warm_start_s=warm_s, fallback_s=fallback_s, total_s=total_s,
        warm_target_s=warm_target_s, applied_target_s=applied,
        regression=fell_back and total_s > applied,
    )


# ---------------------------------------------------------------------------
# The per-round log (sec 6: recorded, and countable). Each `gp race --mode
# post-quali` is its own process handling one round, so the log is a file
# appended to rather than an object held in memory.
# ---------------------------------------------------------------------------

def append_record(path: str, record: ConditioningRecord) -> str:
    """Append one round to the conditioning log, writing a header if new."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    pd.DataFrame([asdict(record)]).to_csv(path, mode="a" if exists else "w",
                                           header=not exists, index=False)
    return path


def read_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fallback_rate(log: pd.DataFrame) -> float:
    """Sec 6: most rounds falling back means the warm start needs revisiting."""
    return float(log["fell_back"].mean()) if len(log) else 0.0


def regressions(log: pd.DataFrame) -> pd.DataFrame:
    """Rounds that missed the target that applies to them -- sec 6's "a real
    regression, not an accepted exception"."""
    return log[log["regression"].astype(bool)]


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------

def _fit_to_shape(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Reshape a pre-weekend posterior mean onto the conditioned model's shape.
    The only axis that moves is the random walk's round axis, which grows by one
    when the conditioned round is new; that step warm-starts at zero."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape == tuple(shape):
        return arr
    out = np.zeros(shape, dtype=float)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
    out[slices] = arr[slices]
    return out


def warm_start_initvals(prior_idata, data) -> dict:
    """Posterior means from the pre-weekend fit, shaped for the conditioned
    model -- the `initvals` sec 6's warm start needs. Team/driver vocabularies
    are built from the full dataset and so are stable across fits (see
    `PaceData`'s docstring); only the round axis changes."""
    shapes = {
        "car0": (data.n_teams,),
        "car_step_raw": (data.n_teams, max(data.n_rounds - 1, 0)),
        "driver_skill": (data.n_drivers,),
        "quali_offset": (data.n_drivers,),
        "driver_track": (data.n_drivers, data.n_circuit_types),
        "race_nu": (), "race_sigma": (), "quali_sigma": (),
    }
    initvals = {}
    for name, shape in shapes.items():
        if name not in prior_idata.posterior:
            continue
        mean = prior_idata.posterior[name].mean(dim=("chain", "draw")).to_numpy()
        initvals[name] = _fit_to_shape(mean, shape) if shape else float(mean)
    return initvals


WARM_TUNE, FULL_TUNE, DRAWS = 100, 400, 400  # sec 6's warm start is "a short warmup"


def condition_on_quali(driver_rounds: pd.DataFrame, driver_meta, season: int, round_: int,
                        prior_idata, chains: int = 2, seed: int = 0, clock=time.perf_counter,
                        warm_target_s: float = WARM_START_TARGET_S,
                        relaxed_target_s: float = RELAXED_TARGET_S):
    """Refit the pace model with the round's quali added as an observation.
    Returns (idata, PaceData, ConditioningRecord). The round's *race* rows are
    held out -- they are what is being forecast."""
    from model.pace import build_pace_data, fit_nuts

    data = build_pace_data(driver_rounds, driver_meta, through_season=season,
                            through_round=round_, hold_out_race_round=(season, round_))
    initvals = warm_start_initvals(prior_idata, data)

    idata, record = run_conditioning(
        season, round_,
        lambda: fit_nuts(data, draws=DRAWS, tune=WARM_TUNE, chains=chains,
                          seed=seed, initvals=initvals),
        lambda: fit_nuts(data, draws=DRAWS, tune=FULL_TUNE, chains=chains, seed=seed),
        clock=clock, warm_target_s=warm_target_s, relaxed_target_s=relaxed_target_s,
    )
    return idata, data, record
