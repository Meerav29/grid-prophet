"""Per-session wet/dry flag (spec sec 5, the weather row).

Sec 5 gives weather one line: FastF1 supplies `weather (dry/wet, air temp)`
per session, and **"wet races get their own noise scale"**. This module is
the single place that decides what counts as wet, so the pace likelihood
(`model.pace`) and the Phase 0 validator (`validate_driver_rounds`) cannot
drift apart on the question.

**Three states, not two.** `data/driver_rounds.csv` carries `is_wet` per
session row, written by `collect_v2._weather_summary` from FastF1's
`Rainfall` channel. A session whose weather FastF1 could not supply is
*unknown*, written as a blank cell, and that is not the same fact as a dry
session. Sec 5 sources the flag from FastF1, so nothing here invents one:
`wet_state` keeps unknown as unknown, and only `wet_flags` -- what the model
consumes -- collapses it to dry. Calling unknown *wet* would hand a wet noise
scale to rounds nobody measured; giving it a third likelihood branch would be
a noise scale fitted to "we don't know".

Fixtures and toy frames written before this slice have no `is_wet` column at
all. That is the same situation -- no recorded weather -- so a missing column
degrades to dry rather than raising, and a frame with no wet row produces
exactly the model it produced before this slice (see `model.pace.build_model`).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WET_COLUMN = "is_wet"

# The CSV round-trips `is_wet` as a bool, but a hand-edited fixture or an
# older collection can land a string or a 0/1 in the column. Anything not
# recognised is unknown rather than an error: a typo must not silently become
# a dry round, and must not stop a fit either.
_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "wet"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "dry"}


def _coerce_one(value) -> bool | None:
    """One cell -> True (wet), False (dry), or None (not recorded)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return None


def wet_state(frame: pd.DataFrame) -> pd.Series:
    """Per-row True / False / None, aligned to `frame.index`.

    A frame with no `is_wet` column is all-unknown, not all-dry -- the
    difference matters to the validator's coverage report even though the
    model treats the two the same way.
    """
    if WET_COLUMN not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)
    raw = frame[WET_COLUMN]
    states = raw.map(_coerce_one).astype(object)
    # Blank cells are legitimately "not recorded"; a non-blank cell that still
    # came out unknown is a value nobody recognised, and would otherwise fit
    # as a dry round without a word.
    unrecognised = raw[states.isna() & raw.notna()]
    if not unrecognised.empty:
        log.warning("is_wet: %d unrecognised value(s) %s treated as not recorded (dry in the fit)",
                    len(unrecognised), sorted({repr(v) for v in unrecognised})[:5])
    return states


def wet_flags(frame: pd.DataFrame) -> np.ndarray:
    """Per-row boolean the likelihood consumes: unknown collapses to dry."""
    return np.array([state is True for state in wet_state(frame)], dtype=bool)


def weather_coverage(frame: pd.DataFrame) -> dict:
    """Row counts by state -- how much of the window actually has weather."""
    states = list(wet_state(frame))
    return {
        "wet": sum(1 for s in states if s is True),
        "dry": sum(1 for s in states if s is False),
        "unknown": sum(1 for s in states if s is None),
    }


def observation_sigma_scale(is_wet, wet_ratio):
    """Per-observation multiplier on the dry race noise scale.

    1.0 on a dry row, `wet_ratio` on a wet one. Written as arithmetic rather
    than a branch so the same function serves a plain float `wet_ratio` (unit
    tests, diagnostics) and a pytensor scalar (inside `build_model`'s graph),
    which keeps the rule the model applies and the rule the tests check from
    being two separate pieces of code.
    """
    wet = np.asarray(is_wet, dtype=float)
    return 1.0 + wet * (wet_ratio - 1.0)
