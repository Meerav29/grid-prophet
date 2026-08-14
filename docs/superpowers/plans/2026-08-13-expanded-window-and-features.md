# Expanded Early-Season Window & Feature/Model Complexity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen Grid Prophet's early-season signal window from a hardcoded 2 rounds to the number of rounds actually completed so far (applied consistently to training and prediction), add four new season-level features derived from already-collected data, and add LassoCV feature selection ahead of the Ridge/XGBoost comparison so the larger feature set doesn't overfit the ~120-row dataset.

**Architecture:** `features.py` gains a parametrized early-round window and an auto-detection helper; `train.py` gains a feature-selection step that narrows the column list before model comparison and stores the selected list in the saved model bundle; `__main__.py`'s `run`/`update` commands compute the round count once and thread it through `features.py` (training AND the live 2026 row) and `predict.py` so train/predict windows never mismatch.

**Tech Stack:** Python, pandas, numpy, scikit-learn (`LassoCV`, `StandardScaler`, existing `Ridge`/`SimpleImputer`/`Pipeline`), XGBoost, FastF1, pytest.

## Global Constraints

- Season-level unit of analysis is a locked decision — do not introduce race-level training rows (`Project.md`).
- No new FastF1 data collection (no `'Q'` qualifying session pulls) — new features must be derivable from columns already in `data/race_results.csv` (`grid_position`, `finish_position`, `classification`).
- `predict.py --rounds N` must keep working as a standalone override, independent of the new auto-detection path.
- Existing public function signatures gain new *optional* parameters with backward-compatible defaults wherever a caller/test doesn't need to change (e.g. `retrain_and_save`, `_print_summary`), except where the spec explicitly requires the call site to change (`_early_season_features`, `build_features`, `update`/`run` CLI flow).
- `is_rule_change_year` must never be dropped by feature selection — `leave_one_season_out_cv` and `retrain_and_save` depend on it for sample weighting.

---

### Task 1: `features.py` — parametrize the early-round window + auto-detection helper

**Files:**
- Modify: `src/features.py:1-20` (imports, constants), `src/features.py:79-98` (`_early_season_features`), `src/features.py:256-329` (`build_features`, `main`)
- Test: `tests/test_features.py` (new file)

**Interfaces:**
- Produces: `_early_season_features(results: pd.DataFrame, early_rounds: int = EARLY_ROUNDS) -> pd.DataFrame` (existing columns unchanged; new feature columns added in Task 2)
- Produces: `build_features(early_rounds: int | None = None) -> pd.DataFrame` — `None` falls back to the `EARLY_ROUNDS` constant.
- Produces: `latest_completed_round(year: int) -> int` — returns 1 if no rounds of `year` have completed yet.
- Produces: `features.main()` now supports `--early-rounds N` (optional; unset = `EARLY_ROUNDS`).

- [ ] **Step 1: Write failing tests for the parametrized window and auto-detection**

Create `tests/test_features.py`:

```python
import unittest.mock as mock
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from features import _early_season_features, build_features, latest_completed_round, EARLY_ROUNDS


def _make_results():
    return pd.DataFrame([
        {"year": 2024, "round": 1, "constructor_canonical": "TeamA", "driver": "Alice",
         "points": 25.0, "grid_position": 2, "finish_position": 1, "classification": "Finished"},
        {"year": 2024, "round": 1, "constructor_canonical": "TeamA", "driver": "Bob",
         "points": 0.0, "grid_position": 5, "finish_position": np.nan, "classification": "DNF"},
        {"year": 2024, "round": 1, "constructor_canonical": "TeamB", "driver": "Carl",
         "points": 18.0, "grid_position": 1, "finish_position": 2, "classification": "Finished"},
        {"year": 2024, "round": 1, "constructor_canonical": "TeamB", "driver": "Dana",
         "points": 15.0, "grid_position": 3, "finish_position": 3, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamA", "driver": "Alice",
         "points": 18.0, "grid_position": 1, "finish_position": 2, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamA", "driver": "Bob",
         "points": 25.0, "grid_position": 3, "finish_position": 1, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamB", "driver": "Carl",
         "points": 10.0, "grid_position": 2, "finish_position": 4, "classification": "Finished"},
        {"year": 2024, "round": 2, "constructor_canonical": "TeamB", "driver": "Dana",
         "points": 12.0, "grid_position": 4, "finish_position": 3, "classification": "Finished"},
        {"year": 2024, "round": 3, "constructor_canonical": "TeamA", "driver": "Alice",
         "points": 15.0, "grid_position": 1, "finish_position": 3, "classification": "Finished"},
        {"year": 2024, "round": 3, "constructor_canonical": "TeamB", "driver": "Carl",
         "points": 18.0, "grid_position": 2, "finish_position": 2, "classification": "Finished"},
    ])


def test_early_season_features_default_window_uses_EARLY_ROUNDS():
    result = _early_season_features(_make_results())
    # default EARLY_ROUNDS=2 excludes round 3 entirely
    teamA = result[result["constructor_canonical"] == "TeamA"].iloc[0]
    # round1 pts=25 (Alice) + round2 pts=18+25=43 -> team total 68; round3 (15) excluded
    assert teamA["early_points_share"] == pytest.approx(68 / (68 + 18 + 25))


def test_early_season_features_custom_window_includes_more_rounds():
    result = _early_season_features(_make_results(), early_rounds=3)
    teamA = result[result["constructor_canonical"] == "TeamA"].iloc[0]
    # now round 3 (15 pts) is included in the team total and the window total
    window_total = (25 + 0) + (18 + 25) + 15 + (18 + 10 + 12 + 18)
    team_total = 25 + 18 + 25 + 15
    assert teamA["early_points_share"] == pytest.approx(team_total / window_total)


def test_build_features_accepts_early_rounds_override():
    results = pd.DataFrame([
        {"year": 2024, "round": r, "constructor": "TeamA", "driver": "Alice",
         "points": p, "grid_position": 1, "finish_position": pos, "classification": "Finished"}
        for r, p, pos in [(1, 25.0, 1), (2, 18.0, 2), (3, 15.0, 3)]
    ])
    standings = pd.DataFrame({"year": [2024], "constructor": ["TeamA"], "total_points": [58.0]})

    with mock.patch("features.pd.read_csv", side_effect=[results, standings]):
        feats_default = build_features()
    with mock.patch("features.pd.read_csv", side_effect=[results, standings]):
        feats_wide = build_features(early_rounds=3)

    row_default = feats_default[feats_default["constructor"] == "TeamA"].iloc[0]
    row_wide = feats_wide[feats_wide["constructor"] == "TeamA"].iloc[0]
    assert row_default["early_points_share"] != row_wide["early_points_share"]


def test_latest_completed_round_returns_max_completed_round():
    schedule = pd.DataFrame({
        "RoundNumber": [1, 2, 3, 4],
        "EventDate": pd.to_datetime(["2026-03-01", "2026-03-15", "2026-08-01", "2026-08-20"]),
    })
    fake_today = dt.date(2026, 8, 13)
    with mock.patch("fastf1.get_event_schedule", return_value=schedule), \
         mock.patch("datetime.date") as mock_date:
        mock_date.today.return_value = fake_today
        result = latest_completed_round(2026)
    assert result == 3


def test_latest_completed_round_returns_1_when_none_completed():
    schedule = pd.DataFrame({
        "RoundNumber": [1, 2],
        "EventDate": pd.to_datetime(["2026-03-01", "2026-03-15"]),
    })
    fake_today = dt.date(2026, 1, 1)
    with mock.patch("fastf1.get_event_schedule", return_value=schedule), \
         mock.patch("datetime.date") as mock_date:
        mock_date.today.return_value = fake_today
        result = latest_completed_round(2026)
    assert result == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_features.py -v`
Expected: FAIL — `_early_season_features` doesn't accept `early_rounds`, `build_features` doesn't accept `early_rounds`, `latest_completed_round` doesn't exist.

- [ ] **Step 3: Implement the parametrized window and auto-detection helper**

In `src/features.py`, add `import argparse` to the top-level imports (alongside `logging`, `os`).

Replace the `_early_season_features` signature and its filter line:

```python
def _early_season_features(results: pd.DataFrame, early_rounds: int = EARLY_ROUNDS) -> pd.DataFrame:
    """
    early_points_share  — team's share of all points from rounds 1..early_rounds
    early_avg_finish    — average finishing position in those rounds
    """
    early = results[results["round"] <= early_rounds].copy()
```//existing body below this line is unchanged for this task (columns added in Task 2)

Add a new function near the bottom of the "Helpers" section (after `_first_rebrand_season`, before "Feature builders"):

```python
def latest_completed_round(year: int) -> int:
    """Return the latest completed round number for `year` from FastF1.
    Falls back to 1 if no rounds of `year` have completed yet."""
    import fastf1
    import datetime

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    today = datetime.date.today()
    completed = schedule[schedule["EventDate"].dt.date < today]
    if completed.empty:
        return 1
    return int(completed["RoundNumber"].max())
```

Update `build_features` to accept and thread the override:

```python
def build_features(early_rounds: int | None = None) -> pd.DataFrame:
    effective_early_rounds = early_rounds if early_rounds is not None else EARLY_ROUNDS

    race_csv = os.path.join(DATA_DIR, "race_results.csv")
    standings_csv = os.path.join(DATA_DIR, "constructor_standings.csv")

    log.info("Loading %s ...", race_csv)
    results = pd.read_csv(race_csv)

    log.info("Loading %s ...", standings_csv)
    standings_raw = pd.read_csv(standings_csv)
```

(rest of the function body unchanged until the early-season call, which becomes:)

```python
    log.info("Computing early-season features (window: rounds 1-%d) ...", effective_early_rounds)
    early = _early_season_features(results, early_rounds=effective_early_rounds)
```

Update `main()` to add the CLI flag:

```python
def main():
    parser = argparse.ArgumentParser(description="Build Grid Prophet feature matrix.")
    parser.add_argument(
        "--early-rounds", type=int, default=None, dest="early_rounds",
        help="Override the early-season round window (default: EARLY_ROUNDS constant)",
    )
    args = parser.parse_args()

    features = build_features(early_rounds=args.early_rounds)
    out_path = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    log.info("Saved features → %s", out_path)
    log.info("\n%s", features.head(20).to_string())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_features.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Run full existing suite to check for regressions**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: PASS except `tests/test_main.py::test_update_subcommand_calls_predict_with_round` and `tests/test_train.py` fixture-dependent tests — those are addressed in Tasks 4 and 6. Confirm no *other* unexpected failures.

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: parametrize early-season round window in features.py"
```

---

### Task 2: `features.py` — add four new engineered features

**Files:**
- Modify: `src/features.py:79-98` (`_early_season_features`)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `_early_season_features(results, early_rounds)` from Task 1.
- Produces: `_early_season_features` output now also includes `constructor_dnf_rate`, `avg_grid_to_finish_delta`, `teammate_head_to_head`, `development_trend` columns.
- Produces (private helpers): `_teammate_h2h_rate(group: pd.DataFrame) -> float`, `_development_trend(group: pd.DataFrame) -> float`.

- [ ] **Step 1: Write failing tests for the new feature columns**

Append to `tests/test_features.py`:

```python
def test_new_features_present_and_correct():
    result = _early_season_features(_make_results(), early_rounds=2)
    teamA = result[result["constructor_canonical"] == "TeamA"].iloc[0]
    teamB = result[result["constructor_canonical"] == "TeamB"].iloc[0]

    # TeamA: round1 Alice finished, Bob DNF; round2 both finished -> 1 DNF / 4 entries
    assert teamA["constructor_dnf_rate"] == pytest.approx(0.25)
    # TeamB: no DNFs in the window
    assert teamB["constructor_dnf_rate"] == pytest.approx(0.0)

    # TeamA classified entries: round1 Alice(grid2,fin1,+1); round2 Alice(grid1,fin2,-1), Bob(grid3,fin1,+2)
    assert teamA["avg_grid_to_finish_delta"] == pytest.approx((1 + (-1) + 2) / 3)
    # TeamB: round1 Carl(-1), Dana(0); round2 Carl(-2), Dana(1)
    assert teamB["avg_grid_to_finish_delta"] == pytest.approx((-1 + 0 - 2 + 1) / 4)

    # TeamA: round1 only Alice classified (Bob DNF) -> race excluded; round2 both classified,
    # alphabetically-first driver "Alice" finishes 2nd vs Bob's 1st -> loses that race -> 0/1
    assert teamA["teammate_head_to_head"] == pytest.approx(0.0)
    # TeamB: round1 "Carl" (alpha-first) finishes 2nd = best -> win; round2 Carl finishes 4th,
    # Dana 3rd = best -> Carl loses -> 1/2
    assert teamB["teammate_head_to_head"] == pytest.approx(0.5)

    # TeamA: round1 mean finish = mean(1, NaN) = 1.0; round2 mean finish = mean(2,1) = 1.5
    # slope = (1.5 - 1.0) / (2 - 1) = 0.5
    assert teamA["development_trend"] == pytest.approx(0.5)
    # TeamB: round1 mean = 2.5, round2 mean = 3.5 -> slope = 1.0
    assert teamB["development_trend"] == pytest.approx(1.0)


def test_development_trend_is_zero_with_single_round_window():
    result = _early_season_features(_make_results(), early_rounds=1)
    teamA = result[result["constructor_canonical"] == "TeamA"].iloc[0]
    assert teamA["development_trend"] == pytest.approx(0.0)


def test_teammate_head_to_head_excludes_races_with_lt_2_classified_drivers():
    # TeamA round1 has only 1 classified driver (Bob DNF) -> excluded from denominator
    result = _early_season_features(_make_results(), early_rounds=1)
    teamA = result[result["constructor_canonical"] == "TeamA"].iloc[0]
    assert np.isnan(teamA["teammate_head_to_head"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_features.py -v -k "new_features or development_trend_is_zero or excludes_races"`
Expected: FAIL — `KeyError` on the new column names.

- [ ] **Step 3: Implement the new features**

Replace the full body of `_early_season_features` in `src/features.py`:

```python
def _teammate_h2h_rate(group: pd.DataFrame) -> float:
    """Fraction of races (within the window) where the alphabetically-first
    driver on the team beat their teammate(s) on finish position. Races
    where the team fielded fewer than 2 classified drivers are excluded
    from the denominator. NaN if no races qualify."""
    wins = 0
    total = 0
    for _, race in group.groupby("round"):
        drivers = sorted(race["driver"].unique())
        if len(drivers) < 2:
            continue
        first_driver = drivers[0]
        first_finish = race.loc[race["driver"] == first_driver, "finish_position"].min()
        best_finish = race["finish_position"].min()
        total += 1
        if first_finish == best_finish:
            wins += 1
    return wins / total if total else np.nan


def _development_trend(group: pd.DataFrame) -> float:
    """Slope of avg finish position per round vs round number, over the
    window. Negative = improving (finish position numbers decreasing).
    0.0 if the window contains fewer than 2 rounds with data."""
    per_round = group.groupby("round")["finish_position"].mean().dropna()
    if len(per_round) < 2:
        return 0.0
    x = per_round.index.values.astype(float)
    y = per_round.values.astype(float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _early_season_features(results: pd.DataFrame, early_rounds: int = EARLY_ROUNDS) -> pd.DataFrame:
    """
    early_points_share            — team's share of all points from rounds 1..early_rounds
    early_avg_finish               — average finishing position in those rounds
    constructor_dnf_rate           — fraction of the team's entries classified DNF
    avg_grid_to_finish_delta       — mean(grid_position - finish_position), classified entries only
    teammate_head_to_head          — fraction of races where the alphabetically-first-named
                                      teammate beat the other teammate(s) on finish position
    development_trend              — slope of avg finish position per round (negative = improving)
    """
    early = results[results["round"] <= early_rounds].copy()

    total_early_points = early.groupby("year")["points"].sum().rename("total_early_pts")

    team_early = (
        early.groupby(["year", "constructor_canonical"])
        .agg(team_early_pts=("points", "sum"), team_early_finish=("finish_position", "mean"))
        .reset_index()
    )
    team_early = team_early.merge(total_early_points, on="year")
    team_early["early_points_share"] = (
        team_early["team_early_pts"] / team_early["total_early_pts"]
    )
    team_early["early_avg_finish"] = team_early["team_early_finish"]

    dnf_rate = (
        early.groupby(["year", "constructor_canonical"])["classification"]
        .apply(lambda s: (s == "DNF").mean())
        .rename("constructor_dnf_rate")
        .reset_index()
    )

    classified = early.dropna(subset=["grid_position", "finish_position"]).copy()
    classified["grid_to_finish_delta"] = classified["grid_position"] - classified["finish_position"]
    grid_delta = (
        classified.groupby(["year", "constructor_canonical"])["grid_to_finish_delta"]
        .mean()
        .rename("avg_grid_to_finish_delta")
        .reset_index()
    )

    h2h = (
        classified.groupby(["year", "constructor_canonical"])
        .apply(_teammate_h2h_rate, include_groups=False)
        .rename("teammate_head_to_head")
        .reset_index()
    )

    trend = (
        early.groupby(["year", "constructor_canonical"])
        .apply(_development_trend, include_groups=False)
        .rename("development_trend")
        .reset_index()
    )

    team_early = team_early.merge(dnf_rate, on=["year", "constructor_canonical"], how="left")
    team_early = team_early.merge(grid_delta, on=["year", "constructor_canonical"], how="left")
    team_early = team_early.merge(h2h, on=["year", "constructor_canonical"], how="left")
    team_early = team_early.merge(trend, on=["year", "constructor_canonical"], how="left")

    return team_early[[
        "year", "constructor_canonical",
        "early_points_share", "early_avg_finish",
        "constructor_dnf_rate", "avg_grid_to_finish_delta",
        "teammate_head_to_head", "development_trend",
    ]]
```

Note: `include_groups=False` requires pandas >= 2.2. Check the installed version first with `python -c "import pandas; print(pandas.__version__)"` — if it's older than 2.2, drop the `include_groups=False` kwarg from both `.apply(...)` calls instead (older pandas doesn't accept it and doesn't emit the deprecation warning it silences).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_features.py -v`
Expected: PASS (all tests, including Task 1's)

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add DNF rate, grid-to-finish delta, teammate H2H, and development trend features"
```

---

### Task 3: `predict.py` — thread the early-round window into 2026 feature building

**Files:**
- Modify: `src/predict.py:98-99` (`build_2026_features`)
- Test: `tests/test_predict.py` (verify existing tests still pass; no new columns need new tests since `FEATURE_COLS`-driven tests already iterate dynamically)

**Interfaces:**
- Consumes: `_early_season_features(results, early_rounds)` from Task 1/2.
- No signature changes to `build_2026_features(n_rounds: int = EARLY_ROUNDS)` — only its internal call to `_early_season_features` changes.

- [ ] **Step 1: Update the call site**

In `src/predict.py`, inside `build_2026_features`, change:

```python
    early = _early_season_features(results)
```

to:

```python
    early = _early_season_features(results, early_rounds=n_rounds)
```

- [ ] **Step 2: Run the predict test suite**

Run: `PYTHONPATH=src python -m pytest tests/test_predict.py -v`
Expected: PASS — existing tests already assert `FEATURE_COLS` (from `train.py`) are all present in the output; they don't hardcode column counts, so they pass as-is here. (`FEATURE_COLS` itself is expanded in Task 4 — re-run this file again after Task 4 to confirm it still passes with the larger list.)

- [ ] **Step 3: Commit**

```bash
git add src/predict.py
git commit -m "fix: use the requested round window for 2026 early-season features"
```

---

### Task 4: `train.py` — expand `FEATURE_COLS` and update fixtures

**Files:**
- Modify: `src/train.py:28-37` (`FEATURE_COLS`)
- Modify: `tests/test_train.py` (fixture dicts that hardcode feature columns)

**Interfaces:**
- Produces: `FEATURE_COLS` now has 12 entries (8 existing + 4 new).

- [ ] **Step 1: Update `FEATURE_COLS`**

In `src/train.py`, replace:

```python
FEATURE_COLS = [
    "early_points_share",
    "early_avg_finish",
    "is_rule_change_year",
    "rule_change_adaptation_score",
    "prev_year_points_share",
    "prev_year_standing",
    "constructor_win_rate_5yr",
    "avg_driver_career_points_per_race",
]
```

with:

```python
FEATURE_COLS = [
    "early_points_share",
    "early_avg_finish",
    "constructor_dnf_rate",
    "avg_grid_to_finish_delta",
    "teammate_head_to_head",
    "development_trend",
    "is_rule_change_year",
    "rule_change_adaptation_score",
    "prev_year_points_share",
    "prev_year_standing",
    "constructor_win_rate_5yr",
    "avg_driver_career_points_per_race",
]
```

- [ ] **Step 2: Update `tests/test_train.py` fixtures to include the new columns**

In `test_load_data_returns_train_split`, add the four new keys to each row dict (both the loop-generated rows and the 2026 NaN-target row):

```python
                "constructor_dnf_rate": 0.1, "avg_grid_to_finish_delta": 0.0,
                "teammate_head_to_head": 0.5, "development_trend": 0.0,
```

placed right after `"early_avg_finish": ...,` in each of the two row-construction blocks in that test.

In `_make_fake_data()`, add the same four keys (with the same constant placeholder values) to its row dict, right after `"early_avg_finish": 5.0,`.

In `test_main_ensemble_flag_creates_ensemble_pkl`, add the same four keys to its row dict, right after `"early_avg_finish": 5.0,`.

- [ ] **Step 3: Run the train test suite to verify it fails first (columns missing), then passes after the fixture edit**

Run: `PYTHONPATH=src python -m pytest tests/test_train.py -v`
Expected before fixture edit: FAIL with `KeyError` on the new column names inside `load_data`.
Expected after fixture edit: PASS (all tests).

- [ ] **Step 4: Re-run `tests/test_predict.py` to confirm it still passes with the expanded `FEATURE_COLS`**

Run: `PYTHONPATH=src python -m pytest tests/test_predict.py -v`
Expected: PASS (its fixtures build feature dicts dynamically from `FEATURE_COLS`, so no edits needed there).

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add new engineered features to FEATURE_COLS"
```

---

### Task 5: `train.py` — LassoCV feature selection ahead of model comparison

**Files:**
- Modify: `src/train.py` (imports, new `select_features`, `retrain_and_save`, `_print_summary`, `main`)
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `select_features(X: pd.DataFrame, y: pd.Series) -> list[str]` — returns the subset of `X.columns` (in original order) with a nonzero LassoCV coefficient, always including `is_rule_change_year`, falling back to all of `X.columns` if selection would otherwise leave no informative columns.
- Modifies: `retrain_and_save(pipeline, X, y, meta, winner_name, rule_change_weight, model_path, feature_cols=None)` — `feature_cols` defaults to the module-level `FEATURE_COLS` when omitted (backward compatible).
- Modifies: `_print_summary(winner_name, best_weight, weight_scores, cv_df, pipeline, feature_cols=None)` — same default-to-`FEATURE_COLS` behavior.
- Modifies: `main()` — runs `select_features` right after `load_data`, narrows `X` to the selected columns before weight tuning and model comparison, and passes `feature_cols=selected_cols` to `retrain_and_save`, `_print_summary`, and the `--ensemble` bundle dict.

- [ ] **Step 1: Write failing tests for `select_features`**

Add to `tests/test_train.py`:

```python
from train import select_features


def test_select_features_keeps_signal_drops_pure_noise():
    rng = np.random.RandomState(0)
    n = 100
    signal = np.linspace(0, 1, n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({
        "early_points_share": signal,
        "pure_noise": noise,
        "is_rule_change_year": np.zeros(n),
    })
    y = pd.Series(signal * 2 + 0.01)

    selected = select_features(X, y)

    assert "early_points_share" in selected
    assert "pure_noise" not in selected


def test_select_features_always_keeps_is_rule_change_year():
    # is_rule_change_year is constant here, so Lasso would naturally zero it —
    # select_features must force-include it anyway.
    n = 30
    X = pd.DataFrame({
        "early_points_share": np.linspace(0, 1, n),
        "is_rule_change_year": np.zeros(n),
    })
    y = pd.Series(np.linspace(0, 1, n) * 2)

    selected = select_features(X, y)

    assert "is_rule_change_year" in selected


def test_select_features_preserves_original_column_order():
    n = 30
    X = pd.DataFrame({
        "early_points_share": np.linspace(0, 1, n),
        "avg_grid_to_finish_delta": np.linspace(1, 0, n),
        "is_rule_change_year": np.zeros(n),
    })
    y = pd.Series(np.linspace(0, 1, n))

    selected = select_features(X, y)

    assert selected == [c for c in X.columns if c in selected]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_train.py -v -k select_features`
Expected: FAIL — `ImportError: cannot import name 'select_features'`.

- [ ] **Step 3: Implement `select_features` and thread it through `main()`**

In `src/train.py`, add imports (alongside the existing `sklearn` imports):

```python
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
```

(Replace the existing `from sklearn.linear_model import Ridge` line with the combined import above.)

Add a constant near `RULE_CHANGE_WEIGHT_CANDIDATES`:

```python
ALWAYS_KEEP_FEATURES = ["is_rule_change_year"]
```

Add the new function (place it after `load_data`, before `leave_one_season_out_cv`):

```python
def select_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Run LassoCV on standardized, imputed features and return the columns
    with a nonzero coefficient, in their original order. Always includes
    ALWAYS_KEEP_FEATURES regardless of the Lasso outcome (leave_one_season_out_cv
    and retrain_and_save depend on is_rule_change_year for sample weighting).
    Falls back to the full column set if Lasso zeroes out everything else."""
    imputer = SimpleImputer()
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_splits = max(2, min(5, len(X)))
    lasso = LassoCV(cv=n_splits, random_state=42, max_iter=5000).fit(X_scaled, y)

    selected = {col for col, coef in zip(X.columns, lasso.coef_) if coef != 0}
    for col in ALWAYS_KEEP_FEATURES:
        if col in X.columns:
            selected.add(col)

    meaningful = selected - set(ALWAYS_KEEP_FEATURES)
    if not meaningful:
        log.warning("LassoCV dropped nearly all features; keeping full feature set.")
        return list(X.columns)

    dropped = [c for c in X.columns if c not in selected]
    if dropped:
        log.info("Feature selection dropped: %s", dropped)
    return [c for c in X.columns if c in selected]
```

Update `retrain_and_save` signature and body:

```python
def retrain_and_save(pipeline, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                     winner_name: str, rule_change_weight: float, model_path: str,
                     feature_cols: list[str] | None = None):
    """Retrain pipeline on all data with best weight and save a bundle to model_path."""
    feature_cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS

    rc_col = X["is_rule_change_year"]
    sample_weights = np.where(rc_col == 1, rule_change_weight, 1.0)

    if isinstance(pipeline, Pipeline):
        last_step_name = pipeline.steps[-1][0]
        pipeline.fit(X, y, **{f"{last_step_name}__sample_weight": sample_weights})
    else:
        pipeline.fit(X, y)

    bundle = {
        "model": pipeline,
        "feature_cols": feature_cols,
        "winner_name": winner_name,
        "rule_change_weight": rule_change_weight,
    }
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved model bundle -> %s", model_path)
```

Update `_print_summary` signature and its two `zip(FEATURE_COLS, ...)` call sites:

```python
def _print_summary(winner_name: str, best_weight: float, weight_scores: dict,
                   cv_df: pd.DataFrame, pipeline, feature_cols: list[str] | None = None):
    """Print training summary: weight tuning results, model comparison, feature importances."""
    feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS

    print("\n" + "=" * 60)
    print("GRID PROPHET - TRAINING SUMMARY")
    print("=" * 60)

    print("\n-- Rule-change weight tuning --")
    for w, s in sorted(weight_scores.items()):
        marker = " << BEST" if w == best_weight else ""
        print(f"  weight {w:.1f}  avg Spearman {s:.4f}{marker}")

    print("\n-- Model comparison (leave-one-season-out CV) --")
    summary = cv_df.groupby("model")[["spearman", "mae"]].mean()
    for model_name, row in summary.iterrows():
        marker = " << WINNER" if model_name == winner_name else ""
        print(f"  {model_name}: avg Spearman {row['spearman']:.4f}, avg MAE {row['mae']:.4f}{marker}")

    print(f"\n-- Winner: {winner_name} (weight={best_weight:.1f}) --")

    last_estimator = pipeline.steps[-1][1]
    if hasattr(last_estimator, "feature_importances_"):
        print("\n-- Feature importances (XGBoost) --")
        for col, imp in sorted(zip(feature_cols, last_estimator.feature_importances_), key=lambda x: -x[1]):
            print(f"  {col:<42} {imp:.4f}")
    elif hasattr(last_estimator, "coef_"):
        print("\n-- Ridge coefficients --")
        for col, coef in sorted(zip(feature_cols, last_estimator.coef_), key=lambda x: -abs(x[1])):
            print(f"  {col:<42} {coef:+.4f}")

    print("=" * 60 + "\n")
```

Update `main()` to run selection and pass `feature_cols` through:

```python
def main():
    parser = argparse.ArgumentParser(description="Train Grid Prophet model.")
    parser.add_argument(
        "--features", default=os.path.join(DATA_DIR, "features.csv"),
    )
    parser.add_argument(
        "--model-out", default=os.path.join(MODELS_DIR, "Grid_Prophet_model.pkl"),
    )
    parser.add_argument(
        "--cv-out", default=os.path.join(DATA_DIR, "cv_results.csv"),
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="After training base model, build and save ensemble bundle",
    )
    parser.add_argument(
        "--ensemble-out",
        default=os.path.join(MODELS_DIR, "Grid_Prophet_ensemble.pkl"),
    )
    args = parser.parse_args()

    log.info("Loading data from %s ...", args.features)
    X, y, meta = load_data(args.features)
    log.info("Training set: %d rows, %d seasons", len(X), meta["year"].nunique())

    log.info("Selecting features via LassoCV ...")
    selected_cols = select_features(X, y)
    log.info("Selected %d/%d features: %s", len(selected_cols), len(FEATURE_COLS), selected_cols)
    X = X[selected_cols]

    tune_pipeline = Pipeline([("imp", SimpleImputer()), ("reg", Ridge(alpha=1.0))])
    log.info("Tuning rule-change sample weight ...")
    best_weight, weight_scores = tune_rule_change_weight(tune_pipeline, X, y, meta)
    log.info("Best rule-change weight: %.1f", best_weight)

    log.info("Comparing XGBoost vs Ridge ...")
    winner_name, winner_pipeline, cv_df = compare_models(X, y, meta, best_weight)

    cv_df.to_csv(args.cv_out, index=False)
    log.info("CV results saved -> %s", args.cv_out)

    log.info("Retraining %s on all data ...", winner_name)
    retrain_and_save(
        winner_pipeline, X, y, meta,
        winner_name=winner_name,
        rule_change_weight=best_weight,
        model_path=args.model_out,
        feature_cols=selected_cols,
    )
    log.info("Done. Model saved to %s", args.model_out)

    _print_summary(winner_name, best_weight, weight_scores, cv_df, winner_pipeline, feature_cols=selected_cols)

    if args.ensemble:
        from ensemble import build_ensemble, save_ensemble
        log.info("Building ensemble model ...")
        ens_bundle = build_ensemble(
            {"model": winner_pipeline, "feature_cols": selected_cols,
             "winner_name": winner_name, "rule_change_weight": best_weight},
            X, y, meta,
        )
        save_ensemble(ens_bundle, args.ensemble_out)
        log.info("Ensemble saved -> %s", args.ensemble_out)
```

Note `compare_models` and `leave_one_season_out_cv` need no changes — they already operate generically on whatever `X` is passed in, and `X["is_rule_change_year"]` is guaranteed present because `select_features` always keeps it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_train.py -v`
Expected: PASS (all tests, including the pre-existing ones — `test_retrain_and_save_creates_pkl` still passes since it doesn't pass `feature_cols` and the default falls back to `FEATURE_COLS`).

- [ ] **Step 5: Run the full suite for regressions**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: PASS except `tests/test_main.py::test_update_subcommand_calls_predict_with_round` (addressed in Task 6).

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add LassoCV feature selection ahead of Ridge/XGBoost comparison"
```

---

### Task 6: `__main__.py` — auto-detect round count, rebuild features, retrain on `run`/`update`

**Files:**
- Modify: `src/__main__.py` (remove `_detect_latest_round`, rewrite `run`/`update` branches)
- Modify: `tests/test_main.py` (`test_run_subcommand_calls_all_four`, replace `test_update_subcommand_calls_predict_with_round`)

**Interfaces:**
- Consumes: `features.latest_completed_round(year: int) -> int` from Task 1.
- Removes: `__main__._detect_latest_round` (superseded by `features.latest_completed_round`).

- [ ] **Step 1: Update failing/changing tests first**

In `tests/test_main.py`, update `test_run_subcommand_calls_all_four`:

```python
def test_run_subcommand_calls_all_four():
    import __main__ as m
    with mock.patch("collect.main") as mc, \
         mock.patch("features.latest_completed_round", return_value=5) as mlcr, \
         mock.patch("features.main") as mf, \
         mock.patch("train.main") as mt, \
         mock.patch("predict.main") as mp:
        with mock.patch("sys.argv", ["grid_prophet", "run"]):
            m.main()
    mc.assert_called_once()
    mlcr.assert_called_once_with(2026)
    mf.assert_called_once()
    mt.assert_called_once()
    mp.assert_called_once()
```

Replace `test_update_subcommand_calls_predict_with_round` with:

```python
def test_update_subcommand_rebuilds_features_retrains_and_predicts():
    import __main__ as m
    captured = {}

    def capture_features():
        captured["features_argv"] = sys.argv[:]

    def capture_predict():
        captured["predict_argv"] = sys.argv[:]

    with mock.patch("features.latest_completed_round", return_value=5) as mlcr, \
         mock.patch("features.main", side_effect=capture_features) as mf, \
         mock.patch("train.main") as mt, \
         mock.patch("predict.main", side_effect=capture_predict) as mp:
        with mock.patch("sys.argv", ["grid_prophet", "update"]):
            m.main()

    mlcr.assert_called_once_with(2026)
    mf.assert_called_once()
    mt.assert_called_once()
    mp.assert_called_once()
    assert captured["features_argv"] == ["grid_prophet", "--early-rounds", "5"]
    assert captured["predict_argv"] == ["grid_prophet", "--rounds", "5"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_main.py -v`
Expected: FAIL — `test_run_subcommand_calls_all_four` fails because `features.latest_completed_round` is never called by the current `run` branch; the new `test_update_subcommand_rebuilds_...` fails with `AttributeError` (`_detect_latest_round` still referenced, `features.main`/`train.main` not called by current `update` branch).

- [ ] **Step 3: Rewrite `src/__main__.py`**

Replace the entire file:

```python
"""CLI entry point: python -m grid_prophet <command>."""

import sys


def main():
    import argparse
    # Imported inside main() to avoid heavy import-time side effects at module load
    import collect
    import features
    import train
    import predict

    parser = argparse.ArgumentParser(
        prog="grid_prophet",
        description="Grid Prophet F1 championship predictor.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    sub.add_parser("collect", help="Pull historical race data from FastF1")
    sub.add_parser("features", help="Build feature matrix")
    sub.add_parser("train", help="Train model")
    sub.add_parser("predict", help="Generate 2026 predictions")
    sub.add_parser("run", help="Run full pipeline: collect > features > train > predict")
    sub.add_parser("update", help="Auto-detect latest round, rebuild features, retrain, and re-predict")
    sub.add_parser("plots", help="Generate all visualisation charts")

    args, remaining = parser.parse_known_args()
    prog_name = sys.argv[0]  # save before any mutation
    # Pass remaining args through to sub-module so their own argparse flags work
    sys.argv = [prog_name] + remaining

    if args.command == "collect":
        collect.main()
    elif args.command == "features":
        features.main()
    elif args.command == "train":
        train.main()
    elif args.command == "predict":
        predict.main()
    elif args.command == "run":
        from predict import PREDICT_YEAR
        collect.main()
        n = features.latest_completed_round(PREDICT_YEAR)
        sys.argv = [prog_name, "--early-rounds", str(n)]
        features.main()
        sys.argv = [prog_name]
        train.main()
        sys.argv = [prog_name, "--rounds", str(n)]
        predict.main()
    elif args.command == "update":
        from predict import PREDICT_YEAR
        n = features.latest_completed_round(PREDICT_YEAR)
        sys.argv = [prog_name, "--early-rounds", str(n)]
        features.main()
        sys.argv = [prog_name]
        train.main()
        sys.argv = [prog_name, "--rounds", str(n)]
        predict.main()
    elif args.command == "plots":
        try:
            import visualize
        except ImportError:
            print("Error: visualize module not yet available. Run all pipeline stages first.", file=sys.stderr)
            sys.exit(1)
        visualize.plot_all()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_main.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: PASS (all tests, full suite green)

- [ ] **Step 6: Commit**

```bash
git add src/__main__.py tests/test_main.py
git commit -m "feat: auto-detect round count and rebuild/retrain on run and update"
```

---

### Task 7: Regenerate pipeline outputs and update README

**Files:**
- Modify (regenerated, not hand-edited): `data/features.csv`, `data/cv_results.csv`, `models/Grid_Prophet_model.pkl`, `models/Grid_Prophet_ensemble.pkl`, `data/predictions_2026.csv`, `plots/*.png`
- Modify: `README.md` (predictions table, and the "Based on rounds 1-N" language if the shipped snapshot no longer uses 2 rounds)

**Interfaces:**
- Consumes: all tasks above, via `PYTHONPATH=src python -m src update` (or `run` for a from-scratch rebuild).

- [ ] **Step 1: Run the full test suite one more time before touching data**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: PASS (confirms Tasks 1-6 are solid before regenerating real artifacts)

- [ ] **Step 2: Regenerate the pipeline outputs**

Run:
```bash
PYTHONPATH=src python -m src update
PYTHONPATH=src python -m src plots
```

This rebuilds `data/features.csv` and retrains the model using `features.latest_completed_round(2026)` rounds (the real 2026 season data through the summer break, via FastF1), regenerates `data/predictions_2026.csv`, and regenerates the four PNG charts. Watch the console output for the reported round count, winner model, and selected feature list — note the round count `N` for Step 3.

- [ ] **Step 3: Update `README.md`**

Update the "2026 Predictions (after rounds 1–2)" heading to reflect the actual round count from Step 2 (e.g. "2026 Predictions (after rounds 1–N)"), and replace the table rows with the freshly generated values from `data/predictions_2026.csv`.

- [ ] **Step 4: Commit the regenerated artifacts and README**

```bash
git add data/features.csv data/cv_results.csv data/predictions_2026.csv \
        models/Grid_Prophet_model.pkl models/Grid_Prophet_ensemble.pkl \
        plots/*.png README.md
git commit -m "chore: regenerate model/predictions with expanded window and features"
```

(If `data/*.csv`, `models/*.pkl`, or `plots/*.png` are gitignored in this repo, skip adding those paths — check `.gitignore` first with `git check-ignore -v data/features.csv` before running `git add` on them; only commit `README.md` in that case.)

---

## Self-Review Notes

- **Spec coverage:** §1 (window widening + auto-detect) → Tasks 1, 6. §2 (four new features) → Task 2. §3 (feature selection + ensemble) → Tasks 4, 5 (ensemble already consumes `feature_cols` generically, verified via Task 5 Step 3's `build_ensemble` call using `selected_cols`). Rollout → Task 7.
- **Placeholder scan:** no TBDs; every step has literal code or an exact command.
- **Type/signature consistency:** `_early_season_features(results, early_rounds)` used identically in `features.py` (Task 1/2) and `predict.py` (Task 3). `FEATURE_COLS` (Task 4) is the superset `select_features` (Task 5) narrows from — `load_data` still indexes the full `FEATURE_COLS`, then `main()` narrows `X` afterward, so no mismatch. `retrain_and_save`/`_print_summary`/`build_ensemble` all consistently receive `selected_cols` in Task 5.
