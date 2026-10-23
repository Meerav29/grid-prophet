"""Grid Prophet v2 CLI (spec sec 6): `gp fit`, `gp race --mode pre`.

Run as: `python -m src.cli fit --through 2026:16` (PYTHONPATH doesn't matter
here since this module resolves its own sibling imports at call time; run
from the repo root).

Implemented: `fit`, `race --mode pre`, `race --mode post-quali` (Phase 2,
spec sec 6 conditioning), sprint weekends (sec 3.4), and `backtest` (the
sec 7 evaluation), and `season` (sec 3.4 chaining, Phase 3).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

DATA_DIR = os.path.join(_THIS_DIR, "..", "data")
OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
DRIVER_ROUNDS_CSV = os.path.join(DATA_DIR, "driver_rounds.csv")
DRIVER_META_CSV = os.path.join(DATA_DIR, "driver_meta.csv")
CIRCUITS_CSV = os.path.join(DATA_DIR, "circuits.csv")
LATEST_POINTER = os.path.join(OUTPUTS_DIR, "latest.txt")


def _parse_through(s: str) -> tuple[int, int]:
    """'2026:16' -> (2026, 16)."""
    season_s, round_s = s.split(":")
    return int(season_s), int(round_s)


def _load_driver_rounds() -> pd.DataFrame:
    return pd.read_csv(DRIVER_ROUNDS_CSV)


def _load_driver_meta() -> pd.DataFrame | None:
    if os.path.exists(DRIVER_META_CSV):
        return pd.read_csv(DRIVER_META_CSV)
    return None


def _load_circuits() -> pd.DataFrame:
    return pd.read_csv(CIRCUITS_CSV)


# ---------------------------------------------------------------------------
# gp fit
# ---------------------------------------------------------------------------

def cmd_fit(args):
    from model.pace import build_pace_data, fit_nuts, fit_advi, posterior_summary
    from model.reliability import build_reliability_data
    from model.challenger import build_challenger_frame, fit_challenger

    season, rnd = _parse_through(args.through)
    driver_rounds = _load_driver_rounds()
    driver_meta = _load_driver_meta()

    print(f"Fitting through {season}:{rnd} using method={args.method} ...")
    t0 = time.time()

    data = build_pace_data(driver_rounds, driver_meta, through_season=season, through_round=rnd)
    print(f"  {len(data.race_y)} race obs, {len(data.quali_y)} quali obs, "
          f"{data.n_teams} teams, {data.n_drivers} drivers, {data.n_rounds} rounds in index")

    if args.method == "advi":
        idata = fit_advi(data, n=args.advi_steps, seed=args.seed, draws=args.draws)
    else:
        idata = fit_nuts(data, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed)

    reliability_data = build_reliability_data(driver_rounds, through_season=season, through_round=rnd)

    challenger_model = None
    if not args.skip_challenger:
        try:
            challenger_frame = build_challenger_frame(driver_rounds, through_season=season, through_round=rnd)
            if len(challenger_frame) >= 20:
                challenger_model = fit_challenger(challenger_frame, seed=args.seed)
        except Exception as exc:  # pragma: no cover - best-effort, backtest can also fit its own
            print(f"  WARNING: challenger fit failed: {exc}")

    elapsed = time.time() - t0
    print(f"Fit complete in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUTPUTS_DIR, ts)
    os.makedirs(outdir, exist_ok=True)

    idata.to_netcdf(os.path.join(outdir, "idata.nc"))
    with open(os.path.join(outdir, "pace_data.pkl"), "wb") as f:
        pickle.dump(data, f)
    with open(os.path.join(outdir, "reliability_data.pkl"), "wb") as f:
        pickle.dump(reliability_data, f)
    if challenger_model is not None:
        with open(os.path.join(outdir, "challenger.pkl"), "wb") as f:
            pickle.dump(challenger_model, f)

    summary = posterior_summary(idata, data)
    summary.to_csv(os.path.join(outdir, "posterior_summary.csv"), index=False)

    meta = {
        "through_season": season, "through_round": rnd, "method": args.method,
        "elapsed_s": elapsed, "n_race_obs": len(data.race_y), "n_quali_obs": len(data.quali_y),
        "n_teams": data.n_teams, "n_drivers": data.n_drivers, "n_rounds": data.n_rounds,
        "has_challenger": challenger_model is not None,
        "fitted_at": ts,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(LATEST_POINTER, "w") as f:
        f.write(outdir)

    print(f"Saved fit -> {outdir}")
    if elapsed > 600:
        print("WARNING: fit exceeded the 10-minute production-fit budget (spec sec 7).")
    return outdir


# ---------------------------------------------------------------------------
# gp race --mode pre
# ---------------------------------------------------------------------------

def load_fit(fit_dir: str):
    import arviz as az

    idata = az.from_netcdf(os.path.join(fit_dir, "idata.nc"))
    with open(os.path.join(fit_dir, "pace_data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(fit_dir, "reliability_data.pkl"), "rb") as f:
        reliability_data = pickle.load(f)
    challenger_model = None
    challenger_path = os.path.join(fit_dir, "challenger.pkl")
    if os.path.exists(challenger_path):
        with open(challenger_path, "rb") as f:
            challenger_model = pickle.load(f)
    return idata, data, reliability_data, challenger_model


def _entrants_for_round(driver_rounds: pd.DataFrame, season: int, round_: int) -> pd.DataFrame:
    """(driver, team) pairs entered for a round, read from the round's own
    rows in driver_rounds.csv (works for a held-out backtest round, where the
    entry list is already known even though the outcome is being predicted;
    a genuinely future round needs an explicit entry list, not yet wired up
    -- phase 1 scope is the backtest, where this is always available)."""
    rows = driver_rounds[(driver_rounds["season"] == season) & (driver_rounds["round"] == round_)
                          & (driver_rounds["session_type"].isin(["Q", "R"]))]
    if rows.empty:
        raise ValueError(f"No entrant data found for {season}:{round_} in driver_rounds.csv")
    entrants = rows[["abbreviation", "team"]].drop_duplicates(subset=["abbreviation"])
    return entrants.dropna()


def _round_has_sprint(driver_rounds: pd.DataFrame, season: int, round_: int) -> bool:
    """Whether the round is a sprint weekend, read from the data rather than
    from a hand-kept list of sprint rounds: `collect_v2` writes `session_type
    == "S"` rows exactly when the FastF1 event format says sprint, so the
    presence of those rows *is* the sprint calendar."""
    rows = driver_rounds[(driver_rounds["season"] == season)
                          & (driver_rounds["round"] == round_)
                          & (driver_rounds["session_type"] == "S")]
    return not rows.empty


def _real_grid_for_round(driver_rounds: pd.DataFrame, season: int, round_: int,
                          driver_ids: list, session_type: str = "R") -> np.ndarray:
    """The round's actual starting grid as 1-indexed positions aligned to
    `driver_ids` (spec sec 3.3 step 3, "post-quali mode uses the real grid").
    `session_type="S"` reads the sprint's own grid instead of the race's --
    from 2023 the two are set by separate sessions and routinely differ.

    Read from `grid_position` on the round's race rows -- FastF1's
    `GridPosition`, which already reflects penalties, so it is the grid the
    race started from rather than quali classification. FastF1 writes 0 for a
    pit-lane start; those and any missing value rank to the back, not to pole.
    Ranks are dense 1..n, so gaps in the grid numbers still come out as a
    clean permutation. Raises if the round has no usable grid at all: silently
    degrading to a simulated one would misreport which mode produced the
    forecast."""
    rows = driver_rounds[(driver_rounds["season"] == season) & (driver_rounds["round"] == round_)
                          & (driver_rounds["session_type"] == session_type)]
    if rows.empty or "grid_position" not in rows.columns:
        raise ValueError(f"No {session_type} rows with a grid for {season}:{round_}; "
                          "--mode post-quali needs a completed quali session.")

    by_driver = dict(zip(rows["abbreviation"], rows["grid_position"]))
    values = pd.Series([by_driver.get(d) for d in driver_ids], dtype="float64")
    values = values.where(values > 0)  # 0 = pit-lane start, NaN = no recorded start
    if values.isna().all():
        raise ValueError(f"No usable {session_type} grid_position values for {season}:{round_}; "
                          "--mode post-quali needs a completed quali session.")
    return values.rank(method="first", na_option="bottom").to_numpy(dtype=int)


def _quali_grid_for_round(driver_rounds: pd.DataFrame, season: int, round_: int,
                           driver_ids: list) -> np.ndarray:
    """Grand Prix starting grid read from the quali classification, for a round
    whose race has not run yet (so no `grid_position` exists to read). Opt-in via
    `gp race --mode post-quali --grid-from-quali`: it does NOT reflect grid
    penalties or pit-lane starts, which only appear on the race rows, and it
    leaves a sprint weekend's sprint grid alone (that still comes from the S
    rows). A driver with no quali classification (NaN position) ranks to the
    back.

    Raises rather than guessing when the classification cannot be trusted: no Q
    rows, an entrant with no Q row at all, a duplicated driver, a tied or
    non-positive position. Any of those would otherwise produce a valid-looking
    1..n grid whose order came from CSV row order."""
    rows = driver_rounds[(driver_rounds["season"] == season) & (driver_rounds["round"] == round_)
                          & (driver_rounds["session_type"] == "Q")]
    where = f"{season}:{round_}"
    if rows.empty:
        raise ValueError(f"No quali classification for {where}; "
                          "--grid-from-quali needs a completed quali session.")
    dupes = sorted(rows.loc[rows["abbreviation"].duplicated(), "abbreviation"].unique())
    if dupes:
        raise ValueError(f"Duplicate quali rows for {dupes} in {where}; fix the data first.")
    missing = [d for d in driver_ids if d not in set(rows["abbreviation"])]
    if missing:
        raise ValueError(f"No quali row for {missing} in {where}; the grid would put them "
                          "at the back in arbitrary order.")
    positions = rows["finish_position"].dropna()
    if (positions <= 0).any() or positions.duplicated().any():
        raise ValueError(f"Quali positions for {where} are tied or non-positive; "
                          "cannot rank a grid from them.")
    by_driver = dict(zip(rows["abbreviation"], rows["finish_position"]))
    values = pd.Series([by_driver[d] for d in driver_ids], dtype="float64")
    if values.isna().all():
        raise ValueError(f"No quali classification for {where}; "
                          "--grid-from-quali needs a completed quali session.")
    return values.rank(method="first", na_option="bottom").to_numpy(dtype=int)


def _round_event_name(driver_rounds: pd.DataFrame, season: int, round_: int):
    """The round's event name, or None when the round has no rows."""
    rows = driver_rounds[(driver_rounds["season"] == season) & (driver_rounds["round"] == round_)]
    return rows.iloc[0]["event_name"] if not rows.empty else None


def _circuit_info(circuits: pd.DataFrame, driver_rounds: pd.DataFrame, season: int, round_: int,
                   overtaking_data=None) -> tuple[str, float]:
    """(circuit_type, overtaking_difficulty) for one round.

    The overtaking parameter is the fitted per-circuit value when
    `overtaking_data` carries one (spec sec 3.3 step 3, slice-4), and the
    `data/circuits.csv` hand rating when it does not -- so a circuit with no
    race history forecasts exactly as it did before the fit existed.
    """
    from model.overtaking import DEFAULT_DIFFICULTY, circuit_difficulty

    rows = driver_rounds[(driver_rounds["season"] == season) & (driver_rounds["round"] == round_)]
    if rows.empty:
        return "mixed", DEFAULT_DIFFICULTY
    event_name = _round_event_name(driver_rounds, season, round_)
    circuit_type = rows.iloc[0]["circuit_type"] if pd.notna(rows.iloc[0]["circuit_type"]) else "mixed"
    match = circuits[circuits["event_name"] == event_name]
    hand_rating = float(match.iloc[0]["overtaking_difficulty"]) if not match.empty else None
    return circuit_type, circuit_difficulty(overtaking_data, event_name, hand_rating)


def _hazard_grid(grid, quali_pace: np.ndarray) -> np.ndarray:
    """The starting slot sec 3.2's incident hazard is conditioned on: post-quali
    the round's real grid, pre-weekend the ranking of the posterior-mean quali
    pace (lower gap = further forward), because the hazard is drawn before the
    resolver simulates a grid. See docs/autopilot/decisions.md.
    """
    if grid is not None:
        return np.asarray(grid, dtype=float)
    mean_pace = np.asarray(quali_pace, dtype=float).mean(axis=0)
    return (np.argsort(np.argsort(mean_pace)) + 1).astype(float)


def forecast_weekend(idata, data, reliability_data, driver_rounds: pd.DataFrame, circuits: pd.DataFrame,
                      season: int, round_: int, n_trials: int = 50_000, seed: int = 0,
                      mode: str = "pre", overtaking_data=None,
                      grid_from_quali: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Forecast one round's events: `(grand_prix, sprint_or_None)`.

    `mode="pre"` simulates quali for the grid, `mode="post-quali"` uses the
    round's real one (spec sec 3.3 step 3); the grid is resolved here, not
    passed in, so it cannot drift out of alignment with the entrant ordering
    built below. On a sprint weekend (sec 3.4) the sprint is resolved from
    the same per-trial pace draw and scored off the sprint points table; the
    second element is None on every other round.
    """
    from model.pace import posterior_pace_draws, posterior_quali_draws
    from model.reliability import (
        combined_dnf_prob, sample_incident_dnf_prob, sample_mechanical_dnf_prob,
    )
    from model.overtaking import build_overtaking_data
    from sim.race import RaceTrialInputs, simulate_weekend, summarize_trials, load_points_table

    rng = np.random.default_rng(seed)
    entrants = _entrants_for_round(driver_rounds, season, round_)
    # Fitted off rounds strictly *before* this one: the round being forecast
    # must never appear in its own overtaking fit, or a backtest scores the
    # model on a parameter that already saw the answer.
    if overtaking_data is None:
        overtaking_data = build_overtaking_data(driver_rounds, circuits,
                                                 before_season=season, before_round=round_,
                                                 event_name=_round_event_name(driver_rounds, season, round_))
    circuit_type, overtaking_difficulty = _circuit_info(circuits, driver_rounds, season, round_,
                                                         overtaking_data)
    round_idx = data.round_to_idx.get((season, round_))  # None for a genuinely future round -> use latest

    driver_ids = entrants["abbreviation"].tolist()
    teams = entrants["team"].tolist()
    n_drivers = len(driver_ids)
    grid = None
    grid_source = "simulated"
    if mode == "post-quali":
        if grid_from_quali:
            grid_source = "quali_classification"
            grid = _quali_grid_for_round(driver_rounds, season, round_, driver_ids)
            print(f"  WARNING: grid read from the quali classification for {season}:{round_}; "
                  "grid penalties and pit-lane starts are not reflected.", file=sys.stderr)
        else:
            grid_source = "race_grid_position"
            grid = _real_grid_for_round(driver_rounds, season, round_, driver_ids)

    has_sprint = _round_has_sprint(driver_rounds, season, round_)
    # Post-quali, the sprint has a real starting order too -- its own, from
    # sprint quali. A sprint round whose S rows carry no usable grid falls
    # back to simulating it from the shared quali pace rather than failing
    # the whole weekend: the Grand Prix forecast does not depend on it.
    sprint_grid = None
    if has_sprint and mode == "post-quali":
        try:
            sprint_grid = _real_grid_for_round(driver_rounds, season, round_, driver_ids, session_type="S")
        except ValueError as exc:
            print(f"  WARNING: no real sprint grid for {season}:{round_} ({exc}); simulating it.",
                  file=sys.stderr)

    n_posterior = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    draw_idx = rng.integers(0, n_posterior, size=n_trials)

    race_pace = np.zeros((n_trials, n_drivers))
    quali_pace = np.zeros((n_trials, n_drivers))
    mech_prob = np.zeros((n_trials, n_drivers))
    incident_prob = np.zeros((n_trials, n_drivers))

    race_nu_all = np.asarray(idata.posterior["race_nu"]).reshape(-1)[draw_idx]
    race_sigma_all = np.asarray(idata.posterior["race_sigma"]).reshape(-1)[draw_idx]
    quali_sigma_all = np.asarray(idata.posterior["quali_sigma"]).reshape(-1)[draw_idx]

    # one mechanical-DNF draw per team per trial, shared across its two cars
    # (model.reliability.sample_mechanical_dnf_prob's documented contract) --
    # cached per team so teammates don't each get an independent draw.
    # `sorted`, not bare `set`: set iteration order over strings varies with
    # PYTHONHASHSEED, so an unsorted loop draws from `rng` in a different
    # order in every process and --seed stops reproducing a forecast.
    team_dnf_prob = {}
    for team in sorted(set(teams)):
        team_dnf_prob[team] = sample_mechanical_dnf_prob(reliability_data, team, season, n_trials, rng)

    for i, (driver, team) in enumerate(zip(driver_ids, teams)):
        rp = posterior_pace_draws(idata, data, team, driver, circuit_type, round_idx=round_idx)
        qp = posterior_quali_draws(idata, data, team, driver, round_idx=round_idx)
        race_pace[:, i] = rp[draw_idx]
        quali_pace[:, i] = qp[draw_idx]
        mech_prob[:, i] = team_dnf_prob[team]

    hazard_grid = _hazard_grid(grid, quali_pace)
    for i, driver in enumerate(driver_ids):
        incident_prob[:, i] = sample_incident_dnf_prob(reliability_data, driver,
                                                       hazard_grid[i], n_trials, rng)

    inputs = RaceTrialInputs(
        driver_ids=driver_ids, race_pace=race_pace,
        race_noise_nu=race_nu_all, race_noise_sigma=race_sigma_all,
        quali_pace=quali_pace, quali_noise_sigma=quali_sigma_all,
        dnf_prob=combined_dnf_prob(mech_prob, incident_prob),
        # Both channels, not just the fold: on a sprint round they do not
        # shorten alike (sec 3.2's first-lap spike -- see
        # `sim.race.sprint_inputs_from`). The resolver reads `dnf_prob`, so
        # carrying them changes no Grand Prix forecast.
        mechanical_dnf_prob=mech_prob,
        incident_dnf_prob=incident_prob,
        overtaking_difficulty=overtaking_difficulty,
        grid=grid,
    )
    outcome = simulate_weekend(inputs, rng, has_sprint=has_sprint, sprint_grid=sprint_grid)
    grand_prix = summarize_trials(outcome.positions, outcome.dnf, driver_ids, teams,
                                   load_points_table("grand_prix", season))
    # which grid produced the forecast, so a saved CSV can't be mistaken for one
    # built from the real (penalty-adjusted) starting order
    grand_prix["grid_source"] = grid_source
    if not outcome.has_sprint:
        return grand_prix, None
    sprint = summarize_trials(outcome.sprint_positions, outcome.sprint_dnf, driver_ids, teams,
                               load_points_table("sprint", season))
    sprint["grid_source"] = "sprint_grid_position" if sprint_grid is not None else "simulated"
    return grand_prix, sprint


def forecast_race(*args, **kwargs) -> pd.DataFrame:
    """The Grand Prix forecast alone -- `forecast_weekend`'s first element.
    Kept because sec 6's `race_forecast.csv` is the Grand Prix table and
    nothing about a sprint round changes its shape."""
    return forecast_weekend(*args, **kwargs)[0]


def cmd_race(args):
    if args.grid_from_quali and args.mode != "post-quali":
        print("--grid-from-quali only applies to --mode post-quali; refusing to run a "
              "pre-weekend forecast that would ignore it.", file=sys.stderr)
        sys.exit(2)
    fit_dir = args.fit_dir
    if fit_dir is None:
        if not os.path.exists(LATEST_POINTER):
            print("No fit found. Run `gp fit --through ...` first, or pass --fit-dir.", file=sys.stderr)
            sys.exit(1)
        fit_dir = open(LATEST_POINTER).read().strip()

    idata, data, reliability_data, _ = load_fit(fit_dir)
    driver_rounds = _load_driver_rounds()
    circuits = _load_circuits()

    out_dir = args.out_dir or fit_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "post-quali":
        from model.conditioning import append_record, condition_on_quali

        print(f"Conditioning on {args.season}:{args.round} quali (warm-started NUTS refit) ...")
        idata, data, record = condition_on_quali(
            driver_rounds, _load_driver_meta(), args.season, args.round, idata,
            chains=args.chains, seed=args.seed,
        )
        # the per-round fallback record sec 6 requires: appended, not overwritten
        log_path = append_record(os.path.join(out_dir, "conditioning_log.csv"), record)
        print(f"  path={record.path} warm_start={record.warm_start_s:.1f}s "
              f"total={record.total_s:.1f}s target={record.applied_target_s:.0f}s -> {log_path}")
        if record.regression:
            print("  REGRESSION: missed the target that applies to this round. Spec sec 6: "
                  "a late-season miss is a real regression, not an accepted exception.",
                  file=sys.stderr)

    forecast, sprint = forecast_weekend(idata, data, reliability_data, driver_rounds, circuits,
                                         args.season, args.round, n_trials=args.n_trials,
                                         seed=args.seed, mode=args.mode,
                                         grid_from_quali=args.grid_from_quali)

    out_path = os.path.join(out_dir, f"race_forecast_{args.season}_{args.round}.csv")
    forecast.to_csv(out_path, index=False)
    print(forecast.to_string(index=False))
    print(f"Saved -> {out_path}")

    # A sprint is a second event, not extra columns on the first: sec 6 fixes
    # race_forecast.csv's columns as the Grand Prix's, and a reader summing
    # exp_points over a weekend should have to say which event it means.
    if sprint is not None:
        sprint_path = os.path.join(out_dir, f"sprint_forecast_{args.season}_{args.round}.csv")
        sprint.to_csv(sprint_path, index=False)
        print(f"\nSprint (spec sec 3.4, shares the weekend pace draw):")
        print(sprint.to_string(index=False))
        print(f"Saved -> {sprint_path}")


# ---------------------------------------------------------------------------
# gp season
# ---------------------------------------------------------------------------

def _season_so_far(driver_rounds: pd.DataFrame, season: int, through_round: int):
    """Points and Grand Prix wins already scored in `season` up to and
    including `through_round`, as ({driver: pts}, {team: pts}, {driver: wins},
    {team: wins}). Read from the recorded `points` of R and S rows, so team
    totals follow whoever actually drove the car that weekend."""
    run = driver_rounds[(driver_rounds["season"] == season) & (driver_rounds["round"] <= through_round)
                         & (driver_rounds["session_type"].isin(["R", "S"]))]
    pts = run.assign(points=run["points"].fillna(0.0))
    won = run[(run["session_type"] == "R") & (run["finish_position"] == 1)]
    return (pts.groupby("abbreviation")["points"].sum().to_dict(),
            pts.groupby("team")["points"].sum().to_dict(),
            won.groupby("abbreviation").size().to_dict(),
            won.groupby("team").size().to_dict())


def forecast_season(idata, data, reliability_data, driver_rounds: pd.DataFrame, circuits: pd.DataFrame,
                     n_trials: int = 10_000, seed: int = 0, schedule_loader=None):
    """Chain the rest of the fitted season (spec sec 3.4).
    Returns `(drivers_championship, constructors_championship)`.

    The season and the last completed round are the fit's own cutoff, so the
    chain starts exactly where the posterior ends. Entrants for the remaining
    rounds are the last completed round's line-up."""
    from model.overtaking import build_overtaking_data, circuit_difficulty
    from model.pace import CIRCUIT_TYPES
    from model.reliability import sample_incident_dnf_prob, sample_mechanical_dnf_prob
    from sim.race import load_points_table
    from sim.season import (
        SeasonRound, SeasonTrialInputs, constructors_championship, drivers_championship,
        remaining_calendar, simulate_season, step_sigma_for_round,
    )

    season, through = max(data.round_to_idx, key=data.round_to_idx.get)
    calendar = remaining_calendar(season, through, loader=schedule_loader)
    rng = np.random.default_rng(seed)

    entrants = _entrants_for_round(driver_rounds, season, through)
    driver_ids, teams = entrants["abbreviation"].tolist(), entrants["team"].tolist()
    drv_pts, team_pts, drv_wins, team_wins = _season_so_far(driver_rounds, season, through)
    team_names = sorted(set(teams) | set(team_pts))

    post = idata.posterior
    n_posterior = post.sizes["chain"] * post.sizes["draw"]
    draw_idx = rng.integers(0, n_posterior, size=n_trials)

    def flat(name):
        arr = np.asarray(post[name])
        return arr.reshape(n_posterior, *arr.shape[2:])[draw_idx]

    car_all = np.asarray(post["car"].isel(round=data.round_to_idx[(season, through)])
                         .transpose("chain", "draw", "team")).reshape(n_posterior, -1)[draw_idx]
    car0_mean = flat("car0").mean(axis=1)
    fitted_teams = list(post["car"].coords["team"].values)
    car = np.column_stack([car_all[:, fitted_teams.index(t)] if t in fitted_teams else car0_mean
                           for t in team_names])

    fitted_drivers = list(post["driver_skill"].coords["driver"].values)
    skill_all, offset_all, track_all = flat("driver_skill"), flat("quali_offset"), flat("driver_track")
    n_drivers = len(driver_ids)
    skill, offset = np.zeros((n_trials, n_drivers)), np.zeros((n_trials, n_drivers))
    track = np.zeros((n_trials, n_drivers, len(CIRCUIT_TYPES)))
    for i, d in enumerate(driver_ids):
        if d in fitted_drivers:  # an unseen driver keeps the field-average zeros
            j = fitted_drivers.index(d)
            skill[:, i], offset[:, i], track[:, i] = skill_all[:, j], offset_all[:, j], track_all[:, j]

    # Hazards are posterior parameters: one draw per trial, held for the season.
    team_mech = {t: sample_mechanical_dnf_prob(reliability_data, t, season, n_trials, rng)
                 for t in sorted(set(teams))}
    mech = np.column_stack([team_mech[t] for t in teams])
    quali_pace = car[:, [team_names.index(t) for t in teams]] + skill + offset
    hazard_grid = _hazard_grid(None, quali_pace)
    incident = np.column_stack([sample_incident_dnf_prob(reliability_data, d, hazard_grid[i], n_trials, rng)
                                for i, d in enumerate(driver_ids)])

    overtaking = build_overtaking_data(driver_rounds, circuits, before_season=season, before_round=through + 1)
    rounds = []
    for row in calendar.itertuples(index=False):
        match = circuits[circuits["event_name"] == row.event_name]
        ctype = match.iloc[0]["circuit_type"] if not match.empty else "mixed"
        hand = float(match.iloc[0]["overtaking_difficulty"]) if not match.empty else None
        rounds.append(SeasonRound(
            round=int(row.round), has_sprint=bool(row.has_sprint),
            circuit_type_idx=CIRCUIT_TYPES.index(ctype) if ctype in CIRCUIT_TYPES else 0,
            overtaking_difficulty=circuit_difficulty(overtaking, row.event_name, hand),
            step_sigma=step_sigma_for_round(season, int(row.round))))

    inputs = SeasonTrialInputs(
        driver_ids=driver_ids, teams=teams, team_names=team_names, car=car,
        driver_skill=skill, driver_track=track, quali_offset=offset,
        race_noise_nu=flat("race_nu"), race_noise_sigma=flat("race_sigma"),
        quali_noise_sigma=flat("quali_sigma"),
        mechanical_dnf_prob=mech, incident_dnf_prob=incident)
    outcome = simulate_season(inputs, rounds, load_points_table("grand_prix", season),
                              load_points_table("sprint", season), rng)

    # Drivers who scored this season but are not on the current line-up keep
    # their points and score nothing more.
    all_drivers = driver_ids + sorted(set(drv_pts) - set(driver_ids))
    last_team = (driver_rounds[driver_rounds["season"] == season]
                 .sort_values("round").groupby("abbreviation")["team"].last())
    all_teams = teams + [last_team[d] for d in all_drivers[n_drivers:]]
    future = np.zeros((n_trials, len(all_drivers)))
    future_wins = np.zeros_like(future)
    future[:, :n_drivers], future_wins[:, :n_drivers] = outcome.driver_points, outcome.driver_wins
    cur_d = np.array([drv_pts.get(d, 0.0) for d in all_drivers])
    cur_dw = np.array([drv_wins.get(d, 0) for d in all_drivers])
    cur_t = np.array([team_pts.get(t, 0.0) for t in team_names])
    cur_tw = np.array([team_wins.get(t, 0) for t in team_names])
    drivers = drivers_championship(cur_d + future, cur_dw + future_wins, all_drivers, all_teams, cur_d)
    constructors = constructors_championship(cur_t + outcome.team_points, cur_tw + outcome.team_wins,
                                             team_names, cur_t)
    return drivers, constructors


def cmd_season(args):
    from sim.season import ScheduleUnavailable

    fit_dir = args.fit_dir
    if fit_dir is None:
        if not os.path.exists(LATEST_POINTER):
            print("No fit found. Run `gp fit --through ...` first, or pass --fit-dir.", file=sys.stderr)
            sys.exit(1)
        fit_dir = open(LATEST_POINTER).read().strip()
    idata, data, reliability_data, _ = load_fit(fit_dir)
    try:
        drivers, constructors = forecast_season(idata, data, reliability_data, _load_driver_rounds(),
                                                _load_circuits(), n_trials=args.n_trials, seed=args.seed)
    except ScheduleUnavailable as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or fit_dir
    os.makedirs(out_dir, exist_ok=True)
    for name, table in (("drivers_championship.csv", drivers),
                        ("constructors_championship.csv", constructors)):
        path = os.path.join(out_dir, name)
        table.to_csv(path, index=False)
        print(table.to_string(index=False))
        print(f"Saved -> {path}\n")


# ---------------------------------------------------------------------------
# gp backtest
# ---------------------------------------------------------------------------

def cmd_backtest(args):
    from backtest import run_backtest

    seasons = [int(s) for s in args.seasons.split("-")] if "-" in args.seasons else [int(args.seasons)]
    if len(seasons) == 2:
        seasons = list(range(seasons[0], seasons[1] + 1))

    if args.season_level:
        from backtest import run_season_backtest

        run_season_backtest(seasons=seasons, cutoffs=[int(c) for c in args.cutoffs.split(",")],
                            n_trials=args.n_trials, method=args.method,
                            advi_steps=args.advi_steps, out_dir=args.out_dir)
        return

    run_backtest(
        seasons=seasons, n_trials=args.n_trials, method=args.method,
        rounds=args.rounds, advi_steps=args.advi_steps, out_dir=args.out_dir,
        post_quali=args.post_quali, post_quali_chains=args.post_quali_chains,
    )


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gp", description="Grid Prophet v2 CLI")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p_fit = sub.add_parser("fit", help="Fit the pace/reliability models through a given round")
    p_fit.add_argument("--through", required=True, help="e.g. 2026:16")
    p_fit.add_argument("--method", choices=["nuts", "advi"], default="nuts")
    p_fit.add_argument("--draws", type=int, default=400)
    p_fit.add_argument("--tune", type=int, default=400)
    p_fit.add_argument("--chains", type=int, default=2)
    p_fit.add_argument("--advi-steps", type=int, default=8000)
    p_fit.add_argument("--seed", type=int, default=0)
    p_fit.add_argument("--skip-challenger", action="store_true")
    p_fit.set_defaults(func=cmd_fit)

    p_race = sub.add_parser("race", help="Next-race forecast")
    p_race.add_argument("--round", type=int, required=True)
    p_race.add_argument("--season", type=int, default=2026)
    p_race.add_argument("--mode", choices=["pre", "post-quali"], default="pre")
    p_race.add_argument("--fit-dir", default=None)
    p_race.add_argument("--out-dir", default=None)
    p_race.add_argument("--n-trials", type=int, default=50_000)
    p_race.add_argument("--chains", type=int, default=2, help="chains for the post-quali refit")
    p_race.add_argument("--seed", type=int, default=0)
    p_race.add_argument("--grid-from-quali", action="store_true",
                        help="with --mode post-quali, before the race has run: take the Grand Prix "
                             "grid from the quali classification (ignores grid penalties and "
                             "pit-lane starts; a sprint's own grid is unaffected)")
    p_race.set_defaults(func=cmd_race)

    p_season = sub.add_parser("season", help="Chain the rest of the fitted season: both championships")
    p_season.add_argument("--fit-dir", default=None)
    p_season.add_argument("--out-dir", default=None)
    p_season.add_argument("--n-trials", type=int, default=10_000)
    p_season.add_argument("--seed", type=int, default=0)
    p_season.set_defaults(func=cmd_season)

    p_bt = sub.add_parser("backtest", help="Rolling backtest (spec sec 7)")
    p_bt.add_argument("--seasons", default="2022-2025")
    p_bt.add_argument("--rounds", default="coarse", choices=["coarse", "all"])
    p_bt.add_argument("--method", choices=["nuts", "advi"], default="advi")
    p_bt.add_argument("--n-trials", type=int, default=5000)
    p_bt.add_argument("--advi-steps", type=int, default=4000)
    p_bt.add_argument("--out-dir", default=None)
    p_bt.add_argument("--post-quali", action="store_true",
                      help="Also condition on each round's quali and score the post-quali forecast "
                           "against pre-weekend (Phase 2 gate); writes conditioning_log.csv")
    p_bt.add_argument("--post-quali-chains", type=int, default=2,
                      help="chains for each round's post-quali conditioning refit")
    p_bt.add_argument("--season-level", action="store_true",
                      help="Season-level backtest instead (spec sec 7, Phase 3): fit at each "
                           "--cutoffs round, chain the rest of the season, score both championships")
    p_bt.add_argument("--cutoffs", default="2,6,12",
                      help="with --season-level: comma-separated rounds to fit through")
    p_bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
