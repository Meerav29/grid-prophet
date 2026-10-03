"""Grid Prophet v2 CLI (spec sec 6): `gp fit`, `gp race --mode pre`.

Run as: `python -m src.cli fit --through 2026:16` (PYTHONPATH doesn't matter
here since this module resolves its own sibling imports at call time; run
from the repo root).

Implemented: `fit`, `race --mode pre`, `race --mode post-quali` (Phase 2,
spec sec 6 conditioning), sprint weekends (sec 3.4), and `backtest` (the
sec 7 evaluation). `season` is Phase 3.
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
    event_name = rows.iloc[0]["event_name"]
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
                      mode: str = "pre", overtaking_data=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
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
                                                 before_season=season, before_round=round_)
    circuit_type, overtaking_difficulty = _circuit_info(circuits, driver_rounds, season, round_,
                                                         overtaking_data)
    round_idx = data.round_to_idx.get((season, round_))  # None for a genuinely future round -> use latest

    driver_ids = entrants["abbreviation"].tolist()
    teams = entrants["team"].tolist()
    n_drivers = len(driver_ids)
    grid = _real_grid_for_round(driver_rounds, season, round_, driver_ids) if mode == "post-quali" else None

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
    if not outcome.has_sprint:
        return grand_prix, None
    sprint = summarize_trials(outcome.sprint_positions, outcome.sprint_dnf, driver_ids, teams,
                               load_points_table("sprint", season))
    return grand_prix, sprint


def forecast_race(*args, **kwargs) -> pd.DataFrame:
    """The Grand Prix forecast alone -- `forecast_weekend`'s first element.
    Kept because sec 6's `race_forecast.csv` is the Grand Prix table and
    nothing about a sprint round changes its shape."""
    return forecast_weekend(*args, **kwargs)[0]


def cmd_race(args):
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
                                         seed=args.seed, mode=args.mode)

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
# gp backtest
# ---------------------------------------------------------------------------

def cmd_backtest(args):
    from backtest import run_backtest

    seasons = [int(s) for s in args.seasons.split("-")] if "-" in args.seasons else [int(args.seasons)]
    if len(seasons) == 2:
        seasons = list(range(seasons[0], seasons[1] + 1))

    run_backtest(
        seasons=seasons, n_trials=args.n_trials, method=args.method,
        rounds=args.rounds, advi_steps=args.advi_steps, out_dir=args.out_dir,
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
    p_race.set_defaults(func=cmd_race)

    p_bt = sub.add_parser("backtest", help="Rolling backtest (spec sec 7)")
    p_bt.add_argument("--seasons", default="2022-2025")
    p_bt.add_argument("--rounds", default="coarse", choices=["coarse", "all"])
    p_bt.add_argument("--method", choices=["nuts", "advi"], default="advi")
    p_bt.add_argument("--n-trials", type=int, default=5000)
    p_bt.add_argument("--advi-steps", type=int, default=4000)
    p_bt.add_argument("--out-dir", default=None)
    p_bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
