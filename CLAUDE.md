# grid-prophet

F1 forecasting. **v2** is a Bayesian pace + reliability model with a race simulator: `src/model/`, `src/sim/`, `src/cli.py`, `src/backtest.py`. The older season-level regression (`src/train.py`, `predict.py`, `confidence.py`) is kept as a benchmark until Phase 3 passes.

## Where things stand (Nov 10, 2026)

- **Phases 0-2 are done.** [docs/phase2-status.md](docs/phase2-status.md) has the backtest numbers. Spec: [docs/grid-prophet-v2-spec.md](docs/grid-prophet-v2-spec.md), §8 for phases.
- **Phase 3's code is built but not on `main`, and Phase 3 has not passed.** Autopilot window 2 (Oct 20-26) is over. It merged slices 7-10 into `auto/queue` (PRs #20-23): Phase 2 carry-forwards, season chaining with `gp season`, the season-level backtest harness, and post-quali calibration. `auto/queue` has not been merged to `main` yet. Nobody has run the season-level backtest, which is the Phase 3 gate. See [AUTOPILOT.md](AUTOPILOT.md) for what's waiting on Meerav.
- **No routines are scheduled.** The window-2 log is issue #19.
- **Live 2026 forecasts run from the `forecast-data` branch.** R18 (US), R19 (Mexico) and R20 (São Paulo) were forecast and scored there, and its data snapshot now runs through R20. That branch also has three code fixes that exist nowhere else (see below).

## Code that only exists on `forecast-data`

These were committed alongside data and forecast outputs while the live forecasts were being run. Neither `main` nor `auto/queue` has them:

- `src/jolpica_laps.py`: race clean-air pace from jolpica lap times when FastF1 has no laps. F1 live timing now needs an F1TV token, so FastF1 falls back to Ergast results with no laps. `collect_v2` also maps Ergast team names. (`4ea0fb6`)
- `src/event_names.py`: accent-free event-name matching, plus pooling renamed tracks (Brazilian → São Paulo) for the overtaking fit. (`bf80383`)
- `gp race --mode post-quali --grid-file PATH`: forecast from the FIA's penalty-adjusted grid. (`fcce09e`)

**Never merge `forecast-data` into `main`.** Port these by hand onto `auto/queue` or `main`, src and tests only. Cherry-picks conflict: the commits also touch `data/driver_rounds.csv`, and they overlap slice-7's edits to `collect_v2.py` and `cli.py`.

## Commands

```bash
pip install -r requirements.txt pytest
PYTHONPATH=src pytest tests -q                                  # unit suite, ~2 min
PYTHONPATH=src python -m collect_v2 --start-year 2018 --end-year 2026   # builds data/driver_rounds.csv (resumable, FastF1-rate-limited)
PYTHONPATH=src python src/repair_round.py 2026 18 Q             # fetch/replace one session's rows
PYTHONPATH=src python -m cli fit --through 2026:17              # NUTS by default
PYTHONPATH=src python -m cli race --round 18 --mode post-quali --grid-from-quali
PYTHONPATH=src python -m cli backtest --seasons 2022-2025 --rounds all --method nuts --post-quali   # ~2.7 h; owner runs this
```

Only on `auto/queue` until it merges:

```bash
PYTHONPATH=src python -m cli season                             # drivers_championship.csv + constructors_championship.csv
PYTHONPATH=src python -m cli backtest --seasons 2022-2025 --season-level --method nuts --n-trials 5000 --out-dir outputs/season_bt   # Phase 3 gate; owner runs this
```

## Data

`data/driver_rounds.csv` and most of `data/` are **gitignored**. Tests use `data/driver_rounds_fixture.csv`. The `forecast-data` branch carries a snapshot (through 2026 R20; R20 race pace is blank until jolpica publishes laps) for cloud forecasts.

Cloud sessions can't reach FastF1's `livetiming-mirror.fastf1.dev` (the proxy returns 403), so the Oct 24 post-quali routine failed. `api.jolpi.ca` is reachable. In some containers plain `python` is 3.11 while `pip` installs into 3.13 (issue #19, Oct 25).

## Rules

- Never run the full backtest in a routine or cloud session unless Meerav asked for it. It's hours long and it's the owner's gate.
- Don't claim a model change improves forecasts on unit tests alone. Only the backtest says that.
- Design calls the spec doesn't settle go in [docs/autopilot/decisions.md](docs/autopilot/decisions.md) (Question / Chosen / Rejected / Blast radius). Window-2 entries are on `auto/queue`.
- Don't retire `confidence.py` or the pkl until the season-level backtest has passed.
