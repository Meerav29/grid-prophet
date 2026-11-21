# grid-prophet

F1 forecasting. **v2** is a Bayesian pace + reliability model with a race simulator: `src/model/`, `src/sim/`, `src/cli.py`, `src/backtest.py`. The older season-level regression (`src/train.py`, `predict.py`, `confidence.py`) is kept as a benchmark until Phase 3 passes.

## Where things stand (Nov 18, 2026)

- **Phases 0-2 are done.** [docs/phase2-status.md](docs/phase2-status.md) has the backtest numbers. Spec: [docs/grid-prophet-v2-spec.md](docs/grid-prophet-v2-spec.md), §8 for phases.
- **The plan to the 2027 opener is [docs/roadmap.md](docs/roadmap.md)**, serving [docs/vision.md](docs/vision.md) (both locked Nov 14). Roadmap Phase 1 ("one source of truth", to Nov 30) is in progress.
- **Phase 3's code is on `main` but Phase 3 has not passed.** Slices 7-10 (PRs #20-23) reached `main` through PR #24 on Nov 10: Phase 2 carry-forwards, season chaining with `gp season`, the season-level backtest harness, and post-quali calibration. Nobody has run the season-level backtest, which is the Phase 3 gate. See [AUTOPILOT.md](AUTOPILOT.md) for what's waiting on Meerav.
- **Live 2026 forecasts run from the `forecast-data` branch.** R18 (US), R19 (Mexico) and R20 (São Paulo) were forecast and scored there, and its data snapshot runs through R20, race pace included. The latest review of those forecasts is [docs/weekly/2026-11-11-model-review.md](docs/weekly/2026-11-11-model-review.md).
- **The three live-forecast code fixes are now on `main`** (ported by hand, Nov 18): `src/jolpica_laps.py` (race pace from jolpica laps when FastF1 has none, plus Ergast team-name mapping in `collect_v2`), `src/event_names.py` (accent-free event names and renamed-track pooling, including in the single-circuit overtaking fit), and the FIA penalty grid (`gp race --mode post-quali --grid-file PATH`, which landed separately in PR #27; its CSV columns are `abbreviation,grid_position`). **Never merge `forecast-data` into `main`.** Its commits also carry data and outputs.

## Commands

```bash
pip install -r requirements.txt pytest
PYTHONPATH=src pytest tests -q                                  # unit suite, ~2 min
PYTHONPATH=src python -m collect_v2 --start-year 2018 --end-year 2026   # builds data/driver_rounds.csv (resumable, FastF1-rate-limited)
PYTHONPATH=src python src/repair_round.py 2026 18 Q             # fetch/replace one session's rows
PYTHONPATH=src python -m cli fit --through 2026:17              # NUTS by default
PYTHONPATH=src python -m cli race --round 18 --mode post-quali --grid-from-quali
PYTHONPATH=src python -m cli race --round 20 --mode post-quali --grid-file grid_2026_20.csv   # columns abbreviation,grid_position
PYTHONPATH=src python -m cli backtest --seasons 2022-2025 --rounds all --method nuts --post-quali   # ~2.7 h; owner runs this
```

Phase 3:

```bash
PYTHONPATH=src python -m cli season                             # drivers_championship.csv + constructors_championship.csv
PYTHONPATH=src python -m cli backtest --seasons 2022-2025 --season-level --method nuts --n-trials 5000 --out-dir outputs/season_bt   # Phase 3 gate; owner runs this
```

## Data

`data/driver_rounds.csv` and most of `data/` are **gitignored**. Tests use `data/driver_rounds_fixture.csv`. The `forecast-data` branch carries a snapshot (through 2026 R20) for cloud forecasts.

Cloud sessions can't reach FastF1's `livetiming-mirror.fastf1.dev` (the proxy returns 403), so the Oct 24 post-quali routine failed. `api.jolpi.ca` is reachable. In some containers plain `python` is 3.11 while `pip` installs into 3.13 (issue #19, Oct 25).

## Rules

- Never run the full backtest in a routine or cloud session unless Meerav asked for it. It's hours long and it's the owner's gate.
- Don't claim a model change improves forecasts on unit tests alone. Only the backtest says that.
- Design calls the spec doesn't settle go in [docs/autopilot/decisions.md](docs/autopilot/decisions.md) (Question / Chosen / Rejected / Blast radius).
- Don't retire `confidence.py` or the pkl until the season-level backtest has passed.
