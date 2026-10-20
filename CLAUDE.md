# grid-prophet

F1 forecasting. **v2** is a Bayesian pace + reliability model with a race simulator: `src/model/`, `src/sim/`, `src/cli.py`, `src/backtest.py`. The older season-level regression (`src/train.py`, `predict.py`, `confidence.py`) is kept as a benchmark until Phase 3 passes.

## Where things stand (Oct 20, 2026)

- **Phases 0-2 are done.** [docs/phase2-status.md](docs/phase2-status.md) has the backtest numbers and the open items. Phase 3 (season chaining, both championships, legacy retirement) is next. See spec §8 in [docs/grid-prophet-v2-spec.md](docs/grid-prophet-v2-spec.md).
- **Autopilot is running Oct 20-26.** Read [AUTOPILOT.md](AUTOPILOT.md). Slice work lives on `auto/queue`. The queue is [docs/autopilot/queue.md](docs/autopilot/queue.md), the log is issue #19.

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

## Data

`data/driver_rounds.csv` and most of `data/` are **gitignored**. Tests use `data/driver_rounds_fixture.csv`. The `forecast-data` branch carries a snapshot (through 2026 R17) for cloud forecasts. Never merge that branch into `main`.

## Rules

- Never run the full backtest in a routine or cloud session unless Meerav asked for it. It's hours long and it's the owner's gate.
- Don't claim a model change improves forecasts on unit tests alone. Only the backtest says that.
- Design calls the spec doesn't settle go in [docs/autopilot/decisions.md](docs/autopilot/decisions.md) (Question / Chosen / Rejected / Blast radius).
