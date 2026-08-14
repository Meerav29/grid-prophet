# Grid Prophet

ML model trained on ~10 years of F1 data (via FastF1) to predict the 2026 constructor championship standings. Emphasizes early-season performance signals — auto-widened to cover however much of the season has actually run — and a rule-change-year adaptation variable to capture how teams historically perform during major regulation shifts like 2026.

## Quick Start

Using [Claude Code](https://claude.com/claude-code)? Clone the repo and run `/pit-wall` — it installs dependencies and runs the full pipeline for you. Pass args to run a specific stage, e.g. `/pit-wall predict --ensemble` or `/pit-wall update`. See [.claude/skills/pit-wall/SKILL.md](.claude/skills/pit-wall/SKILL.md).

Otherwise, run it manually:

```bash
pip install -r requirements.txt

# Run the full pipeline in one command
PYTHONPATH=src python -m src run

# Or step by step
PYTHONPATH=src python -m src collect       # Pull historical data from FastF1
PYTHONPATH=src python -m src features      # Build feature matrix
PYTHONPATH=src python -m src train         # Train model + cross-validate
PYTHONPATH=src python -m src predict       # Generate 2026 predictions (with bootstrap CI)
PYTHONPATH=src python -m src plots         # Generate visualisation charts

# Or via Makefile
make run
make plots
```

## 2026 Predictions (after rounds 1–11)

| Rank | Constructor | Predicted Share | 80% CI |
|------|-------------|----------------|--------|
| 1 | Mercedes | 26.2% | ± 2.3% |
| 2 | Ferrari | 23.6% | ± 1.2% |
| 3 | McLaren | 15.0% | ± 1.2% |
| 4 | Red Bull Racing | 15.0% | ± 1.1% |
| 5 | Alpine | 4.9% | ± 0.5% |
| 6 | Racing Bulls | 4.7% | ± 0.5% |
| 7 | Haas F1 Team | 1.7% | ± 0.3% |
| 8 | Audi | 1.5% | ± 0.3% |
| 9 | Williams | 1.4% | ± 0.3% |
| 10 | Cadillac | 0.2% | ± 0.2% |
| 11 | Aston Martin | 0.1% | ± 0.1% |

## CLI Reference

```bash
PYTHONPATH=src python -m src <command> [flags]
```

| Command | Description |
|---------|-------------|
| `collect` | Pull historical race results from FastF1 |
| `features` | Build per-constructor, per-season feature matrix |
| `train` | Train + cross-validate model; optionally build ensemble |
| `predict` | Generate 2026 predictions with bootstrap confidence intervals |
| `run` | Run full pipeline: collect → features → train → predict |
| `update` | Auto-detect latest completed round and re-predict |
| `plots` | Generate all four visualisation charts |

### Key flags

```bash
# predict.py
--rounds N        # Ad-hoc override: use N rounds of early-season data (default: EARLY_ROUNDS=2)
--ensemble        # Use the ensemble model instead of base model
--no-ci           # Skip bootstrap confidence intervals (faster)

# features.py
--early-rounds N  # Override the early-season round window (default: EARLY_ROUNDS=2)

# train.py
--ensemble        # Build ensemble bundle after training base model
--ensemble-out    # Path for ensemble pkl (default: models/Grid_Prophet_ensemble.pkl)
```

### Mid-season updates

```bash
make update
# or
PYTHONPATH=src python -m src update
```

Auto-detects the latest completed 2026 round from FastF1, rebuilds `features.csv` using that many rounds as the early-season window for **both** historical training seasons and the live 2026 row (so training and prediction never see a mismatched window), retrains the model, and re-predicts. Saves a round-stamped snapshot (`data/predictions_2026_r{N}.csv`) alongside the current `predictions_2026.csv`. `run` does the same after a fresh `collect`.

## How It Works

The model uses per-constructor, per-season features — including early-season dominance signals (points share, average finish, DNF rate, grid-to-finish delta, teammate head-to-head, in-window development trend), historical rule-change adaptation scores, prior-year momentum, and driver quality proxies — to predict each team's share of total championship points. Rule-change years are weighted more heavily during training to better capture the dynamics of regulation shifts. Before comparing Ridge and XGBoost, `train.py` runs LassoCV feature selection to drop features that don't earn their keep against the ~120-row dataset.

An optional ensemble model blends the full-data model with a rule-change-years-only sub-model (trained on 2014, 2015, 2022, 2023). The blend weight α is tuned via leave-one-season-out CV on rule-change seasons.

Bootstrap confidence intervals (N=500) are computed after prediction and shown in the terminal table alongside each constructor's predicted share.

## Outputs

| File | Description |
|------|-------------|
| `data/race_results.csv` | Raw race results 2014–2026 |
| `data/constructor_standings.csv` | Historical championship standings |
| `data/features.csv` | Per-constructor, per-season feature matrix |
| `data/cv_results.csv` | Leave-one-season-out CV results |
| `data/predictions_2026.csv` | 2026 predictions with CI columns |
| `models/Grid_Prophet_model.pkl` | Trained base model bundle |
| `models/Grid_Prophet_ensemble.pkl` | Ensemble model bundle (if built) |
| `plots/cv_accuracy.png` | Predicted vs actual rank per CV season |
| `plots/spearman_by_season.png` | Spearman ρ by season, RC years highlighted |
| `plots/predictions_2026.png` | 2026 standings chart with 80% CI error bars |
| `plots/feature_importance.png` | Feature importances / Ridge coefficients |

## Makefile Targets

```
make collect    make features    make train    make predict
make run        make update      make plots    make clean
```

`make clean` removes derived data files and model pkls. Raw collected data (`race_results.csv`, `constructor_standings.csv`) is preserved.

See [PLANNING.md](PLANNING.md) for the staged implementation plan and [PROJECT.md](Project.md) for detailed project context and design decisions.
