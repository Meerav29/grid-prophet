---
name: pit-wall
description: Use when the user wants to run the Grid Prophet F1 constructor championship pipeline in this repo — generate 2026 predictions, retrain the model, refresh mid-season data, build charts, or run a specific stage (collect/features/train/predict/plots/update) instead of the full run.
---

# Pit Wall

Runs the Grid Prophet prediction pipeline in this repo (`collect → features → train → predict`) and reports results. Named for the pit wall — the command center teams use to make real-time calls during a race; this is ours for the model.

## When invoked with no arguments

Run the full pipeline and summarize the result:

1. Check dependencies are installed: `pip show fastf1 >/dev/null 2>&1 || pip install -r requirements.txt`
2. Run `PYTHONPATH=src python -m src run`
3. Read `data/predictions_2026.csv` and present the standings as a table (rank, constructor, predicted share, CI if present)
4. Mention where outputs landed (`data/predictions_2026.csv`, `models/Grid_Prophet_model.pkl`)

## When invoked with arguments

Treat the arguments as a subcommand + flags and forward them directly:

```bash
PYTHONPATH=src python -m src <args>
```

Valid subcommands and flags (see [README.md](../../../README.md) for full detail):

| Subcommand | Purpose |
|---|---|
| `collect` | Pull historical race data from FastF1 |
| `features` | Rebuild the feature matrix |
| `train [--ensemble]` | Train + cross-validate; optionally build ensemble bundle |
| `predict [--rounds N] [--ensemble] [--no-ci]` | Generate 2026 predictions |
| `run` | Full pipeline: collect → features → train → predict |
| `update` | Auto-detect latest completed round, re-predict, save round-stamped snapshot |
| `plots` | Generate all four charts into `plots/` |

Examples:
- `/pit-wall predict --ensemble` → run only the ensemble prediction step
- `/pit-wall predict --rounds 5` → re-predict using 5 rounds of early-season data
- `/pit-wall update` → pull the latest completed round and re-predict
- `/pit-wall plots` → regenerate the four charts

If a subcommand fails because upstream data/artifacts are missing (e.g. `predict` before `train` has ever run), say so and offer to run the missing prior stage(s) rather than guessing.

## After running

Always show the user what happened: the predictions table for `predict`/`run`/`update`, CV Spearman correlation for `train`, or the list of generated files for `plots`. Don't just report exit code 0.
