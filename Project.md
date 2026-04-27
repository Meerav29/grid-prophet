# Grid Prophet — Project Description & Architecture

## Vision

Predict the 2026 F1 Constructor Championship standings using a lightweight ML model trained on historical F1 data. The core insight driving this project: 2026 is a major regulation change year, and teams that dominate the first 2 races of a regulation era tend to carry that advantage through the season. We want a model that understands this pattern.

## Why "Grid Prophet"?

The pit wall is where F1 teams make real-time strategic decisions during races — it's the command center. This project is our pit wall for predicting championship outcomes.

## Core Hypotheses

1. **Early-season performance is highly predictive of final standings**, especially in regulation-change years where the competitive order is established early and convergence is slow.
2. **Some constructors adapt better to rule changes than others.** Teams like Red Bull, Mercedes, and Ferrari have historical patterns of either gaining or losing ground during regulation resets. This is a learnable signal.
3. **A simple model with good features beats a complex model with raw data** for this problem, because we have limited training examples (~10 seasons × ~10 constructors = ~100 rows).

## Data Source

**FastF1** — A Python library that provides access to F1 timing data, session results, and telemetry. We use it for race results (finishing positions, points, grid positions, DNFs) across all sessions from 2014–2025, plus 2026 early-season rounds as they become available.

We do NOT use telemetry or lap-level data. Our unit of analysis is **per-constructor, per-season**.

## Key Design Decisions

### Granularity: Season-level, not race-level
We predict final championship points share per constructor per season. Race-level prediction (predicting each race outcome then summing) would give more training rows but introduces noise from weather, incidents, and strategy variance that isn't relevant to our goal. Season-level keeps the signal clean.

### Target variable: Points share (not raw points or position)
Points systems have changed across eras (e.g., sprint race points). Points share (team's total points / all points awarded that season) normalizes across these changes and gives us a continuous 0–1 target that's directly comparable across years.

### Constructor identity: Stats-based, not categorical
With only ~10 years of data, encoding "Ferrari" as a categorical feature would overfit. Instead, each constructor is described by its computed stats (prior-year performance, rule-change adaptation score, win rate, etc.). This also lets us handle new entrants and rebrands cleanly.

### Rule-change year handling: Dual approach
1. A binary `is_rule_change_year` feature flag.
2. A per-constructor `rule_change_adaptation_score` capturing how that specific team has historically handled regulation changes.
3. Sample weights: rule-change year rows get extra weight during training (exact weight tuned via CV across [1.0, 1.5, 2.0, 2.5, 3.0]).

### Model selection: XGBoost vs Ridge, winner takes all
With ~120 rows, overfitting is a real risk. We train both XGBoost and Ridge regression with leave-one-season-out CV and pick whichever achieves higher Spearman rank correlation. Ridge may win due to the small dataset.

### Ensemble model
An optional ensemble blends the full-data winning model with a rule-change-years-only sub-model (trained exclusively on {2014, 2015, 2022, 2023}). The blend weight α (full-data fraction) is tuned via LOOCV on rule-change seasons {2014, 2022} across candidates [0.3, 0.4, 0.5, 0.6, 0.7]. Enabled via `--ensemble` flag on `train.py` and `predict.py`.

### Confidence intervals: Bootstrap
After prediction, 500 bootstrap iterations resample the training data, retrain, and predict. The 10th and 90th percentiles per constructor give an 80% CI shown in the terminal table and saved to CSV.

### Mid-season updates
`predict.py` accepts `--rounds N` to use N rounds of early-season data instead of the default 2. The `update` command auto-detects the latest completed round from FastF1 and re-predicts, saving a round-stamped snapshot (`predictions_2026_r{N}.csv`).

### No budget tier feature
We considered adding a budget tier feature (top/mid/back) but decided to keep the model purely performance-based. Subjective categorizations add noise and don't capture teams moving between tiers over time.

### Constructor name continuity
F1 teams rebrand frequently. We maintain a lineage mapping:
- **Alpine** ← Renault ← Lotus (2014 era)
- **RB** ← AlphaTauri ← Toro Rosso
- **Aston Martin** ← Racing Point ← Force India
- **Kick Sauber → Audi** ← Alfa Romeo ← Sauber
- **McLaren, Ferrari, Mercedes, Red Bull, Williams, Haas** — stable names

This mapping allows us to compute multi-year features across rebrands. For **major rebrands** (Sauber → Audi), historically-derived features are dampened by 50% in the first season under the new identity.

## Architecture

```
src/
├── collect.py      — Data ingestion from FastF1 → CSV
├── features.py     — Feature engineering → feature matrix CSV
├── train.py        — Model training, CV, evaluation → saved model
├── predict.py      — 2026 prediction using trained model + early-season data
├── confidence.py   — Bootstrap confidence intervals
├── ensemble.py     — Rule-change blend model + alpha tuning
├── visualize.py    — Four chart functions + plot_all()
└── __main__.py     — CLI entry point (python -m src <command>)
```

Data flows linearly: `collect → features → train → predict`. Each script reads from `data/` and writes to `data/` or `models/`. No shared state between scripts except files on disk. This makes it easy to re-run any stage independently.

The CLI (`__main__.py`) wraps all stages and is the recommended entry point. The `Makefile` provides convenient shortcuts.

## Feature Summary

| Feature | Type | Description |
|---|---|---|
| `early_points_share` | Float | Team's share of all points from rounds 1–2 |
| `early_avg_finish` | Float | Avg finishing position in rounds 1–2 |
| `is_rule_change_year` | Binary | 1 if major regulation change year |
| `rule_change_adaptation_score` | Float | Historical avg rank shift in rule-change years |
| `prev_year_points_share` | Float | Points share from previous season |
| `prev_year_standing` | Int | Championship position from previous season |
| `constructor_win_rate_5yr` | Float | Top-3 finish rate over last 5 seasons |
| `avg_driver_career_points_per_race` | Float | Driver quality proxy |

All features are available at prediction time (after 2 rounds of the current season). We intentionally excluded full-season stats (avg finish position, podium rate, DNF rate) to avoid feature mismatch between training and prediction.

## Visualisations

Four charts are generated by `PYTHONPATH=src python -m src plots` (or `make plots`):

| Chart | File | Description |
|---|---|---|
| CV Accuracy | `plots/cv_accuracy.png` | Predicted vs actual rank scatter per held-out season, identity line, Spearman ρ |
| Spearman by Season | `plots/spearman_by_season.png` | Bar chart of per-season CV correlation, rule-change years highlighted |
| 2026 Predictions | `plots/predictions_2026.png` | Horizontal bar chart with 80% CI error bars |
| Feature Importance | `plots/feature_importance.png` | XGBoost importances or Ridge \|coefficients\| |

Charts are also rendered inline in `notebooks/exploration.ipynb`.

## Limitations & Future Work

- **Small dataset**: ~100 training rows. We mitigate with simple models and careful feature engineering, but confidence intervals will be wide.
- **Team personnel changes**: We don't model designer/engineer moves between teams, which historically drive regulation-year shakeups (e.g., Adrian Newey).
- **Budget cap era**: Post-2021 budget caps may reduce the advantage of big spenders in ways not captured by pre-2021 data.
- **Mid-season development**: Some teams develop faster than others through the season. Use `make update` after each race weekend to re-predict with expanded early-season data.
