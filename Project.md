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

**FastF1** — A Python library that provides access to F1 timing data, session results, and telemetry. We use it for race results (finishing positions, points, grid positions, DNFs) across all sessions from 2014–2025.

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

### No budget/resource tier
We considered adding a budget tier feature (top/mid/back) but decided to keep the model purely performance-based. Subjective categorizations add noise and don't capture teams moving between tiers over time.

### Constructor name continuity
F1 teams rebrand frequently. We maintain a lineage mapping:
- **Alpine** ← Renault ← Lotus (2014 era)
- **RB** ← AlphaTauri ← Toro Rosso
- **Aston Martin** ← Racing Point ← Force India
- **Kick Sauber → Audi** ← Alfa Romeo ← Sauber
- **McLaren, Ferrari, Mercedes, Red Bull, Williams, Haas** — stable names

This mapping allows us to compute multi-year features across rebrands.

## Architecture

```
src/
├── collect.py    — Data ingestion from FastF1 → CSV
├── features.py   — Feature engineering → feature matrix CSV
├── train.py      — Model training, CV, evaluation → saved model
└── predict.py    — 2026 prediction using trained model + early-season data
```

Data flows linearly: `collect → features → train → predict`. Each script reads from `data/` and writes to `data/` or `models/`. No shared state between scripts except files on disk. This makes it easy to re-run any stage independently.

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

## Rebranding & Lineage

Constructor identity is tracked across rebrands via a lineage mapping. For **major rebrands** where the team's fundamental identity changes (e.g., Sauber → Audi), all historically-derived features are dampened by 50% in the first season to reflect uncertainty about how much of the old team's patterns carry over. Minor rebrands (e.g., Toro Rosso → AlphaTauri → RB) keep full lineage.

## Limitations & Future Work

- **Small dataset**: ~100 training rows. We mitigate with simple models and careful feature engineering, but confidence intervals will be wide.
- **Team personnel changes**: We don't model designer/engineer moves between teams, which historically drive regulation-year shakeups (e.g., Adrian Newey).
- **Budget cap era**: Post-2021 budget caps may reduce the advantage of big spenders in ways not captured by pre-2021 data.
- **Mid-season development**: Some teams develop faster than others through the season. A future version could re-predict after every race.