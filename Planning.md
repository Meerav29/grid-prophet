# Grid Prophet — Implementation Plan

This document breaks the project into sequential stages. Each stage is a self-contained unit of work. To implement, tell Claude Code: **"Implement Stage X of PLANNING.md"** — it has everything it needs.

---

## Locked Decisions

These decisions are final. Do not deviate from them during implementation.

- **Training range:** 2014–2025 (turbo-hybrid era only, includes rule-change years 2014 and 2022).
- **Feature discipline:** Only train on features that will be available at 2026 prediction time. No full-season stats (avg finish, podium rate, DNF rate) in the model — these are leaky. The model sees only early-season signals, historical/momentum features, and rule-change variables.
- **Constructor rebrands:** Maintain lineage across rebrands but **dampen historical features by 50%** for major identity changes (Sauber → Audi). Minor rebrands (Toro Rosso → AlphaTauri → RB) keep full lineage. Define a `MAJOR_REBRAND` list in `features.py`.
- **No budget tier feature:** Keep the model purely performance-based. No subjective resource categorization.
- **Model comparison:** Train both XGBoost and Ridge regression, compare via leave-one-season-out CV using Spearman rank correlation. Use whichever scores better.
- **Rule-change weight:** Tune via CV across [1.0, 1.5, 2.0, 2.5, 3.0]. Pick the weight that maximizes average Spearman correlation.
- **Primary metric:** Spearman rank correlation (predicted vs actual constructor standings). Secondary: MAE on points share.
- **Early-season window:** Hardcoded to rounds 1–2 for initial build. Make it a constant that's easy to change later.

---

## Stage 0 — Project Scaffold

**Goal:** Initialize the repo structure with empty modules, dependencies, and config.

Create a Python project called `Grid Prophet` with this structure:

```
Grid Prophet/
├── README.md              (copy from README.md in project docs)
├── PLANNING.md            (this file)
├── PROJECT.md             (copy from PROJECT.md in project docs)
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── collect.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── notebooks/
    └── exploration.ipynb  (empty jupyter notebook shell)
```

`requirements.txt`: `fastf1`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `jupyter`, `matplotlib`, `seaborn`

`.gitignore`: Standard Python gitignore plus `data/*.csv`, `data/*.parquet`, `models/*.pkl`, `.ipynb_checkpoints/`

Each `.py` file should have only a module-level docstring (no implementation). Initialize git repo and make first commit.

---

## Stage 1 — Data Collection (`collect.py`)

**Goal:** Pull historical F1 race results from FastF1 and save them as a clean CSV.

Implement `collect.py` to do the following:

1. Accept a year range via CLI args (default: 2014–2025).
2. For each year, use `fastf1.get_event_schedule(year)` to get the race calendar, then for each round load the Race session via `fastf1.get_session(year, round, 'R')` and extract from `session.results`: year, round number, driver name, driver abbreviation, constructor/team name, grid position, finishing position, classification status (finished/DNF/DNS), points scored.
3. Concatenate everything into a single DataFrame and save to `data/race_results.csv`.
4. Also pull final constructor championship standings per year (either derive from summing points or pull from ergast/FastF1 if available) and save to `data/constructor_standings.csv`.
5. Add basic logging so progress is visible (e.g., "Collecting 2014 Round 1...").
6. Handle errors gracefully — if a session fails to load, log it and skip.

**Expected outputs:** `data/race_results.csv` and `data/constructor_standings.csv`

---

## Stage 2 — Feature Engineering (`features.py`)

**Goal:** Transform raw race results into a per-constructor, per-season feature matrix. Only include features that will be available at 2026 prediction time (no leaky full-season stats).

Implement `features.py` to read from `data/race_results.csv` and `data/constructor_standings.csv` and produce `data/features.csv` with one row per constructor per season.

**Constants to define at the top of the file:**
- `EARLY_ROUNDS = 2` — number of early-season rounds to use as signal.
- `RULE_CHANGE_YEARS = [2009, 2014, 2017, 2022, 2026]`
- `MAJOR_REBRANDS = {"Audi": "Sauber"}` — mapping of new name → old lineage where historical features should be dampened by 50%.
- `CONSTRUCTOR_LINEAGE = {...}` — full mapping of current names to historical names (Alpine ← Renault, RB ← AlphaTauri ← Toro Rosso, Aston Martin ← Racing Point ← Force India, Kick Sauber/Audi ← Alfa Romeo ← Sauber).

**Features to compute (all prediction-safe):**

1. **Early-season signal (rounds 1–2):**
   - `early_points_share`: Team's points from rounds 1–EARLY_ROUNDS as a fraction of total points awarded in those rounds.
   - `early_avg_finish`: Average finishing position of the team's drivers in rounds 1–EARLY_ROUNDS.

2. **Rule-change variables:**
   - `is_rule_change_year`: Binary flag.
   - `rule_change_adaptation_score`: For constructors with history in prior rule-change years, the average championship rank shift (current year rank minus prior year rank, so negative = improved). Set to 0 for teams with no rule-change history.

3. **Momentum / historical strength:**
   - `prev_year_points_share`: The team's points share from the previous season.
   - `prev_year_standing`: Championship finishing position from the previous season.
   - `constructor_win_rate_5yr`: Fraction of the last 5 seasons where the team finished in the top 3 of the constructors championship.

4. **Driver quality proxy:**
   - `avg_driver_career_points_per_race`: Average career points-per-race for the team's current-season drivers (computed from all prior data in the dataset, NOT including the current season to avoid leakage).

**Target variable:** `season_points_share` (continuous, 0–1). Computed from full-season data. Present for 2014–2025, NaN for 2026.

**Rebrand dampening:** For constructors in `MAJOR_REBRANDS`, multiply all historically-derived features (`rule_change_adaptation_score`, `prev_year_points_share`, `prev_year_standing`, `constructor_win_rate_5yr`) by 0.5 in their first season under the new identity.

**Expected output:** `data/features.csv`

## Stage 3 — Model Training (`train.py`)

**Goal:** Train and compare XGBoost and Ridge regression, tune rule-change weight, select the best model.

Implement `train.py` to:

1. Load `data/features.csv`.
2. Split into train set (all complete seasons, i.e., where `season_points_share` is not NaN). Drop `year` and `constructor` from feature columns.
3. **Tune rule-change sample weight** via leave-one-season-out CV: for each candidate weight in [1.0, 1.5, 2.0, 2.5, 3.0], assign that weight to rows where `is_rule_change_year == 1` (all others get 1.0), run full CV, record average Spearman rank correlation. Pick the best weight.
4. **Compare models** using the best rule-change weight:
   - XGBoost regressor (max_depth=4, n_estimators=200, learning_rate=0.05).
   - Ridge regression (alpha tuned via inner CV or default alpha=1.0).
   - Run leave-one-season-out CV for both. For each held-out season, train on all other seasons, predict the held-out season, compute Spearman rank correlation of predicted vs actual standings and MAE on points share.
5. Select the model with the higher average Spearman correlation.
6. Print: best rule-change weight, model comparison results, feature importances for the winning model.
7. Retrain the winning model on all data with the best weight.
8. Save to `models/Grid_Prophet_model.pkl`.
9. Save CV results to `data/cv_results.csv`.

---

## Stage 4 — 2026 Prediction (`predict.py`)

**Goal:** Generate and display 2026 constructor championship predictions.

Implement `predict.py` to:

1. Load the trained model from `models/Grid_Prophet_model.pkl`.
2. Load `data/features.csv` and filter to the 2026 rows (which will have early-season features filled in but no `season_points_share`).
3. If 2026 rows don't exist yet, construct them by: taking the 2025 constructor lineup, setting `is_rule_change_year = 1`, computing `prev_year_*` features from 2025, computing `rule_change_adaptation_score` from historical data, and prompting the user to input round 1–2 results OR pulling them from FastF1 if available.
4. Run prediction → output predicted `season_points_share` per constructor.
5. Rank constructors by predicted share and print a formatted standings table.
6. Save predictions to `data/predictions_2026.csv`.

---

## Stage 5 — Polish & Iteration (Optional)

**Goal:** Improvements once the core pipeline works.

- Add a `Makefile` or CLI entry point so the full pipeline runs with one command (`python -m Grid_Prophet run`).
- Add confidence intervals to predictions (bootstrap or quantile regression).
- Build a simple visualization: predicted vs actual standings for CV seasons, plus the 2026 prediction chart.
- Experiment with adding mid-season update capability (re-predict after each new race).
- Try ensemble: blend the full-data XGBoost with a rule-change-years-only model.