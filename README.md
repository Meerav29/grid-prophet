# Grid Prophet

ML model trained on ~10 years of F1 data (via FastF1) to predict the 2026 constructor championship standings. Emphasizes early-season performance signals from the first 2 races and a rule-change-year adaptation variable to capture how teams historically perform during major regulation shifts like 2026.

## Quick Start

```bash
pip install -r requirements.txt
python src/collect.py          # Pull historical data from FastF1
python src/features.py         # Build feature matrix
python src/train.py            # Train model + cross-validate
python src/predict.py          # Generate 2026 predictions
```

## How It Works

The model uses per-constructor, per-season features — including early-season dominance signals, historical rule-change adaptation scores, prior-year momentum, and driver quality proxies — to predict each team's share of total championship points. Rule-change years are weighted more heavily during training to better capture the dynamics of regulation shifts.

See [PLANNING.md](PLANNING.md) for the staged implementation plan and [PROJECT.md](PROJECT.md) for detailed project context and design decisions.