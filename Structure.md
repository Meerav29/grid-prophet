# Grid Prophet — Claude Code Setup Prompt

Copy this entire block into Claude Code to initialize the project.

---

## Prompt

Initialize a Python project called `Grid Prophet`. Follow these instructions exactly.

### 1. Create this directory structure:

```
Grid Prophet/
├── README.md
├── PLANNING.md
├── PROJECT.md
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
    └── exploration.ipynb
```

### 2. File contents:

**`requirements.txt`:**
```
fastf1>=3.3
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
jupyter>=1.0
matplotlib>=3.7
seaborn>=0.12
```

**`.gitignore`:**
```
__pycache__/
*.pyc
.ipynb_checkpoints/
data/*.csv
data/*.parquet
models/*.pkl
models/*.json
.venv/
*.egg-info/
dist/
build/
```

**`src/__init__.py`:**
```python
"""Grid Prophet — F1 Constructor Championship Predictor"""
```

**`src/collect.py`:**
```python
"""Pull historical race results from the FastF1 API and save to CSV."""
```

**`src/features.py`:**
```python
"""Build per-constructor, per-season feature matrix from raw race results."""
```

**`src/train.py`:**
```python
"""Train XGBoost model on historical seasons with rule-change-year weighting."""
```

**`src/predict.py`:**
```python
"""Generate 2026 constructor championship predictions from early-season data."""
```

**`notebooks/exploration.ipynb`:**
Create a minimal valid Jupyter notebook with one markdown cell containing: "# Grid Prophet — Data Exploration"

### 3. Copy in the full contents of README.md, PLANNING.md, and PROJECT.md from the project docs provided.

### 4. Initialize a git repo and make the first commit with message: "chore: scaffold Grid Prophet project structure"

Do NOT implement any logic. Only scaffold.