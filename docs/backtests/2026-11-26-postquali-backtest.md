# Post-quali backtest re-run: 2026-11-26

Roadmap Phase 1 asked for this re-run, for two reasons: to get post-quali calibration numbers on current code, and to confirm that slices 6–10 did not move race forecasts.

**Result:** confirmed on both counts.
- The race forecasts are unchanged, to within noise, since the October run.
- Post-quali podium probabilities are well calibrated (ECE 0.008).
- Pre-weekend calibration is the weak spot (ECE 0.057).

Run on `main@228ca96`. The data is `driver_rounds.csv` after the 2026 team-name repair, though 2026 is not in this backtest. It took 7,565 s (2.1 h), against 9,605 s in October.

```bash
PYTHONPATH=src python -m cli backtest --seasons 2022-2025 --rounds all --method nuts --n-trials 5000 --post-quali --out-dir outputs/postquali_bt_2026-11-26
```

All 88 rounds were scored, with 0 post-quali failures, 0 fallbacks and `gate_valid: true`. The per-round scores and the post-quali calibration bins are in the two CSVs next to this file.

## Against October (`outputs/phase2_2022_2025`, pre-slice-6 code)

Lower is better for log loss and Brier; higher is better for Spearman.

| Metric (mean over 88 rounds) | Oct | Nov 26 |
|---|---|---|
| Winner log loss, pre-weekend | 1.801 | 1.804 |
| Winner log loss, post-quali | 1.258 | 1.266 |
| Podium Brier, pre / post-quali | 0.0886 / 0.0661 | 0.0885 / 0.0664 |
| Points Brier, pre / post-quali | 0.181 / 0.149 | 0.182 / 0.149 |
| Spearman, pre / post-quali | 0.599 / 0.664 | 0.600 / 0.665 |
| Challenger winner log loss | 2.064 | 2.039 |

Every difference is under 0.01. So slices 6–10 did not move race forecasts, and the October numbers quoted in the posts (1.258) still hold to two decimals.

v2 still beats both the pole baseline (3.219) and the pace-rating baseline (2.099). Post-quali still beats pre-weekend, by 0.54 nats.

## Calibration (podium)

| Predicted | Pre-weekend: n, observed | Post-quali: n, observed |
|---|---|---|
| 0–10% | 990, 0.7% (pred 4.5%) | 1223, 1.3% (pred 1.5%) |
| 10–20% | 202, 7.4% (pred 13.9%) | 85, 15.3% |
| 20–30% | 222, 29.3% | 80, 22.5% |
| 30–40% | 209, 42.1% | 69, 31.9% |
| 40–50% | 102, 62.7% (pred 43.7%) | 62, 45.2% |
| 50–60% | 29, 75.9% (pred 53.9%) | 71, 53.5% |
| 60–100% | 4 | 168, 68.2–100% |
| **ECE** | **0.057** | **0.008** |

- **Post-quali** tracks the diagonal in every bin with real counts.
- **Pre-weekend** is too timid about favourites and too generous to backmarkers:
  - Drivers given 40–60% reach the podium 63–76% of the time.
  - Drivers given about 4.5% reach it 0.7% of the time.
  - This is roadmap Phase 3 item 2, and the same overconfident-noise story the season-level gate found (`2026-11-26-season-level-gate.md`).

## Rounds where post-quali was more than 1 nat worse than pre-weekend

| Round | Pre | Post-quali | Pole | Winner |
|---|---|---|---|---|
| 2022 R13 (Hungary) | 1.56 | 4.16 | RUS | VER |
| 2022 R14 (Belgium) | 1.49 | 3.84 | SAI | VER |
| 2024 R21 (São Paulo) | 2.16 | 4.16 | NOR | VER |
| 2023 R5 (Miami) | 1.18 | 2.43 | PER | VER |

All four are Verstappen winning from well down the grid. Post-quali leans on grid position and gives too little weight to a dominant car's recovery. That is the same failure as Piastri and Sainz at Vegas. It is evidence for roadmap Phase 3 item 1, the grid-position curve, and the pole-rate check should look at it directly.
