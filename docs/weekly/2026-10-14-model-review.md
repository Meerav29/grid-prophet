# Weekly model review — 2026-10-14

**Race weekend check:** No. Round 17 (Singapore) ran Oct 9–11; round 18 (United
States, COTA) is Oct 23–25. This weekend (Oct 17–18) is a gap, so this run did
the prediction-vs-outcome review instead of the pipeline refresh.

## What I could and couldn't compare

- Stored prediction snapshots (`data/predictions_2026_r*.csv`) are gitignored, so
  none exist in this cloud checkout. The only dated prediction in the repo is the
  README table (v1, after rounds 1–11).
- A full `collect` was blocked by FastF1's 500-calls/hour limit (2020–2025
  schedules failed), so I did not retrain. Actuals below come from the
  Jolpica/Ergast API (standings after round 17).
- Actual shares are after R17 of 23, not final. Treat deltas as a mid-season
  read, not a verdict.

## v1 (R1–11 prediction) vs actual standings after R17

| Constructor | Predicted | Actual share | Delta (pred − actual) |
|---|---|---|---|
| Mercedes | 26.2% | 29.9% | −3.7 |
| Ferrari | 23.6% | 23.0% | +0.6 |
| McLaren | 15.0% | 17.5% | −2.5 |
| Red Bull | 15.0% | 17.2% | −2.2 |
| Racing Bulls | 4.7% | 5.2% | −0.5 |
| Alpine | 4.9% | 3.5% | +1.4 |
| Haas | 1.7% | 1.6% | +0.1 |
| Audi | 1.5% | 1.2% | +0.3 |
| Williams | 1.4% | 0.6% | +0.8 |
| Cadillac | 0.2% | 0.0% | +0.2 |
| Aston Martin | 0.1% | 0.4% | −0.3 |

Findings:

1. **Ranking is good.** Top four order is exactly right (Mercedes, Ferrari,
   McLaren, Red Bull). Only Alpine/Racing Bulls (5/6) and Aston Martin/Cadillac
   (10/11) are swapped.
2. **Shares are biased low at the top.** The 11 predictions sum to 94.3%, not
   100%. The top four are all under-predicted (−3.7, +0.6, −2.5, −2.2); the
   missing mass is about 5.7 points, nearly all of it in the front four. Mercedes
   is the largest miss and sits outside its stated 80% CI (±2.3).
3. **Intervals are too narrow.** Ferrari (±1.2) and Alpine (±0.5) land inside or
   near; Mercedes, McLaren and Red Bull miss by 2–3x their CI half-width. The
   bootstrap only captures resampling noise, not model error.
4. **Back of the grid is over-predicted slightly** (Alpine +1.4, Williams +0.8),
   consistent with the v2 calibration note that low-end probabilities are too
   generous.

## v2 evidence already in the repo (docs/phase2-status.md)

- Post-quali log loss 1.258 vs pre-weekend 1.801, but post-quali is high variance
  (13 rounds worse, some by 1.5–2 nats).
- Pre-weekend podium probabilities are too generous at the back (~3% predicted
  vs ~0.5% observed) and too timid at the front.
- Single run, no CI on the delta; no ablation per slice; post-quali calibration
  never checked.

## Options to improve

| # | Change | Why | Effort | How |
|---|---|---|---|---|
| 1 | Renormalise predicted shares to sum to 1 | Removes the ~5.7 pt systematic low bias for free; mostly fixes top-four under-prediction | Low (≈10 lines) | In `src/predict.py`, after the model output, divide by the column sum before CIs. Add a unit test that shares sum to 1. |
| 2 | Predict with a Dirichlet/log-ratio target instead of raw share | Shares are compositional; a direct regression per team ignores the sum constraint and shrinks extremes | Medium | Fit on log-ratio (ALR/CLR) in `src/train.py`, invert with softmax in `predict.py`. Compare LOSO CV Spearman and MAE vs current. |
| 3 | Widen CIs with a conformal/residual term | Current 80% CI misses by 2–3x at the top | Low–Medium | Add LOSO CV residual std to the bootstrap spread in `src/confidence.py`; check empirical coverage. |
| 4 | Store a snapshot every round and score it | There is nothing to evaluate against week to week, because snapshots are gitignored | Low | Un-ignore `data/predictions_2026_r*.csv` (small), or write `docs/weekly/snapshots/`. Then this review can compute error by horizon automatically. |
| 5 | Post-quali calibration check + shrinkage | Confident misses (2022 R13/R14, 2024 R21) dominate the downside | Medium | Run reliability bins on `conditioning_log.csv`/`calibration_podium.csv` from the post-quali arm; add a temperature scale or blend toward pre-weekend probs. |
| 6 | Slice ablation + re-run with slice 6 | Unknown which Phase 2 slices help | High compute (~2.7h per run) | `gp backtest --post-quali` with slices toggled. |
| 7 | Phase 3 `gp season` | Direct comparison of v2 championship forecast vs the legacy 0.88 benchmark | High | Per the v2 spec §8. |

**Recommended order:** 1 and 4 now (cheap, and 4 makes every later review
possible), then 3, then 2 if the bias persists after renormalisation.

## Automating race weeks (follow-up)

The race-weekend branch is just `python -m src update && python -m src plots`
plus a commit/push of `plots/` and the README table. It is not scripted yet.
Two blockers to handle when writing it: FastF1's 500 calls/hour limit (needs a
resumable loop and a persisted cache between runs) and the fact that `data/` is
gitignored, so a fresh cloud checkout starts empty. Persisting the FastF1 cache
or committing snapshots solves both.
