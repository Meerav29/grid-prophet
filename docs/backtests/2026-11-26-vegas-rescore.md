# Las Vegas 2026 (R21) re-scored on the repaired team names

**Date:** 2026-11-26 · **Code:** main@eda63b6 · **Fit:** NUTS through 2026:20, 111 s, `outputs/20261126_212126`
**Scores:** `2026-11-26-vegas-rescore-scores.csv`

## Why

The live Vegas forecast (Nov 20) was fit on data where 2026 R18–20 used the Ergast team names ("Red Bull", "RB F1 Team", "Alpine F1 Team", "Cadillac F1 Team") alongside the FastF1 ones. The fit saw 25 teams instead of 21, so each team's most recent races were treated as a brand-new team with an uncertain car. PR #33 fixed the data and added a guard that refuses to load split names. This re-runs the same three forecasts on the repaired data, using the same inputs as before:

- **Post-penalty grid:** `--grid-file outputs/vegas21_penalised/grid_post_penalty.csv` (the published FIA grid)
- **Raw qualifying grid:** `--grid-from-quali`
- **Pre-qualifying:** `--mode pre`

All three are scored with `backtest.score_round` against the real result, the same way as the backtest.

## Result

| Version | Winner log loss | Podium Brier | Points Brier | Spearman | In 80% range | VER p(win) |
|---|---|---|---|---|---|---|
| Post-penalty, as published | 0.938 | 0.093 | 0.167 | 0.59 | 17/22 | 39.1% |
| **Post-penalty, repaired** | **0.761** | **0.092** | **0.144** | **0.66** | 16/22 | 46.7% |
| Raw quali grid, as published | 1.101 | 0.093 | 0.156 | 0.68 | 17/22 | 33.2% |
| Raw quali grid, repaired | 0.990 | 0.088 | 0.134 | 0.71 | 17/22 | 37.2% |
| Pre-qualifying, as published | 1.465 | 0.092 | 0.158 | 0.69 | 18/22 | 23.1% |
| Pre-qualifying, repaired | 1.373 | 0.086 | 0.135 | 0.78 | 17/22 | 25.3% |

Lower is better for log loss and Brier, and higher is better for Spearman. For comparison, the backtest's post-quali average winner log loss is 1.266 (PR #34).

## What changed

- **Every version improves on every scoring metric.** The split names cost about 0.1–0.18 nats of winner log loss and about 0.02 of points Brier.
- **Racing Bulls were the biggest distortion.** As published, Lindblad and Lawson each had ~4.7% to win and ~16% for a podium, because the "new" RB F1 Team car was very uncertain. Repaired, they have ~0.4% and ~2%. They finished 20th and 14th.
- **Verstappen is more confident: 46.7% instead of 39.1%.** Some of the "open forecast" in the Vegas posts came from that spread-out uncertainty rather than from the race itself.
- **Alpine moved the other way.** With its history joined up, the repaired fit is more pessimistic about Alpine. Gasly (5th) and Colapinto (6th) fall outside their 80% ranges, which they did not before. Hadjar is now inside his range. The repaired post-penalty forecast has 16 of 22 drivers inside their 80% range, against 17 as published. The six outside are Piastri, Sainz, Lindblad, Gasly, Colapinto and Albon.
- **The ranking of the three versions is unchanged.** The penalty grid still gives the best winner call and the worst finishing order (Spearman 0.66, against 0.71 for the raw grid and 0.78 for pre-quali). Piastri (started 17th, finished 3rd) and Sainz (started 20th, finished 8th) are still the big misses.

## Effect on the Vegas posts

The forecast in `docs/substack/vegas-preview.md` and `vegas-retro.md` is what the model actually said on Nov 20, so the published numbers stay as a true record. What no longer holds:

- The score table (0.94 / 1.10 / 1.47) as a measure of the model. The repaired model scores 0.76 / 0.99 / 1.37.
- "It was an open forecast, and nobody on the grid was close to a lock" was partly caused by the bug.
- Lindblad's ~16% podium chance was a data artifact, not a real read on Racing Bulls.

The conclusions survive: the right winner, Piastri as the biggest miss, and no proof either way on penalties.
