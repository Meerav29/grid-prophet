# Pole-rate check, 2022–2025

**Date:** 2026-11-27 · **Code:** branch `pole-rate-check` @ 5eca012 (main@5630184 plus the new metric) · **Run:** `backtest --seasons 2022-2025 --rounds all --method nuts --post-quali`, 88 of 88 rounds ok, no fallbacks
**Per-round metrics:** `2026-11-27-pole-rate-metrics.csv` (new columns `p_win_pole_postquali` and `p_win_pole_model`)

## Question

The Nov 11 review (item B) found that the post-quali model gave the pole sitter 26%, 15% and 33% in three 2026 races where pole converts about 68% of the time. Is that a real, model-wide underweighting of pole, or three unlucky races? If it is real, the front-loaded grid penalty (item A) is worth building in Phase 3.

## Answer: pole is underweighted, by about 13 points

| Season | Rounds | Pole won | Pole win rate | Model's mean p(win) for the pole sitter, post-quali | Gap | z |
|---|---|---|---|---|---|---|
| 2022 | 21 | 9 | 43% | 39% | −4 pts | 0.4 |
| 2023 | 21 | 13 | 62% | 46% | −16 pts | 1.6 |
| 2024 | 23 | 10 | 43% | 39% | −5 pts | 0.5 |
| 2025 | 23 | 15 | 65% | 37% | −28 pts | 2.9 |
| **2022–25** | **88** | **47** | **53%** | **40%** | **−13 pts** | **2.7** |

z is the excess pole wins divided by the spread the model's own probabilities imply. Across the four seasons pole won 11.7 more races than the model expected, about 2.7 standard deviations. That is unlikely to be luck.

The gap is in every probability band, not just at the top:

| Model's p(win) for the pole sitter | Rounds | Mean predicted | Pole actually won |
|---|---|---|---|
| ≤ 20% | 11 | 15% | 36% |
| 20–35% | 25 | 28% | 36% |
| 35–50% | 27 | 43% | 48% |
| 50–65% | 19 | 56% | 79% |
| > 65% | 6 | 74% | 100% |

## What it means

- **The grid-position flaw is real**, so item A (a front-loaded grid penalty in `src/sim/race.py`, P1 vs P2 costing more than P20 vs P21) is justified as the first Phase 3 change.
- **It is worst in seasons with a dominant pole sitter.** 2025 and 2023 carry most of the gap. 2022 and 2024, with more varied winners, are within noise.
- **The four big post-quali misses in the Nov 26 backtest point the other way.** In each, Verstappen won from well down the grid. A concave curve fits both findings: a large gap between the front slots and a smaller one further back, so pole gets more weight and a fast car starting 10th loses less.
- **Pre-weekend column, for reference.** `p_win_pole_model` is the pre-weekend forecast's win probability for the driver who went on to take pole (mean 16%). It doesn't know the grid, so it isn't a calibration check, and it is kept only to make the post-quali shift visible.

Sanity check: the mean post-quali winner log loss is 1.260, against 1.266 in the Nov 26 run. Only the seeds' sampling noise differs.
