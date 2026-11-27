# Grading my Vegas forecast, and the three drivers it got wrong

*Images are in docs/blog-assets/. Upload as PNG, one at a time, alt text below each.*

Before the Las Vegas GP I put out a forecast that, for the first time, knew about the grid penalties (if you missed it, the short version is that Antonelli, Piastri, Sainz, Stroll and Bearman all got dropped and I fed the post-penalty grid into the simulator, which then ran the race 50,000 times). The race is over now so I can finally do the part I was most curious about, which is checking it against what actually happened and being honest about where it was wrong.

## What the model said

![Forecast table for all 22 drivers](11_vegas_forecast.png)

*Alt: Forecast table for all 22 drivers: Verstappen leads with ~39% to win, ~65% for a podium and ~16.2 expected points, then Russell ~15%, Antonelli ~11% and Norris ~10%. Penalised drivers are tagged, and Piastri has under 1% to win and ~3% for a podium.*

Verstappen was the favourite at ~39% to win and ~65% for a podium, with Russell at ~15%, Antonelli at ~11.5% and Norris at ~10%. It was an open forecast, and nobody on the grid was close to a lock.

## What actually happened

Verstappen won from pole, Norris was second and Piastri was third, from 17th. Antonelli finished 4th, then Gasly, Colapinto, Hamilton, Sainz, Russell and Leclerc, and Albon didn't start (I said in the first version that the statuses looked unreliable, and that was wrong, his "did not start" is real and I've since re-collected the round 21 data to check).

![Forecast range against the real result](11_vegas_range.png)

*Alt: Each driver's forecast 10th to 90th percentile finishing range with the actual finish marked, ordered by actual finish; 17 of 22 landed inside, and Piastri, Sainz, Hadjar, Lindblad and Albon did not.*

So the winner was right, and that's the good news. Verstappen at ~39% was the most likely single outcome, and Norris at ~39% for a podium and Antonelli finishing 4th (right where the model's average had him) were fine too. 17 of the 22 drivers landed inside their 80% range, against ~17.6 you'd expect from a well calibrated forecast, so on the whole it wasn't badly off.

## The three it missed

**Piastri.** The model pushed him back to 17th for the penalty and gave him ~3% for a podium, and he finished third. That's the biggest miss by a long way.

**Sainz.** Started 20th, the forecast had him around 17th, and he finished 8th.

**Russell.** The mirror image. He started 2nd, the model had him finishing around 4th, and he came home 9th. Hadjar (started 4th, finished 15th) and Lindblad (started 5th, finished 20th) also finished outside their ranges, though neither of them was as far off as these three.

I don't know why Piastri and Sainz recovered as far as they did, and I'm not going to guess from a results table. What I can say is that the model expects cars to drift back toward where their pace says they belong, and a 10-place penalty is exactly where that assumption gets tested hardest.

## The penalty comparison, again

![Expected points before and after penalties](11_vegas_penalty.png)

*Alt: Expected points from the raw qualifying grid (hollow) versus the real post-penalty grid (filled); Piastri loses ~4.6 and Antonelli ~3.6 while Verstappen, Norris, Russell, Hadjar and Leclerc each gain about a point.*

In the Singapore post ignoring the penalty scored better than including it, and I said that was luck and not a reason to prefer the wrong grid. Vegas gave me a second race to check that against, and I scored three versions the same way I score the backtest (lower is better for the first three, higher for Spearman).

- Post-penalty grid: winner log loss 0.94, podium Brier 0.093, points Brier 0.167, Spearman ~0.59
- Raw qualifying grid: winner log loss 1.10, podium Brier 0.093, points Brier 0.156, Spearman ~0.68
- Pre-qualifying: winner log loss 1.47, podium Brier 0.092, points Brier 0.158, Spearman ~0.69

Including the penalties gave the best winner call and the worst finishing order, because Piastri and Sainz recovered much further than the model expected and the raw grid had them closer to where they ended up. For context the backtest's post-quali average for winner log loss is 1.258, so 0.94 is better than average, though nowhere near Singapore's 0.376.

The part that's a little humbling is that the pre-quali forecast, made before anyone had turned a lap, scored about the same as the post-quali ones on everything except the winner.

## What I'm taking from it

That's one race each way now, so the honest summary is that I can't say whether the penalty-aware grid is better yet, and arguing it from two races would be the same mistake as arguing it from one. I'm also not claiming any of this is a verdict on the model, since a single weekend can't say that, and only the backtest can.

One correction while I'm here: the first version of this analysis said the round 21 data had no pace numbers. That has since been fixed, the pace is back in via a different lap data source, and the refit through Vegas is what I'm using for the Qatar forecast.

**Update, Nov 26:** I found a bug in the data behind this forecast. For rounds 18 to 20 one of my data sources spelled some team names differently ("RB F1 Team" instead of "Racing Bulls", "Alpine F1 Team" instead of "Alpine", "Red Bull" instead of "Red Bull Racing"), so the model treated those as brand-new teams it knew nothing about and spread its uncertainty to match. That's where most of Lindblad's and Lawson's ~16% podium chances came from, and it's part of why Verstappen was only ~39%. With the names fixed and the same three forecasts re-run, Verstappen goes to ~47%, Lindblad and Lawson drop to ~2% for a podium, and every score gets better:

| | Winner log loss | Podium Brier | Points Brier | Spearman |
|---|---|---|---|---|
| Post-penalty grid | 0.94 → 0.76 | 0.093 → 0.092 | 0.167 → 0.144 | ~0.59 → ~0.66 |
| Raw qualifying grid | 1.10 → 0.99 | 0.093 → 0.088 | 0.156 → 0.134 | ~0.68 → ~0.71 |
| Pre-qualifying | 1.47 → 1.37 | 0.092 → 0.086 | 0.158 → 0.135 | ~0.69 → ~0.78 |

The numbers above are what the model actually said before the race, so I've left them alone. The verdict doesn't change either: penalties still gave the best winner call and the worst finishing order, and Piastri and Sainz are still the big misses. The fixed model is gloomier about Alpine, though, so Gasly (5th) and Colapinto (6th) now land outside their ranges and it's 16 of 22 inside instead of 17. The data loader now refuses to run if team names get split like this again.

Verdict: right winner, a decent podium call with one big hole in it, and no proof either way on penalties.
