# Grid Prophet takes on Vegas, with the grid penalties this time

*Images are in docs/blog-assets/. Upload as PNG, one at a time, alt text below each.*

In the last post I scored the model on Singapore and said, in bold, that **grid penalties are not in here**. Russell started 19th instead of 6th and the forecast had no idea, so the first thing I wanted before the Las Vegas GP was a forecast that does know. Everything down to "What I'm not sure about" was written before the race and I haven't touched it since, and the scoring is at the bottom.

Getting there took two fixes to the command line tool. Vegas hasn't happened yet, so it has no rows in my data, and `gp race` just crashed with "No entrant data found" when I pointed it at round 21. I made it build a placeholder round from an entry list (by default the round 20 line-up, with a notice saying so) and the calendar, which meant I could run the pre-weekend forecast at all. The second fix was a `--grid-file` flag that takes an explicit starting grid, because the old `--grid-from-quali` option takes the qualifying order and ignores penalties entirely.

## The grid

Five drivers took penalties and I got the new order from the-race.com plus a screenshot of the penalty list (so this is not the official FIA classification, and if it's wrong the forecast is wrong with it):

- Antonelli: 5 places, now starts 6th
- Piastri: 10 places, now starts 17th
- Sainz: 10 places, now starts 20th
- Stroll: 10 places, now starts 21st
- Bearman: 35 places, now starts last

I checked the file by reinserting the penalised drivers into the qualifying order and it matched the published grid. Then I refit through round 20, conditioned on qualifying with the warm-start refit, and simulated the race 50,000 times from that grid.

## The forecast

![Top 10 by win chance and expected points](10_vegas_top10.png)
*Alt: Verstappen leads with ~39% to win and ~16.2 expected points, then Russell at ~15%, Antonelli ~11.5% and Norris ~10%.*

Verstappen is the favourite at **~39%** to win and ~65% for a podium, which sounds big until you remember the model gave him ~69% in Singapore from pole. I haven't checked why Vegas looks that much more open to the model, so I'm not going to make up a reason. Russell is next at ~15%, then Antonelli (~11.5%), Norris (~10%) and Hadjar (~6.6%).

The part I like is how far the order moves once the penalties are in.

![Slope chart from starting slot to forecast finish](10_vegas_slope.png)
*Alt: Slope chart from starting slot to forecast finish order; Antonelli, Piastri, Sainz and Bearman gain places while Stroll stays last.*

Antonelli starts 6th and the model still has him finishing 4th on average, and Piastri starts 17th and comes out ~10th, while Leclerc (starting 10th) lands ~6th. The model is expecting a lot of the field to unwind back toward where the car pace says they belong, which is the thing it's actually built to do.

![Heatmap of finishing positions for the top 12](10_vegas_heatmap.png)
*Alt: Heatmap of finishing-position probabilities for the top 12 drivers, with each driver's starting slot outlined in orange.*

The heatmap shows the same thing with the uncertainty left in. Verstappen is a dark block at P1 and then a long smear, Piastri's mass sits around 8th to 10th nowhere near his starting box, and the faint stripe on the far right of every row is the DNF tail (~[the DNF number here] for most of the front).

## What the penalties cost

![Dumbbell chart of expected points before and after penalties](10_vegas_penalty_cost.png)
*Alt: Expected points from the raw qualifying grid versus the real grid; Piastri loses ~4.6 and Antonelli ~3.6 while the drivers around them gain ~1.*

The raw qualifying grid and the real one give very different forecasts. Antonelli went from a ~27.9% win chance and ~15.4 expected points to ~11.5% and ~11.8, and Piastri went from ~8.1 expected points to ~3.5. Everyone around them picks up roughly a point because there are simply fewer cars ahead.

I'll be honest though, the Singapore re-run taught me not to read too much into this. Ignoring the penalty there happened to score *better* than including it, which was luck and not a reason to prefer the wrong grid. I don't know yet whether the penalty-aware version is better across many races, since only the backtest can say that and I haven't run one with real penalties in it.

## What I'm not sure about

- The grid came from a website and a screenshot, not the FIA.
- The entry list is a guess (round 20's), so any driver change would break it.
- The model was too timid at the front in my calibration check, so ~39% for Verstappen might be low (though that was the pre-weekend model, and I never checked the post-quali one).
- `data/driver_rounds.csv` isn't committed, so rounds 18 to 21 only exist on my laptop right now.

## What actually happened

Verstappen won from pole, Norris was second and Piastri was third, from 17th. Antonelli was fourth, then Gasly, Colapinto, Hamilton, Sainz, Russell and Leclerc, and Albon is recorded as not starting.

![Forecast range against the real result](10_vegas_result.png)
*Alt: Each driver's forecast 10th to 90th percentile finishing range with the actual finish marked; 17 of 22 landed inside, and Piastri, Sainz, Hadjar, Lindblad and Albon did not.*

Scored the same way as the backtest and Singapore, the post-penalty forecast got a winner log loss of **0.94**, against 1.258 for the backtest's post-quali average (so better than average, though nowhere near Singapore's 0.376), a podium Brier of 0.093, a points Brier of 0.167 and a Spearman of ~0.59 against the real order. Verstappen at ~39% was the right call but not a confident one, and two of the three podium finishers had decent odds (Verstappen ~65%, Norris ~39%).

The third was Piastri, who the model gave ~3% for a podium once it had pushed him back for the penalty, and that is the biggest miss. Sainz is the other one, forecast around 17th and finished 8th, and Russell started second, was forecast around 4th and finished 9th. 17 of the 22 drivers landed inside their 80% range against ~17.6 expected, and the five outside were Piastri, Sainz, Hadjar (15th), Lindblad (20th) and Albon.

I don't know why Piastri and Sainz recovered as far as they did, and I'm not going to guess from a results table.

## The penalty comparison, again

This is the part I was most curious about, since Singapore went the other way.

| | Winner log loss | Podium Brier | Points Brier | Spearman |
|---|---|---|---|---|
| Post-penalty grid | 0.94 | 0.093 | 0.167 | ~0.59 |
| Raw qualifying grid | 1.10 | 0.093 | 0.156 | ~0.68 |
| Pre-qualifying | 1.47 | 0.092 | 0.158 | ~0.69 |

Including the penalties gave the best winner call and the worst finishing order, because Piastri and Sainz recovered much further than the model expected and the raw grid had them closer to where they ended up. In Singapore ignoring the penalty scored better by luck and here it did on order, so that's one race each way and the honest summary is that I can't say which version is better yet. The pre-quali forecast scoring about the same as the post-quali ones on everything except the winner is also a little humbling.

A caveat on the data: FastF1 returned no lap data for Vegas, so that round's row has finishing positions and points but no clean-air pace, and the status column looks unreliable (Albon is "did not start" while three drivers are marked retired and classified). I only used finish position and points here, but I'm going to check it before the next refit uses round 21.

Verdict: right winner, a decent podium call with one big hole in it, and no proof either way on penalties.


**Update, Nov 26:** I found a bug in the data behind this forecast. For rounds 18 to 20 one of my data sources spelled some team names differently ("RB F1 Team" instead of "Racing Bulls", "Alpine F1 Team" instead of "Alpine", "Red Bull" instead of "Red Bull Racing"), so the model treated those as brand-new teams it knew nothing about and spread its uncertainty to match. That's where most of Lindblad's and Lawson's ~16% podium chances came from, and it's part of why Verstappen was only ~39%. With the names fixed and the same three forecasts re-run, Verstappen goes to ~47%, Lindblad and Lawson drop to ~2% for a podium, and every score gets better:

| | Winner log loss | Podium Brier | Points Brier | Spearman |
|---|---|---|---|---|
| Post-penalty grid | 0.94 → 0.76 | 0.093 → 0.092 | 0.167 → 0.144 | ~0.59 → ~0.66 |
| Raw qualifying grid | 1.10 → 0.99 | 0.093 → 0.088 | 0.156 → 0.134 | ~0.68 → ~0.71 |
| Pre-qualifying | 1.47 → 1.37 | 0.092 → 0.086 | 0.158 → 0.135 | ~0.69 → ~0.78 |

The numbers above are what the model actually said before the race, so I've left them alone. The verdict doesn't change either: penalties still gave the best winner call and the worst finishing order, and Piastri and Sainz are still the big misses. The fixed model is gloomier about Alpine, though, so Gasly (5th) and Colapinto (6th) now land outside their ranges and it's 16 of 22 inside instead of 17. The data loader now refuses to run if team names get split like this again.
