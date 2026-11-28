# X thread: Grid Prophet overhaul (@memeerav)

Drafted from `docs/blog-grid-prophet-overhaul.md`. Post it after the Substack post is live and swap `[LINK]` in post 6 for the real URL (X counts any link as 23 characters, so the limit holds whatever the URL is). All posts are 214 to 254 characters. Images come from `docs/blog-assets/`. Use the PNGs, since X won't take the SVG.

Numbers are copied from the blog. The charts round winner log loss to 1.80 and 1.26, while the thread uses 1.801 and 1.258 like the TL;DR. If you'd rather the text match the chart labels, swap them.

---

## Post 1: pick one hook

**Hook A (the setup)**

```
I went away for six days and left scheduled AI agents building the next phase of my F1 prediction model.

Then I graded it on the Singapore GP. What worked, what the numbers don't prove, and the one thing that broke:
```

**Hook B (the result)**

```
My F1 model's top three podium odds for Singapore were Verstappen, Leclerc and Antonelli. That was the podium, in that order.

It proves less than it sounds. The rebuild, the backtest, and the one thing that broke:
```

Chart: Hook A goes with `06_routines_infographic.png`. Hook B goes with `09_cover.png`, or leave it bare and let post 5 carry the podium chart.

---

## Post 2: the rebuild and the routines

```
v1 was a season-level XGBoost model for the 2026 constructors' title. v2 is a Bayesian pace model per driver per race, simulated 50,000 times.

I split the next phase into slices. Routines built and reviewed them on a side branch, so main never moved.
```

Chart: `06_routines_infographic.png` (with Hook B). With Hook A it's already used, so leave this one bare.

---

## Post 3: the backtest

```
Backtest: 88 rounds, 2022-2025, refit before every round.

It beats all three baselines, and knowing qualifying cuts winner log loss from 1.801 to 1.258, better in 75 of 88 rounds.

Most of that gain is just knowing the grid, so I don't trust 1.258 much.
```

Chart: `01_baselines.png`. `02_paired_delta.png` is the alternative if you want the 75/13 split to be the picture.

---

## Post 4: where it's wrong

```
Is it the best model I could have made? No.

When it says ~55% for a podium, that happens ~77% of the time. When it says ~3%, it's ~0.5%. Too timid at the front, too generous at the back.

No ablation yet, so I can't say which slices earn their keep.
```

Chart: `05_calibration.png`.

---

## Post 5: Singapore

```
Singapore: ~69% on Verstappen to win. He won from pole, with Leclerc 2nd and Antonelli 3rd, the model's top three podium odds (86%, 79%, 55%).

Winner log loss: 0.376. But the pole-wins baseline scored 0.001 and beat it. Easy races are easy to call.
```

Chart: `08_singapore_podium.png`. If you use Hook B, cut the first sentence here so the podium doesn't show up twice.

---

## Post 6: what broke, plus the link

```
What broke: re-running from the real starting grid, the warm-start refit took 1,744 seconds against a 120 second target. In the backtest it averaged ~73.

I don't know why yet, so that's next.

Full write-up, charts and all: [LINK]
```

Chart: none, so the link preview card shows. If X drops the preview when there's an image, keep it bare.
