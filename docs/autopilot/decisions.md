# Decisions made without you — grid-prophet

Every entry here is a design question the v2 spec did not settle, which a build
routine resolved on its own so the slice could ship. Appended by the routines;
newest last.

Per the design's ambiguity policy, a routine picks the most defensible option,
records it here and in the PR body, and proceeds. It does not stop, and it does
not guess silently.

**Read this before reviewing the `auto/queue` diff.** A decision you disagree
with is cheaper to catch here than in the code.

Format:

```
## <date> — slice-<n>, PR #<n>
Question:    <what the spec left open>
Chosen:      <what was done>
Rejected:    <the alternative, and why not>
Blast radius: <files or functions affected, and how reversible>
```

---

## 2026-09-19 — slice-1, PR #5
Question:    §6 calls the post-quali target "a couple of minutes", relaxed for "roughly the first ~6 rounds", but names no numbers and does not say what the clock measures.
Chosen:      120s warm-start target, 300s relaxed allowance, early window = rounds 1–6 inclusive. Both measured as total wall clock from the start of conditioning (warm-start attempt *plus* fallback), because that is when results actually land for the user.
Rejected:    Timing the fallback refit alone — a round that burned 119s failing a warm start would be scored as if that time were free, hiding exactly the slow rounds §6 wants counted. Also rejected: leaving the numbers unnamed, which makes "missed its target" untestable.
Blast radius: Three constants at the top of `src/model/conditioning.py`. Changing them changes which rounds are *recorded* as fallbacks or regressions, never a forecast.

## 2026-09-19 — slice-1, PR #5
Question:    §3.3 says post-quali mode uses "the real grid" but not where that grid is read from.
Chosen:      `grid_position` on the round's race rows (FastF1's `GridPosition`, penalties already applied). 0 (pit-lane start) and missing values rank to the back; ranks are dense 1..n; a round with no usable grid raises.
Rejected:    Quali classification order — it ignores grid penalties, so it is not the grid the race started from. Also rejected: silently simulating a grid when the real one is missing, which would label a pre-weekend forecast as post-quali.
Blast radius: `_real_grid_for_round` in `src/cli.py`, reached only by `--mode post-quali`.

## 2026-09-19 — slice-1, PR #5
Question:    §6 says quali "is added as an observation" but does not say what happens to the round's race row during conditioning.
Chosen:      Hold out the round's race observations, keep its quali, and keep the round on the random-walk axis (`build_pace_data(hold_out_race_round=...)`).
Rejected:    Fitting through round N−1 and treating quali as an extra round — that drops the round's own `car[team, round]` node, so the forecast would come off a stale round. Letting the race rows in is leakage and was never an option.
Blast radius: One new optional argument to `build_pace_data`; every existing caller's behaviour is unchanged.
