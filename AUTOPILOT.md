# Autopilot: window 2 finished (Oct 20-26, 2026)

**No routines are running.** Window 2 built slices 7-10 from [docs/autopilot/queue.md](docs/autopilot/queue.md). The review routine merged all four into `auto/queue`, and none were rejected or held. The log is issue [#19](https://github.com/Meerav29/grid-prophet/issues/19).

**`auto/queue` has not been merged into `main`.** `main`'s copies of `queue.md` and `decisions.md` still show slices 7-10 as queued. The `auto/queue` copies are the current ones.

## What window 2 produced

| Slice | PR | What it added |
|---|---|---|
| 7 | #20 | Overtaking-fit call sites are test-protected, each forecast fits only its own circuit, and `collect_v2` has a Miami-style quali fallback (`q_segment="laps"`) |
| 8 | #21 | Season chaining and `gp season`: drivers' and constructors' championships come from the same trials |
| 9 | #22 | `gp backtest --season-level`: cutoffs 2/6/12, constructor Spearman against legacy, title Brier, 80% coverage. **Built, not run** |
| 10 | #23 | Post-quali podium calibration (its own CSV, ECE for each arm), plus a list of rounds where post-quali is more than 1.0 nat worse than pre-weekend |

The suite on `auto/queue` reached 335 passing. Every slice merged on unit tests and review, not on the backtest.

## Live forecasts

These ran on the `forecast-data` branch, outside the queue:

- **R18 US GP:** pre-weekend forecast on Oct 23. The Oct 24 post-quali routine failed twice because the cloud proxy blocks `livetiming-mirror.fastf1.dev`. A post-quali forecast was produced later and scored against the race (committed Oct 30).
- **R19 Mexico City and R20 São Paulo:** pre-weekend and post-quali forecasts, all scored. For R20, winner log loss was 2.24 pre-weekend, 1.44 post-quali on quali order, and 1.11 on the penalty grid.
- Getting these forecasts out needed three code fixes (jolpica race pace, accent-free event names, `--grid-file`). They live **only** on `forecast-data`. [CLAUDE.md](CLAUDE.md) lists them and says how to port them.

## Waiting on Meerav

1. Read the `decisions.md` entries dated 2026-10-20 or later, **on `auto/queue`**. The one most likely to need your call is the Oct 23 entry: the legacy comparison is a rolling Ridge on earlier seasons, not LOSO, so expect it to come in below 0.88.
2. Run the season-level backtest. **This is the Phase 3 gate** (see #22's review pack):
   ```bash
   PYTHONPATH=src python -m cli backtest --seasons 2022-2025 --season-level --method nuts --n-trials 5000 --out-dir outputs/season_bt
   ```
   It needs `data/race_results.csv` and `data/constructor_standings.csv` for the legacy column.
3. Re-run the post-quali backtest to get the real post-quali ECE (#23, review pack step 4).
4. Owner checks the routines left you: does `repair_round.py 2025 6 Q` match your hand-repaired Miami rows (#20)? Does `gp season` on a real fit produce `current_points` that match the real 2026 standings (#21)?
5. Merge `auto/queue` → `main`, then port the `forecast-data` code fixes.
6. Retire `confidence.py` and the pkl only after step 2 passes.

## If you're asking a cloud session to do something here

- Read [CLAUDE.md](CLAUDE.md).
- For Phase 3 work, branch from `auto/queue` until it's merged.
- For forecasts, use the `forecast-data` branch. FastF1's live-timing mirror is blocked from the cloud, so new rounds come in through jolpica.

---

Window 1 (Sep 20 → Oct 4) built slices 1-6. All six were merged to `main` via PRs #13 and #16, and the Phase 2 backtest result is in [docs/phase2-status.md](docs/phase2-status.md). Its run log is issue #4.

Design, branch model, review contract, PR template: [docs/superpowers/specs/2026-09-19-autopilot-design.md](docs/superpowers/specs/2026-09-19-autopilot-design.md).
