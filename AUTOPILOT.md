# Autopilot: window 2, Oct 20-26, 2026

**Read this before you touch anything else.** Scheduled cloud routines are building slices 7-10 from [docs/autopilot/queue.md](docs/autopilot/queue.md) while Meerav is travelling. They open PRs against `auto/queue`, and a separate review routine merges the ones that pass.

**`main` does not move.** `auto/queue` was fast-forwarded to `main` at `303e957` on Oct 20, so everything after that point is autopilot work.

## What's queued

| Day (ET) | Slice |
|---|---|
| Tue Oct 20 (4pm, one-off) | 7: Phase 2 carry-forwards (overtaking call-site test, per-circuit fit, Miami quali fallback) |
| Wed Oct 21 | 8: season chaining and `gp season` |
| Fri Oct 23 | 9: season-level backtest harness (built and tested on the fixture, **not run**) |
| **Sat Oct 24, 8pm** | **US GP (2026 R18) post-quali forecast.** One-off, runs from the `forecast-data` branch and posts to #19 |
| Sun Oct 25 | 10: post-quali calibration |

Builds run around 9am ET, review around 2pm ET, and the digest around 7pm ET. Tuesday's one-off build gets a one-off 9pm review so Wednesday doesn't halt.

## Review in this order when you're back

1. **[docs/autopilot/decisions.md](docs/autopilot/decisions.md)**: entries dated 2026-10-20 or later.
2. **The tracking issue, [#19](https://github.com/Meerav29/grid-prophet/issues/19)**: digests, halts, and the US GP forecast.
3. **Any PR still open against `auto/queue`.** These were held or rejected.
4. **The `auto/queue` → `main` diff.**

## The thing to be skeptical about

Same as window 1: slices merge on unit tests and review, not on the backtest. Slice 9 *builds* the Phase 3 season-level backtest. It doesn't *run* it. **Phase 3 isn't passed until you run it** (the command is in slice 9's PR). Don't retire `confidence.py` or the pkl before then.

The US GP forecast is the first forecast made without you there. It used a model fitted in the cloud from a data snapshot on the `forecast-data` branch, plus R18 quali fetched on the day. Check its PR or comment for the commands it ran.

## The `forecast-data` branch

It holds `data/driver_rounds.csv` (through 2026 R17) and `data/driver_meta.csv`, force-added because `data/` is gitignored. It exists only so the cloud forecast doesn't re-collect 2018-2026 against FastF1's rate limit. **Never merge it into `main`.**

## If you're asking a cloud session to do something here

- Read [CLAUDE.md](CLAUDE.md).
- To continue autopilot work, branch from `auto/queue`. If a slice's PR is open, comment on it rather than pushing to its branch, unless you mean to take it over. In that case comment `HOLD` first so the review routine leaves it alone.
- To make a forecast, use the `forecast-data` branch.

## Stopping it

Comment `HOLD` on a PR to stop that slice. To stop the routines: `RemoteTrigger` update with `enabled: false` (IDs are in Meerav's Claude memory, not in git), or the web UI at <https://claude.ai/code/routines>.

---

Window 1 (Sep 20 → Oct 4) built slices 1-6. All six were merged to `main` via PRs #13 and #16, and the Phase 2 backtest result is in [docs/phase2-status.md](docs/phase2-status.md). Its run log is issue #4.

Design, branch model, review contract, PR template: [docs/superpowers/specs/2026-09-19-autopilot-design.md](docs/superpowers/specs/2026-09-19-autopilot-design.md).
