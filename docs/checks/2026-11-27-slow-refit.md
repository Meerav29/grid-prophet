# Why one warm-start refit took 1,744 s

**Date:** 2026-11-27 · **Roadmap:** Phase 1, "find out why one warm-start refit took 1,744 seconds against a 120-second target"

## The run

`outputs/sgp17_realgrid/conditioning_log.csv`, Oct 11: the post-quali re-run of Singapore (2026 R17) from the real starting grid, conditioned on the Oct 11 fit through R16 (`outputs/20261011_004402`). The warm start took 1,743.96 s, then fell back to a full refit that took 67.6 s, for 1,811.5 s in total. Two earlier post-quali runs of the same round on the same fit took 73.3 s and 47.5 s.

## Cause: the laptop was asleep for most of it

- The output folder was last written at 15:24:37, so the run started at about 14:54:26.
- Windows' System log (Kernel-Power events 506 and 507, entering and leaving Modern Standby) shows the machine in standby from 14:55:20, waking briefly at 15:01:30, and staying asleep again until 15:22:58.
- That is about 27½ minutes of sleep inside a 29-minute warm start. The remaining ~90 s is in line with every other warm start.
- `run_conditioning` times with `time.perf_counter`, which on Windows keeps counting through standby, so the sleep was booked as sampling time.

## Confirmation

I re-ran the same refit on the same Oct 11 fit with today's code (`outputs/refit_repro`). The warm start took 86.5 s (65 s of sampling) with no fallback. The 88-round post-quali backtests on Nov 26 and Nov 27 had no fallbacks and no regressions either.

## Conclusion

This wasn't a model or sampler regression, and the spec's "late-season miss is a real regression" rule doesn't apply. Two small things are worth doing in Phase 2, neither urgent:

- **Keep the machine awake during long runs.** The live race-week routine and the backtest should hold a keep-awake for as long as they run.
- **The warm start has no time limit.** `run_conditioning` only checks the target after the warm start finishes, so a warm start that really did stall would run to completion before falling back, and the fallback would add time rather than save it. A deadline on the warm start, for example a sampling callback that stops it at the target, would cap the worst case.

The overhaul blog post (`docs/blog-grid-prophet-overhaul.md`, the Substack copy and `x-thread.md`) still says the cause is unknown. If you want, add a one-line update there.
