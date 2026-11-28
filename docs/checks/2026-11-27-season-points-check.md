# Owner check: `gp season` current points vs the real 2026 standings

**Date:** 2026-11-27 · **Roadmap:** Phase 1, "does `gp season` report current points that match the real 2026 standings"

## Result: they didn't match, and now they do

I compared `gp season`'s `current_points` after R21 with jolpica's official standings after round 21 (`/ergast/f1/2026/21/driverstandings` and `/constructorstandings`).

**Before the fix**, 10 drivers were short, by a total of exactly 101 points. Antonelli had 387 against an official 402, Verstappen 286 against 304, and Norris 239 against 264. Mercedes was 645 against 666.

**The cause: 2026 R11 (Hungary) had race pace but no race result.** All 22 race rows had clean-air lap times but no grid, finish, status or points. A Grand Prix pays 101 points, which accounts for the whole gap. No other race or sprint from 2018 to 2026 has this problem. I checked every one for missing finishes or zero points.

**The fix:**
- I re-fetched the round with `src/repair_round.py 2026 11 R`, which filled in grid, finish, status and 101 points for all 22 drivers.
- The repair wrote the Ergast team names ("RB F1 Team", "Red Bull") back in, because the script skipped the name mapping that `collect_v2` applies. That would have tripped the team-name guard from #33. This PR fixes the script: repaired rows now go through `canonical_team_names`, and the file is checked with `check_team_names` before it's written.
- A backup of the data from before the fix is in `data/driver_rounds.pre-r11fix-2026-11-27.csv`. The data file itself is gitignored.

**After the fix**, all 25 drivers and all 11 constructors match the official standings exactly.

| | Official | `gp season` before | After |
|---|---|---|---|
| ANT | 402 | 387 | 402 |
| VER | 304 | 286 | 304 |
| HAM | 267 | 257 | 267 |
| RUS | 264 | 258 | 264 |
| NOR | 264 | 239 | 264 |
| LEC | 250 | 238 | 250 |
| Mercedes | 666 | 645 | 666 |
| Ferrari | 517 | 495 | 517 |
| Red Bull Racing | 440 | 414 | 440 |
| McLaren | 431 | 406 | 431 |

## Effect on the forecasts

- **The pace model wasn't affected.** The refit through R21 (`outputs/20261127_231618`, 72 s) has the same 3,654 race observations as the Nov 26 fit, because the pace data for R11 was already there.
- **Qatar pre-quali barely moves.** No driver's win chance changes by more than 0.5 percentage points, and no DNF chance by more than 1.2. `latest.txt` now points to the new fit, so Saturday's post-quali run will use it.
- **The championship table does change.** Verstappen is clear in second (304 against 267 for Hamilton). The fight for third is now Hamilton 267, Russell 264 and Norris 264, with Leclerc on 250. Russell is the favourite for third at 42%.
