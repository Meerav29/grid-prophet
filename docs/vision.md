# Grid Prophet — Vision and Goal Posts

Locked 2026-11-14. Supersedes the vision in [Project.md](../Project.md) (April 2026, the season-level constructors' regression, now the benchmark only). The model design in [grid-prophet-v2-spec.md](grid-prophet-v2-spec.md) still stands; this document says what that model is for.

## Vision

Grid Prophet is a self-improving F1 race forecaster.

Each race weekend it forecasts on a schedule, scores itself against the result and against bookmaker odds, diagnoses its misses, and proposes changes to itself. A change ships only if it passes a multi-season replay. Meerav approves changes at first, and the system earns the right to merge its own.

It keeps running and keeps improving through the 2027 season. It does not freeze at the first race.

## Deadline

The best model we can build is in place before the first race of 2027. From then on it runs live every weekend.

## Goal posts

1. **Accuracy against the market.** Match bookmaker closing odds on race-winner log loss over a full season, then beat them. Weak baselines (pole wins, pace rating) stay in the report but are not the bar.
2. **Honest probabilities.** Stated ranges hold at their stated rate: an 80% range contains the result about 80% of the time, and a 30% chance happens about 30% of the time.
3. **No manual steps.** A full race weekend (pre-weekend forecast, post-quali forecast, scoring, review) runs without Meerav touching it.
4. **Every variable earns its place.** Each input to the model has a replay result showing it improves the score. Ones that don't are removed.

## How it learns

There are two speeds, and they must not be confused.

- **Every round, automatically: the model updates its beliefs.** New results move its estimates of each car and driver. Car strength drifts through the season, so upgrade packages and development swings show up within a round or two. This is refitting, not a code change, and needs no approval.
- **Occasionally, through a gate: the model changes its structure.** A new variable, a new formula, a retuned assumption. Live misses are where these ideas come from. The multi-season replay is the only thing that can accept one. A handful of races is never enough evidence on its own.

Autonomy over structural changes is staged:

1. The system proposes a change, replays it and reports. Meerav merges.
2. Once the gate has agreed with Meerav's own judgment over a run of changes, the system merges changes that pass, sends a digest, and Meerav can revert.

## Variables

The aim is to account for everything that measurably moves a race result, tested one at a time. Weights are learned from data and never set by hand. Weekend-specific facts are fed in before each run and the simulation responds to them.

Known gaps at lock time, as candidates to test:

- Car-by-track effects. Today a car is assumed equally strong at every circuit. Tracks should be described by measured traits (corner speeds, straight-line share, tyre wear) and each car's strength should vary with them.
- Grid position effect that is not linear: pole is worth more than one slot, and the back of the grid is cheaper to climb.
- Wet-weather skill by driver and car, and a weather forecast as a pre-race input.
- The published penalty grid as a standard input.
- Declared upgrade packages as a signal that a car's strength may step.
- Retirement causes for 2023 onward, which the data source no longer labels.
- Tyres and strategy.

## Proving ground

A time-gated sandbox that replays any past weekend through the full pipeline as if it were live: everything known before the weekend goes in, the pre-weekend forecast runs, quali is released, the post-quali forecast runs, then the result is released and scored.

Rules:

- The sandbox only ever sees data dated before the moment it is pretending to be.
- Hand-maintained files (circuit ratings, track traits) are either fixed before a season starts or dated like any other data.
- One season stays sealed. Changes are tuned on the other seasons and checked on the sealed one. No change is chosen because of its score on the sealed season.

## How it runs

Scheduled Claude Code routines that complement each other, not a hosted service. Cloud routines handle the weekend forecast, scoring and review. The multi-hour replay runs on Meerav's laptop, which holds the full dataset. Hosting is revisited only if routines become the limit.

## Out of scope

- Live in-race, lap-by-lap prediction.
- Placing bets. Odds are a yardstick only.
- A hosted service or public site, for now.

## Open risks

- **Historical odds.** Goal post 1 needs bookmaker odds for past races to grade replays against. No source has been checked yet. Without one, that goal post only starts counting from live 2027 races.
- **Data access.** FastF1's live timing now needs a paid token and is blocked from cloud sessions. Lap data currently arrives through jolpica, later than it used to.
- **Thin evidence.** About 190 races since 2018. Small gains will be hard to tell from noise, so the gate needs a stated margin, not just "better".
- **2027 is the second year of new regulations.** 2026 is the only season of the current cars, so early-2027 forecasts lean heavily on it.
