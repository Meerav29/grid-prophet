# Grid Prophet — Roadmap to the 2027 Season

Written 2026-11-14. Serves [vision.md](vision.md). Dates assume the 2027 opener is in early March; adjust once the calendar is confirmed.

Each phase gets its own design and plan when it starts. This document fixes the order, the contents and what "done" means for each.

Standing rule for every phase: a model change is only called an improvement when the multi-season replay says so. Unit tests show the code works, not that the forecast is better.

---

## Phase 1 — One source of truth (now → Nov 30)

**Goal:** `main` holds everything, the last piece of the old plan is settled, and the rest of 2026 is forecast live.

Repo:

- Move the local checkout to `main`. Commit `vision.md` and this roadmap.
- Merge the two doc-only branches: the Nov 10 status update (`feat/eager-turing-3zk4wb`) and the Nov 11 weekly review (`claude/beautiful-cerf-f3um3i`).
- Port the three fixes that live only on `forecast-data`, by hand, source and tests only: lap times from jolpica, accent-free event names with renamed-track pooling, and `--grid-file`. Do not merge that branch.
- Clear out finished branches and worktrees, after Meerav confirms the list.

Settle the old plan:

- Two owner checks left by the routines: does the automated Miami 2025 repair match the hand repair, and does `gp season` report current points that match the real 2026 standings.
- Run the season-level backtest, the gate that has never been run. Record pass or fail against the spec's bar: constructor ranking at least as good as the old model at every cutoff, 80% ranges holding 70–90% of the time.
- If it passes, retire the old regression's production path and keep it as a benchmark only.
- Re-run the post-quali backtest on current code, to get the post-quali calibration numbers and confirm slices 6–10 did not move race forecasts.
- Find out why one warm-start refit took 1,744 seconds against a 120-second target.

Live:

- Forecast and score Las Vegas (Nov 19–21) and every round after it, with the published penalty grid as the Sunday run.
- Produce the first real drivers' and constructors' championship forecast from the new model.
- Run the cheap pole-rate check from the Nov 11 review across 2022–2025, to learn whether the grid-position flaw is real before Phase 3 spends time on it.

**Done when:** no code exists only on a side branch; the season-level gate has a recorded result; every remaining 2026 race has a forecast and a score.

---

## Phase 2 — Proving ground (December)

**Goal:** a sandbox that can judge any change, quickly and honestly. Everything in Phase 3 depends on it.

- **Time gate.** One place where all data is read, taking an "as of" moment (season, round, and stage of the weekend). Nothing dated after that moment can be read. A test fails if anything tries.
- **Weekend replay.** One command replays a past weekend in live order: pre-weekend forecast, release quali, post-quali forecast, release the result, score. It runs through the same code the live routines will use.
- **2026 included.** The replay covers 2022–2026. Changes are tuned on 2022–2025. 2026 is the sealed check season: looked at to confirm a change, never to choose one.
- **Dated reference files.** Circuit ratings and track traits are fixed per season or carry dates, so hindsight cannot leak in through them.
- **One scorecard.** Every replay writes the same report: winner log loss, podium and points accuracy, full finishing-order score, calibration, and how often the 80% ranges hold, for each forecast stage.
- **The gate, written down before it is used.** A change passes if it improves the main score by more than a stated margin on the tuning seasons, with the round-by-round differences showing the gain is not noise; does not worsen calibration; and does not get worse on 2026.
- **Odds.** Find a source of historical closing odds for 2022–2026, convert them to probabilities, and add them to the scorecard as the bar. If no usable source exists, record that and start collecting odds live from 2027 round 1.
- **Speed.** The full replay takes 2.7 hours today. Run seasons side by side, warm-start each round from the last, and keep a coarse mode for quick looks. Target: a quick look in about 15 minutes, a full gate run overnight on the laptop with nobody starting it.
- **First use: grade what already exists.** Switch off each existing piece in turn (sprints, reliability split, overtaking fit, wet noise) and record whether it earns its place. This starts the variable ledger.

**Done when:** any weekend from 2022–2026 can be replayed with one command; the gate is documented; the current model has a baseline scorecard including 2026; the odds question has an answer; the ledger covers every existing variable.

---

## Phase 3 — Model sprint (mid-December → end of January)

**Goal:** the best model that the evidence supports, one tested change at a time.

Rules: one change per replay. Every result goes in the ledger, kept or rejected, with its numbers. Rejected ideas are recorded so they are not retried blindly.

Candidates, in the order I would test them (largest expected gain for least work first):

1. **Grid-position curve.** Replace the flat cost per grid slot with one where the front is worth more and the back is cheaper to climb. Already supported by three live races; Phase 1's check decides if it is real.
2. **Calibration.** The model is too timid about favourites and too generous to backmarkers. Fix the noise structure, or as a fallback, rescale the probabilities.
3. **Car-by-track traits.** Describe each track by measured traits (corner-speed mix, straight-line share, tyre wear) and let each car's strength vary with them. Likely the biggest single gain, and the most work. Needs a track-trait table built from lap data first.
4. **Practice pace.** Friday long runs as a third forecast stage between pre-weekend and post-quali. Not in the model at all today.
5. **Recent-form weighting.** Let late-season upgrades show up faster in a car's estimate.
6. **Wet-weather skill** by driver and car, plus a weather forecast as a pre-race input.
7. **Retirement causes for 2023 onward**, so the reliability model can tell who is likely to retire, not just how many.
8. **Declared upgrade packages** as a signal that a car's strength may step.
9. **Tyres and strategy.** Largest and least certain. Attempt only if time remains; otherwise it carries into 2027.
10. **A different method.** Re-run the rival model and a blend of the two on the new sandbox, so the choice of method is re-tested and not assumed.

**Done when:** every candidate attempted has a ledger entry; the kept set beats the Phase 2 baseline on the tuning seasons and holds up on 2026; a model version is frozen for the 2027 opener by the end of January.

---

## Phase 4 — Close the loop (mid-January → end of February)

**Goal:** the weekend runs itself, and the improvement loop exists end to end.

The routines, each doing one job:

| Routine | When | Job |
|---|---|---|
| Data refresh | After each session, and again when lap data is published | Pull new results and laps |
| Pre-weekend forecast | Thursday | Refit, forecast the race and both championships |
| Post-quali forecast | Saturday | Condition on quali, re-forecast |
| Grid update | Sunday morning | Re-run on the published penalty grid |
| Score | Monday | Grade every forecast against the result and the odds |
| Review | Tuesday | Look across recent rounds, name the worst misses, suggest causes |
| Propose | After a review finds something | Build one change on a branch and request a gate run |
| Gate | Overnight on the laptop | Replay the change, attach the scorecard, pass or fail |
| Digest | Weekly | One summary for Meerav: forecasts, scores, proposals waiting |

Also in this phase:

- **Approval log.** Each proposal records the gate's verdict and Meerav's decision. Agreement over a set run of changes is what unlocks self-merging.
- **2027 setup.** New driver line-ups, the calendar, any new circuits, season-start assumptions, and pre-season testing as an input.
- **Dress rehearsal.** Replay the first six rounds of 2026 through the routines as if live, with nobody stepping in.

**Done when:** the rehearsal completes six weekends with zero manual steps; one proposed change has travelled the whole path from review to gate to decision; the pre-weekend forecast for 2027 round 1 is produced by the routines.

---

## Phase 5 — Live 2027 (March onward)

**Goal:** run every weekend, learn in season, widen the system's autonomy as it earns it.

- Every round: forecast, score, refit. The model's view of each car and driver moves with results, which is how upgrades are absorbed.
- Structural changes continue through the gate. As 2027 rounds accumulate they join the replay.
- Monthly check against the four goal posts, with the gap to the odds as the headline number.
- Move to self-merging when the approval log supports it.
- Mid-season: decide whether routines are still enough or hosting is now worth it.

---

## Time, effort and who does what

These are estimates. The unit of agent work is a "slice": one reviewed pull request of up to about 600 lines, the size the autopilot windows used. Those windows delivered about three slices a week.

| Phase | Agent work | Replay compute | Meerav's time | Self-run share |
|---|---|---|---|---|
| 1. One source of truth | 6–8 slices | Two long runs, about 3 h each | 5–6 h in total | About 70% |
| 2. Proving ground | 8–10 slices | A few full runs while building | 6–8 h in total | About 85% |
| 3. Model sprint | 20–25 slices | 10+ overnight gate runs, many quick looks | 1–2 h a week | About 85% |
| 4. Close the loop | 8–10 slices | Six-weekend rehearsal | About 8 h in total | About 75% |
| 5. Live 2027 | As proposals arise | One gate run per proposal | 30–45 min a week | 90% and rising |

Totals to the 2027 opener: roughly 45–55 slices of agent work and 40–50 hours of Meerav's time, about 3 hours a week over 16 weeks.

**The schedule is tight.** At three slices a week, 50 slices takes 17 weeks, and there are 16. Two things make it fit: Phases 3 and 4 overlap in January because they touch different code, and the build cadence rises to four or five slices a week. If it still does not fit, the cut line is the bottom of the Phase 3 list: tyres and strategy first, then upgrade packages. Both carry into 2027 without harming the opener.

**The true limit is replay compute, not build speed.** Phase 3 needs something like 30 replay runs. At today's 2.7 hours each, on one laptop, that alone is most of the sprint. This is why replay speed is a Phase 2 deliverable and not a nice-to-have.

### What needs Meerav, and why

Agents can write, test, review, forecast, score and diagnose. Four kinds of thing stay with Meerav:

- **Decisions that set the rules.** The gate's pass rule, the sealed season, whether to pay for data, and which model version is frozen for the opener.
- **Merging model changes**, until the approval log supports self-merging. About one decision per Phase 3 candidate, each a few minutes with the scorecard in hand.
- **Access.** Creating and deleting routines, account tokens, any paid subscription, and keeping the laptop on for overnight runs.
- **Sanity checks on hand-made data.** The track-trait table, relabelled retirement causes, and the 2027 line-ups. An agent drafts them; a person who watches the sport should read them once.

Things that look like they need Meerav but should not by the end of Phase 2: starting a replay, reading raw backtest output, and repairing a round's data by hand.

### Which model does what

- **Fable:** the design for each phase, the gate's pass rule, and the hard diagnoses, such as why a change helped on one season and hurt on another.
- **Opus:** changes to the statistical model, the time gate, and reviewing every pull request against its acceptance criteria.
- **Sonnet:** porting and plumbing, data collection, and the recurring routines (forecast, score, digest), which run often and follow a fixed script.

### Week by week

| Weeks | Dates | Focus |
|---|---|---|
| 1–2 | Nov 14 – Nov 30 | Phase 1. Las Vegas runs live in week 1. |
| 3–6 | Dec 1 – Dec 28 | Phase 2. The grid-position curve starts in week 5 as the sandbox's first real customer. |
| 7–11 | Dec 29 – Jan 31 | Phase 3 at full pace. Phase 4's routines start in week 9. |
| 12–15 | Feb 1 – Feb 28 | Phase 4: rehearsal, 2027 setup, pre-season testing. Model frozen; only fixes go in. |
| 16 | Early March | 2027 round 1, live. |

---

## Decisions needed from Meerav

1. **Sealed season.** I have assumed 2026 is the check season. It is the only season with the current cars, so it is the most honest test, but it also means no change can be tuned on it. Confirm or pick another.
2. **Paid data.** FastF1's live timing now needs an F1TV token, and good historical odds may cost money. Decide whether either is worth paying for when Phase 2 reports what is available.
3. **Laptop as the gate machine.** Overnight replay runs assume the laptop is on and free. If that is unreliable, Phase 2 needs another answer.
4. **Branch clean-up list**, when Phase 1 presents it.
