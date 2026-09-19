# Autopilot — unattended slice building for grid-prophet and aperture

Date: 2026-09-19
Status: approved, not yet implemented
Scope: both `grid-prophet` and `aperture`. This spec lives in grid-prophet;
aperture's `AUTOPILOT.md` points here.

## 1. Problem

The owner is away 2026-09-20 through 2026-09-25 (six days). Both projects have
detailed specs describing work that is well-understood but unbuilt. The prompt
to start each slice is the only thing that requires the owner to be at a desk.
Automating that prompt lets the projects keep moving while he is away, provided
the automation cannot do damage he has to undo when he returns.

The owner will check a phone once a day: an email digest and, if something looks
interesting, the PR diff in the Claude mobile app.

## 2. Constraints

- **`main` must not move.** Six days of unattended merges to trunk is not
  reviewable and not cheaply reversible.
- **No CI exists today.** Neither repo has `.github/workflows`. Without it,
  "merge on acceptance criteria" is an LLM reading a diff and deciding.
- **Aperture's remaining work is mostly not automatable.** Most open items
  require real-host validation on installed Windows/macOS hosts.
  `docs/requirements.md` and `AGENTS.md` are explicit that a build, a synthetic
  event, or a CI compile cannot establish runtime behavior, and that missing
  evidence must never be recorded as success. An agent optimizing for a green PR
  is structurally tempted to violate this.
- **grid-prophet's real acceptance gate is slow.** Spec §7 budgets up to two
  hours for a full backtest pass. That cannot gate every PR.
- **Phases are not slices.** Spec §8 Phase 2 is scoped at two weeks. Handing a
  routine "build Phase 2" produces one unreviewable PR.
- **The scheduler must survive a closed laptop.** The desktop app's scheduled
  tasks run only while the app is open and defer a missed fire to next launch,
  which would stack six days of work into the moment of return. In-session cron
  dies with the session. Only cloud routines are machine-independent, and that
  choice costs connector access (§13) and local Tauri tooling (§7.6).

## 3. Non-goals

- Completing Phase 2 of grid-prophet. Three of its six slices are in scope.
- Any aperture validation evidence, macOS work, or host-compatibility claim.
- Running the grid-prophet backtest. That is the Phase 2 acceptance gate and is
  run with the owner present, after his return.
- Self-modifying automation. The routines do not edit their own crons, CI, or
  `AUTOPILOT.md`.

## 4. Branch model

```
main ──────────────────────────────────────────  (prep commit only, 09-19)
  └── auto/queue ← slice-1 ← slice-2 ← slice-3   (all autopilot merges)
        └── auto/<repo>-slice-<n>-<slug>          (per-slice working branches)
```

- `auto/queue` is cut from `main` during setup and is the base for every PR.
- Each slice branches from `auto/queue` HEAD at build time, so slices stack.
- On return, the owner reviews one PR: `auto/queue` → `main`. If it went wrong,
  the branch is deleted and `main` is untouched.

## 5. Routines

Four **cloud** routines, created via `RemoteTrigger` (the `/schedule` mechanism).
Each fire spawns an isolated cloud session with its own git checkout in
Anthropic's infrastructure. Deliberately not the desktop app's scheduled tasks,
which only run while the app is open and would fire six days of backlog at once
on the owner's return; and not in-session `CronCreate`, which dies with the
session.

Environment: `env_01Fs4r4jiPKRTEHieU7uXwUu` (Default, anthropic_cloud).
Cron is **UTC**; the owner is America/New_York (UTC-4 on these dates).

| Routine | Local | Cron (UTC) | Model | Purpose |
|---|---|---|---|---|
| `gp-build` | 09:07 Sun/Tue/Thu | `7 13 * * 0,2,4` | opus | build next grid-prophet slice, open PR |
| `ap-build` | 09:07 Mon/Wed/Fri | `7 13 * * 1,3,5` | opus | build next aperture slice, open PR |
| `autopilot-review` | 14:12 daily | `12 18 * * *` | opus | review open PRs, merge or reject |
| `autopilot-digest` | 19:23 daily | `23 23 * * *` | sonnet | post the day's summary |

Build and review run on Opus: six days of unsupervised spec interpretation is
not where to economize. The digest only summarizes existing artifacts.

The five-hour gap between build and review is the owner's **veto window**.
Silence means proceed; no action is required for the happy path.

Because every fire is a cold session with zero conversation context, the prompt
and `docs/autopilot/queue.md` are the routine's only inputs. Both must be
self-contained.

## 6. The queue

`docs/autopilot/queue.md` in each repo is the routines' only memory across cold
starts. Format:

```markdown
## slice-<n> — <title>
Status: todo
Spec: <file> §<sections>
Acceptance:
- [ ] <criterion, objectively checkable>
- [ ] <criterion>
```

Status values: `todo` → `in-progress` → `in-review` → `merged`, plus terminal
`held` (owner vetoed) and `blocked` (review rejected).

### grid-prophet queue (spec §8 Phase 2, decomposed)

1. **Quali conditioning + `gp race --mode post-quali`** — §3.3, §6 fallback protocol
2. **Sprint events** — sprint scoring, points reconciliation against standings
3. **Reliability split** — mechanical vs incident, grid-position effect (§3.2)

Slices 4–6 of Phase 2 (weather, fitting the overtaking parameter, the Phase 2
backtest and status doc) are out of scope for this window.

### aperture queue (code-only)

1. **Repository/worktree grouping** — R6, currently "Planned" in `docs/goals.md`
2. **Process liveness signal** — named as future work in the Windows follow-up
3. **Search/filter + details pane** — R11

Chosen because each has a real unit-test gate and none requires host evidence.

## 7. Build routine contract

1. Fetch, check out `auto/queue`, pull.
2. Read `docs/autopilot/queue.md`.
   - If every slice is `todo` (the first run), proceed with slice 1.
   - Otherwise, if the highest-numbered non-`todo` slice is not `merged`,
     **halt**. Do not stack on unreviewed, held, or rejected work. Report the
     halt.
   - If no `todo` slice remains, report "queue empty" and exit.
3. Mark the slice `in-progress`; branch `auto/<repo>-slice-<n>-<slug>`.
4. Read the referenced spec sections, plus `AGENTS.md` / `CLAUDE.md` / the
   project's own conventions. These override this spec on style and approach.
5. Implement with tests, following the repo's existing patterns.
6. Run whatever of the repo's CI commands the sandbox supports, and do not push
   red. For grid-prophet that is the full suite after
   `pip install -r requirements.txt`. For aperture the Tauri toolchain may be
   unavailable on a Linux sandbox (`cargo test` needs webkit2gtk and friends);
   run what works, state in the PR body which checks ran locally and which did
   not, and let GitHub Actions be the authoritative gate.
7. Push; open a PR targeting `auto/queue` using the template in §9.
8. Mark the slice `in-review`.

### Ambiguity policy

When the spec does not settle a design question, the routine picks the most
defensible option, **documents the decision and the alternative it rejected** in
the PR body and in `docs/autopilot/decisions.md`, and proceeds. It does not stop
and it does not silently guess. Every such decision surfaces in the digest under
DECIDED FOR YOU.

## 8. Review routine contract

Runs against every open PR targeting `auto/queue`, in slice order.

1. **Veto check.** Any human comment containing `HOLD` (case-insensitive), or a
   changes-requested review from `Meerav29` → mark `held`, skip, report. Never
   override.
2. **CI check.** Not green → request changes, mark `blocked`.
3. **Independent criteria check.** Verify each acceptance criterion **against the
   diff**, not against the PR body's claims about the diff.
4. **Prohibition check** (§10). Any violation → reject, do not merge.
5. **Size check.** Additions plus deletions over 600 lines, excluding lockfiles
   and generated files → mark `held`, do not auto-merge, flag for the owner.
   Large diffs are not reviewable on a phone.
6. **General bar**, in addition to per-slice criteria:
   - New behavior has tests that would fail without the change.
   - No unrelated files touched.
   - Decisions documented where the ambiguity policy required it.
7. All pass → merge to `auto/queue`, mark `merged`. Otherwise request changes
   with specifics and mark `blocked`.

The review routine writes no implementation code. It is a separate agent from
the build routine and does not inherit its reasoning.

## 9. PR body template

Required sections. A missing section is a review rejection.

```markdown
## What this slice does
## Acceptance criteria
<each criterion, with the command run and its output as evidence>
## Decisions made without you
<each, with the alternative rejected and why — or "none">
## Deliberately not included
## How to revert
```

## 10. Prohibitions

Binding on all routines:

- Never push to `main`. Never force-push. Never delete another slice's branch.
- Never modify `.github/workflows/**`, `AUTOPILOT.md`, or the cron definitions.
- Never modify or discard another task's in-flight work.
- **aperture only:** never write to `docs/validation/**`, and never assert host
  compatibility, runtime behavior, or validation status. Unit tests and builds
  are evidence of code correctness only. This mirrors `AGENTS.md`.
- Never fabricate evidence. If a criterion cannot be checked, say so and let the
  slice be rejected.

## 11. Halting

- Two consecutive rejected slices in a repo → that build routine halts for the
  remainder of the window; the digest says so.
- Previous slice not `merged` → halt (§7.2).
- Queue exhausted → idle, report.

Halting is the correct outcome. A halted routine leaves reviewable work; a
grinding one leaves a mess.

## 12. CI

Added to `main` during setup. Runs on PRs targeting `auto/queue` and `main`.

- **grid-prophet:** `pytest tests` — measured 2026-09-19 at 101 passed in 2m03s.
- **aperture:** `npm run build` (includes `tsc`) and
  `cargo test --manifest-path src-tauri/Cargo.toml --locked` — 22 unit tests
  plus one helper CLI integration test as of the 2026-09-08 follow-up. The
  Ubuntu runner needs Tauri's Linux system dependencies installed before
  `cargo test`; this is setup work, and until it is green the aperture queue
  does not start.

CI is the authoritative gate for aperture, since the build sandbox may not be
able to run `cargo test` at all (§7.6).

The grid-prophet backtest is **not** in CI (§2). Slices 1–3 therefore merge on
unit tests plus reviewer judgment, not on Phase 2's real "done when" bar from
spec §8. This is a known soft spot and the main reason `auto/queue` exists.

## 13. Digest

19:00 daily. Decisions-and-problems only; routine successes get one line.

```
Autopilot — <day> <date>

MERGED (n)          one line per PR
DECIDED FOR YOU (n) each decision, where it lives, how isolated
BLOCKED / HELD (n)  what and why, what it needs
TOMORROW            next slice and repo
```

Delivery: a comment on the tracking issue (§14), which reaches the owner's inbox
via GitHub notification. **Not** email via connector — cloud routines cannot use
MCP servers configured in Claude Code, only claude.ai connectors, and this
session's token cannot enumerate those. Choosing the GitHub channel avoids
depending on something unverifiable from here, and keeps the run log and the
digest in the same place.

## 14. Setup commit

One commit to `main` in each repo, by the owner and Claude together on 09-19:

- `AUTOPILOT.md` — read-me-first. What ran, when, where it lives, what to review
  before anything else, and how to tear it down.
- `.github/workflows/ci.yml`
- `docs/autopilot/queue.md`
- `docs/autopilot/decisions.md` (empty, with a header)

Also created at setup: one pinned tracking issue per repo, titled
`Autopilot run log — 2026-09-20 to 09-25`. It is the digest's fallback channel
(§13) and the place halts and rejections are recorded regardless of channel.

Then `auto/queue` is cut from `main` and the four crons are created.

## 14.1 Smoke test — not optional

Several assumptions cannot be verified from this machine: that the cloud
environment can clone both repos, install dependencies, **push a branch**, and
open a PR as the owner. Nothing in the design works if push access is absent,
and discovering that on Sunday morning wastes the whole window.

So before the owner leaves, `gp-build` is fired manually (`RemoteTrigger`
`action: "run"`) and watched end to end. Its output is grid-prophet slice 1 — a
real slice, not a throwaway — reviewed together and merged by hand. The Sunday
fire then picks up slice 2.

This costs one slice's worth of waiting and buys proof that the loop closes.
If the smoke test fails, the window is spent fixing it rather than trusting it,
and nothing is scheduled until it passes.

## 15. Open items

- Whether the Ubuntu runner can build and test the Tauri crate once the system
  dependencies are installed. Verified during setup; the aperture queue does not
  start until CI is green (§12).
- Whether the cloud sandbox can run `cargo test` for pre-flight. Assumed no;
  §7.6 does not depend on it.

Resolved during design: connector-based email is unavailable to cloud routines
(§13). Timezone is America/New_York, confirmed UTC-4 on 2026-09-19.

## 16. Teardown and return

Early return: **disable** the routines (`RemoteTrigger` update with
`enabled: false`) and the queue stops where it is. Nothing is left half-merged,
because a slice is either merged to `auto/queue` or still an open PR. Routines
cannot be deleted from Claude Code — permanent deletion is the web UI at
<https://claude.ai/code/routines>. Disabling is sufficient to stop all work.

On return, in order:

1. Read `AUTOPILOT.md` and `docs/autopilot/decisions.md` first.
2. Review the single `auto/queue` → `main` PR per repo.
3. Run the grid-prophet backtest — Phase 2's actual acceptance gate.
4. Resume aperture's validation lane, which autopilot did not touch.
