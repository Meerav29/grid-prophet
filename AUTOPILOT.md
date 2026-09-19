# Autopilot ran while you were away

**Window: 2026-09-20 → 2026-09-25. Read this before you touch anything else.**

While you were away, scheduled cloud routines built slices from
[docs/autopilot/queue.md](docs/autopilot/queue.md), opened PRs, and merged the
ones that passed CI and review.

**None of it is on `main`.** Everything landed on the `auto/queue` branch.
`main` has only the setup commit that added this file, CI, and the queue.

## Review this in order

1. **[docs/autopilot/decisions.md](docs/autopilot/decisions.md)** — every design
   call a routine made on your behalf because the spec did not settle it, with
   the alternative it rejected. This is the highest-value page. Read it first.
2. **The `auto/queue` → `main` pull request.** One diff, all merged slices.
3. **The tracking issue** `Autopilot run log — 2026-09-20 to 09-25` — daily
   digests, halts, and anything blocked or vetoed.
4. **Any still-open PR** against `auto/queue` — these were rejected or held and
   never merged.

## The thing to be skeptical about

Slices merged on the unit suite (about two minutes) plus reviewer judgment.
They did **not** merge on Phase 2's actual "done when" bar from
[the v2 spec](docs/grid-prophet-v2-spec.md) §8, which needs the full rolling
backtest — up to two hours per §7, too slow for CI.

So it is entirely possible that three slices of plausible, well-tested code do
not improve log loss at all. **Run the backtest before you believe any of it.**
That is the real gate, and it is why this work is on a branch.

If the backtest says no, delete `auto/queue`. `main` never moved.

## Stopping it

Routines are disabled from Claude Code (`RemoteTrigger` update,
`enabled: false`). Permanent deletion is the web UI at
<https://claude.ai/code/routines> — Claude Code cannot delete them.

## How it was set up

[docs/superpowers/specs/2026-09-19-autopilot-design.md](docs/superpowers/specs/2026-09-19-autopilot-design.md)
is the design, including the branch model, the review contract, the
prohibitions, and the reasoning behind each. The same design governs the
`aperture` repo, which ran on the alternating days.
