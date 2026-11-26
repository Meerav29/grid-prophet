# Season-level gate (spec §7, Phase 3): 2026-11-26

**Result: fail.** Keep the legacy regression (`confidence.py`, the pkl) in place. Don't retire it.

Run on `main@93aae59`:

```bash
PYTHONPATH=src python -m cli backtest --seasons 2022-2025 --season-level --method nuts --n-trials 5000 --out-dir outputs/season_bt
```

- 12 of 12 cutoffs scored, with no failures (`gate_valid: true`).
- It took 227 s in total, at 15–24 s per fit.
- The per-season numbers are in `2026-11-26-season-level-metrics.csv`, next to this file.

## Against the spec's bar

The spec's bar: "v2 matches or beats legacy Spearman at every cutoff, interval coverage is within 70–90%, and title-odds Brier is documented."

| Cutoff | Constructor Spearman, v2 | Legacy | 80% coverage, drivers | 80% coverage, constructors | Title Brier, drivers | Title Brier, constructors |
|---|---|---|---|---|---|---|
| R2 | 0.819 | **0.855** | 0.50 | 0.53 | 0.352 | 0.561 |
| R6 | 0.895 | 0.895 | 0.56 | 0.53 | 0.257 | 0.409 |
| R12 | **0.957** | 0.908 | 0.66 | 0.50 | 0.107 | 0.230 |
| Mean | 0.890 | 0.886 | | | | |

- **Spearman: fail at R2.**
  - Legacy is ahead by 0.036. It wins 3 of the 4 seasons there, and 2023 is the big miss (0.697 against 0.857).
  - R6 is a tie (0.8947 against 0.8946). v2 is clearly ahead at R12.
  - Legacy here trains only on earlier seasons, as decided in slice 9, so it scores below the reference 0.88.
- **Coverage: fail at every cutoff.**
  - The 80% points ranges held only 30–80% of the time, with means of 50–66%.
  - The ranges are too narrow: the model is overconfident about where each team ends the season.
  - 2024 is the worst season, at 30–40% coverage and a constructors' title Brier of 1.47 at R2 and R6. That was the season McLaren overtook Red Bull mid-season.
- **Title Brier:** documented above.

## Notes

- The harness's `v2_beats_legacy_spearman: true` compares the **mean** over all cutoffs. The spec asks for every cutoff, so the flag passes a run that the spec fails. Fix the flag, or read the per-cutoff numbers instead of it.
- What this points at, in the roadmap's Phase 3 order:
  - Item 2 (calibration and the noise structure) is the direct fix for the coverage gap.
  - Item 5 (recent-form weighting, letting upgrades show up faster) fits the 2024 miss.
- Re-run this gate on the Phase 2 sandbox once it exists. Any change that claims to fix the gate needs this run as its "before".
