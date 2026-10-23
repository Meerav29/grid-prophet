"""Pre-weekend forecast for 2026 R18 (US GP) before any R18 session exists.

`forecast_weekend` reads entrants and the circuit from the round's own rows,
so this adds stub R18 Q rows (R17's entry list, outcome columns empty) to the
frame passed to the forecast only. The fit never sees them.
Run from the forecast-data worktree with PYTHONPATH=src.
"""
import os
import sys

import numpy as np
import pandas as pd

import cli

SEASON, ROUND, ENTRY_FROM = 2026, 18, 17

fit_dir = open(cli.LATEST_POINTER).read().strip()
idata, data, reliability_data, _ = cli.load_fit(fit_dir)
dr = cli._load_driver_rounds()
assert dr[(dr.season == SEASON) & (dr["round"] == ROUND)].empty, "R18 rows exist; use the CLI"

src = dr[(dr.season == SEASON) & (dr["round"] == ENTRY_FROM) & (dr.session_type == "R")]
stub = pd.DataFrame({c: np.nan for c in dr.columns}, index=range(len(src)))
stub = stub.astype(dr.dtypes.to_dict(), errors="ignore")
stub["season"], stub["round"] = SEASON, ROUND
stub["event_name"], stub["circuit_type"] = "United States Grand Prix", "mixed"
stub["session_type"] = "Q"
for c in ("driver", "abbreviation", "team"):
    stub[c] = src[c].to_numpy()
dr = pd.concat([dr, stub], ignore_index=True)

forecast, sprint = cli.forecast_weekend(idata, data, reliability_data, dr, cli._load_circuits(),
                                        SEASON, ROUND, n_trials=50_000, seed=0, mode="pre")
assert sprint is None
out = os.path.join(sys.argv[1] if len(sys.argv) > 1 else fit_dir,
                   f"race_forecast_{SEASON}_{ROUND}_pre.csv")
forecast.to_csv(out, index=False)
print(forecast.to_string(index=False))
print(f"fit_dir={fit_dir}\nSaved -> {out}")
