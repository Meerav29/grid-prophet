import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
f = pd.read_csv(f"{d}/race_forecast_2026_19_pre.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
a = f.sort_values("p_win").tail(10)
ax[0].barh(a.driver, a.p_win * 100, color="#d62728")
ax[0].set(title="Win probability (%)", xlabel="%")
b = f.sort_values("exp_points")
ax[1].barh(b.driver, b.exp_points, color="#1f77b4")
ax[1].set(title="Expected points", xlabel="points")
fig.suptitle("2026 R19 Mexico City GP: pre-weekend forecast (fit through R18, grid simulated)")
fig.tight_layout()
fig.savefig(f"{d}/forecast_2026_19_pre.png", dpi=130)
