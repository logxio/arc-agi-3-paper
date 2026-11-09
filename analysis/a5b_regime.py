"""A5b: sweep the per-game time limit under the live-test setting.

The simulation uses Qwen3.8-Flash endpoint traces, 8 slots, 25 games x 2
passes and an 80-action cutoff. Both live rounds are overlaid; confirmation
used 5 passes per arm and is not a direct 2-pass simulation check.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a5_theory import simulate
from a8_confirmation import DATA, paired, validate
from common import ROOT, dump, load

runs = load("api_qwen38flash")
lat = {id(r): r["t_first"] - (r["t_final"] - 1800.0) for r in runs if r["state"] == "gave_up"}
med = float(np.median(list(lat.values())))
latency = lambda r: min(max(lat.get(id(r), med), 0.0), 600.0)

LIMITS = [5, 7.5, 10, 12.5, 15, 20, 25, 30]
res = {}
for m in LIMITS:
    o = simulate(runs, (None, 80), "slots", K=8, passes=2, reps=1000, limit=60.0 * m, latency=latency)
    res[str(m)] = o
    d = o["80"]["d_levels"]
    q = o["dyn_only"]["d_levels"]
    print(f"{m:5.1f} min: control {o['none']['levels'][0]:.3f} levels, d_levels {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] "
          f"schedule-only {q[0]:+.3f} [{q[1]:+.3f},{q[2]:+.3f}] "
          f"cutoff effect {o['80']['cutoff_effect_levels'][0]:+.3f} [{o['80']['cutoff_effect_levels'][1]:+.3f},{o['80']['cutoff_effect_levels'][2]:+.3f}] "
          f"d_score {o['80']['d_score'][0]:+.2f}  stopped {o['80']['plays_stopped_by_cutoff']:.1f}/50 "
          f"slow control clears {o['80']['control_plays_clearing_after_c']:.1f}/50")
dump("a5b_regime", res)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.array(LIMITS)
d = np.array([res[str(m)]["80"]["d_levels"] for m in LIMITS])
ax.fill_between(x, d[:, 1], d[:, 2], color="#2a78d6", alpha=0.18, lw=0, label="simulation, 95% interval")
ax.plot(x, d[:, 0], color="#2a78d6", lw=2, marker="o", ms=4, label="stop at 80 + dynamic schedule (simulated)")
q = np.array([res[str(m)]["dyn_only"]["d_levels"] for m in LIMITS])
ax.plot(x, q[:, 0], color="#eb6834", lw=1.6, ls="--", marker="o", ms=3, label="dynamic schedule only (simulated)")
ax.errorbar([15], [0.140], yerr=[[0.140], [0.160]], fmt="s", color="#0b0b0b", ms=6, capsize=3,
            label="live screening round (2 runs x 25 games)")
confirmation_rows = [json.loads(line) for line in DATA.read_text().splitlines() if line]
confirmation = paired(validate(confirmation_rows), "k")
estimate = confirmation["difference"]
lower, upper = confirmation["ci95"]
ax.errorbar([30], [estimate], yerr=[[estimate - lower], [upper - estimate]],
            fmt="D", color="#9c2c3a", ms=6, capsize=3,
            label="live confirmation (5 runs x 25 games)")
ax.axhline(0, color="#8a8984", lw=0.8)
ax.set_xlabel("per-game time limit (minutes)")
ax.set_ylabel("completed levels per game,\nstop-and-transfer minus control")
ax.set_xlim(4, 32)
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(ROOT / "paper" / "figures" / "regime.png", dpi=200)
