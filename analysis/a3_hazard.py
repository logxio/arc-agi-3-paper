"""A3: completion-hazard curves. Every arrival at a level is one attempt; cleared levels
are events (duration = actions on the level), the final unsolved level is right-censored.
Kaplan-Meier survival, binned conditional pass rates, game-bootstrap intervals, main figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (COLORS, DATASETS, LABELS, ROOT, attempts, by_game, dump, km,
                    load, cond_pass, BOOT, SEED)

W = 20                      # bin width in actions
POINTS = [0, 20, 40, 80, 160]
GRID = np.arange(0, 401, W)
MIN_RISK = 10               # do not plot bins with fewer attempts at risk
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def hazards(rows, points, w):
    d = [r[2] for r in rows]
    e = [r[3] for r in rows]
    return [cond_pass(d, e, n, w) for n in points]


def boot_hazard(runs, points, w, unit="actions"):
    groups = list(by_game(runs).values())
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(groups), len(groups))
        rows = attempts([r for i in idx for r in groups[i]], unit)
        h = hazards(rows, points, w)
        vals.append([np.nan if x is None else x for x in h])
    v = np.array(vals, float)
    return np.nanpercentile(v, 2.5, axis=0), np.nanpercentile(v, 97.5, axis=0)


res = {}
curves = {}
for name in DATASETS:
    runs = load(name)
    rows = attempts(runs)
    d = np.array([r[2] for r in rows], float)
    e = np.array([r[3] for r in rows], bool)
    keep = ~((~e) & (d == 0))            # censored with zero actions carries no information
    d, e = d[keep], e[keep]
    S, risk = km(d, e, GRID)
    h_pts = hazards(list(zip([0] * len(d), [0] * len(d), d, e)), POINTS, W)
    lo, hi = boot_hazard(runs, POINTS, W)
    cleared = d[e]
    base_ratio = [r["a"][i] / r["base"][i] for r in runs for i in range(r["k"]) if r["a"][i] > 0]
    res[name] = {
        "attempts": int(len(d)), "events": int(e.sum()),
        "cond_pass_next20": {str(n): [h, l, u] for n, h, l, u in zip(POINTS, h_pts, lo, hi)},
        "S": {str(n): float(s) for n, s in zip(GRID, S) if n in (40, 80, 160)},
        "median_cleared_actions": float(np.median(cleared)) if len(cleared) else None,
        "median_ratio_to_human": float(np.median(base_ratio)) if base_ratio else None,
        "frac_events_after_80": float(np.mean(cleared > 80)) if len(cleared) else None,
    }
    # binned hazard curve for the figure
    hz = []
    for n in GRID[:-1]:
        _, rk = km(d, e, [n])
        h = cond_pass(d, e, n, W)
        hz.append(h if (h is not None and rk[0] >= MIN_RISK) else np.nan)
    fine = np.arange(0, 401)
    S_fine, risk_fine = km(d, e, fine)
    curves[name] = (S, risk, np.array(hz, float), S_fine, risk_fine)
    fmt = lambda x: "  -  " if x is None else f"{x:.3f}"
    print(f"{name:18s} att={len(d):4d} ev={int(e.sum()):4d} "
          + " ".join(f"h{n}={fmt(h)}[{l:.2f},{u:.2f}]" for n, h, l, u in zip(POINTS, h_pts, lo, hi))
          + f" medClr={res[name]['median_cleared_actions']} ratio={res[name]['median_ratio_to_human']}")

# seconds version (runs with per-level timing)
SEC_W = 120
SEC_POINTS = [0, 120, 300, 600, 1200]
SEC_GRID = np.arange(0, 3601, SEC_W)
sec = {}
sec_curves = {}
for name in DATASETS:
    runs = load(name)
    rows = [r for r in attempts(runs, "seconds") if r[3] or r[2] > 0]
    if len(rows) < 30:
        continue
    d = np.array([r[2] for r in rows]); e = np.array([r[3] for r in rows])
    h_pts = hazards(rows, SEC_POINTS, SEC_W)
    lo, hi = boot_hazard(runs, SEC_POINTS, SEC_W, "seconds")
    sec[name] = {"attempts": len(rows), "median_cleared_seconds": float(np.median(d[e])) if e.any() else None,
                 "cond_pass_next120s": {str(n): [h, l, u] for n, h, l, u in zip(SEC_POINTS, h_pts, lo, hi)}}
    hz = []
    for n in SEC_GRID[:-1]:
        _, rk = km(d, e, [n])
        h = cond_pass(d, e, n, SEC_W)
        hz.append(h if (h is not None and rk[0] >= MIN_RISK) else np.nan)
    sec_curves[name] = np.array(hz, float)
    fmt = lambda x: "  -  " if x is None else f"{x:.3f}"
    print(f"[sec] {name:18s} att={len(rows):4d} medClr={sec[name]['median_cleared_seconds']:.0f}s "
          + " ".join(f"h{n}s={fmt(h)}" for n, h in zip(SEC_POINTS, h_pts)))

dump("a3_hazard", {"actions": res, "seconds": sec, "bin_actions": W, "bin_seconds": SEC_W})

# ---------------------------------------------------------------- main figure
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = GRID[:-1] + W / 2
for name in DATASETS:
    S, risk, hz, S_fine, risk_fine = curves[name]
    lw = 2.6 if name in ("kaggle_v2", "tufa_qwen36_27b", "api_qwen38flash") else 1.4
    m = risk_fine >= MIN_RISK
    fine = np.arange(0, 401)
    axes[0].step(fine[m], S_fine[m], where="post", color=COLORS[name], lw=lw, label=LABELS[name])
    axes[1].plot(x, hz, color=COLORS[name], lw=lw, marker="o", ms=3.5, label=LABELS[name])
axes[0].set_xlabel("actions spent on the level since reaching it")
axes[0].set_ylabel("share of attempts not yet cleared")
axes[0].set_title("(a) Kaplan-Meier survival of a level attempt", loc="left", fontsize=10)
axes[0].set_ylim(0, 1.02)
axes[1].set_xlabel("actions already spent on the level without clearing it")
axes[1].set_ylabel(f"P(clear within next {W} actions)")
axes[1].set_title(f"(b) Conditional completion rate, {W}-action bins", loc="left", fontsize=10)
axes[1].set_ylim(0, 0.5)
for ax in axes:
    ax.set_xlim(0, 400)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.axvspan(40, 80, color="#e6e5e0", alpha=0.5, lw=0)
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "hazard.png", dpi=200)

# per-dataset seconds figure for the appendix
fig, ax = plt.subplots(figsize=(6.5, 3.8))
xs = SEC_GRID[:-1] + SEC_W / 2
for name, hz in sec_curves.items():
    ax.plot(xs / 60, hz, color=COLORS[name], lw=1.8, marker="o", ms=3, label=LABELS[name])
ax.set_xlabel("minutes already spent on the level without clearing it")
ax.set_ylabel(f"P(clear within next {SEC_W // 60} min)")
ax.set_xlim(0, 60); ax.set_ylim(0, 0.6)
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "hazard_seconds.png", dpi=200)

# cover image: survival of a level attempt for three reference runs, 2:1, large type
fig, ax = plt.subplots(figsize=(10, 5))
fine = np.arange(0, 401)
ends = {}
for name in ("api_qwen38flash", "kaggle_v2", "tufa_qwen36_27b"):
    S_fine, risk_fine = curves[name][3], curves[name][4]
    m = risk_fine >= MIN_RISK
    ax.step(fine[m], S_fine[m], where="post", color=COLORS[name], lw=4)
    x_end = fine[m][-1]
    ends[name] = (x_end, S_fine[m][-1])
labels = {"api_qwen38flash": "Qwen3.8-Flash", "kaggle_v2": "our Kaggle notebook", "tufa_qwen36_27b": "Qwen3.6-27B"}
for name, (x_end, y_end) in ends.items():
    ax.text(x_end + 6, y_end, labels[name], color=COLORS[name], fontsize=15, va="center", fontweight="bold")
ax.axvspan(0, 80, color="#e6e5e0", alpha=0.7, lw=0)
ax.text(40, 1.04, "first 80 actions:\n88-100% of all clears", ha="center", va="bottom", fontsize=13, color="#52514e")
ax.set_xlim(0, 420)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("actions spent on a level since reaching it", fontsize=15)
ax.set_ylabel("share of level attempts\nnot yet cleared", fontsize=15)
ax.tick_params(labelsize=12)
ax.set_title("Frontier Lock: ARC-AGI-3 agents clear a level fast or not at all",
             loc="left", fontsize=18, fontweight="bold", pad=14)
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
fig.tight_layout()
fig.savefig(FIG / "cover.png", dpi=160)
print("figures written")
