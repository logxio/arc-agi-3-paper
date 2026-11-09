"""A3b: one panel per dataset: completion hazard in 20-action bins with a 95% game-bootstrap band.
Appendix figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import COLORS, DATASETS, LABELS, ROOT, SEED, attempts, by_game, cond_pass, km, load

W = 20
GRID = np.arange(0, 300, W)
MIN_RISK = 10
NB = 500


def curve(rows):
    d = np.array([r[2] for r in rows], float)
    e = np.array([r[3] for r in rows], bool)
    keep = ~((~e) & (d == 0))
    d, e = d[keep], e[keep]
    out = []
    for n in GRID:
        _, rk = km(d, e, [n])
        h = cond_pass(d, e, n, W)
        out.append(h if (h is not None and rk[0] >= MIN_RISK) else np.nan)
    return np.array(out, float), d, e


fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
for ax, name in zip(axes.flat, DATASETS):
    runs = load(name)
    hz, d, e = curve(attempts(runs))
    groups = list(by_game(runs).values())
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(NB):
        idx = rng.integers(0, len(groups), len(groups))
        boots.append(curve(attempts([r for i in idx for r in groups[i]]))[0])
    b = np.array(boots)
    with np.errstate(all="ignore"):
        lo = np.nanpercentile(b, 2.5, axis=0)
        hi = np.nanpercentile(b, 97.5, axis=0)
    x = GRID + W / 2
    m = ~np.isnan(hz)
    ax.fill_between(x[m], lo[m], hi[m], color=COLORS[name], alpha=0.18, lw=0)
    ax.plot(x[m], hz[m], color=COLORS[name], lw=2, marker="o", ms=3)
    ax.set_title(f"{LABELS[name]} ({len(d)} attempts)", fontsize=9, loc="left")
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0, 300)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("actions spent on the level")
for ax in axes[:, 0]:
    ax.set_ylabel("P(clear in next 20 actions)")
fig.tight_layout()
fig.savefig(ROOT / "paper" / "figures" / "hazard_by_dataset.png", dpi=170)
print("written")
