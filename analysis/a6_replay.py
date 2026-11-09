"""A6: replay "stop any level after N actions" on recorded traces (a replay, not a live run).

A run stopped at N keeps the levels it cleared in at most N actions each, up to the first
level that needed more than N (or the frontier once it reached N). Released actions are
those the run spent after that point. Also: the single per-game action cap that releases
the same share of actions."""
import numpy as np

from common import DATASETS, dump, game_bootstrap, load, score

NS = (50, 80, 150)


def stop_at(r, n):
    kept = 0
    for i in range(r["k"]):
        if r["a"][i] > n:
            return i, kept + n
        kept += r["a"][i]
    if r["k"] < r["L"] and r["a"][r["k"]] > n:
        return r["k"], kept + n
    return r["k"], sum(r["a"])


def cap_game(r, cap):
    """Global cap: the run ends after cap actions in total."""
    k, used = 0, 0
    for i in range(r["k"]):
        if used + r["a"][i] > cap:
            return k, cap
        used += r["a"][i]; k += 1
    return k, min(sum(r["a"]), cap)


def replay(runs, rule):
    s = kept_act = 0.0
    for r in runs:
        k, used = rule(r)
        s += score(r["L"], k, r["a"], r["base"])
        kept_act += used
    total = sum(sum(r["a"]) for r in runs)
    return s / len(runs), 1 - kept_act / total


res = {}
for name in DATASETS:
    runs = load(name)
    base = np.mean([score(r["L"], r["k"], r["a"], r["base"]) for r in runs])
    row = {"mean_score": base, "stop": {}}
    line = f"{name:18s} base {base:6.2f}"
    for n in NS:
        sc, rel = replay(runs, lambda r: stop_at(r, n))
        kept = lambda rs: (replay(rs, lambda r: stop_at(r, n))[0] /
                           np.mean([score(r["L"], r["k"], r["a"], r["base"]) for r in rs])
                           if np.mean([r["score"] for r in rs]) > 0 else np.nan)
        p, lo, hi = game_bootstrap(runs, kept, n=1000) if base > 0 else (np.nan, np.nan, np.nan)
        # global cap releasing the same share of actions
        caps = np.arange(1, max(sum(r["a"]) for r in runs) + 1)
        rels = np.array([replay(runs, lambda r, c=c: cap_game(r, c))[1] for c in caps[::5]])
        c_match = caps[::5][np.argmin(np.abs(rels - rel))]
        gsc, grel = replay(runs, lambda r: cap_game(r, c_match))
        row["stop"][str(n)] = {"mean_score": sc, "score_kept": sc / base if base else None,
                               "score_kept_ci": [lo, hi], "actions_released": rel,
                               "global_cap": int(c_match), "global_cap_score_kept": gsc / base if base else None,
                               "global_cap_released": grel}
        line += (f" | N={n}: {sc:5.2f} kept {sc / base if base else 0:4.0%} [{lo:.0%},{hi:.0%}] "
                 f"released {rel:4.0%} (global cap {c_match}: kept {gsc / base if base else 0:4.0%})")
    res[name] = row
    print(line)
dump("a6_replay", res)
