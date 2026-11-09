"""A5: the cutoff theory and a trace-resampling simulation.

Ideal model. With a fixed per-level cutoff c, an attempt costs min(T, c) actions and clears
iff T <= c. If a fresh attempt is always available, the long-run clearing rate is
    R(c) = P(T <= c) / E[min(T, c)]
(Luby et al. 1993 for Las Vegas restarts; Charnov's marginal value theorem in discrete form:
at the optimum the conditional clearing rate of the current attempt equals R). T's distribution
is the Kaplan-Meier estimate from A3.

Simulation. 25 games (or 25 x passes runs), same total action budget, the per-game budget b
of the control arm. Each replicate resamples games with replacement and, for every game,
draws one recorded run of it: its cleared levels keep their recorded action counts; its
unsolved frontier level (censored at c0) gets T drawn from the dataset's pooled KM
conditional on T > c0; levels no recorded run of this game reached get T drawn from the
pooled KM of all attempts at that level index or deeper. Beyond the last recorded clearing
the KM curve is flat, so such a level is never cleared: extra time on a stuck level buys
nothing unless the traces show late clears. Both arms use the same draws.

Two ways the released budget can be used:
  shared   one engine, all games start together, throughput split equally among running
           games (the Kaggle notebook): a stopped game's share goes to the survivors at once.
  slots    K concurrent games on a hosted endpoint, fixed speed per game, games queue.
           Control: fixed per-game budget b. Intervention: the dynamic grant of evidence/code
           (remaining / ceil(unfinished / K) when a game takes a slot; after all games have
           started, survivors run to the common deadline). A freed slot is used only by a
           game that has not started yet; nobody runs faster.

Conservative relative to the ideal model: the number of games is finite, a stopped game is
never replayed, released budget can only go to games that are still running or queued,
and deeper levels are drawn from the (harder) deep-level pool.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (BOOT, COLORS, DATASETS, LABELS, ROOT, SEED, attempts, by_game, dump,
                    km, load, score)

INF = float("inf")
CUTS = (None, 50, 80, 150)
MIN_RISK = 10
REPS = 2000


# ---------------------------------------------------------------- ideal model

def km_curve(rows, tmax):
    d = np.array([r[2] for r in rows], float)
    e = np.array([r[3] for r in rows], bool)
    keep = ~((~e) & (d == 0))
    d, e = d[keep], e[keep]
    grid = np.arange(0, tmax + 1)
    S, risk = km(d, e, grid)
    return S, risk


def rate_curve(S, risk):
    """R(c) for c = 1..cmax where at least MIN_RISK attempts are still observed."""
    cmax = int(np.max(np.nonzero(risk >= MIN_RISK)[0])) if np.any(risk >= MIN_RISK) else 1
    cs = np.arange(1, cmax + 1)
    cum = np.cumsum(S)                      # cum[c-1] = sum_{n<c} S(n)
    R = (1 - S[cs]) / cum[cs - 1]
    return cs, R


def ideal(runs):
    rows = attempts(runs)
    S, risk = km_curve(rows, 1000)
    cs, R = rate_curve(S, risk)
    i = int(np.argmax(R))
    out = {"c_star": int(cs[i]), "R_star": float(R[i]), "R_cmax": float(R[-1]), "cmax": int(cs[-1])}
    for c in (50, 80, 150):
        out[f"R_{c}"] = float(R[c - 1]) if c <= cs[-1] else None
    # marginal value check: conditional clearing rate per action at c*, averaged over 20 actions
    c = cs[i]
    h = (1 - S[min(c + 20, len(S) - 1)] / S[c]) / 20 if S[c] > 0 else float("nan")
    out["hazard_at_c_star"] = float(h)
    return out, cs, R


# ---------------------------------------------------------------- simulation

class Sampler:
    """Inverse-CDF draws from a KM curve (integer actions), optionally conditional on T > c0."""

    def __init__(self, rows, tmax=2000):
        self.S = km_curve(rows, tmax)[0] if rows else np.ones(tmax + 1)

    def draw(self, rng, c0=0):
        S = self.S
        c0 = int(min(c0, len(S) - 1))
        if S[c0] <= 0:
            return INF
        target = (1 - rng.random()) * S[c0]
        idx = np.nonzero(S[c0 + 1:] <= target)[0]
        return float(c0 + 1 + idx[0]) if len(idx) else INF


def build(runs):
    rows = attempts(runs)
    pooled = Sampler(rows)
    depth_rows = defaultdict(list)
    for r in rows:
        depth_rows[r[1]].append(r)
    max_depth = max(depth_rows)
    deep = {lv: Sampler([r for d, rs in depth_rows.items() if d >= min(lv, max_depth) for r in rs])
            for lv in range(0, 11)}
    return pooled, deep


def run_end(r):
    return r.get("t_final") or r.get("t_last")


class Play:
    """One simulated play of a game, built from one recorded run.

    Time runs from the game's start. The recorded part keeps the run's own clock: level i is
    cleared at its recorded time, actions inside a level are spread evenly over its span.
    Past the recorded end the run continues at its own average action rate; its unsolved
    level needs T actions drawn from the pooled KM given T > (actions already spent), deeper
    levels draw T from the pooled KM of attempts at that depth or deeper.
    """

    def __init__(self, r, pooled, deep, rng, latency=0.0):
        self.r = r
        L, k, a = r["L"], r["k"], r["a"]
        t0 = r["t_first"]
        self.rec_end = latency + max(run_end(r) - t0, 1.0)
        self.rate = max(sum(a), 1) / max(run_end(r) - t0, 1.0)
        self.spans = []                       # (start, end, actions, cleared)
        prev = latency
        for i in range(k):
            ci = latency + max(r["t"][i] - t0, 0.0)
            self.spans.append((prev, max(ci, prev), a[i], True))
            prev = max(ci, prev)
        self.front = None
        if k < L:
            self.spans.append((prev, self.rec_end, a[k], False))
            self.front = pooled.draw(rng, a[k])
        self.deeper = [deep[min(i, 10)].draw(rng) for i in range(k + 1, L)]

    def path(self, c):
        """Clear times and action counts under cutoff c, and the natural end time."""
        clears = []
        for (s, e, n, done) in self.spans:
            if done:
                if c is not None and n > c:
                    return clears, s + (c / n) * (e - s)
                clears.append((e, n))
            else:
                if c is not None and n >= c:
                    return clears, (s + (c / n) * (e - s)) if n else s
                t = self.rec_end
                T = self.front
                if c is not None and T > c:
                    return clears, t + (c - n) / self.rate
                if T == INF:
                    return clears, INF
                t += (T - n) / self.rate
                clears.append((t, int(T)))
                for T in self.deeper:
                    if c is not None and T > c:
                        return clears, t + c / self.rate
                    if T == INF:
                        return clears, INF
                    t += T / self.rate
                    clears.append((t, int(T)))
                return clears, t
        return clears, clears[-1][0] if clears else 0.0

    def result(self, c, budget):
        clears, _ = self.path(c)
        got = [n for (t, n) in clears if t <= budget + 1e-9]
        k = len(got)
        a = got + [0] * (self.r["L"] - k)
        return k, score(self.r["L"], k, a, self.r["base"])

    def slow_clear(self, c, budget):
        """Control play that clears a level needing more than c actions within budget."""
        clears, _ = self.path(None)
        return any(n > c and t <= budget for (t, n) in clears)


def shared_alloc(ends, pool):
    """Equal processor sharing of one engine = water-filling: play i gets min(end_i, lam)."""
    ends = np.asarray(ends, float)
    finite = np.sort(ends[np.isfinite(ends)])
    n = len(ends)
    used, prev, j = 0.0, 0.0, 0
    while j < len(finite):
        cost = (finite[j] - prev) * (n - j)
        if used + cost >= pool:
            break
        used += cost
        prev = finite[j]
        j += 1
    lam = prev + (pool - used) / (n - j) if n - j > 0 else prev
    return np.minimum(ends, lam)


def slot_alloc(ends, budgets, K, dynamic, H):
    """Time each play gets on K slots of fixed speed. Control: its own fixed budget.
    Intervention: the live runner's grant, remaining / ceil(unfinished / K) at slot time,
    and after every play has started, all survivors run to the common deadline H."""
    n = len(ends)
    if not dynamic:
        return np.minimum(ends, budgets)
    got = np.zeros(n)
    start = np.zeros(n)
    deadline = np.zeros(n)
    state = {"unfinished": n, "waiting": n}
    running = []
    nxt = 0

    def launch(g, t):
        waves = max(1, math.ceil(state["unfinished"] / K))
        start[g] = t
        deadline[g] = min(H, t + max(0.0, H - t) / waves)
        state["waiting"] -= 1
        if state["waiting"] == 0:
            deadline[:] = H
        heapq.heappush(running, (min(t + ends[g], deadline[g]), g))

    while nxt < n and len(running) < K:
        launch(nxt, 0.0)
        nxt += 1
    while running:
        t, g = heapq.heappop(running)
        true_end = min(start[g] + ends[g], deadline[g])
        if true_end > t + 1e-9:
            heapq.heappush(running, (true_end, g))
            continue
        got[g] = t - start[g]
        state["unfinished"] -= 1
        if nxt < n:
            launch(nxt, min(t, H))
            nxt += 1
    return got


def simulate(runs, cuts, model, K=8, passes=1, reps=REPS, seed=SEED, limit=None, latency=None):
    """limit=None: every play's control budget is its own recorded time (the control arm then
    reproduces the recorded runs). limit=x: a fixed per-game wall-clock limit x from game start,
    with latency (seconds before the first action) taken from the run.

    Arms: "none" = control; str(c) = cutoff c with the dynamic schedule; on slots also
    "dyn_only" = the dynamic schedule without any cutoff (separates schedule from stopping)."""
    groups = list(by_game(runs).values())
    pooled, deep = build(runs)
    rng = np.random.default_rng(seed)
    arms = [("none", None, False)] + [(str(c), c, True) for c in cuts if c is not None]
    if model == "slots":
        arms.append(("dyn_only", None, True))
    acc = {key: {"levels": [], "score": [], "stops": [], "slow": []} for key, _, _ in arms}
    for _ in range(reps):
        gi = rng.integers(0, len(groups), len(groups))
        plays = []
        for i in gi:
            for _p in range(passes):
                r = groups[i][rng.integers(0, len(groups[i]))]
                lat = 0.0 if latency is None else latency(r)
                plays.append(Play(r, pooled, deep, rng, lat))
        plays = [plays[o] for o in rng.permutation(len(plays))]
        budgets = np.array([p.rec_end if limit is None else limit for p in plays])
        H = math.ceil(len(plays) / K) * float(np.mean(budgets))
        for key, c, dynamic in arms:
            paths = [p.path(c) for p in plays]
            ends = np.array([e for _, e in paths])
            if model == "shared":
                alloc = shared_alloc(ends, float(np.sum(budgets)))
            else:
                alloc = slot_alloc(ends, budgets, K, dynamic, H)
            ks, ss = zip(*[p.result(c, x) for p, x in zip(plays, alloc)])
            acc[key]["levels"].append(np.mean(ks))
            acc[key]["score"].append(np.mean(ss))
            if c is not None:
                stopped = [(e <= x + 1e-9) and np.isfinite(e) and len(cl) < p.r["L"]
                           for p, (cl, e), x in zip(plays, paths, alloc)]
                acc[key]["stops"].append(sum(stopped))
                acc[key]["slow"].append(sum(p.slow_clear(c, b) for p, b in zip(plays, budgets)))
    out = {}
    base_lv = np.array(acc["none"]["levels"])
    base_sc = np.array(acc["none"]["score"])
    for key, c, _ in arms:
        lv = np.array(acc[key]["levels"])
        sc = np.array(acc[key]["score"])
        o = {"levels": [lv.mean(), *np.percentile(lv, [2.5, 97.5])],
             "score": [sc.mean(), *np.percentile(sc, [2.5, 97.5])],
             "d_levels": [(lv - base_lv).mean(), *np.percentile(lv - base_lv, [2.5, 97.5])],
             "d_score": [(sc - base_sc).mean(), *np.percentile(sc - base_sc, [2.5, 97.5])],
             "p_levels_better": float(np.mean(lv - base_lv > 0))}
        if c is not None:
            o["plays_stopped_by_cutoff"] = float(np.mean(acc[key]["stops"]))
            o["control_plays_clearing_after_c"] = float(np.mean(acc[key]["slow"]))
            if "dyn_only" in acc:
                # effect of the cutoff itself: same schedule, same draws, with and without stopping
                e = lv - np.array(acc["dyn_only"]["levels"])
                o["cutoff_effect_levels"] = [e.mean(), *np.percentile(e, [2.5, 97.5])]
        out[key] = o
    return out


def fmt(o):
    return (f"lv {o['levels'][0]:.3f} dlv {o['d_levels'][0]:+.3f} "
            f"[{o['d_levels'][1]:+.3f},{o['d_levels'][2]:+.3f}] "
            f"sc {o['score'][0]:.2f} dsc {o['d_score'][0]:+.2f} [{o['d_score'][1]:+.2f},{o['d_score'][2]:+.2f}]")


if __name__ == "__main__":
    res = {"ideal": {}, "sim_shared": {}, "sim_slots": {}, "live_prediction": {}}
    curves = {}
    print("== ideal R(c) (levels per 100 actions)")
    for name in DATASETS:
        runs = load(name)
        out, cs, R = ideal(runs)
        # bootstrap c* and the gain R(c*)/R(cmax) and R(80)/R(cmax)
        groups = list(by_game(runs).values())
        rng = np.random.default_rng(SEED)
        cst, g80, gst = [], [], []
        for _ in range(BOOT // 4):
            idx = rng.integers(0, len(groups), len(groups))
            o, _, _ = ideal([r for i in idx for r in groups[i]])
            if o["R_cmax"] <= 0:
                continue
            cst.append(o["c_star"]); gst.append(o["R_star"] / o["R_cmax"])
            if o["R_80"] is not None:
                g80.append(o["R_80"] / o["R_cmax"])
        out["c_star_ci"] = np.percentile(cst, [2.5, 97.5]).tolist()
        out["gain_star_ci"] = np.percentile(gst, [2.5, 97.5]).tolist()
        out["gain_80_ci"] = np.percentile(g80, [2.5, 97.5]).tolist() if g80 else None
        res["ideal"][name] = out
        curves[name] = (cs, R)
        g80s = f"{out['R_80'] / out['R_cmax']:.2f}" if out["R_80"] else "  - "
        print(f"{name:18s} c*={out['c_star']:3d} [{out['c_star_ci'][0]:.0f},{out['c_star_ci'][1]:.0f}] "
              f"R*={100 * out['R_star']:.2f} h(c*)={100 * out['hazard_at_c_star']:.2f} "
              f"R(80)/R(cmax={out['cmax']})={g80s} R*/R(cmax)={out['R_star'] / out['R_cmax']:.2f}")

    print("\n== simulation in time; control budget = each run's recorded time")
    for name in DATASETS:
        runs = [r for r in load(name) if r.get("t_first") is not None and run_end(r)]
        for model, key in (("shared", "sim_shared"), ("slots", "sim_slots")):
            o = simulate(runs, CUTS, model, reps=REPS // 2)
            res[key][name] = {"runs_used": len(runs), **o}
            print(f"{name:18s} {model:6s} none {o['none']['levels'][0]:.3f} {o['none']['score'][0]:.2f} | "
                  + " | ".join(f"c={c}: {fmt(o[str(c)])}" for c in CUTS[1:])
                  + (f" | sched-only: {fmt(o['dyn_only'])}" if "dyn_only" in o else ""))

    # Predictions for the live paired test from the Qwen3.8-Flash endpoint traces
    # (same model and endpoint, 30 min per game, 8 concurrent). Game start = t_final - 1800 s
    # for runs that ended at the limit; other runs get the median start latency.
    runs = load("api_qwen38flash")
    lat = {id(r): r["t_first"] - (r["t_final"] - 1800.0) for r in runs if r["state"] == "gave_up"}
    med = float(np.median(list(lat.values())))
    latency = lambda r: min(max(lat.get(id(r), med), 0.0), 600.0)
    for label, minutes, passes in (("screening_15min_2pass", 15, 2), ("confirmation_30min_5pass", 30, 5)):
        o = simulate(runs, (None, 80), "slots", K=8, passes=passes, reps=REPS,
                     limit=60.0 * minutes, latency=latency)
        res["live_prediction"][label] = {"limit_s": 60 * minutes, "passes": passes, "K": 8, **o}
        print(f"\nprediction {label}: control {o['none']['levels'][0]:.3f} levels, "
              f"{o['none']['score'][0]:.2f} score | c=80: {fmt(o['80'])} P(levels better)={o['80']['p_levels_better']:.2f} "
              f"stopped={o['80']['plays_stopped_by_cutoff']:.1f}/{25 * passes} "
              f"control slow clears={o['80']['control_plays_clearing_after_c']:.1f}/{25 * passes} | "
              f"schedule only: {fmt(o['dyn_only'])}")
    dump("a5_theory", res)

    # appendix figure: R(c) normalised by R at the largest observed cutoff
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name in DATASETS:
        cs, R = curves[name]
        if name == "api_gptoss120b":
            continue
        ax.plot(cs, R / R[-1], color=COLORS[name], lw=1.8, label=LABELS[name])
    ax.axhline(1, color="#8a8984", lw=0.8)
    ax.axvline(80, color="#8a8984", lw=0.8, ls=":")
    ax.set_xlabel("per-level cutoff c (actions)")
    ax.set_ylabel("R(c) / R(largest observed c)")
    ax.set_xlim(0, 300)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "figures" / "rate_curve.png", dpi=200)
