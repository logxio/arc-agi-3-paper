"""A4: is the wall the level or the context?

For every (game, level) with at least three attempts in a dataset, compare
  continue: attempts already stuck W actions -> P(clear within the next W), and
  fresh:    a new attempt on the same level (another run, new context) -> P(clear within its first W).
Both sides use Kaplan-Meier so that time-limit censoring is handled. The fresh rate for a
stuck attempt is estimated leave-one-out from the other attempts on its level and averaged
over stuck attempts, so both sides describe the same mix of levels. 95% intervals resample games.
"""
from collections import defaultdict

import numpy as np

from common import BOOT, SEED, by_game, dump, km, load

WS = (40, 80)
MIN_ATT = 3


def level_attempts(runs):
    lv = defaultdict(list)
    for r in runs:
        for i in range(r["k"]):
            lv[(r["game"], i)].append((r["a"][i], True))
        if r["k"] < r["L"] and r["a"][r["k"]] > 0:
            lv[(r["game"], r["k"])].append((r["a"][r["k"]], False))
    return {key: v for key, v in lv.items() if len(v) >= MIN_ATT}


def p_within(att, w):
    d = [x[0] for x in att]; e = [x[1] for x in att]
    S, _ = km(d, e, [w])
    return 1.0 - S[0]


def compare(runs, w):
    lv = level_attempts(runs)
    resid_d, resid_e, fresh = [], [], []
    for key, att in lv.items():
        for j, (d, e) in enumerate(att):
            if d <= w:            # not stuck at w (cleared by w, or censored before w)
                continue
            resid_d.append(d - w); resid_e.append(e)
            others = att[:j] + att[j + 1:]
            fresh.append(p_within(others, w))
    if not resid_d:
        return None
    S, _ = km(resid_d, resid_e, [w])
    cont = 1.0 - S[0]
    # Unmatched fresh rate: every attempt on the eligible levels counts once.
    pooled = p_within([x for att in lv.values() for x in att], w)
    return cont, float(np.mean(fresh)), len(resid_d), len(lv), pooled


res = {}
for name in ("tufa_qwen36_27b", "api_qwen38flash", "api_gptoss120b", "kaggle_pooled"):
    if name == "kaggle_pooled":
        # Same model and harness under five serving settings; one run each per game.
        runs = [r for v in ("kaggle_v2", "kaggle_v3", "kaggle_v4", "kaggle_v5", "kaggle_v7")
                for r in load(v)]
    else:
        runs = load(name)
    groups = list(by_game(runs).values())
    res[name] = {}
    for w in WS:
        out = compare(runs, w)
        if out is None:
            continue
        cont, fr, n_stuck, n_levels, pooled = out
        share = (pooled - fr) / (pooled - cont) if pooled > cont else float("nan")
        rng = np.random.default_rng(SEED)
        diffs, conts, frs, shares = [], [], [], []
        for _ in range(BOOT):
            idx = rng.integers(0, len(groups), len(groups))
            # relabel resampled games so that duplicates stay separate levels
            sample = [dict(r, game=f"{r['game']}#{n}") for n, i in enumerate(idx) for r in groups[i]]
            o = compare(sample, w)
            if o is None:
                continue
            conts.append(o[0]); frs.append(o[1]); diffs.append(o[0] - o[1])
            if o[4] > o[0]:
                shares.append((o[4] - o[1]) / (o[4] - o[0]))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res[name][str(w)] = {"stuck_attempts": n_stuck, "levels": n_levels,
                             "continue": [cont, *ci(conts)], "fresh": [fr, *ci(frs)],
                             "difference": [cont - fr, *ci(diffs)],
                             "fresh_unmatched": pooled,
                             "share_of_drop_from_level_mix": [share, *ci(shares)]}
        print(f"{name:16s} W={w:3d} stuck={n_stuck:4d} levels={n_levels:3d} "
              f"continue={cont:.3f} [{ci(conts)[0]:.3f},{ci(conts)[1]:.3f}] "
              f"fresh={fr:.3f} [{ci(frs)[0]:.3f},{ci(frs)[1]:.3f}] "
              f"diff={cont - fr:+.3f} [{ci(diffs)[0]:+.3f},{ci(diffs)[1]:+.3f}] "
              f"unmatched={pooled:.3f} level-mix share={share:.2f} [{ci(shares)[0]:.2f},{ci(shares)[1]:.2f}]")
dump("a4_continue_restart", res)
