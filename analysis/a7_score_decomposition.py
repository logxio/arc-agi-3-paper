"""A7: where the score is. Scorer counterfactuals (not agent results):
  efficiency: every completed level at exactly the human baseline (= the depth cap);
  depth: one more completed level per game at the run's own efficiency
         (mean depth cap at k+1 times the ratio of actual score to depth cap).
Also the distribution of completed levels' actions relative to the human baseline."""
import numpy as np

from common import DATASETS, depth_cap, dump, game_bootstrap, load, score


def parts(runs):
    real = np.mean([score(r["L"], r["k"], r["a"], r["base"]) for r in runs])
    cap = np.mean([depth_cap(r["k"], r["L"]) for r in runs])
    plus = np.mean([depth_cap(min(r["k"] + 1, r["L"]), r["L"]) for r in runs])
    return np.array([real, cap, plus * (real / cap if cap else 0.0)])


res = {}
for name in DATASETS:
    runs = load(name)
    p, lo, hi = game_bootstrap(runs, parts, n=1000)
    ratios = [r["a"][i] / r["base"][i] for r in runs for i in range(r["k"]) if r["a"][i] > 0]
    ks = np.bincount([r["k"] for r in runs])
    res[name] = {"score": [p[0], lo[0], hi[0]], "all_at_human_baseline": [p[1], lo[1], hi[1]],
                 "one_more_level": [p[2], lo[2], hi[2]],
                 "median_actions_over_baseline": float(np.median(ratios)) if ratios else None,
                 "share_levels_at_or_below_baseline": float(np.mean(np.array(ratios) <= 1)) if ratios else None,
                 "completed_level_histogram": ks.tolist()}
    print(f"{name:18s} score {p[0]:5.2f} -> human-efficiency {p[1]:5.2f} ({p[1] / p[0] - 1:+.0%}) "
          f"-> +1 level {p[2]:5.2f} ({p[2] / p[0] - 1 if p[0] else 0:+.0%})  "
          f"median ratio {res[name]['median_actions_over_baseline']}  levels hist {ks.tolist()}")
dump("a7_score_decomposition", res)
