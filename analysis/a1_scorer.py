"""A1: recompute the official score for every trace record and report the largest error."""
from common import DATASETS, dump, load, score

rows = {}
for name in DATASETS:
    runs = load(name)
    errs = [(abs(score(r["L"], r["k"], r["a"], r["base"]) - r["score"]), r) for r in runs]
    worst, wr = max(errs, key=lambda e: e[0])
    mean_rec = sum(r["score"] for r in runs) / len(runs)
    mean_new = sum(score(r["L"], r["k"], r["a"], r["base"]) for r in runs) / len(runs)
    rows[name] = {"runs": len(runs), "max_abs_err": worst, "worst_game": wr["game"],
                  "mean_recorded": mean_rec, "mean_recomputed": mean_new,
                  "n_err_gt_0.01": sum(e > 0.01 for e, _ in errs)}
    print(f"{name:18s} runs={len(runs):4d} max|err|={worst:.4f} ({wr['game']}) "
          f"n>0.01={rows[name]['n_err_gt_0.01']} mean {mean_rec:.3f} vs {mean_new:.3f}")
dump("a1_scorer", rows)
