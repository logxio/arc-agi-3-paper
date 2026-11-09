"""Render every appendix table from analysis/out/*.json into analysis/out/tables.md."""
import json

from common import DATASETS, LABELS, OUT

J = {k: json.load(open(OUT / f"{k}.json")) for k in
     ("a1_scorer", "a2_frontier_lock", "a3_hazard", "a4_continue_restart", "a5_theory",
      "a5b_regime", "a6_replay", "a7_score_decomposition")}
NATIVE = {"kaggle_v2": "shared", "kaggle_v3": "shared", "kaggle_v4": "shared", "kaggle_v5": "shared",
          "kaggle_v7": "shared", "tufa_qwen36_27b": "shared", "api_qwen38flash": "slots",
          "api_gptoss120b": "slots"}
pct = lambda x: "–" if x is None else f"{100 * x:.1f}"
ci = lambda v, f="{:.3f}": "–" if v is None else (f.format(v[0]) + " [" + f.format(v[1]) + ", " + f.format(v[2]) + "]")
cip = lambda v: "–" if v is None else f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"
out = []
w = out.append

w("### A1. Scorer check\n")
w("| Dataset | Runs | Max abs. error | Records off by > 0.01 | Mean recorded | Mean recomputed |")
w("|---|---:|---:|---:|---:|---:|")
for d in DATASETS:
    r = J["a1_scorer"][d]
    w(f"| {LABELS[d]} | {r['runs']} | {r['max_abs_err']:.4f} | {r['n_err_gt_0.01']} | {r['mean_recorded']:.3f} | {r['mean_recomputed']:.3f} |")

w("\n### A2. Frontier Lock (share of actions, seconds, generated tokens on the final unsolved level, %)\n")
w("| Dataset | Runs | Actions | Frontier actions | Action share | Time share | Token share |")
w("|---|---:|---:|---:|---:|---:|---:|")
for d in DATASETS:
    r = J["a2_frontier_lock"][d]
    w(f"| {LABELS[d]} | {r['runs']} | {r['actions']:,} | {r['frontier_actions']:,} | {cip(r['action_share'])} | {cip(r['time_share'])} | {cip(r['token_share'])} |")

w("\n### A3. Completion hazard: P(clear within the next 20 actions | n actions spent without clearing)\n")
w("| Dataset | Attempts | Clears | n = 0 | 20 | 40 | 80 | 160 | Median actions of a cleared level | Clears after 80 actions | Median actions / human baseline |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for d in DATASETS:
    r = J["a3_hazard"]["actions"][d]
    cells = []
    for n in ("0", "20", "40", "80", "160"):
        h = r["cond_pass_next20"][n]
        cells.append("–" if h[0] is None else f"{h[0]:.3f} [{h[1]:.2f}, {h[2]:.2f}]")
    w(f"| {LABELS[d]} | {r['attempts']} | {r['events']} | " + " | ".join(cells)
      + f" | {r['median_cleared_actions']:.0f} | {pct(r['frac_events_after_80'])}% | {r['median_ratio_to_human']:.2f} |")
w("\nSame in seconds (per-level timing; clock starts at a run's first action; runs cut when a batch stopped are excluded): P(clear within the next 2 minutes | t seconds spent)\n")
w("| Dataset | Attempts | Median seconds of a cleared level | t = 0 | 120 s | 300 s | 600 s | 1200 s |")
w("|---|---:|---:|---:|---:|---:|---:|---:|")
for d in DATASETS:
    r = J["a3_hazard"]["seconds"].get(d)
    if not r:
        continue
    cells = []
    for n in ("0", "120", "300", "600", "1200"):
        h = r["cond_pass_next120s"][n]
        cells.append("–" if h[0] is None else f"{h[0]:.3f}")
    w(f"| {LABELS[d]} | {r['attempts']} | {r['median_cleared_seconds']:.0f} | " + " | ".join(cells) + " |")

w("\n### A4. Continue or restart the same level\n")
w("| Dataset | W | Stuck attempts | Levels (≥ 3 attempts) | Continue: clear in next W | Fresh attempt, same level: clear in first W | Difference | Fresh attempt, any eligible level | Share of the drop explained by which levels remain |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
names = {"tufa_qwen36_27b": LABELS["tufa_qwen36_27b"], "api_qwen38flash": LABELS["api_qwen38flash"],
         "api_gptoss120b": LABELS["api_gptoss120b"], "kaggle_pooled": "Kaggle V2, V3, V4, V5, V7 pooled (one run each)"}
for d, rows in J["a4_continue_restart"].items():
    for W, r in rows.items():
        sh = r["share_of_drop_from_level_mix"]
        # The decomposition only applies when continuing does worse than a fresh attempt at the same level.
        shs = ("– (continuing is not worse)" if sh[0] != sh[0] or r["difference"][0] >= 0
               else f"{100 * sh[0]:.0f}% [{100 * sh[1]:.0f}, {100 * sh[2]:.0f}]")
        w(f"| {names[d]} | {W} | {r['stuck_attempts']} | {r['levels']} | {ci(r['continue'])} | {ci(r['fresh'])} | {ci(r['difference'], '{:+.3f}')} | {r['fresh_unmatched']:.3f} | {shs} |")

w("\n### A5. Ideal cutoff theory: R(c) = P(T ≤ c) / E[min(T, c)] (levels per 100 actions)\n")
w("| Dataset | Largest c observed (≥ 10 at risk) | c* [95%] | R(c*) | R at largest c | R(c*) / R(largest c) [95%] | R(80) / R(largest c) |")
w("|---|---:|---:|---:|---:|---:|---:|")
for d in DATASETS:
    r = J["a5_theory"]["ideal"][d]
    g80 = "–" if r["R_80"] is None else f"{r['R_80'] / r['R_cmax']:.2f}"
    w(f"| {LABELS[d]} | {r['cmax']} | {r['c_star']} [{r['c_star_ci'][0]:.0f}, {r['c_star_ci'][1]:.0f}] | {100 * r['R_star']:.2f} | {100 * r['R_cmax']:.2f} | "
      f"{r['R_star'] / r['R_cmax']:.2f} [{r['gain_star_ci'][0]:.2f}, {r['gain_star_ci'][1]:.2f}] | {g80} |")

w("\n### A5. Simulation, each dataset's own recorded budget (control arm = the recorded runs)\n")
w("Change in mean completed levels per game against the control, mean [95% interval] over 1,000 replicates that resample games. `shared`: one engine, all games start together (Kaggle notebooks, the Duck authors' run). `slots`: 8 concurrent games of fixed speed on a hosted endpoint, queue, dynamic grants.\n")
w("| Dataset | Model | Control levels | Cut 50 | Cut 80 | Cut 150 | Schedule only (no cut) | Cut 80 minus schedule only | Cut 80: change in score |")
w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for d in DATASETS:
    m = NATIVE[d]
    r = J["a5_theory"]["sim_" + m][d]
    so = ci(r["dyn_only"]["d_levels"], "{:+.3f}") if "dyn_only" in r else "–"
    ce = ci(r["80"]["cutoff_effect_levels"], "{:+.3f}") if "cutoff_effect_levels" in r["80"] else "–"
    w(f"| {LABELS[d]} | {m} | {r['none']['levels'][0]:.3f} | {ci(r['50']['d_levels'], '{:+.3f}')} | {ci(r['80']['d_levels'], '{:+.3f}')} | "
      f"{ci(r['150']['d_levels'], '{:+.3f}')} | {so} | {ce} | {ci(r['80']['d_score'], '{:+.2f}')} |")

w("\n### A5. Prediction for the live paired test (Qwen3.8-Flash endpoint traces, 8 slots, cutoff 80)\n")
w("| Quantity | Screening: simulated | Screening: live | Confirmation: simulated (made before its results) |")
w("|---|---:|---:|---:|")
s = J["a5_theory"]["live_prediction"]["screening_15min_2pass"]
c = J["a5_theory"]["live_prediction"]["confirmation_30min_5pass"]
w(f"| Runs per arm | 50 (25 × 2, 15 min) | 50 (25 × 2, 15 min) | 125 (25 × 5, 30 min) |")
w(f"| Control: mean completed levels | {ci(s['none']['levels'])} | 0.680 | {ci(c['none']['levels'])} |")
w(f"| Stop + schedule minus control: levels | {ci(s['80']['d_levels'], '{:+.3f}')} | +0.140 [0.000, 0.300] | {ci(c['80']['d_levels'], '{:+.3f}')} |")
w(f"| P(levels difference > 0) | {s['80']['p_levels_better']:.2f} | 0.96 | {c['80']['p_levels_better']:.2f} |")
w(f"| Stop + schedule minus control: score | {ci(s['80']['d_score'], '{:+.2f}')} | +1.289 [0.176, 2.682] | {ci(c['80']['d_score'], '{:+.2f}')} |")
w(f"| Schedule only minus control: levels | {ci(s['dyn_only']['d_levels'], '{:+.3f}')} | not run | {ci(c['dyn_only']['d_levels'], '{:+.3f}')} |")
w(f"| Cutoff itself (stop + schedule minus schedule only): levels | {ci(s['80']['cutoff_effect_levels'], '{:+.3f}')} | not run | {ci(c['80']['cutoff_effect_levels'], '{:+.3f}')} |")
w(f"| Intervention runs stopped by the cutoff | {s['80']['plays_stopped_by_cutoff']:.1f} / 50 | 6 / 50 | {c['80']['plays_stopped_by_cutoff']:.1f} / 125 |")
w(f"| Control runs clearing a level after > 80 actions | {s['80']['control_plays_clearing_after_c']:.1f} / 50 | 1 / 50 | {c['80']['control_plays_clearing_after_c']:.1f} / 125 |")

w("\n### A5b. Per-game time limit sweep (25 games × 2 runs, 8 slots, cutoff 80)\n")
w("| Limit (min) | Control levels | Stop + schedule: Δ levels | Schedule only: Δ levels | Cutoff itself (stop + schedule − schedule only) | Stop + schedule: Δ score | Runs stopped by cutoff (of 50) | Control runs clearing after 80 actions (of 50) |")
w("|---:|---:|---:|---:|---:|---:|---:|---:|")
for m, r in J["a5b_regime"].items():
    w(f"| {m} | {r['none']['levels'][0]:.3f} | {ci(r['80']['d_levels'], '{:+.3f}')} | {ci(r['dyn_only']['d_levels'], '{:+.3f}')} | "
      f"{ci(r['80']['cutoff_effect_levels'], '{:+.3f}')} | "
      f"{r['80']['d_score'][0]:+.2f} | {r['80']['plays_stopped_by_cutoff']:.1f} | {r['80']['control_plays_clearing_after_c']:.1f} |")

w("\n### A6. Replay: stop any level after N actions (recomputed scores, not live runs)\n")
w("| Dataset | Mean score | N = 50: score kept / actions released | N = 80 | N = 150 | Global per-game cap releasing the same share as N = 80: score kept |")
w("|---|---:|---:|---:|---:|---:|")
for d in DATASETS:
    r = J["a6_replay"][d]
    cells = []
    for n in ("50", "80", "150"):
        x = r["stop"][n]
        kept = "–" if x["score_kept"] is None else f"{100 * x['score_kept']:.0f}%"
        cells.append(f"{kept} / {100 * x['actions_released']:.0f}%")
    g = r["stop"]["80"]["global_cap_score_kept"]
    w(f"| {LABELS[d]} | {r['mean_score']:.2f} | " + " | ".join(cells) + f" | {'–' if g is None else f'{100 * g:.0f}%'} |")

w("\n### A7. Where the score is (scorer counterfactuals, not agent results)\n")
w("| Dataset | Score [95%] | Every completed level at the human baseline | One more completed level per game | Median actions / human baseline | Games by completed levels (0, 1, 2, …) |")
w("|---|---:|---:|---:|---:|---|")
for d in DATASETS:
    r = J["a7_score_decomposition"][d]
    s0 = r["score"][0]
    w(f"| {LABELS[d]} | {ci(r['score'], '{:.2f}')} | {r['all_at_human_baseline'][0]:.2f} ({100 * (r['all_at_human_baseline'][0] / s0 - 1):+.0f}%) | "
      f"{r['one_more_level'][0]:.2f} ({100 * (r['one_more_level'][0] / s0 - 1):+.0f}%) | {r['median_actions_over_baseline']:.2f} | {', '.join(map(str, r['completed_level_histogram']))} |")

(OUT / "tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
