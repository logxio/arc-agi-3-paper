# Analysis and Figure Generation Suite

Deterministic Python 3.13 scripts that compute every statistical table, theorem verification, rational receipt inversion, and publication figure in `paper/frontier-lock.md` and `paper/appendix.md`.

| Script | Purpose | Output Artifact |
| :- | :- | :- |
| `common.py` | Trace loader, official ARC-AGI-3 triangular scorer, Kaplan-Meier estimator, game-level bootstrap | Library module |
| `a1_scorer.py` | Recomputes every recorded game score against the official triangular formula | `out/a1_scorer.json` |
| `a2_frontier_lock.py` | Computes action, wall-clock, and token shares spent on the final unsolved level | `out/a2_frontier_lock.json` |
| `a3_hazard.py` | Computes 20-action completion hazard $h(n)$ in actions and seconds | `out/a3_hazard.json` |
| `a3b_by_dataset.py` | Per-dataset completion hazard with 2,000-replicate bootstrap bands | `out/a3b_by_dataset.json` |
| `a4_continue_restart.py` | Matched vs. unmatched level-or-attempt hazard collapse decomposition | `out/a4_continue_restart.json` |
| `a5_theory.py` | Charnov marginal value theorem renewal rate $R(c)$ and finite-pool simulation | `out/a5_theory.json` |
| `a5b_regime.py` | Per-game time-limit sweep with 15-min screening and 30-min confirmation overlays | `out/a5b_regime.json` |
| `a6_replay.py` | Counterfactual per-level action cutoff replay across 825 baseline runs | `out/a6_replay.json` |
| `a7_score_decomposition.py` | Triangular depth weight vs. quadratic step-efficiency counterfactuals | `out/a7_score_decomposition.json` |
| `a8_confirmation.py` | 250-run paired live confirmation bootstrap and net level-clear test | `out/a8_confirmation.json` |
| `a9_d4_and_reachability_audit.py` | Verifies 768 $D_4$ Cayley checks, 597 cross-level latencies, `0/219` false transfers, and 9-suite reachability | `out/a9_d4_and_reachability_audit.json` |
| `compute_ablation_ladder.py` | Computes 4-arm `Public-25` ablation ladder, 98-game transfer CIs, and 825-run dilution bounds | `out/ablation_ladder.json` |
| `tables.py` | Renders statistical tables A1 through A7 | `out/tables.md` |
| `x0_specimens.py` | Extracts `tn36` pixel frames, 342 human replay pairs, and 30 Kaggle receipts | `out/x0_specimens.json` |
| `palette.py`, `figcheck.py` | Publication color palette and automated zero-collision / zero-green-pixel auditor | Library modules |
| `generate_xray_figures.py` | Renders `cover.png`, `hazard.png`, `architecture.png`, `regime.png`, and `hazard_by_dataset.png` (`0` collisions, `0` green pixels) | `../paper/figures/*.png` |
| `wordcount.py` | Audits `frontier-lock.md` word count ($\le 1,500$ words) | Console / JSON receipt |
| `build_pdf.py` & `verify_pdf.py` | Synchronizes `.md` sources to `.tex`, compiles the 20-page PDF, and runs visual/margin QA | `../paper/ARC_Prize_2026_Breaking_Frontier_Lock.pdf` |
| `audit_visual_maturity.py` | Measures page geometry, type scale, table alignment, figure type and identifier breaks in benchmark papers and our PDF | `out/top_papers_visual_statistics.json` |
| `build_repro_notebook.py` | Generates and executes the 20-cell reproducibility notebook | `../notebooks/frontier_lock_reproduction.ipynb` |
| `audit_submission_bundle.py` | Runs the 4-invariant, 20-claim preflight submission bundle audit | `out/submission_audit/preflight_audit_receipt.json` |

All code in this directory is released under [MIT-0](LICENSE).
