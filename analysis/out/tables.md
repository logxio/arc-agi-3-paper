### A1. Scorer check

| Dataset | Runs | Max abs. error | Records off by > 0.01 | Mean recorded | Mean recomputed |
|---|---:|---:|---:|---:|---:|
| Kaggle V2 (submitted) | 25 | 0.0000 | 0 | 8.796 | 8.796 |
| Kaggle V3 | 25 | 0.0000 | 0 | 3.123 | 3.123 |
| Kaggle V4 | 25 | 0.0000 | 0 | 6.379 | 6.379 |
| Kaggle V5 | 25 | 0.0000 | 0 | 7.295 | 7.295 |
| Kaggle V7 (16K ctx) | 25 | 0.0000 | 0 | 1.367 | 1.367 |
| Qwen3.8-Flash, endpoint | 125 | 0.0000 | 0 | 9.066 | 9.066 |
| Qwen3.6-27B, Duck authors | 500 | 0.0000 | 0 | 1.600 | 1.600 |
| gpt-oss-120b, endpoint | 75 | 2.7778 | 2 | 0.160 | 0.232 |

### A2. Frontier Lock (share of actions, seconds, generated tokens on the final unsolved level, %)

| Dataset | Runs | Actions | Frontier actions | Action share | Time share | Token share |
|---|---:|---:|---:|---:|---:|---:|
| Kaggle V2 (submitted) | 25 | 5,436 | 3,746 | 68.9 [54.4, 81.6] | 63.6 [52.4, 74.9] | 61.1 [48.8, 73.3] |
| Kaggle V3 | 25 | 1,536 | 956 | 62.2 [46.5, 76.2] | 57.6 [42.9, 72.0] | 47.9 [33.2, 64.4] |
| Kaggle V4 | 25 | 3,310 | 2,148 | 64.9 [45.6, 80.5] | 67.6 [55.8, 78.4] | 65.5 [52.8, 76.3] |
| Kaggle V5 | 25 | 3,169 | 2,103 | 66.4 [52.4, 78.2] | 69.0 [57.3, 79.8] | 65.4 [53.0, 77.4] |
| Kaggle V7 (16K ctx) | 25 | 607 | 366 | 60.3 [42.5, 79.9] | 89.7 [77.9, 97.6] | 82.4 [65.7, 94.9] |
| Qwen3.8-Flash, endpoint | 125 | 14,238 | 7,463 | 52.4 [42.2, 62.5] | 51.6 [44.0, 60.2] | 49.2 [41.1, 58.3] |
| Qwen3.6-27B, Duck authors | 500 | 68,682 | 56,953 | 82.9 [76.1, 89.2] | 84.4 [78.8, 89.4] | 79.7 [73.4, 85.9] |
| gpt-oss-120b, endpoint | 75 | 17,756 | 17,644 | 99.4 [98.5, 100.0] | 97.4 [93.0, 100.0] | 94.0 [84.3, 100.0] |

### A3. Completion hazard: P(clear within the next 20 actions | n actions spent without clearing)

| Dataset | Attempts | Clears | n = 0 | 20 | 40 | 80 | 160 | Median actions of a cleared level | Clears after 80 actions | Median actions / human baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kaggle V2 (submitted) | 67 | 42 | 0.272 [0.14, 0.38] | 0.298 [0.19, 0.40] | 0.093 [0.00, 0.18] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 22 | 11.9% | 0.82 |
| Kaggle V3 | 45 | 22 | 0.225 [0.10, 0.35] | 0.360 [0.17, 0.54] | 0.259 [0.00, 0.54] | 0.000 [0.00, 0.00] | – | 22 | 0.0% | 0.88 |
| Kaggle V4 | 56 | 33 | 0.313 [0.16, 0.45] | 0.168 [0.07, 0.27] | 0.289 [0.10, 0.50] | 0.091 [0.00, 0.30] | 0.000 [0.00, 0.00] | 20 | 6.1% | 0.78 |
| Kaggle V5 | 58 | 35 | 0.267 [0.12, 0.41] | 0.312 [0.19, 0.41] | 0.215 [0.07, 0.36] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 25 | 2.9% | 0.75 |
| Kaggle V7 (16K ctx) | 37 | 12 | 0.414 [0.22, 0.60] | 0.167 [0.00, 0.60] | 0.667 [0.00, 1.00] | – | – | 17 | 0.0% | 0.68 |
| Qwen3.8-Flash, endpoint | 310 | 200 | 0.276 [0.15, 0.40] | 0.334 [0.23, 0.42] | 0.308 [0.20, 0.42] | 0.085 [0.00, 0.17] | 0.000 [0.00, 0.00] | 24 | 4.5% | 0.82 |
| Qwen3.6-27B, Duck authors | 806 | 309 | 0.164 [0.09, 0.24] | 0.126 [0.07, 0.18] | 0.084 [0.04, 0.13] | 0.054 [0.02, 0.09] | 0.010 [0.00, 0.03] | 25 | 12.3% | 1.11 |
| gpt-oss-120b, endpoint | 70 | 6 | 0.060 [0.00, 0.15] | 0.034 [0.00, 0.08] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 14 | 0.0% | 0.82 |

Same in seconds (per-level timing; clock starts at a run's first action; runs cut when a batch stopped are excluded): P(clear within the next 2 minutes | t seconds spent)

| Dataset | Attempts | Median seconds of a cleared level | t = 0 | 120 s | 300 s | 600 s | 1200 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kaggle V2 (submitted) | 67 | 1254 | 0.015 | 0.000 | 0.016 | 0.071 | 0.067 |
| Kaggle V3 | 47 | 3395 | 0.000 | 0.022 | 0.000 | 0.000 | 0.000 |
| Kaggle V4 | 58 | 1332 | 0.000 | 0.017 | 0.018 | 0.038 | 0.070 |
| Kaggle V5 | 60 | 192 | 0.119 | 0.275 | 0.178 | 0.000 | 0.000 |
| Kaggle V7 (16K ctx) | 37 | 125 | 0.162 | 0.129 | 0.000 | 0.000 | 0.000 |
| Qwen3.8-Flash, endpoint | 296 | 397 | 0.021 | 0.137 | 0.178 | 0.099 | 0.099 |
| Qwen3.6-27B, Duck authors | 806 | 521 | 0.055 | 0.035 | 0.050 | 0.046 | 0.020 |
| gpt-oss-120b, endpoint | 52 | 373 | 0.019 | 0.000 | 0.000 | 0.000 | 0.000 |

### A4. Continue or restart the same level

| Dataset | W | Stuck attempts | Levels (≥ 3 attempts) | Continue: clear in next W | Fresh attempt, same level: clear in first W | Difference | Fresh attempt, any eligible level | Share of the drop explained by which levels remain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B, Duck authors | 40 | 513 | 48 | 0.133 [0.068, 0.199] | 0.112 [0.067, 0.159] | +0.021 [-0.042, +0.090] | 0.271 | – (continuing is not worse) |
| Qwen3.6-27B, Duck authors | 80 | 344 | 48 | 0.129 [0.073, 0.197] | 0.167 [0.104, 0.239] | -0.038 [-0.089, +0.013] | 0.368 | 84% [66, 107] |
| Qwen3.8-Flash, endpoint | 40 | 108 | 62 | 0.478 [0.321, 0.629] | 0.188 [0.109, 0.286] | +0.291 [+0.105, +0.454] | 0.520 | – (continuing is not worse) |
| Qwen3.8-Flash, endpoint | 80 | 38 | 62 | 0.262 [0.108, 0.433] | 0.434 [0.297, 0.615] | -0.172 [-0.395, -0.011] | 0.749 | 65% [34, 97] |
| gpt-oss-120b, endpoint | 40 | 35 | 15 | 0.000 [0.000, 0.000] | 0.029 [0.000, 0.097] | -0.029 [-0.097, +0.000] | 0.143 | 80% [-6, 100] |
| gpt-oss-120b, endpoint | 80 | 31 | 15 | 0.000 [0.000, 0.000] | 0.032 [0.000, 0.111] | -0.032 [-0.111, +0.000] | 0.143 | 77% [-23, 100] |
| Kaggle V2, V3, V4, V5, V7 pooled (one run each) | 40 | 86 | 51 | 0.361 [0.211, 0.518] | 0.221 [0.133, 0.317] | +0.140 [+0.008, +0.268] | 0.517 | – (continuing is not worse) |
| Kaggle V2, V3, V4, V5, V7 pooled (one run each) | 80 | 33 | 51 | 0.196 [0.062, 0.372] | 0.192 [0.086, 0.312] | +0.004 [-0.152, +0.177] | 0.691 | – (continuing is not worse) |

### A5. Ideal cutoff theory: R(c) = P(T ≤ c) / E[min(T, c)] (levels per 100 actions)

| Dataset | Largest c observed (≥ 10 at risk) | c* [95%] | R(c*) | R at largest c | R(c*) / R(largest c) [95%] | R(80) / R(largest c) |
|---|---:|---:|---:|---:|---:|---:|
| Kaggle V2 (submitted) | 154 | 24 [21, 32] | 1.96 | 0.85 | 2.31 [1.70, 3.70] | 1.40 |
| Kaggle V3 | 54 | 30 [11, 49] | 1.76 | 1.59 | 1.11 [1.03, 1.62] | – |
| Kaggle V4 | 86 | 22 [15, 52] | 2.01 | 1.33 | 1.52 [1.13, 2.49] | 0.99 |
| Kaggle V5 | 75 | 34 [9, 57] | 1.67 | 1.33 | 1.26 [1.11, 2.07] | – |
| Kaggle V7 (16K ctx) | 19 | 18 [7, 20] | 2.15 | 2.07 | 1.04 [1.00, 1.95] | – |
| Qwen3.8-Flash, endpoint | 191 | 51 [19, 76] | 1.81 | 1.22 | 1.49 [1.13, 2.33] | 1.40 |
| Qwen3.6-27B, Duck authors | 316 | 23 [8, 44] | 0.89 | 0.26 | 3.47 [2.14, 5.84] | 2.33 |
| gpt-oss-120b, endpoint | 574 | 11 [4, 33] | 0.41 | 0.02 | 23.21 [13.41, 79.79] | 7.03 |

### A5. Simulation, each dataset's own recorded budget (control arm = the recorded runs)

Change in mean completed levels per game against the control, mean [95% interval] over 1,000 replicates that resample games. `shared`: one engine, all games start together (Kaggle notebooks, the Duck authors' run). `slots`: 8 concurrent games of fixed speed on a hosted endpoint, queue, dynamic grants.

| Dataset | Model | Control levels | Cut 50 | Cut 80 | Cut 150 | Schedule only (no cut) | Cut 80 minus schedule only | Cut 80: change in score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kaggle V2 (submitted) | shared | 1.679 | -0.244 [-0.440, +0.000] | -0.122 [-0.360, +0.160] | +0.000 [-0.160, +0.160] | – | – | +0.53 [-0.70, +3.73] |
| Kaggle V3 | shared | 0.845 | -0.011 [-0.160, +0.120] | +0.048 [+0.000, +0.160] | +0.000 [+0.000, +0.000] | – | – | +0.13 [+0.00, +0.64] |
| Kaggle V4 | shared | 1.320 | -0.122 [-0.400, +0.160] | +0.047 [-0.120, +0.200] | +0.006 [-0.080, +0.120] | – | – | +1.02 [-0.14, +2.55] |
| Kaggle V5 | shared | 1.495 | -0.245 [-0.600, +0.080] | +0.101 [-0.040, +0.320] | +0.057 [+0.000, +0.200] | – | – | +1.57 [-0.06, +4.11] |
| Kaggle V7 (16K ctx) | shared | 0.478 | +0.001 [+0.000, +0.040] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | – | – | +0.00 [+0.00, +0.00] |
| Qwen3.8-Flash, endpoint | slots | 1.586 | -0.253 [-0.600, +0.080] | +0.102 [-0.160, +0.360] | +0.209 [+0.000, +0.440] | +0.226 [+0.040, +0.480] | -0.124 [-0.440, +0.120] | +2.09 [-0.35, +5.39] |
| Qwen3.6-27B, Duck authors | shared | 0.610 | -0.168 [-0.400, +0.000] | -0.047 [-0.200, +0.080] | +0.010 [-0.040, +0.080] | – | – | +0.16 [-0.27, +1.07] |
| gpt-oss-120b, endpoint | slots | 0.081 | +0.004 [+0.000, +0.040] | +0.004 [+0.000, +0.040] | +0.003 [+0.000, +0.040] | +0.001 [+0.000, +0.040] | +0.002 [+0.000, +0.040] | +0.01 [+0.00, +0.22] |

### A5. Prediction for the live paired test (Qwen3.8-Flash endpoint traces, 8 slots, cutoff 80)

| Quantity | Screening: simulated | Screening: live | Confirmation: simulated (made before its results) |
|---|---:|---:|---:|
| Runs per arm | 50 (25 × 2, 15 min) | 50 (25 × 2, 15 min) | 125 (25 × 5, 30 min) |
| Control: mean completed levels | 0.829 [0.560, 1.140] | 0.680 | 1.598 [1.192, 2.040] |
| Stop + schedule minus control: levels | +0.096 [+0.000, +0.200] | +0.140 [0.000, 0.300] | -0.014 [-0.144, +0.088] |
| P(levels difference > 0) | 0.96 | 0.96 | 0.41 |
| Stop + schedule minus control: score | +0.83 [+0.03, +2.12] | +1.289 [0.176, 2.682] | +0.72 [-0.15, +1.69] |
| Schedule only minus control: levels | +0.090 [+0.000, +0.200] | not run | +0.030 [-0.016, +0.080] |
| Cutoff itself (stop + schedule minus schedule only): levels | +0.006 [-0.060, +0.080] | not run | -0.044 [-0.168, +0.048] |
| Intervention runs stopped by the cutoff | 6.4 / 50 | 6 / 50 | 44.0 / 125 |
| Control runs clearing a level after > 80 actions | 0.4 / 50 | 1 / 50 | 8.9 / 125 |

### A5b. Per-game time limit sweep (25 games × 2 runs, 8 slots, cutoff 80)

| Limit (min) | Control levels | Stop + schedule: Δ levels | Schedule only: Δ levels | Cutoff itself (stop + schedule − schedule only) | Stop + schedule: Δ score | Runs stopped by cutoff (of 50) | Control runs clearing after 80 actions (of 50) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.175 | +0.043 [+0.000, +0.100] | +0.043 [+0.000, +0.100] | +0.000 [+0.000, +0.000] | +0.17 | 0.3 | 0.0 |
| 7.5 | 0.385 | +0.052 [+0.000, +0.120] | +0.052 [+0.000, +0.120] | +0.000 [-0.020, +0.020] | +0.26 | 1.2 | 0.0 |
| 10 | 0.527 | +0.079 [+0.000, +0.160] | +0.074 [+0.000, +0.160] | +0.005 [-0.020, +0.040] | +0.53 | 3.0 | 0.0 |
| 12.5 | 0.671 | +0.095 [+0.020, +0.200] | +0.087 [+0.000, +0.180] | +0.008 [-0.040, +0.060] | +0.70 | 4.4 | 0.0 |
| 15 | 0.828 | +0.099 [+0.000, +0.200] | +0.092 [+0.000, +0.200] | +0.006 [-0.060, +0.080] | +0.84 | 6.4 | 0.4 |
| 20 | 1.146 | +0.101 [+0.000, +0.220] | +0.096 [+0.000, +0.200] | +0.005 [-0.100, +0.100] | +1.02 | 9.6 | 0.4 |
| 25 | 1.387 | +0.065 [-0.080, +0.200] | +0.101 [+0.000, +0.220] | -0.036 [-0.181, +0.080] | +1.21 | 15.0 | 2.8 |
| 30 | 1.599 | +0.026 [-0.140, +0.180] | +0.099 [+0.000, +0.220] | -0.073 [-0.260, +0.080] | +1.19 | 19.0 | 3.6 |

### A6. Replay: stop any level after N actions (recomputed scores, not live runs)

| Dataset | Mean score | N = 50: score kept / actions released | N = 80 | N = 150 | Global per-game cap releasing the same share as N = 80: score kept |
|---|---:|---:|---:|---:|---:|
| Kaggle V2 (submitted) | 8.80 | 93% / 64% | 96% / 51% | 99% / 29% | 87% |
| Kaggle V3 | 3.12 | 91% / 20% | 100% / 6% | 100% / 0% | 100% |
| Kaggle V4 | 6.38 | 84% / 50% | 98% / 33% | 100% / 18% | 87% |
| Kaggle V5 | 7.29 | 83% / 46% | 99% / 28% | 100% / 11% | 97% |
| Kaggle V7 (16K ctx) | 1.37 | 100% / 0% | 100% / 0% | 100% / 0% | 100% |
| Qwen3.8-Flash, endpoint | 9.07 | 81% / 41% | 97% / 24% | 99% / 11% | 83% |
| Qwen3.6-27B, Duck authors | 1.60 | 86% / 59% | 96% / 39% | 100% / 12% | 93% |
| gpt-oss-120b, endpoint | 0.23 | 100% / 83% | 100% / 74% | 100% / 58% | 100% |

### A7. Where the score is (scorer counterfactuals, not agent results)

| Dataset | Score [95%] | Every completed level at the human baseline | One more completed level per game | Median actions / human baseline | Games by completed levels (0, 1, 2, …) |
|---|---:|---:|---:|---:|---|
| Kaggle V2 (submitted) | 8.80 [4.08, 15.13] | 10.88 (+24%) | 16.26 (+85%) | 0.82 | 4, 11, 4, 4, 1, 0, 0, 1 |
| Kaggle V3 | 3.12 [1.98, 4.55] | 3.53 (+13%) | 8.87 (+184%) | 0.88 | 7, 14, 4 |
| Kaggle V4 | 6.38 [3.62, 9.64] | 6.72 (+5%) | 13.85 (+117%) | 0.78 | 4, 15, 2, 2, 2 |
| Kaggle V5 | 7.29 [3.92, 11.32] | 8.17 (+12%) | 14.63 (+101%) | 0.75 | 6, 11, 3, 3, 1, 1 |
| Kaggle V7 (16K ctx) | 1.37 [0.48, 2.71] | 1.77 (+29%) | 5.24 (+283%) | 0.68 | 15, 8, 2 |
| Qwen3.8-Flash, endpoint | 9.07 [5.64, 12.76] | 10.09 (+11%) | 17.21 (+90%) | 0.82 | 25, 45, 27, 14, 11, 3 |
| Qwen3.6-27B, Duck authors | 1.60 [0.88, 2.50] | 2.41 (+51%) | 5.26 (+229%) | 1.11 | 248, 198, 51, 3 |
| gpt-oss-120b, endpoint | 0.23 [0.00, 0.54] | 0.28 (+19%) | 3.43 (+1381%) | 0.82 | 69, 6 |
