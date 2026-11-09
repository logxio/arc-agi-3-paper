# Level traces

One JSON object per game run. Every run is one ARC-AGI-3 public game played once by a Duck-harness agent (a model writes and runs Python in a REPL, sees image and grid observations, keeps a trimmed interaction history). Only the 25 public games exist outside Kaggle; no training set is involved.

## Fields

| Field | Meaning |
|---|---|
| `game` | Public game id |
| `run` | Index of this run among the runs of the same game in the same dataset |
| `L` | Number of levels in the game |
| `k` | Levels completed (levels are played strictly in order) |
| `a` | Actions spent on each level, length `L`; `a[k]` is the final unsolved level (the frontier) when `k < L`; zeros after it |
| `base` | Human baseline actions per level used by the official scorer |
| `score` | Official game score (0 to 100) as computed by the harness |
| `state` | Terminal state; `gave_up` is the normal end at the time limit, `cancelled` and `playing` mean the run was cut when the batch stopped. All three are right-censored at the frontier |
| `t_final` | Seconds from game start to run end, when recorded |
| `t` | Seconds from game start to the last action of each completed level, length `k` (runs with per-action timing only) |
| `t_first`, `t_last` | Seconds from game start to the first and to the last action |
| `tok` | Generated tokens attributed to each level up to and including the frontier (tokens generated before an action are charged to that action's level) |

Actions include RESET. A completed level's duration in actions is exact. The frontier's actions are right-censored: the run ended before the level was cleared.

## Datasets

| File | Model and serving | Setting | Runs |
|---|---|---|---|
| `kaggle_v2.jsonl` | Qwen3.8-Flash-Next NVFP4 with MTP, vLLM on one RTX Pro 6000 (Kaggle), 28 games scheduled concurrently, 5 GiB KV | Save-and-run commit of the submitted notebook (hidden-set scores 2.93 and 3.22), 7920 s per game | 25 × 1 |
| `kaggle_v3.jsonl` | same model | KV size left to vLLM (2.26 GiB), prefix caching on, 28 sequences: throughput halved | 25 × 1 |
| `kaggle_v4.jsonl` | same model | prefix caching on, otherwise as V2 | 25 × 1 |
| `kaggle_v5.jsonl` | same model | 4 concurrent games, history trimmed in blocks to 60% of budget, dynamic per-game time from a shared 7920 s pool (hidden-set scores 2.51 and 0.99) | 25 × 1 |
| `kaggle_v7.jsonl` | same model | V5 plus a 16K-token context per game: throughput up, capability down | 25 × 1 |
| `api_qwen38flash.jsonl` | Qwen3.8-Flash on a hosted OpenAI-compatible endpoint, game engines local | 30 min per game, 8 concurrent | 25 × 5 |
| `tufa_qwen36_27b.jsonl` | Qwen3.6-27B, the official Duck example run published by its authors (2 GPUs, 32 concurrent games per GPU) | 45 min per game | 25 × 20 |
| `api_gptoss120b.jsonl` | gpt-oss-120b on a hosted endpoint | 30 min per game; 27 runs were still playing when the batch stopped | 25 × 3 |

The live paired screening runs are summarized in `../readings.md`. The 250 run-level confirmation records are in `../confirmation_runs.jsonl`; `../../analysis/a8_confirmation.py` reproduces their paired estimates.
