"""Shared helpers: trace loading, the official scorer, Kaplan-Meier, game bootstrap.

Every analysis script imports this module and reads only evidence/traces/.
Released under MIT-0.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "evidence" / "traces"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Order used in every table and figure.
DATASETS = [
    "kaggle_v2", "kaggle_v3", "kaggle_v4", "kaggle_v5", "kaggle_v7",
    "api_qwen38flash", "tufa_qwen36_27b", "api_gptoss120b",
]
LABELS = {
    "kaggle_v2": "Kaggle V2 (submitted)",
    "kaggle_v3": "Kaggle V3",
    "kaggle_v4": "Kaggle V4",
    "kaggle_v5": "Kaggle V5",
    "kaggle_v7": "Kaggle V7 (16K ctx)",
    "api_qwen38flash": "Qwen3.8-Flash, endpoint",
    "tufa_qwen36_27b": "Qwen3.6-27B, Duck authors",
    "api_gptoss120b": "gpt-oss-120b, endpoint",
}
# Reference categorical palette, fixed order (light surface).
COLORS = {
    "kaggle_v2": "#2a78d6", "kaggle_v3": "#eb6834", "kaggle_v4": "#1baf7a",
    "kaggle_v5": "#eda100", "kaggle_v7": "#e87ba4", "api_qwen38flash": "#008300",
    "tufa_qwen36_27b": "#4a3aa7", "api_gptoss120b": "#e34948",
}
BOOT = 2000
SEED = 0


def load(name: str) -> list[dict]:
    with open(TRACES / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_game(runs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        out[r["game"]].append(r)
    return dict(out)


def game_bootstrap(runs: list[dict], stat, n: int = BOOT, seed: int = SEED):
    """Resample games with replacement (all runs of a game together); return (point, lo, hi)."""
    groups = list(by_game(runs).values())
    rng = np.random.default_rng(seed)
    point = stat(runs)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        sample = [r for i in idx for r in groups[i]]
        v = stat(sample)
        if v is not None and np.all(np.isfinite(v)):
            vals.append(v)
    vals = np.array(vals)
    return point, np.percentile(vals, 2.5, axis=0), np.percentile(vals, 97.5, axis=0)


# ---------------------------------------------------------------- scorer

def level_score(actions: int, base: int) -> float:
    return min(115.0, (base / actions) ** 2 * 100.0)


def score(L: int, k: int, a: list[int], base: list[int]) -> float:
    """Official ARC-AGI-3 game score (docs.arcprize.org/methodology).

    Level i (1-based) has weight i. A completed level scores
    min(115, (baseline/actions)^2 * 100); an uncompleted level scores 0.
    The weighted mean is capped at 100 * completed_weight / total_weight.
    """
    total_w = L * (L + 1) / 2
    s = 0.0
    done_w = 0.0
    for i in range(min(k, L)):
        if a[i] > 0:
            s += (i + 1) * level_score(a[i], base[i])
            done_w += i + 1
    return min(s / total_w, 100.0 * done_w / total_w)


def depth_cap(k: int, L: int) -> float:
    return 100.0 * (k * (k + 1) / 2) / (L * (L + 1) / 2)


# ---------------------------------------------------------------- attempts

def attempts(runs: list[dict], unit: str = "actions"):
    """One record per (run, level reached): (game, level, duration, event).

    Completed levels are events with duration = their actions (or seconds).
    The final unsolved level is right-censored at the actions (seconds) already
    spent on it. Seconds use t (last action of each completed level) and the run
    end (t_final, else t_last), measured from the run's first action t_first:
    in queued batches the clock starts at batch start, not game start. Runs cut
    when a batch stopped (state != gave_up) or without per-level timing are skipped.
    """
    rows = []
    for r in runs:
        L, k, a = r["L"], r["k"], r["a"]
        if unit == "actions":
            for i in range(k):
                rows.append((r["game"], i, a[i], True))
            if k < L:
                rows.append((r["game"], k, a[k], False))
        else:
            t = r.get("t")
            if t is None or len(t) != k or r["state"] != "gave_up":
                continue
            end = r.get("t_final") or r.get("t_last")
            if end is None or r.get("t_first") is None:
                continue
            prev = r["t_first"]
            for i in range(k):
                rows.append((r["game"], i, max(t[i] - prev, 0.0), True))
                prev = t[i]
            if k < L:
                rows.append((r["game"], k, max(end - prev, 0.0), False))
    return rows


def km(dur, ev, grid):
    """Kaplan-Meier survival S(x) = P(T > x) at each grid point, plus at-risk counts.

    Censored records with duration c count as surviving through c.
    """
    dur = np.asarray(dur, float)
    ev = np.asarray(ev, bool)
    times = np.unique(dur[ev])
    S = 1.0
    surv_at = []
    for t in times:
        at_risk = np.sum(dur >= t)
        d = np.sum((dur == t) & ev)
        if at_risk > 0:
            S *= 1.0 - d / at_risk
        surv_at.append(S)
    surv_at = np.array(surv_at)
    grid = np.asarray(grid, float)
    idx = np.searchsorted(times, grid, side="right") - 1
    S_grid = np.where(idx >= 0, surv_at[np.clip(idx, 0, None)] if len(surv_at) else 1.0, 1.0)
    # attempts still unsolved and observed past g
    risk = len(dur) - np.searchsorted(np.sort(dur), grid, side="right")
    return S_grid, risk


def cond_pass(dur, ev, n, w):
    """P(clear within the next w | n spent without clearing) from KM; None if nobody at risk."""
    S, risk = km(dur, ev, [n, n + w])
    if risk[0] == 0 or S[0] <= 0:
        return None
    return 1.0 - S[1] / S[0]


def dump(name: str, obj) -> None:
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=float)
