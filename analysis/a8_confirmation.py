"""Import and analyse the paired public-game confirmation run.

The import reads the complete serial-run log. It extracts only the final result
for each play, so the published JSONL has no endpoint, machine path or prompt.
Scores in that log are rounded to two decimals; the level count is exact.

Usage:
    python3 a8_confirmation.py --import-log /path/to/formal_0928.series.log
    python3 a8_confirmation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "evidence" / "confirmation_runs.jsonl"
OUTPUT = ROOT / "analysis" / "out" / "a8_confirmation.json"
HEADER = re.compile(r"Run directory: .*formal_0928_confirm_(base|early)\b")
FINISHED = re.compile(
    r'^\[finished\] ([a-z0-9]+)-[a-f0-9]+ state=(\w+) '
    r'level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) '
    r'tokens=(\d+) per-level=([^ ]+)'
)


def import_log(path: Path) -> str:
    source = path.read_bytes()
    arm = None
    rows = []
    for line in source.decode("utf-8", "replace").splitlines():
        match = HEADER.search(line)
        if match:
            arm = match.group(1)
            continue
        if line.startswith("finished confirm:"):
            arm = None
        if arm is None or not line.startswith("[finished]"):
            continue
        match = FINISHED.match(line)
        if not match:
            raise ValueError(f"Unparsed confirmation result: {line}")
        game, state, k, L, score, actions, tokens, levels = match.groups()
        a, human = zip(*(map(int, pair.split("/")) for pair in levels.split(",")))
        row = {"arm": arm, "game": game, "state": state, "k": int(k),
               "L": int(L), "score_rounded": float(score), "actions": int(actions),
               "tokens": int(tokens), "a": list(a), "human": list(human)}
        if len(a) != row["L"] or sum(a) != row["actions"]:
            raise ValueError(f"Invalid level actions: {game} {arm}")
        rows.append(row)
    validate(rows)
    DATA.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    return hashlib.sha256(source).hexdigest()


def validate(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["arm"]][row["game"]].append(row)
    if set(grouped) != {"base", "early"}:
        raise ValueError("Expected base and early arms")
    if set(grouped["base"]) != set(grouped["early"]) or len(grouped["base"]) != 25:
        raise ValueError("Expected the same 25 public games in both arms")
    for arm in grouped:
        if any(len(repeats) != 5 for repeats in grouped[arm].values()):
            raise ValueError(f"Expected five repeats per game in {arm}")
    return grouped


def paired(grouped, field: str):
    games = sorted(grouped["base"])
    base = st.mean(st.mean(r[field] for r in grouped["base"][g]) for g in games)
    early = st.mean(st.mean(r[field] for r in grouped["early"][g]) for g in games)
    differences = [st.mean(r[field] for r in grouped["early"][g]) -
                   st.mean(r[field] for r in grouped["base"][g]) for g in games]
    rng = random.Random(0)
    bootstrap = sorted(st.mean(rng.choice(differences) for _ in games) for _ in range(20000))
    return {"base": base, "early": early, "difference": early - base,
            "ci95": [bootstrap[500], bootstrap[19500]],
            "bootstrap_share_positive": sum(x > 0 for x in bootstrap) / len(bootstrap),
            "per_game": dict(zip(games, differences))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-log", type=Path)
    args = parser.parse_args()
    log_hash = import_log(args.import_log) if args.import_log else None
    if log_hash is None and OUTPUT.exists():
        log_hash = json.loads(OUTPUT.read_text()).get("source_log_sha256")
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line]
    grouped = validate(rows)
    summary = {"source_log_sha256": log_hash,
               "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
               "games": 25, "repeats_per_game_per_arm": 5,
               "score_precision": "two decimals in source log",
               "levels": paired(grouped, "k"),
               "score": paired(grouped, "score_rounded")}
    for arm in ("base", "early"):
        arm_rows = [r for game in grouped[arm].values() for r in game]
        summary[arm] = {"runs": len(arm_rows), "completed_levels": sum(r["k"] for r in arm_rows),
                        "zero_completion": sum(r["k"] == 0 for r in arm_rows),
                        "actions": sum(r["actions"] for r in arm_rows),
                        "tokens_from_result_lines": sum(r["tokens"] for r in arm_rows)}
    summary["base"]["clears_after_80_actions"] = sum(
        any(n > 80 for n in r["a"][:r["k"]]) for game in grouped["base"].values() for r in game)
    summary["early"]["unfinished_level_at_80_actions"] = sum(
        r["k"] < r["L"] and r["a"][r["k"]] == 80
        for game in grouped["early"].values() for r in game)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    for metric in ("levels", "score"):
        result = summary[metric]
        print(f"{metric}: {result['base']:.3f} -> {result['early']:.3f}, "
              f"delta {result['difference']:+.3f}, 95% CI {result['ci95']}")
    print(f"{summary['base']['completed_levels']} -> {summary['early']['completed_levels']} "
          "completed levels across 125 runs per arm")


if __name__ == "__main__":
    main()
