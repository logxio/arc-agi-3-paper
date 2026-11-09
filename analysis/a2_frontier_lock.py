"""A2: Frontier Lock = share of actions (and seconds, generated tokens) spent on the
final unsolved level, action-weighted over runs, with game-bootstrap 95% intervals."""
from common import DATASETS, LABELS, dump, game_bootstrap, load


def action_share(runs):
    front = sum(r["a"][r["k"]] for r in runs if r["k"] < r["L"])
    total = sum(sum(r["a"]) for r in runs)
    return front / total if total else None


def timed(runs):
    # Seconds need a trustworthy run end; runs cut when a batch stopped are excluded.
    return [r for r in runs if r.get("t") is not None and len(r["t"]) == r["k"]
            and r["state"] == "gave_up" and (r.get("t_final") or r.get("t_last")) and r.get("t_first") is not None]


def time_share(runs):
    front = total = 0.0
    for r in timed(runs):
        end = r.get("t_final") or r["t_last"]
        # Clock from the first action: queued batches stamp time from batch start.
        start_front = r["t"][-1] if r["k"] else r["t_first"]
        if r["k"] < r["L"]:
            front += max(end - start_front, 0.0)
        total += max(end - r["t_first"], 0.0)
    return front / total if total else None


def token_share(runs):
    front = total = 0
    for r in runs:
        tok = r.get("tok")
        if not tok or r["k"] >= r["L"] or len(tok) != r["k"] + 1:
            continue
        front += tok[r["k"]]
        total += sum(tok)
    return front / total if total else None


res = {}
print(f"{'dataset':18s} {'runs':>4s}  actions            seconds            tokens")
for name in DATASETS:
    runs = load(name)
    row = {"runs": len(runs), "actions": sum(sum(r["a"]) for r in runs),
           "frontier_actions": sum(r["a"][r["k"]] for r in runs if r["k"] < r["L"])}
    out = []
    for key, f in (("action_share", action_share), ("time_share", time_share),
                   ("token_share", token_share)):
        if f(runs) is None:
            row[key] = None
            out.append(" " * 18)
            continue
        p, lo, hi = game_bootstrap(runs, f)
        row[key] = [p, lo, hi]
        out.append(f"{p:.3f} [{lo:.3f},{hi:.3f}]")
    row["timed_runs"] = len(timed(runs))
    res[name] = row
    print(f"{name:18s} {len(runs):4d}  " + "  ".join(out))
dump("a2_frontier_lock", res)
