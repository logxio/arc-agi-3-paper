"""X0: real specimens for the three main figures, extracted once into analysis/out/x0_specimens.json.

Reads the method repository (path in the first argument, default ../arc-agi-3):
  - the Public-25 readout of the certificate-gated planner (per-step actions with level and solver reason),
  - the 342 official human replays (per-level human solutions),
  - the official Kaggle submission receipts.
Replays the planner's own actions in the offline engine to record the first frame of every level of the
specimen game, so the figure draws real pixels. Run with a Python that has arc_agi and arcengine installed.
Released under MIT-0.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import OUT, load

ARC3 = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'arc-agi-3')
READOUT = ARC3 / 'data/2026-10-30-giants-graft/series35b_public25.json'
HUMAN = ARC3 / 'data/2026-10-31-cross-level-invariants/human_solution_candidates.json'
RECEIPTS = ARC3 / 'out/mother-hunt-1102/our_submissions_live.csv'
CENSUS = ARC3 / 'out/mother-hunt-1102/mother_genealogy_and_channels.json'
INVERSION = ARC3 / 'out/mother-hunt-1102/receipt_differential_inversion.json'
ENVS = ARC3 / 'out/kaggle-data/environment_files'
SPECIMEN = 'tn36'
NAMES = {'UP': 'ACTION1', 'DOWN': 'ACTION2', 'LEFT': 'ACTION3', 'RIGHT': 'ACTION4', 'SPACE': 'ACTION5',
         'MOUSE': 'ACTION6', 'RESET': 'RESET'}


def human_base():
    base = {}
    for r in load('kaggle_v2'):
        base[r['game'].split('-')[0]] = r['base']
    return base


def planner_levels(readout):
    """Per game: actions per level, solver reason of the level's first step."""
    out = {}
    for g in readout['games']:
        per, why = defaultdict(int), {}
        for e in g['events']:
            per[e['level']] += 1
            why.setdefault(e['level'], e['reason'])
        L = max(per)
        out[g['game']] = {'levels': g['levels_completed'], 'win': g['win'], 'actions': g['actions'],
                          'per_level': [per[l] for l in range(1, L + 1)],
                          'reason': [why[l] for l in range(1, L + 1)]}
    return out


def specimen_frames(readout, game):
    import arc_agi
    from arcengine import GameAction
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENVS))
    gid = next(e.game_id for e in arcade.available_environments if e.game_id.startswith(game))
    env = arcade.make(gid)
    fd = env.reset()
    events = next(g for g in readout['games'] if g['game'] == game)['events']
    frames = {1: np.array(fd.frame[-1]).tolist()}
    steps = []
    for e in events:
        a = e['action']
        name = a if isinstance(a, str) else a['action']
        data = {} if isinstance(a, str) or name != 'MOUSE' else {'x': a['col'], 'y': a['row']}
        fd = env.step(GameAction[NAMES[name]], data=data)
        steps.append({'level': e['level'], 'action': name, **({'row': a['row'], 'col': a['col']} if data else {})})
        if fd.levels_completed + 1 not in frames and fd.levels_completed < len(set(x['level'] for x in events)):
            frames[fd.levels_completed + 1] = np.array(fd.frame[-1]).tolist()
    assert fd.levels_completed == max(x['level'] for x in events), fd.levels_completed
    return {'game_id': gid, 'level_frames': frames, 'steps': steps}


def human_pairs(human):
    """Cross-level pairs in the human replays: same opening prefix (>= 3 actions) and same click coordinate."""
    prefix, click = defaultdict(int), defaultdict(int)
    for game, levels in human['games'].items():
        sols = {int(l): [tuple(a) for s in v for a in [s['actions']]] for l, v in levels.items()}
        for i in sorted(sols):
            for j in sorted(sols):
                if j <= i:
                    continue
                pre = any(len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3] for a in sols[i] for b in sols[j])
                ci = {tuple(x) for a in sols[i] for x in a if len(x) == 3}
                cj = {tuple(x) for b in sols[j] for x in b if len(x) == 3}
                if pre:
                    prefix[(game, i + 1, j + 1)] += 1
                if ci & cj:
                    click[(game, i + 1, j + 1)] += 1
    return ([{'game': g, 'i': i, 'j': j} for g, i, j in sorted(prefix)],
            [{'game': g, 'i': i, 'j': j} for g, i, j in sorted(click)])


def setter_fingerprints(human):
    """Setter fingerprints: games whose every level declares one grid_size in its source, levels whose human
    recordings click one (x, y) more than once, and the number of cross-level level pairs in the replays."""
    grids = {}
    for f in sorted(ENVS.glob('*/*/*.py')):
        sizes = set(re.findall(r'grid_size=\((\d+),\s*(\d+)\)', f.read_text()))
        if sizes:
            grids[f.parent.parent.name] = len(sizes)
    click_levels = repeat = pairs = 0
    for levels in human['games'].values():
        pairs += len(levels) * (len(levels) - 1) // 2
        for sols in levels.values():
            cl = [[tuple(x) for x in s['actions'] if len(x) == 3] for s in sols]
            if any(cl):
                click_levels += 1
                repeat += any(max(Counter(c).values()) > 1 for c in cl if c)
    return {'games': len(grids), 'constant_grid_games': sum(v == 1 for v in grids.values()),
            'click_levels': click_levels, 'repeat_click_levels': repeat, 'cross_level_pairs': pairs}


def census():
    """Action-route census of the 906 game sources: per route its name and source count, per cohort its routes."""
    d = json.loads(CENSUS.read_text())
    per = defaultdict(lambda: defaultdict(int))
    for g in d['games']:
        for c in g['cohorts']:
            per[c][g['channel']] += 1
    return {'routes': {k: {'name': v['name'], 'sources': v['source_files']} for k, v in d['channels'].items()},
            'cohort_routes': {c: dict(v) for c, v in per.items()}, 'sources': d['scope']['source_files']}


def receipts():
    rows = []
    with open(RECEIPTS) as f:
        for r in csv.DictReader(l for l in f if not l.startswith('Warning')):
            rows.append({'ref': int(r['ref']), 'date': r['date'][:10], 'status': r['status'].split('.')[-1],
                         'score': float(r['publicScore']) if r['publicScore'] else None,
                         'description': r['description']})
    return rows[::-1]


def main():
    readout = json.loads(READOUT.read_text())
    human = json.loads(HUMAN.read_text())
    pre, clk = human_pairs(human)
    out = {
        'readout_summary': readout['summary'],
        'planner': planner_levels(readout),
        'human_base': human_base(),
        'specimen': specimen_frames(readout, SPECIMEN),
        'human_prefix_pairs': pre,
        'human_click_pairs': clk,
        'human_files': human['files'],
        'receipts': receipts(),
        'census': census(),
        'fingerprints': setter_fingerprints(human),
        'inversion': {k: {'ref': v['ref'], 'baseline_ref': v['baseline_ref'], 'delta': v['delta_points']['decimal'],
                          'full_game_equivalents': v['net_full_game_equivalents']['decimal'],
                          'min_games_changed': v['minimum_games_with_changed_scores']}
                      for k, v in json.loads(INVERSION.read_text())['differentials'].items()},
    }
    (OUT / 'x0_specimens.json').write_text(json.dumps(out))
    print('planner levels', sum(v['levels'] for v in out['planner'].values()), 'wins',
          sum(v['win'] for v in out['planner'].values()))
    print('specimen frames', sorted(out['specimen']['level_frames']))
    print('human prefix pairs', len(pre), 'click pairs', len(clk), 'fingerprints', out['fingerprints'])
    print('receipts', len(out['receipts']), 'scored', sum(r['score'] is not None for r in out['receipts']))


if __name__ == '__main__':
    main()
