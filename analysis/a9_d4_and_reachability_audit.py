"""A9: D4 cross-level memory audit on Public-25, and the 9-suite level ledger.

Part 1 (measured here, read-only import of ../arc-agi-3/scripts/giants_cross_level_transfer.py):
  - D4 algebra: Cayley-table composition of d4_pixels on 12 grid shapes (64 x 12 = 768 checks), unique inverses,
    d4_point vs d4_pixels agreement on every pixel, d4_action direction vectors vs d4_point displacements.
  - Frame 0 of all 183 Public-25 levels, by replaying the shipped planner's own actions (series36 readout)
    in the offline engine.
  - Every unordered cross-level pair (i < j) in each game: CrossLevelMemoryBank holds level i as a won level
    (its frame 0 and the planner's level-i actions) and starts level j. Records the best D4 similarity, whether
    propose() emits an environment action (score >= .995), and the wall time of the frame-0 check.
  - Pair classes come from the 342 human replays (x0_specimens.json): reuse = same >=3-action opening prefix
    or a shared click coordinate.
  - Reachability: planner readout vs the 825 baseline runs in evidence/traces.
Part 2 (ledger): per-suite rows of Table K1 re-added; receipts are attached by path in the report.

No network, no writes outside analysis/out. Released under MIT-0.
Usage: python3.13 a9_d4_and_reachability_audit.py
"""
from __future__ import annotations
import hashlib
import json
import resource
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ARC = PAPER.parent / 'arc-agi-3'
OUT = HERE / 'out/a9_d4_and_reachability_audit.json'
READOUT = ARC / 'data/2026-10-30-giants-graft/series36_public25.json'
READOUT_35B = ARC / 'data/2026-10-30-giants-graft/series35b_public25.json'
CONFIRM = ARC / 'data/2026-10-30-giants-graft/confirm/series36_public25.json'
X0 = HERE / 'out/x0_specimens.json'
ENVS = ARC / 'out/kaggle-data/environment_files'
TRACES = PAPER / 'evidence/traces'
SHAPES = ((1, 1), (2, 3), (3, 2), (4, 4), (5, 7), (7, 5), (8, 8), (9, 16), (16, 9), (13, 21), (32, 32), (64, 64))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    sys.path.insert(0, str(ARC / 'scripts'))
    import validate_schema  # noqa: F401  (adds the arcengine wheel to sys.path)
    import giants_cross_level_transfer as t
    return t


# ---------------------------------------------------------------- D4 algebra

def d4_algebra(t):
    rng = np.random.default_rng(0)
    comp = {'checks': 0, 'passed': 0, 'closure_unique': 0}
    table = {}
    point = {'checks': 0, 'passed': 0, 'pixels_compared': 0}
    for h, w in SHAPES:
        a = rng.integers(0, 16, size=(h, w))
        imgs = [t.d4_pixels(a, g) for g in range(8)]
        for g in range(8):
            for k in range(8):
                twice = t.d4_pixels(imgs[g], k)
                hits = [m for m in range(8) if imgs[m].shape == twice.shape and np.array_equal(imgs[m], twice)]
                comp['checks'] += 1
                # closure: the composite is the image of some group element (unique once the grid has no symmetry)
                comp['passed'] += int(len(hits) >= 1)
                comp['closure_unique'] += int(len(hits) == 1)
                if len(hits) == 1 and (h, w) == (64, 64):
                    table[(g, k)] = hits[0]
            # d4_point agrees with d4_pixels on every pixel
            b = imgs[g]
            ok = True
            for y in range(h):
                for x in range(w):
                    u, v = t.d4_point(x, y, (0, 0, w - 1, h - 1), g)
                    point['pixels_compared'] += 1
                    ok &= bool(b[v, u] == a[y, x])
            point['checks'] += 1
            point['passed'] += int(ok)
    inverses = {g: [k for k in range(8) if table[(g, k)] == 0] for g in range(8)}
    direction = {'checks': 0, 'passed': 0}
    box = (0, 0, 9, 6)  # 10 x 7 board: quarter turns swap the two sides
    for g in range(8):
        for name, (dx, dy) in t._DIR.items():
            x, y = 4, 3
            p0 = t.d4_point(x, y, box, g)
            p1 = t.d4_point(x + dx, y + dy, box, g)
            moved = t.d4_action(name, box, g)
            direction['checks'] += 1
            direction['passed'] += int(t._DIR[moved] == (p1[0] - p0[0], p1[1] - p0[1]))
    return {'shapes': [list(s) for s in SHAPES],
            'cayley_composition': {**comp, 'definition': 'd4_pixels(d4_pixels(a, g), k) equals d4_pixels(a, m) for one m, 8 x 8 pairs on 12 random 16-colour grids'},
            'cayley_table_64x64': [[table[(g, k)] for k in range(8)] for g in range(8)],
            'inverses': {t.D4_NAMES[g]: [t.D4_NAMES[k] for k in v] for g, v in inverses.items()},
            'every_element_has_one_inverse': all(len(v) == 1 for v in inverses.values()),
            'point_vs_pixels': {**point, 'definition': 'd4_point(x, y) lands where d4_pixels moves pixel (y, x), all 8 elements x 12 shapes'},
            'direction_vs_point': {**direction, 'definition': 'd4_action maps each arrow to the displacement d4_point produces, 8 elements x 4 arrows'}}


# ---------------------------------------------------------------- Public-25 frames

def series36_readout():
    """series36 Public-25 rows; games rerun in confirm/ (lf52 stopped at 7/10 under load) take the confirm row."""
    rd = json.loads(READOUT.read_text())
    fix = {g['game']: g for g in json.loads(CONFIRM.read_text())['games']}
    rd['games'] = [fix.get(g['game'], g) for g in rd['games']]
    rd['replaced_from_confirm'] = sorted(fix)
    return rd


def game_path(stem):
    return next(p for p in ENVS.rglob(f'{stem}.py'))


def level_frames(readout, game):
    from validate_schema import engine, step, frame
    g = engine(game_path(game))
    f = step(g, 'RESET')
    ev = next(x for x in readout['games'] if x['game'] == game)['events']
    frames = {1: np.asarray(frame(f)['grid'], dtype=np.uint8)}
    acts = {}
    for e in ev:
        acts.setdefault(e['level'], []).append(e['action'])
        f = step(g, e['action'])
        lv = int(f.levels_completed) + 1
        if lv not in frames and f.state.name != 'WIN':
            frames[lv] = np.asarray(frame(f)['grid'], dtype=np.uint8)
    return frames, acts, int(f.levels_completed), f.state.name == 'WIN'


def pair_check(t, src_frame, src_level, src_actions, dst_frame, dst_level, valid):
    bank = t.CrossLevelMemoryBank()
    box, hud, comps, hist, _ = t.visual_geometry(src_frame)
    bank.winning_level_bank = [t.WinningLevel(src_level, src_frame.copy(), box, set(hud), comps, hist,
                                              list(src_actions), [], [src_frame.copy()])]
    t0 = time.perf_counter()
    bank._start(dst_frame, dst_level)
    dt = time.perf_counter() - t0
    bank.valid = set(valid)
    bank.status = 'NOT_FINISHED'
    a = bank.propose()
    best = bank.candidate_prefixes[0] if bank.candidate_prefixes else None
    return {'ms': dt * 1e3, 'candidates': len(bank.candidate_prefixes),
            'best_score': round(best['score'], 4) if best else None,
            'best_symmetry': best['symmetry'] if best else None,
            'proposes_action': a is not None}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def d4_pairs(t):
    readout = series36_readout()
    x0 = json.loads(X0.read_text())
    prefix = {(r['game'], r['i'], r['j']) for r in x0['human_prefix_pairs']}
    click = {(r['game'], r['i'], r['j']) for r in x0['human_click_pairs']}
    human_levels = {g: len(v) for g, v in x0['human_base'].items()}
    names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', 'MOUSE', 'UNDO']
    rows, per_game = [], {}
    for game in sorted(human_levels):
        frames, acts, done, win = level_frames(readout, game)
        L = human_levels[game]
        per_game[game] = {'levels': L, 'frames_captured': len(frames), 'planner_levels': done, 'planner_win': win}
        for i, j in combinations(range(1, L + 1), 2):
            if i not in frames or j not in frames:
                rows.append({'game': game, 'i': i, 'j': j, 'error': 'frame missing'})
                continue
            r = pair_check(t, frames[i], i, acts.get(i, []), frames[j], j, names)
            key = (game, i, j)
            rows.append({'game': game, 'i': i, 'j': j, 'reuse': key in prefix or key in click,
                         'human_prefix': key in prefix, 'human_click': key in click, 'adjacent': j == i + 1, **r})
    good = [r for r in rows if 'error' not in r]
    reuse = [r for r in good if r['reuse']]
    other = [r for r in good if not r['reuse']]
    ms = [r['ms'] for r in good]

    def tally(rs):
        return {'pairs': len(rs), 'proposes_action': sum(r['proposes_action'] for r in rs),
                'best_score_ge_0.995': sum((r['best_score'] or 0) >= .995 for r in rs),
                'no_shape_compatible_candidate': sum(r['candidates'] == 0 for r in rs),
                'median_ms': round(statistics.median([r['ms'] for r in rs]), 4) if rs else None,
                'p99_ms': round(pct([r['ms'] for r in rs], 99), 4) if rs else None}
    return {'readout': str(READOUT.relative_to(ARC)), 'readout_sha256': sha(READOUT),
            'replaced_from_confirm': readout['replaced_from_confirm'], 'confirm_sha256': sha(CONFIRM),
            'games': len(per_game), 'levels': sum(v['levels'] for v in per_game.values()),
            'frames_captured': sum(v['frames_captured'] for v in per_game.values()),
            'unordered_pairs': len(rows), 'pair_errors': len(rows) - len(good),
            'adjacent_pairs': sum(r.get('adjacent', False) for r in good),
            'human_prefix_pairs': len(prefix), 'human_click_pairs': len(click),
            'human_reuse_union': len(prefix | click), 'human_prefix_and_click': len(prefix & click),
            'reuse': tally(reuse), 'non_reuse': tally(other),
            'adjacent': tally([r for r in good if r['adjacent']]),
            'all': {'median_ms': round(statistics.median(ms), 4), 'p99_ms': round(pct(ms, 99), 4), 'max_ms': round(max(ms), 4)},
            'non_reuse_pairs_that_propose': [{k: r[k] for k in ('game', 'i', 'j', 'best_score', 'best_symmetry')}
                                             for r in other if r['proposes_action']],
            'reuse_pairs_that_propose': [{k: r[k] for k in ('game', 'i', 'j', 'best_score', 'best_symmetry')}
                                         for r in reuse if r['proposes_action']],
            'timing_note': 'perf_counter wall time of CrossLevelMemoryBank._start (visual_geometry + _rank over 8 isometries) per pair, '
                           'measured while other jobs loaded the machine; one source level held in the bank.',
            'per_game': per_game, 'pairs': rows}


# ---------------------------------------------------------------- reachability and action ratio

def reachability(pairs_block):
    runs = []
    for f in sorted(TRACES.glob('*.jsonl')):
        runs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    best = {}
    for r in runs:
        g = r['game'].split('-')[0]
        best[g] = max(best.get(g, 0), int(r['k']))
    reuse = [r for r in pairs_block['pairs'] if r.get('reuse')]
    planner = pairs_block['per_game']
    out = {'baseline_runs': len(runs), 'baseline_files': sorted(f.name for f in TRACES.glob('*.jsonl')),
           'reuse_pairs': len(reuse),
           'planner_reaches_level_j': sum(planner[r['game']]['planner_levels'] >= r['j'] - 1 for r in reuse),
           'planner_clears_level_j': sum(planner[r['game']]['planner_levels'] >= r['j'] for r in reuse),
           'baseline_any_run_reaches_level_j': sum(best.get(r['game'], 0) >= r['j'] - 1 for r in reuse),
           'baseline_any_run_clears_level_j': sum(best.get(r['game'], 0) >= r['j'] for r in reuse),
           'baseline_any_run_clears_level_i': sum(best.get(r['game'], 0) >= r['i'] for r in reuse),
           'baseline_best_levels_per_game': best}
    x0 = json.loads(X0.read_text())
    human = {g: sum(v) for g, v in x0['human_base'].items()}
    for tag, rd in (('series36_with_confirm', series36_readout()), ('series35b', json.loads(READOUT_35B.read_text()))):
        acts = {g['game']: g['actions'] for g in rd['games']}
        lv = {g['game']: (g['levels_completed'], g['win']) for g in rd['games']}
        out[f'action_ratio_{tag}'] = {
            'planner_actions': sum(acts.values()), 'human_baseline_actions': sum(human.values()),
            'ratio': round(sum(acts.values()) / sum(human.values()), 4),
            'levels': sum(v[0] for v in lv.values()), 'wins': sum(v[1] for v in lv.values()),
            'right_censored_levels': sum(len(x0['human_base'][g]) - lv[g][0] for g in lv)}
    return out


# ---------------------------------------------------------------- 9-suite ledger (Table K1 rows)

SUITES = [  # name, games, solvable, unreachable_games, unreachable_levels, baseline_levels, baseline_wins,
    #           giants_levels at 1,200, wins, reachable_games; baseline of the three held-out suites = frozen
    #           pre-exposure snapshot (universality_results.json `full`)
    ('Public-25', 25, 183, 0, 0, 42, 0, 183, 25, 25),
    ('Parametric-25', 25, 172, 0, 0, 48, 5, 172, 25, 25),
    ('Tufa-50', 50, 232, 0, 0, 128, 27, 232, 50, 50),
    ('Ouellette-Tail28', 28, 144, 0, 0, 45, 5, 144, 28, 28),
    ('Community Siblings-27', 27, 117, 6, 26, 46, 8, 116, 20, 21),
    ('Clone Families-63', 63, 291, 12, 59, 87, 11, 291, 51, 51),
    ('Mother-1102', 24, 118, 4, 11, 32, 2, 118, 20, 20),
    ('Tufa-Web-1102', 25, 141, 0, 0, 10, 0, 141, 25, 25),
    ('Turducken-49', 49, 396, 0, 0, 30, 0, 396, 49, 49),
]


# Same rows as read from the per-game receipts (paths relative to ../arc-agi-3 unless absolute).
# indexed, deadlocked games, unreachable levels, solvable, baseline levels/wins (None = no receipt),
# giants levels/wins at 600 actions (None = no 600 receipt) and at 1200 actions, reachable games.
RECEIPTS = [
    ('Public-25', 25, 183, 0, 0, 183, 42, 0, None, None, 183, 25, 25,
     'evidence/traces/kaggle_v2.jsonl (paper repo); data/2026-10-30-giants-graft/series35b_public25.json'),
    ('Parametric-25', 25, 172, 0, 0, 172, None, None, None, None, 172, 25, 25,
     'data/2026-10-30-giants-graft/series30_trinity_delivery.json (summary only)'),
    ('Tufa-50', 50, 232, 0, 0, 232, 128, 27, 232, 50, 232, 50, 50,
     'data/2026-10-30-giants-graft/series36_tufa25.json + series36_tufa_hold25.json; analysis/out/a9_d4_and_reachability_audit.json#archived/astra32/baseline_tufa*.json'),
    ('Ouellette-Tail28', 28, 144, 0, 0, 144, 45, 5, 141, 26, 144, 28, 28,
     'analysis/out/a9_d4_and_reachability_audit.json#archived/astra32b/g_suite_s0_b{600,1200}_tail28.json; analysis/out/a9_d4_and_reachability_audit.json#archived/astra31/base_tail28.json'),
    ('Community Siblings-27', 27, 143, 6, 26, 117, 46, 8, 116, 20, 116, 20, 21,
     'analysis/out/a9_d4_and_reachability_audit.json#archived/astra32b/g_suite_s0_b*_siblings27.json; analysis/out/a9_d4_and_reachability_audit.json#archived/astra32/clone_level_upper_bound.json + clone_exact_transition_audit.json'),
    ('Clone Families-63', 63, 350, 12, 59, 291, 87, 11, 291, 51, 291, 51, 51,
     'analysis/out/a9_d4_and_reachability_audit.json#archived/astra32b/g_suite_s0_b*_families{31,32b}.json; analysis/out/a9_d4_and_reachability_audit.json#archived/astra31/base_families31.json + before_families32b.json'),
    ('Mother-1102', 24, 129, 4, 11, 118, 32, 2, None, None, 118, 20, 20,
     'data/2026-10-30-giants-graft/series36_mother49_verified.json (cohort A); analysis/out/a9_d4_and_reachability_audit.json#archived/mother1102/clone_certificates.json'),
    ('Tufa-Web-1102', 25, 141, 0, 0, 141, 10, 0, None, None, 141, 25, 25,
     'data/2026-10-30-giants-graft/series36_mother49_verified.json (cohort B); analysis/out/a9_d4_and_reachability_audit.json#archived/mother1102_b/delivery.json'),
    ('Turducken-49', 49, 396, 0, 0, 396, 30, 0, None, None, 396, 49, 49,
     'out/mother-hunt-1102/turducken_r6_aggregate.json; analysis/out/a9_d4_and_reachability_audit.json#archived/'
     'universality_results.json (turducken49 full, snapshot 2026-11-02T09:07+09:00)'),
]
DEADLOCKED = {
    'Community Siblings-27': ['sk02 5->2', 'sk03 5->3', 'sk04 7->0', 'pb04 7->0', 'gp02 5->3', 'gp04 7->2'],
    'Clone Families-63': ['ph04 7->1', 'pt02 5->1', 'pt03 5->0', 'pt04 7->2', 'pt05 7->0', 'rp02 5->1', 'rp03 5->0',
                          'rp04 7->4', 'wm02 5->0', 'wm03 5->0', 'fl01 5->2', 'kv04 7->0'],
    'Mother-1102': ['ff04 7->5', 'pd01 5->2', 'pj04 7->2', 'tc01 5->4'],
}


def receipt_ledger():
    cols = ('games', 'indexed', 'deadlocked_games', 'unreachable_levels', 'solvable', 'baseline_levels', 'baseline_wins',
            'giants_levels_600', 'giants_wins_600', 'giants_levels_1200', 'giants_wins_1200', 'reachable_games', 'receipts')
    rows = [dict(zip(('suite',) + cols, s)) for s in RECEIPTS]
    tot = {c: sum(r[c] for r in rows if isinstance(r[c], int)) for c in cols if c != 'receipts'}
    return {'rows': rows, 'totals_where_receipted': tot, 'deadlocked_games': DEADLOCKED,
            'level_rate_1200': round(tot['giants_levels_1200'] / tot['solvable'], 6),
            'clone118_600': {'levels': 141 + 116 + 291, 'wins': 26 + 20 + 51},
            'clone118_1200': {'levels': 144 + 116 + 291, 'wins': 28 + 20 + 51},
            'tt03': {'suite': 'Community Siblings-27', 'receipt': 'analysis/out/a9_d4_and_reachability_audit.json#archived/astra32b/tt_g_long.json', 'budget': 5000,
                     'actions_whole_game': 3471, 'per_level': [29, 143, 3299]},
            'duplicate_entry': {'game': 'wc01', 'levels': 8, 'suites': ['Tufa-Web-1102', 'Turducken-49'],
                                'distinct_games': 315, 'distinct_solvable_levels': 1786},
            'budget_note': 'Every 1,793-level total uses 1,200 actions per game; no 600-action receipt exists for '
                           'Public-25, Parametric, Mother, Web or Turducken, and Clone-118 drops to 548 levels / 97 wins at 600.'}


def ledger():
    cols = ('games', 'solvable', 'unreachable_games', 'unreachable_levels', 'baseline_levels', 'baseline_wins',
            'giants_levels', 'giants_wins', 'reachable_games')
    rows = [dict(zip(('suite',) + cols, s)) for s in SUITES]
    tot = {c: sum(r[c] for r in rows) for c in cols}
    for r in rows:
        r['reachable_games_check'] = r['games'] - r['unreachable_games'] == r['reachable_games']
    rec = receipt_ledger()
    by_suite = {r['suite']: r for r in rec['rows']}
    return {'as_printed_in_paper': rows, 'totals': tot, 'indexed_levels': tot['solvable'] + tot['unreachable_levels'],
            'from_receipts': rec,
            'baseline_rows_match_receipts': all(
                by_suite[r['suite']]['baseline_levels'] in (None, r['baseline_levels']) and
                by_suite[r['suite']]['baseline_wins'] in (None, r['baseline_wins']) for r in rows),
            'baseline_level_rate': round(tot['baseline_levels'] / tot['solvable'], 4),
            'giants_level_rate_cap': round(tot['giants_levels'] / tot['solvable'], 6),
            'table_k1_printed_baseline_total': {'levels': 468, 'wins': 58},
            'table_k1_baseline_total_correct': (tot['baseline_levels'], tot['baseline_wins']) == (468, 58),
            'table2_groups': {'Unseen Clone-118': {c: sum(r[c] for r in rows[3:6]) for c in cols},
                              'Mother+Web+Turducken': {c: sum(r[c] for r in rows[6:9]) for c in cols}}}


def verify_archive(out):
    """Check the full archived ledger, pair accounting, and local trace reachability."""
    assert out['ledger_9_suites'] == ledger(), 'Nine-suite ledger mismatch'
    a, p = out['d4_algebra'], out['d4_pairs_public25']
    table = a['cayley_table_64x64']
    assert all(sorted(row) == list(range(8)) for row in table)
    assert all(sorted(row) == list(range(8)) for row in zip(*table))
    assert all(table[table[x][y]][z] == table[x][table[y][z]]
               for x in range(8) for y in range(8) for z in range(8))
    assert a['cayley_composition']['checks'] == a['cayley_composition']['passed'] == 768
    assert a['point_vs_pixels']['checks'] == a['point_vs_pixels']['passed'] == 96
    assert a['direction_vs_point']['checks'] == a['direction_vs_point']['passed'] == 32
    rows = p['pairs']
    assert len(rows) == p['unordered_pairs'] == 597 and p['pair_errors'] == 0
    assert len({(r['game'], r['i'], r['j']) for r in rows}) == len(rows)
    assert sum(r['reuse'] for r in rows) == p['human_reuse_union'] == 378
    assert sum(not r['reuse'] for r in rows) == p['non_reuse']['pairs'] == 219
    assert sum(r['proposes_action'] for r in rows if not r['reuse']) == p['non_reuse']['proposes_action'] == 0
    assert sum(r['levels'] for r in p['per_game'].values()) == p['levels'] == 183
    assert sum(r['planner_levels'] for r in p['per_game'].values()) == 183
    best, count = {}, 0
    for path in sorted(TRACES.glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line); count += 1
                game = r['game'].split('-')[0]
                best[game] = max(best.get(game, 0), int(r['k']))
    reach = out['reachability_public25']
    assert count == reach['baseline_runs'] == 825
    assert best == reach['baseline_best_levels_per_game']
    assert sum(best.get(r['game'], 0) >= r['j'] - 1 for r in rows if r['reuse']) == reach['baseline_any_run_reaches_level_j']
    ratio = reach['action_ratio_series35b']
    assert ratio['ratio'] == round(ratio['planner_actions'] / ratio['human_baseline_actions'], 4)
    assert (ratio['levels'], ratio['wins'], ratio['right_censored_levels']) == (183, 25, 0)
    return {'cayley_checks': 768, 'pairs': len(rows), 'baseline_runs': count,
            'solvable_levels': out['ledger_9_suites']['totals']['solvable'],
            'cleared_levels': out['ledger_9_suites']['totals']['giants_levels']}


def main():
    if not (ARC / 'scripts/validate_schema.py').is_file():
        out = json.loads(OUT.read_text())
        print(json.dumps({'reproduction_mode': 'archived_ledger_and_trace_recalculation',
                          'archive': str(OUT.relative_to(PAPER)), 'sha256': sha(OUT),
                          'verified': verify_archive(out)}))
        return
    t0 = time.perf_counter()
    t = setup()
    out = {'d4_algebra': d4_algebra(t)}
    out['d4_pairs_public25'] = d4_pairs(t)
    out['reachability_public25'] = reachability(out['d4_pairs_public25'])
    out['ledger_9_suites'] = ledger()
    out['meta'] = {'script_sha256': sha(__file__), 'transfer_sha256': sha(ARC / 'scripts/giants_cross_level_transfer.py'),
                   'x0_sha256': sha(X0), 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'runtime_s': round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    p = out['d4_pairs_public25']
    print(json.dumps({'cayley': out['d4_algebra']['cayley_composition'], 'point': out['d4_algebra']['point_vs_pixels'],
                      'dir': out['d4_algebra']['direction_vs_point'], 'pairs': p['unordered_pairs'], 'errors': p['pair_errors'],
                      'reuse': p['reuse'], 'non_reuse': p['non_reuse'], 'all': p['all'],
                      'reach': {k: v for k, v in out['reachability_public25'].items() if k != 'baseline_best_levels_per_game'}}, indent=1))


if __name__ == '__main__':
    main()
