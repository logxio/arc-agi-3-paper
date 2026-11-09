"""Component ablation ladder, Hidden-110 significance, and wording scan.

Part 1 reruns the current GiantsPlanner source (read-only import from ../arc-agi-3) on
the local suites under four arms that remove components one at a time:
  full        every solver as shipped
  no_astra    astra_* suite solvers removed (incl. the wrappers that call them)
  generic     every per-family exact solver removed; general observation models,
              CoupledMechanismSolver and CrossLevelMemoryBank remain
  generic_nod4  generic minus CrossLevelMemoryBank
Part 2 recomputes the Hidden-110 statistics from the official receipt ledger.
Part 3 scans paper/*.md for wording flagged by the blind-review seats.

Single process, CPU clock budget per game, no network, no writes outside analysis/out.
Usage: python3.13 compute_ablation_ladder.py [--suites public25,ouellette_unseen25,...] [--arms ...]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ARC = PAPER.parent / 'arc-agi-3'
OUT = HERE / 'out/ablation_ladder.json'
LEGACY_OUT = HERE / 'out/component_ablation_ladder.json'
ARMS = ('full', 'no_astra', 'generic', 'generic_nod4')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- Part 1: ladder

def setup_imports():
    sys.path.insert(0, str(ARC / 'scripts'))
    sys.path.insert(1, str(ARC / 'out/train-data/ouellette'))
    import validate_schema  # noqa: F401  (adds the arcengine wheel to sys.path)


class NoBank:
    """CrossLevelMemoryBank stand-in that never observes and never proposes."""
    def __init__(self):
        self.metrics = {}; self.level = None; self.last_action = None

    def observe(self, *a, **k):
        return None

    def propose(self):
        return None


def is_astra(m):
    import giants_world_model_planner as gwp
    if type(m).__module__.startswith('astra'):
        return True
    if isinstance(m, gwp.TemplateExact):
        return True
    inner = getattr(m, 'solver', None)
    return inner is not None and type(inner).__module__.startswith('astra')


def make_planner(arm):
    from giants_world_model_planner import GiantsPlanner
    p = GiantsPlanner()
    if arm in ('no_astra',):
        p.exact = [m for m in p.exact if not is_astra(m)]
    elif arm in ('generic', 'generic_nod4'):
        p.exact = [p.coupled]
    if arm != 'full':
        p.mother_exact = tuple(m for m in p.mother_exact if m in p.exact)
    if arm == 'generic_nod4':
        p.bank = NoBank()
    return p


def exact_inventory():
    p = make_planner('full')
    rows = [{'class': type(m).__name__ if not hasattr(m, 'solver') else type(m).__name__ + '(' + type(m.solver).__name__ + ')',
             'module': type(getattr(m, 'solver', m)).__module__, 'astra': is_astra(m)} for m in p.exact]
    return {'exact_chain_length': len(rows), 'astra_entries': sum(r['astra'] for r in rows),
            'non_astra_family_entries': sum(not r['astra'] for r in rows) - 1,
            'generic_kept': ['CoupledMechanismSolver'], 'chain': rows}


def suite_paths(suite):
    public = sorted((ARC / 'out/kaggle-data/environment_files').rglob('*.py'))
    if suite == 'public25':
        return public
    names = {p.stem for p in public}
    if suite.startswith('ouellette'):
        unseen = sorted(p for p in (ARC / 'out/train-data/ouellette/games').glob('*/*.py')
                        if p.stem not in names and p.stem != '__init__')
        return unseen[:25] if suite == 'ouellette_unseen25' else unseen[25:50]
    raise KeyError(suite)


def load_game(suite, path):
    if suite == 'public25':
        from validate_schema import engine
        return engine(path)
    import eval_unseen_ouellette as eo
    return eo.load_engine(path)


PROTOCOL = {'public25': (1200, 12.0), 'ouellette_unseen25': (600, 3.0), 'ouellette_holdout25': (600, 3.0)}


def play(suite, path, arm):
    import numpy as np
    from validate_schema import step, frame, valid
    steps, seconds = PROTOCOL[suite]
    random.seed(0); np.random.seed(0)
    g = load_game(suite, path); f = step(g, 'RESET')
    planner = make_planner(arm)
    t0 = time.process_time(); max_level = 0; actions = 0; reasons = {}
    for _ in range(steps):
        obs = json.loads(json.dumps({'pixels': frame(f)['grid'], 'valid_actions': valid(f),
                                     'level': int(f.levels_completed) + 1, 'status': f.state.name}))
        planner.observe(**obs)
        if f.state.name == 'WIN' or time.process_time() - t0 > seconds:
            break
        a = planner.act()
        if a is None:
            break
        r = planner.reason.split(' ')[0] if isinstance(planner.reason, str) else str(planner.reason)
        reasons[r] = reasons.get(r, 0) + 1
        f = step(g, a); actions += 1; max_level = max(max_level, int(f.levels_completed))
    total = getattr(g, '_levels', None)
    return {'game': path.stem, 'levels_completed': max_level, 'win': f.state.name == 'WIN',
            'actions': actions, 'cpu_s': round(time.process_time() - t0, 3),
            'total_levels': len(total) if total is not None else None,
            'top_reasons': dict(sorted(reasons.items(), key=lambda x: -x[1])[:4])}


def run_ladder(suites, arms, prior):
    setup_imports()
    runs = prior.get('ladder_runs', {})
    for suite in suites:
        paths = suite_paths(suite)
        for arm in arms:
            key = f'{suite}/{arm}'
            rows = []
            for p in paths:
                try:
                    rows.append(play(suite, p, arm))
                except Exception as e:  # surfaced in the receipt, never silently dropped
                    rows.append({'game': p.stem, 'error': repr(e)[:300], 'levels_completed': 0, 'win': False})
                print(json.dumps({'suite': suite, 'arm': arm, **{k: rows[-1].get(k) for k in ('game', 'levels_completed', 'win', 'cpu_s')}}), flush=True)
            runs[key] = {'suite': suite, 'arm': arm, 'budget_actions': PROTOCOL[suite][0],
                         'budget_cpu_s': PROTOCOL[suite][1], 'games': rows,
                         'levels': sum(r['levels_completed'] for r in rows),
                         'wins': sum(bool(r['win']) for r in rows),
                         'errors': sum('error' in r for r in rows)}
            prior['ladder_runs'] = runs
            OUT.write_text(json.dumps(prior, indent=1, ensure_ascii=False) + '\n')
    return runs


# ---------------------------------------------------------------- Part 2: hidden-110

def hidden_stats(base=None):
    import numpy as np
    from scipy import stats
    if base is None:
        inv = json.loads((ARC / 'out/mother-hunt-1102/receipt_differential_inversion.json').read_text())
        base = inv['A_native_baseline']
    x = np.array(base['scores'], dtype=float); n = len(x); mu = x.mean(); s = x.std(ddof=1)
    tcrit = stats.t.ppf(0.975, n - 1)
    duck_peak, duck_mean, duck_sd, duck_n = 3.41, 2.81, 0.66, 6

    def pred_t(v):
        """Single new run v against the 12-run distribution: prediction t with df=n-1."""
        z = (v - mu) / s
        t = (v - mu) / (s * math.sqrt(1 + 1 / n))
        return {'score': v, 'delta_vs_mean': round(v - mu, 4), 'z_sd_units': round(z, 4),
                'prediction_t': round(t, 4), 'df': n - 1,
                'p_two_sided_prediction_t': float(2 * stats.t.sf(abs(t), n - 1)),
                'p_one_sided_below': float(stats.t.cdf(t, n - 1)),
                'p_two_sided_normal_z': float(2 * stats.norm.sf(abs(z))),
                'below_min_of_12': bool(v < x.min()), 'above_max_of_12': bool(v > x.max())}

    out = {'backbone_12': {'refs': base['refs'], 'scores': base['scores'], 'n': n, 'mean': round(mu, 4),
                           'sd': round(s, 4), 'min': float(x.min()), 'max': float(x.max()),
                           'ci95_mean': [round(mu - tcrit * s / math.sqrt(n), 3), round(mu + tcrit * s / math.sqrt(n), 3)],
                           'diff_noise_sd_two_single_runs': round(s * math.sqrt(2), 4)}}
    # Duck V2: only peak 3.41 and mean/sd 2.81±0.68 (n=6) survive in the ledger; individual scores are not stored.
    d_mean = mu - duck_mean
    se = math.sqrt(s ** 2 / n + duck_sd ** 2 / duck_n)
    dfw = se ** 4 / ((s ** 2 / n) ** 2 / (n - 1) + (duck_sd ** 2 / duck_n) ** 2 / (duck_n - 1))
    tw = d_mean / se
    sp = math.sqrt(((n - 1) * s ** 2 + (duck_n - 1) * duck_sd ** 2) / (n + duck_n - 2))
    # Exact Mann-Whitney: every 36K run (min 21.22) exceeds the Duck peak 3.41, so U=72 = n1*n2 regardless of the missing per-run Duck values.
    u = n * duck_n
    p_mw = 2 / math.comb(n + duck_n, duck_n)
    out['backbone_vs_duck_v2'] = {
        'inputs': {'w4a16_36k': f'{mu:.2f}±{s:.2f} (n={n})', 'duck_v2': f'{duck_mean}±{duck_sd} (n={duck_n}, peak {duck_peak})'},
        'mean_difference': round(d_mean, 3), 'welch_t': round(tw, 4), 'welch_df': round(dfw, 2),
        'welch_p_two_sided': float(2 * stats.t.sf(abs(tw), dfw)),
        'cohen_d_pooled': round(d_mean / sp, 2), 'hedges_g': round(d_mean / sp * (1 - 3 / (4 * (n + duck_n) - 9)), 2),
        'mann_whitney_U': u, 'mann_whitney_p_two_sided_exact': p_mw,
        'note_mw': 'U = n1*n2 = 72 follows from min(36K) 21.22 > max(Duck) 3.41; p is the exact complete-separation value 2/C(18,6).',
        'peak_ratio': round(28.38 / duck_peak, 3), 'mean_ratio': round(mu / duck_mean, 3)}
    single = {'14k_cap_57774685': 9.39, '14k_convex_kill_57786717': 7.72, 'open_loop_600_57854657': 20.76,
              'reactive_224_57894564': 21.08, 'cert_gated_57827241': 25.70}
    out['single_runs_vs_backbone'] = {k: pred_t(v) for k, v in single.items()}
    # Pairwise single-run differences: noise of a difference of two independent single runs is s*sqrt(2).
    def diff(a, b, label):
        d = single[a] - single[b]; sd2 = s * math.sqrt(2); z = d / sd2
        return {'pair': label, 'delta': round(d, 2), 'delta_in_backbone_sd': round(d / s, 3),
                'z_two_run_noise': round(z, 3), 'p_one_sided_t11': float(stats.t.sf(z, n - 1)),
                'p_two_sided_t11': float(2 * stats.t.sf(abs(z), n - 1))}
    out['pairwise'] = [diff('cert_gated_57827241', 'open_loop_600_57854657', 'cert-gated 25.70 vs open-loop 20.76'),
                       diff('cert_gated_57827241', 'reactive_224_57894564', 'cert-gated 25.70 vs reactive-224 21.08'),
                       diff('reactive_224_57894564', 'open_loop_600_57854657', 'reactive-224 21.08 vs open-loop 20.76'),
                       diff('14k_convex_kill_57786717', '14k_cap_57774685', 'convex kill 7.72 vs 14K 9.39')]
    # Exact rank test: are the two uncertified-bridge runs exchangeable with the 12 backbone runs?
    unc = [single['open_loop_600_57854657'], single['reactive_224_57894564']]
    pooled = sorted(list(x) + unc)
    ranks = [pooled.index(v) + 1 for v in unc]
    from itertools import combinations
    obs = sum(ranks)
    le = sum(1 for c in combinations(range(1, len(pooled) + 1), 2) if sum(c) <= obs)
    cert = single['cert_gated_57827241']
    out['rank_tests'] = {
        'uncertified_two_runs_vs_backbone_12': {
            'ranks_among_14': ranks, 'rank_sum': obs, 'mann_whitney_U': obs - 3,
            'p_one_sided_exact': le / math.comb(len(pooled), 2),
            'reading': 'Both uncertified-bridge runs (20.76, 21.08) fall below all 12 backbone runs (min 21.22).'},
        'cert_gated_rank_among_13': {'rank': sorted(list(x) + [cert]).index(cert) + 1, 'of': n + 1,
                                     'p_two_sided_exact': 2 * min(sorted(list(x) + [cert]).index(cert) + 1, n + 1 - sorted(list(x) + [cert]).index(cert)) / (n + 1),
                                     'reading': 'The certificate-gated run sits inside the backbone distribution.'}}
    # The brief's own convention (delta / s, treating the comparison arm as the 12-run mean) for the record.
    out['brief_convention_check'] = {
        'cert_vs_open_loop_delta_over_s': round((25.70 - 20.76) / s, 3),
        'p_one_sided_normal_at_that_z': float(stats.norm.sf((25.70 - 20.76) / s)),
        'cert_vs_reactive_delta_over_s': round((25.70 - 21.08) / s, 3),
        'p_one_sided_normal_at_that_z_reactive': float(stats.norm.sf((25.70 - 21.08) / s)),
        'cert_vs_mean_delta_over_s': round((25.70 - mu) / s, 3)}
    return out


# ---------------------------------------------------------------- Part 3: wording

TERMS = [
    (r'\bcausal laws?\b', 'empirical regularities'),
    (r'\bempirical laws?\b', 'empirical regularity / finding'),
    (r'\bfirst empirical law\b|\bsecond empirical law\b', 'first / second finding'),
    (r'\bprove[sd]?\b|\bproofs?\b', 'shows / demonstrates; "proofs" -> "derivations"'),
    (r'\bTheorem\b', 'Observation (or Proposition only for the renewal-reward / lattice results)'),
    (r'Microscopic X-ray', 'per-step trajectory trace'),
    (r'Gateway keepalive armor', 'Scorecard timeout mitigation'),
    (r'Orthogonal compressed sensing', 'Action-signature routing and rational score-lattice inversion'),
    (r'Orthogonal action-channel routing', 'Action-signature routing'),
    (r'\bcausal ablations?\b', 'single-run ablations'),
    (r'\bguarantees?\b', 'yields (state the measured scope)'),
    (r'\bWriteup\b', 'paper'),
    (r'wordcount\.py|word count', 'delete the line (Kaggle submission tooling, not reproduction)'),
    (r'\bGeneralize\b', 'only after the no-astra arm supports it'),
    (r'\bestablish(es)?\b', 'shows (keep only where an interval excludes 0)'),
    (r'\bfirst\b(?= empirical| to)', 'remove priority claim unless sourced'),
]


def wording_scan():
    hits = []
    for name in ('frontier-lock.md', 'appendix.md'):
        for i, line in enumerate((PAPER / 'paper' / name).read_text().splitlines(), 1):
            for pat, rep in TERMS:
                for m in re.finditer(pat, line, flags=re.I if pat.islower() else 0):
                    a = max(0, m.start() - 50); b = min(len(line), m.end() + 50)
                    hits.append({'file': f'paper/{name}', 'line': i, 'term': m.group(0), 'replace_with': rep,
                                 'context': line[a:b].replace('\n', ' ')})
    return hits


def verify_archive(prior):
    """Recompute totals and receipt statistics; no engine rerun is implied."""
    runs = prior['ladder_runs']
    for key, arm in runs.items():
        rows = arm['games']
        assert rows and len({r['game'] for r in rows}) == len(rows), key
        assert arm['levels'] == sum(r['levels_completed'] for r in rows), key
        assert arm['wins'] == sum(bool(r['win']) for r in rows), key
        assert arm['errors'] == sum('error' in r for r in rows) == 0, key
        assert all(0 <= r['levels_completed'] <= r['total_levels'] for r in rows), key
    full = runs['public25/full']
    assert (len(full['games']), full['levels'], full['wins']) == (25, 183, 25)
    assert next(r for r in full['games'] if r['game'].startswith('lf52'))['levels_completed'] == 10
    observed = hidden_stats(prior['hidden110']['backbone_12'])
    assert observed == prior['hidden110'], 'Archived hidden statistics differ from recomputation'
    return {k: {n: v[n] for n in ('levels', 'wins', 'errors')} for k, v in runs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suites', default='public25')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--skip-ladder', action='store_true')
    ap.add_argument('--reuse-archive', action='store_true', help='Verify archived rows without rerunning engines')
    args = ap.parse_args()
    prior = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.reuse_archive or not (ARC / 'scripts/validate_schema.py').is_file():
        summary = verify_archive(prior)
        print(json.dumps({'reproduction_mode': 'archived_rows_and_receipt_recalculation',
                          'archive': str(OUT.relative_to(PAPER)), 'sha256': sha(OUT),
                          'ladder_runs': summary, 'hidden110': prior['hidden110']}))
        return
    t0 = time.perf_counter()
    if not args.skip_ladder:
        run_ladder([s for s in args.suites.split(',') if s], args.arms.split(','), prior)
        prior['exact_inventory'] = exact_inventory()
    prior['hidden110'] = hidden_stats()
    prior['wording_hits'] = wording_scan()
    prior['meta'] = {'script_sha256': sha(__file__), 'planner_sha256': sha(ARC / 'scripts/giants_world_model_planner.py'),
                     'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     'runtime_s_this_call': round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(prior, indent=1, ensure_ascii=False) + '\n')
    runs = prior.get('ladder_runs', {})
    print(json.dumps({k: {'levels': v['levels'], 'wins': v['wins'], 'errors': v['errors']} for k, v in runs.items()}))


if __name__ == '__main__':
    main()
