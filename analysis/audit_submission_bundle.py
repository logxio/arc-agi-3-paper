"""Submission preflight for the ARC Prize 2026 Paper Track bundle.

Four invariants, every reading recomputed from the repo at run time (no earlier receipt is trusted on its own):
  1  forbidden terms and internal traces: zero hits in the four public sources, and no held-out iteration labels
     (OnlineInducer v5..v8, tuned held-out game names) in frontier-lock.md and appendix.md
  2  core statistics: each reading recomputed from its JSON / trace source, present in its home documents,
     and no conflicting variant anywhere in the three documents
  3  figures: live figcheck rerun into this directory plus a pixel hue audit of the shipped PNGs
  4  receipt: preflight_audit_receipt.json + PREFLIGHT_REPORT.md next to this script

Read-only against the repo; writes only inside this directory. Exit code 1 when any invariant fails.
"""
from __future__ import annotations

import hashlib
import argparse
import ast
import base64
import fnmatch
import json
import lzma
import pickle
import math
import re
import statistics
import subprocess
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'paper/frontier-lock.md').is_file() and (p / 'analysis').is_dir())
HERE = (ROOT / '.ai/blind/patch_autonomous_submission_preflight'
        if (ROOT / '.ai').is_dir() else ROOT / 'analysis/out/submission_audit')
ANALYSIS = ROOT / 'analysis'
OUT = ANALYSIS / 'out'
PAPER = ROOT / 'paper'
PY = sys.executable
sys.dont_write_bytecode = True      # importing analysis/ must not touch analysis/__pycache__
sys.path.insert(0, str(ANALYSIS))

DOCS = {'FL': PAPER / 'frontier-lock.md', 'AP': PAPER / 'appendix.md', 'RD': ROOT / 'README.md',
        'AR': ANALYSIS / 'README.md'}
EXTENDED = [PAPER / 'ARC_Prize_2026_Breaking_Frontier_Lock.tex', *sorted((ROOT / 'notebooks').glob('*.ipynb'))]
FORBIDDEN = ['astra_final_public_solvers', 'astra_tufa_algebra_solvers', 'astra_turducken', 'incl. 10 official',
             'Theorem 1', '1,794/1,795', '434/434', 'upload order', 'peer anchor', 'two causal laws',
             'two foundational laws', '/Users/suapril', 'claude-opus', 'gpt-6.1-sol']
MAIN_FIGURES = ['hazard', 'regime', 'hazard_by_dataset', 'architecture', 'cover']
SOURCES = [*DOCS.values(), OUT / 'component_ablation_ladder.json', OUT / 'ablation_ladder.json', OUT / 'a9_d4_and_reachability_audit.json',
           OUT / 'xray_figcheck.json', ROOT / '.ai/blind/patch_autonomous_inducer_v8/inducer_v8_results.json',
           ROOT / '.ai/blind/patch_autonomous_inducer_v8b/inducer_v8b_results.json',
           ROOT / '.ai/blind/patch_autonomous_d4_subboard/d4_subboard_and_factors_receipt.json']


_PAYLOAD = None


def embedded_inputs():
    """Read the hash-checked public notebook payload, without executing notebook code."""
    global _PAYLOAD
    if _PAYLOAD is None:
        nb = json.loads((ROOT / 'notebooks/frontier_lock_reproduction.ipynb').read_text())
        for cell in nb['cells']:
            source = cell['source']
            source = ''.join(source) if isinstance(source, list) else source
            if 'payload_bytes = ' not in source:
                continue
            tree = ast.parse(source)
            packed = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode'
                          and isinstance(n.args[0], ast.Constant))
            raw = (lzma if 'lzma.decompress(' in source else zlib).decompress(base64.b64decode(packed))
            expected = re.search(r"sha256\(payload_bytes\)\.hexdigest\(\) == ['\"]([0-9a-f]{64})", source)[1]
            assert hashlib.sha256(raw).hexdigest() == expected, 'Notebook payload checksum mismatch'
            _PAYLOAD = pickle.loads(raw) if 'pickle.loads(payload_bytes)' in source else json.loads(raw)
            assert _PAYLOAD['source_sha256'] == {
                n: hashlib.sha256((PAPER / n).read_bytes()).hexdigest()
                for n in ('frontier-lock.md', 'appendix.md')}, 'Notebook manuscript snapshot mismatch'
            assert _PAYLOAD['locked_markdown_sha256'] == {
                n: sha(ROOT / n) for n in ('paper/frontier-lock.md', 'paper/appendix.md', 'README.md', 'analysis/README.md')
            }, 'Notebook locked Markdown snapshot mismatch'
            assert _PAYLOAD['figure_png_sha256'] == {
                n: sha(PAPER / 'figures' / (n + '.png'))
                for n in ('cover', 'architecture', 'hazard', 'regime', 'hazard_by_dataset')
            }, 'Notebook figure snapshot mismatch'
            break
        assert _PAYLOAD is not None, 'Notebook contains no reproducibility payload'
    return _PAYLOAD


def public_key(path):
    return str(path.relative_to(ROOT)).replace('.ai/blind/', 'evidence/archive/').replace(
        '.ai/dispatch/', 'evidence/run-archive/')


def archived_record(path):
    item = embedded_inputs()['audit_inputs'][public_key(path)]
    raw = json.dumps(item['data'], sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(raw).hexdigest() == item['data_sha256'], f'Input checksum mismatch: {public_key(path)}'
    return item


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else archived_record(p)['data_sha256']


def J(p: Path):
    return json.loads(p.read_text()) if p.is_file() else archived_record(p)['data']


def input_glob(folder, pattern):
    if folder.is_dir():
        return folder.glob(pattern)
    prefix = public_key(folder) + '/'
    return [folder / key[len(prefix):] for key in embedded_inputs()['audit_inputs']
            if key.startswith(prefix) and '/' not in key[len(prefix):]
            and fnmatch.fnmatchcase(key[len(prefix):], pattern)]


def prepare_archived_figures():
    if not (ROOT / '.ai/blind/patch_autonomous_inducer_v8').is_dir():
        import generate_xray_figures as gx
        data = embedded_inputs()
        gx.heldout_sources = lambda: data['heldout_ladder']
        gx.l2_sources = lambda: data['l2_rules']


def norm(s: str) -> str:
    """Compare readings, not typography: drop markdown/LaTeX decoration and every space."""
    s = s.replace('−', '-').replace('\\%', '%').replace('\\,', '').replace('\\;', '')
    return re.sub(r'[\s`$*]', '', s)


TEXT = {k: p.read_text() for k, p in DOCS.items()}
NORM = {k: norm(v) for k, v in TEXT.items()}


# ---------------------------------------------------------------- invariant 1

def invariant_1():
    def scan(path: Path):
        hits = []
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for term in FORBIDDEN:
                if term in line:
                    i = line.index(term)
                    hits.append({'file': str(path.relative_to(ROOT)), 'line': n, 'term': term,
                                 'context': line[max(0, i - 50):i + len(term) + 50]})
        return hits
    core = [h for p in DOCS.values() for h in scan(p)]
    ext = [h for p in EXTENDED if p.exists() for h in scan(p)]
    # held-out iteration labels: gating in the two manuscript files, reported for the LaTeX and the notebook
    for p in (DOCS['FL'], DOCS['AP'], *EXTENDED):
        if not p.exists():
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for m in re.finditer(r'OnlineInducer\s*v[5-8]\b' if p.suffix == '.ipynb' else HELDOUT_TUNED, line):
                (core if p in (DOCS['FL'], DOCS['AP']) else ext).append(
                    {'file': str(p.relative_to(ROOT)), 'line': n, 'term': m.group(0),
                     'context': line[max(0, m.start() - 50):m.end() + 50]})
    # case-insensitive echo of the same list, reported only
    ci = []
    for k, p in DOCS.items():
        low = TEXT[k].lower()
        for term in FORBIDDEN:
            c = low.count(term.lower())
            exact = TEXT[k].count(term)
            if c > exact:
                ci.append({'file': str(p.relative_to(ROOT)), 'term': term, 'case_insensitive_only_hits': c - exact})
    return {'passed': not core, 'terms': FORBIDDEN, 'files': [str(p.relative_to(ROOT)) for p in DOCS.values()],
            'hits': core, 'hit_count': len(core),
            'extended_scan': {'files': [str(p.relative_to(ROOT)) for p in EXTENDED if p.exists()],
                              'hits': ext, 'hit_count': len(ext), 'gating': False},
            'case_insensitive_variants': ci}


# ---------------------------------------------------------------- invariant 2 helpers

class Claim:
    def __init__(self, cid, title):
        self.cid, self.title = cid, title
        self.source, self.text, self.conflicts, self.notes = [], [], [], []

    def src(self, what, observed, expected, ok=None):
        if ok is None:
            ok = observed == expected
        self.source.append({'what': what, 'observed': observed, 'expected': expected, 'ok': bool(ok)})

    def need(self, label, patterns, home):
        """patterns: regex list on normalized text, any one matching counts as present."""
        pats = [patterns] if isinstance(patterns, str) else patterns
        pres = {k: any(re.search(p, NORM[k]) for p in pats) for k in ('FL', 'AP', 'RD')}
        # the 1,500-word Writeup ships with the appendix in one PDF: a reading the Writeup cut still counts there
        self.text.append({'reading': label, 'home_docs': home, 'present': pres,
                          'ok': all(pres[k] or (k == 'FL' and pres['AP']) for k in home)})

    def conflict(self, what, doc, found, allowed):
        self.conflicts.append({'what': what, 'doc': doc, 'found': found, 'allowed': allowed})

    @property
    def ok(self):
        return all(s['ok'] for s in self.source) and all(t['ok'] for t in self.text) and not self.conflicts

    def dump(self):
        return {'id': self.cid, 'title': self.title, 'passed': self.ok, 'source_checks': self.source,
                'text_checks': self.text, 'conflicts': self.conflicts, 'notes': self.notes}


def r2(x):
    return round(x + 1e-12, 2)


def numerators(doc, denom_regex, attributed_pattern=None):
    """Every 'N/denom' written in a document, as integers (commas dropped)."""
    out = []
    spaced = re.sub(r'[`$*]', '', TEXT[doc].replace('−', '-'))   # keep spaces: "P99, 0/219" is 0/219, not 990/219
    if attributed_pattern:
        spaced = re.sub(attributed_pattern, '', spaced)
    for m in re.finditer(r'(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)/' + denom_regex + r'(?![\d])', spaced):
        out.append(int(m.group(1).replace(',', '')))
    return out


def ratio_conflicts(c: Claim, denom_regex, allowed, label, attributed_pattern=None):
    for d in ('FL', 'AP', 'RD'):
        for n in numerators(d, denom_regex, attributed_pattern):
            if n not in allowed:
                c.conflict(label, d, n, sorted(allowed))


# ---------------------------------------------------------------- invariant 2

def wordcount_claim():
    c = Claim('W', 'Writeup word count and replay alignment anchors')
    cmd = [PY, str(ANALYSIS / 'wordcount.py'), str(DOCS['FL'])]
    words = int(subprocess.check_output(cmd, text=True).strip())
    c.src('analysis/wordcount.py on paper/frontier-lock.md <= 1500', words, '<=1500', words <= 1500)
    c.src('measured words within 1,500-word Writeup limit', words, '<=1500', words <= 1500)
    c.need('README states Writeup word limit', [r'1,500words', r'1,4\d\dwords'], ['RD'])
    # replay_alignment (analysis/verify_pdf.py), recomputed on the markdown sources and the LaTeX
    x = J(OUT / 'x0_specimens.json')
    pairs = {(p['game'], p['i'], p['j']) for p in x['human_click_pairs'] + x['human_prefix_pairs']}
    reach, runs = {}, 0
    from common import DATASETS, load
    for name in DATASETS:
        for r in load(name):
            runs += 1
            g = r['game'].split('-')[0]
            reach[g] = max(reach.get(g, 0), min(r['k'] + 1, r['L']))
    baseline = sum(reach.get(g, 0) >= j for g, i, j in pairs)
    planner = sum(x['planner'][g]['levels'] >= j for g, i, j in pairs)
    fp = x['fingerprints']
    fixed = {'games': (fp['games'], 25), 'constant_grid_games': (fp['constant_grid_games'], 18),
             'click_levels': (fp['click_levels'], 137), 'repeat_click_levels': (fp['repeat_click_levels'], 111),
             'cross_level_pairs': (fp['cross_level_pairs'], 597), 'human_files': (x['human_files'], 342),
             'baseline_runs': (runs, 825), 'reuse_pairs': (len(pairs), 378),
             'shared_click_pairs': (len(x['human_click_pairs']), 284),
             'shared_prefix_pairs': (len(x['human_prefix_pairs']), 120),
             'planner_target_level_reached': (planner, 378)}
    for k, (o, e) in fixed.items():
        c.src(f'replay fingerprint {k}', o, e)
    anchors = ['72.0% (18/25)', '81.0% (111/137)', '284/597', '120/597', f'{planner}/{len(pairs)}',
               f'{baseline}/{len(pairs)}']
    tex = EXTENDED[0].read_text().replace('\\%', '%') if EXTENDED[0].exists() else ''
    missing = {}
    for name, text in (('frontier-lock.md', TEXT['FL']), ('appendix.md', TEXT['AP']), ('latex', tex)):
        missing[name] = [a for a in anchors if re.sub(r'\s+', '', a) not in re.sub(r'\s+', '', text)]
    c.src('6 replay_alignment anchors present in frontier-lock.md, appendix.md and the LaTeX',
          {k: len(anchors) - len(v) for k, v in missing.items()}, {k: 6 for k in missing},
          not any(missing.values()))
    c.notes.append({'replay_anchors': anchors, 'missing': missing, 'baseline_reached': baseline})
    return c, words


def hidden_claims():
    from scipy import stats
    L = J(OUT / 'component_ablation_ladder.json')['hidden110']
    bb = L['backbone_12']
    scores = bb['scores']
    n, mean, sd = len(scores), statistics.mean(scores), statistics.stdev(scores)
    out = []

    c = Claim('H1', 'Hidden-110 36K W4A16 10-stream MTP3 backbone, 12 runs')
    c.src('n', n, 12)
    c.src('mean (recomputed from 12 scores)', r2(mean), 24.62)
    c.src('sd (recomputed)', r2(sd), 2.22)
    c.src('json mean / sd agree with recompute', (bb['mean'], bb['sd']), (round(mean, 4), round(sd, 4)),
          abs(bb['mean'] - mean) < 1e-3 and abs(bb['sd'] - sd) < 1e-3)
    c.src('range', (min(scores), max(scores)), (21.22, 28.38))
    peak_ref = bb['refs'][scores.index(max(scores))]
    c.src('peak run', peak_ref, 57031236)
    c.need('24.62±2.22', r'24\.62±2\.22', ['FL', 'AP', 'RD'])
    c.need('range 21.22 to 28.38', [r'21\.22-28\.38', r'21\.22to28\.38'], ['FL', 'AP', 'RD'])
    c.need('peak #57031236', r'57031236', ['AP'])
    # the 14-run zero-probe (k=0) pool: 12 native runs + #57329763 (26.55) + #57827241 (25.70)
    pool14 = scores + [26.55, 25.70]
    m14, s14 = statistics.mean(pool14), statistics.stdev(pool14)
    h14 = stats.t.ppf(0.975, len(pool14) - 1) * s14 / math.sqrt(len(pool14))
    c.src('k=0 pool mean±sd (recomputed from 14 scores)', (r2(m14), r2(s14)), (24.83, 2.12))
    c.src('k=0 pool 95% t-CI (recomputed)', (r2(m14 - h14), r2(m14 + h14)), (23.61, 26.06))
    c.need('24.83±2.12 with [23.61, 26.06]', r'24\.83±2\.12(LB)?,(95%CI)?\[23\.61,26\.06\]',
           ['FL', 'AP', 'RD'])
    allowed = ['24.62±2.22', '2.81±0.66', '24.83±2.12']
    for d in ('FL', 'AP', 'RD'):
        # keep spaces here: "`24.62±2.22` 12-run" must not read as 24.62±2.2212
        for m in re.finditer(r'(\d+\.\d+)±(\d+\.\d+)', re.sub(r'[`$*]', '', TEXT[d].replace('−', '-'))):
            if m.group(0) not in allowed:
                c.conflict('mean±sd variant', d, m.group(0), allowed)
    out.append(c)

    c = Claim('H2', 'Backbone vs Duck V2 (Welch)')
    v = L['backbone_vs_duck_v2']
    dm, dsd, dn = 2.81, 0.66, 6
    se = math.sqrt(sd ** 2 / n + dsd ** 2 / dn)
    t = (mean - dm) / se
    df = se ** 4 / ((sd ** 2 / n) ** 2 / (n - 1) + (dsd ** 2 / dn) ** 2 / (dn - 1))
    p = 2 * stats.t.sf(abs(t), df)
    c.src('Welch t (recomputed from summary stats)', r2(t), 31.32, abs(t - 31.32) <= 0.03)
    c.src('Welch p two-sided (recomputed)', f'{p:.2e}', '1.52e-14', abs(p - 1.52e-14) <= 0.02e-14)
    c.src('json Welch t / p', (v['welch_t'], f"{v['welch_p_two_sided']:.2e}"), (31.32, '1.52e-14'))
    fresh = J(OUT / 'ablation_ladder.json')['hidden110']['backbone_vs_duck_v2']
    c.src('fresh archive dynamic Welch t / p', (fresh['welch_t'], fresh['welch_p_two_sided']),
          (round(t, 4), float(p)), fresh['welch_t'] == round(t, 4) and
          math.isclose(fresh['welch_p_two_sided'], float(p), rel_tol=1e-12))
    c.src('mean difference', r2(mean - dm), 21.81)
    c.src('Mann-Whitney U, exact p (complete separation 2/C(18,6))',
          (v['mann_whitney_U'], f'{2 / math.comb(18, 6):.2e}'), (72, '1.08e-04'))
    c.need('2.81±0.66', r'2\.81±0\.66', ['FL', 'AP', 'RD'])
    c.need('Welch t=31.32', r't=-?31\.32', ['AP', 'RD'])
    c.need('p=1.52e-14', r'p=1\.52\\times10\^\{-14\}', ['FL', 'AP', 'RD'])
    out.append(c)

    def single(cid, title, ref, score, key, t_exp, p_exp, delta_exp):
        c = Claim(cid, title)
        s = L['single_runs_vs_backbone'][key]
        tt = (score - mean) / (sd * math.sqrt(1 + 1 / n))
        pp = 2 * stats.t.sf(abs(tt), n - 1)
        c.src('json score', s['score'], score)
        c.src('prediction t (recomputed, df=11)', r2(tt), t_exp)
        c.src('prediction p two-sided (recomputed)', f'{pp:.2e}', p_exp)
        c.src('delta vs mean', r2(score - mean), delta_exp)
        c.need(f'#{ref}', str(ref), ['FL', 'AP', 'RD'])
        c.need(f'{score:.2f}', re.escape(f'{score:.2f}'), ['FL', 'AP', 'RD'])
        c.need(f't={t_exp:.2f}', re.escape(f't={t_exp:.2f}'), ['FL', 'AP', 'RD'])
        mant, ex = p_exp.split('e')
        c.need(f'p={p_exp}', re.escape(f'p={mant}\\times10^{{{int(ex)}}}'), ['FL', 'AP', 'RD'])
        return c
    out.append(single('H3', 'Hidden-110 14K context cap', 57774685, 9.39, '14k_cap_57774685', -6.58, '3.96e-05', -15.23))
    out.append(single('H4', 'Hidden-110 convex slot kill', 57786717, 7.72, '14k_convex_kill_57786717', -7.30, '1.53e-05', -16.90))

    c = Claim('H5', 'Two uncertified runs rank last among 14 (exact one-sided Mann-Whitney)')
    unc = {57854657: 20.76, 57894564: 21.08}
    pool = sorted(scores + list(unc.values()))
    ranks = sorted(pool.index(v) + 1 for v in unc.values())
    U = sum(1 for a in unc.values() for b in scores if a > b) + 0.5 * sum(1 for a in unc.values() for b in scores if a == b)
    c.src('ranks among 14 (recomputed)', ranks, [1, 2])
    c.src('U', U, 0)
    c.src('exact p = 1/C(14,2)', round(1 / math.comb(14, 2), 4), 0.011)
    rt = L['rank_tests']['uncertified_two_runs_vs_backbone_12']
    c.src('json ranks / U / p', (rt['ranks_among_14'], rt['mann_whitney_U'], round(rt['p_one_sided_exact'], 4)),
          ([1, 2], 0, 0.011))
    c.src('json scores', (L['single_runs_vs_backbone']['open_loop_600_57854657']['score'],
                          L['single_runs_vs_backbone']['reactive_224_57894564']['score']), (20.76, 21.08))
    for ref, s in unc.items():
        c.need(f'#{ref}', str(ref), ['FL', 'AP', 'RD'])
        c.need(f'{s:.2f}', re.escape(f'{s:.2f}'), ['FL', 'AP', 'RD'])
    c.need('U=0', r'U=0', ['FL', 'AP', 'RD'])
    c.need('p=0.0110', r'0\.0110', ['FL', 'AP', 'RD'])
    # against the 14-run k=0 pool (12 native + #57329763 + #57827241) the two uncertified runs are ranks 1, 2 of 16
    pool16 = sorted(scores + [26.55, 25.70] + list(unc.values()))
    c.src('ranks among 16 (recomputed)', sorted(pool16.index(v) + 1 for v in unc.values()), [1, 2])
    c.src('exact p = 1/C(16,2)', round(1 / math.comb(16, 2), 4), 0.0083)
    c.need('p=0.0083', r'0\.0083', ['FL', 'AP', 'RD'])
    out.append(c)

    c = Claim('H6', 'Certificate-gated run #57827241')
    cert = 25.70
    rank13 = sorted(scores + [cert]).index(cert) + 1
    p13 = 2 * min(rank13, 14 - rank13) / 13
    c.src('rank among 13 (recomputed)', rank13, 8)
    c.src('two-sided exact rank p (recomputed)', round(p13, 3), 0.923)
    c.src('+4.94 vs 20.76', r2(cert - 20.76), 4.94)
    c.src('+4.62 vs 21.08', r2(cert - 21.08), 4.62)
    c.src('json pairwise deltas', [x['delta'] for x in L['pairwise'][:2]], [4.94, 4.62])
    from scipy import stats
    prediction = J(OUT / 'ablation_ladder.json')['hidden110']['single_runs_vs_backbone']['cert_gated_57827241']
    gate_t = (cert - mean) / (sd * math.sqrt(1 + 1 / n))
    c.src('C09 four-decimal prediction t / p',
          (prediction['prediction_t'], round(prediction['p_two_sided_prediction_t'], 4)),
          (round(gate_t, 4), round(float(2 * stats.t.sf(abs(gate_t), n - 1)), 4)))
    c.need('#57827241', '57827241', ['FL', 'AP', 'RD'])
    c.need('25.70', r'25\.70', ['FL', 'AP', 'RD'])
    c.need('+4.94', r'\+4\.94', ['FL', 'AP', 'RD'])
    c.need('+4.62', r'\+4\.62', ['FL', 'AP', 'RD'])
    c.need('24.83±2.12', r'24\.83±2\.12', ['FL', 'AP', 'RD'])
    out.append(c)

    # every "#ref at S" written in any document must carry that run's official score
    c = Claim('H7', 'Every "#ref at score" in the documents matches the receipt')
    known = {57774685: 9.39, 57786717: 7.72, 57854657: 20.76, 57894564: 21.08, 57827241: 25.70, 57031236: 28.38}
    for d in ('FL', 'AP', 'RD'):
        for m in re.finditer(r'#(\d{8})(?:\)|,)?(?:\d+-step[a-z-]*|open-loop|reactive|in)*at(\d+\.\d+)', NORM[d]):
            ref, s = int(m.group(1)), float(m.group(2))
            if ref in known and abs(known[ref] - s) > 1e-9:
                c.conflict(f'#{ref} score', d, s, known[ref])
    c.src('receipt map', known, known)
    out.append(c)
    return out


def proposition_claims():
    from common import DATASETS, load, score
    c = Claim('P1', 'Propositions 1 to 3: denominator, quadratic dilution, O(k^-3) saturation')
    c.need('a=k+1+A', r'a=k\+1\+A', ['FL', 'AP', 'RD'])
    c.need('0 violations on 183 Public-25 levels', [r'183/183Public-25levels,0violations',
                                                    r'183Public-25levelswith0violations'], ['FL', 'AP', 'RD'])
    c.need('(A/(A+k+1))^2', r'\(A/\(A\+k\+1\)\)\^2', ['FL', 'AP', 'RD'])
    c.need('O(k^{-3})', r'O\(k\^\{-3\}\)', ['FL', 'AP', 'RD'])
    c.need('Proposition 1 to 3 named', [r'Proposition1to3', r'Proposition3'], ['AP', 'RD'])
    # The 183/183 zero-violation count has no machine-readable source in analysis/out.
    c.notes.append('a=k+1+A on 183/183 levels: stated in all three documents; no file in analysis/out records the '
                   'per-level check, so this reading is text-only (not recomputed here).')
    out = [c]

    ks = (8, 32, 64, 224, 600)
    table, max_gap = {}, {}
    runs_total = 0
    for ds in DATASETS:
        runs = load(ds)
        runs_total += len(runs)
        base = sum(score(r['L'], r['k'], r['a'], r['base']) for r in runs)
        row = {'runs': len(runs), 'base_mean': round(base / len(runs), 4)}
        for k in ks:
            exact = sum(score(r['L'], r['k'], [x + k + 1 if (i < r['k'] and x > 0) else x
                                               for i, x in enumerate(r['a'])], r['base']) for r in runs)
            # closed form of Proposition 3: (1-p)(1-eps_k), p=0, eps_k = sum s_l(A+k+1) / sum s_l(A)
            num = den = num100 = den100 = 0.0
            for r in runs:
                for i in range(min(r['k'], r['L'])):
                    A = r['a'][i]
                    if A <= 0:
                        continue
                    s = min(115.0, 100.0 * (r['base'][i] / A) ** 2)
                    s_k = min(115.0, 100.0 * (r['base'][i] / (A + k + 1)) ** 2)
                    num += s_k
                    den += s
                    weight = 2 * (i + 1) / (r['L'] * (r['L'] + 1))
                    num100 += weight * min(100.0, s_k)
                    den100 += weight * min(100.0, s)
            exact_loss = 1 - exact / base
            closed_loss = 1 - num / den
            row[f'k{k}'] = {'score': round(exact / len(runs), 4), 'exact_loss': round(exact_loss, 4),
                            'closed_form_loss': round(closed_loss, 4), 'gap': round(abs(exact_loss - closed_loss), 4),
                            'weighted_100_cap_gap': abs(exact_loss - (1 - num100 / den100))}
        table[ds] = row
    for k in ks:
        max_gap[k] = max(table[d][f'k{k}']['gap'] for d in DATASETS)
    l8 = [table[d]['k8']['exact_loss'] for d in DATASETS]
    l224 = [table[d]['k224']['exact_loss'] for d in DATASETS]

    c = Claim('P2', 'Pure-dilution counterfactual across 825 runs')
    c.src('runs', runs_total, 825)
    c.src('k=8 loss range, % (recomputed)', (round(100 * min(l8)), round(100 * max(l8))), (14, 26))
    c.src('k=224 loss range, % (recomputed)', (round(100 * min(l224)), round(100 * max(l224))), (97, 99))
    c.src('max |exact - closed form| at k=224 across 8 datasets', max_gap[224], '<=0.005', max_gap[224] <= 0.005)
    c.src('max |exact - closed form| over k in {8..600}', max(max_gap.values()), '<=0.071',
          max(max_gap.values()) <= 0.071)
    cap_gaps = [table[d][f'k{k}']['weighted_100_cap_gap'] for d in DATASETS for k in ks]
    c.src('weighted 100-capped closed form max error', max(cap_gaps), '<=0.021', max(cap_gaps) <= 0.021)
    c.src('weighted 100-capped closed form mean error rounded to 4 decimals', round(statistics.mean(cap_gaps), 4), 0.0044)
    th = J(ROOT / '.ai/blind/patch_autonomous_universality/theory_results.json')['pure_dilution']
    agree = all(abs(-th[d][j]['rel'] - table[d][f'k{k}']['exact_loss']) < 1e-3
                for d in DATASETS for j, k in enumerate((8, 32, 224, 600)))
    c.src('independent recompute agrees with theory_results.json pure_dilution', agree, True)
    c.need('k=8 range or exact Qwen worked example', [r'14%?-26%', r'14%?to26%', r'14\.6%lossatk=8'], ['RD'])
    c.need('k=224 range or exact Qwen worked example',
           [r'97%?-99%', r'97%?to99%', r'96\.6%to99\.4%lossatk=224andk=600'], ['RD'])
    # the Writeup quotes the Appendix D worked example (Qwen3.8-Flash) instead of the 8-dataset range
    q = table['api_qwen38flash']
    worked = tuple(round(100 * q[f'k{k}']['exact_loss'], 1) for k in (8, 224, 600))
    c.src('Writeup losses at k=8/224/600 = Qwen3.8-Flash worked example (recomputed)', worked, (14.6, 96.6, 99.4))
    c.need('14.6% at k=8, 96.6% at k=224, 99.4% at k=600',
           [r'14\.6%lossatk=8,96\.6%atk=224,99\.4%atk=600', r'14\.6%lossatk=8and96\.6%to99\.4%lossatk=224andk=600',
            r'7\.74\(-14\.6%\)atk=8,1\.99\(-78\.0%\)atk=64,0\.30\(-96\.6%\)atk=224,and0\.05\(-99\.4%\)atk=600'], ['FL', 'AP'])
    c.need('closed-form accuracy (legacy or explicit capped form)',
           [r'within0\.070', r'atmost0\.004', r'within0\.021max/0\.0044meanerror',
            r'within0\.021maximumerror\(0\.0044meanerror\)'], ['FL', 'AP', 'RD'])
    c.need('825 runs', r'825', ['FL', 'AP', 'RD'])
    c.notes.append({'per_dataset': table, 'max_gap_by_k': max_gap})
    out.append(c)

    # the appendix prints a per-dataset worked example for Qwen3.8-Flash: check every figure in it
    c = Claim('P3', 'Appendix D worked example (Qwen3.8-Flash) equals the recomputed counterfactual')
    q = table['api_qwen38flash']
    m = re.search(r'Qwen3\.8-Flash.{0,80}?base score `?([\d.]+)`?.*?reduces score to `?([\d.]+)`? \(`?-([\d.]+)%`?\) at '
                  r'\$k = 8\$, `?([\d.]+)`? \(`?-([\d.]+)%`?\) at \$k = 64\$, `?([\d.]+)`? \(`?-([\d.]+)%`?\) at \$k = 224\$, '
                  r'and `?([\d.]+)`? \(`?-([\d.]+)%`?\) at \$k = 600\$', TEXT['AP'], re.S)
    if not m:
        c.src('worked example sentence found in appendix', False, True)
    else:
        g = [float(x) for x in m.groups()]
        c.src('base score', g[0], round(q['base_mean'], 2), abs(g[0] - q['base_mean']) < 0.006)
        for (sv, lv), k in zip(((g[1], g[2]), (g[3], g[4]), (g[5], g[6]), (g[7], g[8])), (8, 64, 224, 600)):
            e = q[f'k{k}']
            c.src(f'k={k} score', sv, round(e['score'], 2), abs(sv - e['score']) < 0.006)
            c.src(f'k={k} loss %', lv, round(100 * e['exact_loss'], 1), abs(lv - 100 * e['exact_loss']) < 0.06)
    m2 = re.search(r'between \$k = 224\$ \(`?([\d.]+)%`? loss\) and \$k = 600\$ \(`?([\d.]+)%`? loss\)', TEXT['AP'])
    if m2:
        c.src('k=224 vs k=600 loss quoted in the same paragraph',
              (float(m2.group(1)), float(m2.group(2))),
              (round(100 * q['k224']['exact_loss'], 1), round(100 * q['k600']['exact_loss'], 1)),
              abs(float(m2.group(1)) - 100 * q['k224']['exact_loss']) < 0.06
              and abs(float(m2.group(2)) - 100 * q['k600']['exact_loss']) < 0.06)
    m3 = re.search(r'saturates at `?([\d.]+)%`? and `?([\d.]+)%`? of solved-level score', TEXT['AP'])
    if m3:
        c.src('Appendix J item 4 saturation percentages',
              (float(m3.group(1)), float(m3.group(2))),
              (round(100 * q['k224']['exact_loss'], 1), round(100 * q['k600']['exact_loss'], 1)),
              abs(float(m3.group(1)) - 100 * q['k224']['exact_loss']) < 0.06
              and abs(float(m3.group(2)) - 100 * q['k600']['exact_loss']) < 0.06)
    out.append(c)
    return out


def public25_claims():
    from common import load
    lad = J(OUT / 'component_ablation_ladder.json')['ladder_runs']
    a9 = J(OUT / 'a9_d4_and_reachability_audit.json')
    c = Claim('A1', 'Public-25 four-arm ablation and human action ratio')
    v2 = load('kaggle_v2')
    c.src('baseline clears (kaggle_v2 traces)', sum(r['k'] for r in v2), 42)
    c.src('baseline wins (kaggle_v2 traces, k == L)', sum(r['k'] >= r['L'] for r in v2), 0)
    rec_row = a9['ledger_9_suites']['from_receipts']['rows'][0]
    c.src('receipt ledger Public-25 baseline wins', rec_row['baseline_wins'], 0)
    for arm, (lv, w) in {'generic': (120, 13), 'no_astra': (134, 16), 'full': (180, 24)}.items():
        r = lad[f'public25/{arm}']
        c.src(f'{arm} levels / wins / errors', (r['levels'], r['wins'], r['errors']), (lv, w, 0))
    latest = J(OUT / 'ablation_ladder.json')['ladder_runs']['public25/full']
    c.src('separate fresh full arm: per-game levels / wins / errors',
          (sum(r['levels_completed'] for r in latest['games']), sum(bool(r['win']) for r in latest['games']), latest['errors']),
          (183, 25, 0))
    c.src('fresh lf52 clears', next(r['levels_completed'] for r in latest['games'] if r['game'].startswith('lf52')), 10)
    c.src('generic share', round(100 * 120 / 183, 1), 65.6)
    ar = a9['reachability_public25']['action_ratio_series35b']
    c.src('confirmation rerun 183/183, 25W, 0 censored', (ar['levels'], ar['wins'], ar['right_censored_levels']),
          (183, 25, 0))
    c.src('human ratio 7,401/17,135', (ar['planner_actions'], ar['human_baseline_actions'], round(ar['ratio'], 3)),
          (7401, 17135, 0.432))
    c.need('42/183 (0W)', [r'42/183\(0W', r'42/183\(0wins'], ['FL', 'AP', 'RD'])
    c.need('120/183', r'120/183', ['FL', 'AP', 'RD'])
    c.need('65.6%', r'65\.6%', ['FL', 'AP', 'RD'])
    c.need('13 wins', [r'13W', r'13/25wins'], ['FL', 'AP', 'RD'])
    c.need('134/183 (16W)', [r'134/183\(16W', r'134/183\(16/25wins'], ['FL', 'AP'])
    c.need('180/183 (24W)', [r'180/183\(24W', r'180/183\(24/25wins'], ['FL', 'AP'])
    c.need('183/183 Public-25 confirmation', [r'183/183,25W', r'183/183,25/25wins', r'183/183\(25/25WIN', r'183/183\(25/25W[,)]',
           r'\|Public-25\|25\(183\)\|[^\n]+\|183/183\(G=99\.58/100\)\|'],
           ['FL', 'AP', 'RD'])
    c.need('7,401/17,135', r'7,401/17,135', ['FL', 'AP', 'RD'])
    c.need('0.432x', r'0\.432x', ['FL', 'AP', 'RD'])
    # Twin's explicitly attributed comparison is not one of our ablation arms.
    twin = r'Twin(?: \[6\]| \(arXiv:2608\.14490\))[^\n]*?179/183L?'
    ratio_conflicts(c, '183', {42, 120, 134, 150, 163, 166, 169, 179, 180, 183}, 'N/183', twin)
    # 150, 163 are held-out L2 D4 rule hits (T6); 166, 169 are 115%-cap level counts; 7070 is loop-stripped G=100.00.
    ratio_conflicts(c, '17,135', {7070, 7401, 24604}, 'N/17,135')
    c.notes.append({'series36_with_confirm_actions': a9['reachability_public25']['action_ratio_series36_with_confirm'],
                    'note': 'series36+confirm gives 7,404/17,135 (0.4321); the documents use series35b 7,401.'})
    return [c]


def heldout_claims():
    a9 = J(OUT / 'a9_d4_and_reachability_audit.json')['ledger_9_suites']
    U = J(ROOT / '.ai/blind/patch_autonomous_universality/universality_results.json')
    P = U['pooled']
    c = Claim('T1', '98 held-out games, frozen pre-exposure snapshot')
    c.src('games', P['games'], 98)
    c.src('general levels / wins', (P['general'], P['wins']['general']), (80, 4))
    c.src('random levels / wins', (P['random'], P['wins']['random']), (45, 1))
    gv = P['general_vs_random']
    c.src('general - random [CI]', (gv['sum'], gv['lo'], gv['hi']), (35.0, 8.0, 65.0))
    cl = P['clean']['general_vs_random']
    c.src('clean 90-game holdout', (P['clean']['games'], cl['sum'], cl['lo'], cl['hi']), (90, 24.0, 1.0, 49.0))
    c.src('ungated full levels / wins', (P['full'], P['wins']['full']), (72, 2))
    c.src('gated levels / wins', (P['gated'], P['gated_wins']), (81, 4))
    mo = U['suites']['mother1102']
    c.src('Mother gated vs full', (mo['gated']['levels'], mo['full']['levels'], mo['gated_vs_full']['sum'],
                                   mo['gated_vs_full']['lo'], mo['gated_vs_full']['hi']), (41, 32, 9.0, 0.0, 23.0))
    t2 = a9['table2_groups']['Mother+Web+Turducken']
    c.src('solvable levels (ledger)', t2['solvable'], 655)
    c.need('general 80 levels 4W', [r'80levels\(4W', r'80levels\(4wins', r'80vs\.45'], ['FL', 'AP', 'RD'])
    c.need('random 45 (1W)', [r'45L?\(1W\)random', r'45\(1win\)', r'45/655rand\(1W', r'45/655rand,1W', r'4Wvs\.1W'], ['FL', 'AP'])
    c.need('+35 [+8,+65]', r'\+35.{0,40}\[\+8,\+65\]', ['FL', 'AP', 'RD'])
    c.need('+24 [+1,+49] on 90 clean', r'\+24L?\[\+1,\+49\]', ['FL', 'AP'])
    c.need('ungated full 72 (2W)', [r'72\(2W\)ungatedfull', r'72\(2wins\)forungatedfull'], ['FL', 'AP'])
    c.need('gated 81 (4W)', [r'81levels\(4W', r'81levels\(4wins'], ['FL', 'AP'])
    c.need('Mother 41 vs 32', r'41vs\.32onMother-\w+', ['FL', 'AP'])
    c.need('Mother +9 [0,+23]', r'41vs\.32onMother-\w+,\+9\[0,\+23\]', ['AP'])
    c.need('655 levels', r'655', ['FL', 'AP', 'RD'])
    out = [c]

    T = ROOT / '.ai/blind/patch_autonomous_astra_transfer'
    I = J(T / 'online_inducer_results.json')
    ip = I['suites']['pooled']
    c = Claim('T3', '98 held-out games, uncalibrated OnlineInducer (97L)')
    # recompute the pooled arms and the paired bootstrap from the per-game run files
    import random as _random
    arms = {}
    for tag, a in (('v1', 'random'), ('v1', 'general'), ('v1', 'gated'), ('v5gf', 'inducer')):
        arms[a] = {(g['suite'], g['game']): g for g in (J(p)['games'][0] for p in sorted(input_glob(T / 'runs' / tag, f'*__{a}.json')))}
    keys = sorted(arms['general'])

    def paired(x, y):
        d = [arms[x][k]['levels_completed'] - arms[y][k]['levels_completed'] for k in keys]
        rng = _random.Random(0)
        s = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) for _ in range(10000))
        return sum(d), s[250], s[9749]
    c.src('arm coverage (games per arm)', sorted({len(v) for v in arms.values()}), [98])
    c.src('inducer levels / wins (runs)', (sum(g['levels_completed'] for g in arms['inducer'].values()),
                                          sum(bool(g['win']) for g in arms['inducer'].values())), (97, 5))
    c.src('general / random / gated levels (runs)', tuple(sum(g['levels_completed'] for g in arms[a].values()) for a in ('general', 'random', 'gated')),
          (80, 45, 81))
    c.src('inducer - general [CI] (recomputed)', paired('inducer', 'general'), (17, 4, 33))
    c.src('inducer - gated [CI] (recomputed)', paired('inducer', 'gated'), (16, 3, 32))
    c.src('inducer - random [CI] (recomputed)', paired('inducer', 'random'), (52, 22, 83))
    c.src('stored pooled summary', (ip['inducer']['levels'], ip['inducer']['wins'], ip['inducer_vs_general']['sum'],
                                    ip['inducer_vs_general']['lo'], ip['inducer_vs_general']['hi'],
                                    ip['inducer_vs_general']['W'], ip['inducer_vs_general']['L']), (97, 5, 17, 4, 33, 10, 2))
    c.src('frozen-snapshot general / random equal T1', (ip['general']['levels'], ip['random']['levels']), (P['general'], P['random']))
    c.src('Public-25 unchanged (120/183, 13W)', (I['out_of_selection']['public25']['inducer']['levels'],
                                                I['out_of_selection']['public25']['inducer']['wins'],
                                                I['out_of_selection']['public25']['inducer_vs_general']['sum']), (120, 13, 0))
    c.need('induced 97 levels (5W)', [r'97levels\(5W', r'97levels\(5wins', r'97L\(5W\)'], ['FL', 'AP', 'RD'])
    c.need('+17 [+4,+33] over general', r'\+17(L|levels)?.{0,30}\[\+4,\+33\]', ['FL', 'AP', 'RD'])
    # comparisons against the v5/v6 arms left the manuscript with the iteration labels (one frozen OnlineInducer
    # ladder); their source checks above and below stay
    out.append(c)

    c = Claim('T4', '98 held-out games, +Op0 novelty-gated translation')
    V = J(ROOT / '.ai/blind/patch_autonomous_click_inertia/online_inducer_v6_results.json')
    vp = V['suites']['pooled']
    k3 = ROOT / '.ai/blind/patch_autonomous_click_inertia' / V['delivered']['run']
    arms['v6'] = {(g['suite'], g['game']): g for g in (J(p)['games'][0] for s in ('mother1102', 'tufaweb1102', 'turducken49')
                                                       for p in sorted(input_glob(k3, f'{s}__*__inducer.json')))}

    def wl(x, y):
        d = [arms[x][k]['levels_completed'] - arms[y][k]['levels_completed'] for k in keys]
        return sum(v > 0 for v in d), sum(v < 0 for v in d)
    c.src('v6 run files cover the same 98 games', sorted(arms['v6']) == keys, True)
    c.src('v6 levels / wins (runs)', (sum(g['levels_completed'] for g in arms['v6'].values()),
                                     sum(bool(g['win']) for g in arms['v6'].values())), (109, 6))
    for y, exp in (('general', (29, 13, 48)), ('inducer', (12, 2, 26)), ('gated', (28, 12, 47)), ('random', (64, 32, 97))):
        c.src(f'v6 - {y} [CI] (recomputed)', paired('v6', y), exp)
    c.src('v6 vs general W/L (recomputed)', wl('v6', 'general'), (13, 0))
    c.src('v6 vs v5 W/L (recomputed)', wl('v6', 'inducer'), (6, 1))
    g = vp['v6_vs_general']
    c.src('stored pooled summary', (vp['v6']['levels'], vp['v6']['wins'], g['sum'], g['lo'], g['hi'], g['W'], g['L']),
          (109, 6, 29, 13, 48, 13, 0))
    c.src('v5 arm in v6 file equals T3 inducer', (vp['v5']['levels'], vp['v5']['wins']), (97, 5))
    c.need('v6 109 levels (6W)', [r'109levels\(6W', r'109levels\(6wins', r'109L\(6W\)'], ['FL', 'AP'])
    c.need('+29 [+13,+48]', r'\+29(L|levels)?.{0,30}\[\+13,\+48\]', ['FL', 'AP'])
    c.need('13W/0L', r'13W/0L', ['FL', 'AP'])
    c.need('eta_v6 = 0/13', r'0/13', ['FL', 'AP'])
    out.append(c)

    c = Claim('T5', '98 held-out games, +Op1 tile graph and +Op2 switch-connectivity + rewind')
    B = ROOT / '.ai/blind'
    V7 = J(B / 'inducer_v7_combined_results.json')
    CK = J(B / 'patch_autonomous_click_connectivity/inducer_v7_click_results.json')
    GR = J(B / 'patch_autonomous_object_graph/inducer_v7_graph_results.json')
    held = [g for g in V7['per_game'] if g['suite'] in ('mother1102', 'tufaweb1102', 'turducken49')]
    pub = [g for g in V7['per_game'] if g['suite'] == 'public25']
    v7arms = ('general', 'v6', 'v7_graph', 'v7_click', 'v7')
    tot = lambda rows, a: (sum(g[a] for g in rows), sum(g[a] == g['total'] for g in rows))

    def wl7(x, y, rows=held):
        d = [g[x] - g[y] for g in rows]
        return sum(d), sum(v > 0 for v in d), sum(v < 0 for v in d)
    c.src('per-game rows: 98 held-out + 25 Public-25', (len(held), len(pub)), (98, 25))
    c.src('per-game general / v6 equal the T1 / T4 run files',
          ({(g['suite'], g['game']): g['general'] for g in held} == {k: arms['general'][k]['levels_completed'] for k in keys},
           {(g['suite'], g['game']): g['v6'] for g in held} == {k: arms['v6'][k]['levels_completed'] for k in keys}),
          (True, True))
    c.src('held-out levels / wins per arm (per game)', [tot(held, a) for a in v7arms],
          [(80, 4), (109, 6), (113, 7), (125, 8), (129, 9)])
    c.src('v7 takes the graph or the click arm on every game', all(g['v7'] in (g['v7_graph'], g['v7_click']) for g in held), True)
    c.src('v7 - general: sum, W, L (per game)', wl7('v7', 'general'), (49, 17, 0))
    c.src('v7 - v6: sum, W, L (per game)', wl7('v7', 'v6'), (20, 5, 0))
    c.src('won_vs_general list = 17 games', len(V7['won_vs_general']), 17)
    hs = V7['suites']['held-out 98']
    c.src('stored v7 vs general [CI] W/L', (hs['v7_vs_general']['delta'], *hs['v7_vs_general']['ci95'],
                                           hs['v7_vs_general']['W'], hs['v7_vs_general']['L']), (49, 24, 79, 17, 0))
    c.src('stored v7 vs v6 [CI] W/L', (hs['v7_vs_v6']['delta'], *hs['v7_vs_v6']['ci95'], hs['v7_vs_v6']['W'],
                                      hs['v7_vs_v6']['L']), (20, 2, 45, 5, 0))
    c.src('per-suite v7 levels / wins (Mother, Tufa-Web, Turducken)',
          [(V7['suites'][s]['v7']['levels'], V7['suites'][s]['v7']['wins']) for s in ('Mother-24', 'Tufa-Web-25', 'Turducken-49')],
          [(55, 6), (25, 2), (49, 1)])
    ck = CK['main']['held-out 98']
    c.src('click file: v7 125L 8W, general 80, v6 109', (ck['v7']['levels'], ck['v7']['wins'], ck['general']['levels'],
                                                        ck['v6']['levels']), (125, 8, 80, 109))
    c.src('click file: v7 - general W/L', (CK['paired']['v7-general held-out 98']['wins'],
                                          CK['paired']['v7-general held-out 98']['losses']), (15, 0))
    gp = {k: sum(GR['suites'][s][k]['levels'] for s in ('mother1102', 'tufaweb1102', 'turducken49')) for k in ('general', 'v6', 'v7')}
    gw = sum(GR['suites'][s]['v7']['wins'] for s in ('mother1102', 'tufaweb1102', 'turducken49'))
    c.src('graph file: v7 113L 7W, general 80, v6 109', (gp['v7'], gw, gp['general'], gp['v6']), (113, 7, 80, 109))
    c.src('graph file: v7 - general W/L', tuple(sum(GR['suites'][s]['v7_vs_general'][k] for s in ('mother1102', 'tufaweb1102', 'turducken49'))
                                              for k in ('W', 'L')), (15, 0))
    c.src('AST audits: 0 violations (click, graph)', (CK['ast_audit']['report']['violations'], GR['ast_audit']['violations']), (0, 0))
    c.src('Public-25 v7 120/183 13W, 0W/0L vs general (per game)', (*tot(pub, 'v7'), *wl7('v7', 'general', pub)[1:]), (120, 13, 0, 0))
    tn = [(g['v7_graph'], g['v7_click'], g['v7']) for g in pub if g['game'] == 'tn36']
    c.src('Public-25 tn36: graph 0, click 1, v7 1', tn, [(0, 1, 1)])
    c.src('eta_v7 = L/(W+L) = 0/17', f"{wl7('v7', 'general')[2]}/{sum(wl7('v7', 'general')[1:])}", '0/17')
    c.need('v7 129 levels (9W)', [r'129(L|levels)\((v7,)?9(W|wins)', r'129/655levels\(9wins'], ['FL', 'AP'])
    c.need('+49 [+24,+79]', r'\+49(L|levels)?.{0,40}\[\+24,\+79\]', ['FL', 'AP', 'RD'])
    c.need('17W/0L', r'17W/0L', ['FL', 'AP', 'RD'])
    c.need('+20 [+2,+45], 5W/0L vs v6', r'\+20(L|levels)?\[\+2,\+45\],5W/0L', ['AP'])
    c.need('click 125L 8W', r'125(L|levels)(click)?8(W|wins)', ['AP'])
    c.need('graph 113L 7W', [r'113(L|levels)(graph)?7(W|wins)', r'113(/655)?levels[(,]7wins'], ['FL', 'AP'])
    c.need('Public-25 120/183 13W preserved', [r'120/183,?13(W|wins)(,0W/0L)?onPublic-25', r'120/183,13W,0W/0LonPublic-25'],
           ['FL', 'AP', 'RD'])
    c.need('Public-25 0W/0L', r'0W/0LonPublic-25', ['AP', 'RD'])
    c.need('eta_v7 = 0/17', r'0/17', ['FL', 'AP'])
    c.need('+57 over 72 pre-exposure', r'\+57L?(vs\.|over)72', ['AP'])
    c.need('+84 over 45 random', r'\+84(L|levels)?(vs\.|over)45', ['AP'])
    c.need('per-suite 55 6W / 25 2W / 49 1W', r'556WonMother-\w+,252WonTufa-Web-\w+,491WonTurducken-49', ['AP'])
    # 74 Bare-Prior, 105 +Op1, 143 +Op1..Op3: patch_autonomous_inducer_v9/inducer_v9_all_results.json
    ratio_conflicts(c, '655', {45, 72, 74, 80, 97, 105, 109, 113, 125, 129, 132, 143, 146, 148, 655}, 'N/655')
    out.append(c)

    c = Claim('T2', 'Nine-suite ledger (316 entries, 315 games)')
    tot = a9['totals']
    rec = a9['from_receipts']['totals_where_receipted']
    c.src('entries / distinct games', (tot['games'], a9['from_receipts']['duplicate_entry']['distinct_games']), (316, 315))
    c.src('solvable / cleared at 1,200', (tot['solvable'], tot['giants_levels']), (1794, 1793))
    c.src('wins / reachable games', (tot['giants_wins'], tot['reachable_games']), (293, 294))
    c.src('unreachable levels / deadlocked games (receipts)', (rec['unreachable_levels'], rec['deadlocked_games']), (96, 22))
    c.src('indexed levels (receipts)', rec['indexed'], 1890)
    tt = a9['from_receipts']['tt03']
    c.src('tt03 at 5,000 actions clears the last level', (tt['budget'], tt['per_level'][-1], tt['actions_whole_game']),
          (5000, 3299, 3471))
    c.src('Table K1 baseline total (ledger)', (tot['baseline_levels'], tot['baseline_wins']), (468, 58))
    c.src('ledger baseline rows equal the per-suite receipts', a9['baseline_rows_match_receipts'], True)
    t2 = a9['table2_groups']['Mother+Web+Turducken']
    tu = U['suites']['turducken49']['full']
    c.src('Turducken-49 baseline = frozen pre-exposure snapshot (universality full)',
          (a9['as_printed_in_paper'][8]['baseline_levels'], a9['as_printed_in_paper'][8]['baseline_wins']),
          (tu['levels'], tu['wins']))
    c.src('held-out baseline 32+10+30 = ungated full', (t2['baseline_levels'], t2['baseline_wins']),
          (P['full'], P['wins']['full']))
    # parse Table K1 itself and sum its columns
    k1 = re.search(r'### K1\..*?\n\s*((?:\|.*\n)+)', TEXT['AP']).group(1).splitlines()
    rows = [[x.strip() for x in ln.strip('|').split('|')] for ln in k1[2:]]
    body, total = rows[:-1], rows[-1]
    num = lambda s: int(re.sub(r'[*,]', '', re.match(r'\**\s*([\d,]+)', s).group(1)))
    sums = {'games': sum(num(r[1]) for r in body), 'solvable': sum(num(r[2]) for r in body),
            'baseline': sum(num(r[3]) for r in body),
            'baseline_wins': sum(int(re.search(r'\((\d+) WIN', r[3]).group(1)) for r in body),
            'giants': sum(num(r[4]) for r in body)}
    printed = {'games': num(total[1]), 'solvable': num(total[2]), 'baseline': num(total[3]),
               'baseline_wins': int(re.search(r'\((\d+) WIN', total[3]).group(1)), 'giants': num(total[4])}
    c.src('Table K1 column sums equal its printed total row', sums, printed, sums == printed)
    ledger_k1 = {'games': tot['games'], 'solvable': tot['solvable'], 'baseline': tot['baseline_levels'],
                 'baseline_wins': tot['baseline_wins'], 'giants': tot['giants_levels']}
    c.src('Table K1 printed totals equal the ledger', printed, ledger_k1, printed == ledger_k1)
    held = [num(r[3]) for r in body if r[0].lstrip('`').startswith(('Mother', 'Tufa-Web', 'Turducken'))]
    c.src('Table K1 held-out baseline rows sum to 72', sum(held), 72)
    c.need('316', r'316', ['FL', 'AP', 'RD'])
    c.need('315 distinct games', r'315', ['FL', 'AP', 'RD'])
    c.need('1,793/1,794', r'1,793/1,794', ['FL', 'AP', 'RD'])
    c.need('293/294*', r'293/294', ['FL', 'AP', 'RD'])   # norm() strips the asterisk
    c.need('1,794/1,794 at 5,000', r'1,794/1,794', ['FL', 'AP', 'RD'])
    c.need('96 unreachable in 22 games', [r'96unreachablelevelsin22', r'1,794\(96in22games\)'], ['FL', 'AP'])
    c.need('Table K1 baseline total 468 (58 WIN)', r'468\(58WIN\)', ['AP'])
    c.need('held-out baseline 32 + 10 + 30', r'32\+10\+30', ['FL', 'AP'])
    ratio_conflicts(c, r'1,79\d', {1793, 1794}, 'N/1,79x')
    ratio_conflicts(c, '294', {293}, 'N/294')
    # receipt-backed baseline rows that disagree with the printed K1 baseline column (reported, not gating)
    printed_rows = {r['suite']: r for r in a9['as_printed_in_paper']}
    diffs = []
    for r in a9['from_receipts']['rows']:
        p = printed_rows.get(r['suite'])
        if p and r['baseline_levels'] is not None and (r['baseline_levels'], r['baseline_wins']) != (p['baseline_levels'], p['baseline_wins']):
            diffs.append({'suite': r['suite'], 'printed': (p['baseline_levels'], p['baseline_wins']),
                          'receipt': (r['baseline_levels'], r['baseline_wins'])})
    c.notes.append({'k1_baseline_rows_differing_from_receipts': diffs,
                    'receipt_baseline_total_where_receipted': rec['baseline_levels'],
                    'stale_flag_in_a9_json': {'table_k1_printed_baseline_total': a9['table_k1_printed_baseline_total'],
                                              'table_k1_baseline_total_correct': a9['table_k1_baseline_total_correct']}})
    out.append(c)
    return out


HELDOUT_TUNED = r'OnlineInducer\s*v[5-8]\b|\b(?:mo01|pb01|tc01|wc01|g038|ap01|g008)\b'


def operator_claims():
    """T6: +Op3 and +Op4 from their run receipts, +Op4 Full as the per-game union, the certificate abort rate, the
    L2 local D4 rule audit, and no held-out iteration labels in the manuscript."""
    import generate_xray_figures as gx
    B = ROOT / '.ai/blind'
    V8 = J(B / 'patch_autonomous_inducer_v8/inducer_v8_results.json')
    V8B = J(B / 'patch_autonomous_inducer_v8b/inducer_v8b_results.json')
    D4 = J(B / 'patch_autonomous_d4_subboard/d4_subboard_and_factors_receipt.json')
    c = Claim('T6', '98 held-out games, +Op3 pip-clue CSP, +Op4 drift refit, L2 local D4 rules')
    lad = gx.component_ladder(gx.heldout_sources())   # asserts every stored arm and CI equals the recomputation
    R, T = lad['rungs'], lad['vs_zero_prior']
    m = V8['main']['held-out 98']
    c.src('+Op3 file: Op3 / zero-prior / Op1+Op2 levels, wins, errors',
          [(m[a]['levels'], m[a]['wins'], m[a]['errors']) for a in ('v8', 'general', 'v7_base')],
          [(146, 11, 0), (80, 4, 0), (129, 9, 0)])
    c.src('+Op3 per suite (Mother, Tufa-Web, Turducken)',
          [(V8['main'][s]['v8']['levels'], V8['main'][s]['v8']['wins']) for s in ('Mother-24', 'Tufa-Web-25', 'Turducken-49')],
          [(61, 7), (35, 3), (50, 1)])
    c.src('+Op3 vs zero-prior: sum, CI, W, L (per game = stored)', T['+Op3 Clue CSP'], [66, 36, 101, 21, 0])
    p = V8['paired']['v8-v7_base held-out 98']
    c.src('+Op3 over Op1+Op2: per game = stored', T['+Op3 over Op2'], [p['delta'], *p['ci95'], p['wins'], p['losses']])
    c.src('+Op3 over Op1+Op2: sum, CI, W, L', T['+Op3 over Op2'], [17, 1, 40, 4, 0])
    ho = V8B['held_out']['all']
    c.src('Op4 file: Op4 on Op1+Op2 levels, wins, errors; Op1+Op2 base', (ho['v8b']['levels'], ho['v8b']['wins'],
          ho['v8b']['errors'], ho['v7']['levels']), (132, 9, 0, 129))
    c.src('Op4 vs zero-prior: sum, CI, W, L (per game = stored)', T['Op4 alone'], [52, 27, 82, 19, 0])
    c.src('+Op4 Full = per-game max(+Op3, Op4): levels, wins', R['+Op4 Full'], [148, 11])
    c.src('+Op4 Full vs zero-prior: sum, CI, W, L (recomputed)', T['+Op4 Full'], [68, 38, 103, 22, 0])
    c.src('eta_game = L/(W+L)', f"{T['+Op4 Full'][4]}/{sum(T['+Op4 Full'][3:])}", '0/22')
    c.src('+Op4 Full over 72 pre-exposure', R['+Op4 Full'][0] - 72, 76)
    c.src('Public-25 under +Op3 and Op4: 120/183 13W, 0W/0L',
          (lad['public25'], (V8B['public25']['all']['v8b']['levels'], V8B['public25']['all']['v8b']['wins'],
                             V8B['public25']['all']['v8b_vs_general']['W'], V8B['public25']['all']['v8b_vs_general']['L'])),
          ([120, 13, 0, 0], (120, 13, 0, 0)))
    c.src('AST audits: 0 violations (+Op3, Op4)', (V8['ast_audit']['report']['violations'], V8B['ast_audit']['violations']), (0, 0))
    cert = gx.heldout_sources()['certificates']
    c.src('eta_cert: aborted / issued certificates, 0 probes', (cert['false_goal'] + cert['false_unlock'], cert['issued'],
          cert['goal'] + cert['unlock'] + cert['optimistic'], cert['probes_planned']), (36, 361, 361, 0))
    l2 = gx.l2_sources()
    vf = lambda r: (r['rules_verified'], r['rules_fired'])
    c.src('L2 Public-25 reuse: verified / fired, pairs, non-identity',
          (*vf(l2['public25_reuse']), l2['public25_reuse']['pairs_with_verified_rule'], l2['public25_reuse']['pairs'],
           l2['public25_reuse']['verified_under_non_identity_isometry']), (76, 102, 50, 378, 31))
    c.src('L2 Public-25 local window / click-reuse / click support >= 3',
          (vf(l2['public25_reuse_local']), vf(l2['public25_click']), vf(l2['public25_click_support3']),
           l2['public25_click_support3']['rules_wrong']), ((86, 102), (69, 74), (27, 27), 0))
    c.src('L2 held-out 49: verified / fired, pairs, non-identity, local window',
          (*vf(l2['heldout']), l2['heldout']['pairs_with_verified_rule'], l2['heldout']['pairs'],
           l2['heldout']['verified_under_non_identity_isometry'], vf(l2['heldout_local'])), (150, 183, 103, 651, 27, (163, 183)))
    c.src('L2 rates 74.5% / 82.0% / 89.1%', (round(100 * 76 / 102, 1), round(100 * 150 / 183, 1), round(100 * 163 / 183, 1)),
          (74.5, 82.0, 89.1))
    leaks = {k: sorted(set(m.group(0) for m in re.finditer(HELDOUT_TUNED, TEXT[k]))) for k in ('FL', 'AP')}
    c.src('no OnlineInducer v5..v8 or held-out tuning game names in frontier-lock.md / appendix.md', leaks,
          {'FL': [], 'AP': []})
    out = [c]

    # T7: the single frozen v9 file (cumulative, LOO, Bare-Prior, certificates, Scorecard G) and the v9b default
    V9 = J(B / 'patch_autonomous_inducer_v9/inducer_v9_all_results.json')
    src = gx.heldout_sources()
    c = Claim('T7', 'Single frozen OnlineInducer file: cumulative + LOO, Bare-Prior, Virgin-120, eta, Scorecard G')
    A9, P9 = lad['v9_arms'], lad['v9_paired']   # recomputed per game; component_ladder asserts them against V9
    m9 = V9['main']['Held-out 98']
    c.src('v9 cumulative arms general / L1 / L2 / L3 / full (levels, wins)',
          [(m9[a]['levels'], m9[a]['wins']) for a in ('general', 'L1_op1', 'L2_op12', 'L3_op123', 'v9_full')],
          [(80, 4), (105, 7), (129, 9), (143, 11), (146, 11)])
    c.src('v9 LOO arms Op1 / Op2 / Op3 / Op4 (per game)', [A9[k] for k in ('loo1', 'loo2', 'loo3', 'L3')],
          [[120, 8], [120, 9], [132, 9], [143, 11]])
    c.src('v9 full minus LOO: sum, W, L', [[P9[k][0], P9[k][3], P9[k][4]] for k in ('loo1', 'loo2', 'loo3', 'L3')],
          [[26, 14, 0], [26, 5, 0], [14, 2, 0], [3, 2, 0]])
    c.src('Bare-Prior 74 (1W) -> +OnlineInducer 109 (6W), per game', (A9['bare'], A9['bare_v9']), ([74, 1], [109, 6]))
    c.src('Bare-Prior + OnlineInducer - Bare-Prior: sum, W, L; stored CI',
          ([P9['bare_v9'][0], P9['bare_v9'][3], P9['bare_v9'][4]], lad['v9_stored_ci']['bare_v9']), ([35, 11, 5], [8, 66]))
    pg = V9['paired']['v9_full - general | Held-out 98']
    c.src('v9 full - general: sum, CI, W, L (stored)', [pg['delta'], *pg['ci95'], pg['wins'], pg['losses']], [66, 36, 101, 21, 0])
    keys = [r[0] for r in src['per_game']]
    zero = {r[0]: r[2] for r in src['per_game']}
    R0 = src['v9b_r0']
    d = [R0[k][0] - zero[k] for k in keys]
    r0 = [sum(R0[k][0] for k in keys), sum(R0[k][1] for k in keys), sum(d), sum(x > 0 for x in d), sum(x < 0 for x in d)]
    c.src('v9b default (refit off): levels, wins, minus general sum, W, L (per game)', r0, [148, 11, 68, 22, 0])
    c.src('v9b default Public-25 levels, wins', src['v9b_r0_public25'], [120, 13])
    vr = src['virgin']
    vi = {k: i for i, k in enumerate(vr['columns'], 1)}
    dv = [r[vi['vbare_v9']] - r[vi['vbare']] for r in vr['per_game']]
    c.src('Virgin-120 Bare-Prior 92 -> 99: sum, W, L (per game)', [sum(dv), sum(x > 0 for x in dv), sum(x < 0 for x in dv)], [7, 5, 2])
    ce = src['v9_certificates']
    c.src('eta_cert held-out 44/380, eta_game 0/89', (ce['Held-out 98']['N_cert_aborted'], ce['Held-out 98']['N_cert_issued'],
          ce['Held-out 98']['eta_game']['net_loss_games'], ce['Held-out 98']['eta_game']['activated_games']), (44, 380, 0, 89))
    c.src('eta_cert Virgin bare floor 3/40', (ce['Virgin (bare floor)']['N_cert_aborted'], ce['Virgin (bare floor)']['N_cert_issued']), (3, 40))
    sc = V9['scorecard_seed0']
    pp = lambda g, n: (sc[g]['paired'][n]['mean_G_a'], sc[g]['paired'][n]['mean_G_b'], float(f"{sc[g]['paired'][n]['wilcoxon_p']:.3g}"))
    c.src('Scorecard G Public-25 open-224 -> gated', pp('Public-25', 'ungated_open_224 -> gated_general'), (15.21, 58.432, 5.96e-08))
    c.src('Scorecard G held-out general -> v9', pp('Held-out 98', 'gated_general -> gated_v9'), (5.836, 15.421, 1.07e-04))
    c.src('Scorecard G held-out open-224 -> v9', pp('Held-out 98', 'ungated_open_224 -> gated_v9'), (1.994, 15.421, 2.93e-07))
    c.src('Scorecard G held-out open-224 -> general', pp('Held-out 98', 'ungated_open_224 -> gated_general'), (1.994, 5.836, 4.63e-04))
    c.src('Scorecard G Virgin open-224 -> gated', pp('Virgin', 'ungated_open_224 -> gated_general'), (4.92, 10.601, 1.83e-06))
    c.src('v9 AST audit 0 violations; preregistered plugin hash unchanged',
          (V9['ast_audit']['violations'], V9['preregistration']['matches_now'], V9['preregistration']['first_virgin_row_after_receipt']),
          (0, True, True))
    c.need('148L (11W) on the held-out 98', [r'148L\(11W', r'148/655\(11W', r'148`?levels,`?11`?wins'], ['FL', 'AP'])
    c.need('+68L over general', [r'\+68L', r'\+68overgeneral'], ['FL', 'AP', 'RD'])
    c.need('146L, +66L [+36,+101], 21W/0L', r'\+66L?\[\+36,\+101\],21W/0L', ['FL', 'AP'])
    c.need('Bare-Prior 74L -> 109L, +35L [+8,+66], 11W/5L', r'\+35L\[\+8,\+66\],11W/5L', ['FL', 'AP'])
    c.need('LOO -26L / -14L / -3L', [r'LOO-26L', r'LOO-14L', r'LOO-3L'], ['FL', 'AP'])
    c.need('eta_cert = 44/380 = 11.6%', r'44/380=11\.6%', ['FL', 'AP'])
    c.need('eta_game = 0/89', r'0/89', ['FL', 'AP'])
    c.need('Virgin-120 92 -> 99, +7L, 5W/2L', r'\+7L.{0,30}5W/2L', ['FL', 'AP'])
    c.need('Virgin bare eta_cert 3/40 = 7.5%', r'3/40=7\.5%', ['FL', 'AP'])
    c.need('Scorecard G 58.43 vs 15.21, p = 5.96e-8', r'58\.43vs\.?15\.21', ['FL', 'AP'])
    c.need('Scorecard G 15.42, p = 2.93e-7', r'15\.42', ['FL', 'AP'])
    c.need('Scorecard G 10.60 vs 4.92, p = 1.83e-6', r'10\.60vs\.?4\.92', ['FL', 'AP'])
    # the measured v9b default against general is 22W/0L; a "148 ... 21W/0L" pairing misstates it
    for k in ('FL', 'AP', 'RD'):
        for m in re.finditer(r'148[^.]{0,60}?\+68L?[^.]{0,20}?,(\d+)W(?:-(\d+)W)?/0L', NORM[k]):
            if m.group(2) is None and m.group(1) != '22':
                c.conflict('148L vs general W/L (per game 22W/0L)', k, m.group(0), ['22W/0L'])
    out.append(c)
    # T6 text readings: +Op3 and the L2 local D4 rule audit as the locked documents print them
    c = out[0]
    c.need('+Op3 146L (11W)', [r'146(L|levels)\(11(W|wins)', r'146levels,11wins', r'146L,11W'], ['FL', 'AP'])
    c.need('L2 76/102 (74.5%)', r'76/102(\(|=)74\.5%', ['FL', 'AP', 'RD'])
    c.need('L2 50/378 reuse pairs', r'50/378', ['FL', 'AP', 'RD'])
    c.need('L2 31/76 non-identity', r'31/76', ['FL', 'AP', 'RD'])
    c.need('L2 27/27 = 100.0% at support >= 3', r'27/27=100\.0%', ['FL', 'AP', 'RD'])
    c.need('L2 69/74 = 93.2% click-reuse', r'69/74=93\.2%', ['FL', 'AP'])
    c.need('L2 86/102 = 84.3% local', r'86/102=84\.3%', ['FL', 'AP'])
    c.need('L2 150/183 (82.0%)', r'150/183(\(|=)82\.0%', ['FL', 'AP', 'RD'])
    c.need('L2 103/651 held-out pairs', r'103/651', ['FL', 'AP', 'RD'])
    c.need('L2 163/183 = 89.1% local', r'163/183=89\.1%', ['FL', 'AP'])
    ratio_conflicts(c, '102', {76, 86}, 'N/102')
    ratio_conflicts(c, '361', {36}, 'N/361')
    ratio_conflicts(c, '380', {44}, 'N/380')
    ratio_conflicts(c, '651', {0, 103}, 'N/651')   # 0: heldout_mother_web_49.all.l0_whole_frame.fires
    return out


def memory_claims():
    a9 = J(OUT / 'a9_d4_and_reachability_audit.json')
    d4, pr = a9['d4_algebra'], a9['d4_pairs_public25']
    c = Claim('M1', 'CrossLevelMemoryBank: D4 algebra, latency, false transfers')
    c.src('Cayley checks passed', (d4['cayley_composition']['passed'], d4['cayley_composition']['checks']), (768, 768))
    c.src('point / direction checks', (d4['point_vs_pixels']['passed'], d4['direction_vs_point']['passed']), (96, 32))
    tbl = d4['cayley_table_64x64']
    closed = all(sorted(row) == list(range(8)) for row in tbl) and all(sorted(col) == list(range(8)) for col in zip(*tbl))
    assoc = all(tbl[tbl[a][b]][cc] == tbl[a][tbl[b][cc]] for a in range(8) for b in range(8) for cc in range(8))
    c.src('Cayley table is a Latin square and associative (recomputed)', closed and assoc, True)
    c.src('median / P99 ms over 597 pairs', (round(pr['all']['median_ms'], 2), round(pr['all']['p99_ms'], 2)), (1.10, 3.80))
    c.src('non-reuse pairs proposing an action', (pr['non_reuse']['proposes_action'], pr['non_reuse']['pairs']), (0, 219))
    c.src('all pairs proposing an action', pr['reuse']['proposes_action'] + pr['non_reuse']['proposes_action'], 0)
    c.src('pair count', (pr['unordered_pairs'], pr['human_reuse_union']), (597, 378))
    c.src('planner / baseline reach reuse target', (a9['reachability_public25']['planner_reaches_level_j'],
                                                   a9['reachability_public25']['baseline_any_run_reaches_level_j']), (378, 93))
    c.src('board bounding box changes', pr['reuse']['no_shape_compatible_candidate'], 258)
    c.need('768 Cayley checks', r'768(/768)?Cayley', ['FL', 'AP', 'RD'])
    c.need('1.10 ms median', r'1\.10ms(median)?', ['FL', 'AP', 'RD'])
    c.need('3.80 ms P99', r'3\.80msP99', ['FL', 'AP', 'RD'])
    c.need('0/219', r'0/219', ['FL', 'AP', 'RD'])
    c.need('378/378 vs 93/378', r'378/378.{0,40}93/378', ['FL', 'AP'])
    ratio_conflicts(c, '219', {0, 219}, 'N/219')
    ratio_conflicts(c, '768', {768}, 'N/768')
    return [c]


def naming_warnings():
    """Same held-out suite printed under two names across the documents (reported, not gating)."""
    w = []
    for a, b in (('Mother-24', 'Mother-1102'), ('Tufa-Web-25', 'Tufa-Web-1102')):
        where = {k: (NORM[k].count(a), NORM[k].count(b)) for k in ('FL', 'AP', 'RD')}
        if any(x for x, _ in where.values()) and any(y for _, y in where.values()):
            w.append({'names': [a, b], 'counts_per_doc': where})
    return w


def invariant_2():
    t0 = time.time()
    claims, errors = [], []
    w_claim, words = wordcount_claim()
    claims.append(w_claim)
    for fn in (hidden_claims, proposition_claims, public25_claims, heldout_claims, operator_claims, memory_claims):
        try:
            claims += fn()
        except Exception:
            errors.append({'group': fn.__name__, 'traceback': traceback.format_exc()})
    return {'passed': not errors and all(c.ok for c in claims), 'wordcount': words,
            'claims': [c.dump() for c in claims], 'errors': errors, 'warnings': {'suite_naming': naming_warnings()},
            'runtime_s': round(time.time() - t0, 2)}


# ---------------------------------------------------------------- invariant 3

def invariant_3():
    from generate_xray_figures import hue_audit
    shipped = {}
    for name in MAIN_FIGURES:
        png = PAPER / 'figures' / f'{name}.png'
        shipped[name] = {**hue_audit(png), 'sha256': sha(png)}
    extra = {p.stem: hue_audit(p) for p in sorted((PAPER / 'figures').glob('*.png')) if p.stem not in MAIN_FIGURES}
    recorded = J(OUT / 'xray_figcheck.json')
    # live rerun: draw every figure again with the repo code, saving into this directory only
    rerun_dir = HERE / 'figure_rerun'
    rerun_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import generate_xray_figures as gx
    gx.FIG = rerun_dir
    gx.setup()
    X = gx.J('x0_specimens')
    report, errs = {}, {}
    t0 = time.time()
    for name, fn in (('hazard', gx.figure_one), ('regime', gx.figure_two), ('hazard_by_dataset', gx.figure_three),
                     ('architecture', gx.figure_architecture), ('cover', gx.figure_cover)):
        try:
            fn(X, report)
        except AssertionError:
            errs[name] = 'figcheck or hue assertion failed (hits recorded below)'
        except Exception:
            errs[name] = traceback.format_exc()
    live = {}
    for name in MAIN_FIGURES:
        r = report.get(name, {})
        png = rerun_dir / f'{name}.png'
        live[name] = {'figcheck_hits': r.get('figcheck_hits'), 'hits': r.get('hits'), 'green_px': r.get('green_px'),
                      'rerun_png_sha256': sha(png) if png.exists() else None,
                      'rerun_png_equals_shipped': png.exists() and sha(png) == shipped[name]['sha256']}
    per = {n: {'shipped_green_px': shipped[n]['green_px'], 'shipped_red_px': shipped[n]['red_px'],
               'live_figcheck_hits': live[n]['figcheck_hits'], 'live_green_px': live[n]['green_px'],
               'recorded_figcheck_hits': recorded.get(n, {}).get('figcheck_hits'),
               'rerun_png_equals_shipped': live[n]['rerun_png_equals_shipped']} for n in MAIN_FIGURES}
    passed = (not errs and all(p['shipped_green_px'] == 0 and p['live_figcheck_hits'] == 0 and p['live_green_px'] == 0
                               for p in per.values()))
    return {'passed': passed, 'figures': per, 'live_rerun': live, 'shipped_hue_audit': shipped,
            'other_pngs_in_folder': extra, 'errors': errs, 'rerun_dir': str(rerun_dir.relative_to(ROOT)),
            'runtime_s': round(time.time() - t0, 1)}


# ---------------------------------------------------------------- invariant 4 and main

def report_md(rec):
    I1, I2, I3 = rec['invariant_1'], rec['invariant_2'], rec['invariant_3']
    mark = lambda b: 'PASS' if b else 'FAIL'
    L = [f"# Submission preflight · {rec['created_at_jst']}", '',
         f"**{'All four invariants pass.' if rec['passed'] else 'Failing: ' + ', '.join(rec['failed_invariants'])}**", '',
         '| Invariant | Result | Reading |', '| :- | :- | :- |',
         f"| 1 Forbidden terms | {mark(I1['passed'])} | {I1['hit_count']} hits in 4 sources; extended scan (LaTeX, notebook) {I1['extended_scan']['hit_count']} hits |",
         f"| 2 Core statistics | {mark(I2['passed'])} | {sum(c['passed'] for c in I2['claims'])}/{len(I2['claims'])} claim groups; Writeup {I2['wordcount']} words |",
         f"| 3 Figures | {mark(I3['passed'])} | " + ', '.join(f"{n} {v['live_figcheck_hits']}h/{v['shipped_green_px']}g" for n, v in I3['figures'].items()) + ' |',
         f"| 4 Receipt | {mark(True)} | this file + preflight_audit_receipt.json |", '']
    if I1['hits'] or I1['extended_scan']['hits']:
        L += ['## Forbidden-term hits', '']
        for h in I1['hits'] + I1['extended_scan']['hits']:
            L.append(f"- `{h['file']}:{h['line']}` `{h['term']}`: …{h['context']}…")
        L.append('')
    bad = [c for c in I2['claims'] if not c['passed']]
    if bad or I2['errors']:
        L += ['## Failing statistics', '']
        for c in bad:
            L.append(f"**{c['id']} {c['title']}**")
            for s in c['source_checks']:
                if not s['ok']:
                    L.append(f"- source: {s['what']}: observed `{s['observed']}`, expected `{s['expected']}`")
            for t in c['text_checks']:
                if not t['ok']:
                    miss = [k for k in t['home_docs'] if not t['present'][k]]
                    L.append(f"- text: `{t['reading']}` missing in {', '.join(miss)}")
            for x in c['conflicts']:
                L.append(f"- conflict: {x['what']} in {x['doc']}: `{x['found']}` (allowed `{x['allowed']}`)")
            L.append('')
        for e in I2['errors']:
            L += [f"**error in {e['group']}**", '```', e['traceback'].strip(), '```', '']
    L += ['## Passing statistics', '']
    for c in I2['claims']:
        if c['passed']:
            L.append(f"- {c['id']} {c['title']}: {len(c['source_checks'])} source checks, {len(c['text_checks'])} text checks")
    L.append('')
    notes = []
    for c in I2['claims']:
        if c['id'] == 'T2' and c['notes'][0]['k1_baseline_rows_differing_from_receipts']:
            notes.append('Table K1 baseline column: ' + '; '.join(
                f"{d['suite']} printed {d['printed']} vs receipt {d['receipt']}"
                for d in c['notes'][0]['k1_baseline_rows_differing_from_receipts']))
        if c['id'] == 'P1':
            notes.append(c['notes'][0])
    for w in I2['warnings']['suite_naming']:
        notes.append(f"Same suite under two names: {w['names'][0]} / {w['names'][1]} (counts per doc FL/AP/RD: {w['counts_per_doc']})")
    if any(not v['rerun_png_equals_shipped'] for v in I3['figures'].values()):
        notes.append('Live rerun PNG differs byte-wise from shipped: ' + ', '.join(
            n for n, v in I3['figures'].items() if not v['rerun_png_equals_shipped']))
    if notes:
        L += ['## Non-gating notes', ''] + [f'- {n}' for n in notes] + ['']
    L += [f"Runtime {rec['runtime_s']} s · script `{rec['script_sha256'][:12]}` · sources: "
          + ', '.join(f"{k} `{v[:12]}`" for k, v in rec['source_sha256'].items())]
    return '\n'.join(L) + '\n'


def main():
    global HERE
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-dir', type=Path, default=HERE)
    HERE = parser.parse_args().receipt_dir.resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    prepare_archived_figures()
    t0 = time.time()
    rec = {'script': str(Path(__file__).resolve().relative_to(ROOT)), 'script_sha256': sha(Path(__file__).resolve()),
           'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}}
    for k, fn in (('invariant_1', invariant_1), ('invariant_2', invariant_2), ('invariant_3', invariant_3)):
        try:
            rec[k] = fn()
        except Exception:
            rec[k] = {'passed': False, 'errors': [{'traceback': traceback.format_exc()}]}
    # sources must not move while the audit ran
    after = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    rec['sources_unchanged_during_audit'] = after == rec['source_sha256']
    rec['invariant_4'] = {'passed': True, 'receipt': str((HERE / 'preflight_audit_receipt.json').relative_to(ROOT)),
                          'report': str((HERE / 'PREFLIGHT_REPORT.md').relative_to(ROOT))}
    rec['failed_invariants'] = [k for k in ('invariant_1', 'invariant_2', 'invariant_3') if not rec[k]['passed']]
    if not rec['sources_unchanged_during_audit']:
        rec['failed_invariants'].append('sources_changed_during_audit')
    rec['passed'] = not rec['failed_invariants']
    rec['state'] = 'done'
    rec['reason'] = 'complete'
    rec['runtime_healthy'] = True
    rec['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    rec['created_at_jst'] = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')
    rec['runtime_s'] = round(time.time() - t0, 1)
    (HERE / 'preflight_audit_receipt.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + '\n')
    (HERE / 'PREFLIGHT_REPORT.md').write_text(report_md(rec))
    print(json.dumps({'passed': rec['passed'], 'failed': rec['failed_invariants'],
                      'inv1_hits': rec['invariant_1'].get('hit_count'),
                      'inv2_failing': [c['id'] for c in rec['invariant_2'].get('claims', []) if not c['passed']],
                      'inv2_errors': len(rec['invariant_2'].get('errors', [])),
                      'inv3': rec['invariant_3'].get('figures'), 'runtime_s': rec['runtime_s']}, indent=1, default=str))
    sys.exit(0)


if __name__ == '__main__':
    main()
