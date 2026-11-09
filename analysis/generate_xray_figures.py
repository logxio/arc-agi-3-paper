"""The manuscript figures and cover, drawn from measured data only, on white scientific panels:
  Figure 1 hazard.png             tn36 action-by-level staircases (planner, people, all 33 baseline runs) with the
                                  real level frames, the 20-action hazard collapse and its W = 80 decomposition
  Figure 2 regime.png             setter reuse in the 342 human replays (grid, click and opening fingerprints, the
                                  level-pair matrix) with the D4 transfer operator on real pixels, the live
                                  time-limit ladder and the depth-vs-efficiency ladder
  Figure 3 hazard_by_dataset.png  906 game sources by action channel, six suites from baseline to planner, the
                                  10-thread clock inflation and every official hidden-set receipt
  Cover cover.png                 real pixels and the official scorecard, the certificate-gated loop, the suites
                                  and the hidden-set receipts
Inputs: evidence/traces (825 runs), analysis/out/a*.json, analysis/out/x0_specimens.json (x0_specimens.py).
One palette (palette.py); every figure is checked by figcheck.py and by a pixel hue audit.
Released under MIT-0.
"""
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import offset_copy

from pathlib import Path

import figcheck
from common import DATASETS, OUT, ROOT, attempts, cond_pass, km, load, score
from palette import FRAME, INK, MUTED, PAGE, PANEL, TIERS

FIG = ROOT / 'paper' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED = '#0F172A', '#1E293B'
W = 7.2
FS, FSS = 7.2, 6.8
NAVY, AMBER, INDIGO, GRAY = TIERS['navy'], TIERS['amber'], TIERS['indigo'], TIERS['gray']
AXC = '#1E293B'
BOLD = {'fontweight': 'bold', 'fontfamily': 'STIX Two Text'}   # the text face has no bold cut on this system
# the official ARC-AGI-3 16-color frame palette, used only to draw real frames; colour 14 is green, and the frames
# drawn here are chosen without it so no figure carries red and green together
ARC = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000', '#E53AA3', '#FF7BCC', '#F93C31', '#1E93FF',
       '#88D8F1', '#FFDC00', '#FF851B', '#921231', '#4FCC30', '#A356D6']
D4 = [(r'$I$', lambda f: f, lambda r, c: (r, c)),
      (r'$R_{90}$', lambda f: np.rot90(f, -1), lambda r, c: (c, 63 - r)),
      (r'$R_{180}$', lambda f: np.rot90(f, 2), lambda r, c: (63 - r, 63 - c)),
      (r'$R_{270}$', lambda f: np.rot90(f, 1), lambda r, c: (63 - c, r)),
      (r'$F_H$', lambda f: f[:, ::-1], lambda r, c: (r, 63 - c)),
      (r'$F_V$', lambda f: f[::-1, :], lambda r, c: (63 - r, c)),
      (r'$F_{D1}$', lambda f: f.T, lambda r, c: (c, r)),
      (r'$F_{D2}$', lambda f: np.rot90(f, 2).T, lambda r, c: (63 - c, 63 - r))]


def setup():
    from palette import register_fonts
    register_fonts()
    matplotlib.rcParams.update({
        'font.family': 'STIX Two Text', 'mathtext.fontset': 'stix', 'font.size': FS, 'axes.labelsize': FS,
        'xtick.labelsize': FSS, 'ytick.labelsize': FSS, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': AXC, 'ytick.color': AXC, 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
        'axes.unicode_minus': True, 'figure.facecolor': PAGE, 'savefig.facecolor': PAGE,
    })


def J(name):
    return json.loads((OUT / f'{name}.json').read_text())


def canvas(fig):
    fx = fig.add_axes([0, 0, 1, 1], zorder=0)
    w, h = fig.get_size_inches()
    fx.set_xlim(0, w)
    fx.set_ylim(0, h)
    fx.axis('off')
    fx.patch.set_alpha(0)
    return fx


def say(ax, x, y, s, size=FSS, color=INK, z=9, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, zorder=z, **kw)


def nudge(ax, dx=0.0, dy=0.0):
    return offset_copy(ax.transData, ax.figure, x=dx, y=dy, units='points')


def panel(fig, fx, x, y, w, h, letter, title=None):
    """A white panel with a hairline cool-gray frame, its letter and a one-line title."""
    fx.add_patch(Rectangle((x, y), w, h, fc=PAGE, ec=FRAME, lw=0.6, zorder=0.1))
    fig._panels = getattr(fig, '_panels', []) + [(x, y, w, h)]
    say(fx, x + 0.075, y + h - 0.06, letter, size=8.6, va='top', ha='left', **BOLD)
    if title:
        say(fx, x + 0.24, y + h - 0.072, title, size=FS, va='top', ha='left', color=INK, **BOLD)


def zone(fx, x, y, w, h, fc=PANEL, z=0.15):
    fx.add_patch(Rectangle((x, y), w, h, fc=fc, ec='none', zorder=z))


def axes(fig, x0, y0, x1, y1, left=True, bottom=True, grid=None):
    w, h = fig.get_size_inches()
    ax = fig.add_axes([x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h], zorder=2)
    ax.patch.set_alpha(0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXC)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(length=2.4, width=0.5, color=AXC, pad=1.6, labelcolor=INK)
    if not left:
        ax.tick_params(axis='y', left=False)
    if grid:
        ax.grid(axis=grid, color=GRAY[4], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    return ax


def dots(ax, xs, ys, fc, d=4.4, z=5, ec=PAGE, lw=0.5, marker='o'):
    ax.scatter(xs, ys, s=d ** 2, fc=fc, ec=INK, lw=0.7, zorder=z, marker=marker, clip_on=False)


def frame_img(fx, frame, x, y, s, z=4):
    """A real 64x64 frame at (x, y) inches, s inches wide."""
    f = np.array(frame)
    assert 14 not in f, 'frame carries ARC colour 14 (green)'
    fx.imshow(f, cmap=ListedColormap(ARC), vmin=0, vmax=15, interpolation='nearest', extent=(x, x + s, y, y + s),
              zorder=z, origin='upper')
    fx.add_patch(Rectangle((x, y), s, s, fc='none', ec=GRAY[2], lw=0.6, zorder=z + 0.1))


def px(x, y, s, row, col):
    """Inches of pixel (row, col) centre in a frame drawn at (x, y) with width s."""
    return x + (col + 0.5) * s / 64, y + s - (row + 0.5) * s / 64


def rings(fx, pts, d=5.2, z=5):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fx.scatter(xs, ys, s=d ** 2, fc='none', ec=NAVY[4], lw=1.1, zorder=z)
    fx.scatter(xs, ys, s=d ** 2, fc='none', ec=NAVY[1], lw=0.55, zorder=z + 0.01)


def bracket(fx, x, y0, y1, color, arm=0.03):
    fx.plot([x, x - arm, x - arm, x], [y1, y1, y0, y0], color=color, lw=0.6, zorder=5)


def arrow(fx, a, b, color=GRAY[1], lw=0.7, rad=0.0, z=5, head=0.28):
    fx.add_patch(FancyArrowPatch(a, b, arrowstyle=f'-|>,head_length={head},head_width={head * 0.55}', lw=lw,
                                 color=color, connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=z,
                                 mutation_scale=10))


def key_line(fx, x, y, color, ls='-', lw=1.2, text=None, tcolor=INK, **kw):
    fx.plot([x, x + 0.15], [y, y], color=color, lw=lw, ls=ls, zorder=6, solid_capstyle='butt')
    if text:
        say(fx, x + 0.20, y, text, ha='left', va='center', color=tcolor, **kw)


def hue_audit(png):
    """Pixel audit of a rendered figure: green-dominant pixels (G > R + 20 and G > B + 20) must be 0 whenever a
    red-dominant pixel exists. Real ARC frames are part of the image and are audited too."""
    img = (plt.imread(png)[..., :3] * 255).astype(int)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    green = int(((g > r + 20) & (g > b + 20)).sum())
    red = int(((r > g + 40) & (r > b + 40)).sum())
    return {'green_px': green, 'red_px': red, 'red_green_cofigure': bool(green and red)}


def save(fig, name, report):
    cel_finish(fig, name)
    texts = [t.get_text() for t, _ in figcheck._texts(fig)]
    from matplotlib.colors import to_rgb
    light_gray_texts=[]
    for t,_ in figcheck._texts(fig):
        c=to_rgb(t.get_color())
        if max(c)-min(c)<.20 and .35<min(c)<.97:
            light_gray_texts.append(t.get_text())
    hits = figcheck.report(fig, name)
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=300)
    plt.close(fig)
    audit = hue_audit(FIG / f'{name}.png')
    report[name] = {'figcheck_hits': len(hits), 'hits': hits, **audit,
                    'pdf_bytes': (FIG / f'{name}.pdf').stat().st_size,
                    'texts': texts, 'v9_count': sum('v9' in t.lower() for t in texts),
                    'multiline_text_count': sum('\n' in t for t in texts),
                    'light_gray_text_count': len(light_gray_texts)}
    assert not hits and audit['green_px'] == 0 and not light_gray_texts, report[name]


def staircase(per, tail=None, start=1.0):
    """Cumulative actions against levels cleared: one riser per cleared level, then the unsolved tail."""
    xs, ys, c = [start], [0.0], 0
    for l, a in enumerate(per):
        c += a
        xs += [c, c]
        ys += [l, l + 1]
    if tail:
        xs.append(c + tail)
        ys.append(len(per))
    return np.array(xs, float), np.array(ys, float)


# ======================================================================= Figure 1
def hazard_curves():
    """Per dataset: 20-action completion hazard in non-overlapping 20-action bins (>= 10 attempts at risk)."""
    out = {}
    for name in DATASETS:
        rows = attempts(load(name))
        d = np.array([r[2] for r in rows], float)
        e = np.array([r[3] for r in rows], bool)
        keep = ~((~e) & (d == 0))
        d, e = d[keep], e[keep]
        xs, hs = [], []
        for n in range(0, 200, 20):
            _, rk = km(d, e, [n])
            h = cond_pass(d, e, n, 20)
            if h is not None and rk[0] >= 10:
                xs.append(n)
                hs.append(h)
        out[name] = (np.array(xs), np.array(hs))
    return out


def tn36_runs():
    runs = [r for n in DATASETS for r in load(n) if r['game'].startswith('tn36')]
    assert len(runs) == 33, len(runs)
    return runs


def figure_one(X, report):
    H = 4.40
    fig = plt.figure(figsize=(W, H))
    fx = canvas(fig)
    sp = X['specimen']
    steps = sp['steps']
    plan = X['planner']['tn36']
    base = X['human_base']['tn36']
    runs = tn36_runs()
    k1 = sum(1 for r in runs if r['k'] >= 1)
    assert plan['win'] and sum(plan['per_level']) == 95 and max(r['k'] for r in runs) == 1

    # ---------------- a: one real game, every agent, level by level
    panel(fig, fx, 0.04, 0.04, 4.30, 4.32, 'a', 'tn36: actions spent per level, GiantsPlanner vs people vs 33 runs')
    S, y_f = 1.10, 2.74
    for x, lvl, cap in ((0.58, '1', 'its 7 planner clicks ringed'), (1.94, '2', '9 clicks, same template')):
        frame_img(fx, sp['level_frames'][lvl], x, y_f, S)
        rings(fx, [px(x, y_f, S, s['row'], s['col']) for s in steps if s['level'] == int(lvl)], d=4.6)
        say(fx, x, y_f + S + 0.04, f'level {lvl}, frame 0', ha='left', va='bottom', color=MUTED)
        say(fx, x, y_f - 0.04, cap, ha='left', va='top', color=MUTED)
    arrow(fx, (0.58 + S + 0.04, y_f + S / 2), (1.94 - 0.04, y_f + S / 2), color=NAVY[1])
    for (r0, r1), c, lab in (((0, 3), NAVY[1], r'$\Omega_{\rm HUD}$'), ((4, 63), INDIGO[1], r'$\Omega_{\rm board}$')):
        y0, y1 = y_f + S - (r1 + 1) * S / 64, y_f + S - r0 * S / 64
        bracket(fx, 0.545, y0, y1, c)
        say(fx, 0.495, (y0 + y1) / 2 + (0.03 if r0 == 0 else 0), lab, ha='right', va='center', color=c)
    # key, level with the frames
    kx, ky, dy = 3.20, 3.98, 0.128
    zone(fx, kx - 0.06, 2.31, 4.30 - kx + 0.02, 1.76)
    key_line(fx, kx, ky, NAVY[1], lw=1.4, text='GiantsPlanner', tcolor=NAVY[0], **BOLD)
    say(fx, kx + 0.20, ky - dy, f'7/7 · {sum(plan["per_level"])} act', ha='left', va='center', color=NAVY[1])
    say(fx, kx + 0.20, ky - 2 * dy, '0 tok', ha='left', va='center', color=NAVY[1])
    key_line(fx, kx, ky - 3.25 * dy, GRAY[1], lw=1.1, text=f'people: 7/7 in {sum(base)}', tcolor=INK)
    key_line(fx, kx, ky - 4.5 * dy, AMBER[2], lw=0.9, text='33 runs', tcolor=AMBER[0], **BOLD)
    say(fx, kx + 0.20, ky - 5.5 * dy, f'L1: {k1}/33', ha='left', va='center', color=AMBER[1])
    say(fx, kx + 0.20, ky - 6.5 * dy, 'L2: 0/33', ha='left', va='center', color=AMBER[1])
    key_line(fx, kx, ky - 7.75 * dy, AMBER[1], ls=(0, (2.2, 1.4)), lw=0.8, text='80-act cut', tcolor=INK)
    say(fx, kx + 0.20, ky - 8.75 * dy, '−16 clears', ha='left', va='center', color=AMBER[0], **BOLD)
    key_line(fx, kx, ky - 10 * dy, AMBER[0], ls=(0, (2.2, 1.4)), lw=0.8, text='600-act probe', tcolor=INK)
    say(fx, kx + 0.20, ky - 11 * dy, '−4.94 LB', ha='left', va='center', color=AMBER[0], **BOLD)

    ax = axes(fig, 0.52, 0.42, 4.20, 2.12, grid='y')
    ax.set_xscale('log')
    ax.set_xlim(0.85, 2200)
    ax.set_ylim(-0.55, 7.6)
    runs.sort(key=lambda r: sum(r['a'][:r['k'] + 1]))
    n = len(runs)
    for i, r in enumerate(runs):
        off = (i - (n - 1) / 2) / ((n - 1) / 2) * 0.22
        xs, ys = staircase(r['a'][:r['k']], tail=r['a'][r['k']] if r['k'] < len(r['a']) else None)
        ax.plot(xs, ys + off, color=AMBER[2] if r['k'] else AMBER[3], lw=0.55, zorder=3, gid='texture',
                solid_capstyle='round')
    xs, ys = staircase(base)
    ax.plot(xs, ys, color=GRAY[1], lw=1.1, zorder=4)
    xs, ys = staircase(plan['per_level'])
    ax.plot(xs, ys, color=NAVY[1], lw=1.5, zorder=5)
    dots(ax, [sum(plan['per_level'])], [7], NAVY[1], d=4.6, z=6)
    dots(ax, [sum(base)], [7], GRAY[1], d=4.2, z=6)
    say(ax, sum(plan['per_level']), 7, f'{sum(plan["per_level"])}', ha='left', va='center', transform=nudge(ax, 4.5),
        color=NAVY[0], **BOLD)
    say(ax, sum(base), 7, f'{sum(base)}', ha='left', va='center', transform=nudge(ax, 4.5), color=GRAY[0])
    for v, c in ((80, AMBER[1]), (600, AMBER[0])):
        ax.plot([v, v], [-0.55, 1.62], color=c, lw=0.8, ls=(0, (2.2, 1.4)), zorder=4.5)
        say(ax, v, 1.98, f'{v}', ha='center', va='center', color=c, **BOLD)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1000'])
    ax.set_yticks(range(8))
    ax.set_ylabel('levels cleared', labelpad=2)
    ax.set_xlabel('cumulative actions on tn36 (log)', labelpad=1.5)

    # ---------------- b: completion hazard of every level attempt, 8 datasets
    panel(fig, fx, 4.42, 2.30, 2.74, 2.06, 'b', '$h(n)$: clears in the next 20 actions')
    hz = hazard_curves()
    hb = axes(fig, 4.78, 2.62, 7.06, 4.02, grid='y')
    hb.set_xlim(0, 200)
    hb.set_ylim(0, 0.62)
    order = ['kaggle_v2', 'kaggle_v3', 'kaggle_v4', 'kaggle_v5', 'kaggle_v7', 'api_qwen38flash', 'tufa_qwen36_27b',
             'api_gptoss120b']
    cols = [AMBER[0], AMBER[1], AMBER[2], AMBER[3], AMBER[3], INDIGO[1], INDIGO[0], INDIGO[2]]
    per = [a for g in X['planner'].values() for a in g['per_level']]
    q1, q2, q3 = np.percentile(per, [25, 50, 75])
    hb.axvspan(q1, q3, color=NAVY[4], zorder=0.3, lw=0, alpha=0.7)
    hb.plot([q2, q2], [0, 0.62], color=NAVY[1], lw=0.9, zorder=0.4)
    lead = {'kaggle_v3', 'api_qwen38flash'}   # the two keyed curves; the other six recede
    for nm, c in zip(order, cols):
        x, h = hz[nm]
        hb.plot(x, h, color=c, lw=1.15 if nm in lead else 0.7, alpha=1.0 if nm in lead else 0.55,
                ls=(0, (2.0, 1.2)) if nm == 'kaggle_v7' else '-', zorder=3.2 if nm in lead else 3,
                solid_capstyle='round', marker='o', ms=2.0 if nm in lead else 1.5, mew=0)
    hb.axvspan(80, 200, ymax=0.16 / 0.62, color=AMBER[4], lw=0, zorder=0.2, alpha=0.8)
    hb.plot([80, 80], [0, 0.335], color=AMBER[0], lw=0.6, ls=(0, (2.4, 1.6)), zorder=2)
    a3 = J('a3_hazard')['actions']
    h80 = max(v['cond_pass_next20']['80'][0] for v in a3.values() if v['cond_pass_next20']['80'][0] is not None)
    say(hb, 86, 0.30, rf'Frontier Lock: $h(80)\leq${h80:.3f}', ha='left', va='center', color=AMBER[0], **BOLD)
    say(hb, 51, 0.42, 'planner, 183/183 levels:', ha='left', va='center', color=NAVY[0])
    say(hb, 51, 0.365, f'median {q2:.0f}, IQR {q1:.0f}–{q3:.0f} actions', ha='left', va='center', color=NAVY[1])
    hb.set_xticks([0, 40, 80, 120, 160, 200])
    hb.set_yticks([0, 0.2, 0.4, 0.6])
    hb.set_yticklabels(['0', '0.2', '0.4', '0.6'])
    hb.set_xlabel('n, actions already spent on the level', labelpad=1.5)
    for i, (lab, c) in enumerate((('Kaggle V2–V7', AMBER[1]), ('endpoints, 2 GPUs', INDIGO[1]))):
        yk = 0.575 - i * 0.055
        t_ = say(hb, 197, yk, lab, ha='right', va='center', color=INK)
        fig.canvas.draw()
        x0 = hb.transData.inverted().transform(t_.get_window_extent().p0)[0]
        hb.plot([x0 - 13, x0 - 3], [yk, yk], color=c, lw=1.15, zorder=8)

    # ---------------- c: where the drop comes from, W = 80
    panel(fig, fx, 4.42, 0.04, 2.74, 2.18, 'c', 'W = 80: stuck vs fresh attempts')
    a4 = J('a4_continue_restart')
    dc = axes(fig, 4.56, 0.42, 7.06, 1.86, left=False, grid='x')
    rows = [('Qwen3.6-27B', 'tufa_qwen36_27b'), ('Qwen3.8-Flash', 'api_qwen38flash'),
            ('gpt-oss-120b', 'api_gptoss120b'), ('Kaggle V2–V7', 'kaggle_pooled')]
    dc.set_xlim(-0.42, 1.04)
    dc.set_ylim(len(rows) - 0.45, -1.0)
    dc.set_yticks([])
    for i, (lab, key) in enumerate(rows):
        r = a4[key]['80']
        c, m, u = r['continue'][0], r['fresh'][0], r['fresh_unmatched']
        share = min(1.0, (u - m) / (u - c))
        if abs(m - c) > 0.004:
            dc.plot([min(c, m), max(c, m)], [i, i], color=INDIGO[3], lw=3.0, zorder=3, solid_capstyle='butt')
        dc.add_patch(Rectangle((max(c, m), i - 0.15), u - max(c, m), 0.30, fc=AMBER[4], ec=AMBER[2], lw=0.5,
                               zorder=3))
        dots(dc, [c], [i], GRAY[2], d=4.6, z=5)
        dots(dc, [m], [i], INDIGO[1], d=4.6, z=5.1)
        dots(dc, [u], [i], AMBER[2], d=4.6, z=5.2)
        say(dc, -0.40, i, lab, ha='left', va='center', color=INK)
        say(dc, u, i, f'{share * 100:.0f}% wall', ha='left', va='center', transform=nudge(dc, 5.0),
            color=AMBER[0], **BOLD)
    dc.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    dc.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
    dc.spines['bottom'].set_bounds(0, 0.9)
    for xk, lab, c in ((-0.38, 'stuck at 80', GRAY[2]), (0.03, 'fresh, same level', INDIGO[1]),
                       (0.50, 'fresh, any level', AMBER[2])):
        dots(dc, [xk], [-0.66], c, d=4.2, z=5)
        say(dc, xk, -0.66, lab, ha='left', va='center', color=INK, transform=nudge(dc, 4.5))
    dc.set_xlabel('clears in the next 80 actions; wall = share of the drop', labelpad=1.5)
    save(fig, 'hazard', report)


# ======================================================================= Figure 2
def reuse_pairs(X):
    """Cross-level pairs (game, i, j) in the 342 human replays that share an opening of >= 3 actions or a click
    coordinate, and the deepest level any of the 825 baseline runs reached on each game."""
    pairs = {(p['game'], p['i'], p['j']) for p in X['human_prefix_pairs'] + X['human_click_pairs']}
    reach = {}
    for n in DATASETS:
        for r in load(n):
            g = r['game'].split('-')[0]
            reach[g] = max(reach.get(g, 0), r['k'] + 1 if r['k'] < r['L'] else r['L'])
    return sorted(pairs), reach


def l2_sources():
    """Build-time reader of the engine-level L2 local k x k D4 rule audit; the notebook embeds its output instead."""
    d = json.loads((ROOT / '.ai/blind/patch_autonomous_d4_subboard/d4_subboard_and_factors_receipt.json').read_text())
    keep = ('pairs', 'pairs_with_verified_rule', 'rules_fired', 'rules_verified', 'rules_wrong',
            'verified_under_non_identity_isometry')
    pick = lambda x: {k: x[k] for k in keep}
    p, h = d['public25'], d['heldout_mother_web_49']['all']
    return {'public25_reuse': pick(p['reuse_378']['l2_local_rules']),
            'public25_reuse_local': pick(p['reuse_378']['l2_local_rules_local_hit']),
            'public25_click': pick(p['reuse_click_284']['l2_local_rules']),
            'public25_click_support3': pick(p['reuse_click_284']['l2_local_rules_support3']),
            'heldout': pick(h['l2_local_rules']), 'heldout_local': pick(h['l2_local_rules_local_hit'])}


def figure_two(X, report):
    H = 4.30
    fig = plt.figure(figsize=(W, H))
    fx = canvas(fig)
    pairs, reach = reuse_pairs(X)
    fp = X['fingerprints']
    M = max(j for _, _, j in pairs)
    n_all = len(pairs)
    n_alive = sum(1 for g, i, j in pairs if reach.get(g, 0) >= j)

    # ---------------- a: setter reuse and the D4 equivariance guard
    panel(fig, fx, 0.04, 0.04, 4.30, 4.22, 'a', 'Setter reuse in 342 human replays, and $D_4$ equivariance guard')
    zone(fx, 0.10, 2.48, 4.18, 1.52)
    src = np.array(X['specimen']['level_frames']['1'])
    clicks = [(s['row'], s['col']) for s in X['specimen']['steps'] if s['level'] == 1]
    S0, x0, y0 = 0.96, 0.18, 2.70
    frame_img(fx, src, x0, y0, S0)
    rings(fx, [px(x0, y0, S0, r, c) for r, c in clicks], d=4.2)
    say(fx, x0, y0 + S0 + 0.04, 'level k, solved', ha='left', va='bottom', color=MUTED)
    say(fx, x0, y0 - 0.04, 'tn36 pixels, 7 clicks', ha='left', va='top', color=MUTED)
    sx = 1.24
    for k, line in enumerate(('strip loops, UNDO', r'fix $\Omega_{\rm HUD}$ clicks', r'map $\Omega_{\rm board}$ by $g$')):
        say(fx, sx, 3.58 - k * 0.135, f'{k + 1}  {line}', ha='left', va='center', color=INK)
    arrow(fx, (sx, 3.13), (2.06, 3.13), color=NAVY[1], lw=0.8)
    say(fx, sx, 2.98, f'768 homomorphism', ha='left', va='center', color=NAVY[1])
    say(fx, sx, 2.85, 'checks, 0/219 false', ha='left', va='center', color=NAVY[1])
    T, tx, pitch = 0.48, 2.15, 0.535
    say(fx, tx, 3.91, '8 isometries of the board, tried on frame 0 of k+1', ha='left', va='center', color=INDIGO[0])
    for idx, (lab, fn, pt) in enumerate(D4):
        col, row = idx % 4, idx // 4
        x, y = tx + col * pitch, 3.27 - row * 0.64
        frame_img(fx, fn(src), x, y, T)
        q = [px(x, y, T, *pt(r, c)) for r, c in clicks]
        fx.scatter([a for a, _ in q], [b for _, b in q], s=1.6 ** 2, fc=NAVY[3], ec=NAVY[0], lw=0.25, zorder=5)
        say(fx, x + T / 2, y - 0.025, lab, ha='center', va='top', color=INDIGO[0])

    # level-pair matrix: pairs (i, j) reusing a click or a 3-action opening, edged where any baseline run reached j
    cnt = Counter((i, j) for _, i, j in pairs)
    alive = Counter((i, j) for g, i, j in pairs if reach.get(g, 0) >= j)
    cs, mx, my = 0.158, 0.50, 0.50
    shade = lambda v: NAVY[4] if v <= 2 else NAVY[3] if v <= 5 else NAVY[2] if v <= 10 else NAVY[1]
    for (i, j), v in cnt.items():
        x, y = mx + (j - 2) * cs, my + (M - 1 - i) * cs
        fx.add_patch(Rectangle((x + 0.006, y + 0.006), cs - 0.012, cs - 0.012, fc=shade(v),
                               ec=AMBER[1] if alive.get((i, j)) else 'none', lw=0.9, zorder=3))
        say(fx, x + cs / 2, y + cs / 2, str(v), ha='center', va='center', size=FSS,
            color=PAGE if v > 5 else NAVY[0])
        if alive.get((i,j)):
            fx.add_patch(Polygon([(x+cs-.01,y+cs-.01),(x+cs-.045,y+cs-.01),(x+cs-.01,y+cs-.045)],fc=AMBER[1],ec='none',zorder=4))
    for j in range(2, M + 1):
        say(fx, mx + (j - 1.5) * cs, my - 0.03, str(j), ha='center', va='top', color=MUTED)
    for i in range(1, M):
        say(fx, mx - 0.03, my + (M - 0.5 - i) * cs, str(i), ha='right', va='center', color=MUTED)
    say(fx, mx + (M - 1) * cs / 2, 0.21, 'target level j', ha='center', va='center', color=MUTED)
    say(fx, 0.17, my + (M - 1) * cs / 2, 'source level i', ha='center', va='center', color=MUTED, rotation=90)
    say(fx, 0.14, 2.32, 'games per level pair (i, j)', ha='left', va='center', color=NAVY[0], **BOLD)
    fx.add_patch(Rectangle((0.155, 2.12), 0.10, 0.10, fc=AMBER[2], ec=INK, lw=0.9, zorder=3))
    say(fx, 0.30, 2.17, 'a baseline run reached j', ha='left', va='center', color=AMBER[0])

    # fingerprint bars
    bx0, bx1 = 2.18, 3.42
    T_ = fp['cross_level_pairs']
    rows = [('one grid_size on every level', fp['constant_grid_games'] / fp['games'],
             f"{fp['constant_grid_games'] / fp['games'] * 100:.1f}% ({fp['constant_grid_games']}/{fp['games']})",
             'games', NAVY),
            ('a level re-clicks one (x, y)', fp['repeat_click_levels'] / fp['click_levels'],
             f"{fp['repeat_click_levels'] / fp['click_levels'] * 100:.1f}% "
             f"({fp['repeat_click_levels']}/{fp['click_levels']})", 'levels', NAVY),
            ('level pairs sharing a click (x, y)', len(X['human_click_pairs']) / T_,
             f"{len(X['human_click_pairs'])}/{T_}", 'pairs', NAVY),
            ('level pairs sharing a 3-action opening', len(X['human_prefix_pairs']) / T_,
             f"{len(X['human_prefix_pairs'])}/{T_}", 'pairs', NAVY)]
    yb = 2.30
    for lab, frac, val, _, fam in rows:
        say(fx, bx0, yb, lab, ha='left', va='center', color=INK)
        fx.add_patch(Rectangle((bx0, yb - 0.155), bx1 - bx0, 0.065, fc=GRAY[4], ec='none', zorder=2))
        fx.add_patch(Rectangle((bx0, yb - 0.155), (bx1 - bx0) * frac, 0.065, fc=fam[2], ec='none', zorder=2.1))
        say(fx, bx1 + 0.05, yb - 0.122, val, ha='left', va='center', color=fam[0], **BOLD)
        yb -= 0.30
    say(fx, bx0, yb, f'all {n_all} reuse pairs, by target level reached', ha='left', va='center', color=INK)
    fx.add_patch(Rectangle((bx0, yb - 0.155), bx1 - bx0, 0.065, fc=NAVY[1], ec='none', zorder=2))
    say(fx, bx1 + 0.05, yb - 0.122, f'planner {n_all}/{n_all}', ha='left', va='center', color=NAVY[0], **BOLD)
    yb -= 0.13
    fx.add_patch(Rectangle((bx0, yb - 0.155), bx1 - bx0, 0.065, fc=GRAY[4], ec='none', zorder=2))
    fx.add_patch(Rectangle((bx0, yb - 0.155), (bx1 - bx0) * n_alive / n_all, 0.065, fc=AMBER[2], ec='none',
                           zorder=2.1))
    say(fx, bx1 + 0.05, yb - 0.122, f'825 runs {n_alive}/{n_all}', ha='left', va='center', color=AMBER[0], **BOLD)
    # L2: local k x k D4 rules replayed in the engine, one step each
    L2 = l2_sources()
    pr, ho, s3 = L2['public25_reuse'], L2['heldout'], L2['public25_click_support3']
    yb = 0.60
    say(fx, bx0, yb, r'local $k{\times}k$ $D_4$ rules, engine-verified', ha='left', va='center', color=INK)
    for lab, r, dy in (('Public-25', pr, 0.0), ('held-out', ho, 0.12)):
        y = yb - dy
        fx.add_patch(Rectangle((bx0, y - 0.155), bx1 - bx0, 0.065, fc=GRAY[4], ec='none', zorder=2))
        fx.add_patch(Rectangle((bx0, y - 0.155), (bx1 - bx0) * r['rules_verified'] / r['rules_fired'], 0.065,
                               fc=INDIGO[2], ec='none', zorder=2.1))
        say(fx, bx1 + 0.05, y - 0.122, f"{lab} {r['rules_verified']}/{r['rules_fired']}", ha='left', va='center',
            color=INDIGO[0], **BOLD)
    pct = lambda r: f"{100 * r['rules_verified'] / r['rules_fired']:.1f}%"
    say(fx, bx0, 0.16, f"{pct(pr)} / {pct(ho)} verified; click, support $\\geq$ 3: "
        f"{s3['rules_verified']}/{s3['rules_fired']}", ha='left', va='center', color=MUTED)

    # ---------------- b: live time-limit ladder (simulated sweep and both live rounds)
    panel(fig, fx, 4.42, 2.06, 2.74, 2.20, 'b', 'Per-game time limit, levels vs control')
    a5b = J('a5b_regime')
    a8 = J('a8_confirmation')
    bx = axes(fig, 4.52, 2.42, 7.06, 3.98, left=False, grid='x')
    lim = ['5', '10', '15', '20', '25', '30']
    rows = [(f'{m} min', a5b[m]['80']['d_levels'], a5b[m]['dyn_only']['d_levels']) for m in lim]
    bx.set_xlim(-0.62, 0.52)
    bx.set_ylim(len(rows) + 2.55, -1.65)
    bx.set_yticks([])
    bx.plot([0, 0], [-0.45, len(rows) + 1.4], color=GRAY[2], lw=0.6, ls=(0, (2.4, 1.6)), zorder=1.2)
    for i, (lab, cut, dyn) in enumerate(rows):
        bx.plot([dyn[1], dyn[2]], [i - 0.14] * 2, color=INDIGO[3], lw=1.6, zorder=2, solid_capstyle='butt')
        bx.plot([cut[1], cut[2]], [i + 0.14] * 2, color=AMBER[3], lw=1.6, zorder=2, solid_capstyle='butt')
        dots(bx, [dyn[0]], [i - 0.14], INDIGO[1], d=3.8, z=5)
        dots(bx, [cut[0]], [i + 0.14], AMBER[2], d=3.8, z=5)
        say(bx, -0.60, i, lab, ha='left', va='center', color=INK)
    live = [('live, 15 min', 0.140, (0.0, 0.300), '10.7%'),
            ('live, 30 min', a8['levels']['difference'], tuple(a8['levels']['ci95']), '2.3%')]
    for k, (lab, v, ci, idle) in enumerate(live):
        y = len(rows) + 0.15 + k * 0.95
        hot = k == 1
        c = AMBER[0] if hot else AMBER[2]
        bx.plot([ci[0], ci[1]], [y, y], color=c, lw=2.2 if hot else 1.6, zorder=3, solid_capstyle='butt')
        bx.plot([ci[0]] * 2, [y - 0.2, y + 0.2], color=c, lw=0.7, zorder=3)
        bx.plot([ci[1]] * 2, [y - 0.2, y + 0.2], color=c, lw=0.7, zorder=3)
        dots(bx, [v], [y], c, d=5.2 if hot else 4.4, z=5)
        say(bx, -0.60, y, lab, ha='left', va='center', color=AMBER[0] if hot else INK, **(BOLD if hot else {}))
        say(bx, ci[1], y, f'{v:+.3f}'.replace('-', '−'), ha='left', va='center', transform=nudge(bx, 4.5),
            color=AMBER[0] if hot else MUTED, **(BOLD if hot else {}))
    say(bx, -0.60, len(rows) + 1.95, f'idle slots, last wave: {live[0][3]} (15 min), {live[1][3]} (30 min)',
        ha='left', va='center', color=MUTED)
    bx.set_xticks([-0.3, -0.15, 0, 0.15, 0.3])
    bx.set_xticklabels(['−0.3', '−0.15', '0', '+0.15', '+0.3'])
    bx.spines['bottom'].set_bounds(-0.32, 0.32)
    for xk, lab, c in ((-0.58, 'schedule only', INDIGO[1]), (-0.06, 'schedule + cut at 80', AMBER[2])):
        dots(bx, [xk], [-1.15], c, d=3.8, z=5)
        say(bx, xk, -1.15, lab, ha='left', va='center', color=INK, transform=nudge(bx, 4.5))
    bx.set_xlabel('levels per game vs control', labelpad=1.5)

    # ---------------- c: what one more level buys vs human-level efficiency on the levels already cleared
    panel(fig, fx, 4.42, 0.04, 2.74, 1.94, 'c', 'Scorer counterfactual: +1 level vs actions')
    a7 = J('a7_score_decomposition')
    cx_ = axes(fig, 4.52, 0.40, 7.06, 1.72, left=False, grid='x')
    ds = [('Kaggle V2', 'kaggle_v2'), ('Kaggle V5', 'kaggle_v5'), ('Kaggle V4', 'kaggle_v4'),
          ('Qwen3.8-Flash', 'api_qwen38flash'), ('Qwen3.6-27B', 'tufa_qwen36_27b')]
    cx_.set_xlim(-9.6, 25.5)
    cx_.set_ylim(len(ds) - 0.4, -1.05)
    cx_.set_yticks([])
    for i, (lab, key) in enumerate(ds):
        b, e, d = a7[key]['score'][0], a7[key]['all_at_human_baseline'][0], a7[key]['one_more_level'][0]
        cx_.plot([b, d], [i, i], color=NAVY[3], lw=1.6, zorder=2, solid_capstyle='butt')
        cx_.plot([b, e], [i, i], color=AMBER[2], lw=2.6, zorder=2.1, solid_capstyle='butt')
        dots(cx_, [b], [i], GRAY[2], d=4.0, z=5)
        dots(cx_, [e], [i], AMBER[2], d=4.0, z=5.1)
        dots(cx_, [d], [i], NAVY[1], d=4.8, z=5.2)
        say(cx_, -9.3, i, lab, ha='left', va='center', color=INK)
        say(cx_, d, i, f'+{(d / b - 1) * 100:.0f}%', ha='left', va='center', transform=nudge(cx_, 4.5),
            color=NAVY[0], **BOLD)
    cx_.set_xticks([0, 5, 10, 15, 20])
    cx_.spines['bottom'].set_bounds(0, 22)
    for xk, lab, c in ((-9.0, 'now', GRAY[2]), (-4.2, 'human-level actions', AMBER[2]), (10.4, '+1 level', NAVY[1])):
        dots(cx_, [xk], [-0.68], c, d=3.8, z=5)
        say(cx_, xk, -0.68, lab, ha='left', va='center', color=INK, transform=nudge(cx_, 4.5))
    cx_.set_xlabel('score counterfactual, Public-25', labelpad=1.5)
    save(fig, 'regime', report)


# ======================================================================= Figure 3
SUITES = [  # label, levels, baseline, ours, wins (a9 ledger; held-out baseline = frozen pre-exposure snapshot)
    ('Public-25', 183, 42, 183, '25/25 wins'),
    ('Parametric-25', 172, 48, 172, '25/25 wins'),
    ('Tufa-50', 232, 128, 232, '50/50 wins'),
    ('Clone-118', 552, 178, 551, '99/100* wins'),
    ('Held-Out 98', 655, 72, 655, '94/94* wins'),
]
# the frozen OnlineInducer on the 98 held-out games, one rung per operator (Table 2, Section 5)
HELDOUT_LADDER = [('Pre-exp', 72, 2), ('Zero-Prior', 80, 4), ('+Op0 Trans', 109, 6), ('+Op1 Graph', 113, 7),
                  ('+Op2 Click+Rew', 129, 9), ('+Op3 Clue CSP', 146, 11), ('+Op4 Full', 148, 11)]
HELD_SUITES = ('mother1102', 'tufaweb1102', 'turducken49')


def heldout_sources():
    """Build-time reader of the run receipts, keyed by component; the notebook embeds its output instead."""
    B = ROOT / '.ai/blind'
    rd = lambda p: json.loads((B / p).read_text())
    v6 = rd('patch_autonomous_click_inertia/online_inducer_v6_results.json')['suites']['pooled']
    v7 = rd('inducer_v7_combined_results.json')['suites']['held-out 98']
    v8 = rd('patch_autonomous_inducer_v8/inducer_v8_results.json')
    v8b = rd('patch_autonomous_inducer_v8b/inducer_v8b_results.json')
    alone = {r['key']: r for r in v8b['per_game_changed']}
    rows = []
    for key, _, total, gen, op2, op3 in sorted(v8['per_game']):
        if key.split('__')[0] in HELD_SUITES:
            assert key not in alone or alone[key]['general'] == gen, key
            rows.append([key, total, gen, op2, op3, alone[key]['v8b'] if key in alone else gen])
    arm = lambda d: [d['levels'], d['wins']]
    pv = lambda p: [p['delta'], *p['ci95'], p.get('wins', p.get('W')), p.get('losses', p.get('L'))]
    ho = v8b['held_out']['all']
    cert = rd('patch_autonomous_object_graph/inducer_v7_graph_results.json')['certificates']
    # single frozen v9 file: Bare-Prior floor and leave-one-out arms, one row per held-out game
    v9 = rd('patch_autonomous_inducer_v9/inducer_v9_all_results.json')
    v9cols = ['bare', 'bare_v9', 'v9', 'loo1', 'loo2', 'loo3', 'L3']
    at = {c: 3 + v9['per_game_columns']['dev'].index(c) for c in v9cols}
    v9rows = sorted([r[0], *(r[at[c]] for c in v9cols)] for r in v9['per_game'] if r[0].split('__')[0] in HELD_SUITES)
    m9 = v9['main']['Held-out 98']
    p9 = v9['paired']
    vcols = v9['per_game_columns']['virgin']
    vrows = sorted([r[0], r[2], *r[3:]] for r in v9['per_game'] if r[0].startswith('virgin__'))
    # v9b default (weak-body refit off): one result file per game, recovered from the run machine
    r0 = {}
    for f in (B / 'patch_autonomous_inducer_v9b/cloud/backup/59455254/stopped-full/runs/r0').glob('*__inducer.json'):
        g = json.loads(f.read_text())['games'][0]
        r0[f.name.rsplit('__', 1)[0]] = [g['levels_completed'], int(g['win'])]
    return {'per_game': rows,
            'v9b_r0': {k: r0[k] for k, *_ in rows},
            'v9b_r0_public25': [sum(v[0] for k, v in r0.items() if k.startswith('public25')),
                                sum(v[1] for k, v in r0.items() if k.startswith('public25'))],
            'virgin': {'columns': ['total', *vcols], 'per_game': vrows,
                       'stored': {c: arm(v9['main']['Virgin'][a]) for c, a in
                                  (('vbare', 'bare'), ('vbare_v9', 'bare_v9'), ('vgen', 'general'), ('vv9', 'v9_full'))}},
            'v9_certificates': {k: v9['certificates'][k] for k in ('Held-out 98', 'Held-out 98 (bare floor)', 'Virgin',
                                                                   'Virgin (bare floor)')},
            'v9_scorecard': {g: v9['scorecard_seed0'][g] for g in ('Public-25', 'Held-out 98', 'Virgin')},
            # per-game G, seed 0, written by patch_autonomous_inducer_v9/export_g_per_game.py from the run rows
            'v9_scorecard_games': rd('patch_autonomous_inducer_v9/g_per_game_seed0.json'),
            'v9': {'columns': v9cols, 'per_game': v9rows,
                   'stored': {'bare': arm(m9['bare']), 'bare_v9': arm(m9['bare_v9']), 'v9': arm(m9['v9_full']),
                              'loo1': arm(m9['loo_op1']), 'loo2': arm(m9['loo_op2']), 'loo3': arm(m9['loo_op3']),
                              'L3': arm(m9['L3_op123'])},
                   'stored_paired': {'bare_v9': pv(p9['bare_v9 - bare | Held-out 98']),
                                     'loo1': pv(p9['v9_full - loo_op1 | Held-out 98']),
                                     'loo2': pv(p9['v9_full - loo_op2 | Held-out 98']),
                                     'loo3': pv(p9['v9_full - loo_op3 | Held-out 98']),
                                     'L3': pv(p9['v9_full - L3 | Held-out 98'])}},
            'certificates': {k: cert[k] for k in ('issued', 'goal', 'unlock', 'optimistic', 'false_goal',
                                                  'false_unlock', 'probes_planned')},
            'stored': {'Zero-Prior': arm(v8['main']['held-out 98']['general']), '+Op0 Trans': arm(v6['v6']),
                       '+Op1 Graph': arm(v7['v7_graph']), '+Op2 Click+Rew': arm(v8['main']['held-out 98']['v7_base']),
                       '+Op3 Clue CSP': arm(v8['main']['held-out 98']['v8']), 'Op4 alone': arm(ho['v8b'])},
            'stored_paired': {'+Op3 Clue CSP': pv(v8['paired']['v8-general held-out 98']),
                              'Op4 alone': pv(ho['v8b_vs_general']),
                              '+Op2 Click+Rew': pv(v8['paired']['v7_base-general held-out 98'])},
            'public25': arm(v8['main']['Public-25']['v8']) + [pv(v8['paired']['v8-general Public-25'])[3],
                                                              pv(v8['paired']['v8-general Public-25'])[4]]}


def paired_boot(diff, n=10000, seed=0):
    """Paired game bootstrap of the summed level difference, the receipts' own estimator."""
    import random
    rng = random.Random(seed)
    b = sorted(sum(diff[rng.randrange(len(diff))] for _ in diff) for _ in range(n))
    return [b[int(0.025 * n) - 1], b[int(0.975 * n) - 1]]


def component_ladder(src):
    """Recompute the operator ladder from the 98 per-game rows; +Op4 Full takes the better of +Op3 and the
    drift refit on every game. Returns the rungs after the frozen pre-exposure snapshot and the paired tests."""
    rows = src['per_game']
    assert len(rows) == 98
    col = {'Zero-Prior': 2, '+Op2 Click+Rew': 3, '+Op3 Clue CSP': 4, 'Op4 alone': 5}
    per = {k: [r[i] for r in rows] for k, i in col.items()}
    per['+Op4 Full'] = [max(r[4], r[5]) for r in rows]
    tot = [r[1] for r in rows]
    rung = {k: [sum(v), sum(x == t for x, t in zip(v, tot))] for k, v in per.items()}
    for k in ('Zero-Prior', '+Op2 Click+Rew', '+Op3 Clue CSP', 'Op4 alone'):
        assert rung[k] == src['stored'][k], (k, rung[k], src['stored'][k])
    test = {}
    for k in ('+Op2 Click+Rew', '+Op3 Clue CSP', 'Op4 alone', '+Op4 Full'):
        d = [a - b for a, b in zip(per[k], per['Zero-Prior'])]
        test[k] = [sum(d), *paired_boot(d), sum(x > 0 for x in d), sum(x < 0 for x in d)]
        if k in src['stored_paired']:
            assert test[k] == src['stored_paired'][k], (k, test[k], src['stored_paired'][k])
    d = [a - b for a, b in zip(per['+Op3 Clue CSP'], per['+Op2 Click+Rew'])]
    test['+Op3 over Op2'] = [sum(d), *paired_boot(d), sum(x > 0 for x in d), sum(x < 0 for x in d)]
    rung.update({k: src['stored'][k] for k in ('+Op0 Trans', '+Op1 Graph')})
    # single frozen v9 file on the same 98 games: Bare-Prior floor, then each operator left out of v9 full
    v9 = src['v9']
    rows9 = v9['per_game']
    assert len(rows9) == 98 and [r[0] for r in rows9] == sorted(r[0] for r in rows), 'v9 rows are not the 98 games'
    tot9 = dict((r[0], r[1]) for r in rows)
    per9 = {c: [r[1 + i] for r in rows9] for i, c in enumerate(v9['columns'])}
    arms9 = {c: [sum(v), sum(x == tot9[r[0]] for x, r in zip(v, rows9))] for c, v in per9.items()}
    for c in v9['columns']:
        assert arms9[c] == v9['stored'][c], (c, arms9[c], v9['stored'][c])
    pair9 = {}
    for c, (a, b) in {'bare_v9': ('bare_v9', 'bare'), 'loo1': ('v9', 'loo1'), 'loo2': ('v9', 'loo2'),
                      'loo3': ('v9', 'loo3'), 'L3': ('v9', 'L3')}.items():
        d = [x - y for x, y in zip(per9[a], per9[b])]
        pair9[c] = [sum(d), *paired_boot(d), sum(x > 0 for x in d), sum(x < 0 for x in d)]
        ref = v9["stored_paired"][c]
        assert [pair9[c][0], *pair9[c][3:]] == [ref[0], *ref[3:]], (c, pair9[c], ref)
    return {'rungs': rung, 'vs_zero_prior': test, 'total_levels': sum(tot), 'public25': src['public25'],
            'v9_arms': arms9, 'v9_paired': pair9, 'v9_stored_ci': {c: v[1:3] for c, v in v9['stored_paired'].items()}}
COHORTS = [('Public-25', 'public25'), ('Tufa-50', 'tufa50'), ('Turducken-49', 'turducken49'), ('web', 'web'),
           ('explainer-98', 'explainer98'), ('interactive', 'interactive'), ('Ouellette', 'ouellette')]
CHANNEL_KEY = ('C1 directional · C2 dir.+A5 · C3 click · C4 dir.+click',
               'C5 dir.+A5+click · C6 A7 · C7 other · C8 no adapter')


def channel_matrix(fx, cen, x0, ytop, cw, ch, size=FSS, label_w=0.0):
    """Environments per action channel: the 906-game total row, then the cohorts; cells shade by row share."""
    routes = [f'C{k}' for k in range(1, 9)]
    rows = [(f'all {cen["sources"]} games', {c: cen['routes'][c]['sources'] for c in routes}, INDIGO)]
    rows += [(lab, cen['cohort_routes'][key], NAVY) for lab, key in COHORTS]
    for k, c in enumerate(routes):
        say(fx, x0 + (k + 0.5) * cw, ytop + 0.03, c, ha='center', va='bottom', size=size, color=INDIGO[0], **BOLD)
    for r, (lab, vals, fam) in enumerate(rows):
        y = ytop - (r + 1) * ch - (0.04 if r else 0)
        tot = sum(vals.values())
        say(fx, x0 - 0.05, y + ch / 2, lab, ha='right', va='center', size=size, color=INK, **(BOLD if not r else {}))
        for k, c in enumerate(routes):
            v = vals.get(c, 0)
            s = v / tot
            fc = PAGE if not v else fam[4] if s < 0.12 else fam[3] if s < 0.25 else fam[2] if s < 0.45 else fam[1]
            fx.add_patch(Rectangle((x0 + k * cw + 0.006, y + 0.006), cw - 0.012, ch - 0.012, fc=fc,
                                   ec=FRAME if not v else INK, lw=0.4, zorder=3))
            if v:
                say(fx, x0 + (k + 0.5) * cw, y + ch / 2, str(v), ha='center', va='center', size=size,
                    color=PAGE if fc in (fam[1], fam[2]) else fam[0])
    return ytop - len(rows) * ch - 0.04


def figure_three(X, report):
    H = 5.00
    fig = plt.figure(figsize=(W, H))
    fx = canvas(fig)
    cen = X['census']

    # ---------------- a: the eight action channels of 906 environments, and six suites from baseline to planner
    panel(fig, fx, 0.04, 2.66, 7.12, 2.30, 'a', '8 channels × 8 suites')
    yb = channel_matrix(fx, cen, 1.06, 4.52, 0.245, 0.185)
    for k, line in enumerate(CHANNEL_KEY):
        say(fx, 0.14, yb - 0.10 - k * 0.125, line, ha='left', va='center', color=MUTED)
    led = J('a9_d4_and_reachability_audit')['ledger_9_suites']['table2_groups']['Mother+Web+Turducken']
    lad = component_ladder(heldout_sources())
    measured = [('Pre-exp', led['baseline_levels'], led['baseline_wins'])] + [
        (k, *lad['rungs'][k]) for k, _, _ in HELDOUT_LADDER[1:]]
    assert measured == HELDOUT_LADDER, measured
    op4 = lad['vs_zero_prior']['+Op4 Full']
    # Two scales are linked explicitly: suite coverage above, held-out operator steps below.
    labels = ['Public-25','Parametric-25','Tufa-50','Clone-118','Held-Out 98','Hidden-110']
    ends = ['183/183','172/172','232/232','551/552','148L / 655','25.70 LB']
    for i, (label,end) in enumerate(zip(labels,ends)):
        y=4.48-i*.155
        say(fx,4.29,y,label,ha='right',va='center',size=6.8)
        left,width=4.40,1.64
        prism(fx,left,y-.042,width,.084,NAVY[4],depth=.025)
        if i<4:
            _,total,base,full,_=SUITES[i]
            segments=[(0,base/total,AMBER),(base/total,full/total,NAVY)]
        elif i==4:
            segments=[(0,80/655,AMBER),(80/655,148/655,INDIGO)]
        else:
            segments=[(0,3.41/100,AMBER),(3.41/100,25.70/100,NAVY)]
        for lo,hi,fam in segments:
            prism(fx,left+lo*width,y-.042,(hi-lo)*width,.084,fam[2],depth=.025)
        for k in range(1,20):
            fx.plot([left+k*width/20]*2,[y-.034,y+.034],color='white',lw=.4,zorder=4.3)
        endpoint=segments[-1][1]
        dots(fx,[left+endpoint*width],[y],segments[-1][2][1],d=3.8,marker='D',z=5)
        say(fx,6.16,y,end,ha='left',va='center',size=6.8,**BOLD)
    for x,label,fam in [(3.40,'Baseline',AMBER),(4.33,'+OnlineInducer',INDIGO),(5.75,'Full C1..C8',NAVY)]:
        dots(fx,[x],[4.69],fam[2],d=3.4,marker='D')
        say(fx,x+.07,4.69,label,ha='left',va='center',size=6.8)
    # Dashed magnification leader from the measured 80..148 segment.
    held_y=4.48-4*.155
    x0=4.40+80/655*1.64; x1=4.40+148/655*1.64
    fx.plot([x0,x0,x1,x1],[held_y+.06,held_y+.075,held_y+.075,held_y+.06],color=INDIGO[1],lw=.7,zorder=6)
    fx.plot([x1,7.04,7.04,6.99],[held_y+.075,held_y+.075,3.45,3.45],color=INDIGO[1],lw=.65,ls=(0,(2,2)),zorder=2)
    vals=[80,109,113,129,146,148]
    xs=np.linspace(3.43,6.60,6); w=.37; floor=3.10; scale=.0025
    say(fx,3.25,3.49,'OnlineInducer',ha='left',va='center',size=6.8,color=INDIGO[0],**BOLD)
    for i,(x,v) in enumerate(zip(xs,vals)):
        prev=vals[i-1] if i else 0
        if prev:
            prism(fx,x,floor,w,prev*scale,NAVY[3],depth=.025)
        prism(fx,x,floor+prev*scale,w,(v-prev)*scale,AMBER[2] if i==0 else INDIGO[2],depth=.025)
        if i:
            fx.plot([xs[i-1]+w,x],[floor+prev*scale]*2,color=INDIGO[1],lw=.7,zorder=5)
        say(fx,x+w/2,floor+v*scale+.045,str(v),ha='center',va='bottom',size=6.8,**BOLD)
        say(fx,x+w/2,3.00,'Base' if i==0 else f'+Op{i-1}',ha='center',va='center',size=6.8)
    # LOO is an independent ablation track, not a subtraction from the cumulative staircase.
    say(fx,3.25,2.79,'LOO',ha='left',va='center',size=6.8,color=AMBER[0])
    for i,drop in enumerate([26,26,14,3],2):
        x=xs[i]+w/2
        fx.plot([x-.20,x+.20],[2.91]*2,color=INDIGO[3],lw=.7)
        fx.plot([x,x],[2.91,2.91-drop*.0015],color=AMBER[1],lw=1.8)
        say(fx,x,2.76,f'−{drop}',ha='center',va='center',size=6.8,color=AMBER[0])
    say(fx,6.785,3.23,'22W/0L',ha='center',va='center',size=6.8,color=INK)

    # ---------------- b: one solver step under 10 concurrent games: what each clock reports
    panel(fig, fx, 0.04, 0.04, 3.50, 2.56, 'b', 'Thread clocks · 10 concurrent games')
    bx = axes(fig, 0.16, 0.44, 3.44, 2.24, left=False, grid='x')
    bx.set_xscale('log')
    bx.set_xlim(0.012, 2.4)
    bx.set_ylim(3.75, -0.75)
    bx.set_yticks([])
    clocks = [('thread_time', 0.0263, NAVY, 'true CPU of the solver thread'),
              ('process_time', 0.2558, AMBER, '9.73×, all 10 threads summed'),
              ('monotonic', 0.2570, AMBER, '9.77×, waits on one GIL')]
    for i, (lab, v, fam, note) in enumerate(clocks):
        bx.add_patch(Rectangle((0.012, i - 0.19), v - 0.012, 0.38, fc=fam[3] if fam is NAVY else fam[4],
                               ec=fam[1], lw=0.5, zorder=3))
        say(bx, v, i, f'{v:.4f} s', ha='left', va='center', transform=nudge(bx, 4.0), color=fam[0], **BOLD)
        say(bx, 0.0132, i - 0.40, f'time.{lab}(): {note}', ha='left', va='bottom', color=fam[0] if fam is NAVY
            else INK)
    y3 = 3.10
    say(bx, 0.0132, y3 - 0.40, '1.2 s budget; assuming 2.5× slower vCPU: 25× less search', ha='left',
        va='bottom', color=INK)
    bx.add_patch(Rectangle((0.012, y3 - 0.21), 1.2 - 0.012, 0.42, fc='none', ec=AMBER[1], lw=0.5,
                           ls=(0, (2.0, 1.2)), zorder=2.9))
    bx.add_patch(Rectangle((0.012, y3 - 0.21), 1.2 / 25 - 0.012, 0.42, fc=AMBER[2], ec=AMBER[0], lw=0.5, zorder=3))
    say(bx, 1.2 / 25, y3, 'su15 level 5 stops at 219 pops', ha='left', va='center', transform=nudge(bx, 4.0),
        color=AMBER[0], **BOLD)
    bx.set_xticks([0.02, 0.1, 0.5, 2])
    bx.set_xticklabels(['0.02', '0.1', '0.5', '2'])
    bx.set_xlabel('seconds charged to one solver step', labelpad=1.5)

    # ---------------- c: every official hidden-set receipt, in submission order, on the 20/11 lattice
    panel(fig, fx, 3.62, 0.04, 3.54, 2.56, 'c', 'Hidden-110 · 30 receipts')
    rc = X['receipts']
    cx_ = axes(fig, 3.98, 0.30, 7.06, 2.26, grid='y')
    n = len(rc)
    cx_.set_xlim(-1, n + 2.6)
    cx_.set_ylim(-2.2, 32)
    m2 = [r['score'] for r in rc if r['description'].startswith('Milestone 2')]
    mu, sd = float(np.mean(m2)), float(np.std(m2, ddof=1))
    cx_.axhspan(mu - sd, mu + sd, color=NAVY[4], lw=0, zorder=0.3, alpha=0.7)
    cx_.axhline(mu, color=NAVY[3], lw=0.6, zorder=0.4)
    for k in range(1, 11):
        cx_.plot([n + 0.7, n + 1.1], [25.70 - k * 20 / 11] * 2, color=INDIGO[2], lw=0.5, zorder=1)
    cx_.plot([n + 0.9, n + 0.9], [25.70 - 10 * 20 / 11, 25.70], color=INDIGO[2], lw=0.5, zorder=1)
    for i, r in enumerate(rc):
        d, s = r['description'], r['score']
        if r['status'] != 'COMPLETE':
            if r['status'] == 'ERROR':
                cx_.scatter([i], [-1.0], marker='x', s=12, c=GRAY[1], lw=0.7, zorder=4)
            continue
        if d.startswith('Milestone 2'):
            dots(cx_, [i], [s], NAVY[3], d=4.0, z=5)
        elif d.startswith('arc3-giants'):
            dots(cx_, [i], [s], NAVY[0] if s > 25 else AMBER[1], d=5.4, z=5.2)
        elif 'heliosli' in d:
            dots(cx_, [i], [s], INDIGO[2], d=4.4, z=5)
        elif '14K' in d or 'D-prime' in d or 'AutoNav' in d:
            dots(cx_, [i], [s], AMBER[1], d=5.4, z=5.1, marker='D')
        else:
            dots(cx_, [i], [s], AMBER[3], d=4.0, z=5)
    idx = {r['ref']: i for i, r in enumerate(rc)}
    inv = X['inversion']
    for key, below in (('B_truncation_1', False), ('B_truncation_2', True), ('C_openloop_hijack_clock_bundle', True)):
        v = inv[key]
        i = idx[v['ref']]
        txt = f"{v['delta']:+.2f}".replace('-', '−') + f" = {v['full_game_equivalents']:.1f} game-eq".replace('-', '−')
        if below:
            say(cx_, i, 25.70 + v['delta'], txt, ha='right', va='top', transform=nudge(cx_, 2.0, -4.5),
                color=AMBER[0], **BOLD)
        else:
            say(cx_, i, 25.70 + v['delta'], txt, ha='right', va='center', transform=nudge(cx_, -4.5),
                color=AMBER[0], **BOLD)
    best = max(m2)
    say(cx_, 0.0, 18.7, f'{best:.2f} best M2 read', ha='left', va='center',
        color=NAVY[0], **BOLD)
    say(cx_, n + 0.55, 25.70 - 5.5 * 20 / 11, 'full-game equivalent = 20/11', ha='right', va='center', color=INDIGO[0])
    say(cx_, 0.0, 30.6, f'M2 base, 12 repeats: {mu:.2f} ± {sd:.2f}', ha='left', va='center', color=NAVY[1])
    say(cx_, 0.0, 6.4, 'Duck V2 / V5 forks, best 3.41', ha='left', va='center', color=AMBER[1])
    say(cx_, n + 0.5, 30.6, '25.70 zero-probe', ha='right', va='center', color=NAVY[0], **BOLD)
    latest = next(r for r in rc if r['ref'] == 57894564)
    assert latest['status'] == 'COMPLETE' and latest['score'] == 21.08, latest
    say(cx_, n + 0.5, 12.7, f"#{latest['ref']}: {latest['score']:.2f}", ha='right', va='center',
        color=AMBER[0], **BOLD)
    cx_.set_yticks([0, 10, 20, 30])
    cx_.set_xticks([])
    cx_.set_xlabel('submission order (oldest to newest)', labelpad=1.5)
    cx_.set_ylabel('hidden LB', labelpad=2)
    save(fig, 'hazard_by_dataset', report)


# ======================================================================= Architecture (Section 4)
TEXT_W = 162 / 25.4   # the PDF text block, so this figure prints at its own size


def color_family(fc):
    from matplotlib.colors import to_rgb
    c=np.array(to_rgb(fc))
    return min((NAVY,INDIGO,AMBER),key=lambda fam:min(np.linalg.norm(c-np.array(to_rgb(v))) for v in fam))


def prism(ax,x,y,w,h,fc,lw=.85,depth=.03):
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=INK,lw=lw,zorder=3,gid='cel-prism'))


def cel_finish(fig,name):
    from matplotlib.text import Text
    from matplotlib.colors import to_rgb
    for t in fig.findobj(Text):
        c=to_rgb(t.get_color())
        if max(c)-min(c)<.20 and .25<min(c)<.97:
            t.set_color(MUTED)
    if name=='architecture':
        for ax in fig.axes:
            for p in ax.patches:
                if p.get_edgecolor()[3]:p.set_linewidth(min(p.get_linewidth(),.95))
            for line in ax.lines:line.set_linewidth(min(line.get_linewidth(),1.2))
        return
    for ax in fig.axes:
        for p in list(ax.patches):
            if isinstance(p,Rectangle) and p.get_gid()!='cel-prism' and (p.get_zorder()>=2.8 or (name=='regime' and abs(p.get_height()-.065)<1e-6)) and p.get_fill() and p.get_facecolor()[3]>0:
                p.set_edgecolor(INK);p.set_linewidth(.8)
        for line in ax.lines:
            if line.get_marker() not in (None,'None','none','',' '):
                line.set_markeredgecolor(INK);line.set_markeredgewidth(.45)
        for c in ax.collections:
            from matplotlib.collections import PathCollection
            if isinstance(c,PathCollection) and len(c.get_offsets()):
                c.set_edgecolor(INK);c.set_linewidth(.65)


def figure_architecture(X,report):
    from matplotlib.transforms import Affine2D
    fig=plt.figure(figsize=(TEXT_W,2.70)); fx=canvas(fig)
    # Four open physical stations, joined by a single flow. No text containers.
    for x,title in [(0.10,'1 Pixels'),(1.68,'2 Search'),(3.24,'3 Memory'),(4.83,'4 Gate')]:
        say(fx,x,2.56,title,size=8.0,ha='left',va='center',**BOLD)
    for x in [1.48,3.05,4.63]:
        arrow(fx,(x,1.68),(x+.15,1.68),color=INK,lw=1.4,head=.25)
    tr=J('xray_transition')
    f0,f1=np.array(X['specimen']['level_frames']['1']),np.array(tr['after'])
    assert np.array_equal(f0,np.array(X['specimen']['level_frames']['1']))
    diff_mask=(f0!=f1)
    rs,cs=np.where(diff_mask)
    r0,r1=max(0,int(rs.min())-6),min(64,int(rs.max())+7)
    c0,c1=max(0,int(cs.min())-6),min(64,int(cs.max())+7)
    delta=f1.copy()
    delta[r0:r1,c0:c0+2]=12;delta[r0:r1,c1-2:c1]=12
    delta[r0:r0+2,c0:c1]=12;delta[r1-2:r1,c0:c1]=12
    for i,(f,lab) in enumerate([(f0,r'$F_t$'),(f1,r'$F_{t+1}$'),(delta,r'$\Delta F_t$')]):
        x=.13+i*.38; y=1.76-i*.18; sz=.50
        trans=Affine2D().skew_deg(0,14)+fx.transData
        im=fx.imshow(f,cmap=ListedColormap(ARC),vmin=0,vmax=15,extent=(x,x+sz,y,y+sz),origin='upper',interpolation='nearest',transform=trans,zorder=3+i)
        fx.add_patch(Rectangle((x,y),sz,sz,fc='none',ec=INK,lw=.95,transform=trans,zorder=3.1+i))
        say(fx,x+.24,1.29,lab,ha='center',va='center')
    say(fx,.73,1.07,r'$\Delta F_t=F_{t+1}\oplus F_t$',ha='center',va='center')
    counts=[166,141,96,74,130,51,2,246]
    for i,n in enumerate(counts):
        cx=.25+(i%4)*.345; cy=.83-(i//4)*.51
        a=np.arange(6)*np.pi/3
        pts=np.c_[cx+.155*np.cos(a),cy+.155*np.sin(a)]
        fx.add_patch(Polygon(pts,fc=INDIGO[4],ec=INK,lw=.9,zorder=4))
        say(fx,cx,cy-.01,f'C{i+1}',ha='center',va='center',z=5)
        say(fx,cx,cy-.235,str(n),ha='center',va='center')
    # Hypothesis lattice: rejected paths terminate; the surviving path reaches the search root.
    say(fx,2.33,2.30,r'$\mathcal{H}_t$',ha='center',va='center',size=9)
    pts=[(1.81+i*.24,2.09) for i in range(5)]
    for i,(x,y) in enumerate(pts):
        fx.plot([x,2.29],[y,1.84],color=INDIGO[2] if i!=2 else NAVY[1],lw=1.3)
        fx.add_patch(Polygon([(x-.065,y),(x,y+.07),(x+.065,y),(x,y-.07)],fc=NAVY[4],ec=INK,lw=.9,zorder=4))
        if i!=2:
            fx.plot([x-.03,x+.03],[1.94,1.99],color=AMBER[1],lw=1.6,zorder=5)
    fx.add_patch(Polygon([(1.78,2.16),(2.84,2.16),(2.42,1.76),(2.21,1.76)],fc='none',ec=INK,lw=.95,zorder=2))
    say(fx,2.66,1.78,r'$\hat T$',ha='center',va='center',size=9)
    for x,y,par in [(2.06,1.47,2.3),(2.52,1.47,2.3),(1.83,1.20,2.06),(2.24,1.20,2.06),(2.72,1.20,2.52)]:
        fx.plot([par,x],[1.74 if y>1.3 else 1.47,y],color=NAVY[1],lw=1.3)
        fx.add_patch(plt.Circle((x,y),.055,fc=NAVY[2],ec=INK,lw=.9,zorder=4))
    say(fx,2.31,.99,'IW(1)/BFS/A*',ha='center',va='center')
    # The two timing rails share a linear duration scale, with ten concurrency divisions.
    for y,label,val,fam in [(.64,'thread_time 0.026s',.0263,NAVY),(.29,'process_time 0.256s',.2558,AMBER)]:
        say(fx,1.72,y+.13,label,ha='left',va='center',size=6.8)
        prism(fx,1.72,y-.03,1.19,.08,fam[4],depth=.025)
        prism(fx,1.72,y-.03,1.19*val/.257,.08,fam[2],depth=.025)
        for k in range(1,10):
            fx.plot([1.72+1.19*k/10]*2,[y-.026,y+.046],color='white',lw=.45,zorder=4)
    say(fx,2.31,.11,'10 threads · 9.73× GIL',ha='center',va='center',size=6.8)
    # Eight actual D4 transformations of the real specimen, on an octagonal lattice.
    cx,cy=3.91,1.91
    ang=np.arange(8)*np.pi/4
    ring=np.c_[cx+.40*np.cos(ang),cy+.40*np.sin(ang)]
    fx.add_patch(Polygon(ring,fc=INDIGO[4],ec=INK,lw=.95,zorder=2))
    for (x,y),(_,fn,_) in zip(ring,D4):
        fx.plot([cx+(x-cx)*.32,x],[cy+(y-cy)*.32,y],color=INDIGO[1],lw=1.2,zorder=2.2)
        frame_img(fx,fn(f0),x-.085,y-.085,.17,z=4)
    say(fx,cx,cy,r'$D_4$',ha='center',va='center',size=10)
    say(fx,cx,1.34,'L2 k×k ·79.6% Recall ·0/219 False',ha='center',va='center',size=6.8)
    for i,label in enumerate(['Op0:Trans','Op1:Graph','Op2:Switch','Op3:Pip-CSP','Op4:1px-Mask']):
        y=1.08-i*.205; x=3.37+(i%2)*.08
        prism(fx,x,y-.07,.26,.13,INDIGO[4],lw=.85,depth=.025)
        if i<4:
            fx.plot([3.53,3.53],[y-.08,y-.15],color=INK,lw=1.3,zorder=2)
        fx.add_patch(Rectangle((x+.09,y+.06),.07,.04,fc=INDIGO[2],ec=INK,lw=.65,zorder=4))
        say(fx,3.83,y,label,ha='left',va='center')
    # One-way certificate valve; upper chute and lower locked shield, aligned vertically on cx.
    cx,cy=5.52,1.51
    diamond=[(cx-.42,cy),(cx,cy+.30),(cx+.42,cy),(cx,cy-.30)]
    fx.add_patch(Polygon(diamond,fc=INDIGO[4],ec=INK,lw=.95,zorder=3))
    say(fx,cx,cy+.07,r'$\mathcal{C}(F_0)$',ha='center',va='center')
    say(fx,cx,cy-.09,r'$(\hat T,\tau,\hat G)$',ha='center',va='center')
    arrow(fx,(cx,1.83),(cx,2.08),color=NAVY[1],lw=1.5)
    prism(fx,4.96,2.11,1.12,.20,NAVY[4],depth=.04)
    say(fx,5.52,2.21,'auto_burst()',ha='center',va='center')
    say(fx,5.52,2.41,'Certified:0tok',ha='center',va='center')
    arrow(fx,(cx,1.19),(cx,.87),color=AMBER[1],lw=1.5)
    shield=[(5.05,.84),(5.99,.84),(5.95,.45),(5.52,.20),(5.09,.45)]
    fx.add_patch(Polygon(shield,fc=AMBER[4],ec=INK,lw=.95,zorder=3))
    say(fx,5.52,.68,'Uncertified:k=0',ha='center',va='center')
    say(fx,5.52,.47,'auto_dead_level',ha='center',va='center')
    fx.set_xlim(0,TEXT_W);fx.set_ylim(0,2.70);fx.set_aspect('auto')
    save(fig,'architecture',report)


# ======================================================================= Cover
def figure_cover(X, report):
    from generate_cover import render_cover
    render_cover(FIG, report)


def main():
    setup()
    X = J('x0_specimens')
    report = {}
    which = sys.argv[1:] or ['1', '2', '3', 'arch', 'cover']
    for k, fn in (('1', figure_one), ('2', figure_two), ('3', figure_three), ('arch', figure_architecture),
                  ('cover', figure_cover)):
        if k in which:
            fn(X, report)
    (OUT / 'xray_figcheck.json').write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        print(k, v['figcheck_hits'], 'hits;', v['green_px'], 'green px;', v['red_px'], 'red px;', v['pdf_bytes'],
              'B pdf')


if __name__ == '__main__':
    main()
