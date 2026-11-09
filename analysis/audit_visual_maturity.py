"""Layout and visual census of benchmark papers against our compiled PDF.

Measures, from the PDF alone: page geometry and line measure, the font scale, page-1 signposting, table
formatting and numeric alignment, figure density and in-figure type, callout boxes, and identifier splits.
Writes analysis/out/top_papers_visual_statistics.json. Released under MIT-0.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics as st
import subprocess
import sys
from collections import Counter
from pathlib import Path

if importlib.util.find_spec('fitz') is None:
    runtime = subprocess.check_output(['comp', 'paper-python'], text=True).strip()
    os.execv(runtime, [runtime, __file__, *sys.argv[1:]])

import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/out/top_papers_visual_statistics.json'
OURS = ROOT / 'paper/ARC_Prize_2026_Breaking_Frontier_Lock.pdf'
# Benchmark PDFs are third-party files kept outside the repository: SOURCES maps each key below to a local path.
SOURCES = ROOT / '.ai/blind/visual_benchmark_sources.json'
NAMES = {
    'H_iclr2026_oral_micro': 'ICLR 2026 Oral, micro-benchmarking',
    'N_vista_mit': 'VISTA, MIT (arXiv:2610.02200)',
    'K_trm_2025_winner': 'TRM, ARC Prize 2025 paper 1st (arXiv:2510.04871)',
    'F_kepler': 'Kepler (arXiv:2610.00834)',
    'BARC_2024_winner': 'BARC, ARC Prize 2024 paper 1st (arXiv:2411.02272)',
}
MONO = re.compile(r'mono|courier|typewriter|cmtt|sftt|lmtt|txtt|inconsolata|menlo|consol|cursor|beramono', re.I)
HEAD = re.compile(r'^(?:(?P<num>\d+(?:\.\d+)*)|(?P<let>[A-Z](?:\.\d+)*))\.?\s+[A-Z][A-Za-z]')
CAP = re.compile(r'^(Figure|Fig\.|Table)\s*([A-Z]?\d+[a-z]?)\s*[.:|]')
NUMERIC = re.compile(r'^[\s$#~<>≤≥+\-−–()\[\],.%×x/0-9±*LBWn=]+$')
ARCH = re.compile(r'overview|architecture|pipeline|framework|paradigm|workflow|illustration|schematic|system|loop',
                  re.I)
CALLOUT = re.compile(r'^(Theorem|Proposition|Lemma|Definition|Corollary|Takeaway|Finding|Remark|Observation|'
                     r'Key (?:insight|finding)|Hypothesis|Claim)\b')
MM = 25.4 / 72


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def r(x, n=2):
    return None if x is None else round(x, n)


def lines_of(page):
    out = []
    for b in page.get_text('dict', flags=fitz.TEXTFLAGS_TEXT)['blocks']:
        for ln in b.get('lines', []):
            spans = [s for s in ln['spans'] if s['text'].strip()]
            if not spans:
                continue
            text = ''.join(s['text'] for s in ln['spans']).strip()
            sizes = Counter()
            for s in spans:
                sizes[round(s['size'], 1)] += len(s['text'].strip())
            out.append({'text': text, 'bbox': ln['bbox'], 'spans': spans, 'size': sizes.most_common(1)[0][0],
                        'origin': spans[0]['origin'][1], 'block': id(b), 'dir': ln['dir']})
    return out


def is_bold(s):
    return bool(s['flags'] & 16) or bool(re.search(r'bold|black|semibold|\.B$|-B$|cmbx|bx\d', s['font'], re.I))


def fill_rects(page, min_area=0.0):
    rects = []
    for d in page.get_drawings():
        f = d.get('fill')
        if f is None or min(f) > 0.985:
            continue
        rr = d['rect']
        if rr.width * rr.height >= min_area:
            rects.append((rr, f))
    return rects


def hrules(page, min_w):
    out = []
    for d in page.get_drawings():
        for it in d['items']:
            if it[0] == 'l':
                a, b = it[1], it[2]
                if abs(a.y - b.y) < 0.6 and abs(a.x - b.x) >= min_w:
                    out.append((min(a.x, b.x), a.y, max(a.x, b.x), d.get('width') or 0.0))
            elif it[0] == 're':
                rr = it[1]
                if rr.height <= 2.6 and rr.width >= min_w and (d.get('fill') is not None and max(d['fill']) < 0.6):
                    out.append((rr.x0, (rr.y0 + rr.y1) / 2, rr.x1, rr.height))
    return out


def whiteness(page, clip):
    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), clip=clip, alpha=False, colorspace=fitz.csGRAY)
    s = pix.samples
    return sum(1 for v in s if v > 245) / max(1, len(s))


def audit(path):
    doc = fitz.open(path)
    W, H = doc[0].rect.width, doc[0].rect.height
    pages = [lines_of(p) for p in doc]
    chars = Counter()
    fonts = Counter()
    for pl in pages:
        for ln in pl:
            for s in ln['spans']:
                chars[round(s['size'], 1)] += len(s['text'].strip())
                fonts[(s['font'], round(s['size'], 1))] += len(s['text'].strip())
    body = chars.most_common(1)[0][0]
    body_font = Counter({f: n for (f, z), n in fonts.items() if abs(z - body) < 0.15}).most_common(1)[0][0]
    # references / appendix boundary: the first page whose heading line reads References or Bibliography
    ref_at = next(((i, ln['bbox'][1]) for i, pl in enumerate(pages) for ln in pl
                   if re.fullmatch(r'(?:\d+\.?\s*)?(References|REFERENCES|Bibliography)', ln['text'])
                   and ln['size'] >= body), (len(doc) - 1, H))
    ref_page = ref_at[0]

    # ---------------- 1 geometry and measure
    bl = [(i, ln) for i, pl in enumerate(pages) for ln in pl
          if abs(ln['size'] - body) < 0.3 and len(ln['text']) >= 25 and ln['dir'] == (1.0, 0.0)]
    right_col = [ln for _, ln in bl if ln['bbox'][0] > W * 0.45]
    two_col = len(right_col) > 0.25 * len(bl)
    left = [ln for _, ln in bl if ln['bbox'][0] < W * 0.45 and (not two_col or ln['bbox'][2] < W * 0.56)]
    x0 = q([ln['bbox'][0] for ln in left], 0.10)
    col_r = q([ln['bbox'][2] for ln in left], 0.90)
    blk_r = q([ln['bbox'][2] for ln in (right_col if two_col else left)], 0.90)
    col_w = col_r - x0
    full = [ln for ln in left if ln['bbox'][2] - ln['bbox'][0] >= 0.92 * col_w]
    cpl = [len(ln['text']) for ln in full]
    lead = []
    for pl in pages:
        prev = None
        for ln in pl:
            if abs(ln['size'] - body) < 0.3 and prev and prev['block'] == ln['block']:
                d = ln['origin'] - prev['origin']
                if 0.5 * body < d < 2.2 * body:
                    lead.append(d)
            prev = ln
    geometry = {
        'page_w_pt': r(W, 1), 'page_h_pt': r(H, 1), 'page_w_mm': r(W * MM, 1), 'page_h_mm': r(H * MM, 1),
        'columns': 2 if two_col else 1,
        'column_width_pt': r(col_w, 1), 'textblock_width_pt': r(blk_r - x0, 1),
        'textblock_page_ratio': r((blk_r - x0) / W, 3),
        'left_margin_mm': r(x0 * MM, 1), 'right_margin_mm': r((W - blk_r) * MM, 1),
        'chars_per_line_p50': r(q(cpl, 0.5), 0), 'chars_per_line_p90': r(q(cpl, 0.9), 0),
        'leading_pt': r(st.median(lead) if lead else None),
        'leading_ratio': r(st.median(lead) / body if lead else None, 3),
    }

    # ---------------- 2 type scale
    p1 = pages[0]
    tsize = max(ln['size'] for ln in p1)
    tspans = [s for ln in p1 if ln['size'] == tsize for s in ln['spans']]
    heads = {'H1': [], 'H2': []}
    head_style = {'H1': Counter(), 'H2': Counter()}
    for i, pl in enumerate(pages):
        # TeX sets the section number and its title as separate lines on one baseline: merge them into rows
        rows = []
        for ln in sorted(pl, key=lambda l: (round(l['origin']), l['bbox'][0])):
            if rows and abs(rows[-1][-1]['origin'] - ln['origin']) < 1.5 and ln['bbox'][0] - rows[-1][-1]['bbox'][2] < 40:
                rows[-1].append(ln)
            else:
                rows.append([ln])
        for row in rows:
            text = ' '.join(l['text'] for l in row)
            spans = [s for l in row for s in l['spans']]
            size = max(l['size'] for l in row)
            m = HEAD.match(text)
            # the title words, not the (possibly smaller) number, carry the heading style
            s0 = row[-1]['spans'][0] if len(row) > 1 else spans[0]
            if not m or len(text) > 80 or size < body - 0.05 or row[0]['bbox'][0] > x0 + 0.5 * col_w:
                continue
            if not (is_bold(s0) or size > body + 0.4) or re.search(r'\d{3,}|[=<>%]', text):
                continue
            lab = m['num'] or m['let']
            lvl = 'H1' if '.' not in lab else 'H2'
            heads[lvl].append(size)
            words = re.sub(r'^[\dA-Z.]+\s+', '', text)
            sizes = {round(s['size'], 1) for s in spans if s['text'].strip() and not s['text'].strip()[0].isdigit()}
            case = 'upper' if words.isupper() and len(sizes) == 1 else (
                'small_caps' if words.isupper() and len(sizes) > 1 else 'mixed')
            head_style[lvl][(s0['font'], 'bold' if is_bold(s0) else 'regular', case)] += 1
    caps = []
    for i, pl in enumerate(pages):
        for ln in pl:
            m = CAP.match(ln['text'])
            if m:
                caps.append({'page': i + 1, 'kind': 'table' if m[1] == 'Table' else 'figure', 'label': m[2],
                             'size': ln['size'], 'bold_label': is_bold(ln['spans'][0]), 'text': ln['text'],
                             'bbox': ln['bbox'], 'main': (i, ln['bbox'][1]) < ref_at})
    cap_sizes = [c['size'] for c in caps]
    foot = [ln['size'] for i, pl in enumerate(pages) for ln in pl
            if ln['bbox'][1] > H * 0.86 and ln['size'] < body - 0.6 and not re.fullmatch(r'\d+', ln['text'])
            and len(ln['text']) > 15]
    mono = Counter({f: n for (f, z), n in fonts.items() if MONO.search(f)})
    scale = {
        'body_font': body_font, 'body_size_pt': body,
        'title_size_pt': tsize, 'title_font': tspans[0]['font'], 'title_bold': is_bold(tspans[0]),
        'title_ratio': r(tsize / body),
        'h1_size_pt': r(st.median(heads['H1'])) if heads['H1'] else None,
        'h2_size_pt': r(st.median(heads['H2'])) if heads['H2'] else None,
        'h1_ratio': r(st.median(heads['H1']) / body) if heads['H1'] else None,
        'h2_ratio': r(st.median(heads['H2']) / body) if heads['H2'] else None,
        'h1_style': [list(k) + [v] for k, v in head_style['H1'].most_common(2)],
        'h2_style': [list(k) + [v] for k, v in head_style['H2'].most_common(2)],
        'caption_size_pt': r(st.median(cap_sizes)) if cap_sizes else None,
        'caption_ratio': r(st.median(cap_sizes) / body) if cap_sizes else None,
        'caption_label_bold_share': r(sum(c['bold_label'] for c in caps) / len(caps)) if caps else None,
        'footnote_size_pt': r(st.median(foot)) if foot else None,
        'mono_fonts': [f for f, _ in mono.most_common(3)],
        'mono_char_share': r(sum(mono.values()) / sum(fonts.values()), 4),
    }

    # ---------------- 3 page one
    page = doc[0]
    tl = [ln for ln in p1 if ln['size'] == tsize]
    ty0, ty1 = min(ln['bbox'][1] for ln in tl), max(ln['bbox'][3] for ln in tl)
    rules = [h for h in hrules(page, W * 0.45) if h[1] < H * 0.5]
    above = [h for h in rules if h[1] < ty0 + 1]
    below = [h for h in rules if ty1 - 1 < h[1] < ty1 + 60]
    ab = next((ln for ln in p1 if re.fullmatch(r'(?i)abstract[.:]?', ln['text'].strip())
               or re.match(r'(?i)^abstract[.:—\-]\s', ln['text'])), None)
    abstract = {'found': ab is not None}
    if ab:
        abody = [ln for ln in p1 if ln['bbox'][1] > ab['bbox'][1] - 1 and ln is not ab and len(ln['text']) > 30
                 and abs(ln['size'] - p1[p1.index(ln)]['size']) < 0.01]
        # the abstract runs until the first section heading on page 1
        stop = next((ln['bbox'][1] for ln in p1 if ln['bbox'][1] > ab['bbox'][3] + 4 and HEAD.match(ln['text'])
                     and (is_bold(ln['spans'][0]) or ln['size'] > body + 0.4)), H)
        abody = [ln for ln in abody if ln['bbox'][1] < stop]
        if abody:
            ax0, ax1 = q([ln['bbox'][0] for ln in abody], 0.1), q([ln['bbox'][2] for ln in abody], 0.9)
            ref_w = col_w if two_col else (blk_r - x0)
            ref_x0 = x0 if (not two_col or ax0 < W * 0.45) else x0
            abox = fitz.Rect(ax0 - 2, ab['bbox'][1] - 2, ax1 + 2, stop)
            fills = [f for f, c in fill_rects(page) if f.intersects(abox) and f.width > 0.4 * (ax1 - ax0)]
            abstract.update({
                'heading_size_pt': ab['size'], 'heading_bold': is_bold(ab['spans'][0]),
                'heading_centered': abs((ab['bbox'][0] + ab['bbox'][2]) / 2 - (ax0 + ax1) / 2) < 12,
                'heading_case': 'upper' if ab['text'].strip().rstrip('.:').isupper() else 'mixed',
                'body_size_pt': st.median([ln['size'] for ln in abody]),
                'body_ratio': r(st.median([ln['size'] for ln in abody]) / body),
                'inset_left_pt': r(ax0 - ref_x0, 1), 'width_ratio': r((ax1 - ax0) / ref_w, 3),
                'background_or_box': bool(fills), 'lines': len(abody)})
    p1_vis = [fitz.Rect(i['bbox']) for i in page.get_image_info()
              if fitz.Rect(i['bbox']).get_area() > 0.04 * W * H]
    try:
        p1_vis += [c for c in page.cluster_drawings() if c.get_area() > 0.05 * W * H]
    except Exception:
        pass
    page_one = {'title_rules_above': len(above), 'title_rules_below': len(below),
                'title_rule_widths_pt': [r(h[3], 2) for h in above + below],
                'title_lines': len({round(ln['bbox'][1]) for ln in tl}),
                'abstract': abstract, 'page1_figure': bool(p1_vis),
                'columns': geometry['columns']}

    # ---------------- 4 tables
    tables = []
    for i, p in enumerate(doc):
        # booktabs tables: a stack of full-width horizontal rules (top, mid, bottom) next to a "Table N" caption
        tcaps = [c for c in caps if c['kind'] == 'table' and c['page'] == i + 1]
        if not tcaps:
            continue
        hr = sorted(hrules(p, 0.3 * col_w), key=lambda h: h[1])
        groups = []
        for h in hr:
            g = next((g for g in groups if abs(g[0][0] - h[0]) < 3 and abs(g[0][2] - h[2]) < 3
                      and h[1] - g[-1][1] < 260), None)
            g.append(h) if g else groups.append([h])
        found = []
        for g in groups:
            if len(g) < 2:
                continue
            bb = fitz.Rect(g[0][0], g[0][1], g[0][2], g[-1][1])
            if bb.height < 2 * body or not any(-60 < c['bbox'][1] - bb.y1 < 40 or -40 < bb.y0 - c['bbox'][3] < 60
                                               for c in tcaps):
                continue
            try:
                ts = p.find_tables(clip=bb + (-2, -1, 2, 1), horizontal_strategy='text',
                                   vertical_strategy='text').tables
            except Exception:
                ts = []
            if ts:
                found.append((max(ts, key=lambda t: t.row_count * t.col_count), bb))
        for t, bb in found:
            if t.row_count < 2 or t.col_count < 2:
                continue
            words = p.get_text('words', clip=bb)
            sizes = Counter()
            for b in p.get_text('dict', clip=bb, flags=fitz.TEXTFLAGS_TEXT)['blocks']:
                for ln in b.get('lines', []):
                    for s in ln['spans']:
                        if s['text'].strip() and bb.contains(fitz.Rect(s['bbox'])):
                            sizes[round(s['size'], 1)] += len(s['text'].strip())
            if not sizes:
                continue
            tsz = sizes.most_common(1)[0][0]
            rl = [h for h in hrules(p, bb.width * 0.7) if bb.y0 - 4 <= h[1] <= bb.y1 + 4]
            fills = [f for f, c in fill_rects(p) if bb.intersects(f) and f.width > 0.5 * bb.width and f.height < 40]
            # alignment of numeric columns: the edge with the smallest spread across numeric cells
            align = Counter()
            for ci in range(t.col_count):
                l_, r_, c_ = [], [], []
                for row in t.rows[1:]:
                    cell = row.cells[ci] if ci < len(row.cells) else None
                    if not cell:
                        continue
                    cw = [w for w in words if fitz.Rect(w[:4]).intersects(fitz.Rect(cell))]
                    txt = ' '.join(w[4] for w in cw)
                    if not cw or not NUMERIC.match(txt) or not re.search(r'\d', txt):
                        continue
                    a0, a1 = min(w[0] for w in cw), max(w[2] for w in cw)
                    l_.append(a0), r_.append(a1), c_.append((a0 + a1) / 2)
                if len(l_) >= 3:
                    spread = {'left': st.pstdev(l_), 'right': st.pstdev(r_), 'center': st.pstdev(c_)}
                    best = min(spread, key=spread.get)
                    if max(spread.values()) < 0.6:
                        best = 'uniform'
                    align[best] += 1
            tables.append({'page': i + 1, 'rows': t.row_count, 'cols': t.col_count, 'font_pt': tsz,
                           'font_ratio': r(tsz / body), 'rules': len(rl),
                           'rule_widths_pt': sorted({r(h[3], 2) for h in rl}), 'row_fills': len(fills),
                           'numeric_col_alignment': dict(align),
                           'row_pitch_pt': r((bb.height / t.row_count), 1),
                           'row_pitch_ratio': r(bb.height / t.row_count / tsz, 2)})

    # ---------------- 5 figures
    figs = [c for c in caps if c['kind'] == 'figure']
    labels = []
    for c in figs:
        if c['label'] not in [x['label'] for x in labels]:
            labels.append(c)
    main_figs = [c for c in labels if c['main']]
    fig_min, fig_p10, ws = [], [], []
    for i, p in enumerate(doc):
        regions = [fitz.Rect(x['bbox']) for x in p.get_image_info() if fitz.Rect(x['bbox']).get_area() > 0.02 * W * H]
        try:
            regions += [c for c in p.cluster_drawings() if c.get_area() > 0.03 * W * H]
        except Exception:
            pass
        capr = [fitz.Rect(c['bbox']) for c in caps if c['page'] == i + 1]
        for rg in regions:
            sz = [ln['size'] for ln in pages[i] if rg.contains(fitz.Rect(ln['bbox'])) and ln['size'] >= 3.0
                  and not any(fitz.Rect(ln['bbox']).intersects(c) for c in capr)]
            if len(sz) >= 3:
                fig_min.append(min(sz))
                fig_p10.append(q(sz, 0.1))
        ws.append(whiteness(p, fitz.Rect(x0, H * 0.07, blk_r, H * 0.93)))
    panel_counts = []
    for c in main_figs:
        # the caption may wrap: read the whole paragraph block
        blk = doc[c['page'] - 1].get_text('text', clip=fitz.Rect(0, c['bbox'][1], W, min(H, c['bbox'][1] + 120)))
        letters = re.findall(r'\(([a-h])\)', blk[:900])
        panel_counts.append(max([ord(x) - 96 for x in letters], default=1))
    figures = {
        'figure_captions_total': len(labels), 'main_body_figures': len(main_figs),
        'appendix_figures': len(labels) - len(main_figs),
        'architecture_or_overview_figures': [c['label'] for c in labels if ARCH.search(c['text'])],
        'figure_1_page': next((c['page'] for c in labels if c['label'] == '1'), None),
        'subpanels_per_main_figure': panel_counts,
        'subpanels_median': st.median(panel_counts) if panel_counts else None,
        'in_figure_min_font_pt': r(st.median(fig_min)) if fig_min else None,
        'in_figure_p10_font_pt': r(st.median(fig_p10)) if fig_p10 else None,
        'whitespace_ratio_textblock_median': r(st.median(ws), 3),
        'table_captions_total': len({c['label'] for c in caps if c['kind'] == 'table'}),
    }

    # ---------------- 6 callouts and identifier splits
    boxes = 0
    bars = 0
    callout_lines = []
    for i, p in enumerate(doc):
        tb = [fitz.Rect(t.bbox) for t in []]
        for f, c in fill_rects(p):
            if f.width >= 0.5 * col_w and f.height > 1.6 * body and any(
                    f.contains(fitz.Rect(ln['bbox'])) and abs(ln['size'] - body) < 1.2 and len(ln['text']) > 30
                    for ln in pages[i]):
                boxes += 1
            if f.width <= 4 and f.height > 2.5 * body and x0 - 12 < f.x0 < x0 + 8:
                bars += 1
        for ln in pages[i]:
            if CALLOUT.match(ln['text']) and is_bold(ln['spans'][0]):
                callout_lines.append(re.match(r'\w+(?: \w+)?', ln['text'])[0])
    splits = 0
    hyph = 0
    for pl in pages:
        for a, b in zip(pl, pl[1:]):
            if b['block'] != a['block']:
                continue
            sa, sb = a['spans'][-1], b['spans'][0]
            if MONO.search(sa['font']) and MONO.search(sb['font']) and re.search(r'[A-Za-z0-9_]$', a['text']) \
                    and re.match(r'[A-Za-z0-9_]', b['text']):
                splits += 1
            if re.search(r'[a-z]-$', a['text']) and re.match(r'[a-z]', b['text']):
                hyph += 1
    callouts = {'tinted_text_boxes': boxes, 'left_bar_rules': bars,
                'bold_callout_leads': dict(Counter(re.sub(r' \d.*', '', x) for x in callout_lines)),
                'mono_identifier_line_splits': splits, 'hyphenated_line_breaks': hyph,
                'hyphen_per_page': r(hyph / len(doc), 2)}
    return {'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'pages': len(doc), 'main_pages_before_references': ref_page + 1,
            'geometry': geometry, 'type_scale': scale, 'page_one': page_one,
            'tables': {'count': len(tables), 'font_pt_median': r(st.median([t['font_pt'] for t in tables]))
                       if tables else None,
                       'font_ratio_median': r(st.median([t['font_ratio'] for t in tables])) if tables else None,
                       'row_pitch_ratio_median': r(st.median([t['row_pitch_ratio'] for t in tables]))
                       if tables else None,
                       'tables_with_row_fills': sum(t['row_fills'] > 0 for t in tables),
                       'numeric_col_alignment': dict(sum((Counter(t['numeric_col_alignment']) for t in tables),
                                                         Counter())),
                       'rule_widths_pt': sorted({w for t in tables for w in t['rule_widths_pt']}),
                       'detail': tables},
            'figures': figures, 'callouts': callouts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', type=Path, help='our PDF before the typography upgrade')
    ap.add_argument('--label', default='ours_current')
    args = ap.parse_args()
    old = json.loads(OUT.read_text())['papers'] if OUT.exists() else {}
    res = {'papers': {} if args.before or 'ours_before' not in old else {'ours_before': old['ours_before']}}
    paths = json.loads(SOURCES.read_text())
    for key, name in NAMES.items():
        path = Path(paths[key])
        assert path.is_file(), path
        res['papers'][key] = {'name': name, **audit(path)}
    if args.before:
        res['papers']['ours_before'] = {'name': 'Ours, before upgrade', **audit(args.before)}
    res['papers'][args.label] = {'name': 'Ours, current build', **audit(OURS)}
    bench = [v for k, v in res['papers'].items() if not k.startswith('ours')]
    sc = [v for v in bench if v['geometry']['columns'] == 1]
    res['single_column_benchmark_medians'] = {
        'textblock_width_pt': med(v['geometry']['textblock_width_pt'] for v in sc),
        'chars_per_line_p50': med(v['geometry']['chars_per_line_p50'] for v in sc),
        'leading_ratio': med(v['geometry']['leading_ratio'] for v in sc),
        'left_margin_mm': med(v['geometry']['left_margin_mm'] for v in sc),
        'h1_ratio': med(v['type_scale']['h1_ratio'] for v in sc),
        'caption_ratio': med(v['type_scale']['caption_ratio'] for v in sc),
        'table_font_ratio': med(v['tables']['font_ratio_median'] for v in sc),
        'main_body_figures': med(v['figures']['main_body_figures'] for v in bench),
        'in_figure_min_font_pt': st.median(v['figures']['in_figure_min_font_pt'] for v in bench
                                           if v['figures']['in_figure_min_font_pt']),
        'papers': [k for k, v in res['papers'].items() if v in sc],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    for k, v in res['papers'].items():
        g, s, t, f = v['geometry'], v['type_scale'], v['tables'], v['figures']
        print(f"{k:24s} {v['pages']:>2}p col={g['columns']} block={g['textblock_width_pt']}pt "
              f"L/R={g['left_margin_mm']}/{g['right_margin_mm']}mm cpl={g['chars_per_line_p50']}/"
              f"{g['chars_per_line_p90']} lead={g['leading_ratio']} body={s['body_size_pt']} "
              f"H1={s['h1_size_pt']} H2={s['h2_size_pt']} cap={s['caption_size_pt']} tab={t['font_pt_median']}"
              f"x{t['count']} align={t['numeric_col_alignment']} fills={t['tables_with_row_fills']} "
              f"figs={f['main_body_figures']}+{f['appendix_figures']} arch={f['architecture_or_overview_figures']} "
              f"minfig={f['in_figure_min_font_pt']} ws={f['whitespace_ratio_textblock_median']} "
              f"splits={v['callouts']['mono_identifier_line_splits']}")


if __name__ == '__main__':
    main()
