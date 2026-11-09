"""Inspect the synchronized PDF, render its pages, and write a compile receipt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

if importlib.util.find_spec('fitz') is None:
    runtime = subprocess.check_output(['comp', 'paper-python'], text=True).strip()
    print(f'PDF inspection runtime: {runtime}', flush=True)
    os.execv(runtime, [runtime, __file__, *sys.argv[1:]])

import fitz
from PIL import Image, ImageDraw

from build_pdf import BUILD_STATE, PAPER, ROOT, STEM, TABLE_IDS, read_doc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


TITLE_LINES = ["Breaking Frontier Lock: Score Dilution,",
               "Certificate Gating, and Online Operator Induction"]


def page_title_lines(page):
    """Return the rendered lines set in the page's largest font."""
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
             for s in l["spans"] if s["text"].strip()]
    top = max(s["size"] for s in spans)
    lines = {}
    for s in spans:
        if s["size"] > top - 0.5:
            lines.setdefault(round(s["bbox"][1]), []).append(s)
    return [re.sub(r"\s+", " ", unicodedata.normalize("NFKC", "".join(
                s["text"] for s in sorted(v, key=lambda s: s["bbox"][0])))).strip()
            for _, v in sorted(lines.items())]


def source_shapes(path):
    tables = re.findall(r"(?:^\|[^\n]*\n)+", path.read_text() + "\n", re.M)
    return [[len([line for line in t.splitlines() if not re.fullmatch(r"[| :\-]+", line)]),
             len(t.splitlines()[0].strip("|").split("|"))] for t in tables]


def parsed_shapes(path):
    shapes = []
    for b in read_doc(path)["blocks"]:
        if b["t"] != "Table":
            continue
        c = b["c"]
        rows = c[3][1] + [r for body in c[4] for r in body[2] + body[3]]
        n = len(c[2])
        assert all(len(row[1]) == n for row in rows)
        assert all(cell[4] for row in rows for cell in row[1]), "Empty parsed table cell"
        shapes.append([len(rows), n])
    return shapes


def text_lines(page):
    """Rendered lines as (top, bottom, baseline, text), with super/subscripts merged into their line."""
    spans = sorted(((s["bbox"][1], s["bbox"][3], s["origin"][1], s["text"])
                    for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
                    for s in l["spans"] if s["text"].strip()), key=lambda s: s[2])
    lines = []
    for top, bottom, base, text in spans:
        # A row pitch is at least 9pt; script offsets stay under 5pt.
        if lines and base - lines[-1][2] < 5:
            t, b, y, x = lines[-1]
            lines[-1] = (min(t, top), max(b, bottom), y, x + " " + text)
        else:
            lines.append((top, bottom, base, text))
    return lines


def layout_geometry(doc, table_rows):
    """Line count of every main-text table row and the vertical fill of the references page."""
    tables = {}
    for label, head in (("1", "Contrast ("), ("2", "Suite")):
        page = next(p for p in doc if head in p.get_text())
        lines = text_lines(page)
        start = next(i for i, l in enumerate(lines) if l[3].startswith(head))
        end = next(i for i, l in enumerate(lines) if i > start and l[3].startswith(f"Table {label}."))
        tables[label] = {"page": page.number + 1, "rows": table_rows[label],
                         "rendered_lines": end - start,
                         "wrapped_lines": end - start - table_rows[label]}
    top, bottom = 21 * 72 / 25.4, doc[0].rect.height - 21 * 72 / 25.4
    ref_pages = [p.number + 1 for p in doc if "References" in [l[3].strip() for l in text_lines(p)]]
    page = doc[ref_pages[0] - 1]
    body = [l for l in text_lines(page) if l[0] >= top - 1 and l[1] <= bottom + 1]
    first = next(l for l in body if l[3].strip() == "References")
    following = doc[page.number + 1].get_text()
    refs = {"page": page.number + 1, "first_top_pt": round(first[0], 2), "last_bottom_pt": round(body[-1][1], 2),
            "last_line": body[-1][3], "text_height_pt": round(bottom - top, 2),
            "fill": round((body[-1][1] - first[0]) / (bottom - top), 4),
            "last_entry_on_page": "[40]" in page.get_text() and "[40]" not in following,
            # the next page opens, below its running head, with the appendix title
            "appendix_starts_next_page": "Technical Report Appendix" in following.split("\n")[2]}
    return {"tables": tables, "references": refs}


def replay_alignment(main_md, app_md, textext, fulltext):
    """Recompute the figure's cohort and compare it with both locked sources."""
    x = json.loads((ROOT / 'analysis/out/x0_specimens.json').read_text())
    datasets = ['kaggle_v2', 'kaggle_v3', 'kaggle_v4', 'kaggle_v5', 'kaggle_v7',
                'api_qwen38flash', 'tufa_qwen36_27b', 'api_gptoss120b']
    pairs = {(p['game'], p['i'], p['j'])
             for p in x['human_click_pairs'] + x['human_prefix_pairs']}
    reach, run_count, trace_hashes = {}, 0, {}
    for name in datasets:
        path = ROOT / 'evidence/traces' / (name + '.jsonl')
        trace_hashes[path.name] = sha(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            run_count += 1
            g = r['game'].split('-')[0]
            reach[g] = max(reach.get(g, 0), min(r['k'] + 1, r['L']))
    fp = x['fingerprints']
    baseline = sum(reach.get(g, 0) >= j for g, i, j in pairs)
    planner = sum(x['planner'][g]['levels'] >= j for g, i, j in pairs)
    observed = {**fp, 'human_files': x['human_files'], 'baseline_runs': run_count,
                'reuse_pairs': len(pairs), 'shared_click_pairs': len(x['human_click_pairs']),
                'shared_prefix_pairs': len(x['human_prefix_pairs']),
                'planner_target_level_reached': planner,
                'baseline_target_level_reached': baseline}
    fixed = {'games': 25, 'constant_grid_games': 18, 'click_levels': 137,
             'repeat_click_levels': 111, 'cross_level_pairs': 597, 'human_files': 342,
             'baseline_runs': 825, 'reuse_pairs': 378, 'shared_click_pairs': 284,
             'shared_prefix_pairs': 120, 'planner_target_level_reached': 378}
    assert all(observed[k] == v for k, v in fixed.items()), observed
    readings = ['72.0% (18/25)', '81.0% (111/137)', '284/597', '120/597',
                f'{planner}/{len(pairs)}', f'{baseline}/{len(pairs)}']
    targets = {p.name: p.read_text() for p in [main_md, app_md]}
    targets['latex'] = textext.replace(r'\%', '%')
    targets['pdf'] = fulltext
    figure = PAPER / 'figures/regime.pdf'
    with fitz.open(figure) as d:
        targets['regime.pdf'] = '\n'.join(p.get_text() for p in d)
    # PDF extraction may insert line breaks within a ratio or percentage.
    missing = {name: [s for s in readings if re.sub(r'\s+', '', s) not in
                      re.sub(r'\s+', '', text)] for name, text in targets.items()}
    return {'passed': not any(missing.values()), 'observed': observed,
            'required_readings': readings, 'missing_readings': missing,
            'fingerprint_sha256': sha(ROOT / 'analysis/out/x0_specimens.json'),
            'trace_sha256': trace_hashes, 'regime_pdf_sha256': sha(figure),
            'baseline_definition': 'Maximum min(k+1,L) across 825 runs; target level j reached.',
            'source_levels_zero_based_pairs_one_based': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=ROOT / '.ai/dispatch/pdf_receipt.json')
    parser.add_argument("--source-snapshot", type=Path, default=BUILD_STATE)
    parser.add_argument("--previous-pdf-sha256")
    args = parser.parse_args()
    out = args.receipt.parent
    qa = out / "pdf_qa"
    qa.mkdir(parents=True, exist_ok=True)
    pdf = PAPER / (STEM + ".pdf")
    tex = PAPER / (STEM + ".tex")
    log = PAPER / (STEM + ".log")
    aux = PAPER / (STEM + ".aux")
    main_md, app_md = PAPER / "frontier-lock.md", PAPER / "appendix.md"
    # TeX wraps its log at 79 bytes, which can split a UTF-8 character across lines.
    logtext, textext = log.read_bytes().decode('utf-8', 'replace'), tex.read_text()
    count_command = ["/opt/homebrew/bin/python3.13", str(ROOT / "analysis/wordcount.py"), str(main_md)]
    count = int(subprocess.check_output(count_command, text=True).strip())
    doc = fitz.open(pdf)
    for old_page in qa.glob("page-*.png"):
        if int(old_page.stem.split("-")[-1]) > len(doc):
            old_page.unlink()
    bounds = []
    canvas = Image.new("RGB", (1008, ((len(doc) + 2) // 3) * 500), "#d9d9d9")
    draw = ImageDraw.Draw(canvas)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        im.save(qa / f"page-{i + 1:02d}.png")
        im.thumbnail((324, 465))
        x, y = (i % 3) * 336 + 6, (i // 3) * 500 + 25
        canvas.paste(im, (x, y))
        draw.text((x, y - 18), f"Page {i + 1}", fill="black")
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    # microtype deliberately hangs terminal punctuation into
                    # the optical margin; the TeX box itself stays in bounds.
                    margin = 24 * 72 / 25.4
                    if x0 < margin - 4 or x1 > page.rect.width - margin + 4:
                        bounds.append({"page": i + 1, "text": span["text"], "bbox": span["bbox"]})
    canvas.save(qa / "contact.png")
    title_lines = page_title_lines(doc[0])
    fulltext = "\n".join(p.get_text() for p in doc)
    (qa / "extracted_text.txt").write_text(fulltext)
    checks = {
        "latex_fatal_errors": len(re.findall(r"Fatal error", logtext)),
        "latex_errors": len(re.findall(r"^!|LaTeX Error|Package .* Error", logtext, re.M)),
        "undefined_citations": len(re.findall(r"Citation .* undefined", logtext)),
        "undefined_references": len(re.findall(r"(?:Hyper )?[Rr]eference .* undefined|There were undefined references", logtext)),
        "overfull_hbox": len(re.findall(r"Overfull \\hbox", logtext)),
        "overfull_vbox": len(re.findall(r"Overfull \\vbox", logtext)),
        "underfull_hbox": len(re.findall(r"Underfull \\hbox", logtext)),
        "underfull_vbox": len(re.findall(r"Underfull \\vbox", logtext)),
        "missing_characters": len(re.findall(r"Missing character", logtext)),
        "rerun_warnings": len(re.findall(r"Rerun to get|Rerun to get outlines", logtext)),
        "text_spans_outside_horizontal_margins": len(bounds),
        # TeX interword glue between spans carries no space glyph.
        "title_line_mismatches": int([re.sub(r"\s", "", s) for s in title_lines] !=
                                     [re.sub(r"\s", "", s) for s in TITLE_LINES]),
    }
    source_tables = source_shapes(main_md) + source_shapes(app_md)
    parsed_tables = parsed_shapes(main_md) + parsed_shapes(app_md)
    assert source_tables == parsed_tables
    geometry = layout_geometry(doc, {"1": source_tables[0][0], "2": source_tables[1][0]})
    refs = geometry["references"]
    checks["main_table_wrapped_lines"] = sum(t["wrapped_lines"] for t in geometry["tables"].values())
    checks["references_page_layout_misses"] = int(not (refs["page"] == 5 and 0.88 <= refs["fill"] <= 0.96 and
                                                       refs["last_entry_on_page"] and refs["appendix_starts_next_page"]))
    checks["unresolved_question_marks"] = len(re.findall(r"\?\?", fulltext))
    checks["literal_cite_keys"] = len(re.findall(r"\\cite\{ref0\}|ref0\b", textext))
    assert len(re.findall(r"\\bibitem\{ref\d+\}", textext)) == 40
    assert len(re.findall(r"\\begin\{tabular[x*]\}", textext)) == len(TABLE_IDS) == 15
    assert len(re.findall(r"\\includegraphics\[", textext)) == 4
    assert 'figures/architecture.pdf' in textext
    assert len(re.findall(r"\\begin\{equation\}", textext)) == 3
    assert len(re.findall(r"\\bibcite\{ref\d+\}", aux.read_text())) == 40
    assert all(v == 0 for v in checks.values()), checks
    snapshot = json.loads(args.source_snapshot.read_text())
    initial_hashes = snapshot["source_sha256"]
    current_hashes = {p.name: sha(p) for p in [main_md, app_md]}
    assert current_hashes == initial_hashes, "Markdown source changed during compilation"
    assert sha(tex) == snapshot['tex_sha256'], 'LaTeX source changed after compilation'
    required_readings = ['72.0\\% (18/25)', '81.0\\% (111/137)', '48.6\\%', '284/597',
                         '91 optimal', '120/597', '240 directed', '378/378',
                         '316 entries', 'Turducken-49', '396 / 396 (49 / 49 WIN', '655/655(94/94*W)']
    missing_replay = [s for s in required_readings if re.sub(r"[\s{}]", "", s) not in re.sub(r"[\s{}]", "", textext.replace(r"\allowbreak", ""))]
    assert not missing_replay, f'Locked replay readings missing from LaTeX: {missing_replay}'
    # single frozen OnlineInducer file (cumulative + LOO, Bare-Prior, Virgin-120), certificate abort rate and
    # L2 local D4 rules, compared without spacing or TeX grouping
    squash = lambda s: re.sub(r"[\s{}]", "", s.replace(r"\allowbreak", "").replace(r"\hspace{0pt}", ""))
    operator_readings = ['148/655', '148/655,11W', '+68L', '21W/0L', '+66L[+36,+101]', '109/655(6WIN)', '+35L[+8,+66]',
                         '11W/5L', '0/89', '44/380=11.6\\%', '3/40=7.5\\%', '76/102(74.5\\%)', '31/76',
                         '27/27=100.0\\%', '150/183(82.0\\%)', '163/183=89.1\\%', 'K1b']
    missing_ops = [s for s in operator_readings if squash(s) not in squash(textext)]
    assert not missing_ops, f'Operator ladder or L2 readings missing from LaTeX: {missing_ops}'
    leaked = re.findall(r'OnlineInducer v[5-8]\b|\b(?:mo01|pb01|tc01|wc01|g038|ap01|g008)\b', textext)
    assert not leaked, f'Held-out iteration labels in LaTeX: {leaked}'
    channel_counts = [166, 141, 96, 74, 130, 51, 2, 246]
    assert all(f' & {count} (' in textext for count in channel_counts), 'Locked channel count missing'
    assert 12 <= len(doc) <= 20, f"Expected 12 to 20 pages, observed {len(doc)}"
    assert count <= 1500, count
    previous_hash = args.previous_pdf_sha256 or snapshot['pdf_sha256']
    assert snapshot['repeat_compile_identical'] and sha(pdf) == previous_hash, "Consecutive PDF builds differ"
    assert snapshot['figure_sha256'] == {p.name: sha(p) for p in (PAPER/'figures').glob('*.pdf')}, 'Figure changed after compilation'
    assert snapshot['figure_png_sha256'] == {p.name: sha(p) for p in (PAPER/'figures').glob('*.png')}, 'PNG changed after compilation'
    alignment = replay_alignment(main_md, app_md, textext, fulltext)
    receipt = {
        "state": "done" if alignment['passed'] else "failed",
        "reason": "complete" if alignment['passed'] else "replay_reading_mismatch",
        "passed": alignment['passed'], "runtime_healthy": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "compiled" if count <= 1500 else "compiled_word_limit_exceeded",
        "wordcount": {"command": count_command, "measured": count, "limit": 1500,
                      "compression_target": 1485, "passed": count <= 1500,
                      "disposition": "Source preserved byte-for-byte; no compression required."},
        "pdf": {"path": str(pdf), "pages": len(doc), "bytes": pdf.stat().st_size,
                "sha256": sha(pdf), "repeat_compile_identical": sha(pdf) == previous_hash},
        "tex": {"path": str(tex), "sha256": sha(tex)},
        "compile_log": {"path": str(log), "sha256": sha(log)},
        "checks": checks,
        "layout_geometry": geometry,
        "source_reading_alignment": alignment,
        "content": {"source_sha256": current_hashes,
                    "source_snapshot": str(args.source_snapshot),
                    "source_bytes_unchanged": current_hashes == initial_hashes,
                    "locked_replay_readings_present": required_readings,
                    "locked_channel_counts_present": channel_counts,
                    "operator_and_l2_readings_present": operator_readings,
                    "tables": TABLE_IDS, "source_table_shapes": source_tables,
                    "parsed_table_shapes_match": source_tables == parsed_tables,
                    "figures": ["hazard.png", "architecture.png", "regime.png", "hazard_by_dataset.png"],
                    "display_equations": 3, "bibliography_entries": 40,
                    "rendering_repairs": ["Normalize doubled mathematical command escapes while preserving cases row breaks.",
                                          "Escape currency dollars before parsing table cells.",
                                          "Render source numbered lists as lists.",
                                          "Keep appendix subsection headings attached to their tables."]},
        "visual_qa": {"rendered_pages": list(range(1, len(doc) + 1)),
                      "page_1_title_lines": title_lines,
                      "contact_sheet": str(qa / "contact.png"), "page_images": str(qa),
                      "optical_margin_protrusion_allowance_pt": 4,
                      "horizontal_margin_findings": bounds},
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "status": receipt["status"],
                      "wordcount": count, "pages": len(doc), "bytes": pdf.stat().st_size,
                      "checks": checks, "passed": receipt['passed'],
                      "replay_alignment": alignment,
                      "repeat_compile_identical": receipt["pdf"]["repeat_compile_identical"]}))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
