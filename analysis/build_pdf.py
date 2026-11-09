"""Synchronize the current Markdown manuscript and appendix into a LaTeX PDF."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STEM = "ARC_Prize_2026_Breaking_Frontier_Lock"
PANDOC = "/usr/local/bin/pandoc"
BUILD_STATE = ROOT / ".ai/dispatch/pdf_build.json"
LATEX = "/Library/TeX/texbin/pdflatex"


@lru_cache(maxsize=2048)
def pandoc(data, source="json", target="latex"):
    p = subprocess.run([PANDOC, "-f", source, "-t", target, "--wrap=none"],
                       input=data, text=True, capture_output=True, check=True)
    return p.stdout


def raw(text):
    return {"t": "RawBlock", "c": ["latex", text]}


def walk(node):
    if isinstance(node, dict):
        if node.get("t") == "Math":
            # The locked Markdown uses doubled TeX command escapes. Render the
            # intended commands while retaining the source files byte-for-byte.
            node["c"][1] = re.sub(r"\\\\(?=[A-Za-z{};,!])", lambda _: "\\", node["c"][1])
            if node["c"][0]["t"] == "InlineMath":
                for source, target in (
                    (r"s_l(a_l)=\min(115,100(b_l/a_l)^2)", r"s_l(a_l)=\allowbreak\min(115,\allowbreak 100(b_l/a_l)^2)"),
                    (r"G=\min(\sum_lw_ls_l,100\sum_{s_l>0}w_l)", r"G=\allowbreak\min(\sum_lw_ls_l,\allowbreak 100\sum_{s_l>0}w_l)"),
                ):
                    node["c"][1] = node["c"][1].replace(source, target)
        elif node.get("t") == "Code":
            # Identifiers set in the body face; only fenced code blocks keep a monospace font.
            node.update(t="Str", c=node["c"][1])
        elif node.get("t") == "Link" and node["c"][2][0] == "appendix.md":
            node["c"][2][0] = "#technical-appendix"
        for v in list(node.values()):
            walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)


def plain(node):
    if isinstance(node, dict):
        t = node.get("t")
        if t in {"Str", "Code", "Math"}:
            return node["c"] if t == "Str" else node["c"][1]
        if t in {"Space", "SoftBreak", "LineBreak"}:
            return " "
        if t == "Header":
            return plain(node["c"][2])
        return plain(node.get("c", []))
    if isinstance(node, list):
        return "".join(plain(v) for v in node)
    return ""


def outline(s):
    """Plain PDF-outline text: TeX commands and math markup removed."""
    s = re.sub(r"\\[A-Za-z]+", "", s.replace(" ", " "))
    s = re.sub(r"(?<=[A-Za-z])_\{?(\d+)\}?", r"\1", s)
    s = re.sub(r"[^A-Za-z0-9 .,:;()+/'&%_-]", "", s).replace("&", "and").replace("%", " percent")
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip()


ARCH_CAPTION = (r"Figure 2. \textbf{One-way certificate gating and neuro-symbolic verification.} "
                r"(a) Pixel transitions constrain eight action channels. "
                r"(b) Local \(D_4\) memory guards transfer across 8 board isometries with 0/219 false transfers; "
                r"OnlineInducer operators raise held-out clears from 80 to 148. "
                r"(c) Complete certificates authorize execution, uncertified levels receive 0 probes, "
                r"and solver deadlines run on thread CPU time.")


def read_doc(path):
    s = path.read_text()
    # Currency dollars must not capture later mathematical dollar delimiters or
    # the intervening table separators.
    s = re.sub(r"(?<!\\)(?<![A-Za-z}])\$(?=0 API\b|\d{1,3}(?:,\d{3})+(?:/run|-[\\]?[\$]))", lambda _: r"\$", s)
    s = re.sub(r"(?m)([^\n])\n(?=1\. )", r"\1\n\n", s)
    # Pandoc never closes $...$ right before a digit ("$\to$109L", "$\times$5").
    # Zero-width TeX glue preserves the locked source's visible text.
    s = re.sub(r"(\$\\(?:to|times)\$)(?=\d)", lambda m: m[1] + r"\hspace{0pt}", s)
    # Convert only explicit reference markers. Decimal confidence intervals are
    # unaffected. Currency remains literal text in the Markdown reader.
    # Code fences and inline code keep their array subscripts.
    cite = lambda m: r"\cite{" + ",".join("ref" + x.strip() for x in m[1].split(",")) + "}"
    s = "".join(part if k % 2 else re.sub(r"\[([1-9]\d*(?:,\s*[1-9]\d*)*)\]", cite, part)
                for k, part in enumerate(re.split(r"(```.*?```|`[^`\n]*`)", s, flags=re.S)))
    doc = json.loads(pandoc(s, "markdown+raw_tex", "json"))
    walk(doc)
    return doc


def writer(blocks, api):
    return pandoc(json.dumps({"pandoc-api-version": api, "meta": {}, "blocks": blocks}))


TABLE_IDS = ["1", "2", "A0", "A0b", "B", "A0c", "A1", "A2", "A3", "A4", "A5-A7", "G", "K1", "K1b", "K2"]
# Column weights of the appendix tables; the main-text Tables 1 and 2 take natural column widths.
TABLE_WEIGHTS = [
    None, None,
    [1.1, 1.2, .8, 1., .55, 1.35], [0.5, 2.3, .65, .8, .75],
    [1.1, 1.7, 1.1, .45, .65], [2.0, .65, .8, 1.05],
    [1.3, .45, .7, .7, .9, .95], [1.05, .45, .65, .7, 1.05, 1.05, 1.05],
    [1.6, .6, .55, .6, .6, .6, .6, .6, .95, .75, .95],
    [1.1, .55, .6, .6, .7, .85, 1.3, .85, 1.3],
    [1.05, 1., 1.25, 1.25, .85], [1.0, 1.2, .6, 2.2],
    [1.5, .45, .62, .85, 1.5, .53, 1.05], [1.55, .55, .75, 1.25, 1.45],
    [1.45, .8, .85, 1.2, .8, .9],
]
# Main-text tables: font size, baseline and minimum column gap. Every cell sets on one line; a row
# wider than the text block is an Overfull warning, never a silent wrap.
MAIN_TABLE_SIZE = {0: (7.6, 9.2, 1.2), 1: (7.6, 9.1, 2.7)}
REFERENCES = 40
BIB_SIZE = (8.6, 10.2, 1.2)  # font pt, baseline pt, item gap pt


def table_tex(table, index, api, caption, heading=None, prefix=None):
    c = table["c"]
    head = c[3][1]
    rows = [r for body in c[4] for r in body[2] + body[3]]
    n = len(c[2])
    weights = TABLE_WEIGHTS[index] or [1] * n
    weights = [w * n / sum(weights) for w in weights]
    assert len(weights) == n
    natural = index in MAIN_TABLE_SIZE
    # Markdown column alignment: numbers right (-:), centred (:-:), text left.
    align = [{"AlignRight": r"\raggedleft", "AlignCenter": r"\centering"}.get(spec[0]["t"], r"\RaggedCell")
             for spec in c[2]]
    # Packed cells such as "148L(11W,+68L[+38,+103],22W/0L,G=15.42)" may break after a comma; "1,200" and a
    # citation key list stay whole.
    # Inline math such as "\(T=+0.164 [+0.066, +0.252], p=0.006\)" may also break after its ", ".
    texts = [[re.sub(r"\\\(.*?\\\)", lambda m: m[0].replace(", ", r",\allowbreak "),
                     re.sub(r",(?=[^\d\s])(?!ref\d)", r",\\allowbreak{}", writer(cell[4], api).strip()))
              for cell in row[1]] for row in head + rows]
    assert all(len(t) == n for t in texts) and all(cell[2:4] == [1, 1] for row in head + rows for cell in row[1])
    body = texts[len(head):]
    # Our result is the bold entry: shade its rows, or its column when the whole column is bold.
    bold_rows = [i for i, t in enumerate(body) if any(r"\textbf{" in x for x in t)]
    bold_cols = [k for k in range(n) if sum(r"\textbf{" in t[k] for t in body) >= 0.8 * len(body)]
    few = len(bold_rows) <= max(2, len(body) // 3)
    shade_cols = set() if few else set(bold_cols)
    shade_rows = set(bold_rows) if few else set()
    # TeX rounds each weighted X column to scaled points independently. Flexible
    # intercolumn glue absorbs that sub-point residue in the target-width box.
    cols = r"@{\extracolsep{\fill}}" + "".join(
        r">{" + (r"\columncolor{OurRowBg}" if k in shade_cols else "") + (r"}l" if natural else r"\hsize=" +
        f"{w:.6f}" + r"\hsize\linewidth=\hsize" + align[k] + r"\arraybackslash}X") for k, w in enumerate(weights)) + "@{}"
    env = "tabular*" if natural else "tabularx"
    size = MAIN_TABLE_SIZE.get(index) or ((7.6, 9.1, 3.2) if index == 13 else
           ((7.2, 8.6, 3.4) if n >= 9 else ((7.6, 9.1, 3.4) if n >= 6 else (8.4, 10.0, 3.4))))
    # Main-text tables float so the text fills the page; appendix tables stay under their own headings.
    parts = [r"\begin{table}[!tbp]" if index < 2 else r"\begin{table}[H]"]
    if heading:
        parts.append(r"{\normalsize\bfseries " + heading + r"\par}\vspace{5pt}")
    if prefix:
        parts.append(prefix + r"\par\vspace{6pt}")
    parts += [r"\centering", r"\begingroup",
             rf"\fontsize{{{size[0]}}}{{{size[1]}}}\selectfont",
             rf"\setlength{{\tabcolsep}}{{{size[2]}pt}}",
             r"\renewcommand{\arraystretch}{0.90}" if TABLE_IDS[index] == "K1b" else
             (r"\renewcommand{\arraystretch}{1.10}" if index < 2 else r"\renewcommand{\arraystretch}{1.00}"),
             r"\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}" +
             (r"\setlength{\extrarowheight}{0.2pt}" if TABLE_IDS[index] == "K1b" else r"\setlength{\extrarowheight}{1.4pt}"),
             r"\begin{" + env + r"}{\linewidth}{" + cols + "}", r"\toprule[0.8pt]"]
    for i, t in enumerate(texts):
        j = i - len(head)
        parts.append((r"\rowcolor{OurRowBg}" if j in shade_rows else "") + " & ".join(t) + r" \\")
        if i == len(head) - 1:
            parts.append(r"\midrule[0.45pt]")
    parts.extend([r"\bottomrule[0.8pt]", r"\end{" + env + "}", r"\endgroup"])
    label = TABLE_IDS[index]
    text = None
    if caption:
        text = writer([caption], api).strip()
        text = re.sub(r"^\\emph\{(.*)\}$", r"\1", text, flags=re.S)
        if len(plain(caption)) <= 1500:
            parts.append(r"\caption*{" + text + "}")
            text = None
    parts.extend([r"\phantomsection\label{tab:" + label + "}", r"\end{table}"])
    if text:
        # A caption longer than the page cannot sit inside an unbreakable float: set it in caption type
        # directly below the table, where it may continue on the next page.
        parts.append(r"{\fontsize{9.5}{11.0}\selectfont\raggedright\vspace{-4pt}" + text + r"\par}")
    return "\n".join(parts)


def tex_escape_code(s):
    return (s.replace("\\", r"\textbackslash{}")
             .replace("_", r"\_")
             .replace("%", r"\%")
             .replace("#", r"\#")
             .replace("&", r"\&"))


def bash_box_tex(text):
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 14 or lines[0] != "cd analysis":
        raise ValueError(f"Unexpected Section C bash block ({len(lines)} lines): {lines[:2]}")
    rows = []
    for idx, ln in enumerate(lines, 1):
        if "#" in ln:
            cmd_part, comment_part = ln.split("#", 1)
            cmd_str = tex_escape_code(cmd_part.strip())
            comment_str = tex_escape_code(comment_part.strip())
            comment_str = comment_str.replace("R(c)", r"\(R(c)\)")
            row_body = (r"\makebox[162pt][l]{\ttfamily " + cmd_str + r"}"
                        r"{\normalfont\itshape\textcolor{CmdComment}{\#\ " + comment_str + r"}}")
        else:
            cmd_str = tex_escape_code(ln.strip())
            row_body = r"{\ttfamily " + cmd_str + r"}{\normalfont\itshape\textcolor{CmdComment}{}}"
        rows.append(
            r"\makebox[13pt][r]{\normalfont\fontsize{7.4}{9.8}\selectfont\textcolor{AlgLineNo}{"
            + str(idx) + r"}}\hspace{6pt}" + row_body
        )
    return (r"\begin{mdframed}[style=cmdbox]" + "\n" +
            r"{\fontsize{8.0}{9.8}\selectfont\raggedright" + "\n" +
            " \\\\\n".join(rows) + "\n" +
            r"\par}\end{mdframed}")


PROTOCOL_EXPECTED = [
    "inputs: games G, concurrency K, shared deadline H, stall cutoff c (80)",
    "unfinished <- |G|; waiting <- |G|",
    "when game g obtains a slot at time t:",
    "    waves <- max(1, ceil(unfinished / K))",
    "    deadline[g] <- min(H, t + max(0, H - t) / waves)",
    "    waiting <- waiting - 1",
    "deadline_for(g) = H if waiting = 0 else deadline[g]",
    "before every agent step of game g:",
    "    stop g if now >= deadline_for(g) or actions_on_current_level(g) >= c",
]


def protocol_tex(text):
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if lines != PROTOCOL_EXPECTED:
        raise ValueError(f"Section E protocol block mismatch: {lines}")
    rendered = [
        (1, r"\textbf{Input:} games \(G\), concurrency \(K\), shared deadline \(H\), stall cutoff \(c = 80\)"),
        (2, r"\(\text{unfinished} \leftarrow |G|;\quad \text{waiting} \leftarrow |G|\)"),
        (3, r"\textbf{when} game \(g\) obtains a slot at time \(t\)\textbf{:}"),
        (4, r"\hspace*{1.2em}\(\text{waves} \leftarrow \max\!\big(1,\, \lceil \text{unfinished} / K \rceil\big)\)"),
        (5, r"\hspace*{1.2em}\(\text{deadline}[g] \leftarrow \min\!\big(H,\, t + \max(0, H - t) / \text{waves}\big)\)"),
        (6, r"\hspace*{1.2em}\(\text{waiting} \leftarrow \text{waiting} - 1\)"),
        (7, r"\(\text{deadline\_for}(g) = H\) \textbf{if} \(\text{waiting} = 0\) \textbf{else} \(\text{deadline}[g]\)"),
        (8, r"\textbf{before} every agent step of game \(g\)\textbf{:}"),
        (9, r"\hspace*{1.2em}\textbf{stop} \(g\) \textbf{if} \(\text{now} \ge \text{deadline\_for}(g)\) \textbf{or} \(\text{actions\_on\_current\_level}(g) \ge c\)"),
    ]
    parts = [
        r"\par\vspace{3pt}\noindent\hspace*{4pt}\begin{minipage}{\dimexpr\linewidth-8pt\relax}",
        r"\fontsize{8.6}{10.3}\selectfont",
        r"\setlength{\tabcolsep}{0pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\setlength{\aboverulesep}{2pt}\setlength{\belowrulesep}{2.5pt}",
        r"\begin{tabular}{@{}r@{\hspace{7pt}}>{\RaggedCell\arraybackslash}p{\dimexpr\linewidth-21pt\relax}@{}}",
        r"\toprule[0.8pt]",
        r"\multicolumn{2}{@{}l@{}}{\rule{0pt}{9.5pt}\textbf{Protocol E.1:} Dynamic Stop-and-Transfer Water-Filling Schedule} \\[1pt]",
        r"\midrule[0.45pt]",
    ]
    for lno, code in rendered:
        parts.append(r"{\fontsize{7.5}{10.3}\selectfont\textcolor{AlgLineNo}{" + str(lno) + r"}} & " + code + r" \\")
    parts.extend([
        r"\bottomrule[0.8pt]",
        r"\end{tabular}",
        r"\end{minipage}\par\vspace{4pt}",
    ])
    return "\n".join(parts)


ALG1_EXPECTED_HEADING = "Algorithm 1: Unified Frozen OnlineInducer (Op0..Op4) with 1-Step Transition Monitor"
ALG1_EXPECTED_LINES = [
    "inputs: frame F_t in {0..15}^{64x64}, legal actions A_obs, level index l, base planner P",
    "persistent across levels: color classes C = (floor, wall, hazard, boring, effectful, goal),",
    "                          tile-graph rules R_graph, banned scenery vectors V_bg",
    "per-level state: body template B, action vectors {v_a}, visited board hashes H_seen,",
    "                 switch phases Phi, level-1 planner snapshot P_snap",
    "1. [Op4: 1px Scenery-Drift Mask & Small-Body Refit]",
    "   observe diff Delta_t = F_t != F_{t-1} under action a_{t-1}:",
    "   if |v_{a1}| <= 1 and v_{a1} == v_{a2} for two distinct keys a1 != a2:",
    "       add v_{a1} to V_bg; mask pixels shifting by v in V_bg and refit B",
    "   if |B| < 3 pixels and a directional step explains >= 2|B| pixels by shift v_a not in V_bg:",
    "       replace B and v_a with the larger rigid connected component",
    "2. [Op0: Novelty-Gated Translation & Competence Lock]",
    "   hash static-masked board h_t = Hash(F_t \\ DynamicPixels); increment H_seen[h_t]",
    "   if touching object c reaches a novel board state (H_seen[h_t] == 1): mark c in effectful",
    "   if H_seen[h_t] >= 6: count loop; if loops >= 16 or inducer_steps >= 224: demote level to P",
    "   if P cleared >= 2 prior levels alone (p_wins >= 2 and i_wins == 0): stand down to P",
    "3. [Op1: Relational Tile-Signature Graph & 1-Step Transition Monitor]",
    "   partition board into s x s cells S_t; if previous action a_{t-1} was from certificate tau:",
    "       if S_t != T_hat(S_{t-1}, a_{t-1}):",
    "           abort tau immediately at step L_mon = 1; delete refuted rule from R_graph",
    "   else if B and {v_a} are fitted:",
    "       run BFS over T_hat(p, a) = p + v_a avoiding wall/hazard cells in R_graph",
    "       to nearest untested or goal-unlock cell; if path tau found: return tau_1",
    "4. [Op2: Switch-Connectivity Induction & Level-1 Memory Rewind]",
    "   if click-only or 16 directional steps produce zero body displacement:",
    "       on level 1 frame 0, save P_snap <- deepcopy(P)",
    "       cycle each clickable switch s_i to record phase frames Phi(s_i) and 2px dilated contact graph",
    "       search phase assignment z maximizing connected components linked to spreading source color",
    "       if assignment z found: click switches to match z and commit via ACTION5",
    "       if level 1 exhausts click_gate = 224 without clearing:",
    "           restore P <- P_snap and issue 1 RESET (zero memory pollution to P)",
    "5. [Op3: Visual Pip-Clue Run/Count CSP]",
    "   on frame 0 of a clickable lattice with row/column border clue tiles:",
    "       infer cell pitch p = gcd(grid_spacings) and parse border clues (run lengths or pip counts)",
    "       solve exact line propagation + backtracking CSP for cell click mask M* in memory",
    "       verify each click against predicted cell flip; execute M* from frame 0",
]


def algorithm_tex(text, heading):
    if heading.strip() != ALG1_EXPECTED_HEADING:
        raise ValueError(f"Unexpected Algorithm 1 heading: {heading!r}")
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if lines != ALG1_EXPECTED_LINES:
        raise ValueError(f"Algorithm 1 source mismatch: {lines}")
    decls = [
        r"\textbf{Input:} frame \(F_t \in \{0,\dots,15\}^{64\times 64}\), legal actions \(\mathcal{A}_{\text{obs}}\), level index \(l\), base planner \(P\)",
        r"\textbf{Persistent state across levels:} color classes \(\mathcal{C} = (\text{floor}, \text{wall}, \text{hazard}, \text{boring}, \text{effectful}, \text{goal})\), tile-graph rules \(\mathcal{R}_{\text{graph}}\), banned scenery vectors \(V_{\text{bg}}\)",
        r"\textbf{Per-level state:} body template \(B\), action vectors \(\{\mathbf{v}_a\}\), visited board hashes \(\mathcal{H}_{\text{seen}}\), switch phases \(\Phi\), level-1 planner snapshot \(P_{\text{snap}}\)",
    ]
    items = [
        ("stage", r"Stage 1 ($\text{Op}_4$): 1px Scenery-Drift Mask \& Small-Body Refit"),
        (1, r"observe diff \(\Delta_t = (F_t \neq F_{t-1})\) under action \(a_{t-1}\)\textbf{:}"),
        (2, r"\textbf{if} \(|\mathbf{v}_{a_1}| \le 1\) \textbf{and} \(\mathbf{v}_{a_1} = \mathbf{v}_{a_2}\) for two distinct keys \(a_1 \neq a_2\) \textbf{then}\newline\hspace*{1.2em}add \(\mathbf{v}_{a_1}\) to \(V_{\text{bg}}\); mask pixels shifting by \(\mathbf{v} \in V_{\text{bg}}\) and refit \(B\)"),
        (3, r"\textbf{if} \(|B| < 3\) pixels \textbf{and} a directional step explains \(\ge 2|B|\) pixels by shift \(\mathbf{v}_a \notin V_{\text{bg}}\) \textbf{then}\newline\hspace*{1.2em}replace \(B\) and \(\mathbf{v}_a\) with the larger rigid connected component"),
        ("stage", r"Stage 2 ($\text{Op}_0$): Novelty-Gated Translation \& Competence Lock"),
        (4, r"hash static-masked board \(h_t = \operatorname{Hash}(F_t \setminus \text{DynamicPixels})\); increment \(\mathcal{H}_{\text{seen}}[h_t]\)"),
        (5, r"\textbf{if} touching object \(c\) reaches a novel board state (\(\mathcal{H}_{\text{seen}}[h_t] = 1\)) \textbf{then} mark \(c\) in \(\text{effectful}\)"),
        (6, r"\textbf{if} \(\mathcal{H}_{\text{seen}}[h_t] \ge 6\) \textbf{then} count loop; \textbf{if} \(\text{loops} \ge 16\) \textbf{or} \(\text{inducer\_steps} \ge 224\) \textbf{then} demote level to \(P\)"),
        (7, r"\textbf{if} \(P\) cleared \(\ge 2\) prior levels alone (\(\text{p\_wins} \ge 2\) \textbf{and} \(\text{i\_wins} = 0\)) \textbf{then} stand down to \(P\)"),
        ("stage", r"Stage 3 ($\text{Op}_1$): Relational Tile-Signature Graph \& 1-Step Transition Monitor"),
        (8, r"partition board into \(s \times s\) cells \(S_t\); \textbf{if} previous action \(a_{t-1}\) was from certificate \(\tau\) \textbf{then}"),
        (9, r"\hspace*{1.2em}\textbf{if} \(S_t \neq \hat{T}(S_{t-1}, a_{t-1})\) \textbf{then} abort \(\tau\) immediately at step \(L_{\text{mon}} = 1\); delete refuted rule from \(\mathcal{R}_{\text{graph}}\)"),
        (10, r"\textbf{else if} \(B\) and \(\{\mathbf{v}_a\}\) are fitted \textbf{then}"),
        (11, r"\hspace*{1.2em}run BFS over \(\hat{T}(p, a) = p + \mathbf{v}_a\) avoiding wall/hazard cells in \(\mathcal{R}_{\text{graph}}\) to nearest untested or goal-unlock cell"),
        (12, r"\hspace*{1.2em}\textbf{if} path \(\tau\) found \textbf{then} \textbf{return} \(\tau_1\)"),
        ("stage", r"Stage 4 ($\text{Op}_2$): Switch-Connectivity Induction \& Level-1 Memory Rewind"),
        (13, r"\textbf{if} click-only \textbf{or} \(16\) directional steps produce zero body displacement \textbf{then}"),
        (14, r"\hspace*{1.2em}on level \(1\) frame \(0\), save \(P_{\text{snap}} \leftarrow \operatorname{deepcopy}(P)\)"),
        (15, r"\hspace*{1.2em}cycle each clickable switch \(s_i\) to record phase frames \(\Phi(s_i)\) and \(2\text{px}\) dilated contact graph"),
        (16, r"\hspace*{1.2em}search phase assignment \(z\) maximizing connected components linked to spreading source color"),
        (17, r"\hspace*{1.2em}\textbf{if} assignment \(z\) found \textbf{then} click switches to match \(z\) and commit via \(\text{ACTION5}\)"),
        (18, r"\hspace*{1.2em}\textbf{if} level \(1\) exhausts \(\text{click\_gate} = 224\) without clearing \textbf{then}\newline\hspace*{2.4em}restore \(P \leftarrow P_{\text{snap}}\) and issue \(1\) \(\text{RESET}\) (zero memory pollution to \(P\))"),
        ("stage", r"Stage 5 ($\text{Op}_3$): Visual Pip-Clue Run/Count CSP"),
        (19, r"on frame \(0\) of a clickable lattice with row/column border clue tiles\textbf{:}"),
        (20, r"\hspace*{1.2em}infer cell pitch \(p = \gcd(\text{grid\_spacings})\) and parse border clues (run lengths or pip counts)"),
        (21, r"\hspace*{1.2em}solve exact line propagation + backtracking CSP for cell click mask \(M^\star\) in memory"),
        (22, r"\hspace*{1.2em}verify each click against predicted cell flip; execute \(M^\star\) from frame \(0\)"),
    ]
    parts = [
        r"\par\vspace{4pt}\noindent\hspace*{4pt}\begin{minipage}{\dimexpr\linewidth-8pt\relax}",
        r"\fontsize{8.6}{10.2}\selectfont",
        r"\setlength{\tabcolsep}{0pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\setlength{\aboverulesep}{1.8pt}\setlength{\belowrulesep}{2.0pt}",
        r"\begin{tabular}{@{}r@{\hspace{7pt}}>{\RaggedCell\arraybackslash}p{\dimexpr\linewidth-21pt\relax}@{}}",
        r"\toprule[0.8pt]",
        r"\multicolumn{2}{@{}l@{}}{\rule{0pt}{9.5pt}\textbf{Algorithm 1:} Unified Frozen OnlineInducer ($\text{Op}_0..\text{Op}_4$) with 1-Step Transition Monitor} \\[1pt]",
        r"\midrule[0.45pt]",
    ]
    for d in decls:
        parts.append(r"\multicolumn{2}{@{}>{\RaggedCell\arraybackslash}p{\linewidth}@{}}{" + d + r"} \\")
    parts.append(r"\midrule[0.3pt]")
    first_stage = True
    for tag, body in items:
        if tag == "stage":
            pad = r"\rule{0pt}{9.2pt}" if first_stage else r"\rule{0pt}{10.8pt}"
            first_stage = False
            parts.append(r"\multicolumn{2}{@{}l@{}}{" + pad + r"\textbf{" + body + r"}} \\")
        else:
            parts.append(r"{\fontsize{7.5}{10.2}\selectfont\textcolor{AlgLineNo}{" + str(tag) + r"}} & " + body + r" \\")
    parts.extend([
        r"\bottomrule[0.8pt]",
        r"\end{tabular}",
        r"\end{minipage}\par\vspace{4pt}",
    ])
    return "\n".join(parts)


PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\pdfinfoomitdate=1
\pdftrailerid{}
\usepackage[left=24mm,right=24mm,top=21mm,bottom=21mm,headheight=13pt,headsep=6mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{newtxtext,newtxmath}
\usepackage[scaled=0.88,varqu]{zi4}
\makeatletter\input{t1zi4.fd}\makeatother
\DeclareFontShape{T1}{zi4}{m}{it}{<->ssub*zi4/m/n}{}
\DeclareFontShape{T1}{zi4}{b}{it}{<->ssub*zi4/b/n}{}
\usepackage{microtype}
\frenchspacing
\usepackage{graphicx,booktabs,tabularx,array,caption,float,needspace}
\usepackage[table]{xcolor}
\usepackage{fancyhdr,fvextra,upquote}
\usepackage[framemethod=default]{mdframed}
\usepackage[unicode,hidelinks,breaklinks,bookmarksnumbered=false]{hyperref}
\usepackage{bookmark}
\definecolor{OurRowBg}{HTML}{EEF3F8}
\definecolor{AbsBg}{HTML}{F7F9FC}
\definecolor{AbsRule}{HTML}{8EA4BA}
\definecolor{CalloutBg}{HTML}{F8FAFC}
\definecolor{CalloutBar}{HTML}{1B4F72}
\definecolor{CmdBg}{HTML}{F8FAFC}
\definecolor{CmdRule}{HTML}{CBD5E1}
\definecolor{CmdComment}{HTML}{475569}
\definecolor{AlgLineNo}{HTML}{64748B}
\definecolor{AlgHeadBg}{HTML}{F4F7FA}
\mdfdefinestyle{abstract}{leftmargin=18pt,rightmargin=18pt,backgroundcolor=AbsBg,linecolor=AbsRule,linewidth=0.45pt,
  leftline=false,rightline=false,innertopmargin=7pt,innerbottommargin=8pt,innerleftmargin=11pt,innerrightmargin=11pt,
  skipabove=4pt,skipbelow=10pt}
\mdfdefinestyle{callout}{backgroundcolor=CalloutBg,linecolor=CalloutBar,linewidth=2pt,topline=false,bottomline=false,
  rightline=false,innertopmargin=5pt,innerbottommargin=6pt,innerleftmargin=9pt,innerrightmargin=9pt,
  skipabove=7pt,skipbelow=7pt}
\mdfdefinestyle{cmdbox}{leftmargin=4pt,rightmargin=4pt,backgroundcolor=CmdBg,linecolor=CmdRule,linewidth=0.45pt,
  innertopmargin=5pt,innerbottommargin=5.5pt,innerleftmargin=8pt,innerrightmargin=8pt,
  skipabove=5pt,skipbelow=6pt}
\makeatletter
\renewcommand\section{\@startsection{section}{1}{\z@}{-13pt plus -3pt minus -1.5pt}{5.5pt plus 1pt}%
  {\normalfont\large\bfseries\raggedright}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}{-10pt plus -2pt minus -1pt}{4pt plus 1pt}%
  {\normalfont\normalsize\bfseries\raggedright}}
\makeatother
\hypersetup{pdftitle={Breaking Frontier Lock: Score Dilution, Certificate Gating, and Online Operator Induction},pdfauthor={Yan Su},pdfsubject={ARC Prize 2026}}
\urlstyle{same}
\makeatletter
\g@addto@macro\UrlBreaks{\do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j\do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t\do\u\do\v\do\w\do\x\do\y\do\z\do\-\do\_}
\makeatother
\setlength{\parindent}{0pt}
\setlength{\parskip}{4.5pt plus 0.8pt minus 0.4pt}
\linespread{1.05}
\setlength{\emergencystretch}{2em}
\newcommand{\RaggedCell}{\raggedright\hspace{0pt}}
\newcommand{\tightlist}{\setlength{\itemsep}{1.5pt}\setlength{\parskip}{1.5pt}}
\renewcommand{\labelitemi}{\(\bullet\)}
\DefineVerbatimEnvironment{CodeBlock}{Verbatim}{breaklines=true,breakanywhere=true,fontsize=\footnotesize,baselinestretch=1.05}
\DeclareCaptionFont{capsize}{\fontsize{9.5}{11.6}\selectfont}
\captionsetup{font=capsize,labelfont=bf,skip=6pt,justification=raggedright,singlelinecheck=false}
\setcounter{topnumber}{4}\setcounter{bottomnumber}{3}\setcounter{totalnumber}{6}
\renewcommand{\topfraction}{0.95}\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.05}\renewcommand{\floatpagefraction}{0.7}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{9pt plus 1pt minus 1pt}
\setlength{\intextsep}{10pt plus 1pt minus 1pt}
% A page of floats stacks them from the top, as a text page would, and leaves its slack at the foot.
\makeatletter\setlength{\@fptop}{0pt}\setlength{\@fpsep}{14pt plus 2pt}\setlength{\@fpbot}{0pt plus 1fil}\makeatother
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\textit{Breaking Frontier Lock}}
\fancyhead[R]{\small ARC Prize 2026}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\raggedbottom
\begin{document}
"""


def build():
    main_path, app_path = PAPER / "frontier-lock.md", PAPER / "appendix.md"
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (main_path, app_path)}
    main, app = read_doc(main_path), read_doc(app_path)
    api = main["pandoc-api-version"]
    blocks = main["blocks"]
    title = writer([{"t": "Plain", "c": blocks[0]["c"][2]}], api).strip()
    # Break at the clause boundary: two balanced lines, no orphaned last word.
    assert title.count("Dilution, ") == 1, title
    title = title.replace("Dilution, ", "Dilution,\\\\")
    if plain(blocks[1]) == "Yan Su":
        author = writer([blocks[1]], api).strip()
        title_tex = (r"\thispagestyle{plain}\noindent\rule{\linewidth}{1.2pt}\par\vspace{9pt}\begin{center}" + "\n" +
                     r"{\LARGE\bfseries " + title + r"\par}\end{center}" + "\n" +
                     r"\vspace{-3pt}\noindent\rule{\linewidth}{0.5pt}\par\vspace{9pt}" + "\n" +
                     r"\begin{center}{\large " + author + r"\par}\end{center}\vspace{2pt}")
        body_start = 2
    else:
        subtitle = writer([blocks[1]], api).strip()
        author = writer([blocks[2]], api).strip()
        title_tex = (r"\thispagestyle{plain}\noindent\rule{\linewidth}{1.2pt}\par\vspace{9pt}\begin{center}" + "\n" +
                     r"{\LARGE\bfseries " + title + r"\par}" + "\n" +
                     r"\vspace{7pt}{\normalsize " + subtitle + r"\par}\end{center}" + "\n" +
                     r"\vspace{-3pt}\noindent\rule{\linewidth}{0.5pt}\par\vspace{9pt}" + "\n" +
                     r"\begin{center}{\large " + author + r"\par}\end{center}\vspace{2pt}")
        body_start = 3
    refs_index = next(i for i, b in enumerate(blocks) if b["t"] == "Header" and plain(b) == "References")
    refs = blocks[refs_index + 1]
    assert refs["t"] == "OrderedList" and len(refs["c"][1]) == REFERENCES
    allblocks = [raw(title_tex)] + blocks[body_start:refs_index]
    # The bibliography owns page 5; its size and item gap fill that page without spilling.
    bibliography = [r"\clearpage\phantomsection\pdfbookmark[0]{References}{bm-references}",
                    r"\begin{thebibliography}{" + str(REFERENCES) + "}",
                    rf"\fontsize{{{BIB_SIZE[0]}}}{{{BIB_SIZE[1]}}}\selectfont\raggedright\setlength{{\itemsep}}{{{BIB_SIZE[2]}pt}}"
                    r"\setlength{\parsep}{0pt}\setlength{\parskip}{0pt}"]
    for i, item in enumerate(refs["c"][1], 1):
        bibliography.append(r"\bibitem{ref" + str(i) + "} " + writer(item, api).strip())
    bibliography.append(r"\end{thebibliography}")
    allblocks.append(raw("\n".join(bibliography)))
    allblocks.append(raw(r"\clearpage\linespread{1.00}\fontsize{9.5}{11.4}\selectfont"
                        r"\setlength{\parskip}{3pt plus 0.6pt minus 0.3pt}"
                        r"\phantomsection\label{technical-appendix}\hypertarget{technical-appendix}{}"))
    allblocks.extend(app["blocks"])
    result, table_index, figure_index, equation_index = [], 0, 0, 0
    pending_heading = None
    pending_prefix = None
    in_appendix = arch_inserted = False
    i = 0
    while i < len(allblocks):
        b = allblocks[i]
        if b["t"] == "Header":
            lvl, attr, inlines = b["c"]
            text = writer([{"t": "Plain", "c": inlines}], api).strip()
            in_appendix = in_appendix or lvl == 1
            # PDF outline: paper sections and the appendix title at the top, appendix sections below it
            depth = 0 if (lvl == 1 or not in_appendix) else lvl - 1
            mark = r"\phantomsection\pdfbookmark[" + str(depth) + "]{" + outline(plain(b)) + "}{bm-" + \
                str(len(result)) + "}"
            if lvl == 2 and plain(b) == "Abstract":
                assert allblocks[i + 1]["t"] == "Para"
                result.append(raw(mark + r"\begin{mdframed}[style=abstract]" + "\n" +
                                  r"{\centering\normalsize\bfseries Abstract\par}\vspace{3.5pt}" + "\n" +
                                  r"\fontsize{9.2}{11.4}\selectfont " + writer([allblocks[i + 1]], api).strip() +
                                  r"\par\vspace{3pt}\noindent\textbf{Keywords:} Interactive Reasoning, Program Induction, Neuro-Symbolic World Models, Test-Time Verification, Equivariant Rule Induction, Survival Hazard Analysis." +
                                  "\n" + r"\end{mdframed}"))
                i += 2
                continue
            if lvl == 1:
                result.append(raw(r"\needspace{8\baselineskip}" + mark + r"\noindent\rule{\linewidth}{1.2pt}\par" +
                                  r"\vspace{8pt}{\centering\Large\bfseries " + text + r"\par}\vspace{6pt}" +
                                  r"\noindent\rule{\linewidth}{0.5pt}\par\vspace{4pt}"))
            elif lvl == 3 and i + 1 < len(allblocks) and allblocks[i+1]["t"] == "Table":
                pending_heading = mark + text
            else:
                cmd = "section" if lvl <= 2 else "subsection"
                need = "12" if plain(b).startswith("E. Dynamic Stop") else ("8" if plain(b).startswith("I. Multi-Stream") else "4")
                result.append(raw(r"\needspace{" + need + r"\baselineskip}" + mark + "\\" + cmd + "*{" + text + "}"))
            if plain(b).startswith("4. Breaking Frontier Lock"):
                # The Section 4 architecture, numbered as in the text; later figures shift by one.
                result.append(raw(r"\begin{figure}[!tbp]\centering" + "\n" +
                                  r"\includegraphics[width=\linewidth]{figures/architecture.pdf}" + "\n" +
                                  r"\caption*{" + ARCH_CAPTION + "}\n" +
                                  r"\phantomsection\label{fig:2}\end{figure}"))
                arch_inserted = True
        elif b["t"] == "Table":
            caption = None
            if i + 1 < len(allblocks) and allblocks[i+1]["t"] == "Para" and plain(allblocks[i+1]).startswith("Table "):
                caption = allblocks[i+1]
                i += 1
            result.append(raw(table_tex(b, table_index, api, caption, pending_heading, pending_prefix)))
            pending_heading = None
            pending_prefix = None
            table_index += 1
        elif b["t"] == "Figure" and "architecture." in b["c"][2][0]["c"][0]["c"][2][0]:
            i += 1
            continue
        elif b["t"] == "Figure":
            figure_index += 1
            image_block = b["c"][2][0]
            img = image_block["c"][0]
            assert img["t"] == "Image"
            src = img["c"][2][0]
            vector = Path(src).with_suffix('.pdf')
            assert (PAPER / vector).is_file(), vector
            src = vector.as_posix()
            caption = None
            number = str(figure_index + (figure_index > 1 and arch_inserted)) if figure_index < 3 else "A1"
            if i + 1 < len(allblocks) and allblocks[i+1]["t"] == "Para" and plain(allblocks[i+1]).startswith("Figure "):
                caption = writer([allblocks[i+1]], api).strip()
                caption = re.sub(r"^\\emph\{(.*)\}$", r"\1", caption, flags=re.S)
                if figure_index == 2:
                    assert caption.startswith("Figure 2."), caption
                    caption = "Figure " + number + caption[len("Figure 2"):]
                i += 1
            result.append(raw(r"\begin{figure}[!tbp]\centering" + "\n" +
                r"\includegraphics[width=\linewidth,height=0.47\textheight,keepaspectratio]{" + src + "}\n" +
                (r"\caption*{" + caption + "}\n" if caption else "") +
                r"\phantomsection\label{fig:" + number + r"}\end{figure}"))
        elif b["t"] == "Para" and plain(b).startswith("Engine telemetry on baseline serving variants") and i + 1 < len(allblocks) and allblocks[i+1]["t"] == "Table":
            pending_prefix = writer([b], api).strip()
        elif b["t"] == "Para" and plain(b).startswith("Algorithm 1:") and i + 1 < len(allblocks) and allblocks[i+1]["t"] == "CodeBlock":
            result.append(raw(algorithm_tex(allblocks[i+1]["c"][1], plain(b))))
            i += 2
            continue
        elif b["t"] == "Para" and b["c"] and b["c"][0]["t"] == "Strong" and \
                re.match(r"(Theorem|Proposition|Lemma|Corollary|Definition)\b", plain(b["c"][0])):
            statement = writer([b], api).strip()
            if plain(b["c"][0]).startswith("Proposition 1 ("):
                title = pandoc(json.dumps({"pandoc-api-version": api, "meta": {}, "blocks": [{"t": "Plain", "c": [b["c"][0]]}]})).strip()
                assert statement.startswith(title)
                statement = title + r" \itshape " + statement[len(title):].lstrip()
                # Line breaks only: preserve the locked formula and its inequalities.
                statement = statement.replace(r"\[\Delta(k)=", r"\[\begin{aligned}\Delta(k)&=")
                statement = statement.replace(r",\quad D(k)=", r",\\ D(k)&=")
                statement = statement.replace(r"-C_{115}(k)\le", r"-C_{115}(k)\\ &\le")
                statement = statement.replace(r"[1-r_k(A)],\]", r"[1-r_k(A)],\end{aligned}\]")
            result.append(raw(r"\begin{mdframed}[style=callout]" + "\n" + statement +
                              "\n" + r"\end{mdframed}"))
        elif b["t"] == "CodeBlock":
            classes = b["c"][0][1]
            code_text = b["c"][1]
            if "bash" in classes:
                result.append(raw(bash_box_tex(code_text)))
            elif code_text.lstrip().startswith("inputs: games G,"):
                result.append(raw(protocol_tex(code_text)))
            else:
                raise ValueError(f"Unrecognized CodeBlock in manuscript: {code_text[:80]!r}")
        elif b["t"] == "Para" and len(b["c"]) == 1 and b["c"][0]["t"] == "Math" and b["c"][0]["c"][0]["t"] == "DisplayMath":
            equation_index += 1
            result.append(raw(r"\begin{equation}" + "\n" + b["c"][0]["c"][1] +
                              "\n" + r"\label{eq:" + str(equation_index) + r"}\end{equation}"))
        else:
            result.append(b)
        i += 1
    assert table_index == len(TABLE_IDS) == 15 and figure_index == 3 and equation_index == 3 and arch_inserted
    body = writer(result, api)
    # Text-mode ± needs TS1 Times metrics, which this TeX install lacks.
    body = body.replace("±", r"\(\pm\)")
    # Protect explicit figure/table references without changing their text.
    body = re.sub(r"(?<![A-Za-z])Figure ([123])(?![0-9.])", lambda m: r"\hyperref[fig:" + m[1] + "]{Figure " + m[1] + "}", body)
    body = re.sub(r"(?<![A-Za-z])Table ([12])(?![0-9.])", lambda m: r"\hyperref[tab:" + m[1] + "]{Table " + m[1] + "}", body)
    out = PAPER / (STEM + ".tex")
    out.write_text(PREAMBLE + "\n" + body + "\n\\end{document}\n")
    command = [LATEX, '-recorder', '-interaction=nonstopmode', '-halt-on-error', out.name]
    pdf = out.with_suffix('.pdf')
    second_hash = None
    for pass_number in range(3):
        compiled = subprocess.run(command, cwd=PAPER, text=True, errors='replace', capture_output=True)
        if compiled.returncode:
            print(compiled.stdout)
            print(compiled.stderr)
            compiled.check_returncode()
        if pass_number == 1:
            second_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    final_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert final_hash == second_hash, 'Consecutive PDF builds differ'
    assert source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (main_path, app_path)}
    state = {"source_sha256": source_hashes, "pdf_sha256": final_hash,
             "tex_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
             "repeat_compile_identical": True, "compile_passes": 3,
             "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (PAPER/'figures').glob('*.pdf')},
             "figure_png_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (PAPER/'figures').glob('*.png')}}
    BUILD_STATE.parent.mkdir(parents=True, exist_ok=True)
    BUILD_STATE.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({"tex": str(out), "tables": table_index, "figures": figure_index,
                      "display_equations": equation_index, "references": REFERENCES,
                      "pdf_sha256": final_hash, "repeat_compile_identical": True,
                      "sources": source_hashes}))


if __name__ == "__main__":
    build()
