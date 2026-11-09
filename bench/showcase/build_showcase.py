#!/usr/bin/env python3
"""Build the downshift benchmark showcase from the stored result files.

Run from any directory:

    python bench/showcase/build_showcase.py

What it does:
  * It reads files in bench/ (CSV, JSON, one markdown table in docs/compatibility.md).
    It never changes them.
  * It writes HTML and markdown files into bench/showcase/ only.
  * It uses the Python standard library only.
  * It runs no benchmark. It shows the stored results.

Every number on a page comes from a data file. Some sentences depend on the data
(for example "the hand-rolled server is faster"). The function `claim` checks such a
sentence against the data. If the data change and the sentence is no longer true,
the build stops with an error.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
R1 = BENCH / "results" / "v1.0.0"
R5 = BENCH / "results" / "v0.5.0"
R6 = BENCH / "results" / "v0.6.0"
DOCS_COMPAT = REPO / "docs" / "compatibility.md"

REPO_URL = "https://github.com/uNRealCoder/downshift-server"


# ----------------------------------------------------------------------------------------
# Safety: write only inside bench/showcase/. Fail loudly when a source file is missing.
# ----------------------------------------------------------------------------------------


class SourceMissing(Exception):
    """A data file that a page needs does not exist."""


def write(name: str, text: str) -> None:
    path = (HERE / name).resolve()
    if HERE not in path.parents:
        raise RuntimeError(f"refusing to write outside bench/showcase: {path}")
    path.write_text(text, encoding="utf-8", newline="\n")


def need(path: Path) -> Path:
    if not path.is_file():
        raise SourceMissing(str(path.relative_to(REPO)))
    return path


@lru_cache(maxsize=None)
def jload(path: Path):
    return json.loads(need(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def cload(path: Path):
    with need(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


CLAIMS_CHECKED = 0


def claim(condition: bool, message: str) -> None:
    """Stop the build when a sentence on a page is no longer true for the data."""
    global CLAIMS_CHECKED
    CLAIMS_CHECKED += 1
    if not condition:
        raise AssertionError(f"claim does not hold for the data: {message}")


# ----------------------------------------------------------------------------------------
# Number formats
# ----------------------------------------------------------------------------------------


def n0(x: float) -> str:
    return f"{x:.0f}"


def n1(x: float) -> str:
    return f"{x:.1f}"


def n2(x: float) -> str:
    return f"{x:.2f}"


def sci(x: float) -> str:
    return f"{x:.1e}"


def pct(x: float) -> str:
    return f"{x:+.1f}%"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def rel(path: Path) -> str:
    """Path relative to the repo root, with forward slashes."""
    return path.relative_to(REPO).as_posix()


def href(path: Path) -> str:
    """Link from a page in bench/showcase/ to a file in the repo."""
    return "../" + path.relative_to(BENCH).as_posix() if BENCH in path.parents else "../../" + rel(path)


def srclink(path: Path) -> str:
    return f'<a href="{esc(href(path))}"><code>{esc(rel(path))}</code></a>'


# ----------------------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------------------

FLOAT_COLS = (
    "throughput_rps rows_per_s p50_ms p95_ms p99_ms mean_ms window_s ready_s max_abs_err "
    "health_p50_ms health_p99_ms client_ms ms"
).split()
INT_COLS = "batch workers concurrency errors request_bytes".split()


def numeric(rows):
    out = []
    for r in rows:
        r = dict(r)
        for c in FLOAT_COLS:
            if c in r:
                r[c] = float(r[c]) if r[c] != "" else None
        for c in INT_COLS:
            if c in r:
                r[c] = int(r[c]) if r[c] != "" else None
        out.append(r)
    return out


@lru_cache(maxsize=None)
def runs():
    return numeric(cload(R1 / "chart_runs.csv"))


@lru_cache(maxsize=None)
def stages():
    return numeric(cload(R1 / "chart_stages.csv"))


@lru_cache(maxsize=None)
def runs5():
    return numeric(cload(R5 / "chart_runs.csv"))


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def peak(rows):
    return max(rows, key=lambda r: r["throughput_rps"])


FIX_MODELS = ["clean_mlp", "tiny_bert", "mlp_large", "scatter_include_self_false"]
NAIVE_T, NAIVE_O, NAIVE_A = (
    "naive FastAPI + torch",
    "naive FastAPI + ONNX Runtime",
    "naive torch, async def",
)
DS_OLD, DS_OLD_ST = "downshift 0.5.1", "downshift 0.5.1 safetensors"
DS_NEW, DS_NEW_ST = "downshift 1.0.0rc1", "downshift 1.0.0rc1 safetensors"
DS_CACHE, DS_METRICS = "downshift 1.0.0rc1 result cache on", "downshift 1.0.0rc1 --metrics"
FIX_SERVERS = [NAIVE_T, NAIVE_O, NAIVE_A, DS_OLD, DS_OLD_ST, DS_NEW, DS_NEW_ST, DS_CACHE, DS_METRICS]
CORE5 = [NAIVE_T, NAIVE_O, NAIVE_A, DS_OLD, DS_NEW]
CONCS = [1, 2, 4, 8, 16, 32, 64]

SHORT = {
    NAIVE_T: "FastAPI + torch",
    NAIVE_O: "FastAPI + ORT",
    NAIVE_A: "torch, async def",
    DS_OLD: "downshift 0.5.1",
    DS_OLD_ST: "0.5.1 safetensors",
    DS_NEW: "downshift 1.0.0rc1",
    DS_NEW_ST: "1.0.0rc1 safetensors",
    DS_CACHE: "1.0.0rc1 cache on",
    DS_METRICS: "1.0.0rc1 --metrics",
    "naive FastAPI + transformers": "FastAPI + transformers",
    "naive transformers, async def": "transformers, async",
    "downshift 0.5.1 --backend torch": "0.5.1 --backend torch",
    "downshift 1.0.0rc1 --backend torch": "1.0.0rc1 --backend torch",
}


def bar_class(server: str) -> str:
    if server.startswith("naive"):
        return "bn"
    if server.startswith("downshift 0.5.1") or server.startswith("downshift 0.4.0"):
        return "bo"
    if "cache" in server or "--metrics" in server or "--backend torch" in server:
        return "bv"
    return "bw"


# One style for each server in the line charts: colour slot, marker shape, dash pattern.
LINE_STYLE = {
    NAIVE_T: (5, "circle", "hollow", "5 3"),
    NAIVE_O: (4, "square", "hollow", "2 3"),
    NAIVE_A: (6, "triangle", "hollow", "8 3 2 3"),
    DS_OLD: (2, "diamond", "solid", ""),
    DS_NEW: (1, "circle", "solid", ""),
}


# ----------------------------------------------------------------------------------------
# HTML building blocks
# ----------------------------------------------------------------------------------------

PALETTE_LIGHT = {
    "bg": "#f7f6f2",
    "card": "#ffffff",
    "text": "#1d1d1b",
    "muted": "#55554f",
    "line": "#d6d4ca",
    "soft": "#efede6",
    "accent": "#1f5fa8",
    "ok": "#1d6b3f",
    "warn": "#8a5a00",
    "bad": "#a12626",
    "s1": "#1f5fa8",
    "s2": "#b5500f",
    "s3": "#1f7a46",
    "s4": "#7a4fa0",
    "s5": "#5f5f58",
    "s6": "#8a6a00",
}
PALETTE_DARK = {
    "bg": "#14161a",
    "card": "#1c1f24",
    "text": "#e9e8e3",
    "muted": "#a9a9a1",
    "line": "#383c43",
    "soft": "#242830",
    "accent": "#7db3f0",
    "ok": "#6fcf9a",
    "warn": "#e0b45a",
    "bad": "#f08a8a",
    "s1": "#7db3f0",
    "s2": "#f0a070",
    "s3": "#6fcf9a",
    "s4": "#c3a0e6",
    "s5": "#b5b5ad",
    "s6": "#e0c060",
}


def _vars(p: dict) -> str:
    return ";".join(f"--{k}:{v}" for k, v in p.items())


CSS = (
    ":root{"
    + _vars(PALETTE_LIGHT)
    + "}\n@media (prefers-color-scheme: dark){:root:not([data-theme=\"light\"]){"
    + _vars(PALETTE_DARK)
    + "}}\n:root[data-theme=\"dark\"]{"
    + _vars(PALETTE_DARK)
    + "}\n"
    + """
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:0 16px 56px}
a{color:var(--accent)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em;background:var(--soft);padding:.05em .3em;border-radius:3px;overflow-wrap:anywhere}
pre{background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;font-size:.85em;line-height:1.45}
pre code{background:none;padding:0;overflow-wrap:normal}
nav.top{border-bottom:1px solid var(--line);background:var(--card)}
nav.top .wrap{padding-bottom:0}
nav.top ul{list-style:none;margin:0;padding:8px 0;display:flex;flex-wrap:wrap;gap:2px 14px;font-size:.88rem}
nav.top a{text-decoration:none}
nav.top a[aria-current="page"]{font-weight:700;text-decoration:underline}
header.page{padding:28px 0 8px}
h1{font-size:1.75rem;line-height:1.2;margin:0 0 8px}
h2{font-size:1.3rem;margin:34px 0 8px;line-height:1.25;border-top:1px solid var(--line);padding-top:18px}
h3{font-size:1.05rem;margin:22px 0 6px}
.lead{font-size:1.1rem;color:var(--text);margin:0 0 6px}
.meta{color:var(--muted);font-size:.88rem}
p,li{max-width:70ch}
ul,ol{padding-left:1.3em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile .v{font-size:1.45rem;font-weight:700;line-height:1.2;overflow-wrap:anywhere}
.tile .l{font-size:.88rem}
.tile .s{font-size:.78rem;color:var(--muted);margin-top:4px;overflow-wrap:anywhere}
.note{border:1px solid var(--line);border-left:4px solid var(--warn);background:var(--card);border-radius:6px;padding:8px 12px;margin:14px 0;font-size:.93rem}
.note.info{border-left-color:var(--accent)}
.note.bad{border-left-color:var(--bad)}
.note b.k{display:inline-block;margin-right:.4em}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}
.card h3{margin-top:0}
figure{margin:14px 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
figure svg{display:block;width:100%;max-width:560px;height:auto;margin:0 auto}
figcaption{font-size:.85rem;color:var(--muted);margin-top:6px}
.tw{overflow-x:auto;margin:10px 0;border:1px solid var(--line);border-radius:6px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.84rem}
caption{caption-side:top;text-align:left;font-weight:600;padding:8px 8px 4px}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
.chip{display:inline-block;padding:0 .5em;border-radius:4px;border:1px solid currentColor;font-size:.78rem;font-weight:700;letter-spacing:.02em;white-space:nowrap}
.chip.clean{color:var(--ok)}
.chip.degraded{color:var(--warn)}
.chip.failed{color:var(--bad)}
.chip.unverified{color:var(--s4)}
.legend{list-style:none;padding:0;margin:6px 0 0;display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.8rem}
.legend li{display:flex;align-items:center;gap:6px}
.legend svg{width:34px;height:12px;flex:none}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin:14px 0}
.cards a.pg{display:block;text-decoration:none;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.cards a.pg:hover,.cards a.pg:focus{border-color:var(--accent)}
.cards a.pg b{color:var(--accent);display:block}
.cards a.pg span{font-size:.88rem;color:var(--muted)}
footer{margin-top:40px;padding-top:12px;border-top:1px solid var(--line);font-size:.82rem;color:var(--muted)}
/* chart text */
svg text{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;fill:var(--text)}
svg .ax{font-size:10.5px;fill:var(--muted)}
svg .lb{font-size:11px}
svg .hd{font-size:11px;font-weight:700}
svg .vl{font-size:10.5px;font-variant-numeric:tabular-nums}
svg .gl{stroke:var(--line);stroke-width:1}
svg .axl{stroke:var(--muted);stroke-width:1}
svg .ref{stroke:var(--text);stroke-width:1;stroke-dasharray:4 3}
svg .box{fill:var(--card);stroke:var(--text);stroke-width:1.2}
svg .boxs{fill:var(--soft);stroke:var(--muted);stroke-width:1}
svg .arr{stroke:var(--text);stroke-width:1.4;fill:none}
svg .arrh{fill:var(--text)}
svg .bn{stroke:var(--s5);stroke-width:1}
svg .bo{fill:var(--s2)}
svg .bw{fill:var(--s1)}
svg .bv{fill:var(--s3)}
svg .hollow{fill:var(--card)}
svg .solid{fill:currentColor}
svg .mk{stroke:currentColor;stroke-width:1.6}
svg .ln{fill:none;stroke:currentColor;stroke-width:2}
svg .seg{stroke:var(--card);stroke-width:1}
.sw{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin-right:6px;border:1px solid var(--text)}
@media print{
  nav.top{display:none}
  body{background:#fff;color:#000;font-size:11pt}
  .wrap{max-width:none;padding:0}
  figure,.card,.tile,.tw{break-inside:avoid}
}
"""
)

PAGES = [
    ("index", "Overview"),
    ("throughput", "Throughput"),
    ("time-breakdown", "Time per request"),
    ("torch-vs-ort", "Torch vs ORT"),
    ("wire-formats", "Wire formats"),
    ("result-cache", "Result cache"),
    ("correctness", "Correctness"),
    ("compatibility", "Compatibility"),
    ("boot-time", "Boot time"),
    ("scaling", "Workers"),
    ("releases", "Releases"),
    ("SUMMARY", "One page"),
]


def frame(slug: str, title: str, body: str, *, lead: str = "") -> str:
    items = "".join(
        f'<li><a href="{s}.html"{" aria-current=\"page\"" if s == slug else ""}>{esc(t)}</a></li>'
        for s, t in PAGES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{esc(title)} | downshift benchmarks</title>
<style>
{CSS}</style>
</head>
<body>
<nav class="top" aria-label="Pages"><div class="wrap"><ul>{items}</ul></div></nav>
<main class="wrap">
{body}
<footer>
<p>Made by <code>bench/showcase/build_showcase.py</code> from files in <code>bench/</code>. The script changes no source file.
Project: <a href="{REPO_URL}">{REPO_URL}</a>.</p>
</footer>
</main>
</body>
</html>
"""


def header(title: str, question: str, meta: str = "") -> str:
    m = f'<p class="meta">{meta}</p>' if meta else ""
    return f'<header class="page"><h1>{esc(title)}</h1><p class="lead">{question}</p>{m}</header>'


def h2(text: str, anchor: str = "") -> str:
    a = f' id="{anchor}"' if anchor else ""
    return f"<h2{a}>{esc(text)}</h2>"


def h3(text: str) -> str:
    return f"<h3>{esc(text)}</h3>"


def p(text: str) -> str:
    return f"<p>{text}</p>"


def ul(items) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def ol(items) -> str:
    return "<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def note(kind: str, text: str, label: str = "") -> str:
    cls = {"limit": "", "info": " info", "bad": " bad"}[kind]
    lab = label or {"limit": "Limit.", "info": "Note.", "bad": "Warning."}[kind]
    return f'<div class="note{cls}"><b class="k">{esc(lab)}</b>{text}</div>'


def tiles(items) -> str:
    """items: (value, label, source note)"""
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div><div class="s">{s}</div></div>'
        for v, l, s in items
    ) + "</div>"


def chip(status: str) -> str:
    return f'<span class="chip {status.lower()}">{esc(status)}</span>'


def table(headers, rows, *, caption: str = "", right=(), raw=False) -> str:
    """Plain HTML table. `right` lists the column numbers to align right. Cells are escaped
    unless raw is true."""
    head = "".join(
        f'<th scope="col"{" class=\"r\"" if i in right else ""}>{esc(h)}</th>' for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = []
        for i, c in enumerate(row):
            txt = c if raw else esc(c)
            cells.append(f'<td{" class=\"r\"" if i in right else ""}>{txt}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="tw" tabindex="0" role="region" aria-label="{esc(caption or "table")}"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def figure(svg: str, caption: str, legend: str = "") -> str:
    return f"<figure>{svg}{legend}<figcaption>{caption}</figcaption></figure>"


# Collected for RESULTS.md: (section title, caption, source files, headers, rows)
MD_TABLES: list[tuple[str, str, list[Path], list, list]] = []


def mdt(section: str, caption: str, sources, headers, rows) -> None:
    MD_TABLES.append((section, caption, list(sources), list(headers), [list(map(str, r)) for r in rows]))


# ----------------------------------------------------------------------------------------
# SVG charts
# ----------------------------------------------------------------------------------------

_uid = [0]


def uid() -> int:
    _uid[0] += 1
    return _uid[0]


def nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * e * 1.0000001:
            return m * e
    return 10 * e


def marker(shape: str, x: float, y: float, fill: str, tip: str) -> str:
    s = 4.2
    if shape == "circle":
        g = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s}"/>'
    elif shape == "square":
        g = f'<rect x="{x - s:.1f}" y="{y - s:.1f}" width="{2 * s}" height="{2 * s}"/>'
    elif shape == "triangle":
        g = f'<path d="M{x:.1f} {y - s - 0.8:.1f} L{x + s + 0.8:.1f} {y + s:.1f} L{x - s - 0.8:.1f} {y + s:.1f} Z"/>'
    else:  # diamond
        g = f'<path d="M{x:.1f} {y - s - 1:.1f} L{x + s + 1:.1f} {y:.1f} L{x:.1f} {y + s + 1:.1f} L{x - s - 1:.1f} {y:.1f} Z"/>'
    return _mk(g, fill, tip)


def _mk(g: str, fill: str, tip: str) -> str:
    tag = g[1 : g.index(" ")]
    return g.replace("/>", f' class="mk {fill}"><title>{esc(tip)}</title></{tag}>')


def line_chart(cats, series, *, title, desc, ylabel, yfmt=n0, xlabel="concurrent clients") -> tuple[str, str]:
    """Categorical x axis. series: (name, values, style) with style = (slot, shape, fill, dash).
    Returns the svg and the HTML legend."""
    w, h = 400, 250
    ml, mr, mt, mb = 46, 14, 22, 44
    pw, ph = w - ml - mr, h - mt - mb
    vmax = nice_max(max(v for _, vals, _ in series for v in vals if v is not None))
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title><desc>{esc(desc)}</desc>']
    parts.append(f'<text class="ax" x="{ml}" y="12">{esc(ylabel)}</text>')
    for i in range(6):
        v = vmax * i / 5
        y = mt + ph - ph * i / 5
        parts.append(f'<line class="gl" x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ax" x="{ml - 5}" y="{y + 3.5:.1f}" text-anchor="end">{yfmt(v)}</text>')
    n = len(cats)
    xs = [ml + (pw * i / (n - 1) if n > 1 else pw / 2) for i in range(n)]
    for x, c in zip(xs, cats):
        parts.append(f'<text class="ax" x="{x:.1f}" y="{mt + ph + 15}" text-anchor="middle">{esc(c)}</text>')
    parts.append(f'<text class="ax" x="{ml + pw / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    parts.append(f'<line class="axl" x1="{ml}" x2="{w - mr}" y1="{mt + ph}" y2="{mt + ph}"/>')
    legend = ['<ul class="legend">']
    for name, vals, (slot, shape, fill, dash) in series:
        pts = [(x, mt + ph - ph * v / vmax, v, c) for x, v, c in zip(xs, vals, cats) if v is not None]
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y, _, _ in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<g style="color:var(--s{slot})"><path class="ln" d="{d}"{da}/>')
        for x, y, v, c in pts:
            parts.append(marker(shape, x, y, fill, f"{name}, {c}: {yfmt(v)}"))
        parts.append("</g>")
        sw = f'<svg viewBox="0 0 34 12" aria-hidden="true" style="color:var(--s{slot})"><path class="ln" d="M1 6 L33 6"{da}/>' + marker(shape, 17, 6, fill, name) + "</svg>"
        legend.append(f"<li>{sw}{esc(name)}</li>")
    parts.append("</svg>")
    legend.append("</ul>")
    return "".join(parts), "".join(legend)


def hbar(items, fmt, *, title, desc, lw=132, vmax=None, ref=None, ref_label="", w=400) -> str:
    """Horizontal bars. items: ('h', heading) or (label, value, css class)."""
    u = uid()
    pitch, bh = 22, 14
    vals = [i[1] for i in items if i[0] != "h" and i[1] is not None]
    top = vmax or max(vals + ([ref] if ref else []))
    bw_max = w - lw - 62
    y = 6
    body = []
    for it in items:
        if it[0] == "h":
            y += 4
            body.append(f'<text class="hd" x="4" y="{y + 11}">{esc(it[1])}</text>')
            y += 18
            continue
        lab, val, cls = it
        body.append(f'<text class="lb" x="{lw - 6}" y="{y + 11}" text-anchor="end">{esc(lab)}</text>')
        if val is None:
            body.append(f'<text class="vl" x="{lw}" y="{y + 11}">not measured</text>')
        else:
            bw = max(1.5, bw_max * val / top) if val > 0 else 1.5
            style = f' style="fill:url(#hat{u})"' if cls == "bn" else ""
            body.append(f'<rect class="{cls}" x="{lw}" y="{y}" width="{bw:.1f}" height="{bh}"{style}><title>{esc(lab)}: {esc(fmt(val))}</title></rect>')
            body.append(f'<text class="vl" x="{lw + bw + 4:.1f}" y="{y + 11}">{esc(fmt(val))}</text>')
        y += pitch
    if ref is not None:
        rx = lw + bw_max * ref / top
        body.append(f'<line class="ref" x1="{rx:.1f}" x2="{rx:.1f}" y1="2" y2="{y}"/>')
        body.append(f'<text class="ax" x="{rx + 3:.1f}" y="{y + 10}">{esc(ref_label)}</text>')
        y += 8
    h = y + 6
    defs = (
        f'<defs><pattern id="hat{u}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="6" height="6" style="fill:var(--card)"/><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--s5);stroke-width:3"/></pattern></defs>'
    )
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title><desc>{esc(desc)}</desc>{defs}{"".join(body)}</svg>'


def bar_legend(entries) -> str:
    """entries: (css class, text). Shows the fill of each bar class."""
    out = ['<ul class="legend">']
    for cls, text in entries:
        u = uid()
        if cls == "bn":
            fill = f'<defs><pattern id="lg{u}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:var(--card)"/><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--s5);stroke-width:3"/></pattern></defs><rect class="bn" x="1" y="1" width="30" height="10" style="fill:url(#lg{u})"/>'
        else:
            fill = f'<rect class="{cls}" x="1" y="1" width="30" height="10"/>'
        out.append(f'<li><svg viewBox="0 0 34 12" aria-hidden="true">{fill}</svg>{esc(text)}</li>')
    out.append("</ul>")
    return "".join(out)


STAGE_ORDER = ["parse", "prep_wait", "prep", "infer_wait", "infer", "encode", "http_and_handoffs"]
STAGE_NAME = {
    "parse": "parse",
    "prep_wait": "prep wait",
    "prep": "prep",
    "infer_wait": "infer wait",
    "infer": "infer",
    "encode": "encode",
    "http_and_handoffs": "HTTP + hand-offs",
}
STAGE_FILL = {  # css custom property for each stage, plus a pattern kind for a second cue
    "parse": ("s3", "dots"),
    "prep_wait": ("s6", "none"),
    "prep": ("s4", "dots"),
    "infer_wait": ("s6", "hatch"),
    "infer": ("s1", "none"),
    "encode": ("s2", "dots"),
    "http_and_handoffs": ("s5", "hatch"),
}


def _stage_defs(u: int) -> str:
    out = ["<defs>"]
    for st, (slot, kind) in STAGE_FILL.items():
        pid = f"st{u}{st}"
        if kind == "none":
            out.append(f'<pattern id="{pid}" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="4" style="fill:var(--{slot})"/></pattern>')
        elif kind == "dots":
            out.append(f'<pattern id="{pid}" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" style="fill:var(--{slot})"/><circle cx="2.5" cy="2.5" r="1" style="fill:var(--card)"/></pattern>')
        else:
            out.append(f'<pattern id="{pid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:var(--{slot})"/><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--card);stroke-width:2"/></pattern>')
    out.append("</defs>")
    return "".join(out)


def stacked(items, *, title, desc, lw=120, w=400) -> tuple[str, str]:
    """items: (label, {stage: ms}, total_ms). Returns svg and legend."""
    u = uid()
    pitch, bh = 26, 18
    bw_max = w - lw - 46
    top = max(t for _, _, t in items)
    y = 6
    body = []
    for lab, parts_, total in items:
        body.append(f'<text class="lb" x="{lw - 6}" y="{y + 13}" text-anchor="end">{esc(lab)}</text>')
        x = lw
        for st in STAGE_ORDER:
            v = parts_.get(st)
            if not v:
                continue
            seg = bw_max * v / top
            body.append(f'<rect class="seg" x="{x:.2f}" y="{y}" width="{max(seg, 0.6):.2f}" height="{bh}" style="fill:url(#st{u}{st})"><title>{esc(lab)}, {esc(STAGE_NAME[st])}: {v:.2f} ms</title></rect>')
            x += seg
        body.append(f'<text class="vl" x="{x + 4:.1f}" y="{y + 13}">{total:.2f} ms</text>')
        y += pitch
    h = y + 4
    svg = f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title><desc>{esc(desc)}</desc>{_stage_defs(u)}{"".join(body)}</svg>'
    leg = ['<ul class="legend">']
    for st in STAGE_ORDER:
        leg.append(f'<li><svg viewBox="0 0 34 12" aria-hidden="true">{_stage_defs(u + 1000)}<rect x="1" y="1" width="30" height="10" style="fill:url(#st{u + 1000}{st})"/></svg>{esc(STAGE_NAME[st])}</li>')
    leg.append("</ul>")
    return svg, "".join(leg)


def dotplot_log(items, *, title, desc, vline, vline_label, lo=1e-8, hi=1e-3, lw=150, w=400) -> str:
    """Log-scale dot plot. items: (label, value)."""
    ml, mr = lw, 16
    pw = w - ml - mr
    pitch = 24
    top = 26
    h = top + pitch * len(items) + 30

    def X(v):
        return ml + pw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title><desc>{esc(desc)}</desc>']
    e = int(math.log10(lo))
    while e <= int(math.log10(hi)):
        x = X(10.0**e)
        parts.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + pitch * len(items)}"/>')
        parts.append(f'<text class="ax" x="{x:.1f}" y="{top + pitch * len(items) + 13}" text-anchor="middle">1e{e}</text>')
        e += 1
    parts.append(f'<text class="ax" x="{ml}" y="{h - 3}">max abs error against eager PyTorch (log scale)</text>')
    vx = X(vline)
    parts.append(f'<line class="ref" x1="{vx:.1f}" x2="{vx:.1f}" y1="{top - 10}" y2="{top + pitch * len(items)}"/>')
    parts.append(f'<text class="ax" x="{vx:.1f}" y="{top - 13}" text-anchor="middle">{esc(vline_label)}</text>')
    for i, (lab, v) in enumerate(items):
        y = top + pitch * i + pitch / 2
        parts.append(f'<text class="lb" x="{ml - 6}" y="{y + 4}" text-anchor="end">{esc(lab)}</text>')
        x = X(v)
        parts.append(f'<circle class="mk" style="color:var(--s1);fill:var(--s1)" cx="{x:.1f}" cy="{y:.1f}" r="4.5"><title>{esc(lab)}: {sci(v)}</title></circle>')
        anchor, dx = ("end", -8) if x > w - 60 else ("start", 8)
        parts.append(f'<text class="vl" x="{x + dx:.1f}" y="{y + 4}" text-anchor="{anchor}">{sci(v)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def flow_diagram() -> str:
    w, h = 400, 276
    box = lambda x, y, bw, bh, t, cls="box": (  # noqa: E731
        f'<rect class="{cls}" x="{x}" y="{y}" width="{bw}" height="{bh}" rx="5"/>'
        f'<text class="lb" x="{x + bw / 2}" y="{y + bh / 2 + 4}" text-anchor="middle">{t}</text>'
    )
    arrow = lambda x1, y1, x2, y2: (  # noqa: E731
        f'<path class="arr" d="M{x1} {y1} L{x2 - 5 if x2 > x1 else x2} {y2}"/>'
        f'<path class="arrh" d="M{x2} {y2} l-6 -3.5 l0 7 z"/>'
    )
    p_ = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Flow: load, export, verify, verdict, backend"><title>The downshift flow</title><desc>Four steps run in order: load, export, verify, verdict. The verdict selects the backend. CLEAN selects ONNX Runtime. DEGRADED or FAILED selects eager PyTorch. UNVERIFIED selects ONNX Runtime and labels it as not checked.</desc>']
    for i, t in enumerate(["1 Load", "2 Export", "3 Verify", "4 Verdict"]):
        x = 8 + i * 100
        p_.append(box(x, 14, 84, 40, t))
        if i < 3:
            p_.append(arrow(x + 84, 34, x + 100, 34))
    p_.append('<text class="hd" x="8" y="92">5 Backend (the verdict selects it)</text>')
    rows = [("CLEAN", "ONNX Runtime"), ("DEGRADED or FAILED", "eager PyTorch"), ("UNVERIFIED", "ONNX Runtime, marked as not checked")]
    ys = [104, 154, 204]
    for (a, b), y in zip(rows, ys):
        p_.append(f'<rect class="boxs" x="8" y="{y}" width="322" height="38" rx="5"/>')
        p_.append(f'<text class="hd" x="18" y="{y + 16}">{a}</text>')
        p_.append(f'<text class="lb" x="18" y="{y + 31}">serve with: {b}</text>')
        p_.append(arrow(350, y + 19, 332, y + 19))
    p_.append(f'<path class="arr" d="M350 54 L350 {ys[-1] + 19}"/>')
    p_.append(f'<text class="ax" x="8" y="{h - 18}">Load: read the model. Export: make an ONNX graph.</text>')
    p_.append(f'<text class="ax" x="8" y="{h - 5}">Verify: compare the graph with PyTorch on 8 samples.</text>')
    p_.append("</svg>")
    return "".join(p_)


# ----------------------------------------------------------------------------------------
# Shared facts from the data
# ----------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def env1() -> dict:
    return jload(R1 / "results.json")["environment"]


def env_line(env: dict) -> str:
    return (
        f"{env['platform']}, {env['cpu_count']} logical CPUs, Python {env['python']}, torch {env['torch']}, "
        f"onnxruntime {env['onnxruntime']}, torch threads {env['torch_num_threads']}, run {env['timestamp']}"
    )


def sweep(model, server, batch=1, suite="fixture", payload=""):
    rows = [r for r in runs() if r["suite"] == suite and r["model"] == model and r["server"] == server and r["batch"] == batch and r["payload"] == payload]
    return sorted(rows, key=lambda r: r["concurrency"])


def at(model, server, c, batch=1, suite="fixture", payload=""):
    rows = [r for r in sweep(model, server, batch, suite, payload) if r["concurrency"] == c]
    if len(rows) != 1:
        raise SourceMissing(f"no single row for {model} / {server} / c={c} / batch={batch}")
    return rows[0]


def hfrow(server, payload, c, model="all-MiniLM-L6-v2"):
    rows = [r for r in runs() if r["suite"] == "hf" and r["model"] == model and r["server"] == server and r["payload"] == payload and r["concurrency"] == c]
    if len(rows) != 1:
        raise SourceMissing(f"no single hf row for {server} / {payload} / c={c}")
    return rows[0]


def peak_of(model, server):
    return peak(sweep(model, server))


def run_meta() -> str:
    return (
        f"Run {esc(env1()['timestamp'])}. {esc(env1()['platform'])}, {env1()['cpu_count']} logical CPUs, "
        f"Python {esc(env1()['python'])}, torch {esc(env1()['torch'])}, onnxruntime {esc(env1()['onnxruntime'])}. "
        "downshift 1.0.0rc1 from the checkout, 0.5.1 from PyPI. The folder is named <code>v1.0.0</code>."
    )


def section_sources(*paths: Path) -> str:
    return h2("Sources", "sources") + ul([srclink(x) for x in paths])


def common_limits() -> str:
    return (
        "One machine. One run. One 3 s window for each point (5 s for MiniLM). "
        "The file holds no repeats, so the size of run-to-run noise is not measured."
    )


# ----------------------------------------------------------------------------------------
# Page: throughput
# ----------------------------------------------------------------------------------------


def page_throughput() -> str:
    res = jload(R1 / "results.json")
    rerun = jload(R1 / "rerun_mlp_large.json")
    ceiling = {int(k): v for k, v in res["client_ceiling_rps"].items()}

    peaks = {(m, s): peak_of(m, s) for m in FIX_MODELS for s in FIX_SERVERS}
    # Claims used in the text below.
    claim(peaks[("clean_mlp", NAIVE_A)]["throughput_rps"] > peaks[("clean_mlp", DS_NEW)]["throughput_rps"], "async torch beats downshift on clean_mlp")
    claim(peaks[("clean_mlp", NAIVE_O)]["throughput_rps"] > peaks[("clean_mlp", DS_NEW)]["throughput_rps"], "naive ORT beats downshift on clean_mlp")
    claim(peaks[("tiny_bert", NAIVE_O)]["throughput_rps"] > peaks[("tiny_bert", DS_NEW)]["throughput_rps"], "naive ORT beats downshift on tiny_bert")
    claim(peaks[("tiny_bert", DS_NEW)]["throughput_rps"] > 3 * peaks[("tiny_bert", NAIVE_T)]["throughput_rps"], "downshift is above 3x naive torch on tiny_bert")
    claim(peaks[("mlp_large", NAIVE_T)]["throughput_rps"] > peaks[("mlp_large", NAIVE_O)]["throughput_rps"], "naive torch beats naive ORT on mlp_large")

    body = [header("Throughput and latency under load", "How many requests per second does each server handle, and how long does one request take? Batch 1, JSON bodies, CPU only.", run_meta())]

    ct = peaks[("clean_mlp", NAIVE_A)]["throughput_rps"]
    cd = peaks[("clean_mlp", DS_NEW)]["throughput_rps"]
    tb = peaks[("tiny_bert", DS_NEW)]["throughput_rps"]
    tn = peaks[("tiny_bert", NAIVE_T)]["throughput_rps"]
    body.append(
        tiles(
            [
                (f"{n0(cd)} vs {n0(ct)}", "clean_mlp peak, requests/s: downshift 1.0.0rc1 vs hand-rolled async torch", "The hand-rolled server is faster on a tiny model."),
                (f"{n0(tb)} vs {n0(tn)}", "tiny_bert peak, requests/s: downshift 1.0.0rc1 vs hand-rolled FastAPI + torch", "ORT through downshift vs eager torch."),
                (f"{n0(peaks[('tiny_bert', NAIVE_O)]['throughput_rps'])}", "tiny_bert peak of hand-rolled FastAPI + ORT", "Faster than downshift on the same graph."),
            ]
        )
    )

    body.append(h2("The question", "question"))
    body.append(p("A hand-rolled model server is a short FastAPI file. downshift does more work for each request. It validates the body, runs the model on a thread pool, and keeps the verdict and metrics. This page shows what that costs and what it gives, on four models."))

    body.append(h2("Method", "method"))
    body.append(
        ul(
            [
                "Command that made the data: <code>python -m bench.run --out bench/results/v1.0.0/results.json</code>. The file stores its settings. They match <code>--cases clean_mlp tiny_bert mlp_large scatter_include_self_false --skip-workers</code>. The exact command line is not stored.",
                "Chart data: <code>python -m bench.report ... --csv bench/results/v1.0.0/chart_runs.csv</code>. Every point on this page is one row of that CSV.",
                "Load: closed loop. Each of N clients sends its next request when the last one returns. N is the concurrency (1 to 64). One 1.0 s warm-up and one 3.0 s window for each point.",
                "Every server gets the same seeded weights and byte-identical request bodies. Every downshift server is the real <code>downshift serve</code> command. The result cache is off in every row on this page (<code>--result-cache-rows 0</code>).",
                "The hand-rolled servers take a plain <code>dict</code> body and return a plain <code>dict</code>. They do less work than downshift for each request. Source: <code>bench/servers.py</code>.",
                "<code>async def</code> blocks the event loop. It is a warning row, not a recommended design.",
            ]
        )
    )

    body.append(h2("Results", "results"))
    body.append(p("Each chart shows requests per second against concurrent clients. A higher line is better. The table below each chart has the numbers."))
    for m in FIX_MODELS:
        body.append(h3(m))
        series = []
        for s in CORE5:
            vals = [at(m, s, c)["throughput_rps"] for c in CONCS]
            series.append((SHORT[s], vals, LINE_STYLE[s]))
        svg, leg = line_chart([f"c={c}" for c in CONCS], series, title=f"Requests per second against concurrency, {m}", desc="Five lines, one for each server. Values are in the table below.", ylabel="requests per second")
        body.append(figure(svg, f"{m}: requests per second, batch 1. Each marker is one 3 s window.", leg))
        rows = []
        for s in FIX_SERVERS:
            pk = peaks[(m, s)]
            r1, r8 = at(m, s, 1), at(m, s, 8)
            rows.append([s, n0(pk["throughput_rps"]), pk["concurrency"], n0(r1["throughput_rps"]), n2(r1["p50_ms"]), n2(r1["p99_ms"]), n2(r8["p50_ms"]), n2(r8["p99_ms"])])
        hdr = ["server", "peak rps", "at c=", "rps c=1", "p50 ms c=1", "p99 ms c=1", "p50 ms c=8", "p99 ms c=8"]
        body.append(table(hdr, rows, caption=f"{m}: peak throughput and latency, batch 1", right=range(1, 8)))
        if m in ("clean_mlp", "mlp_large"):
            mdt(f"Throughput and latency, batch 1: {m}", "Peak requests/s over the concurrency sweep; p50 and p99 in ms.", [R1 / "chart_runs.csv"], hdr, rows)
    body.append(note("limit", "The row <code>1.0.0rc1 result cache on</code> sends one body again and again. Every request is a hit. It is a best case, not a normal workload. See the result cache page. At high concurrency the latency rises by design. In a closed loop, the median latency is close to concurrency divided by throughput."))

    # rc1 versus 0.5.1
    body.append(h2("Did 1.0.0rc1 get faster than 0.5.1?", "release"))
    rows = []
    rerun_peaks = {}
    for x in rerun["runs"]:
        key = (x["variant"], x["version"])
        if x["batch"] == 1:
            rerun_peaks[key] = max(rerun_peaks.get(key, 0), x["throughput_rps"])
    for m in FIX_MODELS:
        a, b = peaks[(m, DS_OLD)]["throughput_rps"], peaks[(m, DS_NEW)]["throughput_rps"]
        a1, b1 = at(m, DS_OLD, 1), at(m, DS_NEW, 1)
        note_ = ""
        if m == "mlp_large":
            ra, rb = rerun_peaks[("downshift", "0.5.1")], rerun_peaks[("downshift", "1.0.0rc1")]
            note_ = f"re-run: {n0(rb)} vs {n0(ra)} ({pct((rb / ra - 1) * 100)})"
        rows.append([m, n0(a), n0(b), pct((b / a - 1) * 100), n0(a1["throughput_rps"]), n0(b1["throughput_rps"]), pct((b1["throughput_rps"] / a1["throughput_rps"] - 1) * 100), note_])
    hdr = ["model", "peak 0.5.1", "peak 1.0.0rc1", "change", "c=1 0.5.1", "c=1 1.0.0rc1", "change", "note"]
    body.append(table(hdr, rows, caption="downshift 1.0.0rc1 against 0.5.1, same session, batch 1, result cache off", right=range(1, 7)))
    mdt("1.0.0rc1 against 0.5.1, batch 1", "Peak requests/s and requests/s at c=1. The mlp_large row of the main pass is marked in the note column.", [R1 / "chart_runs.csv", R1 / "rerun_mlp_large.json"], hdr, rows)
    mp = {m: peaks[(m, DS_NEW)]["throughput_rps"] / peaks[(m, DS_OLD)]["throughput_rps"] - 1 for m in FIX_MODELS}
    claim(abs(mp["clean_mlp"]) < 0.05 and abs(mp["scatter_include_self_false"]) < 0.05 and -0.15 < mp["tiny_bert"] < 0, "peak changes of clean_mlp, scatter and tiny_bert")
    body.append(
        p(
            "The data shows no gain from 0.5.1 to 1.0.0rc1. Peak throughput is within 5% on two models and "
            f"{pct(mp['tiny_bert'] * 100)} on <code>tiny_bert</code>. The first <code>mlp_large</code> pass of 1.0.0rc1 was slow "
            f"({n0(peaks[('mlp_large', DS_NEW)]['throughput_rps'])} against {n0(peaks[('mlp_large', DS_OLD)]['throughput_rps'])} requests/s). "
            "The report says this does not fit one code path. The case was measured again with batch 1 only. Use the re-run for comparisons on <code>mlp_large</code>."
        )
    )
    claim(
        rerun_peaks[("downshift", "1.0.0rc1")] > 930 and rerun_peaks[("downshift", "1.0.0rc1")] < 945,
        "rerun peak of 1.0.0rc1 default is near 939 to 942",
    )
    body.append(
        note(
            "bad",
            f"The text of <code>results/v1.0.0/REPORT.md</code> gives the re-run peaks as 939, 920 and 989 requests/s. "
            f"The file <code>rerun_mlp_large.json</code> has {n1(rerun_peaks[('downshift', '1.0.0rc1')])}, {n1(rerun_peaks[('downshift_metrics', '1.0.0rc1')])} and {n1(rerun_peaks[('downshift', '0.5.1')])}. "
            "The report values are the c=16 points. The tables on this page use the file values.",
            "Source mismatch.",
        )
    )

    # metrics cost
    body.append(h2("What does --metrics cost?", "metrics"))
    rows = []
    for m in FIX_MODELS:
        d, mt = at(m, DS_NEW, 1), at(m, DS_METRICS, 1)
        us = (1 / mt["throughput_rps"] - 1 / d["throughput_rps"]) * 1e6
        rows.append([m, n0(d["throughput_rps"]), n0(mt["throughput_rps"]), f"{us:+.0f}", pct((mt["throughput_rps"] / d["throughput_rps"] - 1) * 100)])
    hdr = ["model", "rps c=1 default", "rps c=1 --metrics", "extra time per request (us)", "change in rps"]
    body.append(table(hdr, rows, caption="--metrics at concurrency 1 (1/rps difference, in microseconds)", right=range(1, 5)))
    mdt("Cost of --metrics at c=1", "Extra time per request is 1/rps(metrics) - 1/rps(default).", [R1 / "chart_runs.csv"], hdr, rows)
    body.append(p("The main-pass <code>mlp_large</code> row is the slow pass described above. Its c=1 numbers are not a good base for this comparison."))

    # ceiling
    body.append(h2("Was the load generator the limit?", "ceiling"))
    ratios = []
    for r in runs():
        if r["suite"] == "fixture" and r["batch"] == 1 and r["server"] in FIX_SERVERS:
            ratios.append((r["throughput_rps"] / ceiling[r["concurrency"]], r))
    worst, wr = max(ratios, key=lambda t: t[0])
    rows = [[f"c={c}", n0(ceiling[c])] for c in CONCS]
    body.append(p(f"Before the sweep, the harness measures the load generator against <code>GET /health</code> on a server that does nothing. This is the ceiling. The highest ratio of any batch-1 result to the ceiling at the same concurrency is {worst * 100:.0f}% ({esc(wr['server'])}, {esc(wr['model'])}, c={wr['concurrency']}). No result is close to the ceiling."))
    claim(worst < 0.6, "no batch-1 result is above 60% of the load-generator ceiling")
    body.append(table(["concurrency", "ceiling rps"], rows, caption="Load generator ceiling (GET /health on a do-nothing server)", right=(1,)))
    mdt("Load generator ceiling", "Requests/s of GET /health on a do-nothing server.", [R1 / "results.json"], ["concurrency", "ceiling rps"], rows)

    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(
        ul(
            [
                "On the tiny fixtures, a hand-rolled server handles more requests per second than downshift. The HTTP stack is most of the time for one request (see the time page).",
                "On <code>tiny_bert</code>, downshift is more than 3 times faster than the hand-rolled eager torch server. The hand-rolled ORT server is faster than downshift.",
                "1.0.0rc1 and 0.5.1 are close. The data shows no gain and no clear loss, except the slow first <code>mlp_large</code> pass.",
            ]
        )
    )
    body.append(h3("Does not show"))
    body.append(
        ul(
            [
                common_limits(),
                "Only 4 models. The 0.5.0 run had 8 models. The 1.0.0rc1 run did not include <code>dynamic_batch_cnn</code>, <code>gnn_gcn</code>, <code>cnn_large</code> or <code>bert_small</code>.",
                "CPU only, Windows only. GPU, Linux and other CPUs are not measured.",
                "The load generator runs on the same machine as the server. The CPU is shared.",
                "No quantised models. No dynamic request batching (downshift has none by design).",
            ]
        )
    )
    body.append(section_sources(R1 / "chart_runs.csv", R1 / "results.json", R1 / "rerun_mlp_large.json", R1 / "REPORT.md", BENCH / "servers.py", BENCH / "loadgen.py"))
    return frame("throughput", "Throughput and latency", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: time breakdown
# ----------------------------------------------------------------------------------------


def stage_map():
    out = defaultdict(dict)
    totals = {}
    for r in stages():
        key = (r["model"], r["server"])
        out[key][r["stage"]] = r["ms"]
        totals[key] = r["client_ms"]
    return out, totals


def page_time() -> str:
    sm, tot = stage_map()
    body = [header("Where the time of one request goes", "For one request at a time, how many milliseconds go to parsing, waiting, inference, encoding and the HTTP stack?", run_meta())]
    c = ("clean_mlp", DS_NEW)
    t = ("tiny_bert", DS_NEW)
    m_ = ("all-MiniLM-L6-v2", DS_NEW)
    mt = ("all-MiniLM-L6-v2", "downshift 1.0.0rc1 --backend torch")
    share = lambda key, st: sm[key].get(st, 0) / tot[key] * 100  # noqa: E731
    claim(share(c, "http_and_handoffs") > 55, "HTTP share of clean_mlp is above 55%")
    claim(share(m_, "infer") > 55, "inference share of MiniLM on ORT is above 55%")
    body.append(
        tiles(
            [
                (f"{share(c, 'http_and_handoffs'):.0f}%", "clean_mlp: share of the request in HTTP and hand-offs", f"{n2(sm[c]['http_and_handoffs'])} of {n2(tot[c])} ms"),
                (f"{share(t, 'infer'):.0f}%", "tiny_bert: share in model inference", f"{n2(sm[t]['infer'])} of {n2(tot[t])} ms"),
                (f"{share(m_, 'infer'):.0f}% vs {share(mt, 'infer'):.0f}%", "MiniLM: share in inference, ONNX Runtime vs torch backend", f"{n2(sm[m_]['infer'])} vs {n2(sm[mt]['infer'])} ms"),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("A request has a fixed part (HTTP, JSON, thread hand-offs) and a model part. On a tiny model the fixed part is most of the time. On an encoder the model part is most of the time. This page splits the time of one request into stages."))
    body.append(h2("Method", "method"))
    body.append(
        ul(
            [
                "The harness sends 300 requests one after the other on one keep-alive connection. Batch 1 (MiniLM: one sentence). It takes the median of each value.",
                "<code>total</code> is what the client measured. The stage values are the <code>Server-Timing</code> header of the server. Hand-rolled servers have no such header.",
                "<code>HTTP + hand-offs</code> is not measured directly. It is the client total minus every stage. It holds uvicorn, routing, middleware and thread hand-offs back to the event loop. For a hand-rolled server it is the whole total.",
                "Data: <code>bench/results/v1.0.0/chart_stages.csv</code>. Made by <code>bench.report --stages-csv</code> from <code>results.json</code> and <code>results_hf.json</code>.",
            ]
        )
    )
    body.append(h2("Results", "results"))
    body.append(p("Each bar is the median total in milliseconds. The segments are the stages, in the order of the legend. A shorter bar is better."))
    models = FIX_MODELS + ["all-MiniLM-L6-v2"]
    for m in models:
        servers = [s for s in [NAIVE_T, NAIVE_O, NAIVE_A, "naive FastAPI + transformers", "naive transformers, async def", DS_OLD, DS_NEW, DS_CACHE, "downshift 0.5.1 --backend torch", "downshift 1.0.0rc1 --backend torch"] if (m, s) in sm]
        items = []
        for s in servers:
            if s == DS_CACHE:
                continue  # best case with a repeated body; shown on the result cache page
            items.append((SHORT[s], sm[(m, s)], tot[(m, s)]))
        svg, leg = stacked(items, title=f"Stages of one request, {m}", desc="One stacked bar for each server. The numbers are in the table below.")
        body.append(h3(m))
        body.append(figure(svg, f"{m}: median time of one request, in ms, by stage.", leg))
        hdr = ["server", "total ms"] + [STAGE_NAME[s] for s in STAGE_ORDER]
        rows = []
        for s in servers:
            d = sm[(m, s)]
            rows.append([s, n2(tot[(m, s)])] + [n2(d[st]) if st in d else "-" for st in STAGE_ORDER])
        body.append(table(hdr, rows, caption=f"{m}: median ms by stage (concurrency 1, batch 1)", right=range(1, 9)))
        if m in ("clean_mlp", "all-MiniLM-L6-v2"):
            mdt(f"Time of one request by stage: {m}", "Median ms of 300 sequential requests, batch 1. A dash means the server reports no such stage.", [R1 / "chart_stages.csv"], hdr, rows)
    body.append(note("limit", "The cache-on rows are left out of the charts. They answer from memory, so their inference stage is 0. See the result cache page."))
    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(
        ul(
            [
                f"On <code>clean_mlp</code> the model needs {n2(sm[c]['infer'])} ms of {n2(tot[c])} ms. The rest is the fixed cost of the request path.",
                f"On MiniLM the ONNX Runtime backend spends {n2(sm[m_]['infer'])} ms in inference. The torch backend spends {n2(sm[mt]['infer'])} ms on the same input. The other stages are close.",
                "The prep, wait and encode stages of downshift are each below 0.4 ms in this file for all five models.",
            ]
        )
    )
    claim(all(sm[(m, DS_NEW)].get(st, 0) < 0.4 for m in models for st in ("parse", "prep_wait", "prep", "infer_wait", "encode")), "downshift non-infer stages are below 0.4 ms")
    body.append(h3("Does not show"))
    body.append(
        ul(
            [
                "Time under load. These are sequential requests. Queueing at high concurrency is not in this table.",
                "Where the HTTP stack time goes inside. The file has one derived value, not a profile.",
                "Run-to-run spread. The file holds one median for each cell. " + common_limits(),
                "One batch size only (1).",
            ]
        )
    )
    body.append(section_sources(R1 / "chart_stages.csv", R1 / "results.json", R1 / "results_hf.json"))
    return frame("time-breakdown", "Time per request", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: torch vs ORT
# ----------------------------------------------------------------------------------------


def page_torch_ort() -> str:
    res = jload(R1 / "results.json")
    inproc = res["inproc"]
    body = [header("ONNX Runtime against eager PyTorch", "When a model is CLEAN, downshift serves it with ONNX Runtime. How much faster is that than eager PyTorch? First without a server, then through HTTP.", run_meta())]

    rows, ratio_items = [], []
    for e in inproc:
        t, o = e["torch"]["compute"]["p50_ms"], e["onnxruntime"]["compute"]["p50_ms"]
        flag = " (wrong numbers, see note)" if e["verdict"] != "CLEAN" else ""
        rows.append([e["case"], e["batch"], e["verdict"], n2(t) if t >= 0.1 else f"{t:.3f}", n2(o) if o >= 0.1 else f"{o:.3f}", f"{t / o:.1f}x"])
        ratio_items.append((e["case"], e["batch"], t / o, e["verdict"]))
    ratios = [r for _, _, r, v in ratio_items if v == "CLEAN"]
    claim(min(ratios) > 1.0, "ORT compute is faster than torch compute in every CLEAN in-process row")
    lo, hi = min(ratios), max(ratios)

    mini = lambda srv, pay, c: hfrow(srv, pay, c)  # noqa: E731
    sp_b1 = mini(DS_NEW, "short_b1", 8)["throughput_rps"] / mini("downshift 1.0.0rc1 --backend torch", "short_b1", 8)["throughput_rps"]
    sp_l = mini(DS_NEW, "long_b1", 8)["throughput_rps"] / mini("downshift 1.0.0rc1 --backend torch", "long_b1", 8)["throughput_rps"]
    sps = [mini(DS_NEW, pay, 8)["throughput_rps"] / mini("downshift 1.0.0rc1 --backend torch", pay, 8)["throughput_rps"] for pay in ("short_b1", "short_b8", "long_b1")]
    claim(sps == sorted(sps, reverse=True) and min(sps) > 1.1, "MiniLM ORT speed-up falls with input size")

    body.append(
        tiles(
            [
                (f"{lo:.1f}x to {hi:.1f}x", "ORT against torch, compute only, CLEAN models, p50", "9 rows: 3 CLEAN models, 3 batch sizes; the scatter rows are excluded"),
                (f"{sp_b1:.1f}x", "MiniLM, one sentence, c=8: downshift ORT vs downshift --backend torch", f"{n0(mini(DS_NEW, 'short_b1', 8)['throughput_rps'])} vs {n0(mini('downshift 1.0.0rc1 --backend torch', 'short_b1', 8)['throughput_rps'])} rps"),
                (f"{sp_l:.1f}x", "MiniLM, about 180 tokens, c=8: same pair", f"{n0(mini(DS_NEW, 'long_b1', 8)['throughput_rps'])} vs {n0(mini('downshift 1.0.0rc1 --backend torch', 'long_b1', 8)['throughput_rps'])} rps"),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("downshift exists to serve the ONNX graph only when the graph is right. This is worth the work only if ONNX Runtime is faster. This page measures that gain. It does not measure the gate."))

    body.append(h2("Method", "method"))
    body.append(
        ul(
            [
                "<b>In-process.</b> <code>python -m bench.inproc &lt;case&gt; &lt;batch&gt; &lt;iters&gt;</code>, started by <code>bench.run</code> with 200 iterations after 20 warm-up calls. It calls the backend on arrays that are already in memory. No HTTP. Three levels: <code>compute</code> only; <code>compute + JSON</code> (parse the body, infer, write the response); <code>compute + base64</code>. Source: <code>bench/inproc.py</code>. Data: <code>results.json</code>, key <code>inproc</code>.",
                "<b>Over HTTP, MiniLM.</b> <code>python -m bench.hf_models --out bench/results/v1.0.0/results_hf.json</code> (5 s windows, c=1 and c=8). <code>--backend torch</code> forces eager PyTorch in the same downshift server. All other settings are equal.",
                "<b>Over HTTP, fixtures.</b> The two hand-rolled servers (FastAPI + torch, FastAPI + ORT) use the same weights and graph. See the throughput page.",
            ]
        )
    )

    body.append(h2("Results: compute only, no server", "inproc"))
    items = []
    for case in FIX_MODELS:
        items.append(("h", case + ("  (graph gives wrong numbers)" if case.startswith("scatter") else "")))
        for e in inproc:
            if e["case"] == case:
                items.append((f"batch {e['batch']}", e["torch"]["compute"]["p50_ms"] / e["onnxruntime"]["compute"]["p50_ms"], "bw" if e["verdict"] == "CLEAN" else "bn"))
    svg = hbar(items, lambda v: f"{v:.1f}x", title="Torch p50 time divided by ORT p50 time, compute only", desc="Each bar is how many times faster ONNX Runtime is than eager PyTorch. A value above 1 means ORT is faster. Values are in the table.", ref=1.0, ref_label="1x: equal", lw=70)
    body.append(figure(svg, "Speed-up of ONNX Runtime over eager PyTorch, p50 of 200 calls, no HTTP. Hatched bars: the ONNX graph for this model gives wrong numbers.", bar_legend([("bw", "CLEAN model"), ("bn", "DEGRADED model: ORT time of a wrong graph")])))
    hdr = ["model", "batch", "verdict", "torch p50 ms", "ORT p50 ms", "torch / ORT"]
    body.append(table(hdr, rows, caption="Compute only, p50 of 200 calls", right=range(1, 6)))
    mdt("In-process compute: torch against ONNX Runtime", "p50 of 200 calls, no HTTP. The scatter rows time a graph that gives wrong numbers.", [R1 / "results.json"], hdr, rows)
    body.append(note("limit", "For <code>scatter_include_self_false</code> the verdict is DEGRADED. The ONNX graph gives wrong numbers, and downshift does not serve it. Its ORT time is not a speed that a user gets from downshift."))

    body.append(h3("What the JSON codec adds"))
    rows2 = []
    for e in inproc:
        if e["batch"] == 32:
            o = e["onnxruntime"]
            rows2.append([e["case"], n2(o["compute"]["p50_ms"]), n2(o["compute_json"]["p50_ms"]), n2(o["compute_binary"]["p50_ms"]), f"{o['compute_json']['p50_ms'] / o['compute']['p50_ms']:.1f}x", e["request_bytes"], e["request_bytes_binary"]])
    hdr2 = ["model (batch 32)", "ORT compute ms", "+ JSON ms", "+ base64 ms", "JSON / compute", "JSON bytes", "base64 bytes"]
    body.append(table(hdr2, rows2, caption="ORT at batch 32: compute, compute + JSON round trip, compute + base64 round trip (p50)", right=range(1, 7)))
    mdt("In-process: cost of the wire format, ORT, batch 32", "p50 of 200 calls. JSON and base64 columns include parse, infer and write.", [R1 / "results.json"], hdr2, rows2)
    claim(float(rows2[2][1]) * 5 < float(rows2[2][2]), "mlp_large b32 JSON is more than 5x compute")
    body.append(p("For a wide input the JSON round trip takes much more time than the model. The wire formats page shows the effect over HTTP."))

    body.append(h2("Results: MiniLM over HTTP", "minilm"))
    pays = [("short_b1", "one sentence"), ("short_b8", "eight sentences"), ("long_b1", "about 180 tokens")]
    servers = ["naive FastAPI + transformers", "naive transformers, async def", "downshift 0.5.1", "downshift 0.5.1 --backend torch", DS_NEW, "downshift 1.0.0rc1 --backend torch"]
    items = []
    for pay, nm in pays:
        items.append(("h", f"{nm} ({pay}), requests/s at c=8"))
        for s in servers:
            items.append((SHORT[s], mini(s, pay, 8)["throughput_rps"], bar_class(s)))
    svg = hbar(items, n0, title="MiniLM requests per second at concurrency 8", desc="Grouped bars for three payloads. Values are in the table.", lw=140)
    body.append(figure(svg, "all-MiniLM-L6-v2, c=8, requests per second. Longer is better.", bar_legend([("bn", "hand-rolled (torch)"), ("bo", "downshift 0.5.1"), ("bw", "downshift 1.0.0rc1"), ("bv", "downshift with --backend torch")])))
    hdr3 = ["server", "short_b1 rps c=8", "short_b8 rps c=8", "long_b1 rps c=8", "short_b1 p50 c=1 ms", "short_b8 p50 c=1 ms", "long_b1 p50 c=1 ms"]
    rows3 = []
    for s in servers:
        rows3.append([s] + [n0(mini(s, pay, 8)["throughput_rps"]) for pay, _ in pays] + [n1(mini(s, pay, 1)["p50_ms"]) for pay, _ in pays])
    body.append(table(hdr3, rows3, caption="all-MiniLM-L6-v2: requests/s at c=8 and p50 at c=1", right=range(1, 7)))
    mdt("MiniLM over HTTP: torch against ONNX Runtime", "Requests/s at c=8 and p50 ms at c=1, 5 s windows. Cache off.", [R1 / "chart_runs.csv"], hdr3, rows3)
    body.append(p(f"The gain of ONNX Runtime falls with the input size: {sp_b1:.1f}x for one sentence, {mini(DS_NEW, 'short_b8', 8)['throughput_rps'] / mini('downshift 1.0.0rc1 --backend torch', 'short_b8', 8)['throughput_rps']:.1f}x for eight sentences, {sp_l:.1f}x for about 180 tokens. The hand-rolled server runs eager torch. downshift with <code>--backend torch</code> is close to it ({n0(mini('downshift 1.0.0rc1 --backend torch', 'short_b1', 8)['throughput_rps'])} against {n0(mini('naive FastAPI + transformers', 'short_b1', 8)['throughput_rps'])} requests/s). The hand-rolled server cuts over-long text. downshift rejects it."))

    body.append(h2("Results: hand-rolled torch against hand-rolled ORT", "naive"))
    rows4 = []
    for m in FIX_MODELS:
        a, b = peak_of(m, NAIVE_T), peak_of(m, NAIVE_O)
        rows4.append([m, n0(a["throughput_rps"]), n0(b["throughput_rps"]), f"{b['throughput_rps'] / a['throughput_rps']:.2f}x", "wrong numbers" if m.startswith("scatter") else "ok"])
    hdr4 = ["model", "FastAPI + torch peak rps", "FastAPI + ORT peak rps", "ORT / torch", "ORT graph"]
    body.append(table(hdr4, rows4, caption="Peak requests/s of the two hand-rolled servers (same weights, same graph)", right=range(1, 4)))
    mdt("Hand-rolled FastAPI: torch against ORT, peak requests/s", "Batch 1, JSON. The ORT graph of scatter_include_self_false gives wrong numbers.", [R1 / "chart_runs.csv"], hdr4, rows4)
    claim(peak_of("mlp_large", NAIVE_T)["throughput_rps"] > peak_of("mlp_large", NAIVE_O)["throughput_rps"], "naive torch beats naive ORT on mlp_large")
    body.append(p("On <code>mlp_large</code> the eager torch server has the higher peak, although ORT is faster in the in-process table. The data does not say why. Do not assume that ORT wins on every model over HTTP."))

    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(ul([f"Without a server, ORT is {lo:.1f}x to {hi:.1f}x faster than eager torch on the CLEAN models and batch sizes measured.", f"On a real encoder behind HTTP (MiniLM), ORT gives {min(sps):.1f}x to {max(sps):.1f}x more requests per second than eager torch in the same server, and the gain is largest for short inputs.", "The gain does not carry over to every model over HTTP (see mlp_large)."]))
    body.append(h3("Does not show"))
    body.append(ul(["Any GPU result. Torch threads were 8 and the ORT provider was CPU (the file lists Azure and CPU providers).", "Why a result differs. The files hold no profile.", "ORT with graph optimisation or quantisation. downshift does neither.", common_limits(), "The in-process numbers use 200 calls. The p50 is stable, but the tail (p99) of 200 calls is not."]))
    body.append(section_sources(R1 / "results.json", R1 / "results_hf.json", R1 / "chart_runs.csv", BENCH / "inproc.py", BENCH / "hf_models.py"))
    return frame("torch-vs-ort", "Torch vs ONNX Runtime", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: wire formats and batches
# ----------------------------------------------------------------------------------------


def page_wire() -> str:
    body = [header("Batches and wire formats", "What happens when a request carries 8 or 32 rows, and when the body is safetensors and not JSON?", run_meta())]
    rows_ = lambda m, s, b: at(m, s, 8, batch=b)  # noqa: E731
    claim(rows_("mlp_large", DS_NEW_ST, 32)["rows_per_s"] > 3 * rows_("mlp_large", DS_NEW, 32)["rows_per_s"], "safetensors is above 3x JSON on mlp_large b32")
    claim(rows_("clean_mlp", DS_NEW_ST, 32)["rows_per_s"] < rows_("clean_mlp", DS_NEW, 32)["rows_per_s"], "safetensors is slower on clean_mlp b32")
    claim(rows_("tiny_bert", DS_NEW_ST, 32)["request_bytes"] > rows_("tiny_bert", DS_NEW, 32)["request_bytes"], "tiny_bert safetensors body is bigger")
    claim(all(rows_("tiny_bert", st, 32)["rows_per_s"] > rows_("tiny_bert", js, 32)["rows_per_s"] for st, js in ((DS_OLD_ST, DS_OLD), (DS_NEW_ST, DS_NEW))), "tiny_bert safetensors rows/s above JSON")
    claim(all(rows_("clean_mlp", st, 32)["rows_per_s"] < rows_("clean_mlp", js, 32)["rows_per_s"] for st, js in ((DS_OLD_ST, DS_OLD), (DS_NEW_ST, DS_NEW))), "clean_mlp safetensors rows/s below JSON in both versions")
    claim(rows_("mlp_large", DS_OLD_ST, 32)["rows_per_s"] > 3 * rows_("mlp_large", DS_OLD, 32)["rows_per_s"], "0.5.1 mlp_large safetensors above 3x JSON")
    ml_j, ml_s = rows_("mlp_large", DS_NEW, 32), rows_("mlp_large", DS_NEW_ST, 32)
    ml5_j, ml5_s = rows_("mlp_large", DS_OLD, 32), rows_("mlp_large", DS_OLD_ST, 32)
    body.append(
        tiles(
            [
                (f"{n0(ml5_s['rows_per_s'])} vs {n0(ml5_j['rows_per_s'])}", "mlp_large, batch 32, c=8, rows/s: 0.5.1 safetensors vs JSON", "Clean comparison, same pass"),
                (f"{ml_s['request_bytes'] / ml_j['request_bytes']:.2f}x", "mlp_large, batch 32: safetensors body size divided by JSON body size", f"{ml_s['request_bytes']} against {ml_j['request_bytes']} bytes"),
                (f"{n0(rows_('clean_mlp', DS_NEW_ST, 32)['rows_per_s'])} vs {n0(rows_('clean_mlp', DS_NEW, 32)['rows_per_s'])}", "clean_mlp, batch 32, c=8, rows/s: safetensors vs JSON", "safetensors is slower here"),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("downshift runs one inference for each request. A client that has many rows can send them in one request. A client can also send binary tensors (safetensors) in place of JSON text. This page shows what each choice does to the rows per second and to the body size."))
    body.append(h2("Method", "method"))
    body.append(ul(["Same command and files as the throughput page. Batch 8 and 32 are in <code>--batch 1 8 32</code>; the batch sweep ran at concurrency 1, 8 and 32. This page uses c=8.", "<code>rows/s</code> is requests/s times the batch size.", "<code>downshift_safetensors</code> sends the request as safetensors and asks for a safetensors response. Version 0.5.0 added it. The base64 variant of earlier runs is not in this run.", "The hand-rolled async server did not run for batches."]))

    body.append(h2("Results", "results"))
    items = []
    for m in FIX_MODELS:
        items.append(("h", f"{m}, batch 32, c=8, rows/s"))
        items.append(("0.5.1 JSON", rows_(m, DS_OLD, 32)["rows_per_s"], "bo"))
        items.append(("0.5.1 safetensors", rows_(m, DS_OLD_ST, 32)["rows_per_s"], "bo"))
        items.append(("1.0.0rc1 JSON", rows_(m, DS_NEW, 32)["rows_per_s"], "bw"))
        items.append(("1.0.0rc1 safetensors", rows_(m, DS_NEW_ST, 32)["rows_per_s"], "bw"))
    svg = hbar(items, n0, title="Rows per second at batch 32 and concurrency 8", desc="Four bars for each model. Values are in the table below.", lw=128)
    body.append(figure(svg, "Rows per second, batch 32, c=8. Longer is better.", bar_legend([("bo", "downshift 0.5.1"), ("bw", "downshift 1.0.0rc1")])))
    items = []
    for m in FIX_MODELS:
        j, s = rows_(m, DS_NEW, 32), rows_(m, DS_NEW_ST, 32)
        items.append((m if len(m) < 20 else "scatter_include_..", s["request_bytes"] / j["request_bytes"], "bw"))
    svg = hbar(items, lambda v: f"{v:.2f}x", title="Safetensors body size divided by JSON body size, batch 32", desc="A value below 1 means the safetensors body is smaller.", ref=1.0, ref_label="1x: equal", lw=128)
    body.append(figure(svg, "Size of the safetensors request body divided by the JSON body, batch 32. Below 1: smaller.", ""))
    hdr = ["model", "server", "request bytes", "requests/s", "p50 ms", "rows/s"]
    rows = []
    for m in FIX_MODELS:
        for s in (DS_OLD, DS_OLD_ST, DS_NEW, DS_NEW_ST):
            r = rows_(m, s, 32)
            rows.append([m, s, r["request_bytes"], n0(r["throughput_rps"]), n2(r["p50_ms"]), n0(r["rows_per_s"])])
    body.append(table(hdr, rows, caption="Wire encodings, batch 32, concurrency 8", right=range(2, 6)))
    mdt("Wire encodings, batch 32, c=8", "JSON against safetensors. The 1.0.0rc1 mlp_large JSON row comes from the slow main pass.", [R1 / "chart_runs.csv"], hdr, rows)

    hdr2 = ["server", "clean_mlp b8", "clean_mlp b32", "tiny_bert b8", "tiny_bert b32", "mlp_large b8", "mlp_large b32", "scatter b8", "scatter b32"]
    rows2 = []
    for s in [NAIVE_T, NAIVE_O, DS_OLD, DS_OLD_ST, DS_NEW, DS_NEW_ST, DS_CACHE, DS_METRICS]:
        rows2.append([s] + [n0(rows_(m, s, b)["rows_per_s"]) for m in FIX_MODELS for b in (8, 32)])
    body.append(table(hdr2, rows2, caption="Rows per second at concurrency 8 (requests/s times batch)", right=range(1, 9)))
    mdt("Batched requests at c=8, rows/s", "Requests/s times batch size. The cache-on row is a best case with one repeated body.", [R1 / "chart_runs.csv"], hdr2, rows2)
    body.append(note("limit", "In the 1.0.0rc1 JSON row for <code>mlp_large</code>, the main pass was slow (see the throughput page). The re-run covered batch 1 only. The 1.0.0rc1 batch rows for <code>mlp_large</code> are not re-measured. Use the 0.5.1 rows for the clean JSON against safetensors comparison."))

    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(
        ul(
            [
                f"For a wide input (<code>mlp_large</code>, 512 floats a row), safetensors gives more than 3 times the rows per second of JSON at batch 32 ({n0(ml5_s['rows_per_s'])} against {n0(ml5_j['rows_per_s'])} for 0.5.1), and the body is {ml_j['request_bytes'] / ml_s['request_bytes']:.1f} times smaller.",
                "For a tiny input (<code>clean_mlp</code>, 16 floats a row) safetensors is slower than JSON in both versions at batch 32.",
                "For <code>tiny_bert</code> the safetensors body is larger than the JSON body, because the inputs are short integer lists. The rows per second are still higher with safetensors.",
            ]
        )
    )
    body.append(h3("Does not show"))
    body.append(ul(["A real client. The harness encodes the body once and sends the same bytes every time.", "Batch sizes above 32, or batches of different row counts.", "Dynamic request batching. downshift has none; one request makes one inference.", common_limits()]))
    body.append(section_sources(R1 / "chart_runs.csv", R1 / "REPORT.md", BENCH / "cases.py", BENCH / "run.py"))
    return frame("wire-formats", "Batches and wire formats", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: result cache
# ----------------------------------------------------------------------------------------


def page_cache() -> str:
    acc = jload(R6 / "cache_acceptance.json")
    a, b, c = acc["a_repeated_query_server_ms"], acc["b_batch_32_with_30_known"], acc["c_unique_queries_server_ms"]
    claim(all(acc["accepted"].values()), "all three acceptance checks passed")
    ratio = a["no_cache_median"] / a["hit_median"]
    over = (c["cache_on_median"] / c["cache_off_median"] - 1) * 100
    body = [header("The result cache", "How much faster is a request that the cache has seen before, and what does the cache cost a request that it has not seen?", f"Two data sets. A load run of 1.0.0rc1 ({esc(env1()['timestamp'])}) and an in-process check of 0.6.0 ({esc(acc['when'])}).")]
    mini = lambda srv, pay, c_: hfrow(srv, pay, c_)  # noqa: E731
    body.append(
        tiles(
            [
                (f"{a['hit_median']} vs {a['no_cache_median']} ms", "MiniLM, a query seen before: server time, cache hit vs no cache", f"median of {acc['repeats']} repeats (v0.6.0 check)"),
                (f"{n2(over)}%", "MiniLM, unique queries: median server time with cache on, against cache off", f"{c['cache_on_median']} vs {c['cache_off_median']} ms"),
                (f"{n0(mini(DS_CACHE, 'short_b1', 8)['throughput_rps'])} vs {n0(mini(DS_NEW, 'short_b1', 8)['throughput_rps'])}", "MiniLM, one repeated sentence, c=8, rps: cache on vs off", "Best case: every request is a hit"),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("From 1.0.0, downshift can keep the output of a row and answer a repeated row from memory. It does this only if the row check found the rows of the model independent. This page shows the gain for a repeated request and the cost for a new request."))
    body.append(h2("Method", "method"))
    body.append(
        ul(
            [
                "<b>Load run (1.0.0rc1).</b> The variant <code>downshift_cache</code> of <code>bench.run</code> and <code>bench.hf_models</code> uses the default cache. All other 1.0.0rc1 variants add <code>--result-cache-rows 0</code>. The harness sends the same body every time. So after the first request, every request is a hit. This is the best case.",
                "<b>Acceptance check (0.6.0).</b> <code>python -m bench.cache_accept --out bench/results/v0.6.0/cache_acceptance.json</code>. It serves all-MiniLM-L6-v2 in the same process with a test client, with the cache on and off. It reads the sum of the stages in the <code>Server-Timing</code> header, so the client time does not count. It is not part of CI.",
                "The check has three parts: (a) a query seen before must take under 1 ms; (b) a batch of 32 rows with 30 known rows must run the model on 2 rows; (c) unique queries must cost no more than 5% plus 0.2 ms over a server with the cache off.",
            ]
        )
    )
    body.append(h2("Results: load run", "load"))
    rows = []
    for m in FIX_MODELS:
        off, on = peak_of(m, DS_NEW), peak_of(m, DS_CACHE)
        o1, n1_ = at(m, DS_NEW, 1), at(m, DS_CACHE, 1)
        rows.append([m, n0(off["throughput_rps"]), n0(on["throughput_rps"]), pct((on["throughput_rps"] / off["throughput_rps"] - 1) * 100), n2(o1["p50_ms"]), n2(n1_["p50_ms"])])
    hdr = ["model", "peak rps cache off", "peak rps cache on", "change", "p50 ms c=1 off", "p50 ms c=1 on"]
    body.append(table(hdr, rows, caption="Fixtures, batch 1: 1.0.0rc1 with the cache off and on (same body in every request)", right=range(1, 6)))
    mdt("Result cache on the fixtures, batch 1", "1.0.0rc1, cache off against cache on. Every request repeats one body, so this is a best case. The mlp_large off row is from the slow main pass.", [R1 / "chart_runs.csv"], hdr, rows)
    items = []
    for pay, nm in [("short_b1", "one sentence"), ("short_b8", "eight sentences"), ("long_b1", "about 180 tokens")]:
        items.append(("h", f"MiniLM, {nm}, requests/s at c=8"))
        items.append(("cache off", mini(DS_NEW, pay, 8)["throughput_rps"], "bw"))
        items.append(("cache on", mini(DS_CACHE, pay, 8)["throughput_rps"], "bv"))
    svg = hbar(items, n0, title="MiniLM requests per second at c=8, cache off and on", desc="Three payloads, two bars each. Values are in the table.", lw=90)
    body.append(figure(svg, "all-MiniLM-L6-v2, c=8. Every request repeats one body.", bar_legend([("bw", "cache off"), ("bv", "cache on")])))
    hdr2 = ["payload", "rps c=1 off", "rps c=1 on", "rps c=8 off", "rps c=8 on", "p50 ms c=1 off", "p50 ms c=1 on", "worst /health p99 ms off", "worst /health p99 ms on"]
    rows2 = []
    for pay in ("short_b1", "short_b8", "long_b1"):
        def hp(srv):
            return max(mini(srv, pay, cc)["health_p99_ms"] for cc in (1, 8))
        rows2.append([pay] + [n0(mini(s, pay, cc)["throughput_rps"]) for cc in (1, 8) for s in (DS_NEW, DS_CACHE)] + [n1(mini(s, pay, 1)["p50_ms"]) for s in (DS_NEW, DS_CACHE)] + [n0(hp(DS_NEW)), n0(hp(DS_CACHE))])
    body.append(table(hdr2, rows2, caption="all-MiniLM-L6-v2: cache off against cache on", right=range(1, 9)))
    mdt("Result cache on MiniLM", "1.0.0rc1, 5 s windows. Every request repeats one body.", [R1 / "chart_runs.csv"], hdr2, rows2)
    claim(mini(DS_CACHE, "short_b1", 8)["health_p99_ms"] > 5 * mini(DS_NEW, "short_b1", 8)["health_p99_ms"], "cache-on health p99 is far above cache-off")
    body.append(p("With the cache on, the server answers up to about 1900 requests per second (one sentence, c=8). The <code>/health</code> probe, that runs every 50 ms in the same window, sees a higher p99 in the cache-on run than in the cache-off run. The cause is not in the data. The load generator and the server share the machine."))
    body.append(h3("Where the cache does nothing"))
    body.append(p("<code>tiny_bert</code> has dependent rows and <code>scatter_include_self_false</code> has no row check (the numerics check failed). The default cache is off for both. Their cache-on rows are an ordinary run. They differ from the default rows only by run-to-run noise, as the report says. The difference for <code>tiny_bert</code> in the table is a "
              f"{pct((peak_of('tiny_bert', DS_CACHE)['throughput_rps'] / peak_of('tiny_bert', DS_NEW)['throughput_rps'] - 1) * 100)} change in peak. The size of the noise is not measured, because the file holds no repeats."))

    body.append(h2("Results: acceptance check (0.6.0)", "accept"))
    hdr3 = ["check", "measured", "rule", "result"]
    rows3 = [
        ["a. query seen before: server time", f"hit median {a['hit_median']} ms (max {a['hit_max']} ms); no cache {a['no_cache_median']} ms", "hit median under 1 ms", "pass" if acc["accepted"]["a"] else "fail"],
        ["b. batch of 32 with 30 known rows", f"rows in the model call: {b['rows_in_the_model_call']}", "model runs on 2 rows", "pass" if acc["accepted"]["b"] else "fail"],
        ["c. unique queries: server time", f"cache on {c['cache_on_median']} ms; cache off {c['cache_off_median']} ms ({pct(over)})", "on <= 1.05 x off + 0.2 ms", "pass" if acc["accepted"]["c"] else "fail"],
    ]
    body.append(table(hdr3, rows3, caption=f"Cache acceptance, {acc['model']}, {acc['repeats']} repeats, {acc['when']}"))
    mdt("Result cache acceptance check (0.6.0)", f"{acc['model']}, {acc['repeats']} repeats, run {acc['when']}. In-process test client, server-side time from Server-Timing.", [R6 / "cache_acceptance.json"], hdr3, rows3)
    body.append(p(f"A hit takes {ratio:.0f} times less server time than a miss ({a['hit_median']} against {a['no_cache_median']} ms). A request that always misses takes {c['cache_on_median']} ms with the cache on and {c['cache_off_median']} ms with it off."))

    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    claim(0.2 < a["hit_median"] < 0.3 and 4.5 < a["no_cache_median"] < 4.7, "hit about a quarter ms; miss about 4.6 ms")
    claim(mini(DS_CACHE, "short_b1", 8)["throughput_rps"] > 1900, "cache-on MiniLM above 1900 rps")
    body.append(ul(["A repeated query is answered in about a quarter of a millisecond of server time, against about 4.6 ms for the model.", "On a batch with known rows, the model runs on the new rows only (2 of 32).", f"A new query costs {c['cache_on_median'] - c['cache_off_median']:.3f} ms more at the median ({over:.1f}%) in the one check that was run."]))
    claim(abs((c["cache_on_median"] - c["cache_off_median"]) - 0.175) < 1e-9, "unique-query overhead is 0.175 ms")
    body.append(h3("Does not show"))
    body.append(
        ul(
            [
                "A normal workload. The load run repeats one body. A real hit rate is not measured.",
                "The hardware of the 0.6.0 check. The file stores no machine, CPU or package version. It also stores no spread: only medians, one maximum and the repeat count.",
                "The check ran on 0.6.0, not on 1.0.0rc1. It was not run again for 1.0.0.",
                "The timing difference between a hit and a miss is visible to a client. The README says that on a server with more than one tenant, a client can see the difference. This is a property of the design, and the data does not measure it.",
                "Only MiniLM was checked in-process. The cache for graph models is not measured.",
            ]
        )
    )
    body.append(section_sources(R1 / "chart_runs.csv", R6 / "cache_acceptance.json", BENCH / "cache_accept.py", R1 / "REPORT.md"))
    return frame("result-cache", "The result cache", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: correctness
# ----------------------------------------------------------------------------------------


def worst_err(model, server):
    vals = [r["max_abs_err"] for r in runs() if r["suite"] == "fixture" and r["model"] == model and r["server"] == server and r["max_abs_err"] is not None]
    return max(vals)


def page_correct() -> str:
    res = jload(R1 / "results.json")
    hf = jload(R1 / "results_hf.json")
    ver = res["verdicts"]
    total_err = sum(r["errors"] for r in res["runs"]) + sum(x["errors"] for v in hf["models"]["all-MiniLM-L6-v2"]["servers"].values() for x in v["runs"])
    claim(total_err == 0, "zero failed requests")
    sc = "scatter_include_self_false"
    e_naive, e_ds = worst_err(sc, NAIVE_O), worst_err(sc, DS_NEW)
    claim(e_naive > 1.0 and e_ds < 1e-6, "scatter errors: naive ORT wrong, downshift right")
    body = [header("Correctness under load", "Do the servers return the right numbers? Every response in the load run was compared with eager PyTorch.", run_meta())]
    body.append(
        tiles(
            [
                (f"{sci(e_naive)} vs {sci(e_ds)}", "scatter_include_self_false: worst error, hand-rolled ORT vs downshift 1.0.0rc1", "downshift serves this model with torch"),
                (f"{total_err}", "failed requests across every load window", "fixtures and MiniLM"),
                ("DEGRADED", "verdict for scatter_include_self_false", ver[sc]["reason"]),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("A fast server with a wrong answer is not a win. The harness checks the answer of every server for every model, so a wrong row shows up next to its speed."))
    body.append(h2("Method", "method"))
    body.append(ul(["All processes build the same model from seed 1234 (<code>bench/cases.py</code>). The reference output is eager PyTorch.", "<code>max_abs_err</code> is the worst element-wise absolute error against the reference. The table shows the worst value over all rows of a server (every batch and concurrency).", "The verdict is made by <code>downshift check</code> logic at the start of each downshift server. Tolerance: atol 1e-4, rtol 1e-3 for float32. A sample passes when <code>abs(got - expected) &lt;= atol + rtol * abs(expected)</code> (<code>docs/cli.md</code>).", "For MiniLM, every answer is compared with <code>downshift --backend torch</code>, and the file stores the minimum cosine similarity."]))
    body.append(h2("Results", "results"))
    hdr = ["model", "family", "verdict", "backend", "max abs error at check", "reason"]
    rows = [[esc(m), esc(v["family"]), chip(v["status"]), esc(v["backend"]), sci(v["max_abs_err"]), esc(v["reason"])] for m, v in ver.items()]
    body.append(table(hdr, rows, caption="Verdicts of the four models (the check at server start)", raw=True, right=(4,)))
    mdt("Verdicts of the load-run models", "From the gate at the start of each downshift server.", [R1 / "results.json"], hdr, [[m, v["family"], v["status"], v["backend"], sci(v["max_abs_err"]), v["reason"]] for m, v in ver.items()])

    items = []
    for s in [NAIVE_T, NAIVE_O, NAIVE_A, DS_OLD, DS_NEW]:
        items.append((SHORT[s], worst_err(sc, s), bar_class(s)))
    svg = hbar(items, sci, title="Worst error on scatter_include_self_false", desc="Linear bars. Only the hand-rolled ONNX Runtime server has a large error.", lw=132, vmax=1.5)
    body.append(figure(svg, "scatter_include_self_false: worst absolute error against eager PyTorch (linear scale, 0 to 1.5). Only the bar for FastAPI + ORT is visible.", bar_legend([("bn", "hand-rolled"), ("bo", "downshift 0.5.1"), ("bw", "downshift 1.0.0rc1")])))
    hdr2 = ["server"] + FIX_MODELS
    rows2 = [[s] + [sci(worst_err(m, s)) for m in FIX_MODELS] for s in FIX_SERVERS]
    body.append(table(hdr2, rows2, caption="Worst max abs error against eager PyTorch over all rows of a server", right=range(1, 5)))
    mdt("Worst max abs error against eager PyTorch", "Over every row a server answered.", [R1 / "chart_runs.csv"], hdr2, rows2)
    claim(all(worst_err(m, s) < 1e-5 for m in FIX_MODELS[:3] for s in FIX_SERVERS), "all CLEAN models are below 1e-5 on every server")
    body.append(p("On the three CLEAN models, every server stays below 1e-5. On the DEGRADED model, the hand-rolled ONNX Runtime server is off by about 1.2. downshift runs the gate, finds 6 of 8 samples wrong, and serves torch. The gate output for this model is shown in the repo README."))
    hfm = hf["models"]["all-MiniLM-L6-v2"]
    hdr3 = ["server", "max abs err", "min cosine"]
    rows3 = [[k, sci(v["max_abs_err"]), f"{v['min_cosine']:.7f}"] for k, v in hfm["agreement"].items()]
    body.append(table(hdr3, rows3, caption=f"MiniLM, short_b8: agreement with {hfm['reference']}", right=(1, 2)))
    mdt("MiniLM agreement with downshift --backend torch", f"short_b8 output, reference {hfm['reference']}.", [R1 / "results_hf.json"], hdr3, rows3)
    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(ul(["The gate prevented a wrong answer on the one DEGRADED model in the run, and the cost in speed is the torch speed for that model.", "On the CLEAN models, the ORT answers differ from torch by less than 1e-5 under load.", "No request failed."]))
    body.append(h3("Does not show"))
    body.append(ul(["How often a real model is DEGRADED. The run has one DEGRADED model, and the authors chose it to show the hazard. The compatibility page lists more.", "The error on inputs that the harness did not send. Each row of the table compares one request of that shape.", "Weights are random and seeded, not trained. Real trained models are in the compatibility page.", common_limits()]))
    body.append(section_sources(R1 / "results.json", R1 / "chart_runs.csv", R1 / "results_hf.json", BENCH / "cases.py"))
    return frame("correctness", "Correctness", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: compatibility
# ----------------------------------------------------------------------------------------


def parse_matrix():
    text = need(DOCS_COMPAT).read_text(encoding="utf-8")
    head = text.split("<!-- real-models:start -->")[0]
    gen = re.search(r"Generated (\S+) by `scripts/gen_matrix.py` with (.+?)\.\n", head)
    rows = []
    for line in head.splitlines():
        if line.startswith("| `"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append([c.strip("`") for c in cells])
    real_part = text.split("<!-- real-models:start -->")[1]
    gen2 = re.search(r"Generated (\S+) by `scripts/gen_matrix.py` with (.+?)\.\n", real_part)
    return rows, (gen.groups() if gen else ("?", "?")), (gen2.groups() if gen2 else ("?", "?"))


def page_compat() -> str:
    real = jload(R1 / "real_models.json")
    mrows, gen_fix, gen_real = parse_matrix()
    models = real["models"]
    claim(all(m["verdict"]["status"] == "CLEAN" for m in models), "all real models are CLEAN")
    claim(len(models) == 8, "eight real models")
    worst = max(m["verdict"]["numerics"]["max_abs_err"] for m in models)
    counts = defaultdict(int)
    for r in mrows:
        st = r[3]
        counts["skipped" if st.startswith("skipped") else st] += 1
    body = [header("What the gate catches", "Which models pass the gate, and which export hazards does it find? Two sets: 14 small fixtures that each isolate one hazard, and 8 real public models.", f"Real models: <code>real_models.json</code>, date {esc(real['date'])}. Fixture table: <code>docs/compatibility.md</code>, generated {esc(gen_fix[0])}.")]
    body.append(
        tiles(
            [
                ("8 of 8", "real models with verdict CLEAN", f"max abs error {sci(worst)} (limit: atol 1e-4)"),
                (f"{counts['CLEAN']} / {counts['FAILED']} / {counts['DEGRADED']}", "fixtures: CLEAN / FAILED / DEGRADED", f"plus {counts['skipped']} skipped (broken_factory)"),
                ("1 of 14", "fixtures with silent wrong numbers (DEGRADED)", "scatter_include_self_false"),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("Does the verdict agree with what is true about a model? A hazard fixture has a known right answer. A real model shows whether the gate passes models that people use."))
    body.append(h2("Method", "method"))
    body.append(
        ul(
            [
                "<b>Fixtures.</b> <code>python scripts/gen_matrix.py</code> runs <code>downshift.check(..., samples=8)</code> on each model in <code>tests/models/</code>. It writes the table in <code>docs/compatibility.md</code>. This page reads that table.",
                "<b>Real models.</b> <code>python scripts/check_real_models.py</code> runs <code>downshift check --json</code> on each model in its own subprocess, on CPU. It writes <code>bench/results/v1.0.0/real_models.json</code> and the second table of <code>docs/compatibility.md</code>. Models: Hugging Face encoders, torchvision ResNet-18 and a PyG GCN.",
                "Verification: 8 samples. Some have shapes that the exporter did not see. Tolerance atol 1e-4, rtol 1e-3 (float32). Embedding models also need a minimum cosine of 0.9999.",
                "The hub models were read from local copies. downshift never contacts the hub.",
            ]
        )
    )
    body.append(h2("Results: 8 real models", "real"))
    names = {"distilbert/distilbert-base-uncased-finetuned-sst-2-english": "distilbert-sst-2", "pyg/GCN (tests/models/gnn_gcn)": "pyg GCN (fixture)"}
    short = lambda i: names.get(i, i.split("/")[-1])  # noqa: E731
    svg = dotplot_log([(short(m["id"]), m["verdict"]["numerics"]["max_abs_err"]) for m in models], title="Max abs error of eight real models against PyTorch", desc="One dot for each model on a log scale. The dashed line is the absolute tolerance 1e-4.", vline=1e-4, vline_label="atol 1e-4", lo=1e-8, hi=1e-3, lw=150)
    body.append(figure(svg, "Largest absolute difference between ONNX Runtime and PyTorch over 8 samples. Log scale. All dots are left of the tolerance line.", ""))
    hdr = ["model", "family", "verdict", "max abs err", "max rel err", "min cosine", "samples", "sampled batch / seq", "check s"]
    rows, rows_md = [], []
    for m in models:
        v = m["verdict"]
        n = v["numerics"]
        ax = v["axes"]
        samp = " / ".join(f"{a['sampled_min']}-{a['sampled_max']}" for a in ax[:2])
        serve = " / ".join(f"{a['served_min']}-{a['served_max']}" for a in ax[:2])
        cos = f"{n['min_cosine']:.6f}" if n.get("min_cosine") is not None else "-"
        rows.append([esc(m["id"]), esc(v["model_family"]), chip(v["status"]), sci(n["max_abs_err"]), sci(n["max_rel_err"]), cos, n["samples_tested"], f"{samp} (serves {serve})", m["seconds"]])
        rows_md.append([m["id"], v["model_family"], v["status"], sci(n["max_abs_err"]), sci(n["max_rel_err"]), cos, n["samples_tested"], f"{samp} (serves {serve})", m["seconds"]])
    body.append(table(hdr, rows, caption="Real models: verdict, numerics and shape coverage", raw=True, right=(3, 4, 5, 6, 8)))
    mdt("Real models checked with downshift check", f"CPU, one subprocess for each model. Date {real['date']}. For graph models the two axes are nodes / edges.", [R1 / "real_models.json"], hdr, rows_md)
    body.append(note("info", "The pass rule combines atol and rtol. The file also stores the max relative error. It is large for some models (for example 5.9e-03 for MiniLM) because the relative error is high where the expected value is near 0. Those samples pass through the absolute term."))
    gcn = [m for m in models if m["id"].startswith("pyg")][0]["verdict"]
    body.append(p(f"One detail: the PyG GCN has <code>row_independent = {gcn['row_independent']}</code>. The reason in the file: &quot;{esc(gcn['row_reason'])}&quot;. A graph does not have independent batch rows, so the result cache does not apply to it."))
    body.append(h2("Results: 14 hazard fixtures", "fixtures"))
    hdr2 = ["fixture", "hazard", "family", "export", "capture", "numerics", "shape-general", "backend"]
    rows2 = []
    for r in mrows:
        st = r[3]
        cell = chip(st) if st in ("CLEAN", "DEGRADED", "FAILED", "UNVERIFIED") else esc(st)
        rows2.append([esc(r[0]), esc(r[1]), esc(r[2]), cell] + [esc(x) for x in r[4:]])
    body.append(table(hdr2, rows2, caption=f"Fixtures from tests/models (generated {gen_fix[0]})", raw=True))
    mdt("Hazard fixtures", f"Copied from docs/compatibility.md, generated {gen_fix[0]} with {gen_fix[1]}.", [DOCS_COMPAT], hdr2, mrows)
    body.append(
        ul(
            [
                "<code>scatter_include_self_false</code> exports without an error and gives wrong numbers. Only the numerics check stops it.",
                "<code>bf16_weights</code> exports, but ONNX Runtime on CPU has no bfloat16 Gemm kernel. The verdict is FAILED and torch serves it.",
                "<code>data_dependent_branch</code> does not export. The verdict is FAILED.",
                "<code>custom_autograd</code> is CLEAN. <code>torch.export</code> traces through the function.",
            ]
        )
    )
    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(ul(["The gate passes 8 of 8 models that people use, with errors from 1.2e-07 to 6.0e-05.", "The gate finds the three kinds of hazard in the fixtures: wrong numbers, no kernel, no export.", "The check takes 10.6 to 39.0 s for each model on this machine (<code>seconds</code> in the file)."]))
    claim(min(m["seconds"] for m in models) == 10.6 and max(m["seconds"] for m in models) == 39.0, "check seconds range 10.6 to 39.0")
    claim(min(m["verdict"]["numerics"]["max_abs_err"] for m in models) < 1.3e-7 and worst < 6.1e-5, "error range 1.2e-07 to 6.0e-05")
    body.append(h3("Does not show"))
    body.append(
        ul(
            [
                "A false-negative rate. A model that is wrong only on an input that none of the 8 samples hit would pass. The sampled range is small: batch 1 to 4 for the encoders, while the server accepts batch 1 to 4096 (the file marks this as not verified above the sampled range).",
                "Any other model family or size. Qwen3-Embedding-0.6B is in <code>scripts/check_real_models.py</code> behind <code>--large</code>. It is not in the file.",
                "Speed. These runs check the numbers. They do not load-test the real models (only MiniLM has a load run).",
                "The machine. <code>real_models.json</code> stores the date and the package versions. It does not store the operating system or the CPU.",
                "Only CPU. The GPU path is not measured.",
            ]
        )
    )
    mism = []
    if gen_real[0] != real["date"]:
        mism.append(f"The date of <code>real_models.json</code> is {esc(real['date'])}. The second table of <code>docs/compatibility.md</code> says it was generated on {esc(gen_real[0])}.")
    v = real["versions"]
    if f"torch {v['torch']}+" not in gen_real[1] and f"torch {v['torch']}," not in gen_real[1]:
        mism.append(f"The file lists torch {esc(v['torch'])} and onnxruntime {esc(v['onnxruntime'])}. The doc line lists: {esc(gen_real[1])}.")
    if mism:
        body.append(note("bad", " ".join(mism) + " The numbers in both tables agree. The version text is stale in one of the two files. Not changed here.", "Source mismatch."))
    body.append(section_sources(R1 / "real_models.json", DOCS_COMPAT, REPO / "scripts" / "check_real_models.py", REPO / "scripts" / "gen_matrix.py"))
    return frame("compatibility", "Compatibility", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: boot time
# ----------------------------------------------------------------------------------------


def page_boot() -> str:
    hf = jload(R1 / "results_hf.json")
    chk = hf["models"]["all-MiniLM-L6-v2"]["check"]
    ready = {}
    for r in runs():
        ready[(r["model"], r["server"])] = r["ready_s"]
    m = "all-MiniLM-L6-v2"
    claim(ready[(m, DS_NEW)] > ready[(m, "naive FastAPI + transformers")], "downshift boots slower than naive on MiniLM")
    claim(ready[(m, "downshift 1.0.0rc1 --backend torch")] < ready[(m, DS_NEW)], "--backend torch boots faster")
    gaps = [ready[(mo, DS_NEW)] - ready[(mo, NAIVE_T)] for mo in FIX_MODELS]
    claim(0 < min(gaps) and max(gaps) < 6, "fixture boot gap is positive and under 6 s")
    body = [header("Boot time: what the gate costs before the first request", "How many seconds from process start to a ready server?", run_meta())]
    body.append(
        tiles(
            [
                (f"{n1(ready[(m, DS_NEW)])} vs {n1(ready[(m, 'naive FastAPI + transformers')])} s", "MiniLM: downshift 1.0.0rc1 vs hand-rolled FastAPI + transformers", "time to the first 200 answer"),
                (f"{n1(ready[(m, 'downshift 1.0.0rc1 --backend torch')])} s", "MiniLM: downshift with --backend torch", "same server, ORT backend not used"),
                (f"{n1(ready[('clean_mlp', DS_NEW)])} vs {n1(ready[('clean_mlp', NAIVE_T)])} s", "clean_mlp: downshift vs hand-rolled torch", "a tiny model"),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("The gate runs before the first request. It exports the model and runs 8 samples. This work takes time. This page shows how much, as the harness measured it."))
    body.append(h2("Method", "method"))
    body.append(
        ul(
            [
                "<code>ready_s</code> is the time from the start of the server process to the first 200 answer. downshift answers on <code>/ready</code>. The hand-rolled servers answer on <code>/health</code> (<code>bench/hf_models.py</code>). The ends are not the same event.",
                "Each value is one measurement for one start. The CSV repeats it on every row of a server. The file has no repeats.",
                "<code>wall_s</code> of <code>downshift check</code> on MiniLM comes from <code>results_hf.json</code>.",
            ]
        )
    )
    body.append(h2("Results", "results"))
    for mod in FIX_MODELS + [m]:
        servers = [s for s in FIX_SERVERS + ["naive FastAPI + transformers", "naive transformers, async def", "downshift 0.5.1 --backend torch", "downshift 1.0.0rc1 --backend torch"] if (mod, s) in ready and s not in (DS_OLD_ST, DS_NEW_ST, DS_CACHE, DS_METRICS)]
        items = [("h", mod)] + [(SHORT[s], ready[(mod, s)], bar_class(s)) for s in servers]
        svg = hbar(items, lambda v: f"{v:.1f} s", title=f"Seconds to ready, {mod}", desc="One bar for each server.", lw=140)
        body.append(figure(svg, f"{mod}: seconds from process start to ready. Shorter is better.", ""))
    hdr = ["model"] + [SHORT[s] for s in [NAIVE_T, NAIVE_O, NAIVE_A, DS_OLD, DS_NEW]]
    rows = [[mod] + [n1(ready[(mod, s)]) if (mod, s) in ready else "-" for s in [NAIVE_T, NAIVE_O, NAIVE_A, DS_OLD, DS_NEW]] for mod in FIX_MODELS]
    body.append(table(hdr, rows, caption="Seconds to ready, fixtures", right=range(1, 6)))
    mdt("Boot time, fixtures (s)", "Time from process start to first 200 answer. One start for each cell.", [R1 / "chart_runs.csv"], hdr, rows)
    hdr2 = ["server (all-MiniLM-L6-v2)", "ready s"]
    rows2 = [[s, n1(ready[(m, s)])] for s in ["naive FastAPI + transformers", "naive transformers, async def", "downshift 0.5.1", "downshift 0.5.1 --backend torch", DS_NEW, "downshift 1.0.0rc1 --backend torch"]]
    body.append(table(hdr2, rows2, caption="Seconds to ready, MiniLM", right=(1,)))
    mdt("Boot time, MiniLM (s)", "Time from process start to first 200 answer.", [R1 / "chart_runs.csv"], hdr2, rows2)
    body.append(p(f"<code>downshift check</code> on MiniLM took {chk['1.0.0rc1']['wall_s']} s for 1.0.0rc1 and {chk['0.5.1']['wall_s']} s for 0.5.1 (exit code 0 in both)."))
    mdt("downshift check wall time, MiniLM", "From results_hf.json.", [R1 / "results_hf.json"], ["version", "wall s", "exit code"], [[k, v["wall_s"], v["exit_code"]] for k, v in chk.items()])
    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(ul([f"On MiniLM, downshift is ready in {n1(ready[(m, DS_NEW)])} s. The hand-rolled server is ready in {n1(ready[(m, 'naive FastAPI + transformers')])} s. The difference is {n1(ready[(m, DS_NEW)] - ready[(m, 'naive FastAPI + transformers')])} s.", f"With <code>--backend torch</code>, downshift is ready in {n1(ready[(m, 'downshift 1.0.0rc1 --backend torch')])} s.", "On the four fixtures, downshift takes " + f"{min(gaps):.1f} to {max(gaps):.1f}" + " s more than the hand-rolled torch server."]))
    body.append(h3("Does not show"))
    body.append(ul(["How the boot time divides between load, export, verify and warm-up. The stored <code>check</code> report has no timing split. The README example shows such a split for one model.", f"Whether the {n1(ready[(m, DS_NEW)] - ready[(m, 'downshift 1.0.0rc1 --backend torch')])} s gap between the default and <code>--backend torch</code> is the export and the check alone. The data does not say.", "The effect of the export cache (<code>--export-cache-dir</code>). It is not measured here.", "The boot time on Linux, or on another CPU. " + common_limits()]))
    body.append(section_sources(R1 / "chart_runs.csv", R1 / "results_hf.json", BENCH / "hf_models.py", BENCH / "run.py"))
    return frame("boot-time", "Boot time", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: workers (0.5.0 run)
# ----------------------------------------------------------------------------------------


def page_scaling() -> str:
    env5 = jload(R5 / "results.json")["environment"]
    ceil5 = {int(k): v for k, v in jload(R5 / "results.json")["client_ceiling_rps"].items()}
    w = [r for r in runs5() if r["suite"] == "workers"]

    def g(model, ver, nw, c):
        rows = [r for r in w if r["model"] == model and r["server"] == f"downshift {ver} --workers {nw}" and r["concurrency"] == c]
        return rows[0]

    models = ["clean_mlp", "mlp_large", "bert_small"]
    claim(g("clean_mlp", "0.5.0", 4, 8)["throughput_rps"] > 3.5 * g("clean_mlp", "0.5.0", 1, 8)["throughput_rps"], "clean_mlp 4 workers more than 3.5x")
    claim(g("bert_small", "0.5.0", 4, 1)["throughput_rps"] < g("bert_small", "0.5.0", 1, 1)["throughput_rps"], "bert_small 4 workers is slower at c=1")
    body = [header("More worker processes", "Does <code>--workers N</code> raise throughput, and what does it cost?", f"Older data: the 0.5.0 run, {esc(env5['timestamp'])}. {esc(env5['platform'])}, {env5['cpu_count']} logical CPUs, Python {esc(env5['python'])}, torch {esc(env5['torch'])}, onnxruntime {esc(env5['onnxruntime'])}.")]
    body.append(note("bad", "The 1.0.0rc1 run did not include the workers sweep (<code>skip_workers</code> is true in its settings). These numbers are for 0.5.0 and 0.4.0. They were not measured for 1.0.0rc1.", "Old data."))
    a, b = g("clean_mlp", "0.5.0", 1, 8), g("clean_mlp", "0.5.0", 4, 8)
    body.append(
        tiles(
            [
                (f"{n0(b['throughput_rps'])} vs {n0(a['throughput_rps'])}", "clean_mlp, c=8, 0.5.0: 4 workers vs 1 worker, requests/s", f"{b['throughput_rps'] / a['throughput_rps']:.1f}x"),
                (f"{n0(g('bert_small', '0.5.0', 4, 8)['throughput_rps'])} vs {n0(g('bert_small', '0.5.0', 1, 8)['throughput_rps'])}", "bert_small, c=8, 0.5.0: 4 workers vs 1 worker", f"{g('bert_small', '0.5.0', 4, 8)['throughput_rps'] / g('bert_small', '0.5.0', 1, 8)['throughput_rps']:.1f}x"),
                (f"{n0(g('bert_small', '0.5.0', 4, 1)['throughput_rps'])} vs {n0(g('bert_small', '0.5.0', 1, 1)['throughput_rps'])}", "bert_small, c=1, 0.5.0: 4 workers vs 1 worker", "slower with more workers at c=1"),
            ]
        )
    )
    body.append(h2("The question", "question"))
    body.append(p("One downshift process runs inference on one machine. <code>--workers N</code> starts N processes. This page shows what that does to throughput at 1, 8 and 32 concurrent clients."))
    body.append(h2("Method", "method"))
    body.append(ul(["The workers sweep is the last part of <code>python -m bench.run --out bench/results/v0.5.0/results.json</code>. It starts the real CLI with <code>--workers 1</code>, <code>2</code> and <code>4</code> on <code>clean_mlp</code>, <code>mlp_large</code> and <code>bert_small</code>. Batch 1, JSON.", "With more than one worker, the parent exports once and hands the artifact to each worker.", "Data: <code>chart_runs.csv</code> of the 0.5.0 run, rows with <code>suite = workers</code>."]))
    body.append(h2("Results", "results"))
    for mod in models:
        items = []
        for c in (1, 8, 32):
            items.append(("h", f"{mod}, c={c}, requests/s (0.5.0)"))
            for nw in (1, 2, 4):
                items.append((f"{nw} worker" + ("s" if nw > 1 else ""), g(mod, "0.5.0", nw, c)["throughput_rps"], "bw"))
        svg = hbar(items, n0, title=f"Requests per second for 1, 2 and 4 workers, {mod}", desc="Three groups, one for each concurrency.", lw=90)
        body.append(figure(svg, f"{mod}, downshift 0.5.0: requests per second by number of workers.", ""))
    hdr = ["model", "version", "workers", "rps c=1", "rps c=8", "rps c=32", "p99 ms c=8", "boot s"]
    rows = []
    for mod in models:
        for ver in ("0.4.0", "0.5.0"):
            for nw in (1, 2, 4):
                rows.append([mod, ver, nw, n0(g(mod, ver, nw, 1)["throughput_rps"]), n0(g(mod, ver, nw, 8)["throughput_rps"]), n0(g(mod, ver, nw, 32)["throughput_rps"]), n1(g(mod, ver, nw, 8)["p99_ms"]), n1(g(mod, ver, nw, 1)["ready_s"])])
    body.append(table(hdr, rows, caption="The workers sweep, batch 1", right=range(2, 8)))
    mdt("Workers sweep (0.4.0 and 0.5.0 run)", "Requests/s at c=1, 8 and 32; p99 at c=8; seconds to ready. Not measured for 1.0.0rc1.", [R5 / "chart_runs.csv"], hdr, rows)
    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(ul(["With 8 or more clients, throughput rises with the number of workers for all three models in both versions.", "At c=1, more workers do not help. For <code>bert_small</code> 4 workers are slower than 1 (70 against 91 requests/s in 0.5.0).", "Boot time grows with workers: it is longer with 4 workers than with 1 for every model."]))
    claim(round(g("bert_small", "0.5.0", 4, 1)["throughput_rps"]) == 70 and round(g("bert_small", "0.5.0", 1, 1)["throughput_rps"]) == 91, "bert_small c=1 numbers 70 and 91")
    claim(all(g(mo, "0.5.0", 4, 1)["ready_s"] > g(mo, "0.5.0", 1, 1)["ready_s"] for mo in models), "boot is longer with 4 workers")
    body.append(h3("Does not show"))
    body.append(ul([f"Linear scaling. The best case is {b['throughput_rps'] / a['throughput_rps']:.1f}x for 4 workers. The load generator shares the {env5['cpu_count']} logical CPUs with the workers, and its ceiling in this run was {n0(ceil5[8])} requests/s at c=8.", "Results for 1.0.0rc1, for any other model, or for a machine with another core count.", "The README of <code>bench/</code> notes one startup hang in 0.4.0 (<code>--workers 4</code> on <code>bert_small</code> did not answer <code>/ready</code> within 600 s once, then was re-run). The stored boot times are of the re-run.", common_limits()]))
    body.append(section_sources(R5 / "chart_runs.csv", R5 / "results.json", R5 / "REPORT.md"))
    return frame("scaling", "Workers", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: releases
# ----------------------------------------------------------------------------------------


def page_releases() -> str:
    r5 = runs5()
    env5 = jload(R5 / "results.json")["environment"]
    env1_ = env1()
    models5 = ["clean_mlp", "dynamic_batch_cnn", "gnn_gcn", "tiny_bert", "scatter_include_self_false", "mlp_large", "cnn_large", "bert_small"]

    def pk5(m, s):
        rows = [r for r in r5 if r["suite"] == "fixture" and r["model"] == m and r["server"] == s and r["batch"] == 1]
        return peak(rows)

    d = {m: pk5(m, "downshift 0.5.0")["throughput_rps"] / pk5(m, "downshift 0.4.0")["throughput_rps"] - 1 for m in models5}
    claim(all(v > 0 for v in d.values()), "0.5.0 is faster than 0.4.0 at peak on all 8 models in its own run")
    body = [header("Release to release", "Did the peak throughput change between releases? Each comparison is inside one run, so the machine state is the same for both versions.", f"Run A: {esc(env5['timestamp'])} (0.4.0 and 0.5.0). Run B: {esc(env1_['timestamp'])} (0.5.1 and 1.0.0rc1). Both on {esc(env1_['platform'])}, {env1_['cpu_count']} logical CPUs.")]
    body.append(note("bad", "Do not compare numbers between run A and run B. The hand-rolled servers are faster in run B too, so the machine state is not the same. The reports say this. Compare the versions inside one run only.", "Rule."))
    body.append(h2("The question", "question"))
    body.append(p("Each release changes the serving code. This page shows the change in peak requests per second, batch 1, JSON bodies, cache off, for each pair of versions that ran in the same session."))
    body.append(h2("Method", "method"))
    body.append(ul(["Run A: <code>bench/results/v0.5.0/</code> (8 models). Run B: <code>bench/results/v1.0.0/</code> (4 models). Same harness, same commands. See the throughput page for the method.", "Peak is the best requests/s over concurrency 1 to 64, from <code>chart_runs.csv</code>.", "The 0.4.0 run of September 2026 (<code>bench/results_v0.4.0.json</code>, behind the launch post) is a separate run with an older harness. This page does not use it. Its files stay unchanged."]))
    body.append(h2("Results: 0.4.0 to 0.5.0 (run A)", "a"))
    items = [(m if len(m) < 18 else "scatter_include_s..", d[m] * 100, "bw") for m in models5]
    svg = hbar(items, lambda v: f"{v:+.0f}%", title="Change in peak requests per second, 0.5.0 against 0.4.0", desc="One bar for each model.", lw=132)
    body.append(figure(svg, "Run A: change in peak requests/s from 0.4.0 to 0.5.0, batch 1, JSON.", ""))
    hdr = ["model", "0.4.0 peak rps", "0.5.0 peak rps", "change"]
    rows = [[m, n0(pk5(m, "downshift 0.4.0")["throughput_rps"]), n0(pk5(m, "downshift 0.5.0")["throughput_rps"]), pct(d[m] * 100)] for m in models5]
    body.append(table(hdr, rows, caption="Run A: peak requests/s, batch 1", right=range(1, 4)))
    mdt("Release to release, run A: 0.4.0 against 0.5.0", "Peak requests/s, batch 1, JSON, same session.", [R5 / "chart_runs.csv"], hdr, rows)
    body.append(h2("Results: 0.5.1 to 1.0.0rc1 (run B)", "b"))
    mm = {}
    for m in FIX_MODELS:
        mm[m] = peak_of(m, DS_NEW)["throughput_rps"] / peak_of(m, DS_OLD)["throughput_rps"] - 1
    items = [(m if len(m) < 18 else "scatter_include_s..", mm[m] * 100, "bw") for m in FIX_MODELS]
    # negative values cannot be drawn as bars here, so draw magnitude and show the sign in the label
    items2 = [(lab, abs(v), cls) for lab, v, cls in items]
    svg = hbar(items2, lambda v: f"{v:.0f}%", title="Size of the change in peak requests per second, 1.0.0rc1 against 0.5.1", desc="Bars show the size of the change. The signs are in the table.", lw=132)
    body.append(figure(svg, "Run B: size of the change in peak requests/s, 1.0.0rc1 against 0.5.1. Bars show the size only. The sign is in the table.", ""))
    hdr2 = ["model", "0.5.1 peak rps", "1.0.0rc1 peak rps", "change", "note"]
    rows2 = [[m, n0(peak_of(m, DS_OLD)["throughput_rps"]), n0(peak_of(m, DS_NEW)["throughput_rps"]), pct(mm[m] * 100), "main pass was slow; re-run in the throughput page" if m == "mlp_large" else ""] for m in FIX_MODELS]
    body.append(table(hdr2, rows2, caption="Run B: peak requests/s, batch 1", right=range(1, 4)))
    mdt("Release to release, run B: 0.5.1 against 1.0.0rc1", "Peak requests/s, batch 1, JSON, cache off, same session.", [R1 / "chart_runs.csv"], hdr2, rows2)
    body.append(p("In run A, 0.5.0 is faster than 0.4.0 on all eight models. In run B, 1.0.0rc1 and 0.5.1 are close, and 1.0.0rc1 adds the result cache and <code>--metrics</code>."))
    body.append(h2("What this shows and does not show", "limits"))
    body.append(h3("Shows"))
    body.append(ul([f"0.5.0 against 0.4.0: peak throughput rose on all 8 models, from {min(d.values()) * 100:+.0f}% to {max(d.values()) * 100:+.0f}% in the same session.", "1.0.0rc1 against 0.5.1: no gain, and no clear loss except the slow first pass of <code>mlp_large</code>."]))
    body.append(h3("Does not show"))
    body.append(ul(["The final 1.0.0 release. Run B is of <code>1.0.0rc1</code>.", "Any release after 1.0.0rc1.", "0.5.1 against 0.5.0 (never in the same session).", common_limits()]))
    body.append(section_sources(R5 / "chart_runs.csv", R5 / "results.json", R5 / "REPORT.md", R1 / "chart_runs.csv", BENCH / "README.md"))
    return frame("releases", "Releases", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Page: index and summary
# ----------------------------------------------------------------------------------------


def key_facts():
    """The numbers used on the index and on the summary page. All from the data."""
    sc = "scatter_include_self_false"
    m = "all-MiniLM-L6-v2"
    real = jload(R1 / "real_models.json")["models"]
    f = {
        "scatter_naive": worst_err(sc, NAIVE_O),
        "scatter_ds": worst_err(sc, DS_NEW),
        "mini_ds": hfrow(DS_NEW, "short_b1", 8)["throughput_rps"],
        "mini_naive": hfrow("naive FastAPI + transformers", "short_b1", 8)["throughput_rps"],
        "mini_ds_p50": hfrow(DS_NEW, "short_b1", 1)["p50_ms"],
        "mini_naive_p50": hfrow("naive FastAPI + transformers", "short_b1", 1)["p50_ms"],
        "mlp_ds": peak_of("clean_mlp", DS_NEW)["throughput_rps"],
        "mlp_async": peak_of("clean_mlp", NAIVE_A)["throughput_rps"],
        "real_worst": max(x["verdict"]["numerics"]["max_abs_err"] for x in real),
        "real_n": len(real),
        "ready_ds": hfrow(DS_NEW, "short_b1", 1)["ready_s"],
        "ready_naive": hfrow("naive FastAPI + transformers", "short_b1", 1)["ready_s"],
    }
    return f


def page_index() -> str:
    f = key_facts()
    res = jload(R1 / "results.json")
    sc = res["verdicts"]["scatter_include_self_false"]
    body = [header("downshift benchmarks", "downshift serves a PyTorch model over HTTP. Before the first request, it checks that the ONNX version of the model gives the same numbers as PyTorch. These pages show what that check finds and what it costs.", run_meta())]
    body.append(h2("In 60 seconds", "sixty"))
    body.append(
        ol(
            [
                "<b>The problem.</b> <code>torch.onnx.export</code> can finish with no error and still give a graph with wrong numbers.",
                "<b>The check.</b> downshift exports the model in memory. It runs 8 samples through PyTorch and through ONNX Runtime. Some samples have shapes that the exporter did not see. It compares the outputs.",
                "<b>The verdict.</b> CLEAN, DEGRADED, FAILED or UNVERIFIED. The verdict selects the backend: ONNX Runtime when the graph is right, eager PyTorch when it is not.",
                f"<b>An example from the data.</b> The model <code>scatter_include_self_false</code> exports with no error. The check says: {esc(sc['reason'])}. A hand-rolled ONNX Runtime server returns answers that are off by {sci(f['scatter_naive'])}. downshift serves torch and is off by {sci(f['scatter_ds'])}.",
                f"<b>The cost.</b> The check takes time at boot ({n1(f['ready_ds'])} s against {n1(f['ready_naive'])} s for MiniLM). Each request does more work than in a hand-rolled server. On tiny models a hand-rolled server is faster ({n0(f['mlp_async'])} against {n0(f['mlp_ds'])} requests/s peak on <code>clean_mlp</code>).",
            ]
        )
    )
    body.append(figure(flow_diagram(), "The flow. The verdict decides the backend.", ""))
    body.append(
        tiles(
            [
                (f"{sci(f['scatter_naive'])} vs {sci(f['scatter_ds'])}", "Worst error on a model with a silent export fault: hand-rolled ORT vs downshift", "correctness page"),
                (f"{n0(f['mini_ds'])} vs {n0(f['mini_naive'])}", "MiniLM, one sentence, 8 clients, requests/s: downshift vs hand-rolled torch server", "torch vs ORT page"),
                (f"{n0(f['mlp_ds'])} vs {n0(f['mlp_async'])}", "clean_mlp peak requests/s: downshift vs hand-rolled async torch", "throughput page: downshift is slower here"),
                (f"{f['real_n']} of {f['real_n']}", f"real models with verdict CLEAN (worst error {sci(f['real_worst'])})", "compatibility page"),
            ]
        )
    )
    body.append(h2("Pages", "pages"))
    cards = [
        ("throughput", "Throughput and latency", "Requests per second and latency against hand-rolled servers, 0.5.1 and 1.0.0rc1."),
        ("time-breakdown", "Time of one request", "Parse, wait, inference, encode and HTTP time, by stage."),
        ("torch-vs-ort", "ONNX Runtime against PyTorch", "The gain from the ONNX backend, in-process and over HTTP."),
        ("wire-formats", "Batches and wire formats", "JSON against safetensors at batch 8 and 32."),
        ("result-cache", "The result cache", "A repeated request against a new request."),
        ("correctness", "Correctness under load", "Every response against eager PyTorch."),
        ("compatibility", "What the gate catches", "14 hazard fixtures and 8 real models."),
        ("boot-time", "Boot time", "What the check costs before the first request."),
        ("scaling", "More worker processes", "The <code>--workers</code> sweep (0.5.0 data)."),
        ("releases", "Release to release", "0.4.0 to 0.5.0 and 0.5.1 to 1.0.0rc1."),
        ("SUMMARY", "One-page summary", "A page to print or to screenshot."),
    ]
    body.append('<div class="cards">' + "".join(f'<a class="pg" href="{s}.html"><b>{esc(t)}</b><span>{d}</span></a>' for s, t, d in cards) + "</div>")
    body.append(h2("How to read these pages", "howto"))
    body.append(
        ul(
            [
                "Every number comes from a file in <code>bench/</code>. Each page lists its sources at the end.",
                "Each page has the same parts: the question, the method, the results, what the data shows, and what it does not show.",
                "Charts have text labels. A table with the same numbers follows each chart.",
                "Read the limits. All load results are from one machine, in one run, with no repeats.",
            ]
        )
    )
    body.append(h2("What is not measured", "unmeasured"))
    body.append(
        ul(
            [
                "GPU. The machine is CPU only (<code>bench/REPORT_v0.4.0.md</code> states this).",
                "Linux or macOS. The machine is Windows.",
                "TorchServe, BentoML, Triton, TEI and Infinity. No run includes them (<code>bench/REPORT_v0.4.0.md</code>, section Not measured).",
                "The <code>--workers</code> sweep and Prompt Guard for 1.0.0rc1. Both ran in earlier runs only.",
                "Run-to-run spread. No file holds repeats.",
                "Text generation. downshift does not do it.",
            ]
        )
    )
    body.append(section_sources(R1 / "results.json", R1 / "REPORT.md", BENCH / "README.md"))
    return frame("index", "Overview", "\n".join(body))


def page_summary() -> str:
    f = key_facts()
    res = jload(R1 / "results.json")
    mm = {}
    for m in FIX_MODELS:
        mm[m] = peak_of(m, DS_NEW)["throughput_rps"]
    body = [header("downshift benchmarks: one page", "Verified export for a model you already have. What the checks found, and what they cost.", run_meta())]
    body.append(
        tiles(
            [
                (f"{sci(f['scatter_naive'])} vs {sci(f['scatter_ds'])}", "worst error on a model with a silent export fault: hand-rolled ORT vs downshift", "results.json, chart_runs.csv"),
                (f"{f['real_n']} of {f['real_n']}", "real models CLEAN; worst error " + sci(f["real_worst"]), "real_models.json"),
                (f"{n0(f['mini_ds'])} vs {n0(f['mini_naive'])}", "MiniLM requests/s at 8 clients: downshift vs hand-rolled", "chart_runs.csv"),
                (f"{n0(f['mlp_ds'])} vs {n0(f['mlp_async'])}", "clean_mlp peak requests/s: downshift vs hand-rolled async torch", "chart_runs.csv"),
            ]
        )
    )
    # chart 1: MiniLM
    m = "all-MiniLM-L6-v2"
    items = [("h", "MiniLM, one sentence, requests/s at c=8")]
    for s in ["naive FastAPI + transformers", "naive transformers, async def", DS_OLD, DS_NEW, "downshift 1.0.0rc1 --backend torch"]:
        items.append((SHORT[s], hfrow(s, "short_b1", 8)["throughput_rps"], bar_class(s)))
    legend = bar_legend([("bn", "hand-rolled"), ("bo", "downshift 0.5.1"), ("bw", "downshift 1.0.0rc1"), ("bv", "downshift, torch backend")])
    body.append(figure(hbar(items, n0, title="MiniLM requests per second at c=8", desc="Five servers", lw=140), "all-MiniLM-L6-v2 on CPU. 5 s windows. Cache off.", legend))
    items = [("h", "Peak requests/s, batch 1")]
    for mod in ("clean_mlp", "tiny_bert"):
        items.append(("h", mod))
        for s in CORE5:
            items.append((SHORT[s], peak_of(mod, s)["throughput_rps"], bar_class(s)))
    body.append(figure(hbar(items[1:], n0, title="Peak requests per second", desc="Two models, five servers", lw=140), "Fixtures on CPU. 3 s windows.", bar_legend([("bn", "hand-rolled"), ("bo", "downshift 0.5.1"), ("bw", "downshift 1.0.0rc1")])))
    body.append(h2("Read this first", "limits"))
    body.append(
        ul(
            [
                "One Windows machine, 16 logical CPUs, CPU only. One run. No repeats.",
                "downshift 1.0.0rc1 and 0.5.1 are close in speed. The data shows no gain.",
                "On tiny models, hand-rolled servers handle more requests per second. On MiniLM, downshift handles more.",
                "The check costs boot time: " + f"{n1(f['ready_ds'])} s against {n1(f['ready_naive'])} s on MiniLM.",
                "The result cache numbers are a best case: one body, repeated.",
                "Not measured: GPU, Linux, other servers (TorchServe, Triton, TEI), 1.0.0 final.",
            ]
        )
    )
    body.append(p("Full method, tables and limits: <a href=\"index.html\">index.html</a>, <a href=\"RESULTS.md\">RESULTS.md</a>, <a href=\"METHODOLOGY.md\">METHODOLOGY.md</a>."))
    return frame("SUMMARY", "One-page summary", "\n".join(body))


# ----------------------------------------------------------------------------------------
# Markdown outputs
# ----------------------------------------------------------------------------------------


def md_table(headers, rows) -> str:
    esc_ = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(esc_(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc_(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_results_md(failed: dict) -> None:
    out = [
        "# Results",
        "",
        "A plain-text copy of the key tables. The tables are made by `build_showcase.py` from the data files named under each table.",
        "To read the method and the limits of a table, open the matching HTML page. Read [METHODOLOGY.md](METHODOLOGY.md) first.",
        "",
        f"Main run: {env_line(env1())}. downshift 1.0.0rc1 from the checkout, 0.5.1 from PyPI. One machine, one run, no repeats.",
        "",
    ]
    if failed:
        out += ["Pages that could not be built because a source file is missing:", ""] + [f"- `{k}`: missing `{v}`" for k, v in failed.items()] + [""]
    for section, caption, srcs, hdr, rows in MD_TABLES:
        out += [f"## {section}", "", caption, "", md_table(hdr, rows), "", "Source: " + ", ".join(f"[`{rel(s)}`](../{s.relative_to(BENCH).as_posix()})" if BENCH in s.parents else f"[`{rel(s)}`](../../{rel(s)})" for s in srcs), ""]
    write("RESULTS.md", "\n".join(out))


def write_methodology_md() -> None:
    res = jload(R1 / "results.json")
    hf = jload(R1 / "results_hf.json")
    env = res["environment"]
    cfg = res["config"]
    hfcfg = hf["config"]
    real = jload(R1 / "real_models.json")
    e5 = jload(R5 / "results.json")["environment"]
    acc = jload(R6 / "cache_acceptance.json")
    ceil = {int(k): v for k, v in res["client_ceiling_rps"].items()}
    oks = [r["ok"] for r in res["runs"] if r["concurrency"] == 1 and r["batch"] == 1 and r["variant"] == "downshift" and r["version"] == "1.0.0rc1"]
    lines = f"""# Methodology

How the benchmark data in `bench/` is made, what machine and versions made it, how to make it again, and what the data cannot show.
Every value below comes from a data file. The file is named next to the value.

## 1. The question

downshift picks a backend from a correctness verdict. The benchmark does not ask "is downshift fast". It asks three things:

1. What does the verdict cost? (boot time, time of each request)
2. What does it give? (a right answer where a hand-rolled ONNX server gives a wrong one; ONNX Runtime speed where the graph is right)
3. Does the gate agree with what is true? (hazard fixtures with known verdicts, and real models)

## 2. What is measured

| Study | Script | Data file | Page |
|---|---|---|---|
| Throughput and latency under load | `python -m bench.run` | `bench/results/v1.0.0/results.json`, `chart_runs.csv` | [throughput](throughput.html) |
| Time of one request, by stage | `python -m bench.run`, `python -m bench.hf_models` | `chart_stages.csv` | [time-breakdown](time-breakdown.html) |
| ONNX Runtime against torch, in-process | `python -m bench.inproc` (started by `bench.run`) | `results.json` key `inproc` | [torch-vs-ort](torch-vs-ort.html) |
| Real Hugging Face model over HTTP | `python -m bench.hf_models` | `results_hf.json` | [torch-vs-ort](torch-vs-ort.html) |
| Batches and wire formats | `python -m bench.run` | `chart_runs.csv` | [wire-formats](wire-formats.html) |
| Result cache | `bench.run`, `bench.hf_models`, `python -m bench.cache_accept` | `chart_runs.csv`, `bench/results/v0.6.0/cache_acceptance.json` | [result-cache](result-cache.html) |
| Correctness | `bench.run`, `bench.hf_models` | `results.json`, `chart_runs.csv`, `results_hf.json` | [correctness](correctness.html) |
| Real models and hazard fixtures | `python scripts/check_real_models.py`, `python scripts/gen_matrix.py` | `bench/results/v1.0.0/real_models.json`, `docs/compatibility.md` | [compatibility](compatibility.html) |
| Boot time | `bench.run`, `bench.hf_models` | `chart_runs.csv` (`ready_s`), `results_hf.json` | [boot-time](boot-time.html) |
| Workers | `bench.run` (0.5.0 run) | `bench/results/v0.5.0/chart_runs.csv` | [scaling](scaling.html) |
| Release to release | `bench.run` (two runs) | both `chart_runs.csv` files | [releases](releases.html) |

## 3. The harness

- **Servers.** Three hand-rolled servers (`bench/servers.py`): FastAPI + eager torch (sync endpoint), FastAPI + ONNX Runtime (sync), and FastAPI + eager torch with `async def`, which blocks the event loop. For text: FastAPI + transformers (`bench/naive_hf_server.py`). The downshift servers are the real `downshift serve` command, never an in-process shortcut.
- **Fairness.** The hand-rolled servers take a plain `dict` body with no validation and return a plain `dict`. They do less work for each request than downshift. The hand-rolled text server cuts long text. downshift rejects it. All servers get the same seeded weights (seed 1234, `bench/cases.py`). The hand-rolled ORT server gets the graph that the checkout exported.
- **Load.** Closed loop (`bench/loadgen.py`). N clients each send the next request when the last one returns. The body is the same bytes every time. The load generator is split over up to 6 processes on the same machine.
- **Ceiling.** Before the sweep, the harness measures the load generator against `GET /health` on a server that does nothing: {", ".join(f"c={c}: {n0(v)} rps" for c, v in ceil.items())}. A result near these numbers would measure the client.
- **Windows.** Warm-up {cfg["warmup"]} s, then {cfg["duration"]} s for each point on the fixtures. Text models: warm-up {hfcfg["warmup"]} s, window {hfcfg["duration"]} s, concurrency {hfcfg["concurrency"]}. A `/health` probe runs every 50 ms during each text window.
- **Correctness.** Every response is compared element by element with eager PyTorch on the same input. The hand-rolled text servers and downshift answers are compared with `downshift --backend torch`.
- **Result cache.** Every 1.0.0rc1 row except "result cache on" starts with `--result-cache-rows 0`. The harness repeats one body, so a cache would answer from memory.

## 4. Hardware and versions (from the data)

| Item | Value | Source |
|---|---|---|
| Main run date | {env["timestamp"]} (HF part: {hf["environment"]["timestamp"]}) | `results.json`, `results_hf.json` |
| Platform | {env["platform"]}; `{hf["platform"]}` | `results.json`, `results_hf.json` |
| Logical CPUs | {env["cpu_count"]} | `results.json` |
| CPU model, RAM | not recorded in the 1.0.0 files | - |
| Python | {env["python"]} | `results.json` |
| torch | {env["torch"]}; torch threads {env["torch_num_threads"]} | `results.json` |
| onnxruntime | {env["onnxruntime"]}; providers {", ".join(env["ort_providers"])} | `results.json` |
| downshift | checkout {env["downshift_targets"]["checkout"]}; installed {env["downshift_targets"]["installed"]} | `results.json` |
| Load settings | cases {", ".join(cfg["cases"])}; concurrency {cfg["concurrency"]}; batch {cfg["batch"]}; batch concurrency {cfg["batch_concurrency"]}; in-process iterations {cfg["inproc_iters"]} | `results.json` |
| Not run | the workers sweep (`skip_workers` = {cfg["skip_workers"]}); Prompt Guard; `dynamic_batch_cnn`, `gnn_gcn`, `cnn_large`, `bert_small` | `results.json` |
| Real-model check | date {real["date"]}; torch {real["versions"]["torch"]}, onnx {real["versions"]["onnx"]}, onnxruntime {real["versions"]["onnxruntime"]}, transformers {real["versions"]["transformers"]}, torch_geometric {real["versions"]["torch_geometric"]}, torchvision {real["versions"]["torchvision"]}; OS and CPU not recorded | `real_models.json` |
| Cache acceptance (0.6.0) | run {acc["when"]}; {acc["repeats"]} repeats; machine and versions not recorded | `cache_acceptance.json` |
| Earlier run (0.4.0 and 0.5.0) | {e5["timestamp"]}; {e5["platform"]}, {e5["cpu_count"]} logical CPUs, Python {e5["python"]}, torch {e5["torch"]}, onnxruntime {e5["onnxruntime"]} | `bench/results/v0.5.0/results.json` |

The folder `results/v1.0.0/` holds a run of downshift `{env["downshift_targets"]["checkout"]}`, not of the final 1.0.0 release.

## 5. How to reproduce

Run nothing else on the machine while the benchmark runs. A test suite at the same time more than doubles the latencies (`bench/README.md`).

1. Make a venv. Install the checkout dependencies and the previous release (non-editable): `pip install "downshift-server[all]==0.5.1" aiohttp safetensors`.
2. Download the two Hugging Face checkpoints into `bench/models/` (see `bench/README.md`). Prompt Guard is gated.
3. Run the fixture part: `python -m bench.run --out bench/results/vX.Y.Z/results.json`. The stored settings of the 1.0.0rc1 run match `--cases clean_mlp tiny_bert mlp_large scatter_include_self_false --skip-workers`. The exact command line is not stored.
4. Run the text part: `python -m bench.hf_models --out bench/results/vX.Y.Z/results_hf.json` (the 1.0.0rc1 run used all-MiniLM-L6-v2 only).
5. Make the report and the chart CSV files: `python -m bench.report --results ... --hf-results ... --out .../REPORT.md --csv .../chart_runs.csv --stages-csv .../chart_stages.csv`.
6. Check the real models: `python scripts/check_real_models.py`. Make the fixture table: `python scripts/gen_matrix.py`.
7. Run the cache check: `python -m bench.cache_accept --out bench/results/vX.Y.Z/cache_acceptance.json`.
8. Make this site again: `python bench/showcase/build_showcase.py`. Point `R1` (and the other result folders) in the script at the new folder first.

Without `--out`, the commands write to `bench/scratch/`. A trial run never overwrites a published run.

## 6. Known limits and threats to validity

1. **One machine, one run, no repeats.** No file holds repeated runs. The size of run-to-run noise is not measured. A difference of a few percent is not a finding. The report itself calls the difference of the cache-on and cache-off rows of `tiny_bert` and `scatter_include_self_false` "noise".
2. **Short windows.** {cfg["duration"]} s for each point. The p99 at concurrency 1 comes from {min(oks)} to {max(oks)} requests for the 1.0.0rc1 default rows. The p99 at high concurrency has fewer samples for each client.
3. **The load generator shares the machine.** The CPU is shared between the client and the server. Results are valid for this layout only.
4. **Closed loop.** At high concurrency the latency is set by the concurrency, so the latency at c=64 is mostly a queue.
5. **A slow main pass.** The first `mlp_large` pass of 1.0.0rc1 default was slow. The report says it "does not fit one code path". A re-run covers batch 1 only (`rerun_mlp_large.json`). The batch 8 and 32 rows of `mlp_large` for 1.0.0rc1 JSON are not re-measured.
6. **Release candidate.** The newest data is of `1.0.0rc1`.
7. **Small fixtures.** On `clean_mlp` the model takes about 0.13 ms of a 0.70 ms request. The fixtures measure the request path, not the model. The compute-heavy models (`mlp_large`, `tiny_bert`) are still small. Only MiniLM is a real model in the load run.
8. **Random weights.** The fixtures use seeded random weights. The correctness numbers hold for those weights.
9. **Hand-rolled baselines are one person's choice.** They are written to do less work than downshift. A tuned hand-rolled server could be faster or slower. TorchServe, BentoML, Triton, TEI and Infinity are not measured.
10. **The best-case rows.** "Result cache on" repeats one body. The acceptance check of the cache (0.6.0) runs in-process with a test client, stores no machine data, and was not run again for 1.0.0.
11. **The gate is a sample.** The gate runs 8 samples. A fault that none of them hits is not found. The real-model check shows verdicts, not a false-negative rate.
12. **Boot time ends are not equal.** downshift is ready on `/ready`. The hand-rolled servers are ready on `/health`.
13. **CPU and Windows only.** GPU, Linux and macOS are not measured.
14. **Source mismatches.** Some existing files disagree (for example the re-run peaks in `results/v1.0.0/REPORT.md`, and the version text in `docs/compatibility.md`). The pages show the file values and mark the mismatch.

## 7. How the site is made

`build_showcase.py` reads the files above and writes into `bench/showcase/` only. It uses the Python standard library only. Sentences that depend on the data are checked in the script (`claim(...)`). If the data change and a sentence is no longer true, the build stops with an error.
"""
    write("METHODOLOGY.md", lines)


def write_talk_md() -> None:
    f = key_facts()
    res = jload(R1 / "results.json")
    sc = res["verdicts"]["scatter_include_self_false"]
    m = "all-MiniLM-L6-v2"
    hit = jload(R6 / "cache_acceptance.json")["a_repeated_query_server_ms"]
    text = f"""# Talk outline (10 minutes)

For a conference talk or a blog post. Every number comes from the data files named in [RESULTS.md](RESULTS.md). Keep the limits slide.

## 1. The problem (1 min)
- `torch.onnx.export` can finish with no error and give a graph with wrong numbers.
- Show: `scatter_include_self_false`. Verdict DEGRADED. {sc["reason"]}.
- A hand-rolled ONNX Runtime server answers with an error of {sci(f["scatter_naive"])}. (Page: correctness)

## 2. What downshift does (1 min)
- Show the flow diagram (`index.html`): load, export, verify, verdict, backend.
- 8 samples, some with shapes the exporter did not see. Four verdicts. The verdict selects the backend.

## 3. Does the gate agree with the truth? (2 min)
- 14 hazard fixtures with known verdicts. 8 real models: 8 of 8 CLEAN, worst error {sci(f["real_worst"])} against a tolerance of 1e-4. (Page: compatibility)
- Limit: the gate is a sample of 8. The data has no false-negative rate.

## 4. What does the gate cost? (2 min)
- Boot: {n1(f["ready_ds"])} s against {n1(f["ready_naive"])} s on MiniLM. (Page: boot time)
- Request path: on a tiny model, 66% of a request is HTTP and hand-offs. A hand-rolled server wins on `clean_mlp`: {n0(f["mlp_async"])} against {n0(f["mlp_ds"])} requests/s at peak. (Pages: throughput, time of one request)

## 5. What does the gate give? (2 min)
- ONNX Runtime on a real encoder: MiniLM, one sentence, 8 clients: {n0(f["mini_ds"])} against {n0(f["mini_naive"])} requests/s. p50 at one client: {n1(f["mini_ds_p50"])} ms against {n1(f["mini_naive_p50"])} ms. (Page: torch vs ORT)
- The gain falls with the input size. Show the three payloads.
- Wire format: safetensors helps wide inputs, hurts tiny ones. (Page: wire formats)

## 6. Honest notes (1 min)
- 1.0.0rc1 is not faster than 0.5.1. The data shows no gain. (Page: releases)
- The result cache: a hit takes {hit["hit_median"]} ms of server time against {hit["no_cache_median"]} ms. The load-run numbers are a best case. (Page: result cache)
- One machine, one run, no repeats, CPU only. Not measured: GPU, Linux, other servers.

## 7. Q and A: likely questions
- "Why not export once and trust it?" The fixtures show exports that finish with no error and give wrong numbers.
- "Why is a hand-rolled server faster on small models?" It does less work for each request. The time page shows the split. The data does not say more.
- "Is it faster on a GPU?" Not measured.
- "How noisy is this?" Not measured. No file holds repeats.
"""
    write("TALK_OUTLINE.md", text)


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------

BUILDERS = [
    ("throughput", page_throughput),
    ("time-breakdown", page_time),
    ("torch-vs-ort", page_torch_ort),
    ("wire-formats", page_wire),
    ("result-cache", page_cache),
    ("correctness", page_correct),
    ("compatibility", page_compat),
    ("boot-time", page_boot),
    ("scaling", page_scaling),
    ("releases", page_releases),
    ("index", page_index),
    ("SUMMARY", page_summary),
]


def stub(slug: str, missing: str) -> str:
    title = dict(PAGES).get(slug, slug)
    body = header(title, "This page could not be built.") + note("bad", f"The source file <code>{esc(missing)}</code> does not exist. Run the benchmark that makes it, then build again.", "Missing data.")
    return frame(slug, title, body)


def main() -> int:
    failed: dict[str, str] = {}
    for slug, fn in BUILDERS:
        before = len(MD_TABLES)
        try:
            text = fn()
        except SourceMissing as exc:
            del MD_TABLES[before:]
            failed[slug] = str(exc)
            text = stub(slug, str(exc))
        write(f"{slug}.html", text)
        print(f"wrote {slug}.html")
    write_results_md(failed)
    print("wrote RESULTS.md")
    try:
        write_methodology_md()
        print("wrote METHODOLOGY.md")
        write_talk_md()
        print("wrote TALK_OUTLINE.md")
    except SourceMissing as exc:
        failed["md"] = str(exc)
        print(f"skipped METHODOLOGY.md and TALK_OUTLINE.md: missing {exc}")
    print(f"claims checked: {CLAIMS_CHECKED}")
    if failed:
        print("missing sources:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
