import json

g = [r for r in json.load(open("bench/results/v0.6.0/torch_vs_ort_cuda.json")) if r["ok"]]
c = [r for r in json.load(open("bench/results/v0.6.0/torch_vs_ort.json")) if r["ok"]]
SKIP = "scatter_include_self_false"
cases = [x for x in dict.fromkeys(r["case"] for r in g) if x != SKIP]
gi = {(r["case"], r["batch"], r["mode"], r["variant"]): r for r in g}
V = ["torch_eager", "torch_tf32", "torch_compile", "ort_strict", "ort_tf32"]
H = "| model | eager (fp32) | torch TF32 | compile (fp32) | ORT fp32 | ORT TF32 |"
out = []
spread = lambda r: (r["items_per_s_max"] - r["items_per_s_min"]) / r["items_per_s"]

def thr(mode, b):
    out.append(f"@@THR_{mode}_b{b}")
    out.append(H); out.append("|---|---|---|---|---|---|")
    for case in cases:
        cells = []
        for v in V:
            r = gi[(case, b, mode, v)]
            cells.append(f"{r['items_per_s']:,.0f}" + ("†" if spread(r) > 0.5 else ""))
        out.append(f"| `{case}` | " + " | ".join(cells) + " |")
    out.append("")

for mode, b in (("host", 1), ("host", 32), ("host", 256), ("resident", 256)):
    thr(mode, b)

def best(rows, case, b, mode=None, only=None):
    pool = [r for r in rows if r["case"] == case and r["batch"] == b and (mode is None or r.get("mode") == mode)
            and (only is None or r["variant"] in only)]
    return max(pool, key=lambda r: r["items_per_s"])

out.append("@@VSCPU")
out.append("| model | batch | best CPU | best GPU, fp32 | GPU / CPU | best GPU, TF32 allowed | GPU / CPU |")
out.append("|---|---|---|---|---|---|---|")
for case in cases:
    for b in (1, 32):
        cb = best(c, case, b)
        gs = best(g, case, b, "host", ("torch_eager", "torch_compile", "ort_strict"))
        ga = best(g, case, b, "host")
        mark = lambda r: "†" if spread(r) > 0.5 else ""
        out.append(f"| `{case}` | {b} | {cb['items_per_s']:,.0f} | {gs['items_per_s']:,.0f}{mark(gs)} ({gs['variant']}) | {gs['items_per_s']/cb['items_per_s']:.2f}x | {ga['items_per_s']:,.0f}{mark(ga)} ({ga['variant']}) | {ga['items_per_s']/cb['items_per_s']:.2f}x |")
out.append("")

out.append("@@RES")
out.append("| model | variant | GPU util % | GPU power W | VRAM MB | host cores | host peak RSS MB |")
out.append("|---|---|---|---|---|---|---|")
for case in cases:
    for v in ("torch_eager", "ort_strict"):
        r = gi[(case, 256, "resident", v)]
        out.append(f"| `{case}` | {v} | {r['gpu_util_pct']:.0f} | {r['gpu_power_w']:.0f} | {r['gpu_mem_mb']:.0f} | {r['cores_busy']:.2f} | {r['peak_rss_mb']:.0f} |")
out.append("")
out.append("@@RES1")
out.append("| model | variant | GPU util % | GPU power W | VRAM MB |")
out.append("|---|---|---|---|---|")
for case in cases:
    for v in ("torch_eager", "ort_strict"):
        r = gi[(case, 1, "host", v)]
        out.append(f"| `{case}` | {v} | {r['gpu_util_pct']:.0f} | {r['gpu_power_w']:.0f} | {r['gpu_mem_mb']:.0f} |")
out.append("")
out.append("@@ERR")
out.append("| model | fp32 (eager / ORT) | TF32 (torch / ORT) |")
out.append("|---|---|---|")
for case in cases:
    e = lambda v: gi[(case, 256, "resident", v)]["max_abs_err"]
    out.append(f"| `{case}` | {e('torch_eager'):.1e} / {e('ort_strict'):.1e} | {e('torch_tf32'):.1e} / {e('ort_tf32'):.1e} |")
out.append("")
out.append("@@SETUP")
out.append("| model | eager s | compile s | ORT s |")
out.append("|---|---|---|---|")
for case in cases:
    s = lambda v: gi[(case, 1, "host", v)]["setup_s"]
    out.append(f"| `{case}` | {s('torch_eager'):.1f} | {s('torch_compile'):.1f} | {s('ort_strict'):.1f} |")
print("\n".join(out))
