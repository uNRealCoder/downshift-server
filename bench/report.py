"""Turn results.json into the tables the blog post needs.

python -m bench.report --results bench/results.json --out bench/REPORT.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

VARIANT_LABEL = {
    "naive_torch": "naive FastAPI + eager torch",
    "naive_onnx": "naive FastAPI + ONNX Runtime",
    "downshift": "downshift serve (auto)",
    "downshift_base64": "downshift serve (base64)",
    "naive_torch_async": "naive FastAPI + torch, `async def`",
}
ORDER = ["naive_torch", "naive_onnx", "downshift", "downshift_base64", "naive_torch_async"]
# The variants that answer the product question; the `async def` row is a cautionary tale.
MAIN = [v for v in ORDER if v != "naive_torch_async"]


def level(block: dict | None, key: str):
    """p50 of one in-process level, or None when an older results.json never measured it."""
    if not block:
        return None
    return (block.get(key) or {}).get("p50_ms")


def delta(block: dict | None, key: str):
    """What one level adds over bare compute, or None when either side is missing."""
    a, b = level(block, "compute"), level(block, key)
    return None if a is None or b is None else b - a


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    out = ["| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def index_runs(report: dict):
    by = defaultdict(dict)  # (case, batch, variant) -> {concurrency: run}
    for r in report["runs"]:
        by[(r["case"], r["batch"], r["variant"])][r["concurrency"]] = r
    return by


def index_workers(report: dict):
    by = defaultdict(dict)  # (case, workers) -> {concurrency: run}
    for r in report.get("workers_runs", []):
        by[(r["case"], r["workers"])][r["concurrency"]] = r
    return by


def fmt(v, spec="{:.1f}"):
    if v is None:
        return "—"
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def build(report: dict) -> str:
    env = report["environment"]
    cfg = report["config"]
    by = index_runs(report)
    cases = [c for c in cfg["cases"] if c in report["verdicts"]]
    concurrencies = cfg["concurrency"]
    out: list[str] = []

    out.append("# downshift serving benchmark\n")
    out.append(
        f"downshift {env.get('downshift', '?')} on `{env['platform']}`, "
        f"{env['cpu_count']} logical CPUs, Python {env['python']}, "
        f"torch {env['torch']} (CPU), onnxruntime {env['onnxruntime']}, "
        f"torch intra-op threads {env['torch_num_threads']}. Run {env['timestamp']}.\n"
    )
    out.append(
        f"Closed-loop load, {cfg['duration']}s measurement window after {cfg['warmup']}s warmup, "
        f"load generator sharded over up to 6 processes on the same machine as the server. "
        "Every variant is handed the identical prepared module and the identical exported ONNX "
        "graph and receives byte-identical request bodies; the only variable is the server.\n"
    )

    if "client_ceiling_rps" in report:
        ceiling = report["client_ceiling_rps"]
        out.append("## Load generator ceiling\n")
        out.append(
            "How fast this harness can drive a do-nothing endpoint on this machine. Any server "
            "result approaching these numbers is measuring the client, not the server.\n"
        )
        out.append(
            table(
                [[str(c), fmt(ceiling.get(str(c)))] for c in concurrencies],
                ["concurrency", "GET /health rps"],
            )
            + "\n"
        )

    out.append("## Verdicts\n")
    rows = []
    for name in cases:
        v = report["verdicts"][name]
        rows.append(
            [
                f"`{name}`",
                v["family"],
                v["status"],
                v["backend"],
                fmt(v.get("max_abs_err"), "{:.2e}") if v.get("max_abs_err") is not None else "—",
            ]
        )
    out.append(table(rows, ["model", "family", "verdict", "backend chosen", "max abs err"]) + "\n")

    # ---- in-process ------------------------------------------------------------------
    if report.get("inproc"):
        out.append("## Inference cost with no server attached\n")
        out.append(
            "`compute` is `infer(feeds)` on arrays already in memory. `+json` adds parsing the "
            "request body, building the arrays, `.tolist()` on the outputs and serializing the "
            "response — the cost of speaking JSON, before any ASGI or socket work. `+binary` is "
            "the same round trip with base64 tensor bodies: `b64decode` + `np.frombuffer` in, "
            "`b64encode` of the output bytes out. `request` is the nested-list body; `binary req` "
            "is the base64 body carrying the same tensors.\n"
        )
        rows = []
        for d in report["inproc"]:
            t, o = d["torch"], d.get("onnxruntime")
            t_c, o_c = level(t, "compute"), level(o, "compute")
            speedup = f"{t_c / o_c:.2f}x" if o_c else "—"
            rb = d.get("request_bytes_binary")
            rows.append(
                [
                    f"`{d['case']}`",
                    d["batch"],
                    fmt(t_c, "{:.3f}"),
                    fmt(o_c, "{:.3f}"),
                    speedup,
                    fmt(level(t, "compute_json"), "{:.3f}"),
                    fmt(level(o, "compute_json"), "{:.3f}"),
                    fmt(level(t, "compute_binary"), "{:.3f}"),
                    fmt(level(o, "compute_binary"), "{:.3f}"),
                    fmt(d["request_bytes"] / 1024, "{:.1f}") + " KiB",
                    fmt(rb / 1024, "{:.1f}") + " KiB" if rb is not None else "—",
                ]
            )
        out.append(
            table(
                rows,
                [
                    "model",
                    "batch",
                    "torch ms",
                    "ORT ms",
                    "ORT speedup",
                    "torch +json ms",
                    "ORT +json ms",
                    "torch +binary ms",
                    "ORT +binary ms",
                    "request",
                    "binary req",
                ],
            )
            + "\n"
        )

    # ---- throughput vs concurrency ---------------------------------------------------
    out.append("## Throughput vs concurrency (batch 1)\n")
    out.append("Requests per second, closed loop. Higher is better.\n")
    for name in cases:
        present = [v for v in ORDER if (name, 1, v) in by]
        if not present:
            continue
        out.append(
            f"**`{name}`** — {report['verdicts'][name]['status']}, "
            f"downshift serves via {report['verdicts'][name]['backend']}\n"
        )
        rows = []
        for variant in present:
            runs = by[(name, 1, variant)]
            rows.append(
                [VARIANT_LABEL[variant]]
                + [fmt(runs[c]["throughput_rps"]) if c in runs else "—" for c in concurrencies]
            )
        out.append(table(rows, ["server"] + [f"c={c}" for c in concurrencies]) + "\n")

    # ---- peak throughput summary -----------------------------------------------------
    out.append("## Peak throughput and where it lands (batch 1)\n")
    rows = []
    for name in cases:
        base = by.get((name, 1, "naive_torch"), {})
        base_peak = max((r["throughput_rps"] for r in base.values()), default=None)
        for variant in ORDER:
            runs = by.get((name, 1, variant))
            if not runs:
                continue
            best = max(runs.values(), key=lambda r: r["throughput_rps"])
            rel = f"{best['throughput_rps'] / base_peak:.2f}x" if base_peak else "—"
            rows.append(
                [
                    f"`{name}`",
                    VARIANT_LABEL[variant],
                    fmt(best["throughput_rps"]),
                    f"c={best['concurrency']}",
                    rel,
                    fmt(best["p50_ms"], "{:.2f}"),
                    fmt(best["p99_ms"], "{:.2f}"),
                    fmt(best["max_abs_err"], "{:.2e}"),
                ]
            )
    out.append(
        table(
            rows,
            [
                "model",
                "server",
                "peak rps",
                "at",
                "vs naive torch",
                "p50 ms",
                "p99 ms",
                "max abs err",
            ],
        )
        + "\n"
    )

    # ---- latency ---------------------------------------------------------------------
    out.append("## Latency at low and high concurrency (batch 1)\n")
    lo, hi = concurrencies[0], concurrencies[-1]
    rows = []
    for name in cases:
        for variant in ORDER:
            runs = by.get((name, 1, variant))
            if not runs or lo not in runs:
                continue
            a, b = runs[lo], runs.get(hi)
            rows.append(
                [
                    f"`{name}`",
                    VARIANT_LABEL[variant],
                    fmt(a["p50_ms"], "{:.2f}"),
                    fmt(a["p99_ms"], "{:.2f}"),
                    fmt(b["p50_ms"], "{:.2f}") if b else "—",
                    fmt(b["p99_ms"], "{:.2f}") if b else "—",
                ]
            )
    out.append(
        table(rows, ["model", "server", f"p50 c={lo}", f"p99 c={lo}", f"p50 c={hi}", f"p99 c={hi}"])
        + "\n"
    )

    # ---- serving tax -----------------------------------------------------------------
    inproc = {(d["case"], d["batch"]): d for d in report.get("inproc", [])}
    if inproc:
        out.append("## The serving tax\n")
        out.append(
            "`compute` is the model on arrays already in memory; `json` is what parsing the "
            "request and serializing the response adds on top, measured in the same process, so "
            "that subtraction is sound; `binary` is the same addition for base64 tensor bodies, "
            "which is the codec the `downshift serve (base64)` row is actually paying. "
            "`p50 over HTTP` is the single-client latency of the real "
            "server. The gap between the two is **not** subtracted here: they come from different "
            "processes with different allocator and thread-pool state, and for the large-payload "
            "rows the difference is smaller than that discrepancy. Read the last column instead — "
            "the share of end-to-end latency that is actually the model.\n"
        )
        rows = []
        tax_variants = (
            ("naive_torch", "torch"),
            ("naive_onnx", "onnxruntime"),
            ("downshift", None),
            ("downshift_base64", None),
        )
        for name in cases:
            for batch in cfg["batch"]:
                d = inproc.get((name, batch))
                if not d:
                    continue
                for variant, key in tax_variants:
                    runs = by.get((name, batch, variant))
                    if not runs or 1 not in runs:
                        continue
                    backend_key = key
                    if backend_key is None:
                        backend_key = (
                            "onnxruntime"
                            if report["verdicts"][name]["backend"] == "onnxruntime"
                            else "torch"
                        )
                    block = d.get(backend_key)
                    if not block:
                        continue
                    compute = level(block, "compute")
                    http = runs[1]["p50_ms"]
                    rows.append(
                        [
                            f"`{name}`",
                            batch,
                            VARIANT_LABEL[variant],
                            fmt(compute, "{:.3f}"),
                            fmt(delta(block, "compute_json"), "{:.3f}"),
                            fmt(delta(block, "compute_binary"), "{:.3f}"),
                            fmt(http, "{:.3f}"),
                            fmt(100 * compute / http, "{:.0f}") + "%"
                            if compute is not None and http
                            else "—",
                        ]
                    )
        out.append(
            table(
                rows,
                [
                    "model",
                    "batch",
                    "server",
                    "compute ms",
                    "json ms",
                    "binary ms",
                    "p50 over HTTP ms",
                    "compute share",
                ],
            )
            + "\n"
        )

    # ---- what downshift itself costs -------------------------------------------------
    out.append("## What downshift's own serving layer costs\n")
    out.append(
        "downshift against the naive server running **the same backend**, so the only difference "
        "is the HTTP layer: pydantic request validation, a `response_model`, and a response that "
        "also carries per-output shapes and dtypes. Below 1.00x downshift is slower. This is the "
        "price of the product's ergonomics, and it is not free.\n"
    )
    rows = []
    for name in cases:
        peer = (
            "naive_onnx" if report["verdicts"][name]["backend"] == "onnxruntime" else "naive_torch"
        )
        for batch in cfg["batch"]:
            ds, nv = by.get((name, batch, "downshift")), by.get((name, batch, peer))
            if not ds or not nv:
                continue
            d_peak = max(ds.values(), key=lambda r: r["throughput_rps"])["throughput_rps"]
            n_peak = max(nv.values(), key=lambda r: r["throughput_rps"])["throughput_rps"]
            rows.append(
                [
                    f"`{name}`",
                    batch,
                    VARIANT_LABEL[peer],
                    fmt(n_peak),
                    fmt(d_peak),
                    f"{d_peak / n_peak:.2f}x" if n_peak else "—",
                ]
            )
    out.append(
        table(
            rows,
            ["model", "batch", "same-backend peer", "peer peak rps", "downshift peak rps", "ratio"],
        )
        + "\n"
    )

    # ---- the frontier ----------------------------------------------------------------
    degraded = [n for n in cases if report["verdicts"][n]["status"] == "DEGRADED"]
    if degraded:
        out.append("## Throughput against correctness\n")
        out.append(
            "The only models where the two axes actually trade off: the exporter produced a graph, "
            "and the graph is wrong. Peak throughput at batch 1, against the served output's error.\n"
        )
        rows = []
        for name in degraded:
            for variant in ORDER:
                runs = by.get((name, 1, variant))
                if not runs:
                    continue
                best = max(runs.values(), key=lambda r: r["throughput_rps"])
                err = best["max_abs_err"]
                rows.append(
                    [
                        f"`{name}`",
                        VARIANT_LABEL[variant],
                        fmt(best["throughput_rps"]),
                        fmt(err, "{:.2e}"),
                        "correct" if err < 1e-4 else "**WRONG ANSWERS**",
                    ]
                )
        out.append(table(rows, ["model", "server", "peak rps", "max abs err", "verdict"]) + "\n")

    # ---- batch sweep -----------------------------------------------------------------
    batches = [b for b in cfg["batch"] if b != 1]
    if batches:
        out.append("## Batch size sweep\n")
        out.append(
            "Throughput in requests/s, and the same number as items/s, at the concurrency where "
            "each variant peaked. Larger batches amortize the per-request server cost over more work.\n"
        )
        rows = []
        for name in cases:
            for batch in cfg["batch"]:
                for variant in MAIN:
                    runs = by.get((name, batch, variant))
                    if not runs:
                        continue
                    best = max(runs.values(), key=lambda r: r["throughput_rps"])
                    rows.append(
                        [
                            f"`{name}`",
                            batch,
                            VARIANT_LABEL[variant],
                            fmt(best["throughput_rps"]),
                            fmt(best["throughput_rps"] * batch),
                            f"c={best['concurrency']}",
                            fmt(best["p50_ms"], "{:.2f}"),
                        ]
                    )
        out.append(
            table(rows, ["model", "batch", "server", "rps", "items/s", "at", "p50 ms"]) + "\n"
        )

    # ---- correctness -----------------------------------------------------------------
    out.append("## Correctness of the served response\n")
    out.append(
        "Max absolute difference between the served output and eager PyTorch on the same input. "
        "Throughput numbers above are only comparable between rows whose error is at float32 noise.\n"
    )
    rows = []
    for name in cases:
        for variant in MAIN:
            errs = [
                r["max_abs_err"]
                for b in cfg["batch"]
                for r in by.get((name, b, variant), {}).values()
            ]
            if not errs:
                continue
            rows.append([f"`{name}`", VARIANT_LABEL[variant], fmt(max(errs), "{:.2e}")])
    out.append(table(rows, ["model", "server", "max abs err vs eager torch"]) + "\n")

    # ---- workers sweep -----------------------------------------------------------------
    if report.get("workers_runs"):
        out.append("## Workers\n")
        out.append(
            "`--workers N` only exists on the real `downshift serve` CLI, not the in-process "
            "app the rest of this report drives (`bench.servers`), so this section launches the "
            "CLI itself against the import specs in `bench/factories.py` — built under the same "
            "seed as every other fixture here, so its correctness numbers are comparable to the "
            "rest of the report. `workers=1` is bind-first: the port opens immediately and "
            "`boot_s` is purely the export/verify/warmup gate behind `/ready`. `workers>1` "
            "exports once in the parent and hands the artifact to every worker, so `boot_s` "
            "there is that export plus each worker's own load, verify and warmup, running in "
            "parallel with each other but not with the export.\n"
        )
        wby = index_workers(report)
        w_cases = [c for c in cfg.get("workers_cases", []) if c in report.get("verdicts", {})]
        w_sweep = cfg.get("workers_sweep", [])
        w_concurrency = cfg.get("workers_concurrency", [])

        out.append("**Boot time**, i.e. how long `/ready` takes to answer 200.\n")
        rows = []
        for name in w_cases:
            for n in w_sweep:
                runs = wby.get((name, n))
                if not runs:
                    continue
                boot = next(iter(runs.values()))["boot_s"]
                rows.append([f"`{name}`", n, fmt(boot, "{:.2f}")])
        out.append(table(rows, ["model", "workers", "boot_s"]) + "\n")

        out.append(
            "**Throughput at batch 1**, by concurrency. `max abs err` is the worst error seen "
            "across that row's concurrencies — a worker count that is fast but wrong should not "
            "read as a win.\n"
        )
        header = ["model", "workers"]
        for c in w_concurrency:
            header += [f"rps c={c}", f"p99 c={c}"]
        header.append("max abs err")
        rows = []
        footnotes = []
        for name in w_cases:
            for n in w_sweep:
                runs = wby.get((name, n))
                if not runs:
                    continue
                remeasured = [r for r in runs.values() if r.get("remeasured")]
                worker_cell = f"{n} *" if remeasured else n
                row = [f"`{name}`", worker_cell]
                for c in w_concurrency:
                    r = runs.get(c)
                    row += [
                        fmt(r["throughput_rps"]) if r else "—",
                        fmt(r["p99_ms"], "{:.2f}") if r else "—",
                    ]
                row.append(fmt(max(r["max_abs_err"] for r in runs.values()), "{:.2e}"))
                rows.append(row)
                for r in remeasured:
                    note = r.get("remeasured_note")
                    if note and note not in footnotes:
                        footnotes.append(note)
        out.append(table(rows, header) + "\n")
        for note in footnotes:
            out.append(f"\\* {note}\n")

    out.append("## Not measured\n")
    out.append(
        "- TorchServe and BentoML. Deferred; they need their own packaging step and a separate "
        "run to be a fair comparison.\n"
        "- GPU. This box is CPU-only, and `--device cuda` is untested in this release.\n"
        "- `--workers > 1` for anything but the `downshift serve` CLI itself, which is measured "
        "in the Workers section when this run included the sweep. The rest of this matrix, "
        "including the naive baselines, still runs a single uvicorn worker per server.\n"
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results.json")
    ap.add_argument("--out", default="bench/REPORT.md")
    args = ap.parse_args()
    report = load(Path(args.results))
    text = build(report)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
