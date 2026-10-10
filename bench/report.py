"""Turn the two results files into one Markdown report: naive vs previous release vs this one,
plus two long-format CSV files to draw charts from.

  python -m bench.report --results bench/results/v0.5.0/results.json \
      --hf-results bench/results/v0.5.0/results_hf.json --out bench/results/v0.5.0/REPORT.md \
      --csv bench/results/v0.5.0/chart_runs.csv --stages-csv bench/results/v0.5.0/chart_stages.csv

Every number in the report is read from the JSON; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

NAIVE_NAMES = {
    "naive_torch": "naive FastAPI + torch",
    "naive_onnx": "naive FastAPI + ONNX Runtime",
    "naive_torch_async": "naive torch, `async def`",
    "naive_sync": "naive FastAPI + transformers",
    "naive_async": "naive transformers, `async def`",
}
DOWNSHIFT_NAMES = {
    "downshift": "",
    "downshift_base64": " base64",
    "downshift_safetensors": " safetensors",
    "downshift_inline": " `--execution inline`",
    "downshift_auto": "",
    "downshift_torch": " `--backend torch`",
    "downshift_auto_mc4": " `--max-concurrency 4`",
    "downshift_auto_inline": " `--execution inline`",
}


def name(variant: str, version: str | None) -> str:
    if version is None:
        return NAIVE_NAMES[variant]
    return f"downshift {version}{DOWNSHIFT_NAMES[variant]}"


def plain(variant: str, version: str | None) -> str:
    """name() without Markdown, for CSV labels and chart legends."""
    return name(variant, version).replace("`", "")


# Server-Timing stages in pipeline order. 0.4.0 reports `codec` (decode and encode together)
# where 0.5.0 splits prep_wait, prep, infer_wait and encode.
STAGE_ORDER = ("parse", "codec", "prep_wait", "prep", "infer_wait", "infer", "encode")


def server_order(keys, versions: list[str]) -> list[tuple[str, str | None]]:
    """Naive first, then each downshift version oldest first, variants in table order."""
    rank_v = {v: i for i, v in enumerate(versions)}
    rank_n = {v: i for i, v in enumerate([*NAIVE_NAMES, *DOWNSHIFT_NAMES])}
    return sorted(set(keys), key=lambda k: (k[1] is not None, rank_v.get(k[1], -1), rank_n[k[0]]))


def table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def fmt(v, spec: str = "{:.0f}") -> str:
    return "—" if v is None else spec.format(v)


def sorted_versions(env: dict) -> list[str]:
    def key(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())

    return sorted(set(env["downshift_targets"].values()), key=key)


def fixture_section(report: dict) -> list[str]:
    env, cfg = report["environment"], report["config"]
    versions = sorted_versions(env)
    runs = defaultdict(dict)  # (case, batch, variant, version) -> {concurrency: run}
    for r in report["runs"]:
        runs[(r["case"], r["batch"], r["variant"], r["version"])][r["concurrency"]] = r
    cases = [c for c in cfg["cases"] if c in report["verdicts"]]
    out: list[str] = []

    out.append("## Fixture and compute-heavy models\n")
    out.append(
        f"Closed-loop load, {cfg['duration']} s window after {cfg['warmup']} s warm-up, load "
        "generator sharded over up to 6 processes on the same machine. Every server gets the same "
        "seeded weights and byte-identical request bodies, and every row's response is checked "
        "element-wise against eager PyTorch.\n"
    )
    if "client_ceiling_rps" in report:
        ceiling = report["client_ceiling_rps"]
        out.append(
            "Load-generator ceiling (`GET /health` on a do-nothing server): "
            + ", ".join(f"c={c}: {fmt(v)} rps" for c, v in ceiling.items())
            + ". A result near these numbers is measuring the client.\n"
        )

    out.append("### Verdicts\n")
    rows = [
        [
            f"`{c}`",
            report["verdicts"][c]["status"],
            report["verdicts"][c]["backend"],
            fmt(report["verdicts"][c]["max_abs_err"], "{:.1e}"),
        ]
        for c in cases
    ]
    out.append(table(rows, ["model", "verdict", "backend chosen", "max abs err"]))

    out.append("### Peak throughput, batch 1\n")
    out.append(
        "Best requests/s over the concurrency sweep, and p50 latency at concurrency 1. "
        "`async def` blocks the event loop and is shown as the cautionary row.\n"
    )
    servers = server_order({(k[2], k[3]) for k in runs}, versions)
    header = ["server"] + [f"`{c}`" for c in cases]
    peak_rows, p50_rows = [], []
    for variant, version in servers:
        peak, p50 = [name(variant, version)], [name(variant, version)]
        for c in cases:
            sweep = runs.get((c, 1, variant, version), {})
            peak.append(fmt(max((r["throughput_rps"] for r in sweep.values()), default=None)))
            p50.append(fmt((sweep.get(1) or {}).get("p50_ms"), "{:.2f}"))
        peak_rows.append(peak)
        p50_rows.append(p50)
    out.append(table(peak_rows, header))
    out.append("p50 latency (ms) at concurrency 1:\n")
    out.append(table(p50_rows, header))

    out.append("### Concurrency sweep, batch 1 (requests/s)\n")
    for c in cases:
        rows = []
        for variant, version in servers:
            sweep = runs.get((c, 1, variant, version))
            if sweep:
                rows.append(
                    [name(variant, version)]
                    + [fmt((sweep.get(n) or {}).get("throughput_rps")) for n in cfg["concurrency"]]
                )
        out.append(f"`{c}`\n")
        out.append(table(rows, ["server"] + [f"c={n}" for n in cfg["concurrency"]]))

    batches = [b for b in cfg["batch"] if b != 1]
    if batches:
        mid = cfg["batch_concurrency"][len(cfg["batch_concurrency"]) // 2]
        out.append(f"### Batched requests at concurrency {mid} (rows/s)\n")
        out.append("Requests/s times batch size: the work actually done.\n")
        header = ["server"] + [f"`{c}` b{b}" for c in cases for b in batches]
        rows = []
        for variant, version in servers:
            if variant == "naive_torch_async":
                continue
            row = [name(variant, version)]
            for c in cases:
                for b in batches:
                    r = runs.get((c, b, variant, version), {}).get(mid)
                    row.append(fmt(r["throughput_rps"] * b if r else None))
            rows.append(row)
        out.append(table(rows, header))

    wire = [k for k in runs if k[2] in ("downshift", "downshift_base64", "downshift_safetensors")]
    if any(k[2] == "downshift_safetensors" for k in wire):
        b = max(cfg["batch"])
        c = cfg["batch_concurrency"][len(cfg["batch_concurrency"]) // 2]
        out.append(f"### Wire encodings, batch {b}, concurrency {c}\n")
        rows = []
        for case in cases:
            for variant, version in server_order({(k[2], k[3]) for k in wire}, versions):
                r = runs.get((case, b, variant, version), {}).get(c)
                if r:
                    rows.append(
                        [
                            f"`{case}`",
                            name(variant, version),
                            fmt(r["request_bytes"]),
                            fmt(r["throughput_rps"]),
                            fmt(r["p50_ms"], "{:.2f}"),
                        ]
                    )
        out.append(table(rows, ["model", "server", "request bytes", "requests/s", "p50 ms"]))

    if report.get("workers_runs"):
        out.append("### `--workers` sweep, batch 1 (requests/s)\n")
        wr = defaultdict(dict)
        boot = {}
        for r in report["workers_runs"]:
            wr[(r["case"], r["version"], r["workers"])][r["concurrency"]] = r
            boot[(r["case"], r["version"], r["workers"])] = r["boot_s"]
        rows = []
        for case, version, n in sorted(wr, key=lambda k: (k[0], versions.index(k[1]), k[2])):
            rows.append(
                [f"`{case}`", version, str(n)]
                + [
                    fmt((wr[(case, version, n)].get(c) or {}).get("throughput_rps"))
                    for c in cfg["workers_concurrency"]
                ]
                + [fmt(boot[(case, version, n)], "{:.1f}")]
            )
        header = ["model", "downshift", "workers"]
        header += [f"c={c}" for c in cfg["workers_concurrency"]] + ["boot s"]
        out.append(table(rows, header))

    out.append("### Correctness\n")
    out.append("Worst max absolute error against eager PyTorch over every row a server answered.\n")
    worst = defaultdict(dict)
    for r in report["runs"]:
        key = (r["variant"], r["version"])
        worst[key][r["case"]] = max(worst[key].get(r["case"], 0.0), r["max_abs_err"])
    rows = [
        [name(*k)] + [fmt(worst[k].get(c), "{:.1e}") for c in cases] for k in servers if k in worst
    ]
    out.append(table(rows, ["server"] + [f"`{c}`" for c in cases]))
    every = report["runs"] + report.get("workers_runs", [])
    out.append(f"Failed requests across every load window: {sum(r['errors'] for r in every)}.\n")
    for r in (r for r in every if r["errors"]):
        where = f"`{r['case']}`, {name(r['variant'], r['version'])}"
        if "workers" in r:
            where += f" `--workers {r['workers']}`"
        out.append(
            f"- {where}, batch {r['batch']}, c={r['concurrency']}: {r['errors']} of "
            f"{r['ok'] + r['errors']} failed (`{r.get('first_error')}`); that window lasted "
            f"{r['window_s']:.1f} s against {cfg['duration']} s, so its requests/s is understated.\n"
        )
    if "resumed" in report:
        res = report["resumed"]
        out.append(
            f"This run was split in two: {res['why']}, so {res['second_pass_covers']} come from "
            f"a second pass ({res['second_pass']}).\n"
        )
    return out


def hf_section(hf: dict) -> list[str]:
    versions = sorted_versions(hf["environment"])
    cfg = hf["config"]
    out = ["## Real Hugging Face models\n"]
    out.append(
        f"Text in, through the real CLI. {cfg['duration']} s windows at concurrency "
        f"{', '.join(map(str, cfg['concurrency']))}; a `/health` probe runs every 50 ms during "
        "each window (what a liveness probe sees). Payloads: `short_b1` one sentence, `short_b8` "
        "eight sentences, `long_b1` ~180 tokens.\n"
    )
    for model, entry in hf["models"].items():
        out.append(f"### {model}\n")
        checks = entry["check"]
        out.append(
            "`downshift check` wall time: "
            + ", ".join(
                f"{v} {checks[v]['wall_s']:.1f} s (exit {checks[v]['exit_code']})" for v in checks
            )
            + ".\n"
        )
        servers = entry["servers"]
        keys = server_order({(s["variant"], s["version"]) for s in servers.values()}, versions)
        by_key = {(s["variant"], s["version"]): s for s in servers.values()}
        payloads = list(dict.fromkeys(r["payload"] for s in servers.values() for r in s["runs"]))
        top = max(cfg["concurrency"])
        rows = []
        for k in keys:
            s = by_key[k]
            at = {(r["payload"], r["concurrency"]): r for r in s["runs"]}
            row = [name(*k), fmt(s["ready_s"], "{:.1f}")]
            for p in payloads:
                row.append(fmt((at.get((p, top)) or {}).get("throughput_rps")))
            row.append(fmt(max(r["health"].get("p99_ms", 0) for r in s["runs"]), "{:.0f}"))
            rows.append(row)
        header = ["server", "ready s"] + [f"{p} rps c={top}" for p in payloads]
        header.append("worst /health p99 ms")
        out.append(table(rows, header))
        low = min(cfg["concurrency"])
        rows = []
        for k in keys:
            at = {(r["payload"], r["concurrency"]): r for r in by_key[k]["runs"]}
            rows.append(
                [name(*k)]
                + [fmt((at.get((p, low)) or {}).get("p50_ms"), "{:.1f}") for p in payloads]
            )
        out.append(f"p50 latency (ms) at concurrency {low}:\n")
        out.append(table(rows, ["server"] + payloads))
        ref = entry["reference"]
        rows = []
        for k in keys:
            label = next(lbl for lbl, s in servers.items() if (s["variant"], s["version"]) == k)
            cmp = entry["agreement"].get(label)
            if cmp is None:
                continue
            extra = cmp.get("min_cosine", cmp.get("label_agreement"))
            extra = f"{extra:.7f}" if isinstance(extra, float) else extra
            rows.append(
                [name(*k), fmt(cmp["max_abs_err"], "{:.1e}"), "—" if extra is None else extra]
            )
        ref_s = servers[ref]
        out.append(f"Agreement with {name(ref_s['variant'], ref_s['version'])} on `short_b8`:\n")
        third = "min cosine" if model.startswith("all-MiniLM") else "label agreement"
        out.append(table(rows, ["server", "max abs err", third]))
    return out


def stage_entries(report: dict, hf: dict | None) -> list[dict]:
    """Every stage split in both files, flattened: model, server, encoding and the split."""
    entries = [
        {"suite": "fixture", "model": e["case"], "server": plain(e["variant"], e["version"]), **e}
        for e in report.get("stages", [])
    ]
    for model, entry in (hf or {}).get("models", {}).items():
        for srv in entry["servers"].values():
            split = srv.get("stages_short_b1")
            if split:
                entries.append(
                    {
                        "suite": "hf",
                        "model": model,
                        "server": plain(srv["variant"], srv["version"]),
                        "variant": srv["variant"],
                        "version": srv["version"],
                        "encoding": "json",
                        "batch": 1,
                        **split,
                    }
                )
    return entries


def stage_table(entries: list[dict]) -> str:
    """One row per server: client total, each Server-Timing stage, and the unaccounted rest."""
    present = [st for st in STAGE_ORDER if any(st in e["stages_ms"] for e in entries)]
    rows = [
        [e["server"], fmt(e["client_ms"], "{:.2f}")]
        + [fmt(e["stages_ms"].get(st), "{:.2f}") for st in present]
        + [fmt(e["unaccounted_ms"], "{:.2f}")]
        for e in entries
    ]
    return table(rows, ["server", "total ms", *present, "HTTP + hand-offs ms"])


def stages_section(report: dict, hf: dict | None) -> list[str]:
    rank = {v: i for i, v in enumerate([*NAIVE_NAMES, *DOWNSHIFT_NAMES])}
    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in sorted(
        stage_entries(report, hf),
        key=lambda e: (e["version"] is not None, e["version"] or "", rank[e["variant"]]),
    ):
        by_model[e["model"]].append(e)
    if not by_model:
        return []
    out = ["## Where the time goes\n"]
    out.append(
        "Median of 300 sequential requests on one keep-alive connection, batch 1 (HF: one "
        "sentence). `total` is what the client measured; the stage columns are the server's own "
        "`Server-Timing`; `HTTP + hand-offs` is the rest: uvicorn, routing, middleware and thread "
        "hand-offs back to the event loop. Naive servers have no `Server-Timing`, so their total "
        "is all in the last column. 0.4.0 reports one `codec` stage (decode and encode); 0.5.0 "
        "splits it, and its `encode` includes serializing the response.\n"
    )
    for model, entries in by_model.items():
        out.append(f"### {model}\n")
        out.append(stage_table(entries))
    return out


RUN_FIELDS = [
    "suite",
    "model",
    "server",
    "variant",
    "version",
    "backend",
    "encoding",
    "payload",
    "batch",
    "workers",
    "concurrency",
    "throughput_rps",
    "rows_per_s",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "errors",
    "window_s",
    "request_bytes",
    "ready_s",
    "max_abs_err",
    "health_p50_ms",
    "health_p99_ms",
]


def run_rows(report: dict, hf: dict | None) -> list[dict]:
    """One row per measured load window, fixture, --workers and HF alike (long format)."""
    rows = []
    for suite, key in (("fixture", "runs"), ("workers", "workers_runs")):
        for r in report.get(key, []):
            workers = r.get("workers", 1)
            server = plain(r["variant"], r["version"])
            rows.append(
                {
                    **r,
                    "suite": suite,
                    "model": r["case"],
                    "server": f"{server} --workers {workers}" if suite == "workers" else server,
                    "version": r["version"] or "",
                    "workers": workers,
                    "rows_per_s": round(r["throughput_rps"] * r["batch"], 1),
                    "ready_s": r.get("server_ready_s", r.get("boot_s")),
                }
            )
    for model, entry in (hf or {}).get("models", {}).items():
        for srv in entry["servers"].values():
            for r in srv["runs"]:
                batch = int(r["payload"].rsplit("_b", 1)[1])
                rows.append(
                    {
                        **r,
                        "suite": "hf",
                        "model": model,
                        "server": plain(srv["variant"], srv["version"]),
                        "variant": srv["variant"],
                        "version": srv["version"] or "",
                        "backend": srv["served_by"],
                        "encoding": "json",
                        "batch": batch,
                        "workers": 1,
                        "rows_per_s": round(r["throughput_rps"] * batch, 1),
                        "ready_s": srv["ready_s"],
                        "health_p50_ms": r["health"].get("p50_ms"),
                        "health_p99_ms": r["health"].get("p99_ms"),
                    }
                )
    return rows


STAGE_FIELDS = [
    "suite",
    "model",
    "server",
    "variant",
    "version",
    "encoding",
    "batch",
    "client_ms",
    "stage",
    "ms",
]


def stage_rows(report: dict, hf: dict | None) -> list[dict]:
    """One row per (server, stage), with the unaccounted rest as stage `http_and_handoffs`:
    the shape a stacked bar chart wants."""
    rows = []
    for e in stage_entries(report, hf):
        base = {k: e.get(k) for k in STAGE_FIELDS[:8]}
        base["version"] = base["version"] or ""
        parts = [(st, e["stages_ms"][st]) for st in STAGE_ORDER if st in e["stages_ms"]]
        for stage, ms in [*parts, ("http_and_handoffs", e["unaccounted_ms"])]:
            rows.append({**base, "stage": stage, "ms": ms})
    return rows


def write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(report: dict, hf: dict | None) -> str:
    env = report["environment"]
    targets = env["downshift_targets"]
    out = [
        "# downshift benchmark: naive servers vs downshift "
        + " vs ".join(sorted_versions(env))
        + "\n"
    ]
    out.append(
        f"`{env['platform']}`, {env['cpu_count']} logical CPUs, Python {env['python']}, "
        f"torch {env['torch']} (CPU), onnxruntime {env['onnxruntime']}, run {env['timestamp']}. "
        + "; ".join(
            f"downshift {v} from the {'checkout' if t == 'checkout' else 'pip-installed release'}"
            for t, v in targets.items()
        )
        + ". Every downshift server is the real `downshift serve` CLI; all servers ran in one "
        "session on the same machine.\n"
    )
    out += fixture_section(report)
    if hf:
        out += hf_section(hf)
    out += stages_section(report, hf)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/scratch/results.json")
    ap.add_argument("--hf-results", default=None)
    ap.add_argument("--out", default="bench/scratch/REPORT.md")
    ap.add_argument("--csv", default=None, help="Every load window as one CSV row")
    ap.add_argument("--stages-csv", default=None, help="Every (server, stage) as one CSV row")
    args = ap.parse_args()
    report = json.loads(Path(args.results).read_text())
    hf = json.loads(Path(args.hf_results).read_text()) if args.hf_results else None
    Path(args.out).write_text(build(report, hf), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    if args.csv:
        write_csv(args.csv, RUN_FIELDS, run_rows(report, hf))
        print(f"wrote {args.csv}")
    if args.stages_csv:
        write_csv(args.stages_csv, STAGE_FIELDS, stage_rows(report, hf))
        print(f"wrote {args.stages_csv}")


if __name__ == "__main__":
    main()
