"""Condense results.json into the compact shape the published report page embeds.

python -m bench.summarize --results bench/results.json --out bench/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

VARIANTS = ["downshift", "downshift_base64", "naive_onnx", "naive_torch", "naive_torch_async"]


def _p50(block: dict | None, key: str):
    """p50 of one in-process level, or None when an older results.json never measured it."""
    if not block:
        return None
    return (block.get(key) or {}).get("p50_ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results.json")
    ap.add_argument("--out", default="bench/summary.json")
    args = ap.parse_args()

    report = json.loads(Path(args.results).read_text())
    by = defaultdict(dict)
    for r in report["runs"]:
        by[(r["case"], r["batch"], r["variant"])][r["concurrency"]] = r

    cases = list(report["verdicts"])
    concurrencies = report["config"]["concurrency"]
    batches = report["config"]["batch"]

    sweep: dict = {}
    peaks: dict = {}
    for name in cases:
        sweep[name] = {}
        peaks[name] = {}
        for variant in VARIANTS:
            runs = by.get((name, 1, variant))
            if runs:
                sweep[name][variant] = [
                    {
                        "c": c,
                        "rps": runs[c]["throughput_rps"],
                        "p50": runs[c]["p50_ms"],
                        "p99": runs[c]["p99_ms"],
                    }
                    for c in concurrencies
                    if c in runs
                ]
            peaks[name][variant] = {}
            for batch in batches:
                b_runs = by.get((name, batch, variant))
                if not b_runs:
                    continue
                best = max(b_runs.values(), key=lambda r: r["throughput_rps"])
                peaks[name][variant][str(batch)] = {
                    "rps": best["throughput_rps"],
                    "at": best["concurrency"],
                    "p50": best["p50_ms"],
                    "p99": best["p99_ms"],
                    "err": best["max_abs_err"],
                }

    inproc = {}
    for d in report.get("inproc", []):
        entry = {
            "torch": _p50(d["torch"], "compute"),
            "torch_json": _p50(d["torch"], "compute_json"),
            "torch_binary": _p50(d["torch"], "compute_binary"),
            "bytes": d["request_bytes"],
            "bytes_binary": d.get("request_bytes_binary"),
        }
        if d.get("onnxruntime"):
            entry["ort"] = _p50(d["onnxruntime"], "compute")
            entry["ort_json"] = _p50(d["onnxruntime"], "compute_json")
            entry["ort_binary"] = _p50(d["onnxruntime"], "compute_binary")
        inproc[f"{d['case']}|{d['batch']}"] = entry

    out = {
        "environment": report["environment"],
        "config": {
            "concurrency": concurrencies,
            "batches": batches,
            "duration": report["config"]["duration"],
            "warmup": report["config"]["warmup"],
        },
        "ceiling": report.get("client_ceiling_rps", {}),
        "verdicts": report["verdicts"],
        "sweep": sweep,
        "peaks": peaks,
        "inproc": inproc,
        "n_runs": len(report["runs"]),
    }
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
