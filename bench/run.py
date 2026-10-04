"""Run the fixture and compute-heavy matrix and write one results JSON.

  python -m bench.run --out bench/results_v0.5.0.json

For each case it starts one server process per variant, waits for it to be ready, then sweeps
concurrency and batch size against it. The naive servers run once; every downshift variant runs
once per `--targets` entry (the checkout and the previously released, pip-installed version), so
one file holds naive, old and new side by side, measured in the same session on the same weights.

Before trusting any throughput number it calibrates the load generator against a trivial /health
endpoint, so a result that is really a client-side ceiling shows up as one instead of
masquerading as a server limit.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from bench._path import ROOT, TARGETS, cli_env, target_version, verify  # noqa: E402
from bench.cases import (  # noqa: E402
    CASE_NAMES,
    FIXTURE_CASES,
    decode_outputs,
    encode_request,
    make_inputs,
    prepare,
    torch_reference,
)

PYTHON = sys.executable
MAX_LOADGEN_PROCS = 6

NAIVE_VARIANTS = ("naive_torch", "naive_onnx", "naive_torch_async")
# Every downshift variant is `downshift serve` on a bench.factories model; they differ only in
# the wire encoding the client uses and the extra CLI flags.
DOWNSHIFT_VARIANTS = {
    "downshift": ("json", []),
    "downshift_base64": ("base64", []),
    "downshift_safetensors": ("safetensors", []),
    "downshift_inline": ("json", ["--execution", "inline"]),
}
NEW_IN = {"downshift_safetensors": (0, 5), "downshift_inline": (0, 5)}
# Inline execution runs the model on the event loop, which only pays off when compute is short;
# it is measured on the small fixture tier only.
INLINE_CASES = FIXTURE_CASES


def version_tuple(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(
    port: int, proc: subprocess.Popen, timeout: float = 600.0, path: str = "/health"
) -> float:
    """Poll `path` until the server answers 200. Returns seconds to ready (export + warmup).

    `/ready` answers 503 while the model is still loading, and `urlopen` raises that 503 as an
    `HTTPError`, so it has its own "keep polling" branch.
    """
    start = time.perf_counter()
    url = f"http://127.0.0.1:{port}{path}"
    while time.perf_counter() - start < timeout:
        if proc.poll() is not None:
            out = (proc.stdout.read() if proc.stdout else b"") or b""
            raise RuntimeError(
                f"server exited early ({proc.returncode}):\n{out.decode('utf-8', 'replace')[-3000:]}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return time.perf_counter() - start
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise
            time.sleep(0.25)
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError(f"server on port {port} never became ready")


def post_once(port: int, body: bytes, headers: dict) -> dict[str, np.ndarray]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/predict", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return decode_outputs(resp.read(), resp.headers.get("content-type", ""))


def split_concurrency(concurrency: int) -> list[int]:
    """Split N clients over at most MAX_LOADGEN_PROCS processes so the client isn't the limit."""
    procs = min(MAX_LOADGEN_PROCS, concurrency)
    while procs > 1 and concurrency % procs:
        procs -= 1
    return [concurrency // procs] * procs


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def loadgen(url: str, body: bytes, headers: dict, concurrency: int, duration: float, warmup: float):
    """Fan `concurrency` clients out over loadgen processes; returns their raw results."""
    procs = []
    for shard in split_concurrency(concurrency):
        cfg = {
            "url": url,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "headers": headers,
            "concurrency": shard,
            "duration_s": duration,
            "warmup_s": warmup,
        }
        p = subprocess.Popen(
            [PYTHON, "-m", "bench.loadgen"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(ROOT),
        )
        p.stdin.write(json.dumps(cfg).encode())
        p.stdin.close()
        procs.append(p)
    results = []
    for p in procs:
        out, err = p.stdout.read(), p.stderr.read()
        p.wait()
        if not out:
            raise RuntimeError(
                f"loadgen produced nothing: {err.decode('utf-8', 'replace')[-2000:]}"
            )
        results.append(json.loads(out))
    return results


def run_load(
    url: str,
    body: bytes,
    headers: dict,
    concurrency: int,
    duration: float,
    warmup: float,
) -> dict:
    results = loadgen(url, body, headers, concurrency, duration, warmup)
    latencies = sorted(v for r in results for v in r["latencies_ms"])
    ok = sum(r["ok"] for r in results)
    window = statistics.fmean(r["window_s"] for r in results)
    return {
        "concurrency": concurrency,
        "loadgen_procs": len(results),
        "ok": ok,
        "errors": sum(r["errors"] for r in results),
        "first_error": next((r["first_error"] for r in results if r["first_error"]), None),
        "window_s": round(window, 3),
        "throughput_rps": round(ok / window, 2) if window else 0.0,
        "p50_ms": round(percentile(latencies, 0.50), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "p99_ms": round(percentile(latencies, 0.99), 3),
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else float("nan"),
        "request_bytes": results[0]["request_bytes"],
    }


def max_abs_err(outputs: dict[str, np.ndarray], reference: list[np.ndarray]) -> float:
    """Score a response against eager torch, element-wise."""
    errs = []
    for i, ref in enumerate(reference):
        got = outputs.get(f"output_{i}")
        if got is None or got.shape != ref.shape:
            return float("inf")
        errs.append(float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)))))
    return max(errs) if errs else float("nan")


def start_naive(variant: str, case_name: str) -> tuple[subprocess.Popen, int, float]:
    port = free_port()
    proc = subprocess.Popen(
        [PYTHON, "-m", "bench.servers", variant, case_name, str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    return proc, port, wait_ready(port, proc)


def start_downshift(
    case_name: str, target: str, extra: list[str], workers: int = 1
) -> tuple[subprocess.Popen, int, float]:
    """`downshift serve` from `target` on a bench.factories model.

    /health answers 200 before the model is loaded (bind-first, since 0.4.0), so /ready is the
    readiness signal; the returned seconds are the full boot: export, verify and warm-up.
    """
    port = free_port()
    proc = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "downshift",
            "serve",
            f"bench.factories:{case_name}",
            "--inputs",
            f"bench.factories:{case_name}_inputs",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--warmup",
            "5",
            "-k",
            "8",
            "--log-level",
            "warning",
            "--no-access-log",
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env=cli_env(target),
    )
    return proc, port, wait_ready(port, proc, path="/ready")


def stop_server(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        # --workers > 1 spawns uvicorn worker children that proc.terminate() does not reach;
        # they keep the port bound after the parent exits. Kill the whole tree.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def served_backend(port: int) -> str | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metadata", timeout=30) as r:
            backend = json.loads(r.read()).get("backend")
    except Exception:
        return None
    return backend.get("name") if isinstance(backend, dict) else backend


def calibrate_client(concurrencies: list[int], duration: float, warmup: float) -> dict[str, float]:
    """Ceiling check: how fast can this load generator drive /health on a do-nothing server?"""
    proc, port, _ = start_naive("naive_torch", "clean_mlp")
    try:
        out = {}
        for c in concurrencies:
            # /health is a GET route; POSTing to it 405s, which still measures the round trip
            results = loadgen(
                f"http://127.0.0.1:{port}/health",
                b"{}",
                {"content-type": "application/json"},
                c,
                duration,
                warmup,
            )
            total = sum(r["ok"] + r["errors"] for r in results)
            window = statistics.fmean(r["window_s"] for r in results)
            out[str(c)] = round(total / window, 1) if window else 0.0
        return out
    finally:
        stop_server(proc)


def environment(versions: dict[str, str]) -> dict:
    import onnxruntime as ort
    import torch

    return {
        "orchestrator_downshift": verify(),
        "downshift_targets": versions,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "ort_providers": ort.get_available_providers(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def sweep(port, case, steps, encoding, args, label: str, **tags) -> list[dict]:
    """Drive one running server through (batch, concurrency) steps; one result row per step.

    Each step first sends a single request and scores it against eager torch, so a fast row
    with a wrong answer is visible as one.
    """
    rows = []
    for batch, concurrency in steps:
        feeds = make_inputs(case.name, batch, seed=7)
        body, headers = encode_request(feeds, encoding)
        try:
            err = max_abs_err(post_once(port, body, headers), torch_reference(case, feeds))
        except Exception as exc:
            print(f"    !! b{batch} c{concurrency} single request failed: {exc}", flush=True)
            continue
        res = run_load(
            f"http://127.0.0.1:{port}/predict",
            body,
            headers,
            concurrency,
            args.duration,
            args.warmup,
        )
        res.update(case=case.name, batch=batch, max_abs_err=err, encoding=encoding, **tags)
        rows.append(res)
        print(
            f"    {label:<34} b{batch:<3} c{concurrency:<3} {res['throughput_rps']:>9.1f} rps  "
            f"p50 {res['p50_ms']:>7.2f}  p99 {res['p99_ms']:>8.2f}  "
            f"err {res['errors']}  maxabs {err:.2e}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results.json")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--cases", nargs="*", default=list(CASE_NAMES))
    ap.add_argument("--concurrency", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--batch", nargs="*", type=int, default=[1, 8, 32])
    ap.add_argument("--batch-concurrency", nargs="*", type=int, default=[1, 8, 32])
    ap.add_argument("--inproc-iters", type=int, default=200)
    ap.add_argument("--skip-inproc", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument(
        "--http-cases", nargs="*", default=None, help="Subset of --cases to drive over HTTP"
    )
    ap.add_argument(
        "--variants", nargs="*", default=list(NAIVE_VARIANTS) + list(DOWNSHIFT_VARIANTS)
    )
    ap.add_argument("--workers-sweep", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--workers-cases", nargs="*", default=["clean_mlp", "mlp_large", "bert_small"])
    ap.add_argument("--workers-concurrency", nargs="*", type=int, default=[1, 8, 32])
    ap.add_argument("--skip-workers", action="store_true")
    args = ap.parse_args()

    versions = {t: target_version(t) for t in args.targets}
    print("== downshift targets ==", flush=True)
    for t, v in versions.items():
        print(f"  {t:<10} {v}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "environment": environment(versions),
        "config": vars(args),
        "verdicts": {},
        "inproc": [],
        "runs": [],
        "workers_runs": [],
    }

    def save() -> None:
        out_path.write_text(json.dumps(report, indent=1))

    print("== verdicts (checkout) ==", flush=True)
    for case_name in args.cases:
        case = prepare(case_name, export=True)
        report["verdicts"][case_name] = {
            "family": case.family,
            "status": case.verdict_status,
            "backend": case.verdict_backend,
            "reason": case.verdict_reason,
            "max_abs_err": case.max_abs_err,
            "input_names": list(case.input_names),
        }
        print(f"  {case_name:<28} {case.verdict_status:<10} -> {case.verdict_backend}", flush=True)

    if not args.skip_calibration:
        print("== client ceiling (GET /health) ==", flush=True)
        report["client_ceiling_rps"] = calibrate_client(
            args.concurrency, args.duration, args.warmup
        )
        print("  " + json.dumps(report["client_ceiling_rps"]), flush=True)

    if not args.skip_inproc:
        print("== in-process (no HTTP) ==", flush=True)
        for case_name in args.cases:
            for batch in args.batch:
                proc = subprocess.run(
                    [PYTHON, "-m", "bench.inproc", case_name, str(batch), str(args.inproc_iters)],
                    capture_output=True,
                    cwd=str(ROOT),
                )
                if proc.returncode != 0 or not proc.stdout:
                    err = proc.stderr.decode("utf-8", "replace")[-500:]
                    print(f"  !! {case_name} b{batch}: {err}", flush=True)
                    continue
                data = json.loads(proc.stdout)
                report["inproc"].append(data)
                t = data["torch"]["compute"]["p50_ms"]
                o = data.get("onnxruntime", {}).get("compute", {}).get("p50_ms")
                ort_ms = "  n/a" if o is None else f"{o:8.3f}"
                print(
                    f"  {case_name:<28} b{batch:<3} torch {t:8.3f} ms   ort {ort_ms} ms", flush=True
                )
        save()

    plan = [(1, c) for c in args.concurrency]
    plan += [(b, c) for b in args.batch if b != 1 for c in args.batch_concurrency]
    async_plan = [(1, c) for c in args.concurrency]

    http_cases = args.http_cases if args.http_cases is not None else args.cases
    for case_name in http_cases:
        case = prepare(case_name, export=True)
        print(f"== {case_name} ==", flush=True)
        servers = [(v, None) for v in args.variants if v in NAIVE_VARIANTS]
        for target in args.targets:
            for v in args.variants:
                if v not in DOWNSHIFT_VARIANTS:
                    continue
                if version_tuple(versions[target]) < NEW_IN.get(v, (0, 0)):
                    continue
                if v == "downshift_inline" and case_name not in INLINE_CASES:
                    continue
                servers.append((v, target))
        for variant, target in servers:
            if variant == "naive_onnx" and case.onnx_bytes is None:
                print(f"  skip {variant}: no ONNX graph", flush=True)
                continue
            version = versions[target] if target else None
            label = f"{variant}@{version}" if version else variant
            try:
                if target:
                    encoding, extra = DOWNSHIFT_VARIANTS[variant]
                    proc, port, ready_s = start_downshift(case_name, target, extra)
                else:
                    encoding = "json"
                    proc, port, ready_s = start_naive(variant, case_name)
            except Exception as exc:
                print(f"  !! {label} failed to start: {exc}", flush=True)
                continue
            try:
                report["runs"] += sweep(
                    port,
                    case,
                    async_plan if variant == "naive_torch_async" else plan,
                    encoding,
                    args,
                    label,
                    variant=variant,
                    version=version,
                    backend=served_backend(port) if target else variant.split("_")[1],
                    server_ready_s=round(ready_s, 2),
                    verdict=case.verdict_status,
                )
                save()
            finally:
                stop_server(proc)

    if not args.skip_workers:
        print("== workers sweep ==", flush=True)
        steps = [(1, c) for c in args.workers_concurrency]
        for case_name in args.workers_cases:
            case = prepare(case_name, export=True)
            for target in args.targets:
                version = versions[target]
                for n in args.workers_sweep:
                    label = f"downshift@{version} w{n} {case_name}"
                    try:
                        proc, port, boot_s = start_downshift(case_name, target, [], workers=n)
                    except Exception as exc:
                        print(f"  !! {label} failed to start: {exc}", flush=True)
                        continue
                    try:
                        report["workers_runs"] += sweep(
                            port,
                            case,
                            steps,
                            "json",
                            args,
                            label,
                            variant="downshift",
                            version=version,
                            workers=n,
                            boot_s=round(boot_s, 2),
                        )
                        save()
                    finally:
                        stop_server(proc)

    save()
    print(f"\nwrote {out_path} ({len(report['runs'])} runs)", flush=True)


if __name__ == "__main__":
    main()
