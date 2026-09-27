"""Run the matrix and write results.json.

  python -m bench.run --out bench/results.json

For each (case, variant) it starts one server process, waits for it to be ready, then sweeps
concurrency and batch size against it. Before trusting any throughput number it calibrates the
load generator against a trivial /health endpoint, so a result that is really a client-side
ceiling shows up as one instead of masquerading as a server limit.
"""

from __future__ import annotations

import argparse
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

from bench._path import ROOT  # noqa: E402,F401  (must be first: fixes sys.path)
from bench.cases import (  # noqa: E402
    CASE_NAMES,
    decode_array,
    make_inputs,
    prepare,
    to_payload,
    torch_reference,
)

PYTHON = sys.executable
HTTP_VARIANTS = ("naive_torch", "naive_onnx", "downshift", "downshift_base64")
EXTRA_VARIANTS = ("naive_torch_async",)
MAX_LOADGEN_PROCS = 6

# `downshift_base64` is the same `downshift` server process driven with base64 tensor bodies
# (`to_payload(..., encoding="base64")`), so the only variable against `downshift` is the wire
# format. Every other variant sends the nested-list JSON body.
VARIANT_SERVER = {"downshift_base64": "downshift"}
VARIANT_ENCODING = {"downshift_base64": "base64"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(
    port: int, proc: subprocess.Popen, timeout: float = 300.0, path: str = "/health"
) -> float:
    """Poll `path` until the server answers 200. Returns seconds to ready (export + warmup).

    `/ready` answers 503 while the model is still loading, and `urlopen` raises that 503 as an
    `HTTPError` rather than returning it, so it needs its own branch to be treated as "keep
    polling" instead of falling through to the generic `URLError` handler (which it happens to
    be a subclass of, but relying on that would be an accident, not a contract).
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


def post_once(port: int, payload: dict, path: str = "/predict") -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


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


def run_load(url: str, payload: dict, concurrency: int, duration: float, warmup: float) -> dict:
    shards = split_concurrency(concurrency)
    procs = []
    for shard in shards:
        cfg = {
            "url": url,
            "payload": payload,
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
        out = p.stdout.read()
        err = p.stderr.read()
        p.wait()
        if not out:
            raise RuntimeError(
                f"loadgen produced nothing: {err.decode('utf-8', 'replace')[-2000:]}"
            )
        results.append(json.loads(out))

    latencies = sorted(v for r in results for v in r["latencies_ms"])
    ok = sum(r["ok"] for r in results)
    errors = sum(r["errors"] for r in results)
    window = statistics.fmean(r["window_s"] for r in results)
    return {
        "concurrency": concurrency,
        "loadgen_procs": len(shards),
        "ok": ok,
        "errors": errors,
        "window_s": round(window, 3),
        "throughput_rps": round(ok / window, 2) if window else 0.0,
        "p50_ms": round(percentile(latencies, 0.50), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "p99_ms": round(percentile(latencies, 0.99), 3),
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else float("nan"),
        "request_bytes": results[0]["request_bytes"],
    }


def max_abs_err(response: dict, reference: list[np.ndarray]) -> float:
    """Score a response against eager torch. Accepts nested-list or base64 outputs."""
    outputs = response.get("outputs", {})
    errs = []
    for i, ref in enumerate(reference):
        got = outputs.get(f"output_{i}")
        if got is None:
            return float("inf")
        try:
            arr = decode_array(got).astype(np.float64)
        except Exception:
            return float("inf")
        if arr.shape != ref.shape:
            return float("inf")
        errs.append(float(np.max(np.abs(arr - ref.astype(np.float64)))))
    return max(errs) if errs else float("nan")


def start_server(
    variant: str, case_name: str, workers: int = 1
) -> tuple[subprocess.Popen, int, float]:
    port = free_port()
    if variant == "downshift_cli":
        # --workers only exists on the real CLI: bench.servers builds the app in-process and
        # runs a single uvicorn worker, so this variant launches `downshift serve` itself and
        # hands it a case built from bench.factories, so every worker's weights are seeded the
        # same way as the in-process variants.
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
        proc = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "downshift",
                "serve",
                f"bench.factories:{case_name}",
                "--inputs",
                f"bench.factories:{case_name}_inputs",
                "--host",
                "127.0.0.1",
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
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
        )
        ready_s = wait_ready(port, proc, path="/ready")
        return proc, port, ready_s

    # /health answers 200 before the model is loaded (bind-first, since 0.4.0), so only /ready
    # is a valid readiness signal for the downshift variants; the naive servers have no loader
    # thread and no /ready, so /health is still correct for them.
    path = "/ready" if VARIANT_SERVER.get(variant, variant) == "downshift" else "/health"
    proc = subprocess.Popen(
        [PYTHON, "-m", "bench.servers", VARIANT_SERVER.get(variant, variant), case_name, str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    ready_s = wait_ready(port, proc, path=path)
    return proc, port, ready_s


def stop_server(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        # The CLI's --workers>1 path spawns uvicorn worker child processes that proc.terminate()
        # does not reach; they keep the port bound after the parent exits. Kill the whole tree.
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


def calibrate_client(concurrencies: list[int], duration: float, warmup: float) -> dict[str, float]:
    """Ceiling check: how fast can this load generator drive /health on a do-nothing server?"""
    proc, port, _ = start_server("naive_torch", "clean_mlp")
    try:
        out = {}
        for c in concurrencies:
            shards = split_concurrency(c)
            procs = []
            for shard in shards:
                cfg = {
                    "url": f"http://127.0.0.1:{port}/health",
                    "payload": {},
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
                data = p.stdout.read()
                p.stderr.read()
                p.wait()
                results.append(json.loads(data))
            # /health is a GET route; POSTing to it 405s, which still measures the round trip
            total = sum(r["ok"] + r["errors"] for r in results)
            window = statistics.fmean(r["window_s"] for r in results)
            out[str(c)] = round(total / window, 1) if window else 0.0
        return out
    finally:
        stop_server(proc)


def environment() -> dict:
    import onnxruntime as ort
    import torch

    import downshift
    from bench._path import verify

    return {
        "downshift": verify(),
        "downshift_path": downshift.__file__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "ort_providers": ort.get_available_providers(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results.json")
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
        "--http-cases",
        nargs="*",
        default=None,
        help="Subset of --cases to drive over HTTP (default: all)",
    )
    ap.add_argument("--variants", nargs="*", default=list(HTTP_VARIANTS) + list(EXTRA_VARIANTS))
    ap.add_argument("--workers-sweep", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--workers-cases", nargs="*", default=["clean_mlp", "mlp_large", "bert_small"])
    ap.add_argument("--workers-concurrency", nargs="*", type=int, default=[1, 8, 32])
    ap.add_argument("--skip-workers", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "environment": environment(),
        "config": vars(args),
        "runs": [],
        "inproc": [],
        "verdicts": {},
        "workers_runs": [],
    }

    print("== verdicts ==", flush=True)
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
                    print(
                        f"  !! {case_name} b{batch}: {proc.stderr.decode('utf-8', 'replace')[-500:]}",
                        flush=True,
                    )
                    continue
                data = json.loads(proc.stdout)
                report["inproc"].append(data)
                t = data["torch"]["compute"]["p50_ms"]
                t_j = data["torch"]["compute_json"]["p50_ms"]
                t_b = data["torch"]["compute_binary"]["p50_ms"]
                o = data.get("onnxruntime", {}).get("compute", {}).get("p50_ms")
                print(
                    f"  {case_name:<28} b{batch:<3} torch {t:8.3f} ms  +json {t_j:8.3f} ms  "
                    f"+binary {t_b:8.3f} ms   ort {o if o is None else f'{o:8.3f}'} ms",
                    flush=True,
                )

    plan: list[tuple[int, int]] = [(1, c) for c in args.concurrency]
    for batch in args.batch:
        if batch == 1:
            continue
        plan += [(batch, c) for c in args.batch_concurrency]

    variants = list(args.variants)
    http_cases = args.http_cases if args.http_cases is not None else args.cases
    total = len(http_cases) * len(variants)
    done = 0
    for case_name in http_cases:
        case = prepare(case_name, export=True)
        for variant in variants:
            done += 1
            if variant == "naive_onnx" and case.onnx_bytes is None:
                print(f"[{done}/{total}] skip {case_name}/{variant}: no ONNX graph", flush=True)
                continue
            print(f"[{done}/{total}] {case_name} / {variant}", flush=True)
            try:
                proc, port, ready_s = start_server(variant, case_name)
            except Exception as exc:
                print(f"    !! failed to start: {exc}", flush=True)
                continue
            try:
                meta = None
                if VARIANT_SERVER.get(variant, variant) == "downshift":
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/metadata", timeout=30
                        ) as r:
                            meta = json.loads(r.read())
                    except Exception:
                        meta = None

                steps = (
                    plan if variant != "naive_torch_async" else [(1, c) for c in args.concurrency]
                )
                for batch, concurrency in steps:
                    feeds = make_inputs(case_name, batch, seed=7)
                    payload = to_payload(feeds, encoding=VARIANT_ENCODING.get(variant, "json"))
                    # correctness is scored on the raw feeds, independent of the wire encoding
                    reference = torch_reference(case, feeds)
                    try:
                        single = post_once(port, payload)
                        err = max_abs_err(single, reference)
                    except Exception as exc:
                        print(
                            f"    !! b{batch} c{concurrency} single request failed: {exc}",
                            flush=True,
                        )
                        continue
                    res = run_load(
                        f"http://127.0.0.1:{port}/predict",
                        payload,
                        concurrency,
                        args.duration,
                        args.warmup,
                    )
                    res.update(
                        case=case_name,
                        variant=variant,
                        batch=batch,
                        server_ready_s=round(ready_s, 2),
                        max_abs_err=err,
                        backend=(meta or {}).get("backend", {}).get("name") if meta else None,
                        verdict=case.verdict_status,
                        encoding=VARIANT_ENCODING.get(variant, "json"),
                    )
                    report["runs"].append(res)
                    print(
                        f"    b{batch:<3} c{concurrency:<3} {res['throughput_rps']:>9.1f} rps  "
                        f"p50 {res['p50_ms']:>7.2f}  p99 {res['p99_ms']:>8.2f}  "
                        f"err {res['errors']}  maxabs {err:.2e}",
                        flush=True,
                    )
                    out_path.write_text(json.dumps(report, indent=1))
            finally:
                stop_server(proc)

    if not args.skip_workers:
        print("== workers sweep ==", flush=True)
        for case_name in args.workers_cases:
            case = prepare(case_name, export=True)
            for n in args.workers_sweep:
                print(f"[workers] {case_name} / downshift_cli w{n}", flush=True)
                try:
                    proc, port, boot_s = start_server("downshift_cli", case_name, workers=n)
                except Exception as exc:
                    print(f"    !! failed to start: {exc}", flush=True)
                    continue
                try:
                    for concurrency in args.workers_concurrency:
                        feeds = make_inputs(case_name, 1, seed=7)
                        payload = to_payload(feeds)
                        reference = torch_reference(case, feeds)
                        try:
                            single = post_once(port, payload)
                            err = max_abs_err(single, reference)
                        except Exception as exc:
                            print(
                                f"    !! w{n} c{concurrency} single request failed: {exc}",
                                flush=True,
                            )
                            continue
                        res = run_load(
                            f"http://127.0.0.1:{port}/predict",
                            payload,
                            concurrency,
                            args.duration,
                            args.warmup,
                        )
                        res.update(
                            case=case_name,
                            variant="downshift_cli",
                            workers=n,
                            batch=1,
                            boot_s=round(boot_s, 2),
                            max_abs_err=err,
                            verdict=case.verdict_status,
                        )
                        report["workers_runs"].append(res)
                        print(
                            f"    w{n:<2} c{concurrency:<3} {res['throughput_rps']:>9.1f} rps  "
                            f"p50 {res['p50_ms']:>7.2f}  p99 {res['p99_ms']:>8.2f}  "
                            f"err {res['errors']}  maxabs {err:.2e}  boot {boot_s:.2f}s",
                            flush=True,
                        )
                        out_path.write_text(json.dumps(report, indent=1))
                finally:
                    stop_server(proc)

    out_path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_path} ({len(report['runs'])} runs)", flush=True)


if __name__ == "__main__":
    main()
