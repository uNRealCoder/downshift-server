"""Quick throughput check: v0.5.1 vs 0.6.0 (cache on / off), cache-miss and cache-hit traffic.

  python qbench.py <duration_s> <out.json>
"""

import asyncio
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time

sys.path[:0] = ["D:/Projects/downshift-server"]

import aiohttp  # noqa: E402

from bench.cases import encode_request, make_inputs  # noqa: E402
from bench.run import free_port, stop_server, wait_ready  # noqa: E402

ROOT = "D:/Projects/downshift-server"
JOB = "C:/Users/Nikhi/.claude/jobs/569d902c/tmp"
PY = f"{ROOT}/.venv/Scripts/python.exe"
SRC = {"0.5.1": f"{JOB}/v051/src", "0.6.0": f"{ROOT}/src"}
CONFIGS = [("0.5.1", "0.5.1", []), ("0.6.0", "0.6.0", []), ("0.6.0 cache off", "0.6.0", ["--result-cache-rows", "0"])]
CASES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["clean_mlp", "tiny_bert"]
CONCURRENCY, PROCS = 8, 4
POOL = 1500  # distinct bodies for the miss traffic. More than the 100 rows of the cache.


async def _client(url, bodies, headers, n, stop_at, record_from, lat, errors):
    conn = aiohttp.TCPConnector(limit=n + 4)
    async with aiohttp.ClientSession(connector=conn) as s:

        async def worker(k):
            i = k * 37
            while time.perf_counter() < stop_at:
                body = bodies[i % len(bodies)]
                i += 1
                t = time.perf_counter()
                try:
                    async with s.post(url, data=body, headers=headers) as r:
                        await r.read()
                        if r.status != 200:
                            errors.append(r.status)
                        elif t >= record_from:
                            lat.append(time.perf_counter() - t)
                except Exception:
                    errors.append(-1)

        await asyncio.gather(*(worker(k) for k in range(n)))


def _proc(args, q):
    url, bodies, headers, n, duration, warmup = args
    lat, errors = [], []
    start = time.perf_counter()
    asyncio.run(
        _client(url, bodies, headers, n, start + warmup + duration, start + warmup, lat, errors)
    )
    q.put((lat, len(errors)))


def load(port, bodies, headers, duration, warmup=1.0):
    q = mp.Queue()
    url = f"http://127.0.0.1:{port}/predict"
    ps = [
        mp.Process(
            target=_proc, args=((url, bodies, headers, CONCURRENCY // PROCS, duration, warmup), q)
        )
        for _ in range(PROCS)
    ]
    for p in ps:
        p.start()
    lat, errs = [], 0
    for _ in ps:
        a, e = q.get()
        lat += a
        errs += e
    for p in ps:
        p.join()
    lat.sort()
    return {
        "rps": round(len(lat) / duration, 1),
        "p50_ms": round(lat[len(lat) // 2] * 1000, 2),
        "p99_ms": round(lat[int(len(lat) * 0.99)] * 1000, 2),
        "errors": errs,
    }


def serve(case, version, extra):
    port = free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([SRC[version], ROOT])
    proc = subprocess.Popen(
        [PY, "-m", "downshift", "serve", f"bench.factories:{case}", "--inputs",
         f"bench.factories:{case}_inputs", "--port", str(port), "--warmup", "5", "-k", "8",
         "--log-level", "warning", "--no-access-log", *extra],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
    )  # fmt: skip
    wait_ready(port, proc, path="/ready")
    return proc, port


def main():
    duration, out = float(sys.argv[1]), sys.argv[2]
    results = []
    for case in CASES:
        miss = [encode_request(make_inputs(case, 1, seed=s))[0] for s in range(POOL)]
        headers = encode_request(make_inputs(case, 1, seed=0))[1]
        same = [miss[0]]
        for label, version, extra in CONFIGS:
            proc, port = serve(case, version, extra)
            try:
                meta = json.loads(
                    __import__("urllib.request").request.urlopen(
                        f"http://127.0.0.1:{port}/metadata"
                    ).read()
                )
                cache = meta.get("cache")
                for name, bodies in (("distinct rows", miss), ("one row", same)):
                    r = load(port, bodies, headers, duration)
                    r.update(case=case, config=label, traffic=name, cache=bool(cache))
                    results.append(r)
                    print(r, flush=True)
            finally:
                stop_server(proc)
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    mp.freeze_support()
    main()
