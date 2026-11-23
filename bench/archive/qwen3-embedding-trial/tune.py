"""Boot `downshift serve` with each config, load it closed-loop, print one JSON line per run.

usage: python tune.py OUT.jsonl DURATION_S CLIENTS(comma) CONFIG_NAME=FLAGS ...
"""
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

import aiohttp

ROOT = "D:/Projects/downshift-server"
PY = ROOT + "/.venv/Scripts/python.exe"
REPO = ROOT + "/bench/models/Qwen3-Embedding-0.6B"
REPOS = {}  # config name -> repo, from the REPOS env var "name:path,name:path"
for pair in filter(None, os.environ.get("REPOS", "").split(",")):
    name, path = pair.split(":", 1)
    REPOS[name] = path if os.path.isabs(path) else ROOT + "/" + path
TMP = os.path.dirname(os.path.abspath(__file__))
PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"

DOC = (
    "Beijing is the capital of the People's Republic of China. With over 21 million residents, "
    "it is the world's most populous national capital city and is China's second largest city "
    "after Shanghai. It is located in Northern China, and is governed as a municipality under "
    "the direct administration of the State Council with 16 urban, suburban, and rural districts. "
    "Beijing is mostly surrounded by Hebei Province with the exception of neighboring Tianjin to "
    "the southeast; together, the three divisions form the Jingjinji megalopolis and the national "
    "capital region of China. Beijing is a global city and one of the world's leading centres "
    "for culture, diplomacy, politics, finance, business and economics, education, research, "
    "language, tourism, media, sport, science and technology and transportation."
)
WORKLOADS = {  # name -> (body, window multiplier, rows per request)
    "query": ({"text": ["what is the population of the capital of china"]}
              | ({} if os.environ.get("NO_PROMPT") else {"prompt_name": "query"}), 1, 1),
    "doc1": ({"text": [DOC]}, 1, 1),
    "docs8": ({"text": [f"{i}. {DOC}" for i in range(8)]}, 3, 8),
    "docs32": ({"text": [f"{i}. {DOC}" for i in range(32)]}, 3, 32),
    "long2k": ({"text": [" ".join([DOC] * 14)]}, 3, 1),
}
if os.environ.get("WORKLOADS"):
    WORKLOADS = {k: v for k, v in WORKLOADS.items() if k in os.environ["WORKLOADS"].split(",")}


async def load(body: dict, clients: int, duration: float, warmup: float = 3.0) -> dict:
    data = json.dumps(body).encode()
    lat, errors, first_err = [], 0, None
    t0 = time.perf_counter()
    record_from, deadline = t0 + warmup, t0 + warmup + duration
    timeout = aiohttp.ClientTimeout(total=300)

    async def worker(session):
        nonlocal errors, first_err
        while (start := time.perf_counter()) < deadline:
            try:
                async with session.post(BASE + "/predict", data=data,
                                        headers={"content-type": "application/json"}) as r:
                    payload = await r.read()
            except Exception as exc:  # a reset or timeout is a result, not a crash
                errors += 1
                first_err = first_err or f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(0.5)
                continue
            end = time.perf_counter()
            if r.status != 200:
                errors += 1
                first_err = first_err or f"{r.status} {payload[:160]!r}"
            elif start >= record_from and end <= deadline:  # completed inside the window
                lat.append(end - start)

    async with aiohttp.ClientSession(timeout=timeout) as s:
        await asyncio.gather(*(worker(s) for _ in range(clients)))
    lat.sort()
    window = duration
    pct = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))] * 1000, 1) if lat else None
    return {"rps": round(len(lat) / window, 2), "p50_ms": pct(0.5), "p95_ms": pct(0.95),
            "n": len(lat), "errors": errors, "first_error": first_err}


def rss_mb(pid: int) -> int:
    """Total working set of the server and its children (workers)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"$all=Get-CimInstance Win32_Process;$ids=@({pid});$q=@({pid});"
         "while($q){$p=$q[0];$q=@($q|select -Skip 1);"
         "$c=@(($all|?{$_.ParentProcessId -eq $p}).ProcessId);$ids+=$c;$q+=$c};"
         "(Get-Process -Id $ids -ErrorAction SilentlyContinue|Measure-Object WorkingSet64 -Sum).Sum"],
        capture_output=True, text=True).stdout.strip()
    return int(int(out or 0) / 2**20)


def ready(timeout: float) -> float | None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if urllib.request.urlopen(BASE + "/ready", timeout=2).status == 200:
                return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(1)
    return None


def main() -> None:
    out_path, duration, clients = sys.argv[1], float(sys.argv[2]), [int(c) for c in sys.argv[3].split(",")]
    env = {**os.environ, "PYTHONPATH": ROOT + "/src"}
    for spec in sys.argv[4:]:
        name, flags = spec.split("=", 1)
        log = open(os.path.join(TMP, f"serve_{name}.log"), "w")
        if flags.startswith("@"):  # a hand-written server: @script.py args -> script REPO PORT args
            script, *args = shlex.split(flags[1:])
            cmd = [PY, "-I", os.path.join(TMP, script), REPOS.get(name, REPO), str(PORT), *args]
        else:
            cmd = [PY, "-m", "downshift", "serve", REPOS.get(name, REPO), "--port", str(PORT), *shlex.split(flags)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            boot = ready(600)
            row = {"config": name, "flags": flags, "boot_s": boot and round(boot, 1)}
            if boot is None:
                row["error"] = "not ready in 600 s"
                print(json.dumps(row), flush=True)
                continue
            row["rss_mb_idle"] = rss_mb(proc.pid)
            for wl, (body, mult, rows) in WORKLOADS.items():
                for c in clients:
                    r = asyncio.run(load(body, c, duration * mult))
                    res = {**row, "workload": wl, "clients": c, **r,
                           "texts_per_s": round(r["rps"] * rows, 2), "rss_mb": rss_mb(proc.pid)}
                    print(json.dumps(res), flush=True)
                    with open(out_path, "a") as f:
                        f.write(json.dumps(res) + "\n")
        finally:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            log.close()
            time.sleep(3)


main()
