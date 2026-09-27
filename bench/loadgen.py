"""Closed-loop load generator. One process, N coroutines, fixed wall-clock window.

Run as a subprocess: JSON config on stdin, JSON result on stdout. `bench/run.py` fans out
several of these so the client is never the bottleneck (and calibrates that assumption
against /health before trusting any number).

Closed loop means N clients each send the next request as soon as the last one returns, so
throughput and latency are two views of the same measurement rather than independent knobs.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

import aiohttp


async def _worker(
    session: aiohttp.ClientSession,
    url: str,
    body: bytes,
    deadline: float,
    record_from: float,
    latencies: list[float],
    counters: dict[str, int],
    sample_out: list,
) -> None:
    headers = {"content-type": "application/json"}
    while True:
        now = time.perf_counter()
        if now >= deadline:
            return
        start = now
        try:
            async with session.post(url, data=body, headers=headers) as resp:
                payload = await resp.read()
                elapsed = time.perf_counter() - start
                if resp.status != 200:
                    counters["errors"] += 1
                    if len(sample_out) < 1:
                        sample_out.append(
                            {
                                "status": resp.status,
                                "body": payload[:400].decode("utf-8", "replace"),
                            }
                        )
                    continue
                if start >= record_from:
                    latencies.append(elapsed)
                    counters["ok"] += 1
                if not sample_out:
                    sample_out.append(json.loads(payload))
        except Exception as exc:  # connection resets under load are a result, not a crash
            counters["errors"] += 1
            if len(sample_out) < 1:
                sample_out.append({"error": f"{type(exc).__name__}: {exc}"})


async def _run(cfg: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(cfg["payload"]).encode()
    concurrency = cfg["concurrency"]
    connector = aiohttp.TCPConnector(limit=concurrency + 8, force_close=False, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=cfg.get("timeout", 120))
    latencies: list[float] = []
    counters = {"ok": 0, "errors": 0}
    sample_out: list = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start = time.perf_counter()
        record_from = start + cfg["warmup_s"]
        deadline = record_from + cfg["duration_s"]
        tasks = [
            _worker(
                session, cfg["url"], body, deadline, record_from, latencies, counters, sample_out
            )
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        window = time.perf_counter() - record_from

    return {
        "ok": counters["ok"],
        "errors": counters["errors"],
        "window_s": window,
        "latencies_ms": [round(v * 1000, 4) for v in latencies],
        "sample": sample_out[0] if sample_out else None,
        "request_bytes": len(body),
    }


def main() -> None:
    cfg = json.load(sys.stdin)
    result = asyncio.run(_run(cfg))
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
