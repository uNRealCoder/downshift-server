"""The acceptance check of the result cache (0.6.0, step C6). It is not part of CI.

  python -m bench.cache_accept --out bench/results/v0.6.0/cache_acceptance.json

It serves all-MiniLM-L6-v2 in this process, with the cache on and with the cache off, and
measures three things with a TestClient (the numbers are the server-side stages of the
`Server-Timing` header, so the client does not count):

  a. A query that was seen before: the server time is under 1 ms.
  b. A batch of 32 rows with 30 rows that are known: the model runs on 2 rows.
  c. Unique queries: the cache adds no more than noise to a request that always misses.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

from bench._path import ROOT  # noqa: E402
from downshift.loading import LoadSpec, load_model  # noqa: E402
from downshift.serve.app import build_app  # noqa: E402
from downshift.serve.engine import prepare_serving  # noqa: E402
from downshift.serve.options import ServeOptions  # noqa: E402

MODEL = ROOT / "bench" / "models" / "all-MiniLM-L6-v2"
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho".split()


def query(i: int) -> str:
    """A short query that is different for each i (about 10 tokens)."""
    return " ".join(WORDS[(i * 3 + k) % len(WORDS)] for k in range(8)) + f" number {i}"


def server_ms(response) -> float:
    """The sum of the stages of Server-Timing, in ms."""
    return sum(float(v) for v in re.findall(r"dur=([\d.]+)", response.headers["Server-Timing"]))


def serve(rows: int | None):
    loaded = load_model(LoadSpec(str(MODEL)))
    state = prepare_serving(loaded, ServeOptions(result_cache_rows=rows, server_timing=True))
    calls: list[int] = []
    real = state.backend.infer

    def counting(inputs):
        calls.append(len(next(iter(inputs.values()))))
        return real(inputs)

    state.backend.infer = counting  # type: ignore[method-assign]
    return TestClient(build_app(state)), calls


def post(client, texts):
    resp = client.post("/predict", json={"text": texts})
    assert resp.status_code == 200, resp.text
    return resp


def median(values):
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("bench/results/v0.6.0/cache_acceptance.json")
    )
    parser.add_argument("--repeats", type=int, default=60)
    args = parser.parse_args()

    on, calls_on = serve(None)
    off, _ = serve(0)

    # a. a query that was seen before
    for _ in range(2):
        post(on, [query(0)])  # the second sighting stores the row
    hit = [server_ms(post(on, [query(0)])) for _ in range(args.repeats)]
    miss_off = [server_ms(post(off, [query(0)])) for _ in range(args.repeats)]

    # b. a batch of 32 with 30 known rows
    known = [query(1000 + i) for i in range(30)]
    for _ in range(2):
        post(on, known)
    calls_on.clear()
    batch = known + [query(2000), query(2001)]
    post(on, batch)
    rows_computed = list(calls_on)

    # c. unique queries: every request misses
    unique_on = [server_ms(post(on, [query(10_000 + i)])) for i in range(args.repeats)]
    unique_off = [server_ms(post(off, [query(20_000 + i)])) for i in range(args.repeats)]

    result = {
        "model": MODEL.name,
        "repeats": args.repeats,
        "a_repeated_query_server_ms": {
            "hit_median": median(hit),
            "hit_max": max(hit),
            "no_cache_median": median(miss_off),
        },
        "b_batch_32_with_30_known": {"rows_in_the_model_call": rows_computed},
        "c_unique_queries_server_ms": {
            "cache_on_median": median(unique_on),
            "cache_off_median": median(unique_off),
        },
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result["accepted"] = {
        "a": result["a_repeated_query_server_ms"]["hit_median"] < 1.0,
        "b": rows_computed == [2],
        "c": result["c_unique_queries_server_ms"]["cache_on_median"]
        <= 1.05 * result["c_unique_queries_server_ms"]["cache_off_median"] + 0.2,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
