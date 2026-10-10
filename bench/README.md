# Benchmark harness

`downshift serve` against the FastAPI servers people write by hand, and against the previous
downshift release, on the same models, the same weights and byte-identical request bodies.

The question it answers is not "is downshift faster". downshift picks a backend from a
correctness verdict, so the harness measures what that costs and what it buys: throughput,
latency, boot time and `/health` responsiveness under load, with every response checked against
eager PyTorch. A fast row with a wrong answer shows up as one.

## Layout

```
bench/
  README.md
  run.py, hf_models.py, report.py   entry points (python -m bench.<name>)
  *.py, large/                      the harness modules they use (see "What runs")
  results/v0.5.0/                   a published run: REPORT.md, results*.json, chart_*.csv
  REPORT_v0.4.0.md, BLOG_APPENDIX_v0.4.0.md, results*_v0.4.0.json
                                    the 0.4.0 run, left at the root because the post links here
  models/                           downloaded Hugging Face checkpoints (gitignored)
  scratch/                          default output of an ad-hoc run, local trials (gitignored)
```

A new published run goes in `results/vX.Y.Z/` with the same file names as 0.5.0.

## Published runs

| run | report | raw data |
|---|---|---|
| **0.5.0**: naive vs 0.4.0 vs 0.5.0, one session | [`results/v0.5.0/REPORT.md`](results/v0.5.0/REPORT.md) | `results/v0.5.0/`: `results.json`, `results_hf.json`; chart data `chart_runs.csv`, `chart_stages.csv` |
| 0.4.0: naive vs 0.4.0, behind the launch post *Stop hand-rolling model servers* | [`REPORT_v0.4.0.md`](REPORT_v0.4.0.md), [`BLOG_APPENDIX_v0.4.0.md`](BLOG_APPENDIX_v0.4.0.md) | `results_v0.4.0.json`, `results_hf*_v0.4.0.json` |

The 0.4.0 files were produced by the harness as of commit `02419cf` and are kept unchanged
because the post links to them. Their numbers are not comparable row-for-row with the 0.5.0 run,
which re-measured 0.4.0 alongside 0.5.0 in the same session.

## 0.5.0 at a glance

From [`results/v0.5.0/REPORT.md`](results/v0.5.0/REPORT.md) (run 2026-10-04/05); 16-CPU
Windows machine, CPU only, peak requests/s at batch 1 unless stated. Chart data:
`results/v0.5.0/chart_runs.csv`, `results/v0.5.0/chart_stages.csv`.

- **0.5.0 is faster than 0.4.0 on every model.** Peak batch-1 throughput, JSON bodies:
  `clean_mlp` 1030 vs 866 (+19%), `dynamic_batch_cnn` 1031 vs 862, `gnn_gcn` 891 vs 810,
  `tiny_bert` 906 vs 820, `scatter_include_self_false` 1076 vs 940, `mlp_large` 815 vs 696
  (+17%), `cnn_large` 736 vs 657, `bert_small` 106 vs 102. Each request takes two thread hops
  (prep pool, then inference thread) and skips FastAPI's own body handling.
- **Where the time goes** (`clean_mlp`, concurrency 1): 1.00 ms in all, of which 0.63 ms is the
  HTTP stack, 0.19 ms inference, and 0.18 ms parse, prep, waits and encode together; 0.4.0
  takes 1.27 ms. On `bert_small` inference is 7.3 of 10.6 ms, and safetensors cuts the total to
  8.8 ms by shrinking `encode` from 1.8 ms to 0.1 ms.
- **`--execution inline` is the extra gear for tiny models**: `clean_mlp` 1498,
  `scatter_include_self_false` 1545, `dynamic_batch_cnn` 1211, `gnn_gcn` 1160 (all above the
  0.5.0 default). `tiny_bert` is the exception (844 vs 906 default), and on MiniLM it changes
  nothing (281 vs 282), so it is only worth it for models that infer in well under 1 ms.
- **Binary bodies** move the most rows on wide inputs: at batch 32, concurrency 8, `mlp_large`
  does 14.4k rows/s with safetensors against 6.6k as JSON on 0.5.0 (11.2k with 0.4.0 base64);
  `cnn_large` 4.1k vs 1.5k. safetensors matches or beats base64 with bodies about 25% smaller
  (`cnn_large` b32: 393 KB vs 524 KB, 2.0 MB as JSON); on `dynamic_batch_cnn` b32 it is 967 vs
  770 requests/s.
- **Real HF models.** all-MiniLM-L6-v2, one sentence, c=8: 282 vs 239 by default and 483 vs 434
  with `--max-concurrency 4`, against 139 for the hand-rolled transformers server. Prompt Guard
  50 vs 48, and 103 vs 101 with `--max-concurrency 4`, against 16 hand-rolled.
  `--max-concurrency` is still the biggest single lever for small encoders.
- **`--workers` scales both versions**; at 4 workers and concurrency 8, `clean_mlp` 3857 vs 3406,
  `bert_small` 239 vs 227.
- **Correctness is unchanged.** Every downshift row in both versions matches eager PyTorch to
  ≤ 2.7e-6; both HF models agree with eager torch (MiniLM cosine 1.0, Prompt Guard 8/8 labels).
  The hand-rolled ONNX server still returns wrong answers on `scatter_include_self_false`
  (max error 1.2), where both versions route to torch. Failed requests across every window: 0.
- **`/health` stays responsive** under load on every downshift row (worst p99 27 ms). The
  hand-rolled `async def` server reaches 5.2 s on Prompt Guard.
- **One startup hang, in 0.4.0**: `--workers 4` on `bert_small` never answered `/ready` within
  600 s once (it boots in about 24 s otherwise) and was re-run. Together with a low-memory stop
  during `bert_small`, the run was split into passes; the report says which rows came from where.

## Setup

From a checkout, in a venv that also has the previous release pip-installed (non-editable):

```bash
pip install "downshift-server[all]==0.4.0" aiohttp safetensors   # the "installed" target
```

The checkout is never installed: every process the harness starts finds it through `PYTHONPATH`
(`bench/_path.py`). The two Hugging Face checkpoints are downloaded to gitignored paths:

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir bench/models/all-MiniLM-L6-v2
huggingface-cli download meta-llama/Prompt-Guard-86M --local-dir bench/models/Prompt-Guard-86M   # gated: accept the licence first
```

## Run

```bash
# Fixture and compute-heavy models: verdicts, in-process cost, HTTP sweep, --workers sweep (~2 h)
python -m bench.run --out bench/results/v0.5.0/results.json

# Real Hugging Face models through the real CLI (~40 min)
python -m bench.hf_models --out bench/results/v0.5.0/results_hf.json

python -m bench.report --results bench/results/v0.5.0/results.json \
    --hf-results bench/results/v0.5.0/results_hf.json --out bench/results/v0.5.0/REPORT.md \
    --csv bench/results/v0.5.0/chart_runs.csv --stages-csv bench/results/v0.5.0/chart_stages.csv
```

Without `--out`, each command writes to `bench/scratch/`, so a trial run never overwrites a
published one. Both runners take `--targets checkout installed` (the default is both). Drop one to measure a
single version. Other useful flags: `--cases`, `--http-cases`, `--variants`, `--duration`,
`--skip-inproc`, `--skip-calibration`, `--skip-workers` (`bench.run`), `--models` (`bench.hf_models`).

Run nothing else on the machine while it measures; a concurrent test suite more than doubles
latencies. On a 16 GB machine, close memory-heavy apps too: `bert_small` has been stopped by low
memory before.

## Chart data

`bench.report --csv/--stages-csv` flattens both results files into two long-format CSVs, one
measurement per row, so any plotting tool can group and filter without reading the JSON.

**`chart_runs.csv`**: one row per load window.

| column | meaning |
|---|---|
| `suite` | `fixture` (the HTTP sweep), `workers` (the `--workers` sweep), `hf` (real Hugging Face models) |
| `model` | Fixture case or HF model name |
| `server` | Legend label, e.g. `downshift 0.5.0 safetensors`, `naive FastAPI + ONNX Runtime`, `downshift 0.4.0 --workers 4` |
| `variant`, `version` | The same split into the harness variant and the downshift version (empty for naive servers) |
| `backend`, `encoding`, `payload` | What served it, the request body format, and the HF payload (`short_b1`, `short_b8`, `long_b1`) |
| `batch`, `workers`, `concurrency` | Rows per request, worker processes, concurrent clients |
| `throughput_rps`, `rows_per_s` | Requests per second, and requests times batch |
| `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms` | Client-side latency |
| `errors`, `window_s` | Failed requests and the measured window (a long window means a request hung) |
| `request_bytes` | Body size on the wire |
| `ready_s` | Seconds from process start to `/ready` (export, verify, warmup) |
| `max_abs_err` | Worst element-wise error against eager PyTorch on one request of that shape |
| `health_p50_ms`, `health_p99_ms` | HF only: `/health` latency while under load |

Typical charts: throughput against `concurrency` per `server` (filter `suite=fixture`,
`batch=1`, one `model`); `rows_per_s` against `batch` per `encoding`; p99 against
`concurrency`; `throughput_rps` against `workers`; `health_p99_ms` per server on HF.

**`chart_stages.csv`**: where one request's time goes, per server, at batch 1 and
concurrency 1. One row per stage: `parse`, `codec` (0.4.0 only), `prep_wait`, `prep`,
`infer_wait`, `infer`, `encode`, and `http_and_handoffs` (client total minus every stage:
uvicorn, routing, middleware, thread hand-offs). `client_ms` repeats the total on every row.
Stack `ms` by `stage` for one bar per `server`. Naive servers have only `http_and_handoffs`.

## What runs

| module | role |
|---|---|
| `run.py` | Orchestrator for the fixture and compute-heavy models. Calibrates the load generator against `GET /health` first, so a client-side ceiling shows up as one. |
| `hf_models.py` | Real HF checkpoints: `downshift check` per version, boot time, text serving, with a `/health` probe running during every load window. |
| `report.py` | Renders both results files into one Markdown report. Every number comes from the JSON. |
| `_path.py` | Which downshift each process imports: the orchestrator always uses the checkout, and `cli_env(target)` points a `downshift serve` subprocess at the checkout or the installed release. |
| `servers.py`, `naive_hf_server.py` | The hand-rolled baselines: FastAPI + eager torch (sync and `async def`), FastAPI + ONNX Runtime, FastAPI + transformers. |
| `factories.py` | `pkg.module:attr` entry points so `downshift serve` builds the same seeded models as every other process. |
| `loadgen.py` | Closed-loop aiohttp load generator, one process per shard, opaque request bytes. |
| `inproc.py` | No-HTTP baselines: compute, compute + JSON codec, compute + base64 codec. |
| `cases.py`, `large/` | The models and the request codecs. The `tests/models` fixtures are sized for export hazards, so `large/` adds compute-heavy counterparts. All built from a fixed seed. |

### Servers

| server | what it is |
|---|---|
| `naive_torch`, `naive_onnx`, `naive_torch_async` | Hand-rolled FastAPI, sync `def` (threadpool) or `async def` (blocks the event loop). |
| `downshift@V` | `downshift serve` at version V, defaults, JSON bodies. |
| `downshift_base64@V` | Same server, base64 tensor bodies and responses. |
| `downshift_safetensors@V` | Same server, safetensors request and response (0.5.0+). |
| `downshift_inline@V` | `--execution inline` (0.5.0+), on the small fixture tier only: running the model on the event loop only pays off when compute is short. |
| HF: `downshift_auto`, `downshift_torch`, `downshift_auto_mc4`, `downshift_auto_inline` | Verdict backend, forced eager torch, `--max-concurrency 4`, `--execution inline` (MiniLM only). |

## Fairness rules

- The hand-rolled baselines take a plain `dict` body with no pydantic validation and return a
  plain `dict` with no `response_model`. The HF baseline truncates over-long text instead of
  rejecting it. Each does strictly less work per request than downshift, so overhead
  attributed to downshift is real.
- Every server gets the same seeded weights; `naive_onnx` gets the graph the checkout exported.
  Every response is checked element-wise against eager PyTorch, and every HF answer against
  downshift's `--backend torch`.
- Every downshift server is the real `downshift serve` CLI, never an in-process shortcut, and all
  versions are measured in the same session so machine state is not a variable between them.
