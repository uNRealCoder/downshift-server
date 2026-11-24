# Benchmark harness

`downshift serve` against the FastAPI servers people write by hand, and against the previous
downshift release, on the same models, the same weights and byte-identical request bodies.

The question it answers is not "is downshift faster". downshift picks a backend from a
correctness verdict, so the harness measures what that costs and what it buys: throughput,
latency, boot time and `/health` responsiveness under load, with every response checked against
eager PyTorch. A fast row with a wrong answer shows up as one.

## Where to look

| You want | Open |
|---|---|
| The newest numbers | [`results/v1.0.0/REPORT.md`](results/v1.0.0/REPORT.md) |
| Every published run, its files, and a summary of 1.0.0 and 0.5.0 | [`results/README.md`](results/README.md) |
| One page: each downshift capability and the result that shows it | [`CAPABILITIES.md`](CAPABILITIES.md) |
| Charts and tables as web pages | [`showcase/index.html`](showcase/index.html) (made by `showcase/build_showcase.py`; see [`showcase/README.md`](showcase/README.md)) |
| How to run the benchmark | "Setup" and "Run" below |

## Layout

```
bench/
  run.py, hf_models.py, report.py   entry points: python -m bench.<name>
  torch_vs_ort.py, cache_accept.py  in-process checks (no HTTP)
  _path.py, cases.py, factories.py, large/, loadgen.py, servers.py, naive_hf_server.py, inproc.py
                                    the modules that the entry points use (see "What runs")
  results/vX.Y.Z/                   published runs: raw JSON, REPORT.md, chart CSVs
  showcase/                         static web pages for an outside reader
  CAPABILITIES.md                   capability -> result map
  archive/                          old trials with their data (not part of a run)
  models/                           downloaded checkpoints (gitignored, large)
  scratch/                          default output of an ad-hoc run (gitignored, empty in git)
```

`archive/` holds the Qwen3-Embedding trial of 0.5.0 (`qwen3-embedding-trial/`), the scripts that
built the GPU tables of `.docs/REPORT_ONNX_VS_TORCH.md` (`gpu-report/`) and the 0.6.0 throughput
check (`qbench/`). Their paths are not kept up to date. They are a record, not a tool.

## Setup

From a checkout, in a venv that also has the previous release pip-installed (non-editable):

```bash
pip install "downshift-server[all]==0.5.1" aiohttp safetensors   # the "installed" target
```

The checkout is never installed: every process the harness starts finds it through `PYTHONPATH`
(`bench/_path.py`). The two Hugging Face checkpoints are downloaded to gitignored paths:

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir bench/models/all-MiniLM-L6-v2
hf download meta-llama/Prompt-Guard-86M --local-dir bench/models/Prompt-Guard-86M   # gated: accept the licence first
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
published one. Write a published run to `bench/results/vX.Y.Z/`, with the file names of the folder `v1.0.0/`. Both runners take `--targets checkout installed` (the default is both). Drop one to measure a
single version. Other useful flags: `--cases`, `--http-cases`, `--variants`, `--duration`,
`--skip-inproc`, `--skip-calibration`, `--skip-workers` (`bench.run`), `--models` (`bench.hf_models`).

Run nothing else on the machine while it measures; a concurrent test suite more than doubles
latencies. On a 16 GB machine, close memory-heavy apps too: `bert_small` has been stopped by low
memory before.

### GPU

Run from a venv with the CUDA build of torch and `onnxruntime-gpu`. The hand-rolled ONNX server
is CPU only, so leave it out:

```bash
BENCH_DEVICE=cuda python -m bench.run --variants naive_torch downshift     --skip-inproc --skip-workers --out bench/results/vX.Y.Z/results_gpu.json
python -m bench.torch_vs_ort --device cuda --out bench/results/vX.Y.Z/torch_vs_ort_cuda.json   # in process, no HTTP
```

Other checks: `python -m bench.cache_accept` (the result cache, in process, MiniLM) and
`python scripts/check_real_models.py` (verdicts of 8 real models; CI runs it).

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
