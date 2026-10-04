# Benchmark harness

`downshift serve` against the FastAPI servers people write by hand, and against the previous
downshift release, on the same models, the same weights and byte-identical request bodies.

The question it answers is not "is downshift faster". downshift picks a backend from a
correctness verdict, so the harness measures what that costs and what it buys: throughput,
latency, boot time and `/health` responsiveness under load, with every response checked against
eager PyTorch. A fast row with a wrong answer shows up as one.

## Published runs

| run | report | raw data |
|---|---|---|
| **0.5.0**: naive vs 0.4.0 vs 0.5.0, one session | [`REPORT_v0.5.0.md`](REPORT_v0.5.0.md) | `results_v0.5.0.json`, `results_hf_v0.5.0.json` |
| 0.4.0: naive vs 0.4.0, behind the launch post *Stop hand-rolling model servers* | [`REPORT_v0.4.0.md`](REPORT_v0.4.0.md), [`BLOG_APPENDIX_v0.4.0.md`](BLOG_APPENDIX_v0.4.0.md) | `results_v0.4.0.json`, `results_hf*_v0.4.0.json` |

The 0.4.0 files were produced by the harness as of commit `02419cf` and are kept unchanged
because the post links to them. Their numbers are not comparable row-for-row with the 0.5.0 run,
which re-measured 0.4.0 alongside 0.5.0 in the same session.

## 0.5.0 at a glance

From [`REPORT_v0.5.0.md`](REPORT_v0.5.0.md); 16-CPU Windows machine, CPU only, peak requests/s.

- **Correctness is unchanged.** Every downshift row in both versions matches eager PyTorch to
  ≤ 2.7e-6, and both HF models agree with eager torch (MiniLM cosine 1.0, Prompt Guard 8/8 labels).
  The hand-rolled ONNX server still returns wrong answers on `scatter_include_self_false`
  (max error 1.2), where both versions route to torch.
- **Small models, default settings: 0.5.0 is slower than 0.4.0.** Batch-1 peak falls 20–30% on
  every fixture (`clean_mlp` 721 vs 914, `scatter_include_self_false` 653 vs 945). 0.5.0 runs each
  request through four thread-pool hops (parse, prep, infer, encode), and on a model that computes
  in 0.02 ms those hops are the cost.
- **`--execution inline` more than recovers it on small models.** It beats 0.4.0 on four of five
  fixtures (`clean_mlp` 1179, `scatter_include_self_false` 1145, `gnn_gcn` 974,
  `dynamic_batch_cnn` 1046) and has the lowest downshift latency on all five at concurrency 1.
  `tiny_bert` is the exception at peak (779 vs 822). On MiniLM it changes nothing (272 vs 269),
  so it is only worth it for small, fast models.
- **Compute-heavy models are mixed.** `bert_small` is 25% faster on 0.5.0 (128 vs 102);
  `mlp_large` (553 vs 698) and `cnn_large` (533 vs 652) are slower with JSON bodies. With binary
  bodies at batch 32, 0.5.0 moves more rows than 0.4.0 (`mlp_large` 14.9k vs 11.2k rows/s,
  `cnn_large` 4.2k vs 3.5k).
- **safetensors performs like base64 with smaller bodies**: within a few percent on throughput,
  with request bodies ~25% smaller (`cnn_large` b32: 393 KB vs 524 KB, against 2.0 MB as JSON).
- **Real HF models: defaults improved, one tuned row regressed.** MiniLM default: 269 vs 225
  requests/s for one short sentence at c=8. Prompt Guard is the same on both (50 vs 48).
  `--max-concurrency 4` is still the biggest single lever, but on MiniLM single sentences 0.5.0
  falls behind 0.4.0 (358 vs 421, −15%, the same per-request hops). It is ahead on eight-sentence
  batches (138 vs 130). Prompt Guard 95 vs 94, against 15 for the hand-rolled server.
- **`/health` stays responsive** under load on every downshift row (worst p99 28 ms). The
  hand-rolled `async def` server reaches 5.5 s on Prompt Guard.
- **One request hung**: 0.5.0, `--workers 4`, `bert_small`, c=8. It got no response before the
  client's 120 s timeout. It happened once in ~900 windows and is listed in the report.

## Setup

From a checkout, in a venv that also has the previous release pip-installed (non-editable):

```bash
pip install "downshift-server[all]==0.4.0" aiohttp safetensors   # the "installed" target
```

The checkout is never installed: every process the harness starts finds it through `PYTHONPATH`
(`bench/_path.py`). The two Hugging Face checkpoints are downloaded to gitignored paths:

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir bench/all-MiniLM-L6-v2/all-MiniLM-L6-v2
huggingface-cli download meta-llama/Prompt-Guard-86M --local-dir bench/LLAMA-GUARD   # gated: accept the licence first
```

## Run

```bash
# Fixture and compute-heavy models: verdicts, in-process cost, HTTP sweep, --workers sweep (~2 h)
python -m bench.run --out bench/results_v0.5.0.json

# Real Hugging Face models through the real CLI (~40 min)
python -m bench.hf_models --out bench/results_hf_v0.5.0.json

python -m bench.report --results bench/results_v0.5.0.json \
    --hf-results bench/results_hf_v0.5.0.json --out bench/REPORT_v0.5.0.md
```

Both runners take `--targets checkout installed` (the default is both). Drop one to measure a
single version. Other useful flags: `--cases`, `--http-cases`, `--variants`, `--duration`,
`--skip-inproc`, `--skip-calibration`, `--skip-workers` (`bench.run`), `--models` (`bench.hf_models`).

Run nothing else on the machine while it measures; a concurrent test suite more than doubles
latencies. On a 16 GB machine, close memory-heavy apps too: `bert_small` has been stopped by low
memory before.

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
