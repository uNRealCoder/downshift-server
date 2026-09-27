# Benchmark harness

The numbers in [`docs/blog/stop-hand-rolling-model-servers.md`](../docs/blog/stop-hand-rolling-model-servers.md)
come from here. The published run is [`REPORT_v0.4.0.md`](REPORT_v0.4.0.md) (raw data:
`results_v0.4.0.json`) and `results_hf_v0.4.0.json`, both on downshift 0.4.0 as tagged.

The question it answers is not "is downshift faster". downshift picks a backend from a
correctness verdict, so the harness measures what that costs and what it buys: `downshift serve`
against the FastAPI servers people write by hand, on the same model, the same ONNX graph and
byte-identical request bodies.

## Setup

From a checkout:

```bash
pip install -e ".[all]" aiohttp
```

Every entry point imports `bench._path` first, which puts `src/` ahead of site-packages so the
checkout is what gets measured, not an older pip-installed downshift.

The real-model benchmark needs two Hugging Face repos downloaded to these paths (they are
gitignored):

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir bench/all-MiniLM-L6-v2/all-MiniLM-L6-v2
huggingface-cli download meta-llama/Prompt-Guard-86M --local-dir bench/LLAMA-GUARD   # gated: accept the licence first
```

## Run

```bash
# Fixture and large models: verdicts, in-process cost, HTTP sweep, --workers sweep (~1.5 h)
python -m bench.run --out bench/results.json
python -m bench.report --results bench/results.json --out bench/REPORT.md

# Real Hugging Face models through the real CLI (~20 min)
python -m bench.hf_models --out bench/results_hf.json
```

Useful flags on `bench.run`: `--cases`, `--http-cases`, `--variants`, `--skip-inproc`,
`--skip-calibration`, `--skip-workers`, `--duration`. Run nothing else on the machine while it
measures; a concurrent test suite more than doubles latencies. Close memory-heavy apps too: on a
16 GB machine the published run was stopped once by low system memory during `bert_small`.

## What runs

| module | role |
|---|---|
| `run.py` | Orchestrator. Calibrates the load generator against `GET /health` first, so a client-side ceiling shows up as one. |
| `loadgen.py` | Closed-loop aiohttp load generator, one process per shard, JSON config on stdin. |
| `servers.py` | One server process per variant: `downshift` (the real `prepare_serving` + `build_app` path), `naive_torch`, `naive_onnx`, `naive_torch_async`. |
| `inproc.py` | No-HTTP baselines: compute, compute + JSON codec, compute + base64 codec. |
| `cases.py`, `large/` | The models. `tests/models` fixtures are sized for export hazards, so `large/` adds compute-heavy counterparts. All built from a fixed seed. |
| `factories.py` | Import specs so the real `downshift serve --workers N` CLI builds the same seeded models. |
| `hf_models.py`, `naive_hf_server.py` | Real HF checkpoints: `check` verdict, boot time, text serving via downshift (ORT and torch) vs a hand-rolled FastAPI + transformers server (sync and `async def`), with a `/health` probe running during every load window. |
| `report.py`, `summarize.py` | Turn results JSON into the report tables / a compact summary. |

## Fairness rules

- The hand-rolled baselines take a plain `dict` body with no pydantic validation and return a
  plain `dict` with no `response_model`. The HF baseline truncates over-long text instead of
  rejecting it. Each does strictly less work per request than downshift, so overhead
  attributed to downshift is real.
- Every variant gets the identical prepared module and ONNX graph, and every response is
  checked element-wise against eager PyTorch. A fast row with a large error is a wrong answer,
  not a win.
- The published v0.4.0 run was split in two: the first pass was stopped for low memory during
  `bert_small`, so `bert_small`'s HTTP rows and the `--workers` sweep were re-run on their own
  and merged (`"resumed"` in the JSON records this). Everything else is from the first pass.
