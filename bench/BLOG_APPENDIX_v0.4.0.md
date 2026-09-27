# Stop hand-rolling model servers: appendix

The full tables behind the downshift 0.4.0 launch post, *Stop hand-rolling model servers*, published on Medium. Medium doesn't render tables, so they're kept here, next to the harness that produced them.

Every number comes from the v0.4.0 runs in this directory: `results_v0.4.0.json` for the synthetic models and `results_hf*_v0.4.0.json` for Prompt Guard 86M and all-MiniLM-L6-v2.

## A. Method

- **Machine:** Windows 11, 16 logical CPUs, 16 GB RAM, CPU-only. Python 3.12.10, torch 2.14.0+cpu, onnxruntime 1.30.0, downshift 0.4.0 (tag `v0.4.0`).
- **Load:** closed loop (each client sends its next request when the last one returns), 1 s warmup, 3 s measurement window per point (5 s for the HF models), concurrency 1–64, spread over up to 6 load-generator processes.
- **Client ceiling:** about 4,000 rps at concurrency 1 and 7,800 rps at 2–64 against `GET /health`. No server result comes close.
- **Fairness:** every variant gets the identical prepared module, ONNX graph and request bytes. The baselines skip validation and response models, so they do strictly less work per request. Weights are built from a fixed seed in every process, so the correctness numbers compare the same function.
- **Correctness:** each server's first response is compared element by element with eager PyTorch on the same input.

## B. Verdicts for the benchmark models

| model | family | verdict | backend | max abs err |
|---|---|---|---|---|
| `clean_mlp` | generic-torch | CLEAN | onnxruntime | 3.7e-08 |
| `dynamic_batch_cnn` | generic-torch | CLEAN | onnxruntime | 3.0e-08 |
| `gnn_gcn` | pyg | CLEAN | onnxruntime | 2.4e-07 |
| `tiny_bert` | hf-transformers | CLEAN | onnxruntime | 4.8e-07 |
| `scatter_include_self_false` | generic-torch | DEGRADED | torch | 1.59 |
| `mlp_large` | generic-torch | CLEAN | onnxruntime | 2.1e-07 |
| `cnn_large` | generic-torch | CLEAN | onnxruntime | 5.6e-08 |
| `bert_small` | hf-transformers | CLEAN | onnxruntime | 1.9e-06 |

## C. Where the time goes (no server attached)

Compute versus the cost of speaking JSON, measured in-process. Base64 is the same round trip with binary tensor bodies.

| model | batch | ORT compute | + JSON | + base64 | JSON body | base64 body |
|---|---|---|---|---|---|---|
| `clean_mlp` | 1 | 0.020 ms | 0.050 ms | 0.047 ms | 0.3 KiB | 0.2 KiB |
| `dynamic_batch_cnn` | 32 | 0.075 ms | 18.7 ms | 1.1 ms | 498 KiB | 128 KiB |
| `mlp_large` | 32 | 2.06 ms | 30.1 ms | 3.4 ms | 330 KiB | 85 KiB |
| `cnn_large` | 32 | 8.5 ms | 80.7 ms | 12.4 ms | 1,987 KiB | 512 KiB |
| `bert_small` | 1 | 7.2 ms | 39.5 ms | 8.5 ms | 1.3 KiB | 2.8 KiB |

ONNX Runtime versus eager PyTorch, compute only: 1.3x–7.3x faster across the benchmark models (`gnn_gcn` batch 1: 0.78 ms vs 0.11 ms).

## D. Full batch-1 peak throughput

Batch 1, peak across concurrency 1–64; p50/p99 in ms at that peak; error vs eager PyTorch.

| model | server | peak req/s | at | p50 | p99 | max abs err |
|---|---|---|---|---|---|---|
| `clean_mlp` | hand-rolled FastAPI + eager torch | 1339.0 | c=4 | 2.75 | 6.16 | 0.00e+00 |
| `clean_mlp` | hand-rolled FastAPI + ONNX Runtime | 1615.3 | c=4 | 2.26 | 4.75 | 7.45e-09 |
| `clean_mlp` | downshift serve (auto) | 795.0 | c=8 | 9.82 | 17.12 | 9.22e-09 |
| `clean_mlp` | downshift serve (base64) | 879.1 | c=2 | 2.09 | 3.88 | 7.45e-09 |
| `clean_mlp` | hand-rolled FastAPI + torch, `async def` | 2602.0 | c=32 | 11.89 | 16.21 | 0.00e+00 |
| `dynamic_batch_cnn` | hand-rolled FastAPI + eager torch | 929.3 | c=4 | 4.34 | 7.60 | 0.00e+00 |
| `dynamic_batch_cnn` | hand-rolled FastAPI + ONNX Runtime | 1061.0 | c=4 | 3.61 | 8.20 | 2.98e-08 |
| `dynamic_batch_cnn` | downshift serve (auto) | 887.6 | c=8 | 9.93 | 14.82 | 3.17e-08 |
| `dynamic_batch_cnn` | downshift serve (base64) | 936.6 | c=8 | 9.35 | 14.75 | 2.98e-08 |
| `dynamic_batch_cnn` | hand-rolled FastAPI + torch, `async def` | 1039.9 | c=32 | 35.41 | 45.03 | 0.00e+00 |
| `gnn_gcn` | hand-rolled FastAPI + eager torch | 549.4 | c=1 | 1.75 | 3.05 | 0.00e+00 |
| `gnn_gcn` | hand-rolled FastAPI + ONNX Runtime | 1484.3 | c=16 | 10.55 | 15.70 | 5.96e-08 |
| `gnn_gcn` | downshift serve (auto) | 845.2 | c=2 | 2.22 | 4.04 | 6.44e-08 |
| `gnn_gcn` | downshift serve (base64) | 816.8 | c=2 | 2.29 | 4.04 | 5.96e-08 |
| `gnn_gcn` | hand-rolled FastAPI + torch, `async def` | 689.4 | c=16 | 22.75 | 27.59 | 0.00e+00 |
| `tiny_bert` | hand-rolled FastAPI + eager torch | 444.4 | c=2 | 4.20 | 14.03 | 0.00e+00 |
| `tiny_bert` | hand-rolled FastAPI + ONNX Runtime | 1502.5 | c=16 | 10.38 | 15.65 | 4.77e-07 |
| `tiny_bert` | downshift serve (auto) | 807.2 | c=8 | 9.81 | 13.89 | 5.08e-07 |
| `tiny_bert` | downshift serve (base64) | 773.8 | c=16 | 20.50 | 25.45 | 4.77e-07 |
| `tiny_bert` | hand-rolled FastAPI + torch, `async def` | 539.1 | c=1 | 1.78 | 3.17 | 0.00e+00 |
| `scatter_include_self_false` | hand-rolled FastAPI + eager torch | 1267.1 | c=2 | 1.44 | 2.92 | 0.00e+00 |
| `scatter_include_self_false` | hand-rolled FastAPI + ONNX Runtime | 1571.9 | c=4 | 2.33 | 4.57 | 7.94e-01 |
| `scatter_include_self_false` | downshift serve (auto) | 921.3 | c=2 | 2.07 | 3.56 | 2.77e-08 |
| `scatter_include_self_false` | downshift serve (base64) | 915.1 | c=4 | 4.18 | 7.24 | 0.00e+00 |
| `scatter_include_self_false` | hand-rolled FastAPI + torch, `async def` | 2041.4 | c=32 | 15.23 | 20.41 | 0.00e+00 |
| `mlp_large` | hand-rolled FastAPI + eager torch | 768.6 | c=16 | 22.40 | 35.75 | 0.00e+00 |
| `mlp_large` | hand-rolled FastAPI + ONNX Runtime | 877.6 | c=16 | 20.18 | 29.04 | 2.53e-07 |
| `mlp_large` | downshift serve (auto) | 687.9 | c=32 | 53.24 | 65.80 | 2.58e-07 |
| `mlp_large` | downshift serve (base64) | 596.8 | c=8 | 13.24 | 19.30 | 2.53e-07 |
| `mlp_large` | hand-rolled FastAPI + torch, `async def` | 586.8 | c=16 | 30.07 | 42.21 | 0.00e+00 |
| `cnn_large` | hand-rolled FastAPI + eager torch | 358.6 | c=4 | 11.95 | 18.41 | 0.00e+00 |
| `cnn_large` | hand-rolled FastAPI + ONNX Runtime | 430.5 | c=4 | 9.69 | 28.09 | 5.96e-08 |
| `cnn_large` | downshift serve (auto) | 662.4 | c=8 | 13.08 | 20.07 | 5.88e-08 |
| `cnn_large` | downshift serve (base64) | 907.8 | c=16 | 19.36 | 29.66 | 5.96e-08 |
| `cnn_large` | hand-rolled FastAPI + torch, `async def` | 308.6 | c=4 | 14.95 | 18.51 | 0.00e+00 |
| `bert_small` | hand-rolled FastAPI + eager torch | 99.0 | c=4 | 39.49 | 61.62 | 0.00e+00 |
| `bert_small` | hand-rolled FastAPI + ONNX Runtime | 151.5 | c=16 | 103.51 | 151.84 | 1.91e-06 |
| `bert_small` | downshift serve (auto) | 102.6 | c=4 | 38.28 | 59.94 | 1.87e-06 |
| `bert_small` | downshift serve (base64) | 121.5 | c=8 | 65.12 | 88.95 | 1.91e-06 |
| `bert_small` | hand-rolled FastAPI + torch, `async def` | 72.0 | c=4 | 54.97 | 68.69 | 0.00e+00 |

## E. Compatibility matrix

One fixture per export hazard, re-run for this post with torch 2.14.0, onnx 1.22.0, onnxruntime 1.30.0, torch_geometric 2.8.0 and transformers 5.17.0 (`k=8`). The verdicts match the committed [docs/compatibility.md](../docs/compatibility.md); only float-noise error magnitudes differ between runs.

| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `bf16_weights` | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic-torch | FAILED | strict=False | — | — | torch |
| `broken_factory` | Not an export hazard fixture: raises as soon as it's instantiated | — | skipped (RuntimeError) | — | — | — | — |
| `clean_mlp` | Control fixture: no export hazards | generic-torch | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic-torch | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic-torch | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic-torch | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic-torch | CLEAN | strict=False | 1.8e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic-torch | CLEAN | strict=False | 6.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 3.6e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic-torch | DEGRADED | strict=False | 1.4e+00 | — | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic-torch | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a randomly initialised two-layer BERT encoder | hf-transformers | CLEAN | strict=False | 4.8e-07 | ✓ | onnxruntime |

## F. Reproduce

```bash
git clone https://github.com/uNRealCoder/downshift-server && cd downshift-server
pip install -e ".[all]" aiohttp
python -m bench.run --out bench/results.json
python -m bench.report --results bench/results.json --out bench/REPORT.md
python scripts/gen_matrix.py
```

[`bench/README.md`](README.md) covers the real-model run (`python -m bench.hf_models`), where to download the two checkpoints, and the fairness rules. The raw results behind this post are `bench/results_v0.4.0.json` and `bench/results_hf*_v0.4.0.json`, and the full generated report is [`bench/REPORT_v0.4.0.md`](REPORT_v0.4.0.md).

## G. Data behind the charts

The numbers behind each chart in the post, in the order the charts appear.

### One client at a time

| model, request | hand-rolled FastAPI + transformers | downshift |
|---|---|---|
| Prompt Guard, 1 short text | 8.8 req/s (p50 112 ms) | **44.9** (p50 22 ms), 5.1x |
| Prompt Guard, 8 texts | 4.0 (238 ms) | **6.8** (145 ms), 1.7x |
| Prompt Guard, 1 long text | 3.0 (311 ms) | **4.4** (221 ms), 1.5x |
| MiniLM, 1 short text | 98.5 (9.9 ms) | **180.9** (5.3 ms), 1.8x |
| MiniLM, 8 texts | 41.9 (23 ms) | **50.0** (20 ms), 1.2x |
| MiniLM, 1 long text | **34.0** (29 ms) | 29.8 (34 ms), 0.88x |

### Eight clients at once

| model, request | hand-rolled `def` | hand-rolled `async def` | downshift (defaults) | downshift `--max-concurrency 4` |
|---|---|---|---|---|
| Prompt Guard, 1 short text | 15.6 req/s | 7.9 | 47.1 | **92.0** |
| Prompt Guard, 8 texts | 7.8 | 3.2 | 6.0 | **11.3** |
| Prompt Guard, 1 long text | 5.7 | 2.4 | 3.6 | **7.2** |
| MiniLM, 1 short text | 130.5 | 107.9 | 235.3 | **409.6** |
| MiniLM, 8 texts | 81.7 | 44.6 | 52.0 | **133.6** |
| MiniLM, 1 long text | 70.5 | 33.7 | 29.7 | **76.7** |

### Liveness probe latency

`GET /health` every 50 ms during the 8-client runs, worst p99 across payloads.

| model | hand-rolled `def` | hand-rolled `async def` | downshift |
|---|---|---|---|
| Prompt Guard | 16 ms | **5,462 ms** | 19 ms (22 ms at `--max-concurrency 4`) |
| MiniLM | 17 ms | **429 ms** | 20 ms (20 ms) |

*About 17 ms is this Windows machine's floor for a fresh HTTP connection; anything near it is effectively instant.*

### When the fastest server is wrong

| server | peak req/s | max abs error vs PyTorch |
|---|---|---|
| hand-rolled FastAPI + eager torch | 1,267 | 0 |
| hand-rolled FastAPI + ONNX Runtime | **1,572** | **1.20 (wrong)** |
| downshift serve (auto → torch) | 921 | 2.8e-08 |

*Scatter-mean model, batch 1.*

### Throughput on real payloads

| model, batch | hand-rolled FastAPI + ORT | downshift (JSON) | downshift (base64) |
|---|---|---|---|
| `mlp_large`, 8 | 251 req/s | 435 (1.7x) | 701 (2.8x) |
| `mlp_large`, 32 | 74 | 167 (2.3x) | 344 (4.7x) |
| `cnn_large`, 1 | 431 | 662 (1.5x) | 908 (2.1x) |
| `cnn_large`, 8 | 56 | 136 (2.4x) | 342 (6.1x) |
| `cnn_large`, 32 | 12.7 | 36.5 (2.9x) | 110 (8.7x) |
| `dynamic_batch_cnn`, 32 | 51 | 120 (2.4x) | 671 (13.1x) |

*Peak throughput across concurrency 1–64.*

### Where downshift is slower

| model (batch 1) | hand-rolled FastAPI + ORT | downshift | ratio |
|---|---|---|---|
| `clean_mlp` | 1,615 req/s | 795 | 0.49x |
| `tiny_bert` | 1,503 | 807 | 0.54x |
| `gnn_gcn` | 1,484 | 845 | 0.57x |
| `mlp_large` | 878 | 688 | 0.78x |
| `bert_small` | 152 | 103 | 0.68x |

### Scaling out

| model | `--workers 1` | `--workers 2` | `--workers 4` | boot to `/ready` (1 → 4 workers) |
|---|---|---|---|---|
| `clean_mlp` | 865 req/s | 1,772 | 3,126 (3.6x) | 8.0 s → 12.3 s |
| `mlp_large` | 715 | 1,026 | 1,958 (2.7x) | 9.6 s → 12.8 s |
| `bert_small` | 98 | 143 | 224 (2.3x) | 22.6 s → 24.1 s |

*Batch 1, concurrency 32. Outputs were identical across worker counts.*
