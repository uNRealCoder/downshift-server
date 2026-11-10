# Capabilities and the results that show them

This page maps each downshift capability to the benchmark result that shows it. Each row gives the data file and one number. Every number was read from the file named in the row.

Read the limits first:

- **Hardware: not recorded in these runs.** The files hold the platform (`win32`), 16 logical CPUs and 8 torch threads. They do not hold the CPU model or the RAM.
- All load results come from one machine, one run, and no repeats. A difference of a few percent is not a finding.
- The directory `results/v1.0.0/` holds data of downshift **1.0.0rc1** (a release candidate), not of the final 1.0.0.
- All load runs are CPU only, batch 1 unless stated.

## Versions

The current floors are `torch>=2.8`, `onnx>=1.23` and `onnxruntime>=1.20` (`pyproject.toml`). No benchmark ran at a floor.

| Source | torch | onnx | onnxruntime |
|---|---|---|---|
| `results/v1.0.0/results.json` (load runs) | 2.14.0+cpu | not recorded | 1.30.0 |
| `results/v1.0.0/real_models.json` (real-model check) | 2.14.0 | 1.22.0 | 1.30.0 |
| `results/v0.5.0/REPORT.md` and `REPORT_v0.4.0.md` | 2.14.0+cpu | not recorded | 1.30.0 |
| `docs/compatibility.md` (fixture table, 2026-10-12) | 2.14.1+cpu | 1.23.2 | 1.31.0 |

The `onnx` version of the real-model check (1.22.0) is below the new floor of 1.23. The check ran before the floor was raised. Run `python scripts/check_real_models.py` again to get a result on the new floor.

## The map

| Capability | Result that shows it | File | One number |
|---|---|---|---|
| The verdict gate finds a silent export error | `scatter_include_self_false` exports with no error. The gate says DEGRADED and serves from torch. A hand-rolled ONNX Runtime server returns wrong answers. | `results/v1.0.0/results.json` (verdict), `results/v1.0.0/chart_runs.csv` (`max_abs_err`) | Gate: numerics diverge on 6 of 8 samples (max abs error 1.59). Served by downshift: error 2.8e-08. Hand-rolled ONNX Runtime server: error 1.2. |
| ONNX Runtime against the torch fallback | MiniLM served by the verdict backend (ONNX Runtime) against `--backend torch`, one sentence, 8 clients | `results/v1.0.0/chart_runs.csv` | 486 against 146 requests/s |
| In-process compute, ONNX Runtime against torch | `tiny_bert`, batch 1, mean time of the model call | `results/v1.0.0/results.json` (`inproc`) | 0.19 ms against 2.04 ms |
| Throughput and latency | `clean_mlp`, 1.0.0rc1, batch 1 | `results/v1.0.0/chart_runs.csv` | p50 0.77 ms at c=1. Peak 1428 requests/s. |
| Throughput against a hand-rolled server | `tiny_bert` peak, batch 1 | `results/v1.0.0/chart_runs.csv` | 1067 requests/s (downshift) against 350 (hand-rolled FastAPI + torch) |
| Where downshift is slower | `clean_mlp` peak, a tiny model | `results/v1.0.0/chart_runs.csv` | 1428 requests/s (downshift) against 3216 (hand-rolled `async def` torch) |
| Result cache | MiniLM, one sentence, c=1, every request is a hit | `results/v1.0.0/chart_runs.csv` | 1456 against 185 requests/s. p50 0.62 ms against 4.89 ms. |
| Result cache, in-process acceptance check | MiniLM, 60 repeats, server time of a repeated query | `results/v0.6.0/cache_acceptance.json` | Median 0.255 ms for a hit against 4.62 ms with no cache |
| Wire formats | `mlp_large`, batch 32, c=8, JSON against safetensors, 1.0.0rc1 | `results/v1.0.0/chart_runs.csv` | 6561 rows/s (338169 bytes) against 25378 rows/s (65608 bytes) |
| GPU serving check | Not measured by `bench/`. The serving check runs at boot on a GPU and compares the GPU result with the CPU result. | none in `bench/` | No number. Refer to `docs/production.md`, section GPU, for the manual test. |
| Hugging Face adapter | 8 real models through `downshift check`, all CLEAN | `results/v1.0.0/real_models.json` | 8 of 8 CLEAN. Largest error 6.0e-05 (Prompt Guard). |
| Hugging Face adapter, serving | MiniLM over HTTP, agreement with `--backend torch` | `results/v1.0.0/results_hf.json` | Max abs error 1.3e-07, min cosine 1.0000000 |
| PyG adapter | `pyg/GCN (tests/models/gnn_gcn)` in the real-model check | `results/v1.0.0/real_models.json` | CLEAN, error 1.2e-07 |
| PyG adapter, serving | `gnn_gcn` over HTTP, 0.5.0, batch 1, c=16 | `results/v0.5.0/chart_runs.csv` | 891 requests/s (0.4.0: 780). Error 6.4e-08. |
| Boot time | Seconds from process start to `/ready`, MiniLM | `results/v1.0.0/chart_runs.csv` | 19.2 s (downshift) against 11.8 s (hand-rolled FastAPI + transformers) |
| Boot time, the check alone | `downshift check` wall time, MiniLM | `results/v1.0.0/results_hf.json` | 19.11 s |
| Scaling with workers | `clean_mlp`, c=8, 0.5.0, `--workers` 1, 2, 4 | `results/v0.5.0/chart_runs.csv` | 983, 2076 and 3857 requests/s |
| `/health` under load | MiniLM, worst `/health` p99 during the load windows, result cache off | `results/v1.0.0/chart_runs.csv` | 27 ms (downshift) against 420 ms (hand-rolled `async def`) |
| Correctness under load | All fixture rows of downshift, against eager PyTorch | `results/v1.0.0/chart_runs.csv` | Worst error 5.8e-07. Failed requests in all 486 load windows of the file: 0. |

## Notes on the rows

- **Verdict gate.** The gate runs 8 samples, some with shapes that the exporter did not see. A fault that none of the samples hits is not found. The load-run fixtures have random seeded weights. `docs/compatibility.md` lists 14 hazard fixtures. Its number for `scatter_include_self_false` (9.5e-01) differs from the load-run number (1.59). The two are separate runs. The files do not record the reason.
- **Backend choice.** The in-process comparison covers the 4 load-run fixtures. The comparison over HTTP covers MiniLM only.
- **Throughput.** On the tiny `clean_mlp`, a hand-rolled server is faster. The request path of downshift does more work per request than the hand-rolled servers. Each request makes two thread hops (a prep pool, then an inference thread). The data gives no gain from 0.5.1 to 1.0.0rc1.
- **Result cache.** The harness sends one body again and again, so every request is a hit. This is a best case. The cache is off in the headline rows (`--result-cache-rows 0`). It does nothing for `tiny_bert` and `scatter_include_self_false`, where the default is off.
- **Wire formats.** Safetensors helps most on wide inputs. For `tiny_bert` the safetensors body is larger than the JSON body at batch 32 (4248 against 1941 bytes).
- **GPU.** The load runs are CPU only. The serving check on a GPU has tests in the repository and one manual test in `docs/production.md`. It has no benchmark.
- **Real-model serving.** Only MiniLM has a load run. The other 7 real models have a verdict and no load result. Prompt Guard was measured in the 0.5.0 run only.
- **Workers.** The `--workers` sweep ran for 0.4.0 and 0.5.0. It did not run for 1.0.0rc1.
- **Boot time.** downshift is ready on `/ready`. The hand-rolled servers are ready on `/health`. The two ends are not the same.

## Where to look next

- [`README.md`](README.md): how to run the benchmark, the list of runs, the chart data columns.
- [`showcase/index.html`](showcase/index.html): static pages with charts and tables for each row above. Open the file in a browser.
- [`results/v1.0.0/REPORT.md`](results/v1.0.0/REPORT.md): the full report of the 1.0.0rc1 run.
- [`showcase/METHODOLOGY.md`](showcase/METHODOLOGY.md): how the data is made, and what it cannot show.
