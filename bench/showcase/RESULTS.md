# Results

A plain-text copy of the key tables. The tables are made by `build_showcase.py` from the data files named under each table.
To read the method and the limits of a table, open the matching HTML page. Read [METHODOLOGY.md](METHODOLOGY.md) first.

Main run: win32, 16 logical CPUs, Python 3.14.7, torch 2.14.0+cpu, onnxruntime 1.30.0, torch threads 8, run 2026-11-22T18:30:39. downshift 1.0.0 from the checkout, 0.5.1 from PyPI. One machine, one run, no repeats.

## Throughput and latency, batch 1: clean_mlp

Peak requests/s over the concurrency sweep; p50 and p99 in ms.

| server | peak rps | at c= | rps c=1 | p50 ms c=1 | p99 ms c=1 | p50 ms c=8 | p99 ms c=8 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1303 | 16 | 864 | 1.07 | 2.38 | 6.36 | 10.61 |
| naive FastAPI + ONNX Runtime | 1706 | 32 | 1148 | 0.83 | 1.63 | 4.84 | 7.08 |
| naive torch, async def | 2347 | 16 | 1671 | 0.57 | 1.10 | 3.37 | 5.34 |
| downshift 0.5.1 | 1118 | 32 | 845 | 1.09 | 2.26 | 7.62 | 9.79 |
| downshift 0.5.1 safetensors | 1055 | 32 | 876 | 1.07 | 1.86 | 7.88 | 10.24 |
| downshift 1.0.0 | 1164 | 64 | 883 | 1.08 | 1.96 | 7.26 | 9.40 |
| downshift 1.0.0 safetensors | 1079 | 16 | 867 | 1.11 | 1.92 | 7.94 | 10.60 |
| downshift 1.0.0 result cache on | 1407 | 4 | 913 | 1.00 | 2.31 | 6.05 | 8.52 |
| downshift 1.0.0 --metrics | 988 | 32 | 813 | 1.10 | 2.66 | 8.52 | 15.12 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Throughput and latency, batch 1: mlp_large

Peak requests/s over the concurrency sweep; p50 and p99 in ms.

| server | peak rps | at c= | rps c=1 | p50 ms c=1 | p99 ms c=1 | p50 ms c=8 | p99 ms c=8 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 932 | 32 | 316 | 3.08 | 5.42 | 9.84 | 23.17 |
| naive FastAPI + ONNX Runtime | 980 | 32 | 500 | 1.94 | 3.26 | 7.87 | 22.14 |
| naive torch, async def | 656 | 64 | 391 | 2.43 | 4.66 | 16.67 | 26.23 |
| downshift 0.5.1 | 1074 | 64 | 499 | 1.92 | 3.11 | 8.21 | 13.25 |
| downshift 0.5.1 safetensors | 819 | 64 | 550 | 1.77 | 2.71 | 10.40 | 12.88 |
| downshift 1.0.0 | 1013 | 64 | 490 | 1.98 | 3.25 | 8.48 | 14.75 |
| downshift 1.0.0 safetensors | 806 | 64 | 550 | 1.75 | 2.82 | 10.45 | 13.18 |
| downshift 1.0.0 result cache on | 1629 | 64 | 1062 | 0.90 | 1.48 | 5.67 | 8.15 |
| downshift 1.0.0 --metrics | 988 | 64 | 485 | 2.04 | 3.06 | 8.58 | 14.04 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## 1.0.0 against 0.5.1, batch 1

Peak requests/s and requests/s at c=1.

| model | peak 0.5.1 | peak 1.0.0 | change | c=1 0.5.1 | c=1 1.0.0 | change |
|---|---|---|---|---|---|---|
| clean_mlp | 1118 | 1164 | +4.1% | 845 | 883 | +4.4% |
| tiny_bert | 1082 | 1112 | +2.8% | 583 | 632 | +8.4% |
| mlp_large | 1074 | 1013 | -5.6% | 499 | 490 | -1.7% |
| scatter_include_self_false | 1047 | 959 | -8.4% | 853 | 646 | -24.2% |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Cost of --metrics at c=1

Extra time per request is 1/rps(metrics) - 1/rps(default).

| model | rps c=1 default | rps c=1 --metrics | extra time per request (us) | change in rps |
|---|---|---|---|---|
| clean_mlp | 883 | 813 | +97 | -7.9% |
| tiny_bert | 632 | 630 | +6 | -0.4% |
| mlp_large | 490 | 485 | +21 | -1.0% |
| scatter_include_self_false | 646 | 943 | -487 | +46.0% |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Load generator ceiling

Requests/s of GET /health on a do-nothing server.

| concurrency | ceiling rps |
|---|---|
| c=1 | 4453 |
| c=2 | 8168 |
| c=4 | 8304 |
| c=8 | 8769 |
| c=16 | 9437 |
| c=32 | 9636 |
| c=64 | 9902 |

Source: [`bench/results/v1.0.0/results.json`](../results/v1.0.0/results.json)

## Time of one request by stage: clean_mlp

Median ms of 300 sequential requests, batch 1. A dash means the server reports no such stage.

| server | total ms | parse | prep wait | prep | infer wait | infer | encode | HTTP + hand-offs |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.90 | - | - | - | - | - | - | 0.90 |
| naive FastAPI + ONNX Runtime | 0.74 | - | - | - | - | - | - | 0.74 |
| naive torch, async def | 0.48 | - | - | - | - | - | - | 0.48 |
| downshift 0.5.1 | 1.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.22 | 0.04 | 0.66 |
| downshift 1.0.0 | 1.00 | 0.02 | 0.04 | 0.03 | 0.03 | 0.21 | 0.04 | 0.65 |
| downshift 1.0.0 result cache on | 0.83 | 0.02 | 0.03 | 0.06 | 0.00 | 0.00 | 0.03 | 0.69 |

Source: [`bench/results/v1.0.0/chart_stages.csv`](../results/v1.0.0/chart_stages.csv)

## Time of one request by stage: all-MiniLM-L6-v2

Median ms of 300 sequential requests, batch 1. A dash means the server reports no such stage.

| server | total ms | parse | prep wait | prep | infer wait | infer | encode | HTTP + hand-offs |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 9.10 | - | - | - | - | - | - | 9.10 |
| naive transformers, async def | 8.68 | - | - | - | - | - | - | 8.68 |
| downshift 0.5.1 | 4.64 | 0.04 | 0.06 | 0.33 | 0.04 | 3.02 | 0.10 | 1.05 |
| downshift 1.0.0 | 4.67 | 0.04 | 0.06 | 0.34 | 0.04 | 3.01 | 0.10 | 1.09 |
| downshift 1.0.0 result cache on | 0.70 | 0.02 | 0.02 | 0.01 | 0.00 | 0.00 | 0.05 | 0.59 |
| downshift 0.5.1 --backend torch | 9.33 | 0.05 | 0.06 | 0.37 | 0.03 | 7.48 | 0.11 | 1.23 |
| downshift 1.0.0 --backend torch | 9.46 | 0.05 | 0.06 | 0.38 | 0.03 | 7.55 | 0.11 | 1.27 |

Source: [`bench/results/v1.0.0/chart_stages.csv`](../results/v1.0.0/chart_stages.csv)

## In-process compute: torch against ONNX Runtime

p50 of 200 calls, no HTTP. The scatter rows time a graph that gives wrong numbers.

| model | batch | verdict | torch p50 ms | ORT p50 ms | torch / ORT |
|---|---|---|---|---|---|
| clean_mlp | 1 | CLEAN | 0.051 | 0.021 | 2.5x |
| clean_mlp | 8 | CLEAN | 0.060 | 0.021 | 2.8x |
| clean_mlp | 32 | CLEAN | 0.062 | 0.021 | 2.9x |
| dynamic_batch_cnn | 1 | CLEAN | 0.11 | 0.031 | 3.4x |
| dynamic_batch_cnn | 8 | CLEAN | 0.22 | 0.048 | 4.7x |
| dynamic_batch_cnn | 32 | CLEAN | 0.33 | 0.074 | 4.4x |
| gnn_gcn | 1 | CLEAN | 0.78 | 0.12 | 6.5x |
| gnn_gcn | 8 | CLEAN | 0.93 | 0.18 | 5.3x |
| gnn_gcn | 32 | CLEAN | 0.93 | 0.36 | 2.6x |
| tiny_bert | 1 | CLEAN | 1.33 | 0.26 | 5.2x |
| tiny_bert | 8 | CLEAN | 1.52 | 0.48 | 3.2x |
| tiny_bert | 32 | CLEAN | 2.15 | 1.24 | 1.7x |
| scatter_include_self_false | 1 | DEGRADED | 0.059 | 0.034 | 1.8x |
| scatter_include_self_false | 8 | DEGRADED | 0.11 | 0.042 | 2.6x |
| scatter_include_self_false | 32 | DEGRADED | 0.15 | 0.068 | 2.2x |
| mlp_large | 1 | CLEAN | 1.14 | 0.26 | 4.3x |
| mlp_large | 8 | CLEAN | 2.11 | 0.49 | 4.3x |
| mlp_large | 32 | CLEAN | 3.22 | 2.10 | 1.5x |
| cnn_large | 1 | CLEAN | 1.05 | 0.32 | 3.3x |
| cnn_large | 8 | CLEAN | 4.45 | 2.15 | 2.1x |
| cnn_large | 32 | CLEAN | 16.22 | 8.44 | 1.9x |
| bert_small | 1 | CLEAN | 7.85 | 7.02 | 1.1x |
| bert_small | 8 | CLEAN | 35.67 | 47.98 | 0.7x |
| bert_small | 32 | CLEAN | 142.26 | 193.83 | 0.7x |

Source: [`bench/results/v1.0.0/results.json`](../results/v1.0.0/results.json)

## In-process: cost of the wire format, ORT, batch 32

p50 of 200 calls. JSON and base64 columns include parse, infer and write.

| model (batch 32) | ORT compute ms | + JSON ms | + base64 ms | JSON / compute | JSON bytes | base64 bytes |
|---|---|---|---|---|---|---|
| clean_mlp | 0.02 | 0.55 | 0.07 | 25.6x | 10660 | 2831 |
| dynamic_batch_cnn | 0.07 | 19.60 | 1.04 | 263.8x | 510407 | 131178 |
| gnn_gcn | 0.36 | 2.51 | 0.60 | 7.0x | 34956 | 15184 |
| tiny_bert | 1.24 | 5.76 | 1.49 | 4.7x | 1937 | 5636 |
| scatter_include_self_false | 0.07 | 1.22 | 0.14 | 18.0x | 32694 | 10402 |
| mlp_large | 2.10 | 29.68 | 3.32 | 14.1x | 338282 | 87484 |
| cnn_large | 8.44 | 80.44 | 11.77 | 9.5x | 2034441 | 524394 |
| bert_small | 193.83 | 1021.77 | 236.70 | 5.3x | 39626 | 87560 |

Source: [`bench/results/v1.0.0/results.json`](../results/v1.0.0/results.json)

## MiniLM over HTTP: torch against ONNX Runtime

Requests/s at c=8 and p50 ms at c=1, 5 s windows. Cache off.

| server | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | short_b1 p50 c=1 ms | short_b8 p50 c=1 ms | long_b1 p50 c=1 ms |
|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 142 | 83 | 74 | 9.4 | 21.8 | 28.4 |
| naive transformers, async def | 115 | 46 | 36 | 8.9 | 21.3 | 27.1 |
| downshift 0.5.1 | 532 | 139 | 78 | 4.8 | 19.4 | 33.3 |
| downshift 0.5.1 --backend torch | 145 | 86 | 66 | 9.6 | 23.6 | 29.4 |
| downshift 1.0.0 | 528 | 137 | 78 | 4.8 | 19.4 | 33.2 |
| downshift 1.0.0 --backend torch | 148 | 84 | 70 | 9.6 | 23.7 | 29.7 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Hand-rolled FastAPI: torch against ORT, peak requests/s

Batch 1, JSON. The ORT graph of scatter_include_self_false gives wrong numbers.

| model | FastAPI + torch peak rps | FastAPI + ORT peak rps | ORT / torch | ORT graph |
|---|---|---|---|---|
| clean_mlp | 1303 | 1706 | 1.31x | ok |
| tiny_bert | 383 | 1458 | 3.81x | ok |
| mlp_large | 932 | 980 | 1.05x | ok |
| scatter_include_self_false | 1421 | 1692 | 1.19x | wrong numbers |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Wire encodings, batch 32, c=8

JSON against safetensors.

| model | server | request bytes | requests/s | p50 ms | rows/s |
|---|---|---|---|---|---|
| clean_mlp | downshift 0.5.1 | 10659 | 1301 | 6.55 | 41629 |
| clean_mlp | downshift 0.5.1 safetensors | 2120 | 997 | 7.93 | 31899 |
| clean_mlp | downshift 1.0.0 | 10659 | 1302 | 6.63 | 41651 |
| clean_mlp | downshift 1.0.0 safetensors | 2120 | 1062 | 7.46 | 33988 |
| tiny_bert | downshift 0.5.1 | 1941 | 694 | 11.46 | 22219 |
| tiny_bert | downshift 0.5.1 safetensors | 4248 | 1169 | 7.31 | 37419 |
| tiny_bert | downshift 1.0.0 | 1941 | 760 | 10.55 | 24320 |
| tiny_bert | downshift 1.0.0 safetensors | 4248 | 1154 | 7.21 | 36920 |
| mlp_large | downshift 0.5.1 | 338169 | 241 | 35.17 | 7701 |
| mlp_large | downshift 0.5.1 safetensors | 65608 | 839 | 9.64 | 26859 |
| mlp_large | downshift 1.0.0 | 338169 | 223 | 37.31 | 7142 |
| mlp_large | downshift 1.0.0 safetensors | 65608 | 825 | 9.71 | 26416 |
| scatter_include_self_false | downshift 0.5.1 | 32706 | 1070 | 7.74 | 34255 |
| scatter_include_self_false | downshift 0.5.1 safetensors | 7824 | 1380 | 6.12 | 44160 |
| scatter_include_self_false | downshift 1.0.0 | 32706 | 926 | 8.92 | 29631 |
| scatter_include_self_false | downshift 1.0.0 safetensors | 7824 | 1336 | 5.84 | 42740 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Batched requests at c=8, rows/s

Requests/s times batch size. The cache-on row is a best case with one repeated body.

| server | clean_mlp b8 | clean_mlp b32 | tiny_bert b8 | tiny_bert b32 | mlp_large b8 | mlp_large b32 | scatter b8 | scatter b32 |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 7751 | 37182 | 2403 | 7456 | 1958 | 2497 | 9431 | 20470 |
| naive FastAPI + ONNX Runtime | 11577 | 40302 | 8677 | 17882 | 2281 | 2628 | 11144 | 25354 |
| downshift 0.5.1 | 7894 | 41629 | 6927 | 22219 | 4875 | 7701 | 10991 | 34255 |
| downshift 0.5.1 safetensors | 8016 | 31899 | 7872 | 37419 | 8532 | 26859 | 7492 | 44160 |
| downshift 1.0.0 | 8586 | 41651 | 7603 | 24320 | 4620 | 7142 | 10118 | 29631 |
| downshift 1.0.0 safetensors | 8420 | 33988 | 7681 | 36920 | 8561 | 26416 | 6975 | 42740 |
| downshift 1.0.0 result cache on | 9409 | 36611 | 7013 | 21983 | 6137 | 7585 | 10593 | 32004 |
| downshift 1.0.0 --metrics | 7917 | 38453 | 7180 | 20543 | 4565 | 7256 | 9968 | 30706 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Result cache on the fixtures, batch 1

1.0.0, cache off against cache on. Every request repeats one body, so this is a best case.

| model | peak rps cache off | peak rps cache on | change | p50 ms c=1 off | p50 ms c=1 on |
|---|---|---|---|---|---|
| clean_mlp | 1164 | 1407 | +20.9% | 1.08 | 1.00 |
| tiny_bert | 1112 | 1056 | -5.0% | 1.52 | 1.40 |
| mlp_large | 1013 | 1629 | +60.7% | 1.98 | 0.90 |
| scatter_include_self_false | 959 | 1045 | +9.0% | 1.51 | 1.15 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Result cache on MiniLM

1.0.0, 5 s windows. Every request repeats one body.

| payload | rps c=1 off | rps c=1 on | rps c=8 off | rps c=8 on | p50 ms c=1 off | p50 ms c=1 on | worst /health p99 ms off | worst /health p99 ms on |
|---|---|---|---|---|---|---|---|---|
| short_b1 | 204 | 1233 | 528 | 1504 | 4.8 | 0.8 | 22 | 18 |
| short_b8 | 51 | 1039 | 137 | 1239 | 19.4 | 0.9 | 18 | 19 |
| long_b1 | 30 | 1220 | 78 | 1557 | 33.2 | 0.8 | 18 | 19 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Result cache acceptance check (0.6.0)

all-MiniLM-L6-v2, 60 repeats, run 2026-10-14 23:46:17. In-process test client, server-side time from Server-Timing.

| check | measured | rule | result |
|---|---|---|---|
| a. query seen before: server time | hit median 0.255 ms (max 0.42 ms); no cache 4.62 ms | hit median under 1 ms | pass |
| b. batch of 32 with 30 known rows | rows in the model call: [2] | model runs on 2 rows | pass |
| c. unique queries: server time | cache on 4.975 ms; cache off 4.8 ms (+3.6%) | on <= 1.05 x off + 0.2 ms | pass |

Source: [`bench/results/v0.6.0/cache_acceptance.json`](../results/v0.6.0/cache_acceptance.json)

## Verdicts of the load-run models

From the gate at the start of each downshift server.

| model | family | verdict | backend | max abs error at check | reason |
|---|---|---|---|---|---|
| clean_mlp | generic | CLEAN | onnxruntime | 3.7e-08 | exported via strict=False; numerics ok across 8 samples (max abs err 3.73e-08) |
| dynamic_batch_cnn | generic | CLEAN | onnxruntime | 3.0e-08 | exported via strict=False; numerics ok across 8 samples (max abs err 2.98e-08) |
| gnn_gcn | pyg | CLEAN | onnxruntime | 2.4e-07 | exported via strict=False; numerics ok across 8 samples (max abs err 2.38e-07) |
| tiny_bert | hf | CLEAN | onnxruntime | 4.8e-07 | exported via strict=False; numerics ok across 8 samples (max abs err 4.77e-07) |
| scatter_include_self_false | generic | DEGRADED | torch | 1.6e+00 | exported via strict=False but numerics diverge on 6/8 samples (max abs err 1.59e+00) |
| mlp_large | generic | CLEAN | onnxruntime | 2.1e-07 | exported via strict=False; numerics ok across 8 samples (max abs err 2.09e-07) |
| cnn_large | generic | CLEAN | onnxruntime | 5.6e-08 | exported via strict=False; numerics ok across 8 samples (max abs err 5.59e-08) |
| bert_small | hf | CLEAN | onnxruntime | 1.9e-06 | exported via strict=False; numerics ok across 8 samples (max abs err 1.91e-06) |

Source: [`bench/results/v1.0.0/results.json`](../results/v1.0.0/results.json)

## Worst max abs error against eager PyTorch

Over every row a server answered.

| server | clean_mlp | tiny_bert | mlp_large | scatter_include_self_false |
|---|---|---|---|---|
| naive FastAPI + torch | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| naive FastAPI + ONNX Runtime | 6.0e-08 | 4.8e-07 | 2.5e-07 | 1.2e+00 |
| naive torch, async def | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| downshift 0.5.1 | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 0.5.1 safetensors | 6.0e-08 | 4.8e-07 | 2.5e-07 | 0.0e+00 |
| downshift 1.0.0 | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 1.0.0 safetensors | 6.0e-08 | 4.8e-07 | 2.5e-07 | 0.0e+00 |
| downshift 1.0.0 result cache on | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 1.0.0 --metrics | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## MiniLM agreement with downshift --backend torch

short_b8 output, reference downshift_torch@1.0.0.

| server | max abs err | min cosine |
|---|---|---|
| naive_sync | 7.2e-09 | 1.0000000 |
| naive_async | 7.2e-09 | 1.0000000 |
| downshift_auto@1.0.0 | 1.3e-07 | 1.0000000 |
| downshift_torch@1.0.0 | 0.0e+00 | 1.0000000 |
| downshift_auto_mc4@1.0.0 | 1.3e-07 | 1.0000000 |
| downshift_auto_inline@1.0.0 | 1.3e-07 | 1.0000000 |
| downshift_auto_cache@1.0.0 | 1.3e-07 | 1.0000000 |
| downshift_auto@0.5.1 | 1.3e-07 | 1.0000000 |
| downshift_torch@0.5.1 | 0.0e+00 | 1.0000000 |
| downshift_auto_mc4@0.5.1 | 1.3e-07 | 1.0000000 |
| downshift_auto_inline@0.5.1 | 1.3e-07 | 1.0000000 |

Source: [`bench/results/v1.0.0/results_hf.json`](../results/v1.0.0/results_hf.json)

## Real models checked with downshift check

CPU, one subprocess for each model. Date 2026-11-07. For graph models the two axes are nodes / edges.

| model | family | verdict | max abs err | max rel err | min cosine | samples | sampled batch / seq | check s |
|---|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | hf | CLEAN | 2.4e-07 | 5.9e-03 | 1.000000 | 8 | 1-4 / 1-512 (serves 1-4096 / 1-512) | 19.2 |
| BAAI/bge-small-en-v1.5 | hf | CLEAN | 3.0e-07 | 1.0e-02 | 1.000000 | 8 | 1-4 / 1-512 (serves 1-4096 / 1-512) | 21.9 |
| intfloat/e5-small-v2 | hf | CLEAN | 2.4e-07 | 5.4e-03 | 1.000000 | 8 | 1-4 / 1-512 (serves 1-4096 / 1-512) | 21.8 |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | hf | CLEAN | 1.3e-05 | 1.3e-05 | - | 8 | 1-4 / 1-512 (serves 1-4096 / 1-512) | 20.4 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | hf | CLEAN | 6.7e-06 | 9.9e-07 | - | 8 | 1-4 / 1-512 (serves 1-4096 / 1-512) | 18.6 |
| meta-llama/Prompt-Guard-86M | hf | CLEAN | 6.0e-05 | 7.1e-05 | - | 8 | 1-4 / 1-512 (serves 1-4096 / 1-512) | 39.0 |
| torchvision/resnet18 | generic | CLEAN | 3.8e-06 | 4.9e-03 | - | 8 | 1-3 (serves 1-65536) | 10.6 |
| pyg/GCN (tests/models/gnn_gcn) | pyg | CLEAN | 1.2e-07 | 1.8e-05 | - | 8 | 1-12 / 2-20 (serves 1-65536 / 1-65536) | 17.6 |

Source: [`bench/results/v1.0.0/real_models.json`](../results/v1.0.0/real_models.json)

## Hazard fixtures

Copied from docs/compatibility.md, generated 2026-10-12 with torch 2.14.1+cpu, onnx 1.23.2, onnxruntime 1.31.0, torch_geometric 2.8.1, transformers 5.19.0.

| fixture | hazard | family | export | capture | numerics | shape-general | backend |
|---|---|---|---|---|---|---|---|
| bf16_weights | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic | FAILED | strict=False | — | — | torch |
| broken_factory | Not an export hazard fixture: raises as soon as it is instantiated | — | skipped (RuntimeError) | — | — | — | — |
| clean_mlp | Control fixture: no export hazards | generic | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| custom_autograd | custom autograd.Function with no symbolic override | generic | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| data_dependent_branch | data-dependent control flow | generic | FAILED | — | — | — | torch |
| dict_input | dataclass container input | generic | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| dropout_model | stochastic layer | generic | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| dynamic_batch_cnn | batch-dim generalization | generic | CLEAN | strict=False | 6.0e-08 | ✓ | onnxruntime |
| gnn_gat | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| gnn_gcn | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| gnn_sage | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 2.1e-07 | ✓ | onnxruntime |
| scatter_include_self_false | scatter_reduce(include_self=False) has no faithful ONNX translation | generic | DEGRADED | strict=False | 9.5e-01 | — | torch |
| tied_weights | tied embedding/output weight (GPT-2/OPT-style) | generic | CLEAN | strict=False | 3.8e-06 | ✓ | onnxruntime |
| tiny_bert | HF fixture: a BERT encoder with two layers and random initialization | hf | CLEAN | strict=False | 7.2e-07 | ✓ | onnxruntime |

Source: [`docs/compatibility.md`](../../docs/compatibility.md)

## Boot time, fixtures (s)

Time from process start to first 200 answer. One start for each cell.

| model | FastAPI + torch | FastAPI + ORT | torch, async def | downshift 0.5.1 | downshift 1.0.0 |
|---|---|---|---|---|---|
| clean_mlp | 6.8 | 9.6 | 5.1 | 11.3 | 11.6 |
| tiny_bert | 17.4 | 20.9 | 14.6 | 20.8 | 21.3 |
| mlp_large | 4.5 | 9.6 | 4.6 | 8.8 | 9.3 |
| scatter_include_self_false | 4.5 | 8.5 | 4.6 | 9.7 | 11.4 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Boot time, MiniLM (s)

Time from process start to first 200 answer.

| server (all-MiniLM-L6-v2) | ready s |
|---|---|
| naive FastAPI + transformers | 12.4 |
| naive transformers, async def | 11.8 |
| downshift 0.5.1 | 20.5 |
| downshift 0.5.1 --backend torch | 12.4 |
| downshift 1.0.0 | 21.3 |
| downshift 1.0.0 --backend torch | 12.5 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## downshift check wall time, MiniLM

From results_hf.json.

| version | wall s | exit code |
|---|---|---|
| 1.0.0 | 21.89 | 0 |
| 0.5.1 | 20.95 | 0 |

Source: [`bench/results/v1.0.0/results_hf.json`](../results/v1.0.0/results_hf.json)

## Workers sweep (0.5.1 and 1.0.0)

Requests/s at c=1, 8 and 32; p99 at c=8; seconds to ready.

| model | version | workers | rps c=1 | rps c=8 | rps c=32 | p99 ms c=8 | boot s |
|---|---|---|---|---|---|---|---|
| clean_mlp | 0.5.1 | 1 | 1032 | 1150 | 1303 | 8.8 | 8.5 |
| clean_mlp | 0.5.1 | 2 | 963 | 2251 | 2450 | 5.6 | 11.8 |
| clean_mlp | 0.5.1 | 4 | 1032 | 4581 | 4372 | 2.5 | 12.3 |
| clean_mlp | 1.0.0 | 1 | 1005 | 1135 | 1257 | 9.3 | 8.7 |
| clean_mlp | 1.0.0 | 2 | 997 | 2228 | 2373 | 5.9 | 12.1 |
| clean_mlp | 1.0.0 | 4 | 992 | 4123 | 4254 | 3.5 | 12.1 |
| mlp_large | 0.5.1 | 1 | 522 | 1013 | 1050 | 13.3 | 8.5 |
| mlp_large | 0.5.1 | 2 | 520 | 1452 | 1715 | 12.4 | 12.2 |
| mlp_large | 0.5.1 | 4 | 526 | 1858 | 2785 | 7.2 | 12.3 |
| mlp_large | 1.0.0 | 1 | 499 | 981 | 990 | 13.9 | 9.1 |
| mlp_large | 1.0.0 | 2 | 505 | 1483 | 1710 | 11.2 | 12.8 |
| mlp_large | 1.0.0 | 4 | 514 | 1736 | 2759 | 8.1 | 12.8 |
| bert_small | 0.5.1 | 1 | 91 | 254 | 246 | 40.6 | 19.3 |
| bert_small | 0.5.1 | 2 | 92 | 252 | 261 | 50.3 | 22.6 |
| bert_small | 0.5.1 | 4 | 69 | 285 | 346 | 43.8 | 23.1 |
| bert_small | 1.0.0 | 1 | 90 | 252 | 246 | 39.9 | 19.6 |
| bert_small | 1.0.0 | 2 | 94 | 251 | 267 | 53.4 | 23.5 |
| bert_small | 1.0.0 | 4 | 69 | 278 | 327 | 43.2 | 23.1 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Release to release, run A: 0.4.0 against 0.5.0

Peak requests/s, batch 1, JSON, same session.

| model | 0.4.0 peak rps | 0.5.0 peak rps | change |
|---|---|---|---|
| clean_mlp | 866 | 1030 | +19.0% |
| dynamic_batch_cnn | 862 | 1031 | +19.6% |
| gnn_gcn | 810 | 891 | +10.0% |
| tiny_bert | 820 | 906 | +10.5% |
| scatter_include_self_false | 940 | 1076 | +14.5% |
| mlp_large | 696 | 815 | +17.1% |
| cnn_large | 657 | 736 | +12.0% |
| bert_small | 102 | 106 | +3.9% |

Source: [`bench/results/v0.5.0/chart_runs.csv`](../results/v0.5.0/chart_runs.csv)

## Release to release, run B: 0.5.1 against 1.0.0

Peak requests/s, batch 1, JSON, cache off, same session.

| model | 0.5.1 peak rps | 1.0.0 peak rps | change | note |
|---|---|---|---|---|
| clean_mlp | 1118 | 1164 | +4.1% |  |
| tiny_bert | 1082 | 1112 | +2.8% |  |
| mlp_large | 1074 | 1013 | -5.6% |  |
| scatter_include_self_false | 1047 | 959 | -8.4% |  |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)
