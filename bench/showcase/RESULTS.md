# Results

A plain-text copy of the key tables. The tables are made by `build_showcase.py` from the data files named under each table.
To read the method and the limits of a table, open the matching HTML page. Read [METHODOLOGY.md](METHODOLOGY.md) first.

Main run: win32, 16 logical CPUs, Python 3.12.10, torch 2.14.0+cpu, onnxruntime 1.30.0, torch threads 8, run 2026-11-07T22:59:00. downshift 1.0.0rc1 from the checkout, 0.5.1 from PyPI. One machine, one run, no repeats.

## Throughput and latency, batch 1: clean_mlp

Peak requests/s over the concurrency sweep; p50 and p99 in ms.

| server | peak rps | at c= | rps c=1 | p50 ms c=1 | p99 ms c=1 | p50 ms c=8 | p99 ms c=8 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1698 | 32 | 1410 | 0.62 | 1.46 | 4.72 | 10.96 |
| naive FastAPI + ONNX Runtime | 2069 | 16 | 1393 | 0.61 | 1.57 | 3.75 | 19.27 |
| naive torch, async def | 3216 | 8 | 2142 | 0.42 | 0.91 | 2.35 | 3.85 |
| downshift 0.5.1 | 1421 | 32 | 1039 | 0.80 | 2.02 | 5.94 | 8.71 |
| downshift 0.5.1 safetensors | 1416 | 32 | 1015 | 0.83 | 2.17 | 6.04 | 9.67 |
| downshift 1.0.0rc1 | 1428 | 16 | 1114 | 0.77 | 1.79 | 5.80 | 7.76 |
| downshift 1.0.0rc1 safetensors | 1317 | 4 | 1018 | 0.81 | 2.06 | 6.12 | 10.60 |
| downshift 1.0.0rc1 result cache on | 1824 | 4 | 1224 | 0.69 | 1.69 | 4.56 | 7.32 |
| downshift 1.0.0rc1 --metrics | 1325 | 16 | 1054 | 0.81 | 1.81 | 6.15 | 8.60 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Throughput and latency, batch 1: mlp_large

Peak requests/s over the concurrency sweep; p50 and p99 in ms.

| server | peak rps | at c= | rps c=1 | p50 ms c=1 | p99 ms c=1 | p50 ms c=8 | p99 ms c=8 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 984 | 32 | 389 | 2.41 | 4.46 | 9.29 | 21.30 |
| naive FastAPI + ONNX Runtime | 808 | 32 | 383 | 1.80 | 7.58 | 10.60 | 31.62 |
| naive torch, async def | 664 | 16 | 478 | 1.89 | 3.24 | 14.77 | 20.49 |
| downshift 0.5.1 | 965 | 16 | 420 | 1.92 | 7.66 | 9.01 | 17.73 |
| downshift 0.5.1 safetensors | 714 | 16 | 435 | 1.92 | 7.91 | 11.17 | 18.12 |
| downshift 1.0.0rc1 | 765 | 64 | 302 | 2.17 | 9.95 | 10.26 | 24.45 |
| downshift 1.0.0rc1 safetensors | 753 | 32 | 438 | 1.93 | 7.65 | 11.03 | 17.09 |
| downshift 1.0.0rc1 result cache on | 1742 | 16 | 1078 | 0.80 | 1.80 | 5.04 | 8.20 |
| downshift 1.0.0rc1 --metrics | 919 | 16 | 412 | 1.95 | 7.89 | 9.34 | 17.88 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## 1.0.0rc1 against 0.5.1, batch 1

Peak requests/s and requests/s at c=1. The mlp_large row of the main pass is marked in the note column.

| model | peak 0.5.1 | peak 1.0.0rc1 | change | c=1 0.5.1 | c=1 1.0.0rc1 | change | note |
|---|---|---|---|---|---|---|---|
| clean_mlp | 1421 | 1428 | +0.5% | 1039 | 1114 | +7.3% |  |
| tiny_bert | 1183 | 1067 | -9.8% | 591 | 633 | +7.2% |  |
| mlp_large | 965 | 765 | -20.7% | 420 | 302 | -28.2% | re-run: 942 vs 989 (-4.7%) |
| scatter_include_self_false | 1443 | 1387 | -3.8% | 1222 | 1186 | -2.9% |  |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv), [`bench/results/v1.0.0/rerun_mlp_large.json`](../results/v1.0.0/rerun_mlp_large.json)

## Cost of --metrics at c=1

Extra time per request is 1/rps(metrics) - 1/rps(default).

| model | rps c=1 default | rps c=1 --metrics | extra time per request (us) | change in rps |
|---|---|---|---|---|
| clean_mlp | 1114 | 1054 | +51 | -5.4% |
| tiny_bert | 633 | 610 | +61 | -3.7% |
| mlp_large | 302 | 412 | -891 | +36.7% |
| scatter_include_self_false | 1186 | 1167 | +14 | -1.6% |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Load generator ceiling

Requests/s of GET /health on a do-nothing server.

| concurrency | ceiling rps |
|---|---|
| c=1 | 5194 |
| c=2 | 9606 |
| c=4 | 10446 |
| c=8 | 11514 |
| c=16 | 12102 |
| c=32 | 11925 |
| c=64 | 10707 |

Source: [`bench/results/v1.0.0/results.json`](../results/v1.0.0/results.json)

## Time of one request by stage: clean_mlp

Median ms of 300 sequential requests, batch 1. A dash means the server reports no such stage.

| server | total ms | parse | prep wait | prep | infer wait | infer | encode | HTTP + hand-offs |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.61 | - | - | - | - | - | - | 0.61 |
| naive FastAPI + ONNX Runtime | 0.58 | - | - | - | - | - | - | 0.58 |
| naive torch, async def | 0.36 | - | - | - | - | - | - | 0.36 |
| downshift 0.5.1 | 0.71 | 0.01 | 0.03 | 0.02 | 0.03 | 0.14 | 0.03 | 0.43 |
| downshift 1.0.0rc1 | 0.70 | 0.01 | 0.03 | 0.02 | 0.03 | 0.13 | 0.02 | 0.46 |
| downshift 1.0.0rc1 result cache on | 0.66 | 0.02 | 0.03 | 0.04 | 0.00 | 0.00 | 0.02 | 0.55 |

Source: [`bench/results/v1.0.0/chart_stages.csv`](../results/v1.0.0/chart_stages.csv)

## Time of one request by stage: all-MiniLM-L6-v2

Median ms of 300 sequential requests, batch 1. A dash means the server reports no such stage.

| server | total ms | parse | prep wait | prep | infer wait | infer | encode | HTTP + hand-offs |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 8.77 | - | - | - | - | - | - | 8.77 |
| naive transformers, async def | 8.54 | - | - | - | - | - | - | 8.54 |
| downshift 0.5.1 | 4.39 | 0.04 | 0.05 | 0.28 | 0.05 | 2.82 | 0.09 | 0.93 |
| downshift 1.0.0rc1 | 4.81 | 0.04 | 0.06 | 0.32 | 0.05 | 3.00 | 0.09 | 1.05 |
| downshift 1.0.0rc1 result cache on | 0.58 | 0.01 | 0.03 | 0.01 | 0.00 | 0.00 | 0.03 | 0.49 |
| downshift 0.5.1 --backend torch | 9.08 | 0.04 | 0.06 | 0.30 | 0.04 | 7.34 | 0.09 | 1.09 |
| downshift 1.0.0rc1 --backend torch | 10.55 | 0.04 | 0.06 | 0.34 | 0.04 | 8.56 | 0.09 | 1.20 |

Source: [`bench/results/v1.0.0/chart_stages.csv`](../results/v1.0.0/chart_stages.csv)

## In-process compute: torch against ONNX Runtime

p50 of 200 calls, no HTTP. The scatter rows time a graph that gives wrong numbers.

| model | batch | verdict | torch p50 ms | ORT p50 ms | torch / ORT |
|---|---|---|---|---|---|
| clean_mlp | 1 | CLEAN | 0.030 | 0.013 | 2.3x |
| clean_mlp | 8 | CLEAN | 0.025 | 0.011 | 2.4x |
| clean_mlp | 32 | CLEAN | 0.025 | 0.013 | 1.9x |
| tiny_bert | 1 | CLEAN | 1.19 | 0.16 | 7.4x |
| tiny_bert | 8 | CLEAN | 0.99 | 0.31 | 3.2x |
| tiny_bert | 32 | CLEAN | 1.12 | 0.78 | 1.4x |
| mlp_large | 1 | CLEAN | 1.00 | 0.35 | 2.8x |
| mlp_large | 8 | CLEAN | 2.24 | 0.40 | 5.6x |
| mlp_large | 32 | CLEAN | 2.89 | 1.49 | 1.9x |
| scatter_include_self_false | 1 | DEGRADED | 0.037 | 0.020 | 1.8x |
| scatter_include_self_false | 8 | DEGRADED | 0.058 | 0.021 | 2.8x |
| scatter_include_self_false | 32 | DEGRADED | 0.073 | 0.038 | 1.9x |

Source: [`bench/results/v1.0.0/results.json`](../results/v1.0.0/results.json)

## In-process: cost of the wire format, ORT, batch 32

p50 of 200 calls. JSON and base64 columns include parse, infer and write.

| model (batch 32) | ORT compute ms | + JSON ms | + base64 ms | JSON / compute | JSON bytes | base64 bytes |
|---|---|---|---|---|---|---|
| clean_mlp | 0.01 | 0.37 | 0.05 | 29.4x | 10660 | 2831 |
| tiny_bert | 0.78 | 3.78 | 0.95 | 4.9x | 1937 | 5636 |
| mlp_large | 1.49 | 21.91 | 2.49 | 14.7x | 338282 | 87484 |
| scatter_include_self_false | 0.04 | 0.86 | 0.09 | 22.3x | 32694 | 10402 |

Source: [`bench/results/v1.0.0/results.json`](../results/v1.0.0/results.json)

## MiniLM over HTTP: torch against ONNX Runtime

Requests/s at c=8 and p50 ms at c=1, 5 s windows. Cache off.

| server | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | short_b1 p50 c=1 ms | short_b8 p50 c=1 ms | long_b1 p50 c=1 ms |
|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 141 | 84 | 71 | 9.2 | 21.3 | 27.9 |
| naive transformers, async def | 105 | 47 | 35 | 8.4 | 21.1 | 26.0 |
| downshift 0.5.1 | 524 | 165 | 94 | 4.6 | 15.7 | 27.9 |
| downshift 0.5.1 --backend torch | 156 | 90 | 69 | 9.8 | 21.0 | 25.7 |
| downshift 1.0.0rc1 | 486 | 158 | 89 | 4.9 | 17.4 | 28.4 |
| downshift 1.0.0rc1 --backend torch | 146 | 82 | 68 | 10.3 | 22.7 | 27.9 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Hand-rolled FastAPI: torch against ORT, peak requests/s

Batch 1, JSON. The ORT graph of scatter_include_self_false gives wrong numbers.

| model | FastAPI + torch peak rps | FastAPI + ORT peak rps | ORT / torch | ORT graph |
|---|---|---|---|---|
| clean_mlp | 1698 | 2069 | 1.22x | ok |
| tiny_bert | 350 | 1329 | 3.79x | ok |
| mlp_large | 984 | 808 | 0.82x | ok |
| scatter_include_self_false | 1663 | 2043 | 1.23x | wrong numbers |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Wire encodings, batch 32, c=8

JSON against safetensors. The 1.0.0rc1 mlp_large JSON row comes from the slow main pass.

| model | server | request bytes | requests/s | p50 ms | rows/s |
|---|---|---|---|---|---|
| clean_mlp | downshift 0.5.1 | 10659 | 1501 | 5.75 | 48023 |
| clean_mlp | downshift 0.5.1 safetensors | 2120 | 1253 | 6.02 | 40092 |
| clean_mlp | downshift 1.0.0rc1 | 10659 | 1348 | 5.78 | 43150 |
| clean_mlp | downshift 1.0.0rc1 safetensors | 2120 | 1174 | 6.34 | 37558 |
| tiny_bert | downshift 0.5.1 | 1941 | 769 | 10.39 | 24605 |
| tiny_bert | downshift 0.5.1 safetensors | 4248 | 964 | 7.59 | 30833 |
| tiny_bert | downshift 1.0.0rc1 | 1941 | 773 | 10.09 | 24724 |
| tiny_bert | downshift 1.0.0rc1 safetensors | 4248 | 994 | 7.16 | 31813 |
| mlp_large | downshift 0.5.1 | 338169 | 251 | 32.70 | 8022 |
| mlp_large | downshift 0.5.1 safetensors | 65608 | 788 | 10.04 | 25221 |
| mlp_large | downshift 1.0.0rc1 | 338169 | 205 | 39.93 | 6561 |
| mlp_large | downshift 1.0.0rc1 safetensors | 65608 | 793 | 10.15 | 25378 |
| scatter_include_self_false | downshift 0.5.1 | 32706 | 1014 | 8.29 | 32435 |
| scatter_include_self_false | downshift 0.5.1 safetensors | 7824 | 1323 | 6.37 | 42325 |
| scatter_include_self_false | downshift 1.0.0rc1 | 32706 | 912 | 9.08 | 29181 |
| scatter_include_self_false | downshift 1.0.0rc1 safetensors | 7824 | 1268 | 6.64 | 40579 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Batched requests at c=8, rows/s

Requests/s times batch size. The cache-on row is a best case with one repeated body.

| server | clean_mlp b8 | clean_mlp b32 | tiny_bert b8 | tiny_bert b32 | mlp_large b8 | mlp_large b32 | scatter b8 | scatter b32 |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 7972 | 30507 | 2384 | 7314 | 1673 | 2386 | 8970 | 17310 |
| naive FastAPI + ONNX Runtime | 14257 | 46352 | 7986 | 16548 | 2076 | 2916 | 14572 | 29924 |
| downshift 0.5.1 | 10559 | 48023 | 7721 | 24605 | 4724 | 8022 | 10158 | 32435 |
| downshift 0.5.1 safetensors | 10291 | 40092 | 7501 | 30833 | 7339 | 25221 | 7856 | 42325 |
| downshift 1.0.0rc1 | 9788 | 43150 | 7783 | 24724 | 3714 | 6561 | 9700 | 29181 |
| downshift 1.0.0rc1 safetensors | 10076 | 37558 | 7127 | 31813 | 7277 | 25378 | 7597 | 40579 |
| downshift 1.0.0rc1 result cache on | 11922 | 43426 | 6645 | 21697 | 6717 | 9239 | 9199 | 29699 |
| downshift 1.0.0rc1 --metrics | 10093 | 46029 | 7639 | 23325 | 4296 | 7873 | 9221 | 27496 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Result cache on the fixtures, batch 1

1.0.0rc1, cache off against cache on. Every request repeats one body, so this is a best case. The mlp_large off row is from the slow main pass.

| model | peak rps cache off | peak rps cache on | change | p50 ms c=1 off | p50 ms c=1 on |
|---|---|---|---|---|---|
| clean_mlp | 1428 | 1824 | +27.8% | 0.77 | 0.69 |
| tiny_bert | 1067 | 914 | -14.3% | 1.16 | 1.14 |
| mlp_large | 765 | 1742 | +127.7% | 2.17 | 0.80 |
| scatter_include_self_false | 1387 | 1364 | -1.7% | 0.76 | 0.89 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Result cache on MiniLM

1.0.0rc1, 5 s windows. Every request repeats one body.

| payload | rps c=1 off | rps c=1 on | rps c=8 off | rps c=8 on | p50 ms c=1 off | p50 ms c=1 on | worst /health p99 ms off | worst /health p99 ms on |
|---|---|---|---|---|---|---|---|---|
| short_b1 | 185 | 1456 | 486 | 1911 | 4.9 | 0.6 | 27 | 194 |
| short_b8 | 54 | 1222 | 158 | 1627 | 17.4 | 0.7 | 22 | 18 |
| long_b1 | 34 | 1534 | 89 | 1851 | 28.4 | 0.6 | 22 | 159 |

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
| tiny_bert | hf | CLEAN | onnxruntime | 4.8e-07 | exported via strict=False; numerics ok across 8 samples (max abs err 4.77e-07) |
| mlp_large | generic | CLEAN | onnxruntime | 2.1e-07 | exported via strict=False; numerics ok across 8 samples (max abs err 2.09e-07) |
| scatter_include_self_false | generic | DEGRADED | torch | 1.6e+00 | exported via strict=False but numerics diverge on 6/8 samples (max abs err 1.59e+00) |

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
| downshift 1.0.0rc1 | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 1.0.0rc1 safetensors | 6.0e-08 | 4.8e-07 | 2.5e-07 | 0.0e+00 |
| downshift 1.0.0rc1 result cache on | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 1.0.0rc1 --metrics | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## MiniLM agreement with downshift --backend torch

short_b8 output, reference downshift_torch@1.0.0rc1.

| server | max abs err | min cosine |
|---|---|---|
| naive_sync | 7.2e-09 | 1.0000000 |
| naive_async | 7.2e-09 | 1.0000000 |
| downshift_auto@1.0.0rc1 | 1.3e-07 | 1.0000000 |
| downshift_auto_cache@1.0.0rc1 | 1.3e-07 | 1.0000000 |
| downshift_torch@1.0.0rc1 | 0.0e+00 | 1.0000000 |
| downshift_auto@0.5.1 | 1.3e-07 | 1.0000000 |
| downshift_torch@0.5.1 | 0.0e+00 | 1.0000000 |

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

| model | FastAPI + torch | FastAPI + ORT | torch, async def | downshift 0.5.1 | downshift 1.0.0rc1 |
|---|---|---|---|---|---|
| clean_mlp | 3.8 | 7.8 | 3.8 | 8.4 | 8.1 |
| tiny_bert | 12.9 | 18.1 | 12.9 | 17.4 | 17.4 |
| mlp_large | 4.5 | 7.8 | 3.8 | 7.8 | 8.9 |
| scatter_include_self_false | 3.8 | 7.3 | 3.8 | 7.5 | 7.3 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## Boot time, MiniLM (s)

Time from process start to first 200 answer.

| server (all-MiniLM-L6-v2) | ready s |
|---|---|
| naive FastAPI + transformers | 11.8 |
| naive transformers, async def | 11.8 |
| downshift 0.5.1 | 18.5 |
| downshift 0.5.1 --backend torch | 12.2 |
| downshift 1.0.0rc1 | 19.2 |
| downshift 1.0.0rc1 --backend torch | 12.3 |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)

## downshift check wall time, MiniLM

From results_hf.json.

| version | wall s | exit code |
|---|---|---|
| 1.0.0rc1 | 19.11 | 0 |
| 0.5.1 | 18.47 | 0 |

Source: [`bench/results/v1.0.0/results_hf.json`](../results/v1.0.0/results_hf.json)

## Workers sweep (0.4.0 and 0.5.0 run)

Requests/s at c=1, 8 and 32; p99 at c=8; seconds to ready. Not measured for 1.0.0rc1.

| model | version | workers | rps c=1 | rps c=8 | rps c=32 | p99 ms c=8 | boot s |
|---|---|---|---|---|---|---|---|
| clean_mlp | 0.4.0 | 1 | 861 | 887 | 915 | 13.4 | 7.7 |
| clean_mlp | 0.4.0 | 2 | 823 | 1789 | 1923 | 6.1 | 10.6 |
| clean_mlp | 0.4.0 | 4 | 846 | 3406 | 3418 | 4.2 | 10.6 |
| clean_mlp | 0.5.0 | 1 | 907 | 983 | 1111 | 12.7 | 9.2 |
| clean_mlp | 0.5.0 | 2 | 936 | 2076 | 1964 | 6.0 | 11.3 |
| clean_mlp | 0.5.0 | 4 | 974 | 3857 | 3845 | 3.7 | 12.3 |
| mlp_large | 0.4.0 | 1 | 468 | 745 | 719 | 18.6 | 7.9 |
| mlp_large | 0.4.0 | 2 | 418 | 1126 | 1260 | 13.9 | 10.9 |
| mlp_large | 0.4.0 | 4 | 475 | 1325 | 2003 | 11.9 | 11.3 |
| mlp_large | 0.5.0 | 1 | 515 | 819 | 827 | 14.2 | 8.2 |
| mlp_large | 0.5.0 | 2 | 514 | 1006 | 1203 | 19.5 | 11.3 |
| mlp_large | 0.5.0 | 4 | 525 | 1459 | 2089 | 10.6 | 11.3 |
| bert_small | 0.4.0 | 1 | 89 | 105 | 98 | 95.2 | 20.2 |
| bert_small | 0.4.0 | 2 | 90 | 142 | 143 | 111.6 | 23.1 |
| bert_small | 0.4.0 | 4 | 66 | 227 | 231 | 98.0 | 24.1 |
| bert_small | 0.5.0 | 1 | 91 | 108 | 101 | 84.8 | 20.3 |
| bert_small | 0.5.0 | 2 | 92 | 157 | 150 | 82.7 | 23.1 |
| bert_small | 0.5.0 | 4 | 70 | 239 | 248 | 58.9 | 23.6 |

Source: [`bench/results/v0.5.0/chart_runs.csv`](../results/v0.5.0/chart_runs.csv)

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

## Release to release, run B: 0.5.1 against 1.0.0rc1

Peak requests/s, batch 1, JSON, cache off, same session.

| model | 0.5.1 peak rps | 1.0.0rc1 peak rps | change | note |
|---|---|---|---|---|
| clean_mlp | 1421 | 1428 | +0.5% |  |
| tiny_bert | 1183 | 1067 | -9.8% |  |
| mlp_large | 965 | 765 | -20.7% | main pass was slow; re-run in the throughput page |
| scatter_include_self_false | 1443 | 1387 | -3.8% |  |

Source: [`bench/results/v1.0.0/chart_runs.csv`](../results/v1.0.0/chart_runs.csv)
