# downshift benchmark: naive servers vs downshift 0.5.1 vs 1.0.0rc1

`win32`, 16 logical CPUs, Python 3.12.10, torch 2.14.0+cpu (CPU), onnxruntime 1.30.0, run 2026-11-07T22:59:00. downshift 1.0.0rc1 from the checkout; downshift 0.5.1 from the pip-installed release. Every downshift server is the real `downshift serve` CLI; all servers ran in one session on the same machine.

## Fixture and compute-heavy models

Closed-loop load, 3.0 s window after 1.0 s warm-up, load generator sharded over up to 6 processes on the same machine. Every server gets the same seeded weights and byte-identical request bodies, and every row's response is checked element-wise against eager PyTorch.

Load-generator ceiling (`GET /health` on a do-nothing server): c=1: 5194 rps, c=2: 9606 rps, c=4: 10446 rps, c=8: 11514 rps, c=16: 12102 rps, c=32: 11925 rps, c=64: 10707 rps. A result near these numbers is measuring the client.

### Verdicts

| model | verdict | backend chosen | max abs err |
|---|---|---|---|
| `clean_mlp` | CLEAN | onnxruntime | 3.7e-08 |
| `tiny_bert` | CLEAN | onnxruntime | 4.8e-07 |
| `mlp_large` | CLEAN | onnxruntime | 2.1e-07 |
| `scatter_include_self_false` | DEGRADED | torch | 1.6e+00 |

### Peak throughput, batch 1

Best requests/s over the concurrency sweep, and p50 latency at concurrency 1. `async def` blocks the event loop and is shown as the cautionary row.

| server | `clean_mlp` | `tiny_bert` | `mlp_large` | `scatter_include_self_false` |
|---|---|---|---|---|
| naive FastAPI + torch | 1698 | 350 | 984 | 1663 |
| naive FastAPI + ONNX Runtime | 2069 | 1329 | 808 | 2043 |
| naive torch, `async def` | 3216 | 543 | 664 | 2899 |
| downshift 0.5.1 | 1421 | 1183 | 965 | 1443 |
| downshift 0.5.1 safetensors | 1416 | 1005 | 714 | 1362 |
| downshift 1.0.0rc1 | 1428 | 1067 | 765 | 1387 |
| downshift 1.0.0rc1 safetensors | 1317 | 924 | 753 | 1334 |
| downshift 1.0.0rc1 result cache on | 1824 | 914 | 1742 | 1364 |
| downshift 1.0.0rc1 `--metrics` | 1325 | 1060 | 919 | 1313 |

p50 latency (ms) at concurrency 1:

| server | `clean_mlp` | `tiny_bert` | `mlp_large` | `scatter_include_self_false` |
|---|---|---|---|---|
| naive FastAPI + torch | 0.62 | 1.83 | 2.41 | 0.67 |
| naive FastAPI + ONNX Runtime | 0.61 | 0.91 | 1.80 | 0.60 |
| naive torch, `async def` | 0.42 | 1.39 | 1.89 | 0.46 |
| downshift 0.5.1 | 0.80 | 1.09 | 1.92 | 0.76 |
| downshift 0.5.1 safetensors | 0.83 | 1.17 | 1.92 | 0.77 |
| downshift 1.0.0rc1 | 0.77 | 1.16 | 2.17 | 0.76 |
| downshift 1.0.0rc1 safetensors | 0.81 | 1.34 | 1.93 | 0.79 |
| downshift 1.0.0rc1 result cache on | 0.69 | 1.14 | 0.80 | 0.89 |
| downshift 1.0.0rc1 `--metrics` | 0.81 | 1.15 | 1.95 | 0.80 |

### Concurrency sweep, batch 1 (requests/s)

`clean_mlp`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1410 | 1426 | 1606 | 1629 | 1671 | 1698 | 1504 |
| naive FastAPI + ONNX Runtime | 1393 | 1574 | 1975 | 1849 | 2069 | 2016 | 1832 |
| naive torch, `async def` | 2142 | 3157 | 3133 | 3216 | 3200 | 3112 | 3085 |
| downshift 0.5.1 | 1039 | 1284 | 1331 | 1323 | 1311 | 1421 | 1339 |
| downshift 0.5.1 safetensors | 1015 | 1254 | 1277 | 1282 | 1294 | 1416 | 1320 |
| downshift 1.0.0rc1 | 1114 | 1249 | 1306 | 1368 | 1428 | 1388 | 1187 |
| downshift 1.0.0rc1 safetensors | 1018 | 1228 | 1317 | 1249 | 1264 | 1305 | 1239 |
| downshift 1.0.0rc1 result cache on | 1224 | 1615 | 1824 | 1723 | 1642 | 1632 | 1406 |
| downshift 1.0.0rc1 `--metrics` | 1054 | 1250 | 1271 | 1289 | 1325 | 1286 | 1234 |

`tiny_bert`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 342 | 336 | 350 | 331 | 290 | 312 | 268 |
| naive FastAPI + ONNX Runtime | 719 | 937 | 1231 | 1295 | 1298 | 1329 | 1212 |
| naive torch, `async def` | 451 | 494 | 542 | 539 | 494 | 543 | 513 |
| downshift 0.5.1 | 591 | 828 | 1042 | 1035 | 1183 | 966 | 965 |
| downshift 0.5.1 safetensors | 561 | 783 | 925 | 959 | 1005 | 954 | 959 |
| downshift 1.0.0rc1 | 633 | 887 | 1032 | 1020 | 1067 | 1029 | 937 |
| downshift 1.0.0rc1 safetensors | 530 | 748 | 892 | 912 | 924 | 890 | 839 |
| downshift 1.0.0rc1 result cache on | 539 | 757 | 914 | 896 | 887 | 879 | 794 |
| downshift 1.0.0rc1 `--metrics` | 610 | 872 | 1008 | 1060 | 1007 | 981 | 960 |

`mlp_large`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 389 | 499 | 804 | 888 | 984 | 984 | 945 |
| naive FastAPI + ONNX Runtime | 383 | 437 | 735 | 725 | 784 | 808 | 785 |
| naive torch, `async def` | 478 | 569 | 651 | 629 | 664 | 652 | 621 |
| downshift 0.5.1 | 420 | 567 | 888 | 907 | 965 | 964 | 937 |
| downshift 0.5.1 safetensors | 435 | 582 | 691 | 694 | 714 | 684 | 710 |
| downshift 1.0.0rc1 | 302 | 435 | 636 | 746 | 739 | 736 | 765 |
| downshift 1.0.0rc1 safetensors | 438 | 597 | 675 | 711 | 701 | 753 | 688 |
| downshift 1.0.0rc1 result cache on | 1078 | 1416 | 1693 | 1736 | 1742 | 1560 | 1506 |
| downshift 1.0.0rc1 `--metrics` | 412 | 581 | 806 | 870 | 919 | 889 | 900 |

`scatter_include_self_false`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1348 | 1544 | 1663 | 1575 | 1620 | 1614 | 1374 |
| naive FastAPI + ONNX Runtime | 1514 | 1744 | 2000 | 2043 | 2007 | 1990 | 1826 |
| naive torch, `async def` | 2035 | 2740 | 2759 | 2851 | 2899 | 2887 | 2868 |
| downshift 0.5.1 | 1222 | 1406 | 1443 | 1301 | 1348 | 1258 | 1220 |
| downshift 0.5.1 safetensors | 1210 | 1343 | 1362 | 1259 | 1309 | 1248 | 1188 |
| downshift 1.0.0rc1 | 1186 | 1348 | 1387 | 1217 | 1298 | 1323 | 1246 |
| downshift 1.0.0rc1 safetensors | 1169 | 1300 | 1334 | 1243 | 1283 | 1266 | 1198 |
| downshift 1.0.0rc1 result cache on | 1023 | 1285 | 1364 | 1274 | 1286 | 1294 | 1220 |
| downshift 1.0.0rc1 `--metrics` | 1167 | 1313 | 1299 | 1216 | 1253 | 1265 | 1189 |

### Batched requests at concurrency 8 (rows/s)

Requests/s times batch size: the work actually done.

| server | `clean_mlp` b8 | `clean_mlp` b32 | `tiny_bert` b8 | `tiny_bert` b32 | `mlp_large` b8 | `mlp_large` b32 | `scatter_include_self_false` b8 | `scatter_include_self_false` b32 |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 7972 | 30507 | 2384 | 7314 | 1673 | 2386 | 8970 | 17310 |
| naive FastAPI + ONNX Runtime | 14257 | 46352 | 7986 | 16548 | 2076 | 2916 | 14572 | 29924 |
| downshift 0.5.1 | 10559 | 48023 | 7721 | 24605 | 4724 | 8022 | 10158 | 32435 |
| downshift 0.5.1 safetensors | 10291 | 40092 | 7501 | 30833 | 7339 | 25221 | 7856 | 42325 |
| downshift 1.0.0rc1 | 9788 | 43150 | 7783 | 24724 | 3714 | 6561 | 9700 | 29181 |
| downshift 1.0.0rc1 safetensors | 10076 | 37558 | 7127 | 31813 | 7277 | 25378 | 7597 | 40579 |
| downshift 1.0.0rc1 result cache on | 11922 | 43426 | 6645 | 21697 | 6717 | 9239 | 9199 | 29699 |
| downshift 1.0.0rc1 `--metrics` | 10093 | 46029 | 7639 | 23325 | 4296 | 7873 | 9221 | 27496 |

### Wire encodings, batch 32, concurrency 8

| model | server | request bytes | requests/s | p50 ms |
|---|---|---|---|---|
| `clean_mlp` | downshift 0.5.1 | 10659 | 1501 | 5.75 |
| `clean_mlp` | downshift 0.5.1 safetensors | 2120 | 1253 | 6.02 |
| `clean_mlp` | downshift 1.0.0rc1 | 10659 | 1348 | 5.78 |
| `clean_mlp` | downshift 1.0.0rc1 safetensors | 2120 | 1174 | 6.34 |
| `tiny_bert` | downshift 0.5.1 | 1941 | 769 | 10.39 |
| `tiny_bert` | downshift 0.5.1 safetensors | 4248 | 964 | 7.59 |
| `tiny_bert` | downshift 1.0.0rc1 | 1941 | 773 | 10.09 |
| `tiny_bert` | downshift 1.0.0rc1 safetensors | 4248 | 994 | 7.16 |
| `mlp_large` | downshift 0.5.1 | 338169 | 251 | 32.70 |
| `mlp_large` | downshift 0.5.1 safetensors | 65608 | 788 | 10.04 |
| `mlp_large` | downshift 1.0.0rc1 | 338169 | 205 | 39.93 |
| `mlp_large` | downshift 1.0.0rc1 safetensors | 65608 | 793 | 10.15 |
| `scatter_include_self_false` | downshift 0.5.1 | 32706 | 1014 | 8.29 |
| `scatter_include_self_false` | downshift 0.5.1 safetensors | 7824 | 1323 | 6.37 |
| `scatter_include_self_false` | downshift 1.0.0rc1 | 32706 | 912 | 9.08 |
| `scatter_include_self_false` | downshift 1.0.0rc1 safetensors | 7824 | 1268 | 6.64 |

### Correctness

Worst max absolute error against eager PyTorch over every row a server answered.

| server | `clean_mlp` | `tiny_bert` | `mlp_large` | `scatter_include_self_false` |
|---|---|---|---|---|
| naive FastAPI + torch | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| naive FastAPI + ONNX Runtime | 6.0e-08 | 4.8e-07 | 2.5e-07 | 1.2e+00 |
| naive torch, `async def` | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| downshift 0.5.1 | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 0.5.1 safetensors | 6.0e-08 | 4.8e-07 | 2.5e-07 | 0.0e+00 |
| downshift 1.0.0rc1 | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 1.0.0rc1 safetensors | 6.0e-08 | 4.8e-07 | 2.5e-07 | 0.0e+00 |
| downshift 1.0.0rc1 result cache on | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |
| downshift 1.0.0rc1 `--metrics` | 6.4e-08 | 5.8e-07 | 2.6e-07 | 2.8e-08 |

Failed requests across every load window: 0.

## Real Hugging Face models

Text in, through the real CLI. 5.0 s windows at concurrency 1, 8; a `/health` probe runs every 50 ms during each window (what a liveness probe sees). Payloads: `short_b1` one sentence, `short_b8` eight sentences, `long_b1` ~180 tokens.

### all-MiniLM-L6-v2

`downshift check` wall time: 1.0.0rc1 19.1 s (exit 0), 0.5.1 18.5 s (exit 0).

| server | ready s | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | worst /health p99 ms |
|---|---|---|---|---|---|
| naive FastAPI + transformers | 11.8 | 141 | 84 | 71 | 21 |
| naive transformers, `async def` | 11.8 | 105 | 47 | 35 | 420 |
| downshift 0.5.1 | 18.5 | 524 | 165 | 94 | 24 |
| downshift 0.5.1 `--backend torch` | 12.2 | 156 | 90 | 69 | 19 |
| downshift 1.0.0rc1 | 19.2 | 486 | 158 | 89 | 27 |
| downshift 1.0.0rc1 `--backend torch` | 12.3 | 146 | 82 | 68 | 19 |
| downshift 1.0.0rc1 result cache on | 19.2 | 1911 | 1627 | 1851 | 194 |

p50 latency (ms) at concurrency 1:

| server | short_b1 | short_b8 | long_b1 |
|---|---|---|---|
| naive FastAPI + transformers | 9.2 | 21.3 | 27.9 |
| naive transformers, `async def` | 8.4 | 21.1 | 26.0 |
| downshift 0.5.1 | 4.6 | 15.7 | 27.9 |
| downshift 0.5.1 `--backend torch` | 9.8 | 21.0 | 25.7 |
| downshift 1.0.0rc1 | 4.9 | 17.4 | 28.4 |
| downshift 1.0.0rc1 `--backend torch` | 10.3 | 22.7 | 27.9 |
| downshift 1.0.0rc1 result cache on | 0.6 | 0.7 | 0.6 |

Agreement with downshift 1.0.0rc1 `--backend torch` on `short_b8`:

| server | max abs err | min cosine |
|---|---|---|
| naive FastAPI + transformers | 7.2e-09 | 1.0000000 |
| naive transformers, `async def` | 7.2e-09 | 1.0000000 |
| downshift 0.5.1 | 1.3e-07 | 1.0000000 |
| downshift 0.5.1 `--backend torch` | 0.0e+00 | 1.0000000 |
| downshift 1.0.0rc1 | 1.3e-07 | 1.0000000 |
| downshift 1.0.0rc1 `--backend torch` | 0.0e+00 | 1.0000000 |
| downshift 1.0.0rc1 result cache on | 1.3e-07 | 1.0000000 |

## Where the time goes

Median of 300 sequential requests on one keep-alive connection, batch 1 (HF: one sentence). `total` is what the client measured; the stage columns are the server's own `Server-Timing`; `HTTP + hand-offs` is the rest: uvicorn, routing, middleware and thread hand-offs back to the event loop. Naive servers have no `Server-Timing`, so their total is all in the last column. 0.4.0 reports one `codec` stage (decode and encode); 0.5.0 splits it, and its `encode` includes serializing the response.

### clean_mlp

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.61 | — | — | — | — | — | — | 0.61 |
| naive FastAPI + ONNX Runtime | 0.58 | — | — | — | — | — | — | 0.58 |
| naive torch, async def | 0.36 | — | — | — | — | — | — | 0.36 |
| downshift 0.5.1 | 0.71 | 0.01 | 0.03 | 0.02 | 0.03 | 0.14 | 0.03 | 0.43 |
| downshift 0.5.1 safetensors | 0.87 | 0.04 | 0.03 | 0.02 | 0.03 | 0.20 | 0.04 | 0.48 |
| downshift 1.0.0rc1 | 0.70 | 0.01 | 0.03 | 0.02 | 0.03 | 0.13 | 0.02 | 0.46 |
| downshift 1.0.0rc1 safetensors | 0.78 | 0.03 | 0.03 | 0.02 | 0.03 | 0.15 | 0.03 | 0.48 |
| downshift 1.0.0rc1 result cache on | 0.66 | 0.02 | 0.03 | 0.04 | 0.00 | 0.00 | 0.02 | 0.55 |
| downshift 1.0.0rc1 --metrics | 0.77 | 0.02 | 0.03 | 0.02 | 0.04 | 0.17 | 0.03 | 0.48 |

### tiny_bert

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1.75 | — | — | — | — | — | — | 1.75 |
| naive FastAPI + ONNX Runtime | 0.77 | — | — | — | — | — | — | 0.77 |
| naive torch, async def | 1.31 | — | — | — | — | — | — | 1.31 |
| downshift 0.5.1 | 1.06 | 0.02 | 0.03 | 0.02 | 0.04 | 0.41 | 0.04 | 0.51 |
| downshift 0.5.1 safetensors | 1.09 | 0.05 | 0.04 | 0.03 | 0.04 | 0.41 | 0.04 | 0.50 |
| downshift 1.0.0rc1 | 1.03 | 0.02 | 0.04 | 0.02 | 0.04 | 0.40 | 0.04 | 0.50 |
| downshift 1.0.0rc1 safetensors | 1.11 | 0.05 | 0.04 | 0.03 | 0.04 | 0.41 | 0.04 | 0.52 |
| downshift 1.0.0rc1 result cache on | 1.03 | 0.02 | 0.04 | 0.03 | 0.04 | 0.40 | 0.04 | 0.51 |
| downshift 1.0.0rc1 --metrics | 1.04 | 0.02 | 0.03 | 0.02 | 0.04 | 0.41 | 0.04 | 0.49 |

### mlp_large

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 2.26 | — | — | — | — | — | — | 2.26 |
| naive FastAPI + ONNX Runtime | 1.66 | — | — | — | — | — | — | 1.66 |
| naive torch, async def | 1.82 | — | — | — | — | — | — | 1.82 |
| downshift 0.5.1 | 1.79 | 0.07 | 0.04 | 0.06 | 0.04 | 0.81 | 0.07 | 0.70 |
| downshift 0.5.1 safetensors | 1.75 | 0.06 | 0.04 | 0.03 | 0.05 | 0.80 | 0.06 | 0.71 |
| downshift 1.0.0rc1 | 2.04 | 0.06 | 0.05 | 0.06 | 0.04 | 0.84 | 0.07 | 0.85 |
| downshift 1.0.0rc1 safetensors | 1.75 | 0.06 | 0.04 | 0.03 | 0.04 | 0.79 | 0.05 | 0.71 |
| downshift 1.0.0rc1 result cache on | 0.76 | 0.04 | 0.03 | 0.08 | 0.00 | 0.00 | 0.04 | 0.57 |
| downshift 1.0.0rc1 --metrics | 1.80 | 0.06 | 0.05 | 0.06 | 0.04 | 0.80 | 0.07 | 0.72 |

### scatter_include_self_false

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.63 | — | — | — | — | — | — | 0.63 |
| naive FastAPI + ONNX Runtime | 0.54 | — | — | — | — | — | — | 0.54 |
| naive torch, async def | 0.40 | — | — | — | — | — | — | 0.40 |
| downshift 0.5.1 | 0.72 | 0.02 | 0.03 | 0.03 | 0.03 | 0.12 | 0.02 | 0.46 |
| downshift 0.5.1 safetensors | 0.72 | 0.04 | 0.03 | 0.02 | 0.03 | 0.12 | 0.03 | 0.45 |
| downshift 1.0.0rc1 | 0.70 | 0.02 | 0.03 | 0.03 | 0.03 | 0.11 | 0.02 | 0.45 |
| downshift 1.0.0rc1 safetensors | 0.75 | 0.04 | 0.03 | 0.02 | 0.03 | 0.13 | 0.03 | 0.47 |
| downshift 1.0.0rc1 result cache on | 0.85 | 0.02 | 0.04 | 0.04 | 0.04 | 0.15 | 0.03 | 0.53 |
| downshift 1.0.0rc1 --metrics | 0.73 | 0.02 | 0.03 | 0.03 | 0.03 | 0.12 | 0.02 | 0.47 |

### all-MiniLM-L6-v2

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 8.77 | — | — | — | — | — | — | 8.77 |
| naive transformers, async def | 8.54 | — | — | — | — | — | — | 8.54 |
| downshift 0.5.1 | 4.39 | 0.04 | 0.05 | 0.28 | 0.05 | 2.82 | 0.09 | 0.93 |
| downshift 0.5.1 --backend torch | 9.08 | 0.04 | 0.06 | 0.30 | 0.04 | 7.34 | 0.09 | 1.09 |
| downshift 1.0.0rc1 | 4.81 | 0.04 | 0.06 | 0.32 | 0.05 | 3.00 | 0.09 | 1.05 |
| downshift 1.0.0rc1 --backend torch | 10.55 | 0.04 | 0.06 | 0.34 | 0.04 | 8.56 | 0.09 | 1.20 |
| downshift 1.0.0rc1 result cache on | 0.58 | 0.01 | 0.03 | 0.01 | 0.00 | 0.00 | 0.03 | 0.49 |

## Notes on this run

- **The result cache is off in every headline row.** Version 1.0.0 turns the result cache on by default for a model with independent rows. The harness sends the same body again and again, so a cache would answer from memory and the row would not measure the model. Every `1.0.0rc1` row but "result cache on" starts with `--result-cache-rows 0`. Version 0.5.1 has no cache. The "result cache on" rows use the default and send the same body every time, so they show the best case (every request is a hit), not a normal workload.
- **The cache does nothing for two of the four fixtures.** `downshift check` says `Rows dependent` for `tiny_bert` and `Rows not checked` for `scatter_include_self_false` (the numerics check failed). The default cache is off for both, so their "result cache on" rows are an ordinary run and differ from the default rows only by noise.
- **`--metrics` rows** start the same server as the default row with `--metrics` added.
- **Slow `mlp_large` row.** In the main pass, 1.0.0rc1 default on `mlp_large` reached 765 rps (c=64, p99 240 ms) against 965 rps for 0.5.1, and its `--metrics` row reached 919 rps. That does not fit one code path, so the case was measured again with only batch 1 and the variants `downshift` and `downshift_metrics` (file `rerun_mlp_large.json`, same session, nothing else running): 1.0.0rc1 default peak 939 rps, `--metrics` 920 rps, 0.5.1 989 rps. The main-pass row is kept in the tables above. Use the re-run for comparisons on `mlp_large`.
- Windows are 3 s on the fixtures and 5 s on MiniLM, as in the 0.5.0 run. The `--workers` sweep was not part of this run, and Prompt Guard was not measured.
- Machine state differs from the 0.5.0 run (the hand-rolled servers are faster here too), so compare rows inside this report, not with `results/v0.5.0/`.
