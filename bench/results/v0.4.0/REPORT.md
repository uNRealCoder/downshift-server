# downshift serving benchmark

downshift 0.4.0 on `win32`, 16 logical CPUs, Python 3.12.10, torch 2.14.0+cpu (CPU), onnxruntime 1.30.0, torch intra-op threads 8. Run 2026-09-26T23:28:33.

Closed-loop load, 3.0s measurement window after 1.0s warmup, load generator sharded over up to 6 processes on the same machine as the server. Every variant is handed the identical prepared module and the identical exported ONNX graph and receives byte-identical request bodies; the only variable is the server.

## Load generator ceiling

How fast this harness can drive a do-nothing endpoint on this machine. Any server result approaching these numbers is measuring the client, not the server.

| concurrency | GET /health rps |
|-------------|-----------------|
| 1           | 4019.5          |
| 2           | 7596.9          |
| 4           | 7575.4          |
| 8           | 7849.8          |
| 16          | 7909.5          |
| 32          | 7745.6          |
| 64          | 7032.7          |

## Verdicts

| model                        | family          | verdict  | backend chosen | max abs err |
|------------------------------|-----------------|----------|----------------|-------------|
| `clean_mlp`                  | generic-torch   | CLEAN    | onnxruntime    | 3.73e-08    |
| `dynamic_batch_cnn`          | generic-torch   | CLEAN    | onnxruntime    | 2.98e-08    |
| `gnn_gcn`                    | pyg             | CLEAN    | onnxruntime    | 2.38e-07    |
| `tiny_bert`                  | hf-transformers | CLEAN    | onnxruntime    | 4.77e-07    |
| `scatter_include_self_false` | generic-torch   | DEGRADED | torch          | 1.59e+00    |
| `mlp_large`                  | generic-torch   | CLEAN    | onnxruntime    | 2.09e-07    |
| `cnn_large`                  | generic-torch   | CLEAN    | onnxruntime    | 5.59e-08    |
| `bert_small`                 | hf-transformers | CLEAN    | onnxruntime    | 1.91e-06    |

## Inference cost with no server attached

`compute` is `infer(feeds)` on arrays already in memory. `+json` adds parsing the request body, building the arrays, `.tolist()` on the outputs and serializing the response — the cost of speaking JSON, before any ASGI or socket work. `+binary` is the same round trip with base64 tensor bodies: `b64decode` + `np.frombuffer` in, `b64encode` of the output bytes out. `request` is the nested-list body; `binary req` is the base64 body carrying the same tensors.

| model                        | batch | torch ms | ORT ms  | ORT speedup | torch +json ms | ORT +json ms | torch +binary ms | ORT +binary ms | request    | binary req |
|------------------------------|-------|----------|---------|-------------|----------------|--------------|------------------|----------------|------------|------------|
| `clean_mlp`                  | 1     | 0.045    | 0.020   | 2.21x       | 0.088          | 0.050        | 0.090            | 0.047          | 0.3 KiB    | 0.2 KiB    |
| `clean_mlp`                  | 8     | 0.057    | 0.024   | 2.34x       | 0.218          | 0.162        | 0.121            | 0.054          | 2.6 KiB    | 0.8 KiB    |
| `clean_mlp`                  | 32    | 0.056    | 0.024   | 2.31x       | 0.615          | 0.557        | 0.142            | 0.069          | 10.4 KiB   | 2.8 KiB    |
| `dynamic_batch_cnn`          | 1     | 0.100    | 0.026   | 3.80x       | 0.750          | 0.619        | 0.228            | 0.091          | 15.6 KiB   | 4.1 KiB    |
| `dynamic_batch_cnn`          | 8     | 0.277    | 0.050   | 5.53x       | 4.893          | 4.679        | 0.612            | 0.329          | 124.7 KiB  | 32.1 KiB   |
| `dynamic_batch_cnn`          | 32    | 0.340    | 0.075   | 4.53x       | 19.209         | 18.684       | 1.435            | 1.103          | 498.4 KiB  | 128.1 KiB  |
| `gnn_gcn`                    | 1     | 0.777    | 0.107   | 7.25x       | 0.939          | 0.206        | 0.898            | 0.159          | 1.1 KiB    | 0.6 KiB    |
| `gnn_gcn`                    | 8     | 0.832    | 0.169   | 4.94x       | 1.488          | 0.731        | 0.996            | 0.193          | 8.5 KiB    | 3.8 KiB    |
| `gnn_gcn`                    | 32    | 0.942    | 0.362   | 2.60x       | 3.603          | 2.282        | 1.200            | 0.419          | 34.1 KiB   | 14.8 KiB   |
| `tiny_bert`                  | 1     | 1.327    | 0.261   | 5.08x       | 1.547          | 0.431        | 1.427            | 0.328          | 0.1 KiB    | 0.3 KiB    |
| `tiny_bert`                  | 8     | 1.419    | 0.478   | 2.97x       | 2.645          | 1.633        | 1.608            | 0.598          | 0.5 KiB    | 1.5 KiB    |
| `tiny_bert`                  | 32    | 1.598    | 1.216   | 1.31x       | 6.356          | 5.635        | 1.975            | 1.456          | 1.9 KiB    | 5.5 KiB    |
| `scatter_include_self_false` | 1     | 0.060    | 0.031   | 1.94x       | 0.157          | 0.116        | 0.138            | 0.069          | 1.0 KiB    | 0.5 KiB    |
| `scatter_include_self_false` | 8     | 0.094    | 0.043   | 2.20x       | 0.494          | 0.395        | 0.204            | 0.095          | 8.0 KiB    | 2.7 KiB    |
| `scatter_include_self_false` | 32    | 0.149    | 0.071   | 2.09x       | 1.426          | 1.176        | 0.284            | 0.144          | 31.9 KiB   | 10.2 KiB   |
| `mlp_large`                  | 1     | 1.336    | 0.276   | 4.84x       | 1.895          | 1.367        | 1.237            | 0.369          | 10.3 KiB   | 2.8 KiB    |
| `mlp_large`                  | 8     | 2.088    | 0.491   | 4.25x       | 9.409          | 7.771        | 2.683            | 0.863          | 82.6 KiB   | 21.4 KiB   |
| `mlp_large`                  | 32    | 3.248    | 2.063   | 1.57x       | 30.882         | 30.141       | 4.693            | 3.413          | 330.4 KiB  | 85.4 KiB   |
| `cnn_large`                  | 1     | 1.105    | 0.326   | 3.39x       | 3.427          | 2.619        | 1.309            | 0.502          | 62.1 KiB   | 16.1 KiB   |
| `cnn_large`                  | 8     | 4.620    | 2.183   | 2.12x       | 23.027         | 20.188       | 5.731            | 3.203          | 496.8 KiB  | 128.1 KiB  |
| `cnn_large`                  | 32    | 17.626   | 8.538   | 2.06x       | 85.390         | 80.717       | 21.201           | 12.423         | 1986.8 KiB | 512.1 KiB  |
| `bert_small`                 | 1     | 8.132    | 7.165   | 1.13x       | 41.221         | 39.533       | 9.513            | 8.524          | 1.3 KiB    | 2.8 KiB    |
| `bert_small`                 | 8     | 37.124   | 48.920  | 0.76x       | 284.229        | 308.531      | 49.816           | 61.224         | 9.7 KiB    | 21.5 KiB   |
| `bert_small`                 | 32    | 148.069  | 193.011 | 0.77x       | 1014.943       | 1053.389     | 197.237          | 236.832        | 38.7 KiB   | 85.5 KiB   |

## Throughput vs concurrency (batch 1)

Requests per second, closed loop. Higher is better.

**`clean_mlp`** — CLEAN, downshift serves via onnxruntime

| server                             | c=1    | c=2    | c=4    | c=8    | c=16   | c=32   | c=64   |
|------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| naive FastAPI + eager torch        | 1138.4 | 1304.2 | 1339.0 | 1283.8 | 1308.4 | 1298.3 | 1190.6 |
| naive FastAPI + ONNX Runtime       | 1110.9 | 1466.5 | 1615.3 | 1608.5 | 1608.7 | 1600.8 | 1415.4 |
| downshift serve (auto)             | 691.5  | 763.2  | 715.6  | 795.0  | 758.6  | 773.1  | 778.7  |
| downshift serve (base64)           | 794.5  | 879.1  | 852.6  | 810.1  | 864.4  | 823.3  | 726.1  |
| naive FastAPI + torch, `async def` | 1936.7 | 2356.6 | 2394.8 | 2435.3 | 2537.6 | 2602.0 | 2391.7 |

**`dynamic_batch_cnn`** — CLEAN, downshift serves via onnxruntime

| server                             | c=1   | c=2   | c=4    | c=8    | c=16  | c=32   | c=64  |
|------------------------------------|-------|-------|--------|--------|-------|--------|-------|
| naive FastAPI + eager torch        | 654.2 | 770.2 | 929.3  | 853.5  | 818.4 | 762.2  | 722.4 |
| naive FastAPI + ONNX Runtime       | 709.9 | 873.3 | 1061.0 | 981.7  | 881.1 | 886.2  | 791.5 |
| downshift serve (auto)             | 609.3 | 713.1 | 841.8  | 887.6  | 871.9 | 793.2  | 783.9 |
| downshift serve (base64)           | 538.8 | 662.2 | 862.3  | 936.6  | 909.3 | 888.8  | 925.5 |
| naive FastAPI + torch, `async def` | 830.3 | 894.9 | 1009.3 | 1003.9 | 998.4 | 1039.9 | 988.8 |

**`gnn_gcn`** — CLEAN, downshift serves via onnxruntime

| server                             | c=1    | c=2    | c=4    | c=8    | c=16   | c=32   | c=64   |
|------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| naive FastAPI + eager torch        | 549.4  | 539.1  | 504.9  | 444.8  | 401.6  | 359.4  | 354.1  |
| naive FastAPI + ONNX Runtime       | 1010.6 | 1360.0 | 1417.1 | 1440.5 | 1484.3 | 1406.0 | 1308.4 |
| downshift serve (auto)             | 692.3  | 845.2  | 834.5  | 723.5  | 775.4  | 753.9  | 680.2  |
| downshift serve (base64)           | 691.1  | 816.8  | 797.0  | 700.2  | 731.9  | 755.3  | 660.0  |
| naive FastAPI + torch, `async def` | 686.6  | 679.5  | 678.6  | 682.2  | 689.4  | 673.1  | 664.3  |

**`tiny_bert`** — CLEAN, downshift serves via onnxruntime

| server                             | c=1   | c=2    | c=4    | c=8    | c=16   | c=32   | c=64   |
|------------------------------------|-------|--------|--------|--------|--------|--------|--------|
| naive FastAPI + eager torch        | 384.2 | 444.4  | 407.8  | 391.8  | 353.7  | 319.8  | 321.1  |
| naive FastAPI + ONNX Runtime       | 872.9 | 1257.6 | 1475.4 | 1485.8 | 1502.5 | 1454.3 | 1245.7 |
| downshift serve (auto)             | 581.1 | 718.0  | 792.0  | 807.2  | 790.9  | 782.0  | 694.1  |
| downshift serve (base64)           | 575.4 | 687.4  | 769.4  | 769.9  | 773.8  | 756.2  | 680.0  |
| naive FastAPI + torch, `async def` | 539.1 | 516.7  | 523.9  | 525.3  | 528.0  | 524.0  | 521.2  |

**`scatter_include_self_false`** — DEGRADED, downshift serves via torch

| server                             | c=1    | c=2    | c=4    | c=8    | c=16   | c=32   | c=64   |
|------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| naive FastAPI + eager torch        | 1052.0 | 1267.1 | 1242.5 | 1160.2 | 1200.9 | 1213.5 | 1042.6 |
| naive FastAPI + ONNX Runtime       | 1132.0 | 1474.3 | 1571.9 | 1546.9 | 1568.5 | 1550.8 | 1412.1 |
| downshift serve (auto)             | 790.6  | 921.3  | 907.9  | 920.8  | 905.4  | 863.5  | 819.2  |
| downshift serve (base64)           | 792.9  | 893.9  | 915.1  | 912.2  | 900.3  | 862.5  | 863.3  |
| naive FastAPI + torch, `async def` | 1617.1 | 1891.8 | 1897.0 | 2003.7 | 2008.3 | 2041.4 | 1936.2 |

**`mlp_large`** — CLEAN, downshift serves via onnxruntime

| server                             | c=1   | c=2   | c=4   | c=8   | c=16  | c=32  | c=64  |
|------------------------------------|-------|-------|-------|-------|-------|-------|-------|
| naive FastAPI + eager torch        | 358.4 | 410.9 | 643.9 | 696.9 | 768.6 | 746.5 | 709.5 |
| naive FastAPI + ONNX Runtime       | 511.0 | 637.1 | 859.6 | 870.0 | 877.6 | 876.9 | 815.4 |
| downshift serve (auto)             | 424.9 | 553.7 | 651.3 | 678.5 | 682.3 | 687.9 | 663.2 |
| downshift serve (base64)           | 443.6 | 550.7 | 592.8 | 596.8 | 587.7 | 575.3 | 524.8 |
| naive FastAPI + torch, `async def` | 456.1 | 502.9 | 566.7 | 551.7 | 586.8 | 586.1 | 576.2 |

**`cnn_large`** — CLEAN, downshift serves via onnxruntime

| server                             | c=1   | c=2   | c=4   | c=8   | c=16  | c=32  | c=64  |
|------------------------------------|-------|-------|-------|-------|-------|-------|-------|
| naive FastAPI + eager torch        | 257.5 | 297.1 | 358.6 | 336.4 | 326.6 | 305.5 | 295.0 |
| naive FastAPI + ONNX Runtime       | 338.0 | 401.8 | 430.5 | 412.2 | 398.5 | 363.9 | 353.0 |
| downshift serve (auto)             | 430.3 | 537.6 | 626.2 | 662.4 | 612.3 | 580.7 | 529.2 |
| downshift serve (base64)           | 552.7 | 698.8 | 862.6 | 901.6 | 907.8 | 870.0 | 811.2 |
| naive FastAPI + torch, `async def` | 282.4 | 273.9 | 308.6 | 300.7 | 303.1 | 296.8 | 300.3 |

**`bert_small`** — CLEAN, downshift serves via onnxruntime

| server                             | c=1   | c=2   | c=4   | c=8   | c=16  | c=32  | c=64  |
|------------------------------------|-------|-------|-------|-------|-------|-------|-------|
| naive FastAPI + eager torch        | 31.6  | 71.4  | 99.0  | 98.1  | 89.5  | 80.3  | 68.2  |
| naive FastAPI + ONNX Runtime       | 30.9  | 76.0  | 126.3 | 140.0 | 151.5 | 140.1 | 133.6 |
| downshift serve (auto)             | 86.7  | 95.7  | 102.6 | 99.0  | 98.1  | 94.8  | 88.3  |
| downshift serve (base64)           | 104.7 | 111.6 | 119.0 | 121.5 | 118.9 | 112.0 | 106.9 |
| naive FastAPI + torch, `async def` | 31.1  | 64.4  | 72.0  | 72.0  | 68.5  | 65.7  | 57.3  |

## Peak throughput and where it lands (batch 1)

| model                        | server                             | peak rps | at   | vs naive torch | p50 ms | p99 ms | max abs err |
|------------------------------|------------------------------------|----------|------|----------------|--------|--------|-------------|
| `clean_mlp`                  | naive FastAPI + eager torch        | 1339.0   | c=4  | 1.00x          | 2.75   | 6.16   | 0.00e+00    |
| `clean_mlp`                  | naive FastAPI + ONNX Runtime       | 1615.3   | c=4  | 1.21x          | 2.26   | 4.75   | 7.45e-09    |
| `clean_mlp`                  | downshift serve (auto)             | 795.0    | c=8  | 0.59x          | 9.82   | 17.12  | 9.22e-09    |
| `clean_mlp`                  | downshift serve (base64)           | 879.1    | c=2  | 0.66x          | 2.09   | 3.88   | 7.45e-09    |
| `clean_mlp`                  | naive FastAPI + torch, `async def` | 2602.0   | c=32 | 1.94x          | 11.89  | 16.21  | 0.00e+00    |
| `dynamic_batch_cnn`          | naive FastAPI + eager torch        | 929.3    | c=4  | 1.00x          | 4.34   | 7.60   | 0.00e+00    |
| `dynamic_batch_cnn`          | naive FastAPI + ONNX Runtime       | 1061.0   | c=4  | 1.14x          | 3.61   | 8.20   | 2.98e-08    |
| `dynamic_batch_cnn`          | downshift serve (auto)             | 887.6    | c=8  | 0.96x          | 9.93   | 14.82  | 3.17e-08    |
| `dynamic_batch_cnn`          | downshift serve (base64)           | 936.6    | c=8  | 1.01x          | 9.35   | 14.75  | 2.98e-08    |
| `dynamic_batch_cnn`          | naive FastAPI + torch, `async def` | 1039.9   | c=32 | 1.12x          | 35.41  | 45.03  | 0.00e+00    |
| `gnn_gcn`                    | naive FastAPI + eager torch        | 549.4    | c=1  | 1.00x          | 1.75   | 3.05   | 0.00e+00    |
| `gnn_gcn`                    | naive FastAPI + ONNX Runtime       | 1484.3   | c=16 | 2.70x          | 10.55  | 15.70  | 5.96e-08    |
| `gnn_gcn`                    | downshift serve (auto)             | 845.2    | c=2  | 1.54x          | 2.22   | 4.04   | 6.44e-08    |
| `gnn_gcn`                    | downshift serve (base64)           | 816.8    | c=2  | 1.49x          | 2.29   | 4.04   | 5.96e-08    |
| `gnn_gcn`                    | naive FastAPI + torch, `async def` | 689.4    | c=16 | 1.25x          | 22.75  | 27.59  | 0.00e+00    |
| `tiny_bert`                  | naive FastAPI + eager torch        | 444.4    | c=2  | 1.00x          | 4.20   | 14.03  | 0.00e+00    |
| `tiny_bert`                  | naive FastAPI + ONNX Runtime       | 1502.5   | c=16 | 3.38x          | 10.38  | 15.65  | 4.77e-07    |
| `tiny_bert`                  | downshift serve (auto)             | 807.2    | c=8  | 1.82x          | 9.81   | 13.89  | 5.08e-07    |
| `tiny_bert`                  | downshift serve (base64)           | 773.8    | c=16 | 1.74x          | 20.50  | 25.45  | 4.77e-07    |
| `tiny_bert`                  | naive FastAPI + torch, `async def` | 539.1    | c=1  | 1.21x          | 1.78   | 3.17   | 0.00e+00    |
| `scatter_include_self_false` | naive FastAPI + eager torch        | 1267.1   | c=2  | 1.00x          | 1.44   | 2.92   | 0.00e+00    |
| `scatter_include_self_false` | naive FastAPI + ONNX Runtime       | 1571.9   | c=4  | 1.24x          | 2.33   | 4.57   | 7.94e-01    |
| `scatter_include_self_false` | downshift serve (auto)             | 921.3    | c=2  | 0.73x          | 2.07   | 3.56   | 2.77e-08    |
| `scatter_include_self_false` | downshift serve (base64)           | 915.1    | c=4  | 0.72x          | 4.18   | 7.24   | 0.00e+00    |
| `scatter_include_self_false` | naive FastAPI + torch, `async def` | 2041.4   | c=32 | 1.61x          | 15.23  | 20.41  | 0.00e+00    |
| `mlp_large`                  | naive FastAPI + eager torch        | 768.6    | c=16 | 1.00x          | 22.40  | 35.75  | 0.00e+00    |
| `mlp_large`                  | naive FastAPI + ONNX Runtime       | 877.6    | c=16 | 1.14x          | 20.18  | 29.04  | 2.53e-07    |
| `mlp_large`                  | downshift serve (auto)             | 687.9    | c=32 | 0.89x          | 53.24  | 65.80  | 2.58e-07    |
| `mlp_large`                  | downshift serve (base64)           | 596.8    | c=8  | 0.78x          | 13.24  | 19.30  | 2.53e-07    |
| `mlp_large`                  | naive FastAPI + torch, `async def` | 586.8    | c=16 | 0.76x          | 30.07  | 42.21  | 0.00e+00    |
| `cnn_large`                  | naive FastAPI + eager torch        | 358.6    | c=4  | 1.00x          | 11.95  | 18.41  | 0.00e+00    |
| `cnn_large`                  | naive FastAPI + ONNX Runtime       | 430.5    | c=4  | 1.20x          | 9.69   | 28.09  | 5.96e-08    |
| `cnn_large`                  | downshift serve (auto)             | 662.4    | c=8  | 1.85x          | 13.08  | 20.07  | 5.88e-08    |
| `cnn_large`                  | downshift serve (base64)           | 907.8    | c=16 | 2.53x          | 19.36  | 29.66  | 5.96e-08    |
| `cnn_large`                  | naive FastAPI + torch, `async def` | 308.6    | c=4  | 0.86x          | 14.95  | 18.51  | 0.00e+00    |
| `bert_small`                 | naive FastAPI + eager torch        | 99.0     | c=4  | 1.00x          | 39.49  | 61.62  | 0.00e+00    |
| `bert_small`                 | naive FastAPI + ONNX Runtime       | 151.5    | c=16 | 1.53x          | 103.51 | 151.84 | 1.91e-06    |
| `bert_small`                 | downshift serve (auto)             | 102.6    | c=4  | 1.04x          | 38.28  | 59.94  | 1.87e-06    |
| `bert_small`                 | downshift serve (base64)           | 121.5    | c=8  | 1.23x          | 65.12  | 88.95  | 1.91e-06    |
| `bert_small`                 | naive FastAPI + torch, `async def` | 72.0     | c=4  | 0.73x          | 54.97  | 68.69  | 0.00e+00    |

## Latency at low and high concurrency (batch 1)

| model                        | server                             | p50 c=1 | p99 c=1 | p50 c=64 | p99 c=64 |
|------------------------------|------------------------------------|---------|---------|----------|----------|
| `clean_mlp`                  | naive FastAPI + eager torch        | 0.81    | 1.98    | 49.25    | 152.50   |
| `clean_mlp`                  | naive FastAPI + ONNX Runtime       | 0.83    | 1.97    | 41.48    | 239.39   |
| `clean_mlp`                  | downshift serve (auto)             | 1.30    | 2.97    | 68.26    | 324.42   |
| `clean_mlp`                  | downshift serve (base64)           | 1.18    | 2.53    | 75.41    | 253.22   |
| `clean_mlp`                  | naive FastAPI + torch, `async def` | 0.48    | 1.08    | 25.37    | 40.47    |
| `dynamic_batch_cnn`          | naive FastAPI + eager torch        | 1.45    | 2.78    | 99.06    | 211.43   |
| `dynamic_batch_cnn`          | naive FastAPI + ONNX Runtime       | 1.33    | 2.72    | 83.03    | 385.68   |
| `dynamic_batch_cnn`          | downshift serve (auto)             | 1.55    | 3.08    | 79.83    | 311.25   |
| `dynamic_batch_cnn`          | downshift serve (base64)           | 1.73    | 3.29    | 69.92    | 338.95   |
| `dynamic_batch_cnn`          | naive FastAPI + torch, `async def` | 1.15    | 2.26    | 73.03    | 117.19   |
| `gnn_gcn`                    | naive FastAPI + eager torch        | 1.75    | 3.05    | 174.96   | 202.53   |
| `gnn_gcn`                    | naive FastAPI + ONNX Runtime       | 0.92    | 2.13    | 44.47    | 261.56   |
| `gnn_gcn`                    | downshift serve (auto)             | 1.34    | 2.78    | 84.11    | 344.02   |
| `gnn_gcn`                    | downshift serve (base64)           | 1.33    | 2.80    | 87.80    | 312.76   |
| `gnn_gcn`                    | naive FastAPI + torch, `async def` | 1.41    | 2.58    | 92.53    | 105.09   |
| `tiny_bert`                  | naive FastAPI + eager torch        | 2.44    | 4.73    | 192.96   | 220.73   |
| `tiny_bert`                  | naive FastAPI + ONNX Runtime       | 1.09    | 2.23    | 46.53    | 277.03   |
| `tiny_bert`                  | downshift serve (auto)             | 1.66    | 2.96    | 82.15    | 345.58   |
| `tiny_bert`                  | downshift serve (base64)           | 1.69    | 2.94    | 85.51    | 312.20   |
| `tiny_bert`                  | naive FastAPI + torch, `async def` | 1.78    | 3.17    | 120.22   | 127.44   |
| `scatter_include_self_false` | naive FastAPI + eager torch        | 0.88    | 1.96    | 56.90    | 170.94   |
| `scatter_include_self_false` | naive FastAPI + ONNX Runtime       | 0.81    | 1.99    | 41.69    | 228.71   |
| `scatter_include_self_false` | downshift serve (auto)             | 1.17    | 2.52    | 72.64    | 237.82   |
| `scatter_include_self_false` | downshift serve (base64)           | 1.15    | 2.67    | 72.63    | 83.20    |
| `scatter_include_self_false` | naive FastAPI + torch, `async def` | 0.57    | 1.28    | 31.51    | 43.76    |
| `mlp_large`                  | naive FastAPI + eager torch        | 2.52    | 5.88    | 100.63   | 195.23   |
| `mlp_large`                  | naive FastAPI + ONNX Runtime       | 1.85    | 3.75    | 87.62    | 104.08   |
| `mlp_large`                  | downshift serve (auto)             | 2.23    | 4.35    | 112.35   | 126.80   |
| `mlp_large`                  | downshift serve (base64)           | 2.11    | 4.22    | 109.21   | 350.30   |
| `mlp_large`                  | naive FastAPI + torch, `async def` | 2.04    | 4.82    | 126.92   | 152.66   |
| `cnn_large`                  | naive FastAPI + eager torch        | 3.80    | 5.30    | 235.33   | 376.06   |
| `cnn_large`                  | naive FastAPI + ONNX Runtime       | 2.91    | 4.24    | 198.21   | 395.96   |
| `cnn_large`                  | downshift serve (auto)             | 2.25    | 3.67    | 122.81   | 323.98   |
| `cnn_large`                  | downshift serve (base64)           | 1.74    | 3.08    | 86.09    | 289.93   |
| `cnn_large`                  | naive FastAPI + torch, `async def` | 3.50    | 5.09    | 212.84   | 256.44   |
| `bert_small`                 | naive FastAPI + eager torch        | 31.71   | 38.80   | 744.83   | 873.57   |
| `bert_small`                 | naive FastAPI + ONNX Runtime       | 31.71   | 45.89   | 433.36   | 507.47   |
| `bert_small`                 | downshift serve (auto)             | 11.25   | 25.09   | 613.28   | 661.20   |
| `bert_small`                 | downshift serve (base64)           | 9.51    | 11.04   | 519.83   | 561.67   |
| `bert_small`                 | naive FastAPI + torch, `async def` | 31.76   | 36.78   | 631.90   | 744.98   |

## The serving tax

`compute` is the model on arrays already in memory; `json` is what parsing the request and serializing the response adds on top, measured in the same process, so that subtraction is sound; `binary` is the same addition for base64 tensor bodies, which is the codec the `downshift serve (base64)` row is actually paying. `p50 over HTTP` is the single-client latency of the real server. The gap between the two is **not** subtracted here: they come from different processes with different allocator and thread-pool state, and for the large-payload rows the difference is smaller than that discrepancy. Read the last column instead — the share of end-to-end latency that is actually the model.

| model                        | batch | server                       | compute ms | json ms | binary ms | p50 over HTTP ms | compute share |
|------------------------------|-------|------------------------------|------------|---------|-----------|------------------|---------------|
| `clean_mlp`                  | 1     | naive FastAPI + eager torch  | 0.045      | 0.043   | 0.045     | 0.809            | 6%            |
| `clean_mlp`                  | 1     | naive FastAPI + ONNX Runtime | 0.020      | 0.029   | 0.026     | 0.830            | 2%            |
| `clean_mlp`                  | 1     | downshift serve (auto)       | 0.020      | 0.029   | 0.026     | 1.302            | 2%            |
| `clean_mlp`                  | 1     | downshift serve (base64)     | 0.020      | 0.029   | 0.026     | 1.176            | 2%            |
| `clean_mlp`                  | 8     | naive FastAPI + eager torch  | 0.057      | 0.161   | 0.064     | 0.949            | 6%            |
| `clean_mlp`                  | 8     | naive FastAPI + ONNX Runtime | 0.024      | 0.138   | 0.029     | 0.834            | 3%            |
| `clean_mlp`                  | 8     | downshift serve (auto)       | 0.024      | 0.138   | 0.029     | 1.244            | 2%            |
| `clean_mlp`                  | 8     | downshift serve (base64)     | 0.024      | 0.138   | 0.029     | 1.261            | 2%            |
| `clean_mlp`                  | 32    | naive FastAPI + eager torch  | 0.056      | 0.559   | 0.087     | 1.210            | 5%            |
| `clean_mlp`                  | 32    | naive FastAPI + ONNX Runtime | 0.024      | 0.533   | 0.045     | 1.132            | 2%            |
| `clean_mlp`                  | 32    | downshift serve (auto)       | 0.024      | 0.533   | 0.045     | 1.360            | 2%            |
| `clean_mlp`                  | 32    | downshift serve (base64)     | 0.024      | 0.533   | 0.045     | 1.280            | 2%            |
| `dynamic_batch_cnn`          | 1     | naive FastAPI + eager torch  | 0.100      | 0.650   | 0.127     | 1.450            | 7%            |
| `dynamic_batch_cnn`          | 1     | naive FastAPI + ONNX Runtime | 0.026      | 0.592   | 0.065     | 1.334            | 2%            |
| `dynamic_batch_cnn`          | 1     | downshift serve (auto)       | 0.026      | 0.592   | 0.065     | 1.546            | 2%            |
| `dynamic_batch_cnn`          | 1     | downshift serve (base64)     | 0.026      | 0.592   | 0.065     | 1.731            | 2%            |
| `dynamic_batch_cnn`          | 8     | naive FastAPI + eager torch  | 0.277      | 4.616   | 0.335     | 4.854            | 6%            |
| `dynamic_batch_cnn`          | 8     | naive FastAPI + ONNX Runtime | 0.050      | 4.629   | 0.279     | 4.494            | 1%            |
| `dynamic_batch_cnn`          | 8     | downshift serve (auto)       | 0.050      | 4.629   | 0.279     | 2.762            | 2%            |
| `dynamic_batch_cnn`          | 8     | downshift serve (base64)     | 0.050      | 4.629   | 0.279     | 1.526            | 3%            |
| `dynamic_batch_cnn`          | 32    | naive FastAPI + eager torch  | 0.340      | 18.869  | 1.095     | 22.924           | 1%            |
| `dynamic_batch_cnn`          | 32    | naive FastAPI + ONNX Runtime | 0.075      | 18.609  | 1.028     | 21.057           | 0%            |
| `dynamic_batch_cnn`          | 32    | downshift serve (auto)       | 0.075      | 18.609  | 1.028     | 7.529            | 1%            |
| `dynamic_batch_cnn`          | 32    | downshift serve (base64)     | 0.075      | 18.609  | 1.028     | 1.907            | 4%            |
| `gnn_gcn`                    | 1     | naive FastAPI + eager torch  | 0.777      | 0.162   | 0.121     | 1.749            | 44%           |
| `gnn_gcn`                    | 1     | naive FastAPI + ONNX Runtime | 0.107      | 0.099   | 0.052     | 0.919            | 12%           |
| `gnn_gcn`                    | 1     | downshift serve (auto)       | 0.107      | 0.099   | 0.052     | 1.337            | 8%            |
| `gnn_gcn`                    | 1     | downshift serve (base64)     | 0.107      | 0.099   | 0.052     | 1.331            | 8%            |
| `gnn_gcn`                    | 8     | naive FastAPI + eager torch  | 0.832      | 0.656   | 0.164     | 2.205            | 38%           |
| `gnn_gcn`                    | 8     | naive FastAPI + ONNX Runtime | 0.169      | 0.562   | 0.024     | 1.166            | 14%           |
| `gnn_gcn`                    | 8     | downshift serve (auto)       | 0.169      | 0.562   | 0.024     | 1.463            | 12%           |
| `gnn_gcn`                    | 8     | downshift serve (base64)     | 0.169      | 0.562   | 0.024     | 1.407            | 12%           |
| `gnn_gcn`                    | 32    | naive FastAPI + eager torch  | 0.942      | 2.662   | 0.258     | 3.320            | 28%           |
| `gnn_gcn`                    | 32    | naive FastAPI + ONNX Runtime | 0.362      | 1.920   | 0.057     | 2.200            | 16%           |
| `gnn_gcn`                    | 32    | downshift serve (auto)       | 0.362      | 1.920   | 0.057     | 1.879            | 19%           |
| `gnn_gcn`                    | 32    | downshift serve (base64)     | 0.362      | 1.920   | 0.057     | 1.705            | 21%           |
| `tiny_bert`                  | 1     | naive FastAPI + eager torch  | 1.327      | 0.220   | 0.101     | 2.441            | 54%           |
| `tiny_bert`                  | 1     | naive FastAPI + ONNX Runtime | 0.261      | 0.170   | 0.067     | 1.089            | 24%           |
| `tiny_bert`                  | 1     | downshift serve (auto)       | 0.261      | 0.170   | 0.067     | 1.655            | 16%           |
| `tiny_bert`                  | 1     | downshift serve (base64)     | 0.261      | 0.170   | 0.067     | 1.692            | 15%           |
| `tiny_bert`                  | 8     | naive FastAPI + eager torch  | 1.419      | 1.226   | 0.189     | 2.743            | 52%           |
| `tiny_bert`                  | 8     | naive FastAPI + ONNX Runtime | 0.478      | 1.155   | 0.120     | 1.491            | 32%           |
| `tiny_bert`                  | 8     | downshift serve (auto)       | 0.478      | 1.155   | 0.120     | 2.025            | 24%           |
| `tiny_bert`                  | 8     | downshift serve (base64)     | 0.478      | 1.155   | 0.120     | 1.979            | 24%           |
| `tiny_bert`                  | 32    | naive FastAPI + eager torch  | 1.598      | 4.759   | 0.377     | 3.499            | 46%           |
| `tiny_bert`                  | 32    | naive FastAPI + ONNX Runtime | 1.216      | 4.419   | 0.240     | 2.860            | 43%           |
| `tiny_bert`                  | 32    | downshift serve (auto)       | 1.216      | 4.419   | 0.240     | 3.180            | 38%           |
| `tiny_bert`                  | 32    | downshift serve (base64)     | 1.216      | 4.419   | 0.240     | 2.858            | 43%           |
| `scatter_include_self_false` | 1     | naive FastAPI + eager torch  | 0.060      | 0.097   | 0.078     | 0.884            | 7%            |
| `scatter_include_self_false` | 1     | naive FastAPI + ONNX Runtime | 0.031      | 0.085   | 0.038     | 0.812            | 4%            |
| `scatter_include_self_false` | 1     | downshift serve (auto)       | 0.060      | 0.097   | 0.078     | 1.172            | 5%            |
| `scatter_include_self_false` | 1     | downshift serve (base64)     | 0.060      | 0.097   | 0.078     | 1.152            | 5%            |
| `scatter_include_self_false` | 8     | naive FastAPI + eager torch  | 0.094      | 0.400   | 0.110     | 1.253            | 7%            |
| `scatter_include_self_false` | 8     | naive FastAPI + ONNX Runtime | 0.043      | 0.352   | 0.052     | 1.010            | 4%            |
| `scatter_include_self_false` | 8     | downshift serve (auto)       | 0.094      | 0.400   | 0.110     | 1.336            | 7%            |
| `scatter_include_self_false` | 8     | downshift serve (base64)     | 0.094      | 0.400   | 0.110     | 1.286            | 7%            |
| `scatter_include_self_false` | 32    | naive FastAPI + eager torch  | 0.149      | 1.276   | 0.135     | 2.008            | 7%            |
| `scatter_include_self_false` | 32    | naive FastAPI + ONNX Runtime | 0.071      | 1.105   | 0.073     | 1.620            | 4%            |
| `scatter_include_self_false` | 32    | downshift serve (auto)       | 0.149      | 1.276   | 0.135     | 1.659            | 9%            |
| `scatter_include_self_false` | 32    | downshift serve (base64)     | 0.149      | 1.276   | 0.135     | 1.317            | 11%           |
| `mlp_large`                  | 1     | naive FastAPI + eager torch  | 1.336      | 0.559   | -0.099    | 2.525            | 53%           |
| `mlp_large`                  | 1     | naive FastAPI + ONNX Runtime | 0.276      | 1.091   | 0.093     | 1.853            | 15%           |
| `mlp_large`                  | 1     | downshift serve (auto)       | 0.276      | 1.091   | 0.093     | 2.230            | 12%           |
| `mlp_large`                  | 1     | downshift serve (base64)     | 0.276      | 1.091   | 0.093     | 2.107            | 13%           |
| `mlp_large`                  | 8     | naive FastAPI + eager torch  | 2.088      | 7.322   | 0.595     | 7.094            | 29%           |
| `mlp_large`                  | 8     | naive FastAPI + ONNX Runtime | 0.491      | 7.280   | 0.372     | 4.560            | 11%           |
| `mlp_large`                  | 8     | downshift serve (auto)       | 0.491      | 7.280   | 0.372     | 3.256            | 15%           |
| `mlp_large`                  | 8     | downshift serve (base64)     | 0.491      | 7.280   | 0.372     | 2.312            | 21%           |
| `mlp_large`                  | 32    | naive FastAPI + eager torch  | 3.248      | 27.634  | 1.446     | 19.220           | 17%           |
| `mlp_large`                  | 32    | naive FastAPI + ONNX Runtime | 2.063      | 28.078  | 1.350     | 17.188           | 12%           |
| `mlp_large`                  | 32    | downshift serve (auto)       | 2.063      | 28.078  | 1.350     | 8.370            | 25%           |
| `mlp_large`                  | 32    | downshift serve (base64)     | 2.063      | 28.078  | 1.350     | 4.343            | 48%           |
| `cnn_large`                  | 1     | naive FastAPI + eager torch  | 1.105      | 2.323   | 0.205     | 3.797            | 29%           |
| `cnn_large`                  | 1     | naive FastAPI + ONNX Runtime | 0.326      | 2.293   | 0.176     | 2.910            | 11%           |
| `cnn_large`                  | 1     | downshift serve (auto)       | 0.326      | 2.293   | 0.176     | 2.253            | 14%           |
| `cnn_large`                  | 1     | downshift serve (base64)     | 0.326      | 2.293   | 0.176     | 1.741            | 19%           |
| `cnn_large`                  | 8     | naive FastAPI + eager torch  | 4.620      | 18.407  | 1.111     | 25.336           | 18%           |
| `cnn_large`                  | 8     | naive FastAPI + ONNX Runtime | 2.183      | 18.005  | 1.020     | 22.831           | 10%           |
| `cnn_large`                  | 8     | downshift serve (auto)       | 2.183      | 18.005  | 1.020     | 9.261            | 24%           |
| `cnn_large`                  | 8     | downshift serve (base64)     | 2.183      | 18.005  | 1.020     | 4.335            | 50%           |
| `cnn_large`                  | 32    | naive FastAPI + eager torch  | 17.626     | 67.763  | 3.574     | 133.010          | 13%           |
| `cnn_large`                  | 32    | naive FastAPI + ONNX Runtime | 8.538      | 72.179  | 3.885     | 127.231          | 7%            |
| `cnn_large`                  | 32    | downshift serve (auto)       | 8.538      | 72.179  | 3.885     | 80.744           | 11%           |
| `cnn_large`                  | 32    | downshift serve (base64)     | 8.538      | 72.179  | 3.885     | 20.093           | 42%           |
| `bert_small`                 | 1     | naive FastAPI + eager torch  | 8.132      | 33.089  | 1.381     | 31.715           | 26%           |
| `bert_small`                 | 1     | naive FastAPI + ONNX Runtime | 7.165      | 32.368  | 1.359     | 31.706           | 23%           |
| `bert_small`                 | 1     | downshift serve (auto)       | 7.165      | 32.368  | 1.359     | 11.246           | 64%           |
| `bert_small`                 | 1     | downshift serve (base64)     | 7.165      | 32.368  | 1.359     | 9.508            | 75%           |
| `bert_small`                 | 8     | naive FastAPI + eager torch  | 37.124     | 247.104 | 12.692    | 206.543          | 18%           |
| `bert_small`                 | 8     | naive FastAPI + ONNX Runtime | 48.920     | 259.611 | 12.303    | 260.055          | 19%           |
| `bert_small`                 | 8     | downshift serve (auto)       | 48.920     | 259.611 | 12.303    | 142.616          | 34%           |
| `bert_small`                 | 8     | downshift serve (base64)     | 48.920     | 259.611 | 12.303    | 79.596           | 61%           |
| `bert_small`                 | 32    | naive FastAPI + eager torch  | 148.069    | 866.875 | 49.168    | 968.826          | 15%           |
| `bert_small`                 | 32    | naive FastAPI + ONNX Runtime | 193.011    | 860.378 | 43.820    | 708.528          | 27%           |
| `bert_small`                 | 32    | downshift serve (auto)       | 193.011    | 860.378 | 43.820    | 584.938          | 33%           |
| `bert_small`                 | 32    | downshift serve (base64)     | 193.011    | 860.378 | 43.820    | 365.986          | 53%           |

## What downshift's own serving layer costs

downshift against the naive server running **the same backend**, so the only difference is the HTTP layer: pydantic request validation, a `response_model`, and a response that also carries per-output shapes and dtypes. Below 1.00x downshift is slower. This is the price of the product's ergonomics, and it is not free.

| model                        | batch | same-backend peer            | peer peak rps | downshift peak rps | ratio |
|------------------------------|-------|------------------------------|---------------|--------------------|-------|
| `clean_mlp`                  | 1     | naive FastAPI + ONNX Runtime | 1615.3        | 795.0              | 0.49x |
| `clean_mlp`                  | 8     | naive FastAPI + ONNX Runtime | 1387.5        | 783.3              | 0.56x |
| `clean_mlp`                  | 32    | naive FastAPI + ONNX Runtime | 1225.1        | 940.1              | 0.77x |
| `dynamic_batch_cnn`          | 1     | naive FastAPI + ONNX Runtime | 1061.0        | 887.6              | 0.84x |
| `dynamic_batch_cnn`          | 8     | naive FastAPI + ONNX Runtime | 204.8         | 381.6              | 1.86x |
| `dynamic_batch_cnn`          | 32    | naive FastAPI + ONNX Runtime | 51.1          | 120.2              | 2.35x |
| `gnn_gcn`                    | 1     | naive FastAPI + ONNX Runtime | 1484.3        | 845.2              | 0.57x |
| `gnn_gcn`                    | 8     | naive FastAPI + ONNX Runtime | 1217.8        | 862.2              | 0.71x |
| `gnn_gcn`                    | 32    | naive FastAPI + ONNX Runtime | 563.4         | 663.0              | 1.18x |
| `tiny_bert`                  | 1     | naive FastAPI + ONNX Runtime | 1502.5        | 807.2              | 0.54x |
| `tiny_bert`                  | 8     | naive FastAPI + ONNX Runtime | 1104.4        | 658.5              | 0.60x |
| `tiny_bert`                  | 32    | naive FastAPI + ONNX Runtime | 574.6         | 407.8              | 0.71x |
| `scatter_include_self_false` | 1     | naive FastAPI + eager torch  | 1267.1        | 921.3              | 0.73x |
| `scatter_include_self_false` | 8     | naive FastAPI + eager torch  | 937.7         | 1026.5             | 1.09x |
| `scatter_include_self_false` | 32    | naive FastAPI + eager torch  | 538.1         | 744.3              | 1.38x |
| `mlp_large`                  | 1     | naive FastAPI + ONNX Runtime | 877.6         | 687.9              | 0.78x |
| `mlp_large`                  | 8     | naive FastAPI + ONNX Runtime | 251.0         | 435.2              | 1.73x |
| `mlp_large`                  | 32    | naive FastAPI + ONNX Runtime | 73.6          | 167.0              | 2.27x |
| `cnn_large`                  | 1     | naive FastAPI + ONNX Runtime | 430.5         | 662.4              | 1.54x |
| `cnn_large`                  | 8     | naive FastAPI + ONNX Runtime | 55.8          | 136.2              | 2.44x |
| `cnn_large`                  | 32    | naive FastAPI + ONNX Runtime | 12.7          | 36.5               | 2.87x |
| `bert_small`                 | 1     | naive FastAPI + ONNX Runtime | 151.5         | 102.6              | 0.68x |
| `bert_small`                 | 8     | naive FastAPI + ONNX Runtime | 18.4          | 16.0               | 0.87x |
| `bert_small`                 | 32    | naive FastAPI + ONNX Runtime | 4.5           | 3.1                | 0.69x |

## Throughput against correctness

The only models where the two axes actually trade off: the exporter produced a graph, and the graph is wrong. Peak throughput at batch 1, against the served output's error.

| model                        | server                             | peak rps | max abs err | verdict           |
|------------------------------|------------------------------------|----------|-------------|-------------------|
| `scatter_include_self_false` | naive FastAPI + eager torch        | 1267.1   | 0.00e+00    | correct           |
| `scatter_include_self_false` | naive FastAPI + ONNX Runtime       | 1571.9   | 7.94e-01    | **WRONG ANSWERS** |
| `scatter_include_self_false` | downshift serve (auto)             | 921.3    | 2.77e-08    | correct           |
| `scatter_include_self_false` | downshift serve (base64)           | 915.1    | 0.00e+00    | correct           |
| `scatter_include_self_false` | naive FastAPI + torch, `async def` | 2041.4   | 0.00e+00    | correct           |

## Batch size sweep

Throughput in requests/s, and the same number as items/s, at the concurrency where each variant peaked. Larger batches amortize the per-request server cost over more work.

| model                        | batch | server                       | rps    | items/s | at   | p50 ms  |
|------------------------------|-------|------------------------------|--------|---------|------|---------|
| `clean_mlp`                  | 1     | naive FastAPI + eager torch  | 1339.0 | 1339.0  | c=4  | 2.75    |
| `clean_mlp`                  | 1     | naive FastAPI + ONNX Runtime | 1615.3 | 1615.3  | c=4  | 2.26    |
| `clean_mlp`                  | 1     | downshift serve (auto)       | 795.0  | 795.0   | c=8  | 9.82    |
| `clean_mlp`                  | 1     | downshift serve (base64)     | 879.1  | 879.1   | c=2  | 2.09    |
| `clean_mlp`                  | 8     | naive FastAPI + eager torch  | 1059.3 | 8474.2  | c=8  | 7.50    |
| `clean_mlp`                  | 8     | naive FastAPI + ONNX Runtime | 1387.5 | 11100.3 | c=8  | 5.57    |
| `clean_mlp`                  | 8     | downshift serve (auto)       | 783.3  | 6266.5  | c=8  | 9.82    |
| `clean_mlp`                  | 8     | downshift serve (base64)     | 789.7  | 6317.4  | c=32 | 36.67   |
| `clean_mlp`                  | 32    | naive FastAPI + eager torch  | 924.9  | 29596.5 | c=8  | 9.18    |
| `clean_mlp`                  | 32    | naive FastAPI + ONNX Runtime | 1225.1 | 39204.5 | c=8  | 7.02    |
| `clean_mlp`                  | 32    | downshift serve (auto)       | 940.1  | 30084.8 | c=8  | 9.41    |
| `clean_mlp`                  | 32    | downshift serve (base64)     | 759.9  | 24318.1 | c=32 | 37.87   |
| `dynamic_batch_cnn`          | 1     | naive FastAPI + eager torch  | 929.3  | 929.3   | c=4  | 4.34    |
| `dynamic_batch_cnn`          | 1     | naive FastAPI + ONNX Runtime | 1061.0 | 1061.0  | c=4  | 3.61    |
| `dynamic_batch_cnn`          | 1     | downshift serve (auto)       | 887.6  | 887.6   | c=8  | 9.93    |
| `dynamic_batch_cnn`          | 1     | downshift serve (base64)     | 936.6  | 936.6   | c=8  | 9.35    |
| `dynamic_batch_cnn`          | 8     | naive FastAPI + eager torch  | 197.3  | 1578.6  | c=1  | 4.85    |
| `dynamic_batch_cnn`          | 8     | naive FastAPI + ONNX Runtime | 204.8  | 1638.8  | c=1  | 4.49    |
| `dynamic_batch_cnn`          | 8     | downshift serve (auto)       | 381.6  | 3052.8  | c=8  | 21.14   |
| `dynamic_batch_cnn`          | 8     | downshift serve (base64)     | 887.4  | 7098.9  | c=8  | 9.66    |
| `dynamic_batch_cnn`          | 32    | naive FastAPI + eager torch  | 47.1   | 1506.6  | c=8  | 182.15  |
| `dynamic_batch_cnn`          | 32    | naive FastAPI + ONNX Runtime | 51.1   | 1635.2  | c=8  | 164.73  |
| `dynamic_batch_cnn`          | 32    | downshift serve (auto)       | 120.2  | 3845.8  | c=8  | 63.69   |
| `dynamic_batch_cnn`          | 32    | downshift serve (base64)     | 670.7  | 21463.0 | c=8  | 13.36   |
| `gnn_gcn`                    | 1     | naive FastAPI + eager torch  | 549.4  | 549.4   | c=1  | 1.75    |
| `gnn_gcn`                    | 1     | naive FastAPI + ONNX Runtime | 1484.3 | 1484.3  | c=16 | 10.55   |
| `gnn_gcn`                    | 1     | downshift serve (auto)       | 845.2  | 845.2   | c=2  | 2.22    |
| `gnn_gcn`                    | 1     | downshift serve (base64)     | 816.8  | 816.8   | c=2  | 2.29    |
| `gnn_gcn`                    | 8     | naive FastAPI + eager torch  | 453.8  | 3630.3  | c=8  | 20.21   |
| `gnn_gcn`                    | 8     | naive FastAPI + ONNX Runtime | 1217.8 | 9742.6  | c=8  | 6.93    |
| `gnn_gcn`                    | 8     | downshift serve (auto)       | 862.2  | 6897.6  | c=8  | 10.16   |
| `gnn_gcn`                    | 8     | downshift serve (base64)     | 737.2  | 5897.9  | c=8  | 10.64   |
| `gnn_gcn`                    | 32    | naive FastAPI + eager torch  | 314.1  | 10051.8 | c=8  | 28.70   |
| `gnn_gcn`                    | 32    | naive FastAPI + ONNX Runtime | 563.4  | 18027.2 | c=8  | 14.30   |
| `gnn_gcn`                    | 32    | downshift serve (auto)       | 663.0  | 21215.7 | c=8  | 13.07   |
| `gnn_gcn`                    | 32    | downshift serve (base64)     | 869.1  | 27809.9 | c=8  | 9.70    |
| `tiny_bert`                  | 1     | naive FastAPI + eager torch  | 444.4  | 444.4   | c=2  | 4.20    |
| `tiny_bert`                  | 1     | naive FastAPI + ONNX Runtime | 1502.5 | 1502.5  | c=16 | 10.38   |
| `tiny_bert`                  | 1     | downshift serve (auto)       | 807.2  | 807.2   | c=8  | 9.81    |
| `tiny_bert`                  | 1     | downshift serve (base64)     | 773.8  | 773.8   | c=16 | 20.50   |
| `tiny_bert`                  | 8     | naive FastAPI + eager torch  | 343.8  | 2750.6  | c=1  | 2.74    |
| `tiny_bert`                  | 8     | naive FastAPI + ONNX Runtime | 1104.4 | 8835.2  | c=8  | 7.09    |
| `tiny_bert`                  | 8     | downshift serve (auto)       | 658.5  | 5268.0  | c=8  | 12.00   |
| `tiny_bert`                  | 8     | downshift serve (base64)     | 687.8  | 5502.2  | c=8  | 11.48   |
| `tiny_bert`                  | 32    | naive FastAPI + eager torch  | 265.4  | 8493.8  | c=1  | 3.50    |
| `tiny_bert`                  | 32    | naive FastAPI + ONNX Runtime | 574.6  | 18385.9 | c=8  | 12.38   |
| `tiny_bert`                  | 32    | downshift serve (auto)       | 407.8  | 13049.0 | c=8  | 19.44   |
| `tiny_bert`                  | 32    | downshift serve (base64)     | 561.1  | 17954.2 | c=8  | 15.47   |
| `scatter_include_self_false` | 1     | naive FastAPI + eager torch  | 1267.1 | 1267.1  | c=2  | 1.44    |
| `scatter_include_self_false` | 1     | naive FastAPI + ONNX Runtime | 1571.9 | 1571.9  | c=4  | 2.33    |
| `scatter_include_self_false` | 1     | downshift serve (auto)       | 921.3  | 921.3   | c=2  | 2.07    |
| `scatter_include_self_false` | 1     | downshift serve (base64)     | 915.1  | 915.1   | c=4  | 4.18    |
| `scatter_include_self_false` | 8     | naive FastAPI + eager torch  | 937.7  | 7501.7  | c=8  | 8.99    |
| `scatter_include_self_false` | 8     | naive FastAPI + ONNX Runtime | 1298.8 | 10390.2 | c=8  | 6.11    |
| `scatter_include_self_false` | 8     | downshift serve (auto)       | 1026.5 | 8212.0  | c=8  | 8.38    |
| `scatter_include_self_false` | 8     | downshift serve (base64)     | 900.0  | 7199.9  | c=8  | 8.70    |
| `scatter_include_self_false` | 32    | naive FastAPI + eager torch  | 538.1  | 17219.5 | c=8  | 16.13   |
| `scatter_include_self_false` | 32    | naive FastAPI + ONNX Runtime | 727.3  | 23273.0 | c=8  | 11.60   |
| `scatter_include_self_false` | 32    | downshift serve (auto)       | 744.3  | 23817.3 | c=8  | 10.88   |
| `scatter_include_self_false` | 32    | downshift serve (base64)     | 1069.5 | 34222.4 | c=8  | 8.14    |
| `mlp_large`                  | 1     | naive FastAPI + eager torch  | 768.6  | 768.6   | c=16 | 22.40   |
| `mlp_large`                  | 1     | naive FastAPI + ONNX Runtime | 877.6  | 877.6   | c=16 | 20.18   |
| `mlp_large`                  | 1     | downshift serve (auto)       | 687.9  | 687.9   | c=32 | 53.24   |
| `mlp_large`                  | 1     | downshift serve (base64)     | 596.8  | 596.8   | c=8  | 13.24   |
| `mlp_large`                  | 8     | naive FastAPI + eager torch  | 210.7  | 1685.8  | c=8  | 42.50   |
| `mlp_large`                  | 8     | naive FastAPI + ONNX Runtime | 251.0  | 2007.8  | c=8  | 35.44   |
| `mlp_large`                  | 8     | downshift serve (auto)       | 435.2  | 3481.9  | c=8  | 19.94   |
| `mlp_large`                  | 8     | downshift serve (base64)     | 701.0  | 5607.8  | c=8  | 12.22   |
| `mlp_large`                  | 32    | naive FastAPI + eager torch  | 66.0   | 2112.3  | c=8  | 132.00  |
| `mlp_large`                  | 32    | naive FastAPI + ONNX Runtime | 73.6   | 2355.8  | c=8  | 119.57  |
| `mlp_large`                  | 32    | downshift serve (auto)       | 167.0  | 5343.7  | c=8  | 54.33   |
| `mlp_large`                  | 32    | downshift serve (base64)     | 343.5  | 10991.7 | c=8  | 25.72   |
| `cnn_large`                  | 1     | naive FastAPI + eager torch  | 358.6  | 358.6   | c=4  | 11.95   |
| `cnn_large`                  | 1     | naive FastAPI + ONNX Runtime | 430.5  | 430.5   | c=4  | 9.69    |
| `cnn_large`                  | 1     | downshift serve (auto)       | 662.4  | 662.4   | c=8  | 13.08   |
| `cnn_large`                  | 1     | downshift serve (base64)     | 907.8  | 907.8   | c=16 | 19.36   |
| `cnn_large`                  | 8     | naive FastAPI + eager torch  | 51.3   | 410.7   | c=8  | 160.30  |
| `cnn_large`                  | 8     | naive FastAPI + ONNX Runtime | 55.8   | 446.7   | c=8  | 156.93  |
| `cnn_large`                  | 8     | downshift serve (auto)       | 136.2  | 1089.8  | c=8  | 60.55   |
| `cnn_large`                  | 8     | downshift serve (base64)     | 342.3  | 2738.2  | c=8  | 26.02   |
| `cnn_large`                  | 32    | naive FastAPI + eager torch  | 11.3   | 360.6   | c=8  | 693.34  |
| `cnn_large`                  | 32    | naive FastAPI + ONNX Runtime | 12.7   | 407.4   | c=8  | 595.09  |
| `cnn_large`                  | 32    | downshift serve (auto)       | 36.5   | 1168.0  | c=8  | 215.89  |
| `cnn_large`                  | 32    | downshift serve (base64)     | 110.1  | 3522.9  | c=8  | 80.96   |
| `bert_small`                 | 1     | naive FastAPI + eager torch  | 99.0   | 99.0    | c=4  | 39.49   |
| `bert_small`                 | 1     | naive FastAPI + ONNX Runtime | 151.5  | 151.5   | c=16 | 103.51  |
| `bert_small`                 | 1     | downshift serve (auto)       | 102.6  | 102.6   | c=4  | 38.28   |
| `bert_small`                 | 1     | downshift serve (base64)     | 121.5  | 121.5   | c=8  | 65.12   |
| `bert_small`                 | 8     | naive FastAPI + eager torch  | 17.5   | 139.7   | c=8  | 445.16  |
| `bert_small`                 | 8     | naive FastAPI + ONNX Runtime | 18.4   | 146.8   | c=8  | 423.05  |
| `bert_small`                 | 8     | downshift serve (auto)       | 16.0   | 127.7   | c=8  | 457.75  |
| `bert_small`                 | 8     | downshift serve (base64)     | 20.9   | 167.1   | c=8  | 362.48  |
| `bert_small`                 | 32    | naive FastAPI + eager torch  | 3.6    | 114.2   | c=8  | 1989.91 |
| `bert_small`                 | 32    | naive FastAPI + ONNX Runtime | 4.5    | 143.4   | c=8  | 1489.16 |
| `bert_small`                 | 32    | downshift serve (auto)       | 3.1    | 98.9    | c=8  | 1756.67 |
| `bert_small`                 | 32    | downshift serve (base64)     | 3.7    | 117.4   | c=8  | 1595.57 |

## Correctness of the served response

Max absolute difference between the served output and eager PyTorch on the same input. Throughput numbers above are only comparable between rows whose error is at float32 noise.

| model                        | server                       | max abs err vs eager torch |
|------------------------------|------------------------------|----------------------------|
| `clean_mlp`                  | naive FastAPI + eager torch  | 0.00e+00                   |
| `clean_mlp`                  | naive FastAPI + ONNX Runtime | 5.96e-08                   |
| `clean_mlp`                  | downshift serve (auto)       | 6.41e-08                   |
| `clean_mlp`                  | downshift serve (base64)     | 5.96e-08                   |
| `dynamic_batch_cnn`          | naive FastAPI + eager torch  | 0.00e+00                   |
| `dynamic_batch_cnn`          | naive FastAPI + ONNX Runtime | 2.98e-08                   |
| `dynamic_batch_cnn`          | downshift serve (auto)       | 3.38e-08                   |
| `dynamic_batch_cnn`          | downshift serve (base64)     | 2.98e-08                   |
| `gnn_gcn`                    | naive FastAPI + eager torch  | 0.00e+00                   |
| `gnn_gcn`                    | naive FastAPI + ONNX Runtime | 2.38e-07                   |
| `gnn_gcn`                    | downshift serve (auto)       | 2.41e-07                   |
| `gnn_gcn`                    | downshift serve (base64)     | 2.38e-07                   |
| `tiny_bert`                  | naive FastAPI + eager torch  | 0.00e+00                   |
| `tiny_bert`                  | naive FastAPI + ONNX Runtime | 4.77e-07                   |
| `tiny_bert`                  | downshift serve (auto)       | 5.76e-07                   |
| `tiny_bert`                  | downshift serve (base64)     | 4.77e-07                   |
| `scatter_include_self_false` | naive FastAPI + eager torch  | 0.00e+00                   |
| `scatter_include_self_false` | naive FastAPI + ONNX Runtime | 1.20e+00                   |
| `scatter_include_self_false` | downshift serve (auto)       | 2.77e-08                   |
| `scatter_include_self_false` | downshift serve (base64)     | 0.00e+00                   |
| `mlp_large`                  | naive FastAPI + eager torch  | 0.00e+00                   |
| `mlp_large`                  | naive FastAPI + ONNX Runtime | 2.53e-07                   |
| `mlp_large`                  | downshift serve (auto)       | 2.58e-07                   |
| `mlp_large`                  | downshift serve (base64)     | 2.53e-07                   |
| `cnn_large`                  | naive FastAPI + eager torch  | 0.00e+00                   |
| `cnn_large`                  | naive FastAPI + ONNX Runtime | 5.96e-08                   |
| `cnn_large`                  | downshift serve (auto)       | 5.88e-08                   |
| `cnn_large`                  | downshift serve (base64)     | 5.96e-08                   |
| `bert_small`                 | naive FastAPI + eager torch  | 0.00e+00                   |
| `bert_small`                 | naive FastAPI + ONNX Runtime | 2.62e-06                   |
| `bert_small`                 | downshift serve (auto)       | 2.66e-06                   |
| `bert_small`                 | downshift serve (base64)     | 2.62e-06                   |

## Workers

`--workers N` only exists on the real `downshift serve` CLI, not the in-process app the rest of this report drives (`bench.servers`), so this section launches the CLI itself against the import specs in `bench/factories.py` — built under the same seed as every other fixture here, so its correctness numbers are comparable to the rest of the report. `workers=1` is bind-first: the port opens immediately and `boot_s` is purely the export/verify/warmup gate behind `/ready`. `workers>1` exports once in the parent and hands the artifact to every worker, so `boot_s` there is that export plus each worker's own load, verify and warmup, running in parallel with each other but not with the export.

**Boot time**, i.e. how long `/ready` takes to answer 200.

| model        | workers | boot_s |
|--------------|---------|--------|
| `clean_mlp`  | 1       | 7.99   |
| `clean_mlp`  | 2       | 11.32  |
| `clean_mlp`  | 4       | 12.30  |
| `mlp_large`  | 1       | 9.62   |
| `mlp_large`  | 2       | 12.32  |
| `mlp_large`  | 4       | 12.80  |
| `bert_small` | 1       | 22.60  |
| `bert_small` | 2       | 23.62  |
| `bert_small` | 4       | 24.11  |

**Throughput at batch 1**, by concurrency. `max abs err` is the worst error seen across that row's concurrencies — a worker count that is fast but wrong should not read as a win.

| model        | workers | rps c=1 | p99 c=1 | rps c=8 | p99 c=8 | rps c=32 | p99 c=32 | max abs err |
|--------------|---------|---------|---------|---------|---------|----------|----------|-------------|
| `clean_mlp`  | 1       | 845.2   | 2.34    | 893.1   | 14.86   | 864.8    | 220.62   | 9.22e-09    |
| `clean_mlp`  | 2       | 837.5   | 2.17    | 1703.3  | 6.70    | 1771.6   | 26.52    | 9.22e-09    |
| `clean_mlp`  | 4       | 714.5   | 2.60    | 3077.2  | 4.90    | 3126.3   | 19.98    | 9.22e-09    |
| `mlp_large`  | 1       | 402.0   | 4.20    | 695.0   | 22.22   | 714.7    | 64.31    | 2.58e-07    |
| `mlp_large`  | 2       | 395.9   | 4.20    | 926.6   | 16.93   | 1026.1   | 56.42    | 2.58e-07    |
| `mlp_large`  | 4       | 378.2   | 4.80    | 1357.0  | 13.66   | 1957.6   | 40.98    | 2.58e-07    |
| `bert_small` | 1       | 88.9    | 12.99   | 102.6   | 104.75  | 97.7     | 340.29   | 1.87e-06    |
| `bert_small` | 2       | 88.2    | 15.31   | 149.9   | 78.88   | 142.8    | 336.07   | 1.87e-06    |
| `bert_small` | 4       | 66.6    | 17.46   | 230.7   | 75.41   | 224.2    | 210.83   | 1.87e-06    |

## Not measured

- TorchServe and BentoML. Deferred; they need their own packaging step and a separate run to be a fair comparison.
- GPU. This box is CPU-only, and `--device cuda` is untested in this release.
- `--workers > 1` for anything but the `downshift serve` CLI itself, which is measured in the Workers section when this run included the sweep. The rest of this matrix, including the naive baselines, still runs a single uvicorn worker per server.
