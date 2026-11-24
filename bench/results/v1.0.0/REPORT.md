# downshift benchmark: naive servers vs downshift 0.5.1 vs 1.0.0

`win32`, 16 logical CPUs, Python 3.14.7, torch 2.14.0+cpu (CPU), onnxruntime 1.30.0, run 2026-11-22T18:30:39. downshift 1.0.0 from the checkout; downshift 0.5.1 from the pip-installed release. Every downshift server is the real `downshift serve` CLI; all servers ran in one session on the same machine.

## Fixture and compute-heavy models

Closed-loop load, 3.0 s window after 1.0 s warm-up, load generator sharded over up to 6 processes on the same machine. Every server gets the same seeded weights and byte-identical request bodies, and every row's response is checked element-wise against eager PyTorch.

Load-generator ceiling (`GET /health` on a do-nothing server): c=1: 4453 rps, c=2: 8168 rps, c=4: 8304 rps, c=8: 8769 rps, c=16: 9437 rps, c=32: 9636 rps, c=64: 9902 rps. A result near these numbers is measuring the client.

### Verdicts

| model | verdict | backend chosen | max abs err |
|---|---|---|---|
| `clean_mlp` | CLEAN | onnxruntime | 3.7e-08 |
| `dynamic_batch_cnn` | CLEAN | onnxruntime | 3.0e-08 |
| `gnn_gcn` | CLEAN | onnxruntime | 2.4e-07 |
| `tiny_bert` | CLEAN | onnxruntime | 4.8e-07 |
| `scatter_include_self_false` | DEGRADED | torch | 1.6e+00 |
| `mlp_large` | CLEAN | onnxruntime | 2.1e-07 |
| `cnn_large` | CLEAN | onnxruntime | 5.6e-08 |
| `bert_small` | CLEAN | onnxruntime | 1.9e-06 |

### Peak throughput, batch 1

Best requests/s over the concurrency sweep, and p50 latency at concurrency 1. `async def` blocks the event loop and is shown as the cautionary row.

| server | `clean_mlp` | `dynamic_batch_cnn` | `gnn_gcn` | `tiny_bert` | `scatter_include_self_false` | `mlp_large` | `cnn_large` | `bert_small` |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1303 | 913 | 496 | 383 | 1421 | 932 | 405 | 97 |
| naive FastAPI + ONNX Runtime | 1706 | 1182 | 1572 | 1458 | 1692 | 980 | 463 | 148 |
| naive torch, `async def` | 2347 | 1135 | 676 | 486 | 1643 | 656 | 345 | 71 |
| downshift 0.5.1 | 1118 | 1302 | 1076 | 1082 | 1047 | 1074 | 933 | 260 |
| downshift 0.5.1 base64 | 1130 | 1421 | 1069 | 989 | 1028 | 805 | 1345 | 295 |
| downshift 0.5.1 safetensors | 1055 | 1801 | 1108 | 1059 | 1003 | 819 | 1421 | 326 |
| downshift 0.5.1 `--execution inline` | 1557 | 1552 | 1286 | 983 | 1359 | — | — | — |
| downshift 1.0.0 | 1164 | 1208 | 1054 | 1112 | 959 | 1013 | 873 | 259 |
| downshift 1.0.0 base64 | 1061 | 1440 | 1068 | 1061 | 946 | 780 | 1307 | 299 |
| downshift 1.0.0 safetensors | 1079 | 1431 | 1023 | 1064 | 940 | 806 | 1384 | 319 |
| downshift 1.0.0 `--execution inline` | 1619 | 1462 | 1190 | 950 | 1638 | — | — | — |
| downshift 1.0.0 result cache on | 1407 | 1503 | 1072 | 1056 | 1045 | 1629 | 1182 | 252 |
| downshift 1.0.0 `--metrics` | 988 | 1162 | 1078 | 1031 | 1174 | 988 | 864 | 252 |

p50 latency (ms) at concurrency 1:

| server | `clean_mlp` | `dynamic_batch_cnn` | `gnn_gcn` | `tiny_bert` | `scatter_include_self_false` | `mlp_large` | `cnn_large` | `bert_small` |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1.07 | 1.60 | 2.03 | 2.75 | 0.89 | 3.08 | 3.73 | 31.57 |
| naive FastAPI + ONNX Runtime | 0.83 | 1.38 | 0.95 | 1.24 | 0.80 | 1.94 | 2.89 | 31.81 |
| naive torch, `async def` | 0.57 | 1.20 | 1.51 | 1.97 | 0.67 | 2.43 | 3.48 | 31.71 |
| downshift 0.5.1 | 1.09 | 1.34 | 1.25 | 1.60 | 1.14 | 1.92 | 2.01 | 10.89 |
| downshift 0.5.1 base64 | 1.13 | 1.20 | 1.19 | 1.61 | 1.09 | 1.77 | 1.50 | 9.66 |
| downshift 0.5.1 safetensors | 1.07 | 1.41 | 1.25 | 1.82 | 1.23 | 1.77 | 1.44 | 8.98 |
| downshift 0.5.1 `--execution inline` | 0.83 | 0.89 | 0.95 | 1.08 | 0.88 | — | — | — |
| downshift 1.0.0 | 1.08 | 1.43 | 1.22 | 1.52 | 1.51 | 1.98 | 2.00 | 10.73 |
| downshift 1.0.0 base64 | 1.05 | 1.23 | 1.24 | 1.45 | 1.30 | 1.88 | 1.52 | 9.64 |
| downshift 1.0.0 safetensors | 1.11 | 1.22 | 1.26 | 1.47 | 1.32 | 1.75 | 1.47 | 8.92 |
| downshift 1.0.0 `--execution inline` | 0.77 | 0.98 | 0.96 | 1.14 | 0.94 | — | — | — |
| downshift 1.0.0 result cache on | 1.00 | 1.06 | 1.24 | 1.40 | 1.15 | 0.90 | 1.20 | 10.92 |
| downshift 1.0.0 `--metrics` | 1.10 | 1.41 | 1.23 | 1.54 | 1.03 | 2.04 | 2.01 | 10.74 |

### Concurrency sweep, batch 1 (requests/s)

`clean_mlp`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 864 | 1069 | 1279 | 1227 | 1303 | 1247 | 1046 |
| naive FastAPI + ONNX Runtime | 1148 | 1435 | 1659 | 1632 | 1702 | 1706 | 1569 |
| naive torch, `async def` | 1671 | 2251 | 2248 | 2290 | 2347 | 2095 | 2217 |
| downshift 0.5.1 | 845 | 1090 | 1021 | 1043 | 1061 | 1118 | 1067 |
| downshift 0.5.1 base64 | 825 | 1130 | 1027 | 1056 | 1049 | 1071 | 1058 |
| downshift 0.5.1 safetensors | 876 | 1049 | 995 | 1004 | 1048 | 1055 | 1049 |
| downshift 0.5.1 `--execution inline` | 1056 | 1406 | 1506 | 1471 | 1531 | 1484 | 1557 |
| downshift 1.0.0 | 883 | 1139 | 1079 | 1097 | 1149 | 1004 | 1164 |
| downshift 1.0.0 base64 | 922 | 1036 | 963 | 985 | 1033 | 1052 | 1061 |
| downshift 1.0.0 safetensors | 867 | 1073 | 1007 | 991 | 1079 | 1075 | 972 |
| downshift 1.0.0 `--execution inline` | 1218 | 1426 | 1546 | 1619 | 1538 | 1611 | 1572 |
| downshift 1.0.0 result cache on | 913 | 1260 | 1407 | 1314 | 1305 | 1288 | 1236 |
| downshift 1.0.0 `--metrics` | 813 | 982 | 965 | 902 | 985 | 988 | 963 |

`dynamic_batch_cnn`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 591 | 636 | 885 | 896 | 913 | 904 | 821 |
| naive FastAPI + ONNX Runtime | 703 | 844 | 1182 | 1077 | 1115 | 1017 | 953 |
| naive torch, `async def` | 801 | 909 | 1131 | 1126 | 1126 | 1135 | 1088 |
| downshift 0.5.1 | 723 | 953 | 1216 | 1216 | 1288 | 1264 | 1302 |
| downshift 0.5.1 base64 | 815 | 1125 | 1350 | 1399 | 1380 | 1401 | 1421 |
| downshift 0.5.1 safetensors | 669 | 982 | 1283 | 1339 | 1801 | 1470 | 1483 |
| downshift 0.5.1 `--execution inline` | 1072 | 1208 | 1463 | 1490 | 1495 | 1507 | 1552 |
| downshift 1.0.0 | 673 | 863 | 1084 | 1208 | 1183 | 1148 | 1169 |
| downshift 1.0.0 base64 | 789 | 1080 | 1323 | 1373 | 1417 | 1382 | 1440 |
| downshift 1.0.0 safetensors | 767 | 981 | 1290 | 1363 | 1394 | 1374 | 1431 |
| downshift 1.0.0 `--execution inline` | 979 | 1138 | 1404 | 1436 | 1432 | 1443 | 1462 |
| downshift 1.0.0 result cache on | 906 | 1141 | 1465 | 1503 | 1467 | 1438 | 1436 |
| downshift 1.0.0 `--metrics` | 689 | 916 | 1081 | 1160 | 1162 | 1136 | 1151 |

`gnn_gcn`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 457 | 489 | 496 | 369 | 466 | 425 | 420 |
| naive FastAPI + ONNX Runtime | 941 | 1325 | 1572 | 1510 | 1529 | 1454 | 1359 |
| naive torch, `async def` | 650 | 651 | 674 | 653 | 676 | 676 | 627 |
| downshift 0.5.1 | 746 | 894 | 944 | 1000 | 1021 | 1076 | 1055 |
| downshift 0.5.1 base64 | 811 | 1061 | 978 | 1017 | 1045 | 1059 | 1069 |
| downshift 0.5.1 safetensors | 768 | 1028 | 971 | 992 | 940 | 966 | 1108 |
| downshift 0.5.1 `--execution inline` | 991 | 1035 | 1184 | 1242 | 1217 | 1254 | 1286 |
| downshift 1.0.0 | 776 | 1035 | 990 | 987 | 985 | 1054 | 1011 |
| downshift 1.0.0 base64 | 773 | 1010 | 985 | 993 | 1016 | 1068 | 1052 |
| downshift 1.0.0 safetensors | 756 | 959 | 985 | 978 | 1016 | 1023 | 1007 |
| downshift 1.0.0 `--execution inline` | 989 | 1166 | 1174 | 1188 | 1190 | 1184 | 1180 |
| downshift 1.0.0 result cache on | 770 | 1037 | 1009 | 1002 | 1034 | 1072 | 1066 |
| downshift 1.0.0 `--metrics` | 773 | 986 | 940 | 996 | 1035 | 1078 | 1031 |

`tiny_bert`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 350 | 367 | 383 | 358 | 337 | 325 | 304 |
| naive FastAPI + ONNX Runtime | 758 | 1152 | 1435 | 1395 | 1458 | 1417 | 1362 |
| naive torch, `async def` | 486 | 429 | 432 | 454 | 436 | 472 | 459 |
| downshift 0.5.1 | 583 | 842 | 1008 | 1000 | 1001 | 1082 | 1015 |
| downshift 0.5.1 base64 | 597 | 828 | 943 | 962 | 983 | 989 | 982 |
| downshift 0.5.1 safetensors | 514 | 892 | 985 | 952 | 994 | 977 | 1059 |
| downshift 0.5.1 `--execution inline` | 917 | 943 | 939 | 959 | 973 | 972 | 983 |
| downshift 1.0.0 | 632 | 844 | 1007 | 930 | 935 | 967 | 1112 |
| downshift 1.0.0 base64 | 676 | 945 | 1014 | 1006 | 1035 | 1061 | 1059 |
| downshift 1.0.0 safetensors | 665 | 904 | 991 | 988 | 1028 | 1023 | 1064 |
| downshift 1.0.0 `--execution inline` | 870 | 905 | 909 | 928 | 931 | 950 | 943 |
| downshift 1.0.0 result cache on | 701 | 875 | 1056 | 1028 | 920 | 960 | 947 |
| downshift 1.0.0 `--metrics` | 630 | 874 | 972 | 983 | 1017 | 1031 | 1014 |

`scatter_include_self_false`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1097 | 1341 | 1345 | 1277 | 1421 | 1393 | 1256 |
| naive FastAPI + ONNX Runtime | 1200 | 1589 | 1692 | 1667 | 1476 | 1582 | 1236 |
| naive torch, `async def` | 1398 | 1506 | 1568 | 1590 | 1371 | 1588 | 1643 |
| downshift 0.5.1 | 853 | 1047 | 1042 | 926 | 989 | 1035 | 959 |
| downshift 0.5.1 base64 | 880 | 1028 | 1023 | 925 | 910 | 951 | 966 |
| downshift 0.5.1 safetensors | 804 | 1003 | 967 | 903 | 925 | 971 | 970 |
| downshift 0.5.1 `--execution inline` | 1074 | 1270 | 1293 | 1325 | 1287 | 1359 | 1316 |
| downshift 1.0.0 | 646 | 827 | 807 | 845 | 906 | 930 | 959 |
| downshift 1.0.0 base64 | 750 | 946 | 919 | 862 | 923 | 903 | 884 |
| downshift 1.0.0 safetensors | 741 | 940 | 890 | 869 | 886 | 878 | 909 |
| downshift 1.0.0 `--execution inline` | 962 | 1169 | 1638 | 1388 | 1423 | 1467 | 1350 |
| downshift 1.0.0 result cache on | 843 | 1044 | 1003 | 913 | 1026 | 1045 | 983 |
| downshift 1.0.0 `--metrics` | 943 | 1174 | 1039 | 979 | 1014 | 1055 | 1001 |

`mlp_large`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 316 | 416 | 676 | 799 | 882 | 932 | 877 |
| naive FastAPI + ONNX Runtime | 500 | 628 | 893 | 948 | 942 | 980 | 913 |
| naive torch, `async def` | 391 | 422 | 511 | 516 | 509 | 577 | 656 |
| downshift 0.5.1 | 499 | 699 | 955 | 1016 | 1038 | 1070 | 1074 |
| downshift 0.5.1 base64 | 550 | 691 | 758 | 767 | 762 | 793 | 805 |
| downshift 0.5.1 safetensors | 550 | 696 | 750 | 764 | 787 | 808 | 819 |
| downshift 1.0.0 | 490 | 662 | 891 | 985 | 956 | 969 | 1013 |
| downshift 1.0.0 base64 | 513 | 649 | 690 | 725 | 742 | 761 | 780 |
| downshift 1.0.0 safetensors | 550 | 702 | 746 | 762 | 769 | 792 | 806 |
| downshift 1.0.0 result cache on | 1062 | 1348 | 1591 | 1555 | 1547 | 1592 | 1629 |
| downshift 1.0.0 `--metrics` | 485 | 668 | 915 | 978 | 982 | 971 | 988 |

`cnn_large`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 265 | 316 | 405 | 398 | 374 | 347 | 352 |
| naive FastAPI + ONNX Runtime | 344 | 430 | 463 | 461 | 451 | 443 | 428 |
| naive torch, `async def` | 290 | 294 | 345 | 340 | 335 | 330 | 340 |
| downshift 0.5.1 | 493 | 698 | 860 | 885 | 933 | 911 | 906 |
| downshift 0.5.1 base64 | 655 | 928 | 1186 | 1256 | 1294 | 1314 | 1345 |
| downshift 0.5.1 safetensors | 678 | 955 | 1251 | 1307 | 1367 | 1415 | 1421 |
| downshift 1.0.0 | 497 | 687 | 847 | 873 | 872 | 852 | 837 |
| downshift 1.0.0 base64 | 643 | 916 | 1154 | 1229 | 1258 | 1278 | 1307 |
| downshift 1.0.0 safetensors | 666 | 939 | 1224 | 1300 | 1348 | 1378 | 1384 |
| downshift 1.0.0 result cache on | 806 | 976 | 1182 | 1127 | 1124 | 1140 | 1029 |
| downshift 1.0.0 `--metrics` | 494 | 675 | 836 | 858 | 864 | 848 | 811 |

`bert_small`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 32 | 70 | 90 | 97 | 89 | 79 | 73 |
| naive FastAPI + ONNX Runtime | 32 | 76 | 125 | 148 | 144 | 134 | 134 |
| naive torch, `async def` | 31 | 64 | 70 | 71 | 70 | 66 | 58 |
| downshift 0.5.1 | 90 | 137 | 184 | 251 | 260 | 244 | 238 |
| downshift 0.5.1 base64 | 103 | 152 | 221 | 295 | 294 | 286 | 282 |
| downshift 0.5.1 safetensors | 111 | 170 | 280 | 326 | 320 | 315 | 302 |
| downshift 1.0.0 | 91 | 136 | 183 | 259 | 249 | 248 | 225 |
| downshift 1.0.0 base64 | 103 | 153 | 218 | 299 | 292 | 287 | 280 |
| downshift 1.0.0 safetensors | 112 | 176 | 278 | 319 | 319 | 313 | 305 |
| downshift 1.0.0 result cache on | 91 | 135 | 180 | 252 | 252 | 246 | 238 |
| downshift 1.0.0 `--metrics` | 92 | 133 | 182 | 245 | 251 | 252 | 231 |

### Batched requests at concurrency 8 (rows/s)

Requests/s times batch size: the work actually done.

| server | `clean_mlp` b8 | `clean_mlp` b32 | `dynamic_batch_cnn` b8 | `dynamic_batch_cnn` b32 | `gnn_gcn` b8 | `gnn_gcn` b32 | `tiny_bert` b8 | `tiny_bert` b32 | `scatter_include_self_false` b8 | `scatter_include_self_false` b32 | `mlp_large` b8 | `mlp_large` b32 | `cnn_large` b8 | `cnn_large` b32 | `bert_small` b8 | `bert_small` b32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 7751 | 37182 | 1834 | 1806 | 4892 | 13421 | 2403 | 7456 | 9431 | 20470 | 1958 | 2497 | 469 | 420 | 141 | 87 |
| naive FastAPI + ONNX Runtime | 11577 | 40302 | 1912 | 1866 | 11422 | 21840 | 8677 | 17882 | 11144 | 25354 | 2281 | 2628 | 535 | 483 | 160 | 122 |
| downshift 0.5.1 | 7894 | 41629 | 4514 | 6365 | 10628 | 31290 | 6927 | 22219 | 10991 | 34255 | 4875 | 7701 | 1666 | 1586 | 311 | 237 |
| downshift 0.5.1 base64 | 8082 | 32138 | 9590 | 24020 | 10891 | 39344 | 7486 | 34531 | 7981 | 43694 | 7598 | 18506 | 5165 | 6624 | 394 | 331 |
| downshift 0.5.1 safetensors | 8016 | 31899 | 10402 | 40083 | 10418 | 39447 | 7872 | 37419 | 7492 | 44160 | 8532 | 26859 | 6803 | 7578 | 414 | 352 |
| downshift 0.5.1 `--execution inline` | 11939 | 54388 | 4651 | 6495 | 11474 | 29495 | 5885 | 12783 | 12538 | 38698 | — | — | — | — | — | — |
| downshift 1.0.0 | 8586 | 41651 | 4102 | 5291 | 10056 | 29392 | 7603 | 24320 | 10118 | 29631 | 4620 | 7142 | 1464 | 1519 | 303 | 244 |
| downshift 1.0.0 base64 | 8411 | 32862 | 9496 | 23465 | 10695 | 38386 | 7667 | 34591 | 7142 | 42417 | 7433 | 18349 | 5010 | 6727 | 398 | 325 |
| downshift 1.0.0 safetensors | 8420 | 33988 | 10698 | 38320 | 10679 | 40815 | 7681 | 36920 | 6975 | 42740 | 8561 | 26416 | 6825 | 7630 | 416 | 349 |
| downshift 1.0.0 `--execution inline` | 11805 | 56024 | 4421 | 5070 | 11457 | 30143 | 5722 | 13307 | 12376 | 37163 | — | — | — | — | — | — |
| downshift 1.0.0 result cache on | 9409 | 36611 | 4318 | 5542 | 9790 | 29083 | 7013 | 21983 | 10593 | 32004 | 6137 | 7585 | 1950 | 2044 | 298 | 247 |
| downshift 1.0.0 `--metrics` | 7917 | 38453 | 4012 | 4790 | 9814 | 29426 | 7180 | 20543 | 9968 | 30706 | 4565 | 7256 | 1449 | 1568 | 317 | 212 |

### Wire encodings, batch 32, concurrency 8

| model | server | request bytes | requests/s | p50 ms |
|---|---|---|---|---|
| `clean_mlp` | downshift 0.5.1 | 10659 | 1301 | 6.55 |
| `clean_mlp` | downshift 0.5.1 base64 | 2831 | 1004 | 7.90 |
| `clean_mlp` | downshift 0.5.1 safetensors | 2120 | 997 | 7.93 |
| `clean_mlp` | downshift 1.0.0 | 10659 | 1302 | 6.63 |
| `clean_mlp` | downshift 1.0.0 base64 | 2831 | 1027 | 7.74 |
| `clean_mlp` | downshift 1.0.0 safetensors | 2120 | 1062 | 7.46 |
| `dynamic_batch_cnn` | downshift 0.5.1 | 510377 | 199 | 42.88 |
| `dynamic_batch_cnn` | downshift 0.5.1 base64 | 131178 | 751 | 11.35 |
| `dynamic_batch_cnn` | downshift 0.5.1 safetensors | 98384 | 1253 | 7.09 |
| `dynamic_batch_cnn` | downshift 1.0.0 | 510377 | 165 | 50.55 |
| `dynamic_batch_cnn` | downshift 1.0.0 base64 | 131178 | 733 | 11.57 |
| `dynamic_batch_cnn` | downshift 1.0.0 safetensors | 98384 | 1198 | 7.17 |
| `gnn_gcn` | downshift 0.5.1 | 34970 | 978 | 8.59 |
| `gnn_gcn` | downshift 0.5.1 base64 | 15184 | 1229 | 6.81 |
| `gnn_gcn` | downshift 0.5.1 safetensors | 11408 | 1233 | 6.66 |
| `gnn_gcn` | downshift 1.0.0 | 34970 | 919 | 9.32 |
| `gnn_gcn` | downshift 1.0.0 base64 | 15184 | 1200 | 7.20 |
| `gnn_gcn` | downshift 1.0.0 safetensors | 11408 | 1275 | 6.66 |
| `tiny_bert` | downshift 0.5.1 | 1941 | 694 | 11.46 |
| `tiny_bert` | downshift 0.5.1 base64 | 5636 | 1079 | 7.40 |
| `tiny_bert` | downshift 0.5.1 safetensors | 4248 | 1169 | 7.31 |
| `tiny_bert` | downshift 1.0.0 | 1941 | 760 | 10.55 |
| `tiny_bert` | downshift 1.0.0 base64 | 5636 | 1081 | 7.73 |
| `tiny_bert` | downshift 1.0.0 safetensors | 4248 | 1154 | 7.21 |
| `scatter_include_self_false` | downshift 0.5.1 | 32706 | 1070 | 7.74 |
| `scatter_include_self_false` | downshift 0.5.1 base64 | 10402 | 1365 | 6.16 |
| `scatter_include_self_false` | downshift 0.5.1 safetensors | 7824 | 1380 | 6.12 |
| `scatter_include_self_false` | downshift 1.0.0 | 32706 | 926 | 8.92 |
| `scatter_include_self_false` | downshift 1.0.0 base64 | 10402 | 1326 | 6.18 |
| `scatter_include_self_false` | downshift 1.0.0 safetensors | 7824 | 1336 | 5.84 |
| `mlp_large` | downshift 0.5.1 | 338169 | 241 | 35.17 |
| `mlp_large` | downshift 0.5.1 base64 | 87484 | 578 | 14.24 |
| `mlp_large` | downshift 0.5.1 safetensors | 65608 | 839 | 9.64 |
| `mlp_large` | downshift 1.0.0 | 338169 | 223 | 37.31 |
| `mlp_large` | downshift 1.0.0 base64 | 87484 | 573 | 14.50 |
| `mlp_large` | downshift 1.0.0 safetensors | 65608 | 825 | 9.71 |
| `cnn_large` | downshift 0.5.1 | 2034322 | 50 | 176.38 |
| `cnn_large` | downshift 0.5.1 base64 | 524394 | 207 | 38.98 |
| `cnn_large` | downshift 0.5.1 safetensors | 393296 | 237 | 36.40 |
| `cnn_large` | downshift 1.0.0 | 2034322 | 47 | 180.42 |
| `cnn_large` | downshift 1.0.0 base64 | 524394 | 210 | 38.42 |
| `cnn_large` | downshift 1.0.0 safetensors | 393296 | 238 | 34.26 |
| `bert_small` | downshift 0.5.1 | 39700 | 7 | 938.67 |
| `bert_small` | downshift 0.5.1 base64 | 87560 | 10 | 681.13 |
| `bert_small` | downshift 0.5.1 safetensors | 65696 | 11 | 667.62 |
| `bert_small` | downshift 1.0.0 | 39700 | 8 | 1085.45 |
| `bert_small` | downshift 1.0.0 base64 | 87560 | 10 | 664.07 |
| `bert_small` | downshift 1.0.0 safetensors | 65696 | 11 | 619.25 |

### `--workers` sweep, batch 1 (requests/s)

| model | downshift | workers | c=1 | c=8 | c=32 | boot s |
|---|---|---|---|---|---|---|
| `bert_small` | 0.5.1 | 1 | 91 | 254 | 246 | 19.3 |
| `bert_small` | 0.5.1 | 2 | 92 | 252 | 261 | 22.6 |
| `bert_small` | 0.5.1 | 4 | 69 | 285 | 346 | 23.1 |
| `bert_small` | 1.0.0 | 1 | 90 | 252 | 246 | 19.6 |
| `bert_small` | 1.0.0 | 2 | 94 | 251 | 267 | 23.5 |
| `bert_small` | 1.0.0 | 4 | 69 | 278 | 327 | 23.1 |
| `clean_mlp` | 0.5.1 | 1 | 1032 | 1150 | 1303 | 8.5 |
| `clean_mlp` | 0.5.1 | 2 | 963 | 2251 | 2450 | 11.8 |
| `clean_mlp` | 0.5.1 | 4 | 1032 | 4581 | 4372 | 12.3 |
| `clean_mlp` | 1.0.0 | 1 | 1005 | 1135 | 1257 | 8.7 |
| `clean_mlp` | 1.0.0 | 2 | 997 | 2228 | 2373 | 12.1 |
| `clean_mlp` | 1.0.0 | 4 | 992 | 4123 | 4254 | 12.1 |
| `mlp_large` | 0.5.1 | 1 | 522 | 1013 | 1050 | 8.5 |
| `mlp_large` | 0.5.1 | 2 | 520 | 1452 | 1715 | 12.2 |
| `mlp_large` | 0.5.1 | 4 | 526 | 1858 | 2785 | 12.3 |
| `mlp_large` | 1.0.0 | 1 | 499 | 981 | 990 | 9.1 |
| `mlp_large` | 1.0.0 | 2 | 505 | 1483 | 1710 | 12.8 |
| `mlp_large` | 1.0.0 | 4 | 514 | 1736 | 2759 | 12.8 |

### Correctness

Worst max absolute error against eager PyTorch over every row a server answered.

| server | `clean_mlp` | `dynamic_batch_cnn` | `gnn_gcn` | `tiny_bert` | `scatter_include_self_false` | `mlp_large` | `cnn_large` | `bert_small` |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| naive FastAPI + ONNX Runtime | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 1.2e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| naive torch, `async def` | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| downshift 0.5.1 | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | 2.6e-07 | 5.9e-08 | 2.7e-06 |
| downshift 0.5.1 base64 | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 0.0e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| downshift 0.5.1 safetensors | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 0.0e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| downshift 0.5.1 `--execution inline` | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | — | — | — |
| downshift 1.0.0 | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | 2.6e-07 | 5.9e-08 | 2.7e-06 |
| downshift 1.0.0 base64 | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 0.0e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| downshift 1.0.0 safetensors | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 0.0e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| downshift 1.0.0 `--execution inline` | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | — | — | — |
| downshift 1.0.0 result cache on | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | 2.6e-07 | 5.9e-08 | 2.7e-06 |
| downshift 1.0.0 `--metrics` | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | 2.6e-07 | 5.9e-08 | 2.7e-06 |

Failed requests across every load window: 0.

## Real Hugging Face models

Text in, through the real CLI. 5.0 s windows at concurrency 1, 8; a `/health` probe runs every 50 ms during each window (what a liveness probe sees). Payloads: `short_b1` one sentence, `short_b8` eight sentences, `long_b1` ~180 tokens.

### all-MiniLM-L6-v2

`downshift check` wall time: 1.0.0 21.9 s (exit 0), 0.5.1 20.9 s (exit 0).

| server | ready s | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | worst /health p99 ms |
|---|---|---|---|---|---|
| naive FastAPI + transformers | 12.4 | 142 | 83 | 74 | 19 |
| naive transformers, `async def` | 11.8 | 115 | 46 | 36 | 481 |
| downshift 0.5.1 | 20.5 | 532 | 139 | 78 | 22 |
| downshift 0.5.1 `--backend torch` | 12.4 | 145 | 86 | 66 | 18 |
| downshift 0.5.1 `--max-concurrency 4` | 20.5 | 529 | 137 | 78 | 22 |
| downshift 0.5.1 `--execution inline` | 20.4 | 536 | 138 | 78 | 23 |
| downshift 1.0.0 | 21.3 | 528 | 137 | 78 | 22 |
| downshift 1.0.0 `--backend torch` | 12.5 | 148 | 84 | 70 | 18 |
| downshift 1.0.0 `--max-concurrency 4` | 21.4 | 530 | 136 | 78 | 23 |
| downshift 1.0.0 `--execution inline` | 21.3 | 521 | 138 | 78 | 22 |
| downshift 1.0.0 result cache on | 21.2 | 1504 | 1239 | 1557 | 19 |

p50 latency (ms) at concurrency 1:

| server | short_b1 | short_b8 | long_b1 |
|---|---|---|---|
| naive FastAPI + transformers | 9.4 | 21.8 | 28.4 |
| naive transformers, `async def` | 8.9 | 21.3 | 27.1 |
| downshift 0.5.1 | 4.8 | 19.4 | 33.3 |
| downshift 0.5.1 `--backend torch` | 9.6 | 23.6 | 29.4 |
| downshift 0.5.1 `--max-concurrency 4` | 4.8 | 19.5 | 33.3 |
| downshift 0.5.1 `--execution inline` | 4.9 | 19.4 | 33.5 |
| downshift 1.0.0 | 4.8 | 19.4 | 33.2 |
| downshift 1.0.0 `--backend torch` | 9.6 | 23.7 | 29.7 |
| downshift 1.0.0 `--max-concurrency 4` | 4.9 | 19.6 | 33.4 |
| downshift 1.0.0 `--execution inline` | 4.9 | 19.5 | 33.6 |
| downshift 1.0.0 result cache on | 0.8 | 0.9 | 0.8 |

Agreement with downshift 1.0.0 `--backend torch` on `short_b8`:

| server | max abs err | min cosine |
|---|---|---|
| naive FastAPI + transformers | 7.2e-09 | 1.0000000 |
| naive transformers, `async def` | 7.2e-09 | 1.0000000 |
| downshift 0.5.1 | 1.3e-07 | 1.0000000 |
| downshift 0.5.1 `--backend torch` | 0.0e+00 | 1.0000000 |
| downshift 0.5.1 `--max-concurrency 4` | 1.3e-07 | 1.0000000 |
| downshift 0.5.1 `--execution inline` | 1.3e-07 | 1.0000000 |
| downshift 1.0.0 | 1.3e-07 | 1.0000000 |
| downshift 1.0.0 `--backend torch` | 0.0e+00 | 1.0000000 |
| downshift 1.0.0 `--max-concurrency 4` | 1.3e-07 | 1.0000000 |
| downshift 1.0.0 `--execution inline` | 1.3e-07 | 1.0000000 |
| downshift 1.0.0 result cache on | 1.3e-07 | 1.0000000 |

### prompt-guard-86m

`downshift check` wall time: 1.0.0 47.4 s (exit 0), 0.5.1 41.6 s (exit 0).

| server | ready s | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | worst /health p99 ms |
|---|---|---|---|---|---|
| naive FastAPI + transformers | 14.6 | 16 | 8 | 6 | 18 |
| naive transformers, `async def` | 14.6 | 8 | 3 | 2 | 5220 |
| downshift 0.5.1 | 42.6 | 102 | 12 | 7 | 294 |
| downshift 0.5.1 `--backend torch` | 15.2 | 16 | 7 | 5 | 170 |
| downshift 0.5.1 `--max-concurrency 4` | 41.9 | 103 | 12 | 7 | 306 |
| downshift 1.0.0 | 46.6 | 103 | 12 | 7 | 319 |
| downshift 1.0.0 `--backend torch` | 15.3 | 16 | 7 | 5 | 17 |
| downshift 1.0.0 `--max-concurrency 4` | 46.4 | 102 | 12 | 7 | 317 |
| downshift 1.0.0 result cache on | 46.5 | 1308 | 1288 | 1359 | 271 |

p50 latency (ms) at concurrency 1:

| server | short_b1 | short_b8 | long_b1 |
|---|---|---|---|
| naive FastAPI + transformers | 108.1 | 229.6 | 295.0 |
| naive transformers, `async def` | 107.6 | 231.2 | 292.4 |
| downshift 0.5.1 | 21.5 | 141.3 | 216.7 |
| downshift 0.5.1 `--backend torch` | 109.1 | 229.2 | 294.7 |
| downshift 0.5.1 `--max-concurrency 4` | 21.2 | 141.4 | 217.8 |
| downshift 1.0.0 | 21.2 | 141.0 | 216.5 |
| downshift 1.0.0 `--backend torch` | 108.5 | 232.5 | 297.0 |
| downshift 1.0.0 `--max-concurrency 4` | 21.2 | 141.2 | 217.1 |
| downshift 1.0.0 result cache on | 0.8 | 0.9 | 0.8 |

Agreement with downshift 1.0.0 `--backend torch` on `short_b8`:

| server | max abs err | label agreement |
|---|---|---|
| naive FastAPI + transformers | 2.2e-07 | 8/8 |
| naive transformers, `async def` | 2.2e-07 | 8/8 |
| downshift 0.5.1 | 2.3e-05 | 8/8 |
| downshift 0.5.1 `--backend torch` | 0.0e+00 | 8/8 |
| downshift 0.5.1 `--max-concurrency 4` | 2.3e-05 | 8/8 |
| downshift 1.0.0 | 2.3e-05 | 8/8 |
| downshift 1.0.0 `--backend torch` | 0.0e+00 | 8/8 |
| downshift 1.0.0 `--max-concurrency 4` | 2.3e-05 | 8/8 |
| downshift 1.0.0 result cache on | 2.3e-05 | 8/8 |

## Where the time goes

Median of 300 sequential requests on one keep-alive connection, batch 1 (HF: one sentence). `total` is what the client measured; the stage columns are the server's own `Server-Timing`; `HTTP + hand-offs` is the rest: uvicorn, routing, middleware and thread hand-offs back to the event loop. Naive servers have no `Server-Timing`, so their total is all in the last column. 0.4.0 reports one `codec` stage (decode and encode); 0.5.0 splits it, and its `encode` includes serializing the response.

### clean_mlp

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.90 | — | — | — | — | — | — | 0.90 |
| naive FastAPI + ONNX Runtime | 0.74 | — | — | — | — | — | — | 0.74 |
| naive torch, async def | 0.48 | — | — | — | — | — | — | 0.48 |
| downshift 0.5.1 | 1.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.22 | 0.04 | 0.66 |
| downshift 0.5.1 base64 | 0.96 | 0.02 | 0.03 | 0.04 | 0.03 | 0.20 | 0.03 | 0.61 |
| downshift 0.5.1 safetensors | 1.06 | 0.05 | 0.04 | 0.02 | 0.04 | 0.20 | 0.04 | 0.67 |
| downshift 0.5.1 --execution inline | 0.61 | 0.02 | 0.00 | 0.02 | 0.00 | 0.15 | 0.03 | 0.39 |
| downshift 1.0.0 | 1.00 | 0.02 | 0.04 | 0.03 | 0.03 | 0.21 | 0.04 | 0.65 |
| downshift 1.0.0 base64 | 1.14 | 0.03 | 0.04 | 0.05 | 0.03 | 0.23 | 0.04 | 0.68 |
| downshift 1.0.0 safetensors | 0.99 | 0.04 | 0.03 | 0.02 | 0.03 | 0.17 | 0.04 | 0.63 |
| downshift 1.0.0 --execution inline | 0.72 | 0.02 | 0.00 | 0.02 | 0.00 | 0.18 | 0.03 | 0.47 |
| downshift 1.0.0 result cache on | 0.83 | 0.02 | 0.03 | 0.06 | 0.00 | 0.00 | 0.03 | 0.69 |
| downshift 1.0.0 --metrics | 1.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.21 | 0.04 | 0.66 |

### dynamic_batch_cnn

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1.44 | — | — | — | — | — | — | 1.44 |
| naive FastAPI + ONNX Runtime | 1.27 | — | — | — | — | — | — | 1.27 |
| naive torch, async def | 1.08 | — | — | — | — | — | — | 1.08 |
| downshift 0.5.1 | 1.29 | 0.08 | 0.04 | 0.11 | 0.03 | 0.29 | 0.05 | 0.71 |
| downshift 0.5.1 base64 | 1.17 | 0.03 | 0.04 | 0.06 | 0.03 | 0.29 | 0.04 | 0.68 |
| downshift 0.5.1 safetensors | 1.13 | 0.06 | 0.03 | 0.03 | 0.03 | 0.28 | 0.05 | 0.65 |
| downshift 0.5.1 --execution inline | 0.81 | 0.05 | 0.00 | 0.07 | 0.00 | 0.20 | 0.03 | 0.45 |
| downshift 1.0.0 | 1.35 | 0.08 | 0.04 | 0.12 | 0.03 | 0.30 | 0.05 | 0.76 |
| downshift 1.0.0 base64 | 1.17 | 0.03 | 0.03 | 0.07 | 0.03 | 0.29 | 0.04 | 0.68 |
| downshift 1.0.0 safetensors | 1.12 | 0.06 | 0.03 | 0.03 | 0.03 | 0.29 | 0.05 | 0.66 |
| downshift 1.0.0 --execution inline | 0.90 | 0.06 | 0.00 | 0.08 | 0.00 | 0.21 | 0.03 | 0.51 |
| downshift 1.0.0 result cache on | 0.98 | 0.06 | 0.03 | 0.11 | 0.00 | 0.00 | 0.03 | 0.73 |
| downshift 1.0.0 --metrics | 1.30 | 0.08 | 0.03 | 0.11 | 0.03 | 0.29 | 0.04 | 0.73 |

### gnn_gcn

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1.93 | — | — | — | — | — | — | 1.93 |
| naive FastAPI + ONNX Runtime | 0.91 | — | — | — | — | — | — | 0.91 |
| naive torch, async def | 1.37 | — | — | — | — | — | — | 1.37 |
| downshift 0.5.1 | 1.11 | 0.03 | 0.03 | 0.04 | 0.03 | 0.29 | 0.04 | 0.63 |
| downshift 0.5.1 base64 | 1.14 | 0.02 | 0.04 | 0.07 | 0.03 | 0.29 | 0.03 | 0.64 |
| downshift 0.5.1 safetensors | 1.17 | 0.06 | 0.04 | 0.04 | 0.04 | 0.28 | 0.04 | 0.66 |
| downshift 0.5.1 --execution inline | 0.81 | 0.02 | 0.00 | 0.04 | 0.00 | 0.26 | 0.04 | 0.44 |
| downshift 1.0.0 | 1.14 | 0.03 | 0.04 | 0.05 | 0.04 | 0.30 | 0.04 | 0.66 |
| downshift 1.0.0 base64 | 1.18 | 0.02 | 0.04 | 0.07 | 0.04 | 0.30 | 0.04 | 0.66 |
| downshift 1.0.0 safetensors | 1.17 | 0.06 | 0.04 | 0.04 | 0.03 | 0.28 | 0.04 | 0.67 |
| downshift 1.0.0 --execution inline | 0.86 | 0.02 | 0.00 | 0.04 | 0.00 | 0.28 | 0.04 | 0.46 |
| downshift 1.0.0 result cache on | 1.15 | 0.03 | 0.04 | 0.04 | 0.03 | 0.28 | 0.04 | 0.67 |
| downshift 1.0.0 --metrics | 1.17 | 0.03 | 0.04 | 0.05 | 0.04 | 0.29 | 0.04 | 0.69 |

### tiny_bert

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 3.63 | — | — | — | — | — | — | 3.63 |
| naive FastAPI + ONNX Runtime | 1.10 | — | — | — | — | — | — | 1.10 |
| naive torch, async def | 1.86 | — | — | — | — | — | — | 1.86 |
| downshift 0.5.1 | 1.56 | 0.03 | 0.04 | 0.04 | 0.03 | 0.61 | 0.06 | 0.75 |
| downshift 0.5.1 base64 | 1.47 | 0.03 | 0.04 | 0.07 | 0.03 | 0.59 | 0.05 | 0.69 |
| downshift 0.5.1 safetensors | 1.41 | 0.07 | 0.03 | 0.04 | 0.03 | 0.58 | 0.06 | 0.63 |
| downshift 0.5.1 --execution inline | 0.95 | 0.02 | 0.00 | 0.02 | 0.00 | 0.41 | 0.04 | 0.45 |
| downshift 1.0.0 | 1.45 | 0.03 | 0.03 | 0.04 | 0.03 | 0.59 | 0.05 | 0.70 |
| downshift 1.0.0 base64 | 1.37 | 0.03 | 0.03 | 0.07 | 0.03 | 0.58 | 0.05 | 0.62 |
| downshift 1.0.0 safetensors | 1.44 | 0.07 | 0.04 | 0.04 | 0.03 | 0.58 | 0.06 | 0.67 |
| downshift 1.0.0 --execution inline | 1.05 | 0.02 | 0.00 | 0.03 | 0.00 | 0.46 | 0.04 | 0.49 |
| downshift 1.0.0 result cache on | 1.32 | 0.03 | 0.03 | 0.03 | 0.03 | 0.57 | 0.05 | 0.62 |
| downshift 1.0.0 --metrics | 1.50 | 0.03 | 0.04 | 0.04 | 0.03 | 0.59 | 0.05 | 0.74 |

### scatter_include_self_false

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.84 | — | — | — | — | — | — | 0.84 |
| naive FastAPI + ONNX Runtime | 0.78 | — | — | — | — | — | — | 0.78 |
| naive torch, async def | 0.61 | — | — | — | — | — | — | 0.61 |
| downshift 0.5.1 | 1.08 | 0.03 | 0.04 | 0.05 | 0.03 | 0.19 | 0.04 | 0.67 |
| downshift 0.5.1 base64 | 1.04 | 0.02 | 0.04 | 0.06 | 0.04 | 0.16 | 0.03 | 0.68 |
| downshift 0.5.1 safetensors | 1.11 | 0.06 | 0.04 | 0.04 | 0.03 | 0.20 | 0.05 | 0.67 |
| downshift 0.5.1 --execution inline | 0.72 | 0.02 | 0.00 | 0.04 | 0.00 | 0.16 | 0.03 | 0.46 |
| downshift 1.0.0 | 1.41 | 0.04 | 0.05 | 0.07 | 0.04 | 0.27 | 0.05 | 0.87 |
| downshift 1.0.0 base64 | 1.24 | 0.03 | 0.04 | 0.08 | 0.04 | 0.24 | 0.04 | 0.77 |
| downshift 1.0.0 safetensors | 1.31 | 0.07 | 0.04 | 0.05 | 0.04 | 0.24 | 0.05 | 0.79 |
| downshift 1.0.0 --execution inline | 0.79 | 0.02 | 0.00 | 0.05 | 0.00 | 0.17 | 0.04 | 0.51 |
| downshift 1.0.0 result cache on | 1.05 | 0.02 | 0.04 | 0.05 | 0.04 | 0.17 | 0.04 | 0.69 |
| downshift 1.0.0 --metrics | 1.07 | 0.03 | 0.03 | 0.07 | 0.03 | 0.23 | 0.05 | 0.65 |

### mlp_large

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 2.95 | — | — | — | — | — | — | 2.95 |
| naive FastAPI + ONNX Runtime | 1.81 | — | — | — | — | — | — | 1.81 |
| naive torch, async def | 2.17 | — | — | — | — | — | — | 2.17 |
| downshift 0.5.1 | 1.86 | 0.08 | 0.04 | 0.08 | 0.03 | 0.70 | 0.09 | 0.83 |
| downshift 0.5.1 base64 | 1.70 | 0.04 | 0.04 | 0.07 | 0.03 | 0.69 | 0.07 | 0.77 |
| downshift 0.5.1 safetensors | 1.70 | 0.07 | 0.04 | 0.04 | 0.03 | 0.68 | 0.07 | 0.78 |
| downshift 1.0.0 | 1.86 | 0.07 | 0.04 | 0.08 | 0.04 | 0.71 | 0.09 | 0.85 |
| downshift 1.0.0 base64 | 1.78 | 0.04 | 0.04 | 0.07 | 0.03 | 0.70 | 0.07 | 0.82 |
| downshift 1.0.0 safetensors | 1.65 | 0.07 | 0.03 | 0.04 | 0.03 | 0.67 | 0.07 | 0.74 |
| downshift 1.0.0 result cache on | 0.87 | 0.04 | 0.03 | 0.08 | 0.00 | 0.00 | 0.05 | 0.66 |
| downshift 1.0.0 --metrics | 1.94 | 0.07 | 0.04 | 0.08 | 0.04 | 0.76 | 0.09 | 0.87 |

### cnn_large

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 3.62 | — | — | — | — | — | — | 3.62 |
| naive FastAPI + ONNX Runtime | 2.77 | — | — | — | — | — | — | 2.77 |
| naive torch, async def | 3.39 | — | — | — | — | — | — | 3.39 |
| downshift 0.5.1 | 1.91 | 0.22 | 0.04 | 0.30 | 0.03 | 0.57 | 0.04 | 0.73 |
| downshift 0.5.1 base64 | 1.43 | 0.04 | 0.03 | 0.11 | 0.03 | 0.58 | 0.04 | 0.61 |
| downshift 0.5.1 safetensors | 1.39 | 0.05 | 0.04 | 0.03 | 0.03 | 0.57 | 0.05 | 0.65 |
| downshift 1.0.0 | 1.92 | 0.22 | 0.04 | 0.32 | 0.03 | 0.58 | 0.04 | 0.73 |
| downshift 1.0.0 base64 | 1.43 | 0.04 | 0.03 | 0.11 | 0.03 | 0.58 | 0.04 | 0.63 |
| downshift 1.0.0 safetensors | 1.40 | 0.06 | 0.03 | 0.03 | 0.03 | 0.59 | 0.05 | 0.64 |
| downshift 1.0.0 result cache on | 1.13 | 0.14 | 0.03 | 0.24 | 0.00 | 0.00 | 0.03 | 0.70 |
| downshift 1.0.0 --metrics | 1.96 | 0.22 | 0.04 | 0.32 | 0.03 | 0.58 | 0.04 | 0.77 |

### bert_small

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 13.67 | — | — | — | — | — | — | 13.67 |
| naive FastAPI + ONNX Runtime | 12.48 | — | — | — | — | — | — | 12.48 |
| naive torch, async def | 13.99 | — | — | — | — | — | — | 13.99 |
| downshift 0.5.1 | 10.28 | 0.05 | 0.05 | 0.07 | 0.04 | 7.28 | 1.66 | 1.13 |
| downshift 0.5.1 base64 | 9.58 | 0.05 | 0.06 | 0.11 | 0.04 | 7.32 | 0.78 | 1.20 |
| downshift 0.5.1 safetensors | 8.71 | 0.09 | 0.05 | 0.05 | 0.04 | 7.32 | 0.09 | 1.08 |
| downshift 1.0.0 | 10.37 | 0.05 | 0.05 | 0.07 | 0.04 | 7.28 | 1.65 | 1.20 |
| downshift 1.0.0 base64 | 9.42 | 0.05 | 0.05 | 0.11 | 0.04 | 7.24 | 0.76 | 1.14 |
| downshift 1.0.0 safetensors | 8.73 | 0.09 | 0.05 | 0.05 | 0.04 | 7.28 | 0.08 | 1.08 |
| downshift 1.0.0 result cache on | 10.41 | 0.05 | 0.06 | 0.07 | 0.04 | 7.29 | 1.71 | 1.22 |
| downshift 1.0.0 --metrics | 10.38 | 0.05 | 0.05 | 0.07 | 0.04 | 7.22 | 1.71 | 1.26 |

### all-MiniLM-L6-v2

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 9.10 | — | — | — | — | — | — | 9.10 |
| naive transformers, async def | 8.68 | — | — | — | — | — | — | 8.68 |
| downshift 0.5.1 | 4.64 | 0.04 | 0.06 | 0.33 | 0.04 | 3.02 | 0.10 | 1.05 |
| downshift 0.5.1 --backend torch | 9.33 | 0.05 | 0.06 | 0.37 | 0.03 | 7.48 | 0.11 | 1.23 |
| downshift 0.5.1 --max-concurrency 4 | 4.66 | 0.04 | 0.05 | 0.34 | 0.04 | 3.02 | 0.10 | 1.06 |
| downshift 0.5.1 --execution inline | 4.72 | 0.03 | 0.06 | 0.34 | 0.04 | 3.10 | 0.10 | 1.05 |
| downshift 1.0.0 | 4.67 | 0.04 | 0.06 | 0.34 | 0.04 | 3.01 | 0.10 | 1.09 |
| downshift 1.0.0 --backend torch | 9.46 | 0.05 | 0.06 | 0.38 | 0.03 | 7.55 | 0.11 | 1.27 |
| downshift 1.0.0 --max-concurrency 4 | 4.72 | 0.04 | 0.06 | 0.35 | 0.03 | 3.04 | 0.10 | 1.11 |
| downshift 1.0.0 --execution inline | 4.69 | 0.03 | 0.06 | 0.34 | 0.04 | 3.04 | 0.10 | 1.11 |
| downshift 1.0.0 result cache on | 0.70 | 0.02 | 0.02 | 0.01 | 0.00 | 0.00 | 0.05 | 0.59 |

### prompt-guard-86m

| server | total ms | parse | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 107.33 | — | — | — | — | — | — | 107.33 |
| naive transformers, async def | 106.85 | — | — | — | — | — | — | 106.85 |
| downshift 0.5.1 | 21.30 | 0.05 | 0.06 | 0.37 | 0.03 | 19.32 | 0.20 | 1.26 |
| downshift 0.5.1 --backend torch | 107.56 | 0.05 | 0.06 | 0.37 | 0.03 | 105.61 | 0.17 | 1.25 |
| downshift 0.5.1 --max-concurrency 4 | 20.82 | 0.05 | 0.06 | 0.37 | 0.03 | 18.84 | 0.19 | 1.26 |
| downshift 1.0.0 | 20.88 | 0.05 | 0.06 | 0.37 | 0.03 | 18.86 | 0.19 | 1.27 |
| downshift 1.0.0 --backend torch | 107.54 | 0.05 | 0.05 | 0.37 | 0.03 | 105.50 | 0.17 | 1.29 |
| downshift 1.0.0 --max-concurrency 4 | 20.99 | 0.05 | 0.05 | 0.37 | 0.03 | 18.97 | 0.19 | 1.29 |
| downshift 1.0.0 result cache on | 0.74 | 0.02 | 0.03 | 0.01 | 0.00 | 0.00 | 0.07 | 0.61 |

## Notes on this run

- **The result cache is off in every headline row.** Version 1.0.0 turns the result cache on by default for a model with independent rows. The harness sends the same body again and again, so a cache would answer from memory and the row would not measure the model. Every 1.0.0 row but "result cache on" starts with `--result-cache-rows 0`. This includes the `--workers` sweep. Version 0.5.1 has no cache. The "result cache on" rows use the default and send the same body every time, so they show the best case (every request is a hit), not a normal workload.
- **The cache does nothing for two of the four small fixtures.** `downshift check` says `Rows dependent` for `tiny_bert` and `Rows not checked` for `scatter_include_self_false` (the numerics check failed). The default cache is off for both, so their "result cache on" rows are an ordinary run and differ from the default rows only by noise.
- **`--metrics` rows** start the same server as the default row with `--metrics` added.
- **Noise is 10% or more.** Same code, two rows: on `scatter_include_self_false` the `--metrics` row (peak 1174 rps) is faster than the default row (959). Read a difference smaller than about 15% between two rows as noise. Compare the cost of `--metrics` at concurrency 1 and the stage tables, not peak rps.
- **Stalled windows were run again.** Three windows of 1.0.0 had requests that timed out after 30 s or more: `bert_small` `--workers 2` c=32 (14 of 774), `clean_mlp` `--workers 4` c=8 (1 request, 32 s window) and `bert_small` `--workers 4` c=8 (2 requests, 61 s window). Each was measured again and was clean, so the tables show the clean row. **The cause was the harness, not downshift.** It started each server with its output on a pipe that nobody read. A Windows pipe holds about 4 KB, and when it is full the next log write blocks. A py-spy dump of a hung `--workers 2` server showed the `downshift-loader` thread of each worker in `print_ready` → `logger.info` → `write`, so the worker never finished loading and `/ready` stayed on `warmup`. A second look ran the sweep three more times (`workers_repeats.json`) and found the same fault: 3 of 12 server starts never became ready, and 3 of 18 windows (all `--workers 4` c=8) had a timeout. `bench/run.py` now writes the server output to a temp file (`spawn_server`). After the fix, 12 starts of `clean_mlp` `--workers 2` had no hang, and a full sweep had 0 errors in 8 windows (`after_fix` in `workers_repeats.json`). The published `--workers` rows are clean windows and stay valid. The stalled rows are in `failed_rows_workers.json` as a record of the fault. The 0.5.1 sweep ran under the same harness and was not run again after the fix.
- **The first `--workers` pass for 1.0.0 had the cache on** (the harness passed no `--result-cache-rows 0` in that sweep) and showed `mlp_large` at twice the speed of 0.5.1. The harness is now fixed (`bench/run.py`) and the sweep for 1.0.0 was run again with the cache off. The tables show the second pass. The two versions are equal on `--workers` within noise (`clean_mlp` at 4 workers, c=8: 4123 vs 4581 rps).
- **The run came in three passes** because the system ran out of memory once during the `--workers` sweep. The first pass labelled the checkout `1.0.0rc1` and the second `1.0.0`: the version was changed from `1.0.0rc1` to `1.0.0` between the two. No other code changed, and the first-pass rows are relabelled. `results.json` is the merge of the passes.
- Windows are 3 s on the fixtures and 5 s on MiniLM and Prompt Guard. The Python version (3.14.7) differs from the earlier runs (3.12.10), and so does the machine state, so compare rows inside this report, not with `results/v0.5.0/` or with the 1.0.0rc1 run.
