# downshift benchmark: naive servers vs downshift 0.4.0 vs 0.5.0

`win32`, 16 logical CPUs, Python 3.12.10, torch 2.14.0+cpu (CPU), onnxruntime 1.30.0, run 2026-10-04T22:32:59. downshift 0.5.0 from the checkout; downshift 0.4.0 from the pip-installed release. Every downshift server is the real `downshift serve` CLI; all servers ran in one session on the same machine.

## Fixture and compute-heavy models

Closed-loop load, 3.0 s window after 1.0 s warm-up, load generator sharded over up to 6 processes on the same machine. Every server gets the same seeded weights and byte-identical request bodies, and every row's response is checked element-wise against eager PyTorch.

Load-generator ceiling (`GET /health` on a do-nothing server): c=1: 4075 rps, c=2: 7523 rps, c=4: 7889 rps, c=8: 8477 rps, c=16: 8968 rps, c=32: 8879 rps, c=64: 9129 rps. A result near these numbers is measuring the client.

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
| naive FastAPI + torch | 1330 | 885 | 518 | 433 | 1257 | 795 | 363 | 98 |
| naive FastAPI + ONNX Runtime | 1615 | 1004 | 1415 | 1515 | 1355 | 892 | 436 | 149 |
| naive torch, `async def` | 2450 | 1037 | 689 | 572 | 2098 | 585 | 313 | 70 |
| downshift 0.4.0 | 866 | 862 | 810 | 820 | 940 | 696 | 657 | 102 |
| downshift 0.4.0 base64 | 884 | 995 | 765 | 774 | 900 | 605 | 913 | 118 |
| downshift 0.5.0 | 1030 | 1031 | 891 | 906 | 1076 | 815 | 736 | 106 |
| downshift 0.5.0 base64 | 990 | 1260 | 838 | 886 | 1057 | 669 | 1047 | 125 |
| downshift 0.5.0 safetensors | 991 | 1102 | 898 | 871 | 1052 | 681 | 1055 | 128 |
| downshift 0.5.0 `--execution inline` | 1498 | 1211 | 1160 | 844 | 1545 | — | — | — |

p50 latency (ms) at concurrency 1:

| server | `clean_mlp` | `dynamic_batch_cnn` | `gnn_gcn` | `tiny_bert` | `scatter_include_self_false` | `mlp_large` | `cnn_large` | `bert_small` |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.89 | 1.42 | 1.86 | 2.28 | 0.87 | 2.42 | 3.73 | 31.05 |
| naive FastAPI + ONNX Runtime | 0.83 | 1.36 | 1.02 | 1.13 | 0.93 | 1.81 | 2.88 | 30.75 |
| naive torch, `async def` | 0.51 | 1.17 | 1.42 | 1.69 | 0.63 | 2.02 | 3.52 | 31.03 |
| downshift 0.4.0 | 1.28 | 1.52 | 1.50 | 1.64 | 1.16 | 2.23 | 2.25 | 11.44 |
| downshift 0.4.0 base64 | 1.18 | 1.56 | 1.45 | 1.69 | 1.18 | 2.10 | 1.72 | 9.57 |
| downshift 0.5.0 | 1.05 | 1.36 | 1.44 | 1.46 | 1.00 | 2.00 | 2.07 | 11.20 |
| downshift 0.5.0 base64 | 1.13 | 1.22 | 1.49 | 1.52 | 1.03 | 1.96 | 1.56 | 9.40 |
| downshift 0.5.0 safetensors | 1.08 | 1.25 | 1.25 | 1.54 | 1.06 | 1.92 | 1.54 | 9.06 |
| downshift 0.5.0 `--execution inline` | 0.73 | 1.05 | 0.98 | 1.18 | 0.68 | — | — | — |

### Concurrency sweep, batch 1 (requests/s)

`clean_mlp`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1020 | 1325 | 1327 | 1212 | 1330 | 1198 | 1037 |
| naive FastAPI + ONNX Runtime | 1125 | 1469 | 1488 | 1602 | 1615 | 1446 | 1498 |
| naive torch, `async def` | 1763 | 2253 | 2353 | 2264 | 2345 | 2450 | 2394 |
| downshift 0.4.0 | 723 | 849 | 840 | 834 | 866 | 819 | 788 |
| downshift 0.4.0 base64 | 788 | 884 | 750 | 787 | 832 | 771 | 741 |
| downshift 0.5.0 | 858 | 1030 | 880 | 928 | 978 | 1009 | 935 |
| downshift 0.5.0 base64 | 813 | 982 | 924 | 948 | 976 | 990 | 848 |
| downshift 0.5.0 safetensors | 873 | 991 | 844 | 897 | 963 | 964 | 864 |
| downshift 0.5.0 `--execution inline` | 1253 | 1432 | 1425 | 1467 | 1388 | 1424 | 1498 |

`dynamic_batch_cnn`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 667 | 760 | 885 | 859 | 829 | 726 | 637 |
| naive FastAPI + ONNX Runtime | 690 | 828 | 1004 | 977 | 910 | 892 | 769 |
| naive torch, `async def` | 823 | 887 | 1037 | 1034 | 986 | 1010 | 1035 |
| downshift 0.4.0 | 619 | 753 | 824 | 862 | 768 | 854 | 779 |
| downshift 0.4.0 base64 | 609 | 814 | 902 | 916 | 995 | 920 | 897 |
| downshift 0.5.0 | 687 | 883 | 946 | 990 | 1027 | 1031 | 991 |
| downshift 0.5.0 base64 | 777 | 992 | 1070 | 1081 | 1151 | 1168 | 1260 |
| downshift 0.5.0 safetensors | 758 | 935 | 1038 | 1059 | 1098 | 1102 | 1035 |
| downshift 0.5.0 `--execution inline` | 913 | 1018 | 1118 | 1159 | 1177 | 1186 | 1211 |

`gnn_gcn`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 509 | 518 | 509 | 421 | 389 | 363 | 334 |
| naive FastAPI + ONNX Runtime | 903 | 1168 | 1292 | 1415 | 1353 | 1294 | 1181 |
| naive torch, `async def` | 685 | 673 | 687 | 680 | 689 | 662 | 679 |
| downshift 0.4.0 | 619 | 810 | 761 | 758 | 780 | 767 | 756 |
| downshift 0.4.0 base64 | 642 | 764 | 765 | 763 | 745 | 758 | 722 |
| downshift 0.5.0 | 628 | 882 | 789 | 855 | 891 | 870 | 744 |
| downshift 0.5.0 base64 | 629 | 838 | 799 | 816 | 808 | 828 | 684 |
| downshift 0.5.0 safetensors | 752 | 898 | 807 | 820 | 875 | 881 | 825 |
| downshift 0.5.0 `--execution inline` | 931 | 1094 | 1065 | 1081 | 1120 | 1130 | 1160 |

`tiny_bert`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 428 | 433 | 407 | 389 | 357 | 319 | 307 |
| naive FastAPI + ONNX Runtime | 833 | 1259 | 1515 | 1504 | 1491 | 1357 | 1304 |
| naive torch, `async def` | 572 | 521 | 523 | 523 | 526 | 516 | 516 |
| downshift 0.4.0 | 593 | 711 | 795 | 812 | 820 | 802 | 717 |
| downshift 0.4.0 base64 | 577 | 704 | 771 | 774 | 770 | 763 | 681 |
| downshift 0.5.0 | 664 | 868 | 876 | 883 | 906 | 887 | 868 |
| downshift 0.5.0 base64 | 641 | 839 | 837 | 847 | 886 | 864 | 848 |
| downshift 0.5.0 safetensors | 637 | 833 | 832 | 848 | 871 | 862 | 849 |
| downshift 0.5.0 `--execution inline` | 823 | 823 | 825 | 837 | 840 | 844 | 831 |

`scatter_include_self_false`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1071 | 1240 | 1257 | 1207 | 1220 | 1241 | 1070 |
| naive FastAPI + ONNX Runtime | 1014 | 1230 | 1162 | 1355 | 1327 | 1198 | 1185 |
| naive torch, `async def` | 1450 | 1879 | 1951 | 1936 | 2098 | 2036 | 1976 |
| downshift 0.4.0 | 807 | 893 | 933 | 940 | 930 | 925 | 842 |
| downshift 0.4.0 base64 | 782 | 895 | 897 | 897 | 900 | 879 | 878 |
| downshift 0.5.0 | 942 | 1076 | 1015 | 1011 | 1003 | 979 | 972 |
| downshift 0.5.0 base64 | 910 | 1057 | 1000 | 997 | 961 | 962 | 933 |
| downshift 0.5.0 safetensors | 893 | 1052 | 959 | 980 | 964 | 949 | 929 |
| downshift 0.5.0 `--execution inline` | 1329 | 1474 | 1505 | 1532 | 1545 | 1528 | 1509 |

`mlp_large`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 375 | 430 | 679 | 729 | 795 | 790 | 723 |
| naive FastAPI + ONNX Runtime | 526 | 612 | 865 | 892 | 876 | 870 | 813 |
| naive torch, `async def` | 453 | 507 | 576 | 577 | 575 | 578 | 585 |
| downshift 0.4.0 | 426 | 540 | 651 | 688 | 696 | 690 | 622 |
| downshift 0.4.0 base64 | 451 | 554 | 595 | 598 | 605 | 592 | 543 |
| downshift 0.5.0 | 470 | 630 | 750 | 785 | 796 | 815 | 751 |
| downshift 0.5.0 base64 | 480 | 631 | 634 | 641 | 652 | 669 | 619 |
| downshift 0.5.0 safetensors | 488 | 631 | 633 | 648 | 663 | 681 | 619 |

`cnn_large`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 260 | 302 | 363 | 348 | 328 | 305 | 292 |
| naive FastAPI + ONNX Runtime | 342 | 411 | 436 | 407 | 399 | 373 | 359 |
| naive torch, `async def` | 278 | 279 | 313 | 302 | 307 | 299 | 303 |
| downshift 0.4.0 | 435 | 540 | 649 | 657 | 610 | 578 | 542 |
| downshift 0.4.0 base64 | 565 | 722 | 867 | 913 | 913 | 889 | 805 |
| downshift 0.5.0 | 472 | 636 | 712 | 732 | 733 | 736 | 719 |
| downshift 0.5.0 base64 | 620 | 862 | 1000 | 1014 | 1047 | 1045 | 1031 |
| downshift 0.5.0 safetensors | 628 | 877 | 1007 | 1036 | 1055 | 1051 | 1021 |

`bert_small`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 32 | 68 | 94 | 98 | 90 | 81 | 68 |
| naive FastAPI + ONNX Runtime | 33 | 76 | 124 | 147 | 149 | 136 | 125 |
| naive torch, `async def` | 32 | 64 | 70 | 69 | 67 | 64 | 55 |
| downshift 0.4.0 | 84 | 94 | 100 | 102 | 97 | 93 | 88 |
| downshift 0.4.0 base64 | 102 | 110 | 116 | 118 | 115 | 110 | 106 |
| downshift 0.5.0 | 86 | 105 | 106 | 104 | 103 | 99 | 91 |
| downshift 0.5.0 base64 | 105 | 125 | 123 | 124 | 121 | 118 | 109 |
| downshift 0.5.0 safetensors | 109 | 128 | 127 | 126 | 125 | 119 | 112 |

### Batched requests at concurrency 8 (rows/s)

Requests/s times batch size: the work actually done.

| server | `clean_mlp` b8 | `clean_mlp` b32 | `dynamic_batch_cnn` b8 | `dynamic_batch_cnn` b32 | `gnn_gcn` b8 | `gnn_gcn` b32 | `tiny_bert` b8 | `tiny_bert` b32 | `scatter_include_self_false` b8 | `scatter_include_self_false` b32 | `mlp_large` b8 | `mlp_large` b32 | `cnn_large` b8 | `cnn_large` b32 | `bert_small` b8 | `bert_small` b32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 8127 | 31716 | 1481 | 1653 | 3653 | 9375 | 2670 | 8822 | 8129 | 17206 | 1634 | 2092 | 403 | 392 | 134 | 93 |
| naive FastAPI + ONNX Runtime | 11135 | 40536 | 1676 | 1743 | 8717 | 17947 | 8823 | 17968 | 9703 | 18614 | 2007 | 2389 | 476 | 407 | 145 | 134 |
| downshift 0.4.0 | 6562 | 29513 | 3198 | 4361 | 6207 | 16426 | 5451 | 13559 | 8194 | 24898 | 3493 | 5344 | 1095 | 1157 | 120 | 99 |
| downshift 0.4.0 base64 | 6597 | 25174 | 6983 | 21651 | 7233 | 28815 | 5616 | 17520 | 7184 | 33584 | 5562 | 11162 | 2755 | 3490 | 164 | 129 |
| downshift 0.5.0 | 7372 | 34787 | 3743 | 5204 | 7279 | 20466 | 6366 | 15814 | 8990 | 29022 | 3976 | 6590 | 1442 | 1525 | 128 | 98 |
| downshift 0.5.0 base64 | 7729 | 31360 | 8219 | 24629 | 7679 | 26816 | 6556 | 21527 | 7739 | 36624 | 6300 | 14322 | 3490 | 3972 | 166 | 116 |
| downshift 0.5.0 safetensors | 7116 | 29693 | 8335 | 30956 | 8281 | 30977 | 6529 | 21486 | 7578 | 35957 | 6580 | 14372 | 3507 | 4092 | 172 | 131 |
| downshift 0.5.0 `--execution inline` | 10739 | 47074 | 3754 | 5087 | 9651 | 25204 | 5235 | 12694 | 11125 | 32977 | — | — | — | — | — | — |

### Wire encodings, batch 32, concurrency 8

| model | server | request bytes | requests/s | p50 ms |
|---|---|---|---|---|
| `clean_mlp` | downshift 0.4.0 | 10659 | 922 | 9.51 |
| `clean_mlp` | downshift 0.4.0 base64 | 2831 | 787 | 9.89 |
| `clean_mlp` | downshift 0.5.0 | 10659 | 1087 | 8.07 |
| `clean_mlp` | downshift 0.5.0 base64 | 2831 | 980 | 8.04 |
| `clean_mlp` | downshift 0.5.0 safetensors | 2120 | 928 | 8.52 |
| `dynamic_batch_cnn` | downshift 0.4.0 | 510377 | 136 | 57.11 |
| `dynamic_batch_cnn` | downshift 0.4.0 base64 | 131178 | 677 | 13.25 |
| `dynamic_batch_cnn` | downshift 0.5.0 | 510377 | 163 | 51.76 |
| `dynamic_batch_cnn` | downshift 0.5.0 base64 | 131178 | 770 | 11.37 |
| `dynamic_batch_cnn` | downshift 0.5.0 safetensors | 98384 | 967 | 9.04 |
| `gnn_gcn` | downshift 0.4.0 | 34970 | 513 | 15.67 |
| `gnn_gcn` | downshift 0.4.0 base64 | 15184 | 900 | 9.52 |
| `gnn_gcn` | downshift 0.5.0 | 34970 | 640 | 12.21 |
| `gnn_gcn` | downshift 0.5.0 base64 | 15184 | 838 | 10.50 |
| `gnn_gcn` | downshift 0.5.0 safetensors | 11408 | 968 | 9.11 |
| `tiny_bert` | downshift 0.4.0 | 1941 | 424 | 19.07 |
| `tiny_bert` | downshift 0.4.0 base64 | 5636 | 548 | 15.80 |
| `tiny_bert` | downshift 0.5.0 | 1941 | 494 | 15.92 |
| `tiny_bert` | downshift 0.5.0 base64 | 5636 | 673 | 13.78 |
| `tiny_bert` | downshift 0.5.0 safetensors | 4248 | 671 | 13.70 |
| `scatter_include_self_false` | downshift 0.4.0 | 32706 | 778 | 10.78 |
| `scatter_include_self_false` | downshift 0.4.0 base64 | 10402 | 1050 | 8.16 |
| `scatter_include_self_false` | downshift 0.5.0 | 32706 | 907 | 9.84 |
| `scatter_include_self_false` | downshift 0.5.0 base64 | 10402 | 1144 | 7.77 |
| `scatter_include_self_false` | downshift 0.5.0 safetensors | 7824 | 1124 | 7.88 |
| `mlp_large` | downshift 0.4.0 | 338169 | 167 | 53.80 |
| `mlp_large` | downshift 0.4.0 base64 | 87484 | 349 | 24.87 |
| `mlp_large` | downshift 0.5.0 | 338169 | 206 | 44.28 |
| `mlp_large` | downshift 0.5.0 base64 | 87484 | 448 | 21.06 |
| `mlp_large` | downshift 0.5.0 safetensors | 65608 | 449 | 20.83 |
| `cnn_large` | downshift 0.4.0 | 2034322 | 36 | 213.82 |
| `cnn_large` | downshift 0.4.0 base64 | 524394 | 109 | 81.48 |
| `cnn_large` | downshift 0.5.0 | 2034322 | 48 | 177.77 |
| `cnn_large` | downshift 0.5.0 base64 | 524394 | 124 | 72.72 |
| `cnn_large` | downshift 0.5.0 safetensors | 393296 | 128 | 72.83 |
| `bert_small` | downshift 0.4.0 | 39700 | 3 | 2056.30 |
| `bert_small` | downshift 0.4.0 base64 | 87560 | 4 | 1400.24 |
| `bert_small` | downshift 0.5.0 | 39700 | 3 | 2003.73 |
| `bert_small` | downshift 0.5.0 base64 | 87560 | 4 | 1580.50 |
| `bert_small` | downshift 0.5.0 safetensors | 65696 | 4 | 1546.37 |

### `--workers` sweep, batch 1 (requests/s)

| model | downshift | workers | c=1 | c=8 | c=32 | boot s |
|---|---|---|---|---|---|---|
| `bert_small` | 0.4.0 | 1 | 89 | 105 | 98 | 20.2 |
| `bert_small` | 0.4.0 | 2 | 90 | 142 | 143 | 23.1 |
| `bert_small` | 0.4.0 | 4 | 66 | 227 | 231 | 24.1 |
| `bert_small` | 0.5.0 | 1 | 91 | 108 | 101 | 20.3 |
| `bert_small` | 0.5.0 | 2 | 92 | 157 | 150 | 23.1 |
| `bert_small` | 0.5.0 | 4 | 70 | 239 | 248 | 23.6 |
| `clean_mlp` | 0.4.0 | 1 | 861 | 887 | 915 | 7.7 |
| `clean_mlp` | 0.4.0 | 2 | 823 | 1789 | 1923 | 10.6 |
| `clean_mlp` | 0.4.0 | 4 | 846 | 3406 | 3418 | 10.6 |
| `clean_mlp` | 0.5.0 | 1 | 907 | 983 | 1111 | 9.2 |
| `clean_mlp` | 0.5.0 | 2 | 936 | 2076 | 1964 | 11.3 |
| `clean_mlp` | 0.5.0 | 4 | 974 | 3857 | 3845 | 12.3 |
| `mlp_large` | 0.4.0 | 1 | 468 | 745 | 719 | 7.9 |
| `mlp_large` | 0.4.0 | 2 | 418 | 1126 | 1260 | 10.9 |
| `mlp_large` | 0.4.0 | 4 | 475 | 1325 | 2003 | 11.3 |
| `mlp_large` | 0.5.0 | 1 | 515 | 819 | 827 | 8.2 |
| `mlp_large` | 0.5.0 | 2 | 514 | 1006 | 1203 | 11.3 |
| `mlp_large` | 0.5.0 | 4 | 525 | 1459 | 2089 | 11.3 |

### Correctness

Worst max absolute error against eager PyTorch over every row a server answered.

| server | `clean_mlp` | `dynamic_batch_cnn` | `gnn_gcn` | `tiny_bert` | `scatter_include_self_false` | `mlp_large` | `cnn_large` | `bert_small` |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| naive FastAPI + ONNX Runtime | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 1.2e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| naive torch, `async def` | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| downshift 0.4.0 | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | 2.6e-07 | 5.9e-08 | 2.7e-06 |
| downshift 0.4.0 base64 | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 0.0e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| downshift 0.5.0 | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | 2.6e-07 | 5.9e-08 | 2.7e-06 |
| downshift 0.5.0 base64 | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 0.0e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| downshift 0.5.0 safetensors | 6.0e-08 | 3.0e-08 | 2.4e-07 | 4.8e-07 | 0.0e+00 | 2.5e-07 | 6.0e-08 | 2.6e-06 |
| downshift 0.5.0 `--execution inline` | 6.4e-08 | 3.4e-08 | 2.4e-07 | 5.8e-07 | 2.8e-08 | — | — | — |

Failed requests across every load window: 0.

This run was split in two: the first pass was stopped by low system memory during bert_small; in the second pass downshift 0.4.0 --workers 4 on bert_small hung at startup (no /ready within 600 s; it booted in 25 s in every other run) and was re-run on its own, so bert_small HTTP rows and stage splits, and the --workers sweep come from a second pass (2026-10-05T00:14:55).

## Real Hugging Face models

Text in, through the real CLI. 5.0 s windows at concurrency 1, 8; a `/health` probe runs every 50 ms during each window (what a liveness probe sees). Payloads: `short_b1` one sentence, `short_b8` eight sentences, `long_b1` ~180 tokens.

### all-MiniLM-L6-v2

`downshift check` wall time: 0.5.0 23.0 s (exit 0), 0.4.0 22.5 s (exit 0).

| server | ready s | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | worst /health p99 ms |
|---|---|---|---|---|---|
| naive FastAPI + transformers | 12.8 | 139 | 89 | 74 | 18 |
| naive transformers, `async def` | 12.8 | 117 | 47 | 36 | 436 |
| downshift 0.4.0 | 20.9 | 239 | 52 | 30 | 21 |
| downshift 0.4.0 `--backend torch` | 13.4 | 107 | 46 | 36 | 23 |
| downshift 0.4.0 `--max-concurrency 4` | 20.9 | 434 | 130 | 77 | 21 |
| downshift 0.5.0 | 21.6 | 282 | 56 | 32 | 18 |
| downshift 0.5.0 `--backend torch` | 13.5 | 109 | 47 | 37 | 18 |
| downshift 0.5.0 `--max-concurrency 4` | 21.5 | 483 | 137 | 78 | 27 |
| downshift 0.5.0 `--execution inline` | 21.4 | 281 | 56 | 32 | 18 |

p50 latency (ms) at concurrency 1:

| server | short_b1 | short_b8 | long_b1 |
|---|---|---|---|
| naive FastAPI + transformers | 9.4 | 21.8 | 27.8 |
| naive transformers, `async def` | 8.7 | 21.2 | 27.3 |
| downshift 0.4.0 | 5.3 | 19.7 | 33.1 |
| downshift 0.4.0 `--backend torch` | 9.8 | 21.6 | 27.9 |
| downshift 0.4.0 `--max-concurrency 4` | 5.3 | 19.7 | 33.4 |
| downshift 0.5.0 | 5.1 | 19.6 | 33.2 |
| downshift 0.5.0 `--backend torch` | 9.5 | 21.7 | 27.9 |
| downshift 0.5.0 `--max-concurrency 4` | 5.0 | 19.6 | 33.0 |
| downshift 0.5.0 `--execution inline` | 5.1 | 19.6 | 33.3 |

Agreement with downshift 0.5.0 `--backend torch` on `short_b8`:

| server | max abs err | min cosine |
|---|---|---|
| naive FastAPI + transformers | 7.2e-09 | 1.0000000 |
| naive transformers, `async def` | 7.2e-09 | 1.0000000 |
| downshift 0.4.0 | 1.3e-07 | 1.0000000 |
| downshift 0.4.0 `--backend torch` | 0.0e+00 | 1.0000000 |
| downshift 0.4.0 `--max-concurrency 4` | 1.3e-07 | 1.0000000 |
| downshift 0.5.0 | 1.3e-07 | 1.0000000 |
| downshift 0.5.0 `--backend torch` | 0.0e+00 | 1.0000000 |
| downshift 0.5.0 `--max-concurrency 4` | 1.3e-07 | 1.0000000 |
| downshift 0.5.0 `--execution inline` | 1.3e-07 | 1.0000000 |

### prompt-guard-86m

`downshift check` wall time: 0.5.0 43.5 s (exit 0), 0.4.0 42.0 s (exit 0).

| server | ready s | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | worst /health p99 ms |
|---|---|---|---|---|---|
| naive FastAPI + transformers | 15.8 | 16 | 7 | 6 | 22 |
| naive transformers, `async def` | 15.8 | 9 | 4 | 2 | 5213 |
| downshift 0.4.0 | 43.4 | 48 | 6 | 4 | 26 |
| downshift 0.4.0 `--backend torch` | 16.3 | 8 | 4 | 3 | 17 |
| downshift 0.4.0 `--max-concurrency 4` | 43.2 | 101 | 11 | 7 | 20 |
| downshift 0.5.0 | 43.8 | 50 | 6 | 4 | 18 |
| downshift 0.5.0 `--backend torch` | 16.2 | 8 | 4 | 3 | 24 |
| downshift 0.5.0 `--max-concurrency 4` | 43.6 | 103 | 12 | 7 | 26 |

p50 latency (ms) at concurrency 1:

| server | short_b1 | short_b8 | long_b1 |
|---|---|---|---|
| naive FastAPI + transformers | 107.9 | 228.5 | 296.6 |
| naive transformers, `async def` | 107.4 | 227.9 | 293.6 |
| downshift 0.4.0 | 21.3 | 140.9 | 214.7 |
| downshift 0.4.0 `--backend torch` | 109.2 | 231.3 | 296.4 |
| downshift 0.4.0 `--max-concurrency 4` | 21.3 | 141.1 | 215.4 |
| downshift 0.5.0 | 21.1 | 140.7 | 215.2 |
| downshift 0.5.0 `--backend torch` | 108.6 | 230.8 | 295.1 |
| downshift 0.5.0 `--max-concurrency 4` | 21.2 | 140.8 | 214.6 |

Agreement with downshift 0.5.0 `--backend torch` on `short_b8`:

| server | max abs err | label agreement |
|---|---|---|
| naive FastAPI + transformers | 2.2e-07 | 8/8 |
| naive transformers, `async def` | 2.2e-07 | 8/8 |
| downshift 0.4.0 | 2.3e-05 | 8/8 |
| downshift 0.4.0 `--backend torch` | 0.0e+00 | 8/8 |
| downshift 0.4.0 `--max-concurrency 4` | 2.3e-05 | 8/8 |
| downshift 0.5.0 | 2.3e-05 | 8/8 |
| downshift 0.5.0 `--backend torch` | 0.0e+00 | 8/8 |
| downshift 0.5.0 `--max-concurrency 4` | 2.3e-05 | 8/8 |

## Where the time goes

Median of 300 sequential requests on one keep-alive connection, batch 1 (HF: one sentence). `total` is what the client measured; the stage columns are the server's own `Server-Timing`; `HTTP + hand-offs` is the rest: uvicorn, routing, middleware and thread hand-offs back to the event loop. Naive servers have no `Server-Timing`, so their total is all in the last column. 0.4.0 reports one `codec` stage (decode and encode); 0.5.0 splits it, and its `encode` includes serializing the response.

### clean_mlp

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.77 | — | — | — | — | — | — | — | 0.77 |
| naive FastAPI + ONNX Runtime | 0.69 | — | — | — | — | — | — | — | 0.69 |
| naive torch, async def | 0.54 | — | — | — | — | — | — | — | 0.54 |
| downshift 0.4.0 | 1.27 | 0.16 | 0.05 | — | — | — | 0.22 | — | 0.84 |
| downshift 0.4.0 base64 | 1.16 | 0.14 | 0.06 | — | — | — | 0.21 | — | 0.75 |
| downshift 0.5.0 | 1.00 | 0.02 | — | 0.04 | 0.02 | 0.04 | 0.19 | 0.03 | 0.63 |
| downshift 0.5.0 base64 | 1.03 | 0.02 | — | 0.04 | 0.04 | 0.04 | 0.18 | 0.03 | 0.66 |
| downshift 0.5.0 safetensors | 1.06 | 0.04 | — | 0.04 | 0.02 | 0.05 | 0.19 | 0.04 | 0.66 |
| downshift 0.5.0 --execution inline | 0.74 | 0.02 | — | 0.00 | 0.02 | 0.00 | 0.19 | 0.03 | 0.46 |

### dynamic_batch_cnn

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1.43 | — | — | — | — | — | — | — | 1.43 |
| naive FastAPI + ONNX Runtime | 1.25 | — | — | — | — | — | — | — | 1.25 |
| naive torch, async def | 1.11 | — | — | — | — | — | — | — | 1.11 |
| downshift 0.4.0 | 1.56 | 0.23 | 0.12 | — | — | — | 0.29 | — | 0.93 |
| downshift 0.4.0 base64 | 1.40 | 0.16 | 0.08 | — | — | — | 0.30 | — | 0.86 |
| downshift 0.5.0 | 1.24 | 0.07 | — | 0.04 | 0.10 | 0.05 | 0.26 | 0.04 | 0.69 |
| downshift 0.5.0 base64 | 1.17 | 0.03 | — | 0.04 | 0.05 | 0.05 | 0.30 | 0.04 | 0.69 |
| downshift 0.5.0 safetensors | 1.16 | 0.05 | — | 0.05 | 0.02 | 0.05 | 0.24 | 0.04 | 0.69 |
| downshift 0.5.0 --execution inline | 0.98 | 0.06 | — | 0.00 | 0.08 | 0.00 | 0.25 | 0.04 | 0.56 |

### gnn_gcn

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1.75 | — | — | — | — | — | — | — | 1.75 |
| naive FastAPI + ONNX Runtime | 0.93 | — | — | — | — | — | — | — | 0.93 |
| naive torch, async def | 1.31 | — | — | — | — | — | — | — | 1.31 |
| downshift 0.4.0 | 1.41 | 0.16 | 0.06 | — | — | — | 0.33 | — | 0.85 |
| downshift 0.4.0 base64 | 1.42 | 0.16 | 0.08 | — | — | — | 0.32 | — | 0.87 |
| downshift 0.5.0 | 1.74 | 0.04 | — | 0.06 | 0.07 | 0.05 | 0.46 | 0.06 | 0.97 |
| downshift 0.5.0 base64 | 1.35 | 0.03 | — | 0.05 | 0.08 | 0.05 | 0.36 | 0.04 | 0.72 |
| downshift 0.5.0 safetensors | 1.25 | 0.06 | — | 0.04 | 0.04 | 0.05 | 0.30 | 0.04 | 0.70 |
| downshift 0.5.0 --execution inline | 0.76 | 0.02 | — | 0.00 | 0.03 | 0.00 | 0.24 | 0.03 | 0.42 |

### tiny_bert

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 2.22 | — | — | — | — | — | — | — | 2.22 |
| naive FastAPI + ONNX Runtime | 1.12 | — | — | — | — | — | — | — | 1.12 |
| naive torch, async def | 1.61 | — | — | — | — | — | — | — | 1.61 |
| downshift 0.4.0 | 1.60 | 0.15 | 0.05 | — | — | — | 0.57 | — | 0.85 |
| downshift 0.4.0 base64 | 1.66 | 0.16 | 0.09 | — | — | — | 0.56 | — | 0.86 |
| downshift 0.5.0 | 1.41 | 0.02 | — | 0.05 | 0.03 | 0.05 | 0.51 | 0.05 | 0.69 |
| downshift 0.5.0 base64 | 1.47 | 0.02 | — | 0.05 | 0.06 | 0.05 | 0.55 | 0.04 | 0.69 |
| downshift 0.5.0 safetensors | 1.41 | 0.06 | — | 0.05 | 0.03 | 0.05 | 0.45 | 0.04 | 0.68 |
| downshift 0.5.0 --execution inline | 1.07 | 0.02 | — | 0.00 | 0.03 | 0.00 | 0.47 | 0.04 | 0.52 |

### scatter_include_self_false

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.84 | — | — | — | — | — | — | — | 0.84 |
| naive FastAPI + ONNX Runtime | 0.85 | — | — | — | — | — | — | — | 0.85 |
| naive torch, async def | 0.51 | — | — | — | — | — | — | — | 0.51 |
| downshift 0.4.0 | 1.07 | 0.14 | 0.05 | — | — | — | 0.13 | — | 0.74 |
| downshift 0.4.0 base64 | 1.13 | 0.15 | 0.06 | — | — | — | 0.14 | — | 0.76 |
| downshift 0.5.0 | 0.96 | 0.02 | — | 0.04 | 0.04 | 0.04 | 0.15 | 0.03 | 0.62 |
| downshift 0.5.0 base64 | 0.93 | 0.02 | — | 0.04 | 0.05 | 0.04 | 0.14 | 0.03 | 0.61 |
| downshift 0.5.0 safetensors | 0.98 | 0.05 | — | 0.04 | 0.03 | 0.04 | 0.15 | 0.03 | 0.63 |
| downshift 0.5.0 --execution inline | 0.66 | 0.02 | — | 0.00 | 0.03 | 0.00 | 0.13 | 0.03 | 0.44 |

### mlp_large

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 2.35 | — | — | — | — | — | — | — | 2.35 |
| naive FastAPI + ONNX Runtime | 1.70 | — | — | — | — | — | — | — | 1.70 |
| naive torch, async def | 2.06 | — | — | — | — | — | — | — | 2.06 |
| downshift 0.4.0 | 2.13 | 0.21 | 0.10 | — | — | — | 0.74 | — | 1.07 |
| downshift 0.4.0 base64 | 2.01 | 0.17 | 0.09 | — | — | — | 0.72 | — | 1.03 |
| downshift 0.5.0 | 1.89 | 0.08 | — | 0.06 | 0.08 | 0.05 | 0.71 | 0.08 | 0.86 |
| downshift 0.5.0 base64 | 1.81 | 0.04 | — | 0.05 | 0.06 | 0.05 | 0.70 | 0.05 | 0.86 |
| downshift 0.5.0 safetensors | 1.80 | 0.07 | — | 0.05 | 0.04 | 0.05 | 0.71 | 0.06 | 0.84 |

### cnn_large

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 3.64 | — | — | — | — | — | — | — | 3.64 |
| naive FastAPI + ONNX Runtime | 2.79 | — | — | — | — | — | — | — | 2.79 |
| naive torch, async def | 3.40 | — | — | — | — | — | — | — | 3.40 |
| downshift 0.4.0 | 2.17 | 0.39 | 0.31 | — | — | — | 0.61 | — | 0.92 |
| downshift 0.4.0 base64 | 1.66 | 0.16 | 0.08 | — | — | — | 0.59 | — | 0.83 |
| downshift 0.5.0 | 1.97 | 0.23 | — | 0.05 | 0.29 | 0.05 | 0.57 | 0.04 | 0.78 |
| downshift 0.5.0 base64 | 1.49 | 0.04 | — | 0.05 | 0.06 | 0.05 | 0.59 | 0.04 | 0.69 |
| downshift 0.5.0 safetensors | 1.47 | 0.05 | — | 0.05 | 0.02 | 0.05 | 0.57 | 0.05 | 0.68 |

### bert_small

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 13.91 | — | — | — | — | — | — | — | 13.91 |
| naive FastAPI + ONNX Runtime | 12.83 | — | — | — | — | — | — | — | 12.83 |
| naive torch, async def | 14.13 | — | — | — | — | — | — | — | 14.13 |
| downshift 0.4.0 | 10.75 | 0.21 | 0.10 | — | — | — | 7.35 | — | 3.09 |
| downshift 0.4.0 base64 | 9.21 | 0.21 | 0.17 | — | — | — | 7.32 | — | 1.53 |
| downshift 0.5.0 | 10.61 | 0.05 | — | 0.06 | 0.07 | 0.06 | 7.33 | 1.77 | 1.28 |
| downshift 0.5.0 base64 | 9.05 | 0.05 | — | 0.06 | 0.09 | 0.06 | 7.33 | 0.19 | 1.25 |
| downshift 0.5.0 safetensors | 8.78 | 0.09 | — | 0.06 | 0.05 | 0.06 | 7.32 | 0.08 | 1.16 |

### all-MiniLM-L6-v2

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 9.07 | — | — | — | — | — | — | — | 9.07 |
| naive transformers, async def | 8.51 | — | — | — | — | — | — | — | 8.51 |
| downshift 0.4.0 | 5.12 | 0.20 | 0.36 | — | — | — | 3.12 | — | 1.45 |
| downshift 0.4.0 --backend torch | 9.57 | 0.21 | 0.37 | — | — | — | 7.41 | — | 1.57 |
| downshift 0.4.0 --max-concurrency 4 | 5.09 | 0.21 | 0.38 | — | — | — | 3.11 | — | 1.43 |
| downshift 0.5.0 | 4.87 | 0.04 | — | 0.06 | 0.35 | 0.06 | 3.10 | 0.11 | 1.17 |
| downshift 0.5.0 --backend torch | 9.21 | 0.04 | — | 0.07 | 0.36 | 0.06 | 7.30 | 0.10 | 1.28 |
| downshift 0.5.0 --max-concurrency 4 | 4.86 | 0.04 | — | 0.06 | 0.35 | 0.06 | 3.10 | 0.11 | 1.17 |
| downshift 0.5.0 --execution inline | 4.94 | 0.03 | — | 0.06 | 0.35 | 0.06 | 3.15 | 0.11 | 1.19 |

### prompt-guard-86m

| server | total ms | parse | codec | prep_wait | prep | infer_wait | infer | encode | HTTP + hand-offs ms |
|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + transformers | 107.50 | — | — | — | — | — | — | — | 107.50 |
| naive transformers, async def | 106.72 | — | — | — | — | — | — | — | 106.72 |
| downshift 0.4.0 | 21.16 | 0.19 | 0.48 | — | — | — | 18.98 | — | 1.51 |
| downshift 0.4.0 --backend torch | 107.74 | 0.18 | 0.47 | — | — | — | 105.58 | — | 1.48 |
| downshift 0.4.0 --max-concurrency 4 | 21.07 | 0.20 | 0.47 | — | — | — | 18.91 | — | 1.50 |
| downshift 0.5.0 | 20.88 | 0.04 | — | 0.06 | 0.37 | 0.05 | 18.89 | 0.19 | 1.29 |
| downshift 0.5.0 --backend torch | 107.54 | 0.04 | — | 0.06 | 0.36 | 0.06 | 105.58 | 0.17 | 1.28 |
| downshift 0.5.0 --max-concurrency 4 | 20.93 | 0.04 | — | 0.06 | 0.38 | 0.05 | 18.95 | 0.19 | 1.26 |
