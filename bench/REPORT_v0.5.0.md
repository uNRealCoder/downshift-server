# downshift benchmark: naive servers vs downshift 0.4.0 vs 0.5.0

`win32`, 16 logical CPUs, Python 3.12.10, torch 2.14.0+cpu (CPU), onnxruntime 1.30.0, run 2026-10-04T15:05:08. downshift 0.5.0 from the checkout; downshift 0.4.0 from the pip-installed release. Every downshift server is the real `downshift serve` CLI; all servers ran in one session on the same machine.

## Fixture and compute-heavy models

Closed-loop load, 3.0 s window after 1.0 s warm-up, load generator sharded over up to 6 processes on the same machine. Every server gets the same seeded weights and byte-identical request bodies, and every row's response is checked element-wise against eager PyTorch.

Load-generator ceiling (`GET /health` on a do-nothing server): c=1: 4384 rps, c=2: 7750 rps, c=4: 8197 rps, c=8: 8826 rps, c=16: 8997 rps, c=32: 8767 rps, c=64: 9577 rps. A result near these numbers is measuring the client.

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
| naive FastAPI + torch | 1409 | 908 | 555 | 427 | 1283 | 795 | 359 | 96 |
| naive FastAPI + ONNX Runtime | 1744 | 1068 | 1526 | 1567 | 1621 | 884 | 441 | 149 |
| naive torch, `async def` | 2690 | 1045 | 724 | 559 | 2112 | 572 | 308 | 71 |
| downshift 0.4.0 | 914 | 910 | 833 | 822 | 945 | 698 | 652 | 102 |
| downshift 0.4.0 base64 | 903 | 1018 | 836 | 795 | 936 | 606 | 910 | 119 |
| downshift 0.5.0 | 721 | 687 | 659 | 652 | 653 | 553 | 533 | 128 |
| downshift 0.5.0 base64 | 663 | 777 | 650 | 625 | 625 | 486 | 743 | 129 |
| downshift 0.5.0 safetensors | 695 | 779 | 637 | 633 | 636 | 485 | 743 | 130 |
| downshift 0.5.0 `--execution inline` | 1179 | 1046 | 974 | 779 | 1145 | — | — | — |

p50 latency (ms) at concurrency 1:

| server | `clean_mlp` | `dynamic_batch_cnn` | `gnn_gcn` | `tiny_bert` | `scatter_include_self_false` | `mlp_large` | `cnn_large` | `bert_small` |
|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 0.80 | 1.39 | 1.74 | 2.29 | 0.89 | 2.53 | 3.76 | 31.10 |
| naive FastAPI + ONNX Runtime | 0.75 | 1.24 | 0.89 | 1.17 | 0.81 | 1.85 | 2.91 | 31.06 |
| naive torch, `async def` | 0.49 | 1.10 | 1.34 | 1.72 | 0.56 | 2.05 | 3.50 | 31.12 |
| downshift 0.4.0 | 1.18 | 1.46 | 1.33 | 1.65 | 1.17 | 2.24 | 2.24 | 11.31 |
| downshift 0.4.0 base64 | 1.17 | 1.33 | 1.32 | 1.69 | 1.17 | 2.13 | 1.73 | 9.46 |
| downshift 0.5.0 | 1.55 | 1.90 | 1.69 | 2.09 | 1.55 | 2.73 | 2.60 | 11.79 |
| downshift 0.5.0 base64 | 1.59 | 1.79 | 1.76 | 2.12 | 1.58 | 2.62 | 2.18 | 9.88 |
| downshift 0.5.0 safetensors | 1.64 | 1.78 | 1.75 | 2.11 | 1.58 | 2.57 | 2.14 | 9.64 |
| downshift 0.5.0 `--execution inline` | 1.02 | 1.24 | 1.20 | 1.39 | 1.00 | — | — | — |

### Concurrency sweep, batch 1 (requests/s)

`clean_mlp`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1162 | 1409 | 1366 | 1299 | 1381 | 1330 | 1257 |
| naive FastAPI + ONNX Runtime | 1250 | 1479 | 1729 | 1744 | 1717 | 1593 | 1608 |
| naive torch, `async def` | 1894 | 2410 | 2469 | 2593 | 2262 | 2690 | 2536 |
| downshift 0.4.0 | 798 | 914 | 908 | 854 | 878 | 883 | 816 |
| downshift 0.4.0 base64 | 800 | 903 | 867 | 856 | 869 | 857 | 811 |
| downshift 0.5.0 | 601 | 656 | 641 | 659 | 663 | 721 | 688 |
| downshift 0.5.0 base64 | 581 | 661 | 617 | 630 | 663 | 653 | 656 |
| downshift 0.5.0 safetensors | 566 | 642 | 614 | 638 | 653 | 664 | 695 |
| downshift 0.5.0 `--execution inline` | 936 | 1078 | 1179 | 1164 | 1065 | 1029 | 979 |

`dynamic_batch_cnn`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 691 | 791 | 908 | 863 | 823 | 763 | 703 |
| naive FastAPI + ONNX Runtime | 773 | 892 | 1068 | 1030 | 1012 | 951 | 853 |
| naive torch, `async def` | 869 | 909 | 1022 | 1027 | 1030 | 1045 | 1018 |
| downshift 0.4.0 | 659 | 806 | 895 | 910 | 855 | 867 | 822 |
| downshift 0.4.0 base64 | 719 | 878 | 910 | 1018 | 1013 | 995 | 984 |
| downshift 0.5.0 | 505 | 620 | 676 | 671 | 687 | 642 | 653 |
| downshift 0.5.0 base64 | 534 | 667 | 732 | 738 | 755 | 777 | 775 |
| downshift 0.5.0 safetensors | 538 | 670 | 743 | 718 | 759 | 779 | 728 |
| downshift 0.5.0 `--execution inline` | 779 | 866 | 1046 | 997 | 880 | 860 | 788 |

`gnn_gcn`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 555 | 538 | 503 | 487 | 429 | 386 | 366 |
| naive FastAPI + ONNX Runtime | 1056 | 1313 | 1490 | 1490 | 1526 | 1357 | 1372 |
| naive torch, `async def` | 720 | 707 | 696 | 720 | 719 | 724 | 718 |
| downshift 0.4.0 | 702 | 833 | 819 | 813 | 793 | 803 | 726 |
| downshift 0.4.0 base64 | 705 | 836 | 801 | 788 | 780 | 780 | 707 |
| downshift 0.5.0 | 550 | 630 | 624 | 607 | 639 | 659 | 604 |
| downshift 0.5.0 base64 | 529 | 604 | 600 | 593 | 592 | 650 | 590 |
| downshift 0.5.0 safetensors | 536 | 627 | 613 | 594 | 621 | 637 | 590 |
| downshift 0.5.0 `--execution inline` | 803 | 905 | 974 | 931 | 906 | 819 | 818 |

`tiny_bert`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 425 | 427 | 408 | 376 | 360 | 328 | 300 |
| naive FastAPI + ONNX Runtime | 821 | 1283 | 1487 | 1499 | 1506 | 1567 | 1347 |
| naive torch, `async def` | 559 | 519 | 528 | 525 | 526 | 531 | 518 |
| downshift 0.4.0 | 591 | 741 | 806 | 822 | 793 | 789 | 698 |
| downshift 0.4.0 base64 | 575 | 712 | 785 | 795 | 786 | 766 | 693 |
| downshift 0.5.0 | 461 | 606 | 599 | 587 | 613 | 636 | 652 |
| downshift 0.5.0 base64 | 456 | 590 | 578 | 573 | 593 | 611 | 625 |
| downshift 0.5.0 safetensors | 458 | 591 | 583 | 574 | 591 | 610 | 633 |
| downshift 0.5.0 `--execution inline` | 686 | 779 | 745 | 734 | 677 | 684 | 633 |

`scatter_include_self_false`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 1059 | 1283 | 1261 | 1230 | 1219 | 1234 | 1098 |
| naive FastAPI + ONNX Runtime | 1156 | 1422 | 1579 | 1478 | 1621 | 1528 | 1347 |
| naive torch, `async def` | 1666 | 1912 | 1984 | 2037 | 2112 | 2080 | 2017 |
| downshift 0.4.0 | 804 | 900 | 905 | 945 | 925 | 917 | 811 |
| downshift 0.4.0 base64 | 804 | 902 | 905 | 936 | 915 | 899 | 841 |
| downshift 0.5.0 | 603 | 653 | 617 | 614 | 601 | 619 | 590 |
| downshift 0.5.0 base64 | 594 | 625 | 612 | 594 | 595 | 597 | 595 |
| downshift 0.5.0 safetensors | 595 | 636 | 604 | 585 | 598 | 599 | 577 |
| downshift 0.5.0 `--execution inline` | 940 | 1096 | 1145 | 1118 | 1045 | 1039 | 1000 |

`mlp_large`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 359 | 427 | 665 | 716 | 795 | 791 | 730 |
| naive FastAPI + ONNX Runtime | 514 | 628 | 849 | 884 | 869 | 865 | 824 |
| naive torch, `async def` | 448 | 490 | 563 | 567 | 570 | 572 | 566 |
| downshift 0.4.0 | 422 | 542 | 665 | 697 | 698 | 693 | 605 |
| downshift 0.4.0 base64 | 449 | 526 | 597 | 606 | 599 | 596 | 537 |
| downshift 0.5.0 | 349 | 466 | 535 | 546 | 551 | 548 | 553 |
| downshift 0.5.0 base64 | 359 | 460 | 461 | 457 | 468 | 474 | 486 |
| downshift 0.5.0 safetensors | 367 | 465 | 460 | 466 | 465 | 485 | 483 |

`cnn_large`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 259 | 308 | 359 | 343 | 330 | 305 | 300 |
| naive FastAPI + ONNX Runtime | 335 | 411 | 441 | 422 | 399 | 379 | 332 |
| naive torch, `async def` | 277 | 275 | 308 | 304 | 299 | 298 | 299 |
| downshift 0.4.0 | 437 | 533 | 640 | 652 | 603 | 583 | 539 |
| downshift 0.4.0 base64 | 562 | 710 | 875 | 910 | 894 | 886 | 810 |
| downshift 0.5.0 | 372 | 486 | 533 | 527 | 533 | 481 | 457 |
| downshift 0.5.0 base64 | 443 | 613 | 687 | 703 | 715 | 731 | 743 |
| downshift 0.5.0 safetensors | 449 | 624 | 699 | 710 | 732 | 737 | 743 |

`bert_small`

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 33 | 69 | 95 | 96 | 89 | 79 | 67 |
| naive FastAPI + ONNX Runtime | 32 | 77 | 126 | 149 | 146 | 143 | 129 |
| naive torch, `async def` | 32 | 63 | 71 | 69 | 68 | 66 | 58 |
| downshift 0.4.0 | 85 | 95 | 102 | 101 | 98 | 94 | 87 |
| downshift 0.4.0 base64 | 104 | 110 | 118 | 119 | 119 | 111 | 102 |
| downshift 0.5.0 | 83 | 127 | 128 | 127 | 125 | 121 | 112 |
| downshift 0.5.0 base64 | 100 | 129 | 129 | 127 | 125 | 121 | 113 |
| downshift 0.5.0 safetensors | 102 | 130 | 129 | 128 | 125 | 122 | 113 |

### Batched requests at concurrency 8 (rows/s)

Requests/s times batch size: the work actually done.

| server | `clean_mlp` b8 | `clean_mlp` b32 | `dynamic_batch_cnn` b8 | `dynamic_batch_cnn` b32 | `gnn_gcn` b8 | `gnn_gcn` b32 | `tiny_bert` b8 | `tiny_bert` b32 | `scatter_include_self_false` b8 | `scatter_include_self_false` b32 | `mlp_large` b8 | `mlp_large` b32 | `cnn_large` b8 | `cnn_large` b32 | `bert_small` b8 | `bert_small` b32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive FastAPI + torch | 8495 | 30796 | 1524 | 1583 | 3702 | 10239 | 2667 | 8607 | 7994 | 16852 | 1650 | 2102 | 406 | 359 | 138 | 88 |
| naive FastAPI + ONNX Runtime | 11657 | 40480 | 1684 | 1733 | 10120 | 18935 | 8440 | 18727 | 11054 | 22394 | 2005 | 2373 | 459 | 425 | 163 | 117 |
| downshift 0.4.0 | 6596 | 29892 | 3249 | 4516 | 7091 | 20084 | 5442 | 13343 | 8003 | 23867 | 3473 | 5338 | 1098 | 1165 | 128 | 90 |
| downshift 0.4.0 base64 | 6812 | 26778 | 7768 | 22861 | 7292 | 28609 | 5710 | 18322 | 7256 | 33859 | 5545 | 11154 | 2750 | 3546 | 166 | 125 |
| downshift 0.5.0 | 5140 | 22793 | 2784 | 4316 | 5419 | 16971 | 4391 | 15256 | 5622 | 19077 | 2923 | 5572 | 1059 | 1327 | 173 | 122 |
| downshift 0.5.0 base64 | 4911 | 20502 | 5641 | 18380 | 5655 | 21261 | 4453 | 20512 | 4794 | 23044 | 4459 | 14076 | 3553 | 4047 | 169 | 131 |
| downshift 0.5.0 safetensors | 4935 | 20710 | 5837 | 22056 | 5817 | 21688 | 4480 | 20766 | 4414 | 23132 | 4598 | 14911 | 3634 | 4160 | 173 | 131 |
| downshift 0.5.0 `--execution inline` | 8806 | 38318 | 2758 | 4421 | 7900 | 20596 | 4635 | 11752 | 8677 | 24393 | — | — | — | — | — | — |

### Wire encodings, batch 32, concurrency 8

| model | server | request bytes | requests/s | p50 ms |
|---|---|---|---|---|
| `clean_mlp` | downshift 0.4.0 | 10659 | 934 | 9.27 |
| `clean_mlp` | downshift 0.4.0 base64 | 2831 | 837 | 9.32 |
| `clean_mlp` | downshift 0.5.0 | 10659 | 712 | 12.02 |
| `clean_mlp` | downshift 0.5.0 base64 | 2831 | 641 | 12.37 |
| `clean_mlp` | downshift 0.5.0 safetensors | 2120 | 647 | 12.08 |
| `dynamic_batch_cnn` | downshift 0.4.0 | 510377 | 141 | 55.57 |
| `dynamic_batch_cnn` | downshift 0.4.0 base64 | 131178 | 714 | 12.43 |
| `dynamic_batch_cnn` | downshift 0.5.0 | 510377 | 135 | 60.44 |
| `dynamic_batch_cnn` | downshift 0.5.0 base64 | 131178 | 574 | 15.38 |
| `dynamic_batch_cnn` | downshift 0.5.0 safetensors | 98384 | 689 | 12.93 |
| `gnn_gcn` | downshift 0.4.0 | 34970 | 628 | 12.47 |
| `gnn_gcn` | downshift 0.4.0 base64 | 15184 | 894 | 9.61 |
| `gnn_gcn` | downshift 0.5.0 | 34970 | 530 | 16.13 |
| `gnn_gcn` | downshift 0.5.0 base64 | 15184 | 664 | 12.98 |
| `gnn_gcn` | downshift 0.5.0 safetensors | 11408 | 678 | 13.05 |
| `tiny_bert` | downshift 0.4.0 | 1941 | 417 | 19.03 |
| `tiny_bert` | downshift 0.4.0 base64 | 5636 | 573 | 15.15 |
| `tiny_bert` | downshift 0.5.0 | 1941 | 477 | 16.54 |
| `tiny_bert` | downshift 0.5.0 base64 | 5636 | 641 | 13.95 |
| `tiny_bert` | downshift 0.5.0 safetensors | 4248 | 649 | 13.93 |
| `scatter_include_self_false` | downshift 0.4.0 | 32706 | 746 | 10.95 |
| `scatter_include_self_false` | downshift 0.4.0 base64 | 10402 | 1058 | 8.12 |
| `scatter_include_self_false` | downshift 0.5.0 | 32706 | 596 | 14.76 |
| `scatter_include_self_false` | downshift 0.5.0 base64 | 10402 | 720 | 12.21 |
| `scatter_include_self_false` | downshift 0.5.0 safetensors | 7824 | 723 | 12.26 |
| `mlp_large` | downshift 0.4.0 | 338169 | 167 | 54.38 |
| `mlp_large` | downshift 0.4.0 base64 | 87484 | 349 | 25.25 |
| `mlp_large` | downshift 0.5.0 | 338169 | 174 | 52.37 |
| `mlp_large` | downshift 0.5.0 base64 | 87484 | 440 | 20.66 |
| `mlp_large` | downshift 0.5.0 safetensors | 65608 | 466 | 19.90 |
| `cnn_large` | downshift 0.4.0 | 2034322 | 36 | 206.17 |
| `cnn_large` | downshift 0.4.0 base64 | 524394 | 111 | 81.25 |
| `cnn_large` | downshift 0.5.0 | 2034322 | 41 | 191.92 |
| `cnn_large` | downshift 0.5.0 base64 | 524394 | 126 | 71.58 |
| `cnn_large` | downshift 0.5.0 safetensors | 393296 | 130 | 71.66 |
| `bert_small` | downshift 0.4.0 | 39700 | 3 | 2020.47 |
| `bert_small` | downshift 0.4.0 base64 | 87560 | 4 | 1728.11 |
| `bert_small` | downshift 0.5.0 | 39700 | 4 | 1575.53 |
| `bert_small` | downshift 0.5.0 base64 | 87560 | 4 | 1544.64 |
| `bert_small` | downshift 0.5.0 safetensors | 65696 | 4 | 1534.74 |

### `--workers` sweep, batch 1 (requests/s)

| model | downshift | workers | c=1 | c=8 | c=32 | boot s |
|---|---|---|---|---|---|---|
| `bert_small` | 0.4.0 | 1 | 85 | 101 | 94 | 21.4 |
| `bert_small` | 0.4.0 | 2 | 88 | 133 | 132 | 24.1 |
| `bert_small` | 0.4.0 | 4 | 65 | 222 | 211 | 24.9 |
| `bert_small` | 0.5.0 | 1 | 84 | 127 | 123 | 21.2 |
| `bert_small` | 0.5.0 | 2 | 83 | 147 | 127 | 24.4 |
| `bert_small` | 0.5.0 | 4 | 63 | 23 | 223 | 24.8 |
| `clean_mlp` | 0.4.0 | 1 | 798 | 864 | 883 | 8.2 |
| `clean_mlp` | 0.4.0 | 2 | 780 | 1703 | 1651 | 11.3 |
| `clean_mlp` | 0.4.0 | 4 | 781 | 3043 | 3299 | 11.3 |
| `clean_mlp` | 0.5.0 | 1 | 600 | 661 | 720 | 8.7 |
| `clean_mlp` | 0.5.0 | 2 | 573 | 1258 | 1257 | 11.3 |
| `clean_mlp` | 0.5.0 | 4 | 572 | 2378 | 2360 | 11.8 |
| `mlp_large` | 0.4.0 | 1 | 428 | 710 | 675 | 8.7 |
| `mlp_large` | 0.4.0 | 2 | 426 | 909 | 1031 | 11.6 |
| `mlp_large` | 0.4.0 | 4 | 421 | 1407 | 1745 | 11.8 |
| `mlp_large` | 0.5.0 | 1 | 350 | 542 | 554 | 8.9 |
| `mlp_large` | 0.5.0 | 2 | 355 | 804 | 836 | 11.8 |
| `mlp_large` | 0.5.0 | 4 | 345 | 1181 | 1664 | 12.3 |

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

Failed requests across every load window: 1.

- `bert_small`, downshift 0.5.0 `--workers 4`, batch 1, c=8: 1 of 734 failed (`{'error': 'TimeoutError: '}`); that window lasted 32.2 s against 3.0 s, so its requests/s is understated.

This run was split in two: the first pass was stopped by low system memory during bert_small, so bert_small HTTP rows and the --workers sweep come from a second pass (2026-10-04T16:58:07).

## Real Hugging Face models

Text in, through the real CLI. 5.0 s windows at concurrency 1, 8; a `/health` probe runs every 50 ms during each window (what a liveness probe sees). Payloads: `short_b1` one sentence, `short_b8` eight sentences, `long_b1` ~180 tokens.

### all-MiniLM-L6-v2

`downshift check` wall time: 0.5.0 22.4 s (exit 0), 0.4.0 22.4 s (exit 0).

| server | ready s | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | worst /health p99 ms |
|---|---|---|---|---|---|
| naive FastAPI + transformers | 13.6 | 134 | 85 | 70 | 21 |
| naive transformers, `async def` | 14.1 | 111 | 45 | 35 | 484 |
| downshift 0.4.0 | 22.1 | 225 | 51 | 30 | 18 |
| downshift 0.4.0 `--backend torch` | 14.2 | 103 | 45 | 34 | 25 |
| downshift 0.4.0 `--max-concurrency 4` | 22.1 | 421 | 130 | 75 | 24 |
| downshift 0.5.0 | 22.5 | 269 | 56 | 31 | 20 |
| downshift 0.5.0 `--backend torch` | 14.5 | 97 | 44 | 35 | 23 |
| downshift 0.5.0 `--max-concurrency 4` | 22.4 | 358 | 138 | 77 | 28 |
| downshift 0.5.0 `--execution inline` | 22.6 | 272 | 55 | 31 | 22 |

p50 latency (ms) at concurrency 1:

| server | short_b1 | short_b8 | long_b1 |
|---|---|---|---|
| naive FastAPI + transformers | 9.5 | 22.3 | 28.8 |
| naive transformers, `async def` | 9.0 | 22.0 | 27.8 |
| downshift 0.4.0 | 5.4 | 20.0 | 33.9 |
| downshift 0.4.0 `--backend torch` | 10.0 | 22.5 | 28.6 |
| downshift 0.4.0 `--max-concurrency 4` | 5.5 | 20.0 | 33.9 |
| downshift 0.5.0 | 6.0 | 20.5 | 34.4 |
| downshift 0.5.0 `--backend torch` | 10.4 | 22.9 | 29.4 |
| downshift 0.5.0 `--max-concurrency 4` | 5.9 | 20.6 | 34.5 |
| downshift 0.5.0 `--execution inline` | 6.0 | 20.8 | 34.4 |

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

`downshift check` wall time: 0.5.0 44.8 s (exit 0), 0.4.0 44.9 s (exit 0).

| server | ready s | short_b1 rps c=8 | short_b8 rps c=8 | long_b1 rps c=8 | worst /health p99 ms |
|---|---|---|---|---|---|
| naive FastAPI + transformers | 16.8 | 15 | 8 | 6 | 22 |
| naive transformers, `async def` | 16.4 | 8 | 3 | 2 | 5473 |
| downshift 0.4.0 | 46.0 | 48 | 6 | 4 | 19 |
| downshift 0.4.0 `--backend torch` | 17.1 | 8 | 3 | 2 | 23 |
| downshift 0.4.0 `--max-concurrency 4` | 45.8 | 94 | 11 | 7 | 19 |
| downshift 0.5.0 | 45.2 | 50 | 6 | 4 | 27 |
| downshift 0.5.0 `--backend torch` | 17.1 | 8 | 3 | 2 | 23 |
| downshift 0.5.0 `--max-concurrency 4` | 45.4 | 95 | 11 | 7 | 25 |

p50 latency (ms) at concurrency 1:

| server | short_b1 | short_b8 | long_b1 |
|---|---|---|---|
| naive FastAPI + transformers | 112.9 | 237.2 | 307.4 |
| naive transformers, `async def` | 111.3 | 236.1 | 304.7 |
| downshift 0.4.0 | 21.7 | 145.5 | 220.2 |
| downshift 0.4.0 `--backend torch` | 113.4 | 239.2 | 307.6 |
| downshift 0.4.0 `--max-concurrency 4` | 21.8 | 145.3 | 220.0 |
| downshift 0.5.0 | 22.1 | 146.1 | 220.0 |
| downshift 0.5.0 `--backend torch` | 113.4 | 239.5 | 306.6 |
| downshift 0.5.0 `--max-concurrency 4` | 22.3 | 145.7 | 220.5 |

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
