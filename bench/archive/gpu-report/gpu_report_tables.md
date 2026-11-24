@@THR_host_b1
| model | eager (fp32) | torch TF32 | compile (fp32) | ORT fp32 | ORT TF32 |
|---|---|---|---|---|---|
| `clean_mlp` | 3,547 | 3,567 | 2,454 | 4,226 | 3,679 |
| `dynamic_batch_cnn` | 2,436 | 2,176 | 1,874 | 2,902 | 3,631 |
| `gnn_gcn` | 385 | 385 | 435 | 692 | 639 |
| `tiny_bert` | 337 | 299 | 588 | 595 | 424 |
| `mlp_large` | 2,430 | 2,144 | 1,900 | 2,874 | 2,604 |
| `cnn_large` | 1,216 | 1,110 | 1,031 | 1,819 | 1,650 |
| `bert_small` | 168 | 165 | 334 | 317 | 302 |

@@THR_host_b32
| model | eager (fp32) | torch TF32 | compile (fp32) | ORT fp32 | ORT TF32 |
|---|---|---|---|---|---|
| `clean_mlp` | 89,646 | 75,176 | 70,639 | 133,653 | 127,310 |
| `dynamic_batch_cnn` | 68,665 | 68,421 | 51,131 | 98,580 | 99,344 |
| `gnn_gcn` | 10,864 | 10,817 | 14,890 | 20,662 | 21,394 |
| `tiny_bert` | 8,473 | 8,825 | 18,762 | 15,296 | 15,472 |
| `mlp_large` | 62,661 | 65,658 | 52,828 | 41,240† | 81,581 |
| `cnn_large` | 13,849† | 9,150† | 14,426 | 13,308† | 20,839 |
| `bert_small` | 1,416† | 2,470† | 1,524† | 1,241 | 2,619† |

@@THR_host_b256
| model | eager (fp32) | torch TF32 | compile (fp32) | ORT fp32 | ORT TF32 |
|---|---|---|---|---|---|
| `clean_mlp` | 813,733 | 598,097 | 547,557 | 968,803 | 994,896 |
| `dynamic_batch_cnn` | 401,266 | 361,966 | 378,252 | 503,708 | 565,661 |
| `gnn_gcn` | 74,331 | 100,974 | 107,773 | 160,168 | 161,676 |
| `tiny_bert` | 75,387 | 83,603† | 133,223 | 129,367 | 142,279 |
| `mlp_large` | 96,531† | 199,967 | 101,826† | 75,326† | 189,435 |
| `cnn_large` | 14,824 | 7,947† | 17,812 | 14,765 | 18,632† |
| `bert_small` | 1,981† | 2,532 | 4,920 | 3,456 | 3,785 |

@@THR_resident_b256
| model | eager (fp32) | torch TF32 | compile (fp32) | ORT fp32 | ORT TF32 |
|---|---|---|---|---|---|
| `clean_mlp` | 1,152,621 | 934,231 | 739,791 | 1,509,491 | 1,444,271 |
| `dynamic_batch_cnn` | 657,133 | 664,740 | 513,675 | 1,113,353 | 1,214,470 |
| `gnn_gcn` | 92,024 | 84,940 | 81,868 | 146,178 | 153,843 |
| `tiny_bert` | 74,129 | 79,212 | 156,882 | 137,001 | 158,212 |
| `mlp_large` | 113,687† | 312,029 | 110,697† | 156,306 | 226,656† |
| `cnn_large` | 15,840† | 11,172† | 16,115† | 11,196 | 15,648† |
| `bert_small` | 5,157 | 6,315 | 5,530 | 3,622 | 4,084 |

@@VSCPU
| model | batch | best CPU | best GPU, fp32 | GPU / CPU | best GPU, TF32 allowed | GPU / CPU |
|---|---|---|---|---|---|---|
| `clean_mlp` | 1 | 76,725 | 4,226 (ort_strict) | 0.06x | 4,226 (ort_strict) | 0.06x |
| `clean_mlp` | 32 | 2,096,225 | 133,653 (ort_strict) | 0.06x | 133,653 (ort_strict) | 0.06x |
| `dynamic_batch_cnn` | 1 | 46,586 | 2,902 (ort_strict) | 0.06x | 3,631 (ort_tf32) | 0.08x |
| `dynamic_batch_cnn` | 32 | 519,274 | 98,580 (ort_strict) | 0.19x | 99,344 (ort_tf32) | 0.19x |
| `gnn_gcn` | 1 | 10,060 | 692 (ort_strict) | 0.07x | 692 (ort_strict) | 0.07x |
| `gnn_gcn` | 32 | 96,681 | 20,662 (ort_strict) | 0.21x | 21,394 (ort_tf32) | 0.22x |
| `tiny_bert` | 1 | 4,744 | 595 (ort_strict) | 0.13x | 595 (ort_strict) | 0.13x |
| `tiny_bert` | 32 | 37,135 | 18,762 (torch_compile) | 0.51x | 18,762 (torch_compile) | 0.51x |
| `mlp_large` | 1 | 2,537 | 2,874 (ort_strict) | 1.13x | 2,874 (ort_strict) | 1.13x |
| `mlp_large` | 32 | 16,189 | 62,661 (torch_eager) | 3.87x | 81,581 (ort_tf32) | 5.04x |
| `cnn_large` | 1 | 3,096 | 1,819 (ort_strict) | 0.59x | 1,819 (ort_strict) | 0.59x |
| `cnn_large` | 32 | 3,953 | 14,426 (torch_compile) | 3.65x | 20,839 (ort_tf32) | 5.27x |
| `bert_small` | 1 | 150 | 334 (torch_compile) | 2.23x | 334 (torch_compile) | 2.23x |
| `bert_small` | 32 | 232 | 1,524† (torch_compile) | 6.56x | 2,619† (ort_tf32) | 11.27x |

@@RES
| model | variant | GPU util % | GPU power W | VRAM MB | host cores | host peak RSS MB |
|---|---|---|---|---|---|---|
| `clean_mlp` | torch_eager | 8 | 21 | 117 | 0.98 | 810 |
| `clean_mlp` | ort_strict | 14 | 22 | 97 | 0.93 | 452 |
| `dynamic_batch_cnn` | torch_eager | 30 | 47 | 117 | 0.93 | 858 |
| `dynamic_batch_cnn` | ort_strict | 48 | 35 | 103 | 0.96 | 571 |
| `gnn_gcn` | torch_eager | 6 | 21 | 117 | 0.96 | 1179 |
| `gnn_gcn` | ort_strict | 10 | 21 | 99 | 0.93 | 619 |
| `tiny_bert` | torch_eager | 14 | 22 | 117 | 0.95 | 1062 |
| `tiny_bert` | ort_strict | 33 | 20 | 99 | 0.97 | 615 |
| `mlp_large` | torch_eager | 83 | 22 | 137 | 1.00 | 820 |
| `mlp_large` | ort_strict | 83 | 35 | 127 | 0.96 | 440 |
| `cnn_large` | torch_eager | 92 | 27 | 419 | 0.98 | 925 |
| `cnn_large` | ort_strict | 97 | 21 | 451 | 0.97 | 648 |
| `bert_small` | torch_eager | 98 | 104 | 715 | 1.00 | 1270 |
| `bert_small` | ort_strict | 99 | 90 | 2159 | 0.99 | 931 |

@@RES1
| model | variant | GPU util % | GPU power W | VRAM MB |
|---|---|---|---|---|
| `clean_mlp` | torch_eager | 4 | 46 | 117 |
| `clean_mlp` | ort_strict | 8 | 22 | 95 |
| `dynamic_batch_cnn` | torch_eager | 20 | 20 | 117 |
| `dynamic_batch_cnn` | ort_strict | 8 | 21 | 95 |
| `gnn_gcn` | torch_eager | 6 | 20 | 117 |
| `gnn_gcn` | ort_strict | 10 | 21 | 95 |
| `tiny_bert` | torch_eager | 5 | 22 | 117 |
| `tiny_bert` | ort_strict | 22 | 13 | 95 |
| `mlp_large` | torch_eager | 22 | 22 | 144 |
| `mlp_large` | ort_strict | 34 | 18 | 127 |
| `cnn_large` | torch_eager | 9 | 22 | 117 |
| `cnn_large` | ort_strict | 15 | 22 | 97 |
| `bert_small` | torch_eager | 20 | 13 | 161 |
| `bert_small` | ort_strict | 24 | 22 | 173 |

@@ERR
| model | fp32 (eager / ORT) | TF32 (torch / ORT) |
|---|---|---|
| `clean_mlp` | 1.2e-07 / 1.2e-07 | 4.2e-04 / 4.2e-04 |
| `dynamic_batch_cnn` | 6.0e-08 / 6.0e-08 | 1.3e-04 / 6.0e-08 |
| `gnn_gcn` | 4.8e-07 / 4.8e-07 | 2.2e-03 / 2.2e-03 |
| `tiny_bert` | 9.5e-07 / 7.2e-07 | 1.7e-05 / 1.7e-05 |
| `mlp_large` | 2.7e-07 / 2.1e-07 | 2.4e-04 / 2.4e-04 |
| `cnn_large` | 6.0e-08 / 7.5e-08 | 1.1e-04 / 3.7e-05 |
| `bert_small` | 2.1e-06 / 1.9e-06 | 5.7e-04 / 5.6e-04 |

@@SETUP
| model | eager s | compile s | ORT s |
|---|---|---|---|
| `clean_mlp` | 6.5 | 14.7 | 1.1 |
| `dynamic_batch_cnn` | 6.7 | 20.8 | 1.8 |
| `gnn_gcn` | 22.1 | 36.1 | 1.4 |
| `tiny_bert` | 18.9 | 31.4 | 1.6 |
| `mlp_large` | 7.8 | 16.4 | 1.2 |
| `cnn_large` | 8.1 | 21.6 | 1.7 |
| `bert_small` | 19.1 | 33.1 | 1.7 |
