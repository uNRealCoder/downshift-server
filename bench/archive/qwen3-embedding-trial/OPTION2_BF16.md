# Option 2 trial: serve fp32 models in bf16 (2026-10-06/07)

Machine: Ryzen 7 7840HS (8C/16T, native bf16), 16 GB, Windows, CPU only. downshift 0.5.0 working
tree with `--max-concurrency` default 4. Model: all-MiniLM-L6-v2 (ships fp32). The bf16 copy is
`minilm-bf16/` (made by `to_bf16.py`; transformers loads it in bf16 from its config). Driver:
`tune.py`, raw rows: `opt2_minilm.jsonl`. Closed loop, 20 s windows (60 s for 8/32-doc batches).

## Accuracy

bf16 vs fp32, 16 sentences: cosine min 0.99994, max pairwise-similarity change 0.0025, nearest
neighbour unchanged for 16/16.

## Throughput, 8 clients (texts/s; latency at 1 client)

| request | ONNX Runtime fp32 (today's auto) | torch fp32 | torch bf16 (option 2) |
|---|---|---|---|
| short query | **454** (5.8 ms) | 132 (9.8 ms) | 152 (9.4 ms) |
| 1 doc (~150 tok) | 86 (31.8 ms) | 67 (28.7 ms) | **91** (24.9 ms) |
| 8 docs | 81 (211 ms) | 99 (142 ms) | **141** (105 ms) |
| 32 docs | 76 (834 ms) | 88 (592 ms) | **142-149** (445 ms) |
| peak RAM | 2.0 GB | 0.7 GB | **0.55 GB** |

## Read

- bf16 is no blanket win. ONNX Runtime is 3x faster on short single queries, where per-call
  overhead dominates; torch bf16 is ~1.7-2x faster once a request carries real compute
  (batches, long text), and uses far less memory.
- ONNX Runtime fp32 is slower than even torch fp32 on batches here.
- `--backend auto` on the bf16 copy reproduces the Qwen problem: export runs, ORT cannot load
  it (`Add(14) NOT_IMPLEMENTED`), it falls back to torch. That is option 1's fix.
- One machine with native bf16. CPUs without it (pre-Zen 4 AMD, most pre-2021 Intel) are
  expected to run bf16 slower than fp32; unmeasured.
