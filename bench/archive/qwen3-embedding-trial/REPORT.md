# Qwen3-Embedding-0.6B on downshift 0.5.0 (CPU)

Trial run on 2026-10-06: serve `Qwen/Qwen3-Embedding-0.6B` with `downshift serve` the way a user
would, tune the flags, and compare against the FastAPI server people write by hand from the model
card.

- Machine: Ryzen 7 7840HS (8 cores / 16 threads), 16 GB RAM, Windows 11, CPU only.
- downshift 0.5.0 from `src/` (`PYTHONPATH=src`); the model repo is stored as bfloat16.
- Load: `tune.py`, closed loop with aiohttp, 3 s warmup; a request counts only if it starts and
  ends inside the window. `rps` is requests/s, `texts/s` is rows/s. Memory is the working set
  of the server and its children.
- Raw data: `bench.jsonl` (final 3-config run), `tune_round1.jsonl`, `tune_round2.jsonl`,
  `naive.jsonl`. Boot banner and long-text timings: `serve_default_first_boot.log`.

Workloads: `query` (one short query with `prompt_name=query`), `doc1` (one ~150-token
paragraph), `docs8` / `docs32` (8 / 32 such paragraphs in one request), `long2k` (one ~2k-token
text).

## Summary

1. **The default boot is a wasted export.** `--backend auto` spends 89 s exporting to ONNX,
   ORT can't load the graph (`NOT_IMPLEMENTED: Mul(14)`, because the weights load as bf16), and
   downshift falls back to torch eager anyway. Boot is 92-116 s instead of 16-18 s, and the
   failed export holds on to **+1.35 GB RAM** (2954 vs 1599 MB idle). The banner says FAILED
   but doesn't suggest `--backend torch`.
2. **The defaults leave half the throughput unused.** With `--max-concurrency 1` (the 0.5.0
   default) queries top out at about 7 rps however many clients there are. `--max-concurrency 4`
   doubles that to about 14.5 rps.
3. **`--intra-op-threads 16` is the latency flag.** One client sending 8 docs goes from 3.1 s
   to 1.86 s, and a 2k-token text from 9.3 s to 6.5 s. Under concurrent load it costs about 13%
   query throughput.
4. **A hand-written sync FastAPI server matches tuned downshift on throughput** (14.1 vs 14.5
   query rps). downshift only pulls ahead on single big requests, and only through the thread
   flag. The naive `async def` version halves throughput and drops connections under load.
5. **Output is correct.** Cosine similarity vs plain transformers is at least 0.9998, and
   `prompt_name=query` plus last-token pooling match the model card. (These checks came from
   `ref.py`; its output wasn't saved to a file.)

**Recommended flags for this model on this machine:**

| Goal | Flags |
|---|---|
| Throughput (many small requests) | `--backend torch --max-concurrency 4` |
| Latency (few large requests) | `--backend torch --intra-op-threads 16 --max-concurrency 4` |

## Boot and memory

| Config | Boot | Idle RSS |
|---|---:|---:|
| default (`auto`, export fails, falls back to torch) | 91.7 s (115.8 s in round 1) | 2954 MB |
| `--backend torch` | 16-19 s (one outlier at 38.7 s in round 1) | 1593-1613 MB |
| naive FastAPI + torch | 16.1 s | 1562 MB |

Default boot breakdown from the banner: load 16.9 s, **export 89.1 s**, verify 1.7, session 0.3,
warmup 0.4.

## Final comparison (`bench.jsonl`, `naive.jsonl`)

Throughput is in rows/s. Latency is p50.

| Workload | Clients | default | throughput (`torch mc4`) | latency (`torch t16 mc4`) | naive sync | naive async |
|---|---:|---:|---:|---:|---:|---:|
| query | 1 | 7.3 / 136 ms | 7.4 / 135 ms | **8.0 / 123 ms** | 7.1 / 139 ms | 7.2 / 138 ms |
| query | 4 | 7.2 / 542 ms | **14.6 / 268 ms** | 12.8 / 308 ms | 14.1 / 280 ms | 7.1 / 550 ms |
| query | 8 | 7.0 / 1083 ms | **14.5 / 538 ms** | 12.6 / 613 ms | 14.2 / 549 ms | 6.8 / 1102 ms |
| doc1 | 1 | 2.4 / 409 ms | 2.4 / 407 ms | **3.2 / 307 ms** | 2.4 / 408 ms | 2.4 / 407 ms |
| doc1 | 4 | 2.3 / 1624 ms | **4.8 / 812 ms** | 4.3 / 866 ms | 4.1 / 928 ms | 2.4 / 1627 ms |
| docs8 | 1 | 2.6 / 3.11 s | 2.4 / 3.16 s | **4.2 / 1.86 s** | 2.4 / 3.16 s | 2.4 / 3.17 s |
| docs8 | 4 | 2.2 / 12.6 s | 4.0 / 6.7 s | **4.4 / 6.5 s** | 4.2 / 6.6 s | 2.0 / 12.7 s ¹ |
| docs32 | 1 | 2.2 / 12.4 s | 1.6 / 12.7 s | **3.2 / 8.0 s** | 2.2 / 12.5 s | 2.2 / 12.4 s ¹ |
| long2k | 1 | 0.08 / 9.3 s | 0.08 / 9.4 s | **0.13 / 6.5 s** | 0.08 / 9.3 s | 0.08 / 9.3 s ¹ |

¹ From the second naive-async run (`naive_torch_async2`); the first was stopped after docs8 at
1 client.

**What fails under overload:**

- **default:** docs32 and long2k at 4-8 clients return 503s ("waited 37 s in queue, past the
  30 s timeout"). The timeout fires at about 37 s rather than 30 s because the queue is only
  checked when a slot frees up.
- **throughput config:** one 503 at docs32 / 8 clients.
- **naive async:** drops connections (`ServerDisconnectedError`, `WinError 64`) at docs8 / 8
  clients and at docs32 and long2k / 4 clients.

At 8 clients, docs32 and long2k often show 0 completions. Each request there takes longer than
the measurement window, so read that as "nothing finished in the window", not as a failure.

## Tuning rounds (`tune_round1.jsonl`, `tune_round2.jsonl`)

Query rps at 1 / 4 / 8 clients, and p50 for docs8 at 1 client. All configs use `--backend torch`.

| Flags | query rps 1 / 4 / 8 | docs8 p50 (1 client) |
|---|---|---:|
| (none) | 7.4 / 7.2 / 7.0 | 3.13 s |
| `--max-concurrency 2` | 7.4 / 11.7 / 11.5 | 3.16 s |
| `--max-concurrency 4` | 7.3 / **14.7** / 14.4 | 3.18 s |
| `--workers 2` | 7.2 / 12.0 / 12.1 | 3.17 s |
| `--intra-op-threads 16` | 7.9 / 7.9 / 7.6 | **1.85 s** |
| `--intra-op-threads 16 --max-concurrency 2` | 7.6 / 10.7 / 10.4 | 2.03 s |
| `--intra-op-threads 16 --max-concurrency 4` | 8.0 / 12.6 / 12.3 | 1.96 s |
| `--intra-op-threads 8 --max-concurrency 4` | 7.4 / 14.5 / 14.3 | 3.25 s |
| `--intra-op-threads 4 --max-concurrency 4` | 5.4 / 14.6 / 14.3 | 5.48 s |

What the rounds show:

- Concurrency is what buys throughput. Threads are what buy single-request latency. On 16
  logical cores you can't max out both.
- `--workers 2` was slower than `--max-concurrency 4`, and it pushed the 16 GB machine to
  critical memory: Claude Code killed the run. Its RSS column (362 MB) is wrong, because the
  process-tree sum didn't pick up the worker processes.
- Thread counts below 8 hurt single-client latency and don't add throughput.

## Long texts (`long.py`, latency-config server, from the access log)

| Request | Result |
|---|---|
| ~500 tokens | 200 in 1.36 s |
| ~2,000 tokens | 200 in 7.6 s |
| ~8,000 tokens | 200 in 64.1 s |
| over the 32,768-token limit | 400 in 123 ms (refused, not truncated) |
| batch of 32 × ~500 tokens | 200 in 44.2 s |

Attention cost grows faster than linearly with length: each 4× step in tokens took 5.6×, then
8.4× the time.
The model allows up to 32k tokens per row, but on this CPU anything near the top of that range
takes minutes.

## What this means for downshift

- **ONNX fails on bf16 repos.** Either load bf16 weights as fp32 for the export, or skip ONNX
  for bf16 checkpoints and say so in the banner. As it stands, this model pays 90 s and 1.35 GB
  for nothing.
- **The default `--max-concurrency` should be above 1.** A default of 4 was made afterwards
  (uncommitted at the time).
- **The `--intra-op-threads` help text says "ORT threads"**, but the flag sets torch threads
  too, and that's where the latency win comes from.
- **On throughput, downshift's value here is not speed**, since a careful sync FastAPI server
  ties it. The value is everything around inference: refusing over-long input, queueing with
  503s instead of dropped connections, `prompt_name`, pooling read from `modules.json`, and
  `/ready`.

Not measured: a naive ONNX server, because the fp32 export was killed for low memory.
