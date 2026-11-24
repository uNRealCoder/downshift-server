# Published runs

One folder for each downshift release that has a published run. Each folder holds the raw JSON
that the harness wrote, a report that `python -m bench.report` made from it, and (from 0.5.0) the
chart data. Every number in a report comes from the JSON next to it.

**Hardware: the CPU runs record the platform (`win32`), 16 logical CPUs and the torch thread count.**
They do not record the CPU model or the RAM. Compare rows inside one report. Do not compare them
with a run on other hardware, or a CPU run with a GPU run.

| folder | what it is | report | files |
|---|---|---|---|
| `v1.0.0/` | naive vs 0.5.1 vs 1.0.0 (run 2026-11-22, Python 3.14, CPU); real-model verdicts; GPU load test | [`v1.0.0/REPORT.md`](v1.0.0/REPORT.md) | see below |
| `v0.6.0/` | result cache acceptance check; torch vs ONNX Runtime, in process (CPU and CUDA) | none (see `.docs/REPORT_ONNX_VS_TORCH.md`) | `cache_acceptance.json`, `torch_vs_ort.json`, `torch_vs_ort_cuda.json` |
| `v0.5.0/` | naive vs 0.4.0 vs 0.5.0, one session | [`v0.5.0/REPORT.md`](v0.5.0/REPORT.md) | `results.json`, `results_hf.json`, `chart_runs.csv`, `chart_stages.csv` |
| `v0.4.0/` | naive vs 0.4.0, behind the launch post *Stop hand-rolling model servers* | [`v0.4.0/REPORT.md`](v0.4.0/REPORT.md), [`v0.4.0/BLOG_APPENDIX.md`](v0.4.0/BLOG_APPENDIX.md) | `results.json`, `results_hf.json`, `results_hf_tuned_minilm.json`, `results_hf_tuned_pg.json` |

The 0.4.0 files come from the harness as of commit `02419cf`. They are kept unchanged. Their
numbers are not comparable row for row with the 0.5.0 run, which measured 0.4.0 again next to
0.5.0 in one session.

## Files in `v1.0.0/`

| file | what it holds | made by |
|---|---|---|
| `REPORT.md` | The report. | `python -m bench.report` |
| `results.json` | Fixture and compute-heavy models: verdicts, HTTP sweep, batches, wire formats, `--workers` sweep. | `python -m bench.run` |
| `results_hf.json` | Real Hugging Face models (MiniLM, Prompt Guard), with the `/health` probe. | `python -m bench.hf_models` |
| `chart_runs.csv`, `chart_stages.csv` | The two results files as flat tables (columns in `bench/README.md`). | `python -m bench.report --csv --stages-csv` |
| `real_models.json` | `downshift check` verdicts of 8 real models (2026-11-07). The table in `docs/compatibility.md` comes from it. | `python scripts/check_real_models.py` |
| `results_gpu.json` | GPU load test: hand-rolled torch against downshift, both on the GPU (RTX 4050 laptop, Python 3.12, torch `+cu130`). 4 models, 56 runs. | `BENCH_DEVICE=cuda python -m bench.run --variants naive_torch downshift` |
| `torch_vs_ort_cuda.json` | In-process torch against ONNX Runtime on the GPU. No HTTP. | `python -m bench.torch_vs_ort --device cuda` |
| `failed_rows_workers.json`, `workers_repeats.json` | Rows of the `--workers` sweep that were measured again. | `python -m bench.run` |

The GPU run used another Python and another torch build than the CPU run. Do not put a GPU number
and a CPU number in one comparison.

# Summaries of two runs

## 1.0.0 at a glance (`v1.0.0/`)

From [`results/v1.0.0/REPORT.md`](v1.0.0/REPORT.md) (run 2026-11-22; about 3 hours of
measuring with 3 s windows, 5 s on the Hugging Face models); 16-CPU Windows machine, CPU only,
Python 3.14, batch 1, JSON bodies, 1.0.0 from the checkout against 0.5.1 from PyPI. The result
cache is OFF in every headline row, also in the `--workers` sweep (`--result-cache-rows 0`): the
harness sends one body again and again, so a cache would answer from memory. Differences under
about 15% between two rows are noise in this run (see "Notes on this run" in the report). The
earlier run of this directory, of 1.0.0rc1, is replaced by this one.

- **1.0 matches 0.5.1 on throughput and latency, with no gain.** Concurrency 1, `clean_mlp`: 883
  vs 845 rps, p50 1.08 vs 1.09 ms, p99 1.96 vs 2.26 ms. `tiny_bert`: 632 vs 583 rps, p99 2.4 vs
  3.1 ms. `mlp_large`: 490 vs 499 rps. Peak rps: `clean_mlp` 1164 vs 1118, `tiny_bert` 1112 vs
  1082, `mlp_large` 1013 vs 1074, `bert_small` 259 vs 260. `scatter_include_self_false` peak is
  959 vs 1047, but its `--metrics` row of the same code reaches 1174, so that gap is noise.
  MiniLM, one sentence, c=8: 528 vs 532 rps, p50 at c=1 4.8 vs 4.8 ms.
- **`--metrics` costs between 5 and 100 microseconds a request.** Concurrency 1: `clean_mlp` 883
  to 813 rps (+97 us, -8%), `tiny_bert` 632 to 630 (+6 us), `mlp_large` 490 to 485 (+21 us),
  `bert_small` no change. Peak rps differs by -15% (`clean_mlp`), -7%, -3% and -3%.
- **The result cache, all hits, is a best case.** `clean_mlp` peak 1407 vs 1164 rps (+21%);
  `mlp_large` 1629 vs 1013 rps (+61%) and c=1 p50 0.90 vs 1.98 ms; MiniLM `short_b1` at c=8
  1504 vs 528 rps and c=1 p50 0.8 vs 4.8 ms; Prompt Guard 1308 vs 103 rps. It does nothing for
  `tiny_bert`, `scatter_include_self_false` and `bert_small` (252 vs 259 rps), where the
  default cache is off for `tiny_bert` and `scatter_include_self_false`.
- **`--workers` scales, and 1.0.0 equals 0.5.1.** At 4 workers and c=8, `clean_mlp` 4123 vs 4581
  rps, `mlp_large` 1736 vs 1858, `bert_small` 278 vs 285 (c=32: 327 vs 346). At 1 worker,
  c=1, 1.0.0 is within 5% of 0.5.1 on all three models.
- **Prompt Guard** (the first run for 1.0.0): 103 vs 102 rps at c=8 against 16 for the
  hand-rolled transformers server; `--backend torch` gives 16. `downshift check` takes 47 s
  (42 s for 0.5.1).
- **Correctness is unchanged.** Every downshift row matches eager PyTorch to at most 2.7e-06
  (`bert_small`); MiniLM 1.3e-07 against `--backend torch`, cosine 1.0; Prompt Guard 2.3e-05,
  8/8 labels. `scatter_include_self_false` is DEGRADED and still serves from torch (error
  2.8e-08), where the hand-rolled ONNX server is off by 1.2. Failed requests in the published
  rows: 0.
- **Three windows stalled, and the harness was the cause** (`--workers 2` c=32 and `--workers 4`
  c=8: requests that timed out after 30 s or more). The harness started each server with its
  output on a pipe that nobody read. A Windows pipe holds about 4 KB. When it was full, the next
  log line blocked: at start-up (3 of 12 starts in a repeat never became ready) and under load
  (3 more stalled windows). `bench/run.py` now writes server output to a temp file. One sweep
  after the fix had 0 errors in 8 windows and no hang. The stalled rows stay in
  `failed_rows_workers.json` and the repeats in `workers_repeats.json`. downshift is not at fault.
- **`/health` stays responsive** under load on MiniLM with the cache off: worst p99 22 ms for 1.0.0
  and for 0.5.1, against 481 ms for the hand-rolled `async def` server. On Prompt Guard the worst
  p99 is 319 ms for 1.0.0 (294 ms for 0.5.1); the `async def` server reaches 5.2 s.
- **The hand-rolled servers win on the tiny fixtures here.** `clean_mlp` peak: ONNX Runtime
  FastAPI 1706 and `async def` torch 2347 against 1164 for downshift. Downshift wins where it
  matters: MiniLM 528 vs 142 rps at c=8, Prompt Guard 103 vs 16, and the correct answer on
  `scatter_include_self_false`.

## 0.5.0 at a glance

From [`results/v0.5.0/REPORT.md`](v0.5.0/REPORT.md) (run 2026-10-04/05); 16-CPU
Windows machine, CPU only, peak requests/s at batch 1 unless stated. Chart data:
`v0.5.0/chart_runs.csv`, `v0.5.0/chart_stages.csv`.

- **0.5.0 is faster than 0.4.0 on every model.** Peak batch-1 throughput, JSON bodies:
  `clean_mlp` 1030 vs 866 (+19%), `dynamic_batch_cnn` 1031 vs 862, `gnn_gcn` 891 vs 810,
  `tiny_bert` 906 vs 820, `scatter_include_self_false` 1076 vs 940, `mlp_large` 815 vs 696
  (+17%), `cnn_large` 736 vs 657, `bert_small` 106 vs 102. Each request takes two thread hops
  (prep pool, then inference thread) and skips FastAPI's own body handling.
- **Where the time goes** (`clean_mlp`, concurrency 1): 1.00 ms in all, of which 0.63 ms is the
  HTTP stack, 0.19 ms inference, and 0.18 ms parse, prep, waits and encode together; 0.4.0
  takes 1.27 ms. On `bert_small` inference is 7.3 of 10.6 ms, and safetensors cuts the total to
  8.8 ms by shrinking `encode` from 1.8 ms to 0.1 ms.
- **`--execution inline` is the extra gear for tiny models**: `clean_mlp` 1498,
  `scatter_include_self_false` 1545, `dynamic_batch_cnn` 1211, `gnn_gcn` 1160 (all above the
  0.5.0 default). `tiny_bert` is the exception (844 vs 906 default), and on MiniLM it changes
  nothing (281 vs 282), so it is only worth it for models that infer in well under 1 ms.
- **Binary bodies** move the most rows on wide inputs: at batch 32, concurrency 8, `mlp_large`
  does 14.4k rows/s with safetensors against 6.6k as JSON on 0.5.0 (11.2k with 0.4.0 base64);
  `cnn_large` 4.1k vs 1.5k. safetensors matches or beats base64 with bodies about 25% smaller
  (`cnn_large` b32: 393 KB vs 524 KB, 2.0 MB as JSON); on `dynamic_batch_cnn` b32 it is 967 vs
  770 requests/s.
- **Real HF models.** all-MiniLM-L6-v2, one sentence, c=8: 282 vs 239 by default and 483 vs 434
  with `--max-concurrency 4`, against 139 for the hand-rolled transformers server. Prompt Guard
  50 vs 48, and 103 vs 101 with `--max-concurrency 4`, against 16 hand-rolled.
  `--max-concurrency` is still the biggest single lever for small encoders.
- **`--workers` scales both versions**; at 4 workers and concurrency 8, `clean_mlp` 3857 vs 3406,
  `bert_small` 239 vs 227.
- **Correctness is unchanged.** Every downshift row in both versions matches eager PyTorch to
  ≤ 2.7e-6; both HF models agree with eager torch (MiniLM cosine 1.0, Prompt Guard 8/8 labels).
  The hand-rolled ONNX server still returns wrong answers on `scatter_include_self_false`
  (max error 1.2), where both versions route to torch. Failed requests across every window: 0.
- **`/health` stays responsive** under load on every downshift row (worst p99 27 ms). The
  hand-rolled `async def` server reaches 5.2 s on Prompt Guard.
- **One startup hang, in 0.4.0**: `--workers 4` on `bert_small` never answered `/ready` within
  600 s once (it boots in about 24 s otherwise) and was re-run. Together with a low-memory stop
  during `bert_small`, the run was split into passes; the report says which rows came from where.
