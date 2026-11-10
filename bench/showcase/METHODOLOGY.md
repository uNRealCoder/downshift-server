# Methodology

How the benchmark data in `bench/` is made, what machine and versions made it, how to make it again, and what the data cannot show.
Every value below comes from a data file. The file is named next to the value.

## 1. The question

downshift picks a backend from a correctness verdict. The benchmark does not ask "is downshift fast". It asks three things:

1. What does the verdict cost? (boot time, time of each request)
2. What does it give? (a right answer where a hand-rolled ONNX server gives a wrong one; ONNX Runtime speed where the graph is right)
3. Does the gate agree with what is true? (hazard fixtures with known verdicts, and real models)

## 2. What is measured

| Study | Script | Data file | Page |
|---|---|---|---|
| Throughput and latency under load | `python -m bench.run` | `bench/results/v1.0.0/results.json`, `chart_runs.csv` | [throughput](throughput.html) |
| Time of one request, by stage | `python -m bench.run`, `python -m bench.hf_models` | `chart_stages.csv` | [time-breakdown](time-breakdown.html) |
| ONNX Runtime against torch, in-process | `python -m bench.inproc` (started by `bench.run`) | `results.json` key `inproc` | [torch-vs-ort](torch-vs-ort.html) |
| Real Hugging Face model over HTTP | `python -m bench.hf_models` | `results_hf.json` | [torch-vs-ort](torch-vs-ort.html) |
| Batches and wire formats | `python -m bench.run` | `chart_runs.csv` | [wire-formats](wire-formats.html) |
| Result cache | `bench.run`, `bench.hf_models`, `python -m bench.cache_accept` | `chart_runs.csv`, `bench/results/v0.6.0/cache_acceptance.json` | [result-cache](result-cache.html) |
| Correctness | `bench.run`, `bench.hf_models` | `results.json`, `chart_runs.csv`, `results_hf.json` | [correctness](correctness.html) |
| Real models and hazard fixtures | `python scripts/check_real_models.py`, `python scripts/gen_matrix.py` | `bench/results/v1.0.0/real_models.json`, `docs/compatibility.md` | [compatibility](compatibility.html) |
| Boot time | `bench.run`, `bench.hf_models` | `chart_runs.csv` (`ready_s`), `results_hf.json` | [boot-time](boot-time.html) |
| Workers | `bench.run` (0.5.0 run) | `bench/results/v0.5.0/chart_runs.csv` | [scaling](scaling.html) |
| Release to release | `bench.run` (two runs) | both `chart_runs.csv` files | [releases](releases.html) |

## 3. The harness

- **Servers.** Three hand-rolled servers (`bench/servers.py`): FastAPI + eager torch (sync endpoint), FastAPI + ONNX Runtime (sync), and FastAPI + eager torch with `async def`, which blocks the event loop. For text: FastAPI + transformers (`bench/naive_hf_server.py`). The downshift servers are the real `downshift serve` command, never an in-process shortcut.
- **Fairness.** The hand-rolled servers take a plain `dict` body with no validation and return a plain `dict`. They do less work for each request than downshift. The hand-rolled text server cuts long text. downshift rejects it. All servers get the same seeded weights (seed 1234, `bench/cases.py`). The hand-rolled ORT server gets the graph that the checkout exported.
- **Load.** Closed loop (`bench/loadgen.py`). N clients each send the next request when the last one returns. The body is the same bytes every time. The load generator is split over up to 6 processes on the same machine.
- **Ceiling.** Before the sweep, the harness measures the load generator against `GET /health` on a server that does nothing: c=1: 5194 rps, c=2: 9606 rps, c=4: 10446 rps, c=8: 11514 rps, c=16: 12102 rps, c=32: 11925 rps, c=64: 10707 rps. A result near these numbers would measure the client.
- **Windows.** Warm-up 1.0 s, then 3.0 s for each point on the fixtures. Text models: warm-up 1.0 s, window 5.0 s, concurrency [1, 8]. A `/health` probe runs every 50 ms during each text window.
- **Correctness.** Every response is compared element by element with eager PyTorch on the same input. The hand-rolled text servers and downshift answers are compared with `downshift --backend torch`.
- **Result cache.** Every 1.0.0rc1 row except "result cache on" starts with `--result-cache-rows 0`. The harness repeats one body, so a cache would answer from memory.

## 4. Hardware and versions (from the data)

| Item | Value | Source |
|---|---|---|
| Main run date | 2026-11-07T22:59:00 (HF part: 2026-11-07T23:45:53) | `results.json`, `results_hf.json` |
| Platform | win32; `Windows-11-10.0.26200-SP0` | `results.json`, `results_hf.json` |
| Logical CPUs | 16 | `results.json` |
| CPU model, RAM | not recorded in these runs (hardware: not recorded) | - |
| Python | 3.12.10 | `results.json` |
| torch | 2.14.0+cpu; torch threads 8 | `results.json` |
| onnxruntime | 1.30.0; providers AzureExecutionProvider, CPUExecutionProvider | `results.json` |
| downshift | checkout 1.0.0rc1; installed 0.5.1 | `results.json` |
| Load settings | cases clean_mlp, tiny_bert, mlp_large, scatter_include_self_false; concurrency [1, 2, 4, 8, 16, 32, 64]; batch [1, 8, 32]; batch concurrency [1, 8, 32]; in-process iterations 200 | `results.json` |
| Not run | the workers sweep (`skip_workers` = True); Prompt Guard; `dynamic_batch_cnn`, `gnn_gcn`, `cnn_large`, `bert_small` | `results.json` |
| Real-model check | date 2026-11-07; torch 2.14.0, onnx 1.22.0, onnxruntime 1.30.0, transformers 5.17.0, torch_geometric 2.8.0.post1, torchvision 0.29.2+cpu; OS and CPU not recorded | `real_models.json` |
| Cache acceptance (0.6.0) | run 2026-10-14 23:46:17; 60 repeats; machine and versions not recorded | `cache_acceptance.json` |
| Earlier run (0.4.0 and 0.5.0) | 2026-10-04T22:32:59; win32, 16 logical CPUs, Python 3.12.10, torch 2.14.0+cpu, onnxruntime 1.30.0 | `bench/results/v0.5.0/results.json` |

The folder `results/v1.0.0/` holds a run of downshift `1.0.0rc1`, not of the final 1.0.0 release.

## 5. How to reproduce

Run nothing else on the machine while the benchmark runs. A test suite at the same time more than doubles the latencies (`bench/README.md`).

1. Make a venv. Install the checkout dependencies and the previous release (non-editable): `pip install "downshift-server[all]==0.5.1" aiohttp safetensors`.
2. Download the two Hugging Face checkpoints into `bench/models/` (see `bench/README.md`). Prompt Guard is gated.
3. Run the fixture part: `python -m bench.run --out bench/results/vX.Y.Z/results.json`. The stored settings of the 1.0.0rc1 run match `--cases clean_mlp tiny_bert mlp_large scatter_include_self_false --skip-workers`. The exact command line is not stored.
4. Run the text part: `python -m bench.hf_models --out bench/results/vX.Y.Z/results_hf.json` (the 1.0.0rc1 run used all-MiniLM-L6-v2 only).
5. Make the report and the chart CSV files: `python -m bench.report --results ... --hf-results ... --out .../REPORT.md --csv .../chart_runs.csv --stages-csv .../chart_stages.csv`.
6. Check the real models: `python scripts/check_real_models.py`. Make the fixture table: `python scripts/gen_matrix.py`.
7. Run the cache check: `python -m bench.cache_accept --out bench/results/vX.Y.Z/cache_acceptance.json`.
8. Make this site again: `python bench/showcase/build_showcase.py`. Point `R1` (and the other result folders) in the script at the new folder first.

Without `--out`, the commands write to `bench/scratch/`. A trial run never overwrites a published run.

## 6. Known limits and threats to validity

1. **One machine, one run, no repeats.** No file holds repeated runs. The size of run-to-run noise is not measured. A difference of a few percent is not a finding. The report itself calls the difference of the cache-on and cache-off rows of `tiny_bert` and `scatter_include_self_false` "noise".
2. **Short windows.** 3.0 s for each point. The p99 at concurrency 1 comes from 905 to 3559 requests for the 1.0.0rc1 default rows. The p99 at high concurrency has fewer samples for each client.
3. **The load generator shares the machine.** The CPU is shared between the client and the server. Results are valid for this layout only.
4. **Closed loop.** At high concurrency the latency is set by the concurrency, so the latency at c=64 is mostly a queue.
5. **A slow main pass.** The first `mlp_large` pass of 1.0.0rc1 default was slow. The report says it "does not fit one code path". A re-run covers batch 1 only (`rerun_mlp_large.json`). The batch 8 and 32 rows of `mlp_large` for 1.0.0rc1 JSON are not re-measured.
6. **Release candidate.** The newest data is of `1.0.0rc1`.
7. **Small fixtures.** On `clean_mlp` the model takes about 0.13 ms of a 0.70 ms request. The fixtures measure the request path, not the model. The compute-heavy models (`mlp_large`, `tiny_bert`) are still small. Only MiniLM is a real model in the load run.
8. **Random weights.** The fixtures use seeded random weights. The correctness numbers hold for those weights.
9. **Hand-rolled baselines are one person's choice.** They are written to do less work than downshift. A tuned hand-rolled server could be faster or slower. TorchServe, BentoML, Triton, TEI and Infinity are not measured.
10. **The best-case rows.** "Result cache on" repeats one body. The acceptance check of the cache (0.6.0) runs in-process with a test client, stores no machine data, and was not run again for 1.0.0.
11. **The gate is a sample.** The gate runs 8 samples. A fault that none of them hits is not found. The real-model check shows verdicts, not a false-negative rate.
12. **Boot time ends are not equal.** downshift is ready on `/ready`. The hand-rolled servers are ready on `/health`.
13. **CPU and Windows only.** GPU, Linux and macOS are not measured.
14. **Source mismatches.** The pages show the values of the data files. If a text file disagrees with a data file, the page marks the mismatch. No mismatch is known now.

## 7. How the site is made

`build_showcase.py` reads the files above and writes into `bench/showcase/` only. It uses the Python standard library only. Sentences that depend on the data are checked in the script (`claim(...)`). If the data change and a sentence is no longer true, the build stops with an error.
