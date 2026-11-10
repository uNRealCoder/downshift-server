# Benchmark showcase

A set of static pages that show the benchmark data of `bench/` to an outside reader. Every number comes from a stored data file. Nothing here runs a benchmark.

## What is here

| File | Contents |
|---|---|
| `index.html` | Start here. What downshift verifies, a flow diagram, and links to all pages. |
| `throughput.html` | Requests per second and latency against hand-rolled servers, 0.5.1 and 1.0.0rc1. |
| `time-breakdown.html` | Where the time of one request goes. |
| `torch-vs-ort.html` | ONNX Runtime against eager PyTorch, in-process and over HTTP. |
| `wire-formats.html` | Batches, JSON and safetensors. |
| `result-cache.html` | The result cache: best case and cost. |
| `correctness.html` | Every response against eager PyTorch. |
| `compatibility.html` | 14 hazard fixtures and 8 real models. |
| `boot-time.html` | The seconds before the first request. |
| `scaling.html` | The `--workers` sweep (0.5.0 data; not run for 1.0.0rc1). |
| `releases.html` | Release to release, inside one run each. |
| `../CAPABILITIES.md` | Not a page of this folder. One table: each capability, the result file that shows it, and one number. |
| `SUMMARY.html` | One page for print or screenshot. |
| `RESULTS.md` | The key tables as plain text, with the source file of each. |
| `METHODOLOGY.md` | How the data is made, hardware and versions, how to reproduce, limits. |
| `GLOSSARY.md` | Terms, for a reader who is new to the project. |
| `TALK_OUTLINE.md` | A 10-minute talk outline. |
| `build_showcase.py` | The script that makes the HTML files, `RESULTS.md`, `METHODOLOGY.md` and `TALK_OUTLINE.md`. |

`README.md` and `GLOSSARY.md` are written by hand. The script does not change them.

## How to open it

Open `bench/showcase/index.html` in a browser. To find the proof of one capability, read [`../CAPABILITIES.md`](../CAPABILITIES.md). The pages are self-contained: inline CSS, inline SVG, no script, no network. They work from disk, at phone width, and in light and dark mode.

## How to rebuild

```bash
python bench/showcase/build_showcase.py
```

- Use Python 3.12 or later. The script uses the standard library only.
- The script reads files in `bench/` and `docs/compatibility.md`. It never changes them.
- It writes inside `bench/showcase/` only.
- If a source file is missing, the page shows a "Missing data" note and the script exits with code 1.
- Some sentences depend on the data. The script checks them (`claim(...)`). If the data change and a sentence is no longer true, the build stops with an error. Change the sentence, then build again.
- To show a new run, change the paths `R1`, `R5` and `R6` at the top of the script.
- `results/v1.0.0/` holds data of 1.0.0rc1. The hardware (CPU model, RAM) is not recorded in any run. The pages say so.

## Rules of the pages

- No number without a source file.
- Each page says what the data shows and what it does not show.
- A claim that the data does not support is left out, or marked as not measured.
