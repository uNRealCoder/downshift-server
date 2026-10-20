# downshift

Serve a downloaded model over HTTP with one command. Before the first request, downshift exports the model to ONNX and compares the ONNX graph with PyTorch on inputs that the exporter did not see. If the graph gives wrong numbers, downshift serves eager PyTorch instead.

Everything that downshift serves is already on the machine: a PyTorch checkpoint, an ONNX file, a downloaded Hugging Face repo directory, or an `nn.Module` that Python can import. Downshift fetches nothing.

```
$ downshift serve examples.scatter_include_self_false:make_model

2026-10-20 16:20:33 WARNING downshift.serve: DOWNSHIFT_SERVER_API_KEY is not set, so the endpoints (including /metadata) are unauthenticated. Set it to require a fixed API key, or add your own authentication middleware.
2026-10-20 16:20:33 INFO downshift.report: loading examples.scatter_include_self_false:make_model
2026-10-20 16:20:33 INFO downshift.report: will listen on http://127.0.0.1:8000 (not ready yet)
2026-10-20 16:20:36 INFO downshift.report: verify: sample 1/8, x [6, 8], segment_ids [6]
...
2026-10-20 16:20:36 INFO downshift.report: verify: sample 8/8, x [7, 8], segment_ids [7]
2026-10-20 16:20:36 INFO downshift.report: downshift v0.6.0
  Model          examples.scatter_include_self_false:make_model
  Family         generic
  Verdict        DEGRADED  (strict=False, opset 20)
  Numerics       max abs err 1.62e+00 over 8 samples  6/8 failed
                 ! numerics diverge on 6/8 samples (max abs err 1.62e+00)
  Override       --force-onnx to serve the ONNX graph anyway
  Tolerance      atol 1e-04, rtol 1e-03 (float32)
  Worst          output_0[3, 0]: torch 0.2167, onnxruntime 1.8388  (sample 2, x (7,8), segment_ids (7))
  Samples        x: (6,8) (12,8) (7,8) (7,8) (1,8) (12,8) (1,8) (7,8)
                 segment_ids: (6) (12) (7) (7) (1) (12) (1) (7)
  Rows           not checked (the numerics check failed)
  Cache          off (the numerics check failed)
  Warmup         3 inferences, 0.09 ms each
  Boot           2.9 s: load 0.0, export 2.8, verify 0.0, session 0.0, warmup 0.0
  Backend        torch (eager) | cpu  <- auto-selected
  Verified on    CPUExecutionProvider
  Dynamic dims   `dim0` (x[0])  sampled 1-12, serves 1-65536  ! unverified above 12
                 `dim0` (segment_ids[0])  sampled 1-12, serves 1-65536  ! unverified above 12
  Encoding       json  (clients override with output_encoding)
  Capacity       4 inferences at a time, 4 prep threads, 64 queued, 30 s timeout  (--max-concurrency, --prep-threads, --max-queue, --request-timeout)
  Execution      threadpool  (--execution)
  Endpoint       http://127.0.0.1:8000  (GET /schema for the input format)
2026-10-20 16:20:36 INFO downshift.report: ready in 2.9 s
```

This graph exported without an error, but it gives wrong numbers on 6 of 8 inputs. Downshift found the fault before the first request and serves PyTorch instead. The model is [`examples/scatter_include_self_false.py`](examples/scatter_include_self_false.py). To run the command, use a clone of this repo.

## Install

```bash
pip install downshift-server            # core: any nn.Module, any .onnx
pip install "downshift-server[gnn]"     # + PyTorch Geometric adapter
pip install "downshift-server[hf]"      # + Hugging Face repos, text input
pip install "downshift-server[fast]"    # + pybase64, about 12x faster base64 tensors
pip install "downshift-server[all]"
```

Downshift needs Python 3.12 to 3.14. If the `downshift` command is not on `PATH`, use `python -m downshift`.

## Quick start

1. Start the server. The port opens at once. The gate runs in the background.

   ```bash
   downshift serve my_pkg.models:build --port 8000
   ```

2. Wait until `GET /ready` gives `200`. Until then, it gives `503` with the current phase.
3. Ask the server what to send. `GET /schema` reads the input names, dtypes, shapes and bounds from the graph that runs. It also gives an `example_request` that you can send back unchanged.

   ```bash
   curl -s localhost:8000/schema | python -m json.tool
   ```

4. Send a request:

   ```bash
   curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"inputs": {"x": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]}}'
   ```

   ```json
   {"outputs": {"output_0": [[0.199, -0.206, 0.561, 0.405]]},
    "shapes": {"output_0": [1, 4]},
    "dtypes": {"output_0": "float32"}}
   ```

Integer lists become `int64`, and all other lists become `float32`. For large tensors, send base64 or a safetensors body. A Hugging Face repo with a tokenizer also takes `{"text": [...]}`. A graph model takes `x` and `edge_index` on `POST /predict/graph`. Refer to [`docs/http-api.md`](docs/http-api.md).

| Route | Purpose |
|---|---|
| `GET /` | The version, the model, `ready`, and all routes. Start here if you know only the base URL. |
| `GET /health` | Liveness. `200` while the process is up, also during the load. |
| `GET /ready` | Readiness. `503` until the model is loaded, verified and warmed up. |
| `GET /schema` | What to POST, with an example body. |
| `GET /metadata` | The verdict, the backend, the limits and the boot timings. |
| `POST /predict` | Named tensors, or `text`. JSON or safetensors. |
| `POST /predict/graph` | One graph, or a batch of graphs. |

To mount the same app in your own FastAPI service, use `downshift.serve.app_for(model, example_inputs)`. Refer to [`docs/python-api.md`](docs/python-api.md#app_for). To deploy, read [`docs/production.md`](docs/production.md).

## The gate

A successful `torch.onnx.export` does not prove that the graph computes the same function as the model. Before downshift serves anything, it exports the model in memory. It then runs `k` samples (default 8) through PyTorch and through ONNX Runtime, and it varies the dynamic axes, so some samples have shapes that the exporter did not see. The result is one of four verdicts, and the verdict selects the backend:

| Verdict | Meaning | Served by |
|---|---|---|
| CLEAN | The model exports, matches PyTorch on every sample, and works on shapes that the trace did not include. | ONNX Runtime |
| DEGRADED | The model exports, but the numbers differ by more than the tolerance on at least one sample. | eager PyTorch (`--force-onnx` overrides) |
| FAILED | The model does not export, or ONNX Runtime cannot load or run the graph. | eager PyTorch |
| UNVERIFIED | A `.onnx` file with no `--reference` model, or `--no-verify`. | ONNX Runtime, labelled as never checked |

The gate also runs alone:

```bash
downshift check my_pkg.models:build --json > verdict.json   # exit code = verdict: 0 CLEAN, 1 FAILED, 2 DEGRADED, 3 UNVERIFIED
downshift export my_pkg.models:build -o artifacts/            # writes the .onnx file and a manifest
```

`check` can gate CI. Exit code `4` is a usage error, and `5` is a crash. [`docs/cli.md`](docs/cli.md) describes each flag, the tolerances and the report.

## Accepted model forms

| Argument | Meaning |
|---|---|
| `model.onnx` | A downloaded or already exported ONNX file. Served as it is. UNVERIFIED unless you give `--reference model`. |
| `weights.pt` | A downloaded PyTorch checkpoint (state dict). Needs `--model-class pkg.module:Class`. `.pth`, `.bin` and `.ckpt` also work. Downshift loads it with `weights_only=True`. `--unsafe-load` allows a pickled module from a file that you trust. |
| `path/to/repo/dir/` | A downloaded Hugging Face repo: a directory with a `config.json`. Needs the `[hf]` extra. |
| `pkg.module:attr` | An `nn.Module`, or a function that returns one. If the module has a `make_inputs` function, downshift uses it for the example inputs. Otherwise, give `--inputs pkg.module:fn`. |

Downshift serves a Hugging Face repo the way its own files say:

- A classifier (`...ForSequenceClassification`) loads with its head and also returns class probabilities.
- A sentence-transformers repo (`modules.json`, `1_Pooling/`) is pooled and normalised inside the exported graph, so the verification covers the embedding.
- With tokenizer files, `/predict` takes `{"text": ...}`. A row that is longer than the model reads gets a `400`. Downshift never cuts a row.

`downshift serve model.onnx --tokenizer-from repo/` serves a graph that someone else exported (for example with `optimum` or Olive) and keeps text input, with the tokenizer and recipe of the repo.

Downshift never contacts the Hugging Face hub and rejects a hub id. Download the repo first:

```bash
huggingface-cli download bert-base-uncased --local-dir ./bert-base-uncased
downshift serve ./bert-base-uncased
```

## Compatibility matrix

`scripts/gen_matrix.py` generates this table from the fixtures in `tests/models/`. Each fixture isolates one export hazard. CI makes the table again each week against the current torch and onnxruntime. The full file, with versions, is [docs/compatibility.md](docs/compatibility.md).

<!-- matrix:start -->
| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `bf16_weights` | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic | FAILED | strict=False | — | — | torch |
| `broken_factory` | Not an export hazard fixture: raises as soon as it is instantiated | — | skipped (RuntimeError) | — | — | — | — |
| `clean_mlp` | Control fixture: no export hazards | generic | CLEAN | strict=False | 1.8e-07 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic | CLEAN | strict=False | 1.5e-07 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic | CLEAN | strict=False | 3.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 3.6e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic | DEGRADED | strict=False | 2.2e+00 | — | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a BERT encoder with two layers and random initialization | hf | CLEAN | strict=False | 7.2e-07 | ✓ | onnxruntime |
<!-- matrix:end -->

Read these rows carefully:

- `scatter_include_self_false` exports without an error and returns wrong numbers. Only the numerics check stops this graph from reaching production.
- `bf16_weights` exports without an error, but ONNX Runtime cannot run it, because bfloat16 has no CPU Gemm kernel. It is a FAILED verdict and serves on torch.
- `custom_autograd` is CLEAN. `torch.export` traces through a `Function.forward` that has ordinary operations.

## Scope and non-goals

Downshift serves models that give their answer in one forward pass: tensors in, tensors out. The gate compares one pass between the ONNX graph and PyTorch. GNNs, encoders, classifiers, embedders and your own `nn.Module` classes fit. A model that needs a loop around it does not fit.

- **No text generation.** There is no KV cache, sampling loop or OpenAI-compatible endpoint. Use vLLM, TGI or llama.cpp. Downshift refuses a decoder-only repo (`...ForCausalLM`), unless it has an embedding recipe, as Qwen3-Embedding has.
- **No remote code.** Downshift never sets `trust_remote_code`. A repo with an `auto_map` does not load. Export it to ONNX yourself and serve it with `--tokenizer-from`.
- **No multimodal Hugging Face models.** Text in, one tensor out.
- **No quantization or graph optimization.** Use Olive or `onnxruntime.quantization`, then serve the result with `downshift serve model.onnx --reference model.pt`, so downshift verifies it against the original weights. `--fp16` is a cast before the trace.
- **No dynamic request batching.** One request makes one inference. `/predict/graph` joins a batch of graphs only when the outputs are for each node or each edge.
- **No Docker image and no DGL adapter.** Use pip and version pins. PyG is the supported GNN family.

To add a model family, write an adapter. Refer to [`docs/adapters.md`](docs/adapters.md).

## Documentation

- [`docs/cli.md`](docs/cli.md): every command, flag, environment variable and exit code.
- [`docs/http-api.md`](docs/http-api.md): every route, body, status code and wire format.
- [`docs/production.md`](docs/production.md): authentication, HTTPS, probes, workers, caches, sizing, GPUs and Kubernetes.
- [`docs/python-api.md`](docs/python-api.md): `check`, `export`, `intake`, `app_for` and the verdict.
- [`docs/serving.md`](docs/serving.md) and [`docs/adapters.md`](docs/adapters.md): the internals, for contributors.
- [`llms.txt`](llms.txt) for an LLM agent that calls or hosts a server. [`AGENTS.md`](AGENTS.md) for a coding agent that changes this repo.
- [`CHANGELOG.md`](CHANGELOG.md).

## Development

```bash
pip install -e ".[dev,all]"
ruff check .
ruff format --check .
mypy src
pytest
python scripts/gen_matrix.py     # makes docs/compatibility.md and the table above again
```

CI requires 95% line coverage (`--cov-fail-under=95`). The HTTPS tests make a temporary certificate with `openssl`. If `openssl` is not on `PATH`, these tests skip.

## License

MIT.
