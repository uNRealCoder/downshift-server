# downshift — Universal Torch → ONNX Verification & Serving
## Implementation Plan

**Status:** Design draft, v2 — post-naming, post-Olive
**Author:** Nikhil Ranjan
**Package:** `downshift-server` (PyPI) · CLI/import: `downshift`
**Date:** September 2026

---

## 1. The One-Line Pitch

> Point it at any PyTorch `nn.Module`, TorchScript file, or Hugging Face repo. It tells you
> honestly whether the model can be trusted after exporting to ONNX, then serves it — with a
> graceful fallback to native PyTorch when ONNX would be lying to you.

We don't optimize models. We verify them and serve them, including the ones nobody else's
export tooling handles cleanly. That's the whole product.

---

## 2. Positioning: What This Is and Isn't

### 2.1 The competitive reality (be honest about this up front)

| Layer | Incumbent | Maturity | Should we compete? |
|---|---|---|---|
| PyTorch/HF → ONNX conversion | `torch.onnx.export(dynamo=True)` | Default since PyTorch 2.9 | **No — wrap it** |
| Quantization / graph optimization | Microsoft Olive, ORT's own `quantize_dynamic` | Mature, first-party | **No dependency, no reimplementation — see §5.7** |
| LLM generative loop (KV cache, sampling) | `onnxruntime-genai` | Mature, multi-EP | **No — delegate to it (deferred, v0.5+)** |
| High-throughput LLM serving | vLLM, SGLang, TensorRT-LLM | Very mature | **No — out of scope** |
| Export + validate + serve, general | `anydeploy` (PyPI, active) | Young but real, 0.2.x | **Partial overlap — differentiate on §2.4 below** |
| **Universal model triage (CLEAN/DEGRADED/FAILED) + honest fallback + GNN support** | *(nobody)* | — | **Yes. This is the wedge.** |

**Note on `anydeploy`:** a real competitor discovered mid-planning. It already does
export (PyTorch/sklearn → ONNX/TorchScript/TFLite), single-shot validation against the
original model, FastAPI serving, benchmarking, dockerizing, and a plugin-style exporter
registry. It does **not** appear to support GNNs (PyG/DGL), does **not** have a tiered
verdict system (its validation looks like pass/fail, not a documented `DEGRADED` middle
state), and its framing is edge/mobile-first. Skim its repo before Friday; add an explicit
"vs. anydeploy" line to the README. Silence on an obvious competitor reads worse than
addressing it directly.

### 2.2 What we are NOT building

- **Not a vLLM competitor on throughput.** vLLM's moat is PagedAttention + continuous batching
  implemented as custom CUDA kernels against PyTorch's dynamic execution. ONNX's static-graph
  model fights that pattern. We will not win there and shouldn't try.
- **Not a new quantization algorithm, and not a quantization tool at all.** GPTQ/AWQ/RTN exist
  in Olive and elsewhere — not our problem to solve. We accept their output (§5.7).
- **Not a new graph optimizer.** ONNX Runtime has three optimization levels already.
- **Not a Triton Inference Server replacement.** No multi-model orchestration, no model
  repository versioning, no ensemble scheduling — at least not in v1.

### 2.3 What we ARE building

Three things nobody has combined:

1. **An export triage system.** Classify every model into `CLEAN` / `DEGRADED` / `FAILED`
   with *numerical* evidence, not just "did it throw an exception."
2. **A graceful fallback path.** A model that can't export ONNX still gets served — via native
   PyTorch — through the *same* API, same CLI, same endpoint contract. The user never hits a
   dead end.
3. **A single CLI + serving surface** that covers a `GraphSAGE` node classifier, a BERT
   sentiment model, and a 1B-param LLM without the user learning three toolchains.

### 2.4 Why the "universal" framing is technically smart

Supporting arbitrary `torch.nn.Module` is hard *precisely because GNNs are hard* — variable-arity
forward signatures, custom scatter ops, dynamic control flow, sparse tensors, non-tensor container
inputs. If the export layer survives a PyG `MessagePassing` model, it survives almost anything
else, because most non-GNN torch models are structurally simpler.

**GNNs are the stress test, not a separate mode.** Build for the general case, validate against
the worst case.

---

## 3. Naming — DECIDED

- **PyPI package:** `downshift-server`
- **CLI command / import name:** `downshift` (e.g. `downshift serve model.pt`, `import downshift`)
- **Why the split:** the bare word `downshift` is already a well-known JS library (Kent C.
  Dodds' React combobox primitive, ~3.6M weekly npm downloads) — no technical conflict since
  npm and PyPI are separate namespaces, but bare `downshift` would fight it for search
  visibility and GitHub-org naming. `downshift-server` is unambiguous on PyPI and elsewhere;
  the short word is reserved for the part people actually type, via
  `[project.scripts] downshift = "downshift.cli:app"` in `pyproject.toml`. This pattern is
  normal — `beautifulsoup4` installs as `bs4`, `scikit-learn` installs as `sklearn`.
- **Confirmed available** via `pip index versions downshift-server` — no matching distribution.
- **Action:** reserve on PyPI with a stub `0.0.1` in H1, before any feature code.

---

## 4. Architecture

### 4.1 Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│  CLI  (Typer + rich)                                        │
│  serve · export · bench · check · init                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Public Python API                                           │
│  load() · export() · serve() · Verdict                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌───────▼────────┐ ┌──────▼─────────┐
│  ADAPTERS      │ │  EXPORT CORE   │ │  SERVING CORE  │
│                │ │                │ │                │
│ HF Transformers│ │ Signature      │ │ Backend:       │
│ PyG            │ │  introspection │ │  - ORT (EPs)   │
│ DGL            │ │ Dummy input    │ │  - torch eager │
│ Raw nn.Module  │ │  synthesis     │ │  - ORT-GenAI   │
│ TorchScript    │ │ dynamic_shapes │ │                │
│ Checkpoint file│ │  inference     │ │ Dynamic batcher│
│                │ │ Capture ladder │ │ Graph batcher  │
│ Each provides: │ │ Numerical      │ │ HTTP layer     │
│  - example in  │ │  verification  │ │  /predict      │
│  - flatten fn  │ │ Pre-opt intake │ │  /v1/* (LLM)   │
│  - batch fn    │ │ VERDICT        │ │  /health /meta │
└────────────────┘ └────────────────┘ └────────────────┘
```

### 4.2 Core design principle: the CLI is a thin shell

Every CLI command maps to a library function that is independently unit-testable and returns a
structured object, never printed text. `serve` is `render(serve_engine(...))`. This matters
because half the value of the project as a portfolio piece is that it's *well-engineered*, and
untestable CLI-embedded logic is the fastest way to undermine that.

---

## 5. The Export Layer (the hard part)

### 5.1 What PyTorch already gives us — use all of it

`torch.onnx.export(..., dynamo=True)` (default since 2.9) already implements a multi-strategy
capture ladder internally:

1. `torch.export.export(..., strict=False)`
2. `torch.export.export(..., strict=True)`
3. `torch.export.draft_export`
4. Decompose operators for ONNX compatibility
5. Translate graph to ONNX
6. `onnx.checker`
7. Execute with ONNX Runtime
8. Validate output accuracy

It also exposes `report=True`, `verify=True`, `profile=True`, `dump_exported_program=True`,
`artifacts_dir=...`, and `fallback=True` (falls back to the deprecated TorchScript exporter).

**Do not reimplement any of this.** Our job is to (a) *feed* it correctly, (b) *harvest* its
report into a structured verdict, and (c) *decide what to do* when it partially succeeds.

### 5.2 Where the real work is: dummy input synthesis

This is the single biggest engineering problem in the project. `torch.onnx.export` needs example
inputs. Getting them is trivial for a ResNet and genuinely hard for everything else.

Strategy ladder, in priority order:

**Tier 1 — User-supplied.** Always honored, always wins.
```python
kl.export(model, example_inputs=(x, edge_index))
```

**Tier 2 — Adapter-derived.** Each adapter knows how to build inputs for its family:
- **HF Transformers:** pull `OnnxConfig` from `optimum` — it already defines IO config and dummy
  input generators per task. Olive does this; we do the same.
- **PyG:** construct a small synthetic `Data(x=randn(N, F), edge_index=randint(...))` from the
  first layer's `in_channels`. Requires flattening (see 5.4).
- **DGL:** synthetic `DGLGraph` + node feature dict.
- **torchvision-style:** infer from first `Conv2d` / `Linear` shape.

**Tier 3 — Signature introspection + heuristics.** Read `forward()` type annotations and
parameter names. `x: torch.Tensor, edge_index: torch.Tensor` is a strong signal. Names like
`input_ids`, `attention_mask`, `edge_index`, `batch` map to known shape templates.

**Tier 4 — Interactive prompt.** `downshift export ./model.pt` with no inputs and no adapter match →
drop into a short wizard asking for shapes/dtypes. (Olive does something similar with
`olive init`; it's a good pattern.)

**Tier 5 — Fail loudly** with a message that says exactly what to pass.

### 5.3 `dynamic_shapes` inference

`dynamic_axes` is deprecated in the dynamo path; `dynamic_shapes` (with `torch.export.Dim`) is
the replacement, and misusing it produces `torch._dynamo.exc.UserError: Constraints violated`.

Heuristics for marking a dim dynamic:
- Batch dim (usually dim 0) of every input → dynamic by default.
- Sequence dim for HF models → dynamic (from `OnnxConfig`).
- **GNNs: node count `N` and edge count `E` are dynamic *and independent*.** `x` is `(N, F)`,
  `edge_index` is `(2, E)`. `N` and `E` must be separate `Dim` objects with no relation
  constraint. Getting this wrong is why so many exported GNNs work on one input shape and throw
  `INVALID_ARGUMENT` on the next.
- Allow explicit override: `--dynamic "x:0,edge_index:1"`.

### 5.4 Container flattening (the PyG/DGL problem)

`torch.onnx.export` wants tensors. PyG's forward often takes a `Data` or `Batch` object; DGL's
takes a `DGLGraph`. These are non-tensor containers holding tensors.

Solution: each adapter provides a **flatten/unflatten pair**.

```python
class PyGAdapter(Adapter):
    def flatten(self, data: Data) -> tuple[Tensor, ...]:
        return (data.x, data.edge_index, getattr(data, "edge_attr", None), data.batch)
    def unflatten(self, tensors) -> Data: ...
    def wrap(self, model) -> nn.Module:
        """Return a shim module whose forward takes flat tensors."""
```

Export the *shim*, not the original model. The shim's forward signature is a flat tensor tuple,
which `torch.export` handles cleanly. Serving reverses the flattening at request time.

This is reusable well beyond GNNs — any model with dataclass/dict inputs benefits.

### 5.5 Numerical verification — MANDATORY, not optional

**This is the most important design decision in the whole project.**

There is a known class of bug where PyG models (e.g. GAT) export "successfully" to ONNX and then
produce *different results* — sometimes wildly different depending on input data — because of
`scatter_reduce` translation issues. A separate known failure: ONNX doesn't support
`include_self=False` for `scatter_reduce`, which at least fails loudly. The silent-wrong-answer
case is far more dangerous.

Therefore: **a successful export is not a success until numerics are verified.**

Verification protocol:
1. Generate *K* random input samples (default K=8), varying dynamic dims across the sample set.
   Critically, include shapes *different from* the export-time example — this is what catches
   frozen-shape bugs.
2. Run both PyTorch (eval mode, no_grad) and ONNX Runtime.
3. Compare with `torch.testing.assert_close`-style tolerances (`rtol`/`atol` configurable,
   dtype-aware defaults).
4. Record max absolute error, max relative error, and error distribution.
5. If any sample exceeds tolerance → verdict is `DEGRADED`, not `CLEAN`.

For a stochastic model, seed everything and disable dropout. For models where exact-match is
impossible (fp16, int4 quantization), tolerances scale with the precision tier and we report the
observed deviation rather than hard-failing.

### 5.6 The Verdict object — core IP

```python
@dataclass
class ExportVerdict:
    status: Literal["CLEAN", "DEGRADED", "FAILED", "UNVERIFIED"]
    #  UNVERIFIED = input was already an .onnx file with no reference torch model supplied;
    #  served as-is, numerics never checked. Distinct from CLEAN — never conflate the two
    #  in the CLI banner or the compatibility matrix.
    model_family: str              # "hf-transformers" | "pyg" | "dgl" | "generic-torch"
    capture_strategy: str          # which torch.export strategy succeeded
    onnx_path: Path | None
    opset: int | None
    dynamic_dims: dict[str, list[int]]
    numerics: NumericsReport       # max_abs_err, max_rel_err, samples_tested, failures
    shape_generalization: bool     # did it survive shapes != export shape?
    unsupported_ops: list[str]     # ops that forced decomposition or failed
    warnings: list[str]
    recommended_backend: Literal["onnxruntime", "torch"]
    reason: str                    # human-readable, one sentence
```

Four tiers:

- **`CLEAN`** — exports, verifies numerically, generalizes across shapes. Serve via ONNX Runtime.
- **`DEGRADED`** — exports but something is off: numerics drift beyond tolerance, or shapes are
  effectively frozen, or ops were decomposed into slow primitives. *Report it prominently.*
  Default to torch backend; allow `--force-onnx` with a loud warning.
- **`FAILED`** — cannot export. Serve via native PyTorch. Not an error condition; a supported path.
- **`UNVERIFIED`** — input was already `.onnx`, no source model to check numerics against
  (§5.7). Served as-is. Not a quality claim either way — just an honest statement of what
  wasn't checked.

The `DEGRADED` tier is the thing that doesn't exist anywhere else and the reason someone would
pick this over hand-writing an `onnxruntime.quantization.quantize_dynamic` script themselves.

### 5.7 No built-in optimization — accept pre-optimized artifacts as input (DECISION)

**Olive is dropped entirely — not vendored, not shelled out to, not an optional extra.**
Reasoning: most of what Olive buys on non-LLM models is directly reachable via
`onnxruntime.quantization.quantize_dynamic` (int8), `onnxconverter_common.float16` (fp16),
and `SessionOptions` graph-optimization levels — but building even that thin layer ourselves
adds surface area for a marginal, easily-replicated feature. Olive's real concentration of
value (GPTQ/AWQ/int4, NPU/QNN/DirectML targets) is LLM- and edge-specific — outside this
project's niche, and outside v0.1's scope regardless.

**The contract instead: `downshift` has two input paths, not one.**

1. **Raw PyTorch / HF model** (`.pt`, checkpoint dir, HF repo) → full pipeline: export (§5.1–5.6),
   mandatory numerical verification against the source model, manifest, serve. This is the
   path with a verdict.
2. **Pre-optimized `.onnx` file** (produced by Olive, TensorRT-Model-Optimizer, a hand-rolled
   script, whatever) → served directly via `OnnxRuntimeBackend`, **no export, no numerical
   verification** — there's no PyTorch reference to verify against unless the user also
   supplies the original checkpoint (`--reference model.pt`), in which case verification runs
   exactly as in path 1. Verdict status for an unreferenced `.onnx` input is a fourth state:
   **`UNVERIFIED`** — distinct from `CLEAN`, and the banner says so plainly.

**Why this is a *better* identity than "we optimize too," not a lesser one:** "we don't
optimize, we verify and serve" is a sharper, one-sentence pitch than "we do a bit of
everything Olive does, slightly worse." Verifying somebody else's int4-quantized ONNX against
the original weights is exactly the service nobody else provides, and it works identically
regardless of which tool produced the artifact — the verification layer doesn't care whether
Olive, TensorRT-Model-Optimizer, or a grad student's notebook made the `.onnx` file.

**What this removes from scope entirely (not deferred — cut):** `--precision int4/int8/gptq`
export flags, any Olive dependency, any custom quantization code. `downshift export` produces
a plain export (fp32, with an optional `--fp16` flag that's just a `.half()` cast before
tracing — not a quantization pass) plus the verdict and manifest. Anyone who wants int4 runs
their own tool and feeds the result to `downshift serve model.onnx --reference model.pt`.

### 5.8 Export layer risk register

| Risk | Mitigation |
|---|---|
| `torch.export` breaks on custom autograd / data-dependent control flow | Ladder down to TorchScript (`fallback=True`), then to `FAILED` + torch backend |
| Tied weights (GPT-2, OPT) duplicate tensors on export | Detect via param identity check; warn; point at an external optimizer's model-builder path if the user needs it |
| ONNX opset lags ATen (RoPE, GQA, SwiGLU decompose into dozens of primitives) | Detect decomposition blowup via node-count ratio; flag as `DEGRADED` for perf reasons |
| `scatter_reduce(include_self=False)` unsupported | Detect in pre-flight op scan; give a specific, actionable error naming the offending layer |
| Silent numerical drift in GNNs | Mandatory verification (§5.5) |
| PyTorch version churn breaks our assumptions | Pin a supported torch range; CI matrix across 2–3 minor versions |

---

## 6. The Serving Layer

### 6.1 Backends

A `Backend` protocol with three implementations:

```python
class Backend(Protocol):
    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
    def metadata(self) -> BackendMeta: ...
```

1. **`OnnxRuntimeBackend`** — `InferenceSession` with EP selection
   (`CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`,
   `OpenVINOExecutionProvider`). IO binding for GPU to avoid host round-trips.
2. **`TorchBackend`** — native eager `nn.Module` under `torch.inference_mode()`. Optional
   `torch.compile`. This is the fallback path and it must be first-class, not a stub.
3. **`GenAIBackend`** — delegate to `onnxruntime-genai` for causal LMs. It manages the full
   generative loop: pre/post-processing, inference, sampling, and KV cache management. Building
   our own token loop would be strictly worse.

Backend selection: `verdict.recommended_backend`, overridable by `--backend`.

### 6.2 Request batching — be honest about what this is

**Dynamic batching, not continuous batching.** Server-side request coalescing:

- Queue incoming requests up to `--max-batch-size` or `--max-wait-ms`, whichever hits first.
- Stack compatible requests, run one forward, scatter results back.
- Requires shape compatibility checking — reject or separate-queue mismatched shapes.

We are not implementing PagedAttention. Say so in the README. For LLM workloads that need real
continuous batching, the README should point users at vLLM without embarrassment — being honest
about scope buys more credibility than overclaiming.

### 6.3 Graph batching — a real differentiator

Batching GNN requests is *not* stacking tensors. Multiple graphs batch as a single
block-diagonal graph: concatenate node features, offset each graph's `edge_index` by the running
node count, and build a `batch` vector mapping nodes → source graph. This is what PyG's
`Batch.from_data_list` does, and no generic serving framework does it correctly.

Implement `GraphBatcher` as the PyG/DGL adapter's batching strategy. This alone is a legitimately
novel serving feature and worth a blog post.

### 6.4 HTTP surface

FastAPI + uvicorn.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Generic tensor in/out. JSON for small payloads; `application/octet-stream` (msgpack or raw numpy) for large ones. |
| `POST /predict/graph` | Graph-shaped payload: `{x, edge_index, edge_attr?, batch?}`. Handles offsetting. |
| `POST /v1/chat/completions` | OpenAI-compatible, LLM models only, backed by `GenAIBackend`. Streaming via SSE. |
| `POST /v1/completions` | Same. |
| `GET /health` | Liveness. |
| `GET /ready` | Readiness (model loaded, warmup complete). |
| `GET /metadata` | Model family, backend, verdict summary, input/output specs. |
| `GET /metrics` | Prometheus: latency histogram, queue depth, batch size distribution, error rate. |

OpenAI compatibility for the LLM path matters — it's a large part of why vLLM got adopted without
anyone rewriting client code.

### 6.5 Warmup

Run *n* dummy inferences at startup before flipping `/ready` to true. ONNX Runtime EPs (especially
TensorRT) do expensive first-call compilation. Without warmup the first real request eats a
multi-second penalty and every benchmark looks wrong.

---

## 7. The CLI

Typer + rich. This is the part that should feel *fancy*, because first impressions drive adoption.

### 7.1 Commands

```bash
# The hero command — one line to a live endpoint
downshift serve meta-llama/Llama-3.2-1B-Instruct --port 8000
downshift serve ./gnn_checkpoint.pt --adapter pyg --port 8000
downshift serve ./model.pt --backend torch          # skip ONNX entirely

# Serve a pre-optimized ONNX artifact directly — UNVERIFIED unless a reference is given
downshift serve ./model.onnx --port 8000
downshift serve ./model.onnx --reference ./model.pt # re-enables numerical verification

# Standalone export — for people who just want the artifact (no quantization; see §5.7)
downshift export meta-llama/Llama-3.2-1B-Instruct --fp16 -o ./exported
downshift export ./gnn.pt --adapter pyg --dynamic "x:0,edge_index:1"
downshift export ./model.pt --no-verify             # escape hatch, prints a warning

# Triage without exporting anything — fast, read-only
downshift check ./model.pt
downshift check meta-llama/Llama-3.2-1B-Instruct

# Throughput / latency measurement
downshift bench ./optimized --requests 500 --concurrency 16
downshift bench ./optimized --compare torch          # ONNX vs eager, side by side

# Interactive wizard for the confused
downshift init
```

### 7.2 What makes it feel good

**Boot banner.** A rich table printed at startup, vLLM-style:

```
╭─ downshift v0.1.0 ────────────────────────────────────────────╮
│ Model          bert-base-uncased                              │
│ Family         hf-transformers (encoder)                      │
│ Export         CLEAN · strict=False · opset 20                │
│ Numerics       max_abs_err 4.1e-05 over 8 samples  ✓          │
│ Backend        onnxruntime · CUDAExecutionProvider             │
│ Dynamic dims   input_ids[0,1]                                 │
│ Endpoint       http://0.0.0.0:8000                            │
╰─────────────────────────────────────────────────────────────╯
```

There is no LLM/quantization row in v0.1's banner — no Olive, no int4, no OpenAI-compatible
endpoint. Those return in v0.5+ only if the GenAI backend gets built (§9). Don't imply them
in the README before they exist.

**The `UNVERIFIED` case** — an honest label the moment someone hands it a pre-optimized `.onnx`
with no source model to check against:

```
╭─ downshift v0.1.0 ────────────────────────────────────────────╮
│ Model          ./quantized_bert.onnx                          │
│ Verdict        UNVERIFIED — no reference model supplied       │
│                └ served as-is; numerics were never checked    │
│ Backend        onnxruntime · CPUExecutionProvider              │
│ Tip            pass --reference <model.pt> to verify           │
╰─────────────────────────────────────────────────────────────╯
```

And the failure case is just as informative — this is the moment the product proves its value:

```
╭─ downshift v0.1.0 ──────────────────────────────────────────────╮
│ Model          ./gat_fraud_v3.pt                           │
│ Family         pyg (GATConv ×3)                            │
│ Export         DEGRADED  ⚠                                 │
│                └ numerics: max_abs_err 0.41 on 3/8 samples │
│                └ likely cause: scatter_reduce translation  │
│ Backend        torch (eager)  ← auto-selected              │
│ Override       --force-onnx to serve the ONNX graph anyway │
╰────────────────────────────────────────────────────────────╯
```

**Other polish:**
- Progress bars (`rich.progress`) during download, export, quantization — never a silent hang.
- Grouped, colorized `--help` via `rich-click` or Typer's rich formatting.
- `downshift check` outputs a table; `downshift check --json` outputs machine-readable for CI use.
- Non-zero exit codes that mean something (`0` clean, `1` failed, `2` degraded) so `downshift check` can
  gate a CI pipeline.
- Structured logging with a `--log-level` flag; JSON logs behind `--log-format json`.

---

## 8. Package Structure

```
downshift-server/
├── pyproject.toml            # hatchling; extras: [optimize] [gnn] [genai] [all]
├── src/downshift/
│   ├── __init__.py           # public API: load, export, serve, check
│   ├── cli/
│   │   ├── main.py           # Typer app
│   │   ├── serve.py  export.py  bench.py  check.py  init.py
│   │   └── render.py         # ALL rich formatting lives here, nowhere else
│   ├── adapters/
│   │   ├── base.py           # Adapter protocol
│   │   ├── registry.py       # detection + dispatch
│   │   ├── hf.py  pyg.py  dgl.py  generic.py  torchscript.py
│   ├── export/
│   │   ├── inputs.py         # dummy input synthesis (§5.2)
│   │   ├── shapes.py         # dynamic_shapes inference (§5.3)
│   │   ├── capture.py        # torch.onnx.export driver + report harvesting
│   │   ├── verify.py         # numerical verification (§5.5)
│   │   ├── prevalidated.py   # intake for already-optimized .onnx (§5.7, UNVERIFIED path)
│   │   └── verdict.py        # ExportVerdict — CLEAN/DEGRADED/FAILED/UNVERIFIED (§5.6)
│   ├── serve/
│   │   ├── backends/         # ort.py  torch.py  genai.py
│   │   ├── batching.py       # DynamicBatcher
│   │   ├── graph_batching.py # GraphBatcher (§6.3)
│   │   ├── app.py            # FastAPI
│   │   └── schemas.py        # pydantic request/response models
│   ├── bench/
│   └── _typing.py
├── tests/
│   ├── models/               # tiny fixture models, one per hazard class
│   ├── test_export_*.py
│   ├── test_serve_*.py
│   └── test_cli_*.py
├── docs/
│   └── compatibility.md      # THE compatibility matrix (§10)
├── examples/
└── .github/workflows/        # CI matrix + PyPI OIDC trusted publisher
```

**Dependency discipline.** Base install must be light: `torch`, `onnx`, `onnxruntime`, `typer`,
`rich`, `fastapi`, `uvicorn`, `numpy`, `pydantic`. Everything else is an extra. A user serving a
plain `nn.Module` should not be forced to install `transformers` and `torch-geometric`. No Olive
dependency exists anywhere in the tree (§5.7).

---

## 9. Milestones

### v0.1 — Export core, library only, no CLI (2–3 weekends)
- `Adapter` protocol + `generic` and `hf` adapters
- Dummy input synthesis tiers 1–3
- `torch.onnx.export` driver with report harvesting
- Numerical verification
- `ExportVerdict`
- **Ship criterion:** `downshift.export(model)` returns a correct verdict for 10 fixture models,
  including at least two designed to fail.

### v0.2 — Serving core (2–3 weekends)
- `OnnxRuntimeBackend` + `TorchBackend`
- FastAPI `/predict`, `/health`, `/metadata`
- Warmup
- Dynamic batching
- **Ship criterion:** a model that fails export still serves correctly, same endpoint contract.

### v0.3 — The CLI (1–2 weekends)
- `serve`, `export`, `check`
- Boot banner, progress bars, exit codes
- **Ship criterion:** `pip install downshift-server && downshift serve <hf-model>` works cold on a clean machine.
- **This is the first public release.** Post it.

### v0.4 — GNN support (2–3 weekends)
- `pyg` and `dgl` adapters with flatten/unflatten
- Independent `N`/`E` dynamic dims
- `GraphBatcher` + `/predict/graph`
- Pre-flight scan for known-hazardous ops
- **Ship criterion:** a 3-layer GraphSAGE node classifier serves end-to-end with correct
  batched results across variable graph sizes.
- **This is the differentiated release.** Write the blog post here.

### v0.5 — Pre-optimized artifact ingestion + LLM path (stretch)
- Harden the `--reference` path: verify a pre-optimized `.onnx` (from Olive, TensorRT-Model-
  Optimizer, anything) against a supplied source checkpoint. This is where "your int4
  quantization introduced a 12% max deviation on GAT outputs" becomes a real, demoable feature.
- `GenAIBackend` (`onnxruntime-genai`) + OpenAI-compatible endpoints for causal LMs — independent
  of whether *we* ran any optimization; just a serving backend for whatever `.onnx` shows up.
- `bench` command

### v0.6 — Production polish
- Prometheus metrics
- Dockerfile + compose example
- Graceful shutdown, request timeouts, concurrency limits
- Optional: revisit device-routing/OOM-prevention logic (parked, formerly its own project) as
  a module on the torch backend — natural fit, since GNN memory footprints vary wildly with
  graph size

---

## 10. The Compatibility Matrix

**Build this before writing serving code.** It is simultaneously the test suite, the
documentation, and the honesty mechanism.

A table in `docs/compatibility.md`, generated by a CI job that runs `downshift check` against a corpus:

| Model | Family | Export | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|
| `resnet18` | torchvision | CLEAN | 1e-6 | ✓ | ort |
| `bert-base-uncased` | hf | CLEAN | 4e-5 | ✓ | ort |
| `Llama-3.2-1B` | hf-causal | CLEAN | — | ✓ | genai |
| `GCNConv ×2` | pyg | CLEAN | 2e-6 | ✓ | ort |
| `GATConv ×3` | pyg | DEGRADED | 4e-1 | ✗ | torch |
| `SAGEConv ×2` | pyg | DEGRADED | — | ✗ | torch |
| `AttentionalAggregation` | pyg | FAILED | — | — | torch |

Regenerate on every release. If a PyTorch update fixes `scatter_reduce`, the matrix shows it
immediately and the fix becomes a release note.

---

## 11. Testing Strategy

**Fixture models, one per hazard class** — each tiny (< 1s to run), each isolating one failure mode:

- Clean feedforward (control)
- Data-dependent control flow (`if x.sum() > 0`)
- Dynamic shape dependence
- Custom autograd function
- `scatter_reduce` with `include_self=False` (known hard fail)
- Tied weights
- Dict/dataclass input (flattening test)
- Stochastic layer (dropout — must be disabled in eval)

**Test levels:**
1. **Unit** — input synthesis, shape inference, verdict logic, batching math (especially
   `edge_index` offsetting; test it exhaustively, it's easy to get subtly wrong).
2. **Integration** — full export → verify → serve → assert response, per fixture.
3. **Contract** — the same request against ORT backend and torch backend must produce equivalent
   responses. This guarantees the fallback is genuinely transparent.
4. **CLI** — Typer's `CliRunner`; assert exit codes and JSON output, not banner text.

**CI matrix:** Python 3.10–3.13 × torch {stable, stable-1} × {cpu, cuda-if-available}. GPU tests
optional/skipped on public runners — document how to run them locally.

---

## 12. Benchmarking

`downshift bench` should measure and report:
- p50 / p95 / p99 latency
- Throughput (req/s) at a given concurrency
- Cold-start time (import → ready)
- Peak memory
- `--compare torch`: side-by-side ONNX vs eager

**Be scientific in the README.** Report the machine, the batch size, the EP, and the model. Don't
cherry-pick. A benchmark table showing "ONNX is 1.4× faster on BERT, 0.9× on this GNN" is more
credible — and more useful — than a marketing number, and the negative result is itself an
interesting finding worth writing about.

---

## 13. Risks and Kill Criteria

| Risk | Severity | Response |
|---|---|---|
| An incumbent (Olive, `anydeploy`, or similar) ships a verdict/fallback system like ours | High | Our GNN adapters + `DEGRADED`/`UNVERIFIED` distinction are still the harder-to-replicate part; pivot messaging further toward GNN triage specifically |
| `torch.export` matures enough that everything just works | Medium | Good for users; the value shifts to serving + batching, which still has no universal option |
| Scope creep into building a real inference engine | **High** | Hard rule: no custom CUDA kernels, no attention implementations. If a task requires either, it's out of scope |
| GNN ONNX support stays broken indefinitely | Medium | Fine — that's the case *for* the fallback path, not against it |
| Nobody needs "universal" and everyone uses one family | Medium | Watch adoption; if 90% is HF, narrow the marketing, keep the architecture |

**Kill criteria — be willing to stop.** If after v0.3 nobody outside your own use cases tries it,
the honest read is that the DX gap wasn't real. Salvage value is still high: the export triage
logic and the numerical verification harness are independently useful and publishable as a much
smaller focused tool (`onnx-triage`).

---

## 14. Open Design Questions

1. **Does the torch fallback backend belong in v1 at all, or does it dilute the "ONNX serving"
   message?** Current lean: it's essential — it's what makes "any torch model" true rather than
   aspirational. But it does make the elevator pitch harder.
2. **RESOLVED — no quantization ownership at all.** We don't quantize; we verify what others
   produce (§5.7). `downshift export` defaults to fp32, `--fp16` is a plain cast, nothing lower.
   Removes an entire category of "silent degradation" risk by not being in that business.
3. **Multi-model serving in one process?** Deferred. Needs a model registry, per-model routing, and
   memory accounting. Not v1.
4. **Should `check` be able to run against a HF repo without downloading full weights?** Would be
   a great fast-triage feature (config-only static analysis of the architecture). Possibly a
   later "static mode."
5. **Streaming for non-LLM models?** Probably unnecessary. Skip unless asked.
6. **Rust core for the batcher?** Tempting for the perf story, kills contributor accessibility.
   No, at least not before v1.0.

---

## 15. First Concrete Steps

1. Check PyPI name availability; reserve with a stub `0.0.1`.
2. Scaffold repo: `pyproject.toml` (hatchling), ruff, mypy, pytest, GH Actions with OIDC
   trusted-publisher flow.
3. Write the **eight fixture models** in `tests/models/` first — before any export code. They
   define the problem.
4. Implement `ExportVerdict` + `verify.py`. Numerical verification is the heart; build it first.
5. Implement `capture.py` around `torch.onnx.export(dynamo=True)` with report harvesting.
6. Run it against the fixtures. Publish the resulting compatibility matrix as the first commit to
   `docs/`. That table *is* the project's opening argument.
