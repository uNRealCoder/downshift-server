# v0.1 Sprint Plan — Ship by Monday Night

**Window:** Fri 11 Sep (evening) → Mon 14 Sep (evening)
**Budget:** ~22 hours optimistic, ~16 productive
**Definition of shipped:** public GitHub repo + working README + generated compatibility matrix + blog post draft
**PyPI:** stretch goal, Tuesday. Do not let packaging eat Sunday.
**Package:** `downshift-server` (PyPI) · **CLI/import:** `downshift` — DECIDED, reserve in H1.

---

## 1. Scope Lock

### IN (v0.1)

- Export triage: `torch.onnx.export(dynamo=True)` driver + report harvesting
- Numerical verification across varying dynamic shapes
- `ExportVerdict`: `CLEAN` / `DEGRADED` / `FAILED` / `UNVERIFIED`
- Adapters: `generic` (any `nn.Module`), `pyg` (GCNConv, SAGEConv, GATConv), `hf` (encoder-only)
- Adapter plugin discovery via entry points
- Backends: `OnnxRuntimeBackend` (CPU + CUDA EP), `TorchBackend` (eager fallback)
- Two input paths: raw torch model (full verify pipeline) **and** pre-optimized `.onnx`
  (served as-is, `UNVERIFIED` unless `--reference <model.pt>` is also given)
- Serving: FastAPI — `/predict`, `/predict/graph`, `/health`, `/ready`, `/metadata`
- Middleware injection (`--middleware`, vLLM-style)
- Artifact manifest (provenance sidecar JSON)
- CLI: `check`, `export`, `serve` + rich boot banner + meaningful exit codes
- `weights_only=True` by default on checkpoint load
- Generated `docs/compatibility.md`

### OUT (explicitly deferred or cut outright, say so in README)

| Cut | Why |
|---|---|
| **Any quantization/optimization — Olive or self-built** | **Cut permanently, not deferred.** We accept pre-optimized `.onnx` as input instead of producing it ourselves (see design doc §5.7). No `--precision int8` flag, no Olive dependency, ever. |
| LLM path + `onnxruntime-genai` | v0.5+, independent of the (now nonexistent) quantization plan |
| OpenAI-compatible endpoints | Follows the LLM path out |
| Dynamic request batching | v0.2 |
| `GraphBatcher` | v0.2 — `/predict/graph` accepts one graph for now |
| `bench` command | v0.2 |
| Prometheus metrics | v0.2 — middleware hook makes this easy to add later |
| DGL adapter | v0.2 |
| `init` wizard | v0.3 |
| Docker | v0.2 — pip + version pins is the primary path for now (no Docker resources) |

**Do not put "vLLM-like" in the README.** No continuous batching, no PagedAttention, no LLM path.
People will arrive expecting them and leave annoyed. The CLI is a visual homage; say that plainly.

---

## 2. Hour-by-Hour

### Friday evening — 3h · Foundation

**H1 — Name + scaffold**
- Pick the name. Check PyPI. Reserve only if it takes < 10 min; otherwise move on.
- `src/` layout, `pyproject.toml` (hatchling), ruff, pytest, mypy (lenient for now)
- GH Actions: lint + test on 3.11/3.12, CPU only
- Push an empty-but-green repo. **Gate: CI green before writing feature code.**

**H2–H3 — Fixture models + walking skeleton**

Write eight fixtures in `tests/models/` — each tiny, each isolating one hazard:

| Fixture | Hazard | Expected verdict |
|---|---|---|
| `clean_mlp` | none (control) | CLEAN |
| `dynamic_batch_cnn` | batch-dim generalization | CLEAN |
| `data_dependent_branch` | `if x.sum() > 0` | FAILED or DEGRADED |
| `custom_autograd` | custom `autograd.Function` | FAILED |
| `tied_weights` | shared embedding/output weight | CLEAN + warning |
| `dict_input` | dataclass/dict forward arg | CLEAN (via flatten) |
| `dropout_model` | stochastic — must be eval'd | CLEAN |
| `scatter_include_self_false` | `scatter_reduce(include_self=False)` | FAILED, loud |

Then the **walking skeleton**: hand-export `clean_mlp`, load in ORT, serve behind a 20-line FastAPI
app, `curl` it. Ugly, hardcoded, throwaway.

**Gate: end-to-end pipe proven before any abstraction exists.** This is the single most important
hour of the weekend — it de-risks every integration assumption at once.

---

### Saturday — 8h · Export core (the product)

**H4–H6 — `export/inputs.py`, `export/shapes.py`, `export/capture.py`**

```python
# inputs.py — synthesis ladder
def synthesize(model, adapter, user_inputs=None) -> tuple[Tensor, ...]
#   T1 user-supplied → T2 adapter-derived → T3 signature introspection → T5 fail loudly
#   (T4 interactive wizard deferred to v0.3)

# shapes.py
def infer_dynamic_shapes(model, adapter, inputs) -> dict
#   batch dim 0 dynamic by default
#   GNN: N and E as INDEPENDENT torch.export.Dim objects, no relation constraint
#   override via --dynamic "x:0,edge_index:1"

# capture.py
def capture(model, inputs, dynamic_shapes, opset) -> CaptureResult
#   torch.onnx.export(..., dynamo=True, report=True, verify=True,
#                     artifacts_dir=tmp, fallback=True)
#   parse the report → which strategy won, which ops decomposed, node count
```

Do not reimplement the strategy ladder — `torch.onnx.export` already runs
`strict=False → strict=True → draft_export → decompose → checker → ORT → accuracy`. Harvest it.

**H7–H8 — `export/verify.py` + `export/verdict.py`**

This is the heart. Build it carefully; a wrong verdict is worse than no tool.

```python
def verify(torch_model, onnx_path, adapter, k=8) -> NumericsReport
```
- Generate K samples, **varying dynamic dims — include shapes ≠ the export-time example.**
  This is what catches frozen-shape bugs.
- `model.eval()`, `torch.inference_mode()`, seed everything
- Compare max_abs_err / max_rel_err; dtype-aware tolerances
- Any sample over tolerance → `DEGRADED`, never `CLEAN`

`ExportVerdict` per the design doc: status, family, capture_strategy, numerics,
shape_generalization, unsupported_ops, recommended_backend, one-sentence reason.

**H9–H10 — PyG adapter + GNN fixtures**

`adapters/pyg.py`: `flatten`/`unflatten`/`wrap`. Export the **shim** (flat tensor forward),
never the raw model — `Data`/`Batch` containers don't survive `torch.export`.

Fixtures: 2-layer GCN, 2-layer SAGE, 3-layer GAT.

**This is the money moment.** If GAT comes back `DEGRADED` with real numerical divergence,
you have your blog post and your headline. Screenshot the terminal output when it happens.

**H11 — CLI `check` + `cli/render.py`**

All rich formatting lives in `render.py`. `check` returns a structured object; render prints it.
Exit codes: `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED. `--json` for machine output.

**Gate: `downshift check` produces a correct verdict table for all 11 fixtures (8 base hazard
fixtures + GCN/SAGE/GAT). If this isn't done by Saturday night, cut the HF adapter and
`/predict/graph` on Sunday without hesitation.**

---

### Sunday — 8h · Serving + CLI

**H12 — `export/manifest.py`**

Sidecar JSON next to every `.onnx`: source model SHA256 (or `null` if input was already
`.onnx` with no reference), torch/onnx/ORT versions, opset, EP target, observed dtype
(fp32/fp16 — whatever the export actually produced, not a quantization setting), full verdict
including `UNVERIFIED` where applicable, numerics report, UTC timestamp, package version.
Cheap now, and it's the feature a regulated buyer asks about first.

**H13–H14 — Backends**

`Backend` protocol; `OnnxRuntimeBackend` (EP selection, IO binding for CUDA) and `TorchBackend`
(`inference_mode`, device placement). Warmup: *n* dummy inferences before `/ready` flips true.

**Timebox CUDA EP setup to 60 minutes.** `onnxruntime-gpu` version/CUDA pinning is a known
time sink. If it isn't working at the hour mark, ship CPU-only and note it in the README —
do not let it eat Sunday.

**H15–H16 — FastAPI app + middleware**

Endpoints: `/predict` (generic tensors), `/predict/graph` (single graph payload),
`/health`, `/ready`, `/metadata` (family, backend, verdict summary, IO specs).

Middleware loader:
```python
# --middleware mypkg.auth:APIKeyMiddleware  (repeatable)
for spec in middleware:
    obj = import_from_string(spec)
    if inspect.isclass(obj) and issubclass(obj, BaseHTTPMiddleware):
        app.add_middleware(obj)
    elif inspect.iscoroutinefunction(obj):
        app.middleware("http")(obj)
    else:
        raise ValueError(...)
```
Empty list → no-op passthrough, zero overhead.

**H17 — CLI `serve` + boot banner**

The banner is the first impression — spend real time on it. Both the CLEAN case and the
DEGRADED case (see the design doc for both layouts). The DEGRADED banner *is* the product demo.

**H18 — CLI `export`** — standalone artifact + manifest, `--fp16` (plain `.half()` cast, not
quantization — nothing lower than fp16 exists in this tool, see design doc §5.7), plus
`--no-verify` escape hatch with a loud warning. Also wire `downshift serve model.onnx
[--reference model.pt]` here — the `UNVERIFIED`-vs-verified branch for pre-optimized input.

**H19 — HF adapter (encoder-only) + entry-point registration**

Skip `optimum`; build `input_ids`/`attention_mask` dummies directly from the tokenizer config.
Target: `bert-base-uncased`, `distilbert`. Register adapters via
`[project.entry-points."downshift.adapters"]` so third parties can extend without a PR.

**Most droppable hour of the weekend. Cut it first if behind.**

---

### Monday evening — 3h · Ship

**H20 — Matrix generation**
`scripts/gen_matrix.py` runs `check` across the corpus → writes `docs/compatibility.md`.
Wire it to a scheduled weekly CI job (the matrix is your core claim; a stale one is worse than none).

**H21 — README**
Lead with the DEGRADED banner screenshot, not the install instructions. Then: what it does,
the matrix, explicit "not vLLM / no LLM path / no quantization — accepts pre-optimized ONNX
as input instead" scope note, a one-line "vs. `anydeploy`" positioning note, security note
on `weights_only`.

**H22 — Blog post draft + buffer**
Working title: *"Your GNN exports to ONNX cleanly. It's also wrong."*
Lead with the GAT numerical divergence, generalize to why export success ≠ correctness,
land on the tool. Publish Tuesday after a re-read.

---

## 3. Cut Ladder

Fall down this list, in order, the moment you're behind:

1. HF adapter (H19)
2. `/predict/graph` — `/predict` covers it with manual flattening
3. CUDA EP — CPU-only, documented
4. `export` CLI command — `serve` exports inline anyway
5. `tied_weights` + `dict_input` fixtures
6. GAT fixture — **never cut this one; it's the headline**

## 4. Non-Negotiables

- Numerical verification correctness. Everything else can be rough.
- The GAT / `DEGRADED` result. It is the entire differentiation.
- `weights_only=True` default. A security footgun in a v0.1 follows you.
- Honest README scope section.
- CI green at every push.

## 5. Anti-Patterns for This Weekend

- Building the `Adapter` abstraction before two adapters exist. Write `generic` and `pyg`
  concretely, extract the protocol on the third.
- Polishing the banner at H5 instead of H17.
- Perfecting `pyproject.toml` extras. One `[all]` extra is fine for now.
- Chasing a CUDA install past the timebox.
- Starting the blog post before the GAT result exists — you don't yet know what it says.
