# downshift — a guided tour of the code

This is the doc I'd hand you on day one. It explains how the package is laid out, why each
piece exists, and the handful of decisions that look odd until you know what they're working
around. `README.md` tells you what the tool does from the outside; this tells you what it
does on the inside.

Read it top to bottom the first time. After that, §2 (the map) and §11 (invariants) are the
bits you'll come back to.

> **As of / line numbers.** Written against `b0c1faa` ("Updated code simplified and session
> threads"), which is where `settings.py` and the current `cli/main.py`,
> `export/{verdict,verify,prevalidated}.py` and `serve/{engine,backends}.py` landed. Every
> `file.py:NN` reference below was checked against that commit. Code moves; the **symbol names
> are authoritative** — if a line number looks wrong, grep for the function or class name
> rather than trusting the number.

---

## 1. The one idea

Everything in this codebase exists to produce one object and then act on it:

```python
ExportVerdict(status="DEGRADED", recommended_backend="torch", numerics=NumericsReport(...), ...)
```

The premise of the tool is that **`torch.onnx.export` succeeding is not evidence that the
export is correct.** Our own `scatter_include_self_false` fixture exports with zero errors,
zero warnings, and returns wrong numbers on 7 of 8 inputs. So "did it export?" is not the
question we answer. The question is "should you trust it?", and the answer is one of four
statuses:

| Status | Means | Serve via |
|---|---|---|
| `CLEAN` | Exports, matches PyTorch on every sample, survives shapes it wasn't traced on | ONNX Runtime |
| `DEGRADED` | Exports fine, numbers drift past tolerance on ≥1 sample | eager PyTorch |
| `FAILED` | Won't export at all | eager PyTorch |
| `UNVERIFIED` | A `.onnx` with no reference, or `--no-verify` | ONNX Runtime, labelled |

Two consequences shape the whole design:

1. **Verification is not optional and not a side quest.** It sits in the middle of the export
   path, not bolted on after. `build_verdict` refuses to return `CLEAN` without it.
2. **`FAILED` is a supported outcome, not an error.** Eager PyTorch is a first-class backend
   behind the same HTTP endpoint. That's why there's a serving layer at all — without the
   fallback, a verdict would just be a report.

If you only remember one thing: `ExportVerdict` is the spine. Every layer either produces it
or reads it.

---

## 2. Map of the repo

```
src/downshift/
├── __init__.py          public API: check(), export(), intake(), the dataclasses
├── loading.py           "what did the user actually point me at?" → LoadedModel
├── settings.py          DOWNSHIFT_* env defaults (tolerances, host/port, threads)
│
├── adapters/            model-family knowledge: how to build inputs, how to flatten
│   ├── base.py          the Adapter protocol + the Prepared dataclass
│   ├── registry.py      discovery, ordering, entry-point plugins
│   ├── _flatten.py      the shim that gives torch.export a flat tensor signature
│   ├── generic.py       any nn.Module (always matches; always tried last)
│   ├── pyg.py           PyTorch Geometric — independent node/edge dynamic dims
│   └── hf.py            Hugging Face encoder-only
│
├── export/              produce and judge the ONNX graph
│   ├── inputs.py        the example-input ladder
│   ├── shapes.py        dynamic-dim inference, the --dynamic override, trace-safety
│   ├── capture.py       drives torch.export + torch.onnx.export, reports which worked
│   ├── verify.py        the numerics check — the heart of the product
│   ├── verdict.py       ExportVerdict + build_verdict(): capture, verify, decide
│   ├── manifest.py      provenance sidecar written next to every .onnx
│   └── prevalidated.py  intake for a .onnx someone else produced
│
├── serve/               turn a verdict into a running server
│   ├── backends.py      OnnxRuntimeBackend + TorchBackend behind one Protocol
│   ├── engine.py        verdict → backend choice → warmed-up ServingState
│   ├── schemas.py       pydantic request/response models + JSON↔numpy
│   ├── app.py           the FastAPI routes
│   └── middleware.py    --middleware attachment
│
└── cli/
    ├── main.py          typer commands: check, export, serve, version
    └── render.py        every byte of rich output. Commands never print.

tests/
├── models/              the hazard corpus — one export hazard per fixture
├── conftest.py          session-scoped fixtures (export is the slow part)
└── test_*.py            one module per source concern
```

~2,600 lines of source. Small enough that you can hold all of it in your head, and it's worth
doing that once.

### The dependency rule

Dependencies point **downward only**:

```
cli  →  serve  →  export  →  adapters  →  torch
                      ↘  loading  ↗
```

- `adapters/` imports from `export/shapes.py` (for `infer_dynamic_shapes` /
  `alternative_sizes`) and nothing else in `export/`. That's the one upward-looking edge and
  it's deliberate: shapes is leaf-level utility code.
- `export/` knows nothing about serving or HTTP.
- `serve/` knows nothing about typer or rich.
- `cli/` is the only place that prints, and `render.py` is the only place inside `cli/` that
  prints. A command function builds objects and hands them to `render`.

If you find yourself wanting to `import fastapi` inside `export/`, or wanting to `print()`
inside `serve/`, the layering has gone wrong.

---

## 3. `loading.py` — from a string to a model

The CLI takes one positional argument and it can be four different things. `load_model()`
(`loading.py:113`) disambiguates, in this order:

1. **`.onnx` suffix** → `LoadedModel(onnx_path=...)`, no torch model at all.
2. **An existing file with a `.pt/.pth/.bin/.ckpt` suffix** → a checkpoint.
3. **Matches the import-spec regex `pkg.module:attr`** → import it.
4. **Otherwise** → assume a Hugging Face repo id, and say something useful if the `[hf]`
   extra isn't installed.

Order matters: the suffix checks come first so a file named `my.module:thing` on disk can't
be mistaken for an import spec, and the HF fallback is last because it's the only branch that
hits the network.

Things worth knowing:

- **`LoadedModel` is a dumb bag** (`loading.py:29`). It carries `model`, `onnx_path`,
  `example_inputs`, and `adapter_hint`. The hint is only set by the HF path — when we
  downloaded it from the hub we already know the family, so the registry doesn't need to
  guess.
- **The sibling `make_inputs` convention** (`loading.py:75`). If you point at
  `tests.models.clean_mlp:make_model` and that module also defines `make_inputs`, we call it
  automatically. That's why every fixture in `tests/models/` has both. It makes the whole
  corpus usable from the CLI with no `--inputs` flag.
- **`torch.load(weights_only=True)` by default** (`loading.py:89`). A pickled full module
  won't load that way, and the error message says so and names `--unsafe-load`. We don't
  silently downgrade — loading a pickle is running arbitrary code and that should be the
  user's explicit choice. `cli/main.py:162` prints a warning whenever the flag is used.
- **`_instantiate`** accepts either an `nn.Module` instance or a zero-arg callable returning
  one, so `attr` can be a model object or a factory.

`import_object()` lives here too and is reused by `serve/middleware.py` — it's the generic
"turn `pkg.module:attr` into a Python object" helper.

---

## 4. `adapters/` — family knowledge

### What an adapter is for

`torch.export` wants a module whose `forward` takes a **flat, fixed-arity tuple of tensors**.
Real models don't look like that. A PyG model takes a `Data` object. An HF model takes
kwargs and returns a `ModelOutput`. Someone's research model takes a dataclass.

So an adapter does exactly two jobs:

1. **Guess example inputs** when the user gave none (`example_inputs`).
2. **Turn (model, inputs) into export-ready form** (`prepare` → `Prepared`).

That's the whole `Adapter` protocol (`adapters/base.py:36`). It's a `typing.Protocol`, not a
base class — third parties implement it structurally, no import of ours required beyond
`Prepared`.

### `Prepared` is the contract

```python
@dataclass
class Prepared:
    model: nn.Module          # export-ready; forward takes flat tensors
    inputs: tuple             # flat example inputs, one per input_names entry
    input_names: tuple[str, ...]
    dynamic_shapes: tuple     # per input: {axis: torch.export.Dim} or None
    vary_fn: VaryFn | None    # sample i -> inputs; None = default shared-axis-0 sampler
    family: str
```

Everything downstream reads `Prepared` and nothing else about the model family. The
`dynamic_dims` property (`base.py:27`) zips names against specs to produce the
`{"x": [0], "edge_index": [1]}` dict you see in the CLI output and the manifest.

`vary_fn` is the sneaky-important field. It's how an adapter says *"here's how to generate a
differently-shaped but still-valid input for this family."* For a plain batch model the
default (resize axis 0 of everything together) is right. For a graph, it very much isn't —
see below.

### The flatten shim, and why it uses `exec`

`_flatten.py:29` generates a subclass whose `forward` has one **named positional parameter
per field**:

```python
src = f"def forward(self, {params}):\n    return self._call(({params},))\n"
exec(src, namespace)
```

`exec` in a library is a smell, so here's the justification, which is also in the docstring:

- `def forward(self, *tensors)` would bind everything into one `VAR_POSITIONAL` parameter.
  `torch.export` then sees a **single tuple input**, which doesn't line up with the per-input
  `dynamic_shapes` tuple we need to pass. You get a spec/signature mismatch.
- The parameter names become the **ONNX graph's input names**. `x`, `edge_index` in the
  served API come from here. A `*args` signature gives you `args_0`, `args_1`.

The input to `exec` is field names from a dataclass or a `Data` object, filtered through
`isidentifier()` — no user-controlled text, hence the `# noqa: S102`.

### The registry, and the `generic`-goes-last rule

`registry.available()` (`registry.py:31`) builds an ordered dict:

1. Entry points in the `downshift.adapters` group (ours are declared in `pyproject.toml`;
   third parties declare their own).
2. The built-in specs, via `setdefault` — so an entry point wins if both are present.
3. **`generic` is popped and re-appended** (`registry.py:43`).

That last step is the important one. `GenericAdapter.matches()` returns `True`
unconditionally — it's the catch-all. If dict ordering ever put it before `pyg`, every graph
model would silently take the generic path, get a single shared axis-0 `Dim`, and produce an
export that works on the example graph and throws `INVALID_ARGUMENT` on the next one. So the
ordering isn't stylistic; it's load-bearing.

`_load_spec` swallows `ImportError` and returns `None` — that's how `pyg` and `hf` stay
optional. No torch_geometric installed means the adapter is skipped silently, not a crash.

> **Testing note:** in a dev checkout the package usually isn't pip-installed (it's on a
> `.pth` path), so entry points aren't registered and only the `_BUILTIN_SPECS` path runs.
> Registry tests monkeypatch `entry_points` rather than relying on installation.

### The three built-ins

**`generic.py`** — any `nn.Module`. Two behaviours:

- If the single argument is a **dataclass instance**, flatten it into its tensor fields via
  the shim (`generic.py:80`). `torch.export` rejects unregistered dataclasses outright, so
  without this the whole category fails. If any field isn't a tensor we bail out and return
  `None`, deliberately letting `torch.export` produce the real error rather than a worse one
  of ours.
- If no inputs were given, guess a shape from the **first `Linear`/`Conv*`/`Embedding`
  layer** (`generic.py:31`). This is a heuristic and only fires when `forward` takes exactly
  one positional parameter. It handles the torchvision-style case and honestly nothing else.

Input names come from `forward`'s signature (`generic.py:22`), falling back to `input_0`,
`input_1`, ….

**`pyg.py`** — the reason the project has a GNN story:

```python
n_dim = torch.export.Dim("num_nodes", min=1, max=1 << 16)
e_dim = torch.export.Dim("num_edges", min=1, max=1 << 16)
axis_by_field = {"x": {0: n_dim}, "edge_index": {1: e_dim}, "edge_attr": {0: e_dim}}
```

**Node count and edge count are independent dims.** Tying them to one `Dim` is the classic
GNN export bug: it works on the example graph (where by coincidence the trace satisfies the
constraint) and fails on the next graph with a different N/E ratio. `edge_attr` shares the
*edge* dim, not the node one.

`make_vary_fn` (`pyg.py:85`) is the second half of the same insight. It resamples N and E
independently, and it **redraws `edge_index` against the sample's own node count**, not
against the original tensor's value range. Resize a graph from 8 nodes to 3 with the generic
axis-0 sampler and you get edges pointing at node 7 — an out-of-bounds gather, and a
"failure" that's our sampler's fault rather than the export's. That's a false DEGRADED, which
is the worst kind of bug this tool can have.

Note `matches()` (`pyg.py:47`): a PyG `Data` input is conclusive; otherwise, *only if no
inputs were supplied at all*, we sniff for a `MessagePassing` module. If the user handed us
plain tensors we return `False` and let the generic adapter take it — they've told us the
signature they want.

**`hf.py`** — encoder-only transformers. Builds `input_ids`/`attention_mask` straight from
`model.config` rather than pulling in `optimum`. Two details:

- `_FirstOutputShim` unwraps `ModelOutput` to `out[0]` so export sees a plain tensor
  (`last_hidden_state` for base models, `logits` for heads).
- The seq `Dim` is capped at `max_position_embeddings` (`hf.py:59`). A looser bound trips
  export's own guards against position-embedding indexing — you get a confusing constraint
  violation instead of a clean trace.

---

## 5. `export/` — produce the graph and judge it

### The example-input ladder (`inputs.py`)

Four rungs, in order, and the file is 25 lines because that's all it should be:

1. User-supplied inputs always win.
2. Adapter-derived (`adapter.example_inputs`).
3. The generic signature guess (which is rung 2 for the generic adapter).
4. **Fail loudly**, with a message that names both the Python and CLI ways to fix it.

Rung 4 matters more than it looks. "Couldn't figure out your inputs" is the single most
common way this tool tells a user something, so the message spells out
`check(model, example_inputs=(tensor, ...))` and `--inputs module:function`.

### `shapes.py` — small file, three sharp edges

- **`infer_dynamic_shapes`** — the default: axis 0 of every tensor input, all sharing one
  `Dim`. That's the batch assumption, right for most models, wrong for graphs (hence the PyG
  override).
- **`alternative_sizes(base)`** returns `{1, 2, 3, base+1, base*2} - {base}`. The small sizes
  catch off-by-one and squeeze/broadcast bugs; `base*2` catches anything that baked in a
  buffer size. Excluding `base` guarantees every non-zero verification sample has a shape the
  exporter never traced.
- **`safe_capture_inputs`** (`shapes.py:51`) is the subtle one. `torch.export` will
  **specialise a size-1 dimension to a constant even when you mark it dynamic** — it treats
  the 1 as meaningful (broadcasting). So if your example input has batch size 1, you silently
  get a static graph. We double any size-1 dynamic axis *for the trace only*; verification
  still runs against the real inputs. Without this, `--dynamic` on a size-1 axis is a no-op
  and you'd never know.
- **`parse_dynamic_spec`** handles `"x:0,edge_index:1"` and `"x:0:1"`, and
  `apply_dynamic_override` gives every `(name, axis)` pair its **own independent `Dim`** —
  if you're overriding by hand, we assume you don't want anything tied together.

### `capture.py` — drive the exporter, report what worked

```python
_STRATEGIES = (("strict=False", False), ("strict=True", True))
```

We run `torch.export.export` ourselves rather than letting `torch.onnx.export` do it, for one
reason: **the winning strategy becomes a value we return** (`capture_strategy`), not a string
we scrape out of console output. That value ends up in the verdict, the manifest, and the
compatibility matrix.

`strict=False` is tried **first** because it's the more permissive tracer and succeeds more
often; on torch 2.14 it's what essentially every fixture lands on. `strict=True` is the
fallback for the rare case where the non-strict path chokes.

The `contextlib.redirect_stderr` block (`capture.py:47`) exists because torch prints **entire
FX graphs to stderr** when a data-dependent guard fails. On `data_dependent_branch` that's
hundreds of lines of noise in front of a one-line answer. We capture it into
`CaptureResult.stderr`, available at debug level, and keep the terminal clean. Same reasoning
behind the `logging.getLogger("torch.onnx._internal.exporter._registration")` silencing at
module scope — torch logs a warning per missing torchvision op on *every* export, and it's
never actionable.

Failure is split into two shapes: trace failure (no `ExportedProgram`, `capture_strategy` is
`None`) and translation failure (traced, but `torch.onnx.export` raised — `capture_strategy`
is set). Both are `success=False`, but the verdict reason can tell you which wall you hit.

### `verify.py` — the actual product

```python
for i in range(k):
    sample = vary_fn(i)
    torch_outs = model(*sample)
    ort_outs = session.run(None, feeds)
```

Sample 0 is always the baseline (the exact example inputs). Samples 1..k-1 have varied
dynamic dims. Points to internalise:

- **The failure condition is `and`, not `or`** (`verify.py:149`):

  ```python
  if sample_abs > atol and sample_rel > rtol:
  ```

  A sample fails only when it blows **both** the absolute and the relative tolerance. This is
  intentional: a tiny absolute error on a near-zero value has a huge relative error, and a
  large absolute error on a huge value has a tiny relative one. Neither is a real divergence.
  Requiring both keeps false DEGRADEDs down. If you ever "fix" this to `or`, half the corpus
  goes yellow.

- **`shape_generalization` is a separate signal** from pass/fail. It's
  `non_baseline_failures == 0` — did every sample with a *novel* shape pass? You can have
  `failures > 0` (so `DEGRADED`) while `shape_generalization` is `True`, if only the baseline
  sample failed. That combination means "the graph is wrong everywhere", as opposed to "the
  graph is right on the traced shape and wrong elsewhere", which is the frozen-shape
  signature.

- **`torch.random.fork_rng`** (`verify.py:129`) wraps the loop. We seed for reproducibility,
  and the fork means we don't leave the caller's global RNG state perturbed. `check()` called
  from inside someone's training script must not change their next `torch.randn`.

- **Tolerances come from the widest float dtype in the model's parameters**
  (`default_tolerances`, `verify.py:39`), with bf16 checked before fp16 because it's the
  coarser of the two. The table itself lives in `settings.py` so it's env-overridable.

- **`_to_session`** accepts a path, bytes, or an `ONNXProgram` — the same `verify()` serves
  both fresh exports (in-memory proto) and `intake()` (a file on disk).

- **`_resize_dim0`** for the default sampler regenerates float tensors with `randn`, but
  keeps integer tensors **inside their observed `[min, max]` range** — integers in this
  position are almost always indices, and random int64s would just crash the gather.

### `verdict.py` — where the decision gets made

`build_verdict` (`verdict.py:129`) is the function to read if you read only one:

1. Collect warnings: **tied weights** (`_tied_weight_warnings`, which walks
   `named_parameters(remove_duplicate=False)` and flags shared storage — the GPT-2 embedding
   pattern), and **training mode** (we call `.eval()` and say so; verifying a model with
   active dropout is meaningless).
2. `capture(...)` on `safe_capture_inputs(...)`.
3. Not successful → `FAILED`, reason = the exception's first line, and
   `unsupported_ops` scraped from the message with the `_ATEN_OP` regex. That regex is
   best-effort string mining, and it's fine that it is — it's a hint in a table, not a
   contract.
4. `verify_numerics=False` → `UNVERIFIED` via ORT. Note it can **never** be `CLEAN`. The
   `--no-verify` escape hatch gives you the graph, not our blessing.
5. Otherwise verify, and hand the report to `numerics_outcome`.

`numerics_outcome` (`verdict.py:84`) is factored out so that `prevalidated.intake` applies
**exactly the same rule** to a third-party `.onnx` as we do to our own export. Only the
reason-string prefixes differ. One place decides pass→CLEAN/ORT, fail→DEGRADED/torch.

Two enums that are easy to confuse, so here's the distinction:

- `BackendName` (`verdict.py:27`) — the concrete backends a verdict can recommend:
  `onnxruntime`, `torch`. Never `auto`.
- `BackendChoice` (`engine.py:17`) — what the *user asked for*, which includes `auto`.

`auto` is a selection sentinel, not a backend. Keeping them as separate types means you can't
accidentally try to instantiate an "auto" backend. Both are `str`-Enums so they serialize to
plain JSON and compare equal to strings (`meta.name == "torch"` in `render.py` works because
of this).

`ExportVerdict` carries two heavyweight fields marked `repr=False`: `onnx_program` (the whole
in-memory graph) and `prepared` (the model and its example inputs). They're there because the
serving layer needs both without re-doing the work — but it means **a verdict is not a cheap
object to hold onto**, and `to_dict()` deliberately drops them.

### `manifest.py` — provenance

Written next to every `.onnx`. The parts that earn their keep:

- **SHA-256 of the artifact and of the source checkpoint** (when the model came from a file).
  That's the pair that lets you answer "is this `.onnx` really built from that `.pt`?" months
  later.
- **`observed_dtype`** (`manifest.py:25`) reads the dtype off the graph's **initializers**,
  i.e. what the export actually produced — not what you asked for. `--fp16` that silently
  didn't take shows up here.
- The full verdict, numerics report included. A `DEGRADED` export is still written to disk
  precisely because the manifest records how far off it is.
- torch/onnx/onnxruntime versions and available EPs, because "it worked last month" is a
  version question.

### `prevalidated.py` — someone else's `.onnx`

`intake()` handles the Olive/notebook/black-box case. No reference model → `UNVERIFIED`,
served as-is, and we say plainly that we never checked it. With `--reference`, it's the
normal `prepare_model` + `verify` path and the same `numerics_outcome`.

One detail in `_graph_summary` (`prevalidated.py:17`): input names are filtered against the
initializer set. Older ONNX graphs list initializers (weights) as graph inputs too, and
without the filter you'd tell the user their model needs a `layer1.weight` input.

---

## 6. `serve/` — from verdict to running server

### Backends behind one Protocol

`backends.py` defines a two-method contract: `infer(dict[str, ndarray]) -> dict[str, ndarray]`
and `metadata() -> BackendMeta`. Both implementations satisfy it, so the HTTP layer genuinely
does not know which one is behind it.

- **`OnnxRuntimeBackend`** — session with `ORT_ENABLE_ALL` graph optimization (that's ORT's
  own fusion pass, not quantization; we do no optimization of our own). Thread counts of `0`
  mean "ORT chooses", which is also ORT's default, so passing them unconditionally is safe.
  Provider selection falls back to CPU whenever CUDA was asked for but isn't available.
- **`TorchBackend`** — eager. The docstring calls it "the fallback path, and a first-class
  one" and that's the intent: it's not a degraded mode of the server, it's an equal
  implementation. If given `example_inputs` it runs **one pass at construction** to fill in
  output dtypes and shapes for `/metadata`, since unlike ORT a torch module can't introspect
  its own signature.

**Outputs are keyed positionally** — `output_0`, `output_1` (`backends.py:99`). ORT knows the
graph's real output names and torch doesn't, so using them would make the response shape
depend on the backend, which is exactly the thing this layer exists to hide. The real ONNX
names are kept in `onnx_output_names` for the `session.run` call.

### `engine.py` — the orchestrator

`prepare_serving()` is four steps: get a verdict, choose a backend, build it, warm it up.

`_verdict_for` (`engine.py:58`) branches three ways: a `.onnx` goes to `intake`; `--backend
torch` **skips the export entirely** and fabricates an `UNVERIFIED` verdict (you asked for
eager, so burning 30 seconds on an export you won't use would be rude); otherwise the full
`build_verdict`.

`choose_backend` (`engine.py:88`) returns `(name, notes)` — the notes are strings destined for
the boot banner, which is how the engine tells the user "I overrode you" without importing
rich. The rules:

- `auto` → the verdict's recommendation.
- `--force-onnx` only does anything when the status is `DEGRADED` (see `ServingState.forced_onnx`,
  `engine.py:54`) and attaches a red note. Forcing ONNX on a `FAILED` model is meaningless —
  there's no graph.
- ORT requested but no graph exists → fall back to torch with a note.
- Torch requested but no PyTorch model (i.e. the user handed us a bare `.onnx`) → that's a
  hard `ValueError`, because there's nothing to fall back *to*.

`warmup()` runs N inferences and then flips `state.ready`, which is what `/ready` reports.
First-call costs (ORT arena allocation, lazy kernel init) shouldn't land on a real user's
request, and in a rolling deploy the load balancer should hold traffic until they're paid.

### `schemas.py` — JSON in, tensors out

The `to_numpy` rules, in precedence order:

1. An explicit `dtype` in a `{"data": ..., "dtype": ..., "shape": ...}` payload.
2. Whatever the backend **declared** for that input (ORT's `tensor(float)` names get
   normalized to numpy names by `normalize_dtype`).
3. Nothing declared → **integer lists become `int64`, everything else `float32`**. That's
   what torch models expect, and the torch backend declares no dtypes, so this rung is the
   one that actually fires for eager serving.

Ragged or non-numeric arrays (which numpy silently turns into `dtype=object`) are caught and
rejected with the input's name in the message.

### `app.py` — the routes

Five routes, all thin. `run_predict` is the shared body: check for missing inputs, convert,
infer, and turn **any** exception from the backend into a `400`. That last choice is
deliberate — a shape or dtype mismatch coming out of ORT or torch is the client's mistake,
not a server fault, and returning 500 for it makes the endpoint look flaky when it's working
correctly.

`/predict/graph` is sugar over `/predict`: it checks `x` and `edge_index` are actually among
the model's inputs (with a message that lists the real ones if not), forces `edge_index` to
`int64`, and delegates. One graph per request — no batching. Concatenate client-side with
offset edge indices if you need more.

`/ready` returns 200/503 via an explicit `JSONResponse` because the status code carries the
signal for k8s-style probes.

---

## 7. `cli/` — thin commands, all output in one place

`main.py` has four commands and each follows the same three-beat shape: **load → call the
library → hand the result to `render`**. No command computes anything a library function
could compute.

### Exit codes are the API

```python
EXIT_CODES = {"CLEAN": 0, "FAILED": 1, "DEGRADED": 2, "UNVERIFIED": 3}  # verdict.py:34
EXIT_USAGE = 4   # bad spec, bad option, unloadable file
EXIT_CRASH = 5   # anything we didn't anticipate
```

This is what makes `downshift check` usable as a CI gate. The `_exit_on_error` context
manager (`main.py:145`) enforces the split: `typer.Exit` passes through untouched,
`ValueError` (which `LoadError` subclasses, along with bad `--dynamic` and bad backend
combinations) becomes 4, everything else becomes 5 with a traceback at debug level.

### Logging keeps stdout clean

`_setup_logging` sends everything to **stderr** (`main.py:133`) so `--json` output on stdout
is pipeable. `render.warn`/`render.error` use a separate stderr `Console` for the same
reason. Below debug level we silence `onnxscript` and `onnx_ir`, which log every graph
rewrite at INFO.

### The multi-worker trick

This is the one genuinely non-obvious bit of the CLI. `uvicorn.run(app_object, workers=N)`
doesn't work — multi-worker mode needs an **import string**, because each worker is a fresh
process that has to construct its own app.

So: `ServeArgs` (`main.py:281`) is a dataclass of **plain JSON-able types only** — everything
needed to rebuild a `ServingState` from scratch. The parent serializes it into the
`_DOWNSHIFT_SERVE_ARGS` env var, then points uvicorn at
`downshift.cli.main:_serve_app_factory` (`main.py:331`), which each worker calls on its own,
independently reloading, re-exporting, and re-warming the model.

That means **memory and startup time scale linearly with `--workers`**, and the CLI warns
about it explicitly. The parent also builds one throwaway state with `warmup=0` first, purely
so the banner can be printed and a bad model fails fast before N processes spawn.

If you ever add a serve option, it has to be a JSON-able field on `ServeArgs` or it silently
won't reach the workers.

### `render.py`

Every table, banner, warning, and error. Commands pass objects; this module decides how they
look. The reason for the hard split is testability: `tests/test_cli.py` asserts **exit codes
and JSON**, never banner text, so the visual layer can be reworked without touching a single
CLI test.

`_sym(utf, ascii)` (`render.py:27`) falls back to ASCII when the console can't encode
unicode — Windows `cp1252` pipes will otherwise raise `UnicodeEncodeError` mid-render and
take down the command. Every glyph in the banner goes through it.

---

## 8. The verdict, end to end

Three traces. Follow one in the source with the file open.

### `downshift check pkg.mod:make_model`

```
cli/main.py:check_cmd
  └─ loading.load_model            → LoadedModel(model=..., example_inputs=...)
  └─ downshift.check               (__init__ → export/verdict.py)
       ├─ prepare_model
       │    ├─ registry.detect     → pyg? hf? generic (last)
       │    ├─ inputs.synthesize   → user > adapter > guess > raise
       │    └─ adapter.prepare     → Prepared(shim, flat tensors, dims, vary_fn)
       └─ build_verdict
            ├─ warnings           tied weights, training mode
            ├─ capture            safe_capture_inputs → torch.export → torch.onnx.export
            └─ verify             k samples, varied shapes, torch vs ORT
                 └─ numerics_outcome → (status, backend, reason)
  └─ render.print_verdict / json
  └─ raise typer.Exit(verdict.exit_code)
```

Nothing touches the disk.

### `downshift export ... -o artifacts/`

Same path, then `__init__.export` saves the program, sets `verdict.onnx_path`, and calls
`write_manifest`. `FAILED` → `onnx_program is None` → nothing written, and `render.print_artifacts`
says so.

### `downshift serve ...` and one request

```
serve_cmd → _build_serving_app
  ├─ load_model (+ reference)
  ├─ prepare_serving
  │    ├─ _verdict_for      intake / skip-export / build_verdict
  │    ├─ choose_backend    (name, notes)
  │    ├─ _build_backend    ORT session or TorchBackend
  │    └─ warmup            N inferences, then ready = True
  └─ build_app             FastAPI routes over ServingState

POST /predict
  └─ run_predict: missing-input check → to_numpy (declared dtypes) → backend.infer
                  → {outputs, shapes, dtypes}
```

---

## 9. `settings.py` — configuration

One rule, stated at the top of the file: **flag > env > default**.

The CLI achieves this by using `settings.X` as the *default value* of the typer option
(`k: SamplesOpt = settings.SAMPLES`), so passing the flag overrides it and not passing it
falls through to the env-derived default.

The consequence to remember: **values are read once at import time**. Setting a `DOWNSHIFT_*`
variable after the module is imported does nothing — you restart the process. That's why
`tests/test_settings.py` reloads the module around each case instead of importing the
constants at collection time.

What's configurable: `HOST`, `PORT`, `DEVICE`, `BACKEND`, `WARMUP`, `SAMPLES`, the two ORT
thread counts, `WORKERS`, and the **per-dtype tolerance table** (`DOWNSHIFT_TOL_FLOAT16_ATOL`
and friends). That last one exists so a genuinely noisy fp16 model can be given room without
anyone editing `verify.py` or passing tolerances through five call layers.

Bad values raise at import with a message naming the variable — failing at startup beats
silently falling back to a default you didn't want.

---

## 10. `tests/` — the corpus is the point

### `tests/models/` is a hazard corpus, not a grab bag

Each fixture isolates **exactly one export hazard**, and each has a `make_model` and a
`make_inputs` so it's usable from the CLI and by `conftest`. The interesting ones:

- `scatter_include_self_false` — exports with no errors, returns wrong numbers on 7/8
  samples. This is the fixture that justifies the entire product.
- `custom_autograd` — was *expected* to fail. On torch 2.14 it's CLEAN, because
  `torch.export` traces straight through a `Function.forward` built from ordinary ops. We
  left it in and updated the expectation.
- `data_dependent_branch` — a Python `if` on a tensor value. FAILED under both strict modes,
  and the fixture that produces the stderr firehose `capture.py` suppresses.
- `dict_input` — the dataclass-container case, i.e. PyG's flattening problem without the PyG
  dependency.
- `tied_weights` — GPT-2-style shared storage. CLEAN, but surfaces the verdict warning.
- `dropout_model` — meaningless to verify unless `eval()` happened, so it's the test that
  `build_verdict` switches modes.

The expected statuses in `test_export.py` are **what torch actually does today**, not what we
wished it did. When torch changes, that file changes, and the compatibility matrix regenerates
to match. That's the honest way to run this kind of corpus.

### Structure

- `conftest.py` — session-scoped fixtures. Export is the slow part (every fixture traces
  through `torch.export`), so anything more than one module needs is built exactly once.
- One test module per source concern; the `test_*_internals.py` ones exist to cover small
  helpers that are otherwise only hit indirectly.
- **`test_cli.py` asserts exit codes and JSON, never banner text.** `test_render.py` executes
  every rendering branch without asserting its content. Presentation stays free to change.
- Coverage gate is **95%**, enforced in `pyproject.toml`'s `addopts`, so a plain `pytest` run
  fails on a coverage drop.

CI (`.github/workflows/ci.yml`) runs `ruff check .`, `mypy src`, and `pytest` on Python 3.11
and 3.12 with CPU torch. A second workflow regenerates the compatibility matrix weekly
against current torch/ORT and opens a PR when it changes — that's how we find out that torch
changed its behaviour on a hazard before a user does.

---

## 11. Invariants — the things not to break

If you change one of these, you're changing the product, not refactoring it.

1. **`generic` is always tried last.** `registry.py:43`. Break it and graph models silently
   get batch-shaped dims.
2. **Numerics failure is `abs AND rel`, not `or`.** `verify.py:149`.
3. **`UNVERIFIED` can never become `CLEAN`.** `--no-verify` hands you the graph, not a
   verdict.
4. **Sample 0 is the baseline; samples ≥1 must have shapes the export never saw.** That's
   what `alternative_sizes` excluding `base` guarantees, and it's the whole basis of
   `shape_generalization`.
5. **A `vary_fn` must generate *semantically valid* inputs, not just correctly-shaped ones.**
   The PyG edge-index redraw is the worked example. Violating this manufactures false
   DEGRADEDs.
6. **Size-1 dynamic axes get doubled for the trace.** `shapes.py:51`. Otherwise torch
   specialises them to constants and the dynamic marking is a lie.
7. **Exit codes are a public contract.** People gate CI on them.
8. **`serve/` must not print and `cli/` must not compute.** Output lives in `render.py`.
9. **Anything a worker needs must be a JSON-able field on `ServeArgs`.**
10. **Both backends return positionally-keyed outputs.** The response must not reveal which
    backend answered.
11. **No quantization, ever.** Not deferred — cut. Hand an optimized graph back in via
    `serve model.onnx --reference model.pt` and it gets verified like anything else.

---

## 12. Where to add things

**A new model family** → a new adapter. Implement `matches`/`example_inputs`/`prepare`,
expose `ADAPTER`, register under the `downshift.adapters` entry-point group. Think hardest
about `dynamic_shapes` (what varies independently?) and `vary_fn` (what makes a *valid*
sample of this family?). Add a fixture to `tests/models/` isolating whatever hazard the family
brings.

**A new verdict input** (a new signal that should influence the status) → a field on
`ExportVerdict`, set in `build_verdict`, rendered in `render.py`, and included in `to_dict`
so it reaches the manifest and `/metadata` for free.

**A new serving option** → a field on `ServeOptions`, a field on `ServeArgs` (JSON-able!), a
typer option defaulting to a `settings` constant, and a line in `_build_serving_app`.

**A new route** → `app.py`, with the request/response models in `schemas.py`. Keep the body
thin enough that it delegates to something testable without a TestClient.

**A new backend** → satisfy the `Backend` protocol in `backends.py`, add it to `BackendName`
and `BackendChoice`, and teach `choose_backend` when to pick it. Make sure the output keys
stay positional.

---

## 13. Known rough edges

Stated plainly so you don't rediscover them as surprises:

- **`_ATEN_OP` regex-mines exception text** for unsupported ops (`verdict.py:36`). It's a
  hint, not a contract, and torch changing its message format degrades it silently.
- **`generic.example_inputs` is a crude heuristic** — first `Linear`/`Conv`, single-argument
  forward only. It's right for torchvision-shaped models and gives up otherwise, which is the
  intended behaviour but not a general solution.
- **`ExportVerdict` holds the ONNX program and the prepared model in memory.** Fine for a CLI
  invocation and for serving; not something to accumulate in a list.
- **CUDA is implemented but untested in this release.** `--device cuda` selects the EP and
  moves the torch module; nobody has run it on a GPU in CI.
- **`--workers N` does N full independent loads.** No shared memory, no copy-on-write model
  weights.
- **No dynamic request batching, no graph batching.** One request, one inference, one graph.
