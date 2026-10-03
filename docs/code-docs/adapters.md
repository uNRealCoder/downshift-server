# Adapters

How downshift turns an arbitrary `nn.Module` into something `torch.export` can trace, how
adapters are found, and how to write your own.

## What an adapter does

An adapter knows one model family well enough to do two things: build example inputs when
the caller gave none, and turn the model plus inputs into a flat, fixed-arity tensor
signature that `torch.export` can trace, along with everything the verification step needs
to generate more samples. `downshift.core.verdict.prepare_model` is the only caller: it
resolves an adapter, synthesizes inputs if needed (`downshift.core.inputs.synthesize`),
and calls `adapter.prepare(...)`.

## The `Adapter` protocol

```python
from downshift.adapters.base import Adapter, Prepared


@runtime_checkable
class Adapter(Protocol):
    name: str
    family: str

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool: ...
    def example_inputs(self, model: nn.Module) -> tuple | None: ...
    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared: ...
```

It's a `typing.Protocol` marked `@runtime_checkable`, so `isinstance(obj, Adapter)` works
structurally - a custom adapter doesn't need to inherit from anything, it needs these five
attributes with matching signatures.

- **`name: str`** - the identifier used to select it: `--adapter name`, `adapter="name"`
  on `check()`/`export()`/`intake()`/`prepare_model()`, and what `registry.get()` looks up.
- **`family: str`** - usually the same string as `name`, but conceptually distinct: it's
  what ends up in `ExportVerdict.model_family` and `/metadata`'s `family` field, i.e. the
  thing a human reads, not the thing a user types on `--adapter`.
- **`matches(model, example_inputs) -> bool`** - called by `registry.detect()`, in
  registration order (see below), when no `--adapter`/`adapter=` was given. The first
  adapter whose `matches()` returns `True` wins; write it to check something structural
  (the built-in `pyg` adapter checks for a `MessagePassing` layer) rather than something
  incidental, since a false positive here silently steals a model from a better-fitting
  adapter.
- **`example_inputs(model) -> tuple | None`** - called only when the caller supplied no
  `example_inputs` at all. Return a tuple of forward-arguments you can construct from the
  model alone (shapes read off a first `Linear`/`Conv` layer, a Hugging Face config's
  `hidden_size`, ...), or `None` if you can't guess - `synthesize()` then raises
  `ValueError` naming exactly how to pass inputs explicitly.
- **`prepare(model, example_inputs) -> Prepared`** - the adapter's core job: return
  export-ready everything. `example_inputs` here is always a tuple (already resolved, by
  the caller or by `example_inputs()` above), never `None`.

## `Prepared`

```python
@dataclass
class Prepared:
    model: nn.Module  # export-ready module; forward takes flat tensors
    inputs: tuple  # flat example inputs, one per input_names entry
    input_names: tuple[str, ...]
    dynamic_shapes: tuple  # per input: {axis: torch.export.Dim} or None
    vary_fn: VaryFn | None  # sample i -> inputs; None means the shared-axis-0 default
    family: str
```

- `model` doesn't have to be the same object the caller passed in - wrap it if the real
  model's `forward` takes a dataclass, a dict, or keyword-only arguments that
  `torch.export` can't trace directly. Whatever `model` is, its `forward` must take
  exactly `len(input_names)` positional tensor arguments.
- `inputs` and `input_names` are parallel tuples: `inputs[i]` is the example value for
  `input_names[i]`.
- `dynamic_shapes` is parallel to both: one entry per input, either `None` (fully static)
  or a `{axis_index: torch.export.Dim(...)}` dict. `downshift.core.shapes` has the helpers
  built-in adapters use to build these (`apply_dynamic_override` for the `--dynamic` CLI
  override, `dim_bounds`/`alternative_sizes` for sampling within them).
- `vary_fn`, when not `None`, is called as `vary_fn(i)` for `i` in `range(k)` during
  verification and must return a tuple shaped like `inputs`; `vary_fn(0)` must return
  exactly the baseline `inputs` (sample 0 is never varied - it's the "does this even work
  at the traced shape" check). Leave it `None` to get `make_shared_axis0_vary_fn`'s
  default: every dynamic input's axis 0 moves together, staying close to the example
  (tiled rows plus small noise for floats, values sampled within the observed range for
  integers) rather than generating unrelated noise.
- `dynamic_dims` is a derived property, not a field: `{input_name: sorted(axes)}` for
  every input whose `dynamic_shapes` entry is non-empty - this is what ends up in
  `ExportVerdict.dynamic_dims`.

## Resolution: `detect()` / `available()` / `get()`

All in `downshift.adapters.registry`.

- **`get(name)`** loads exactly one adapter by name, without importing any other family's
  optional dependency. Three forms, tried in order: a `path/to/adapter.py[:attr]` spec
  (split on the literal `.py:` so a Windows drive letter's colon is never mistaken for the
  separator) loads straight from a file via `load_from_file`; a built-in name (`generic`,
  `pyg`, `hf`) imports that one module directly; anything else is looked up among
  registered entry points. Raises `LoadError` (a `ValueError` subclass, defined in
  `downshift._imports` and re-exported by `downshift.loading`) if nothing matches.
- **`available()`** returns every adapter that can currently be loaded: entry points under
  the `downshift.adapters` group (skipping any whose `ImportError` means an optional
  dependency isn't installed), plus the built-ins whose gating module (`transformers` for
  `hf`, `torch_geometric` for `pyg`; `generic` has none) is already in `sys.modules` - so
  discovery for a plain PyTorch model never imports either. `generic` is always moved to
  the end regardless of registration order. Results are cached per process, keyed by which
  optional families were in play at the time (a plugin importing its own optional
  dependency later can only add entries to that cache, never remove one).
- **`detect(model, example_inputs)`** calls `available()` and returns the first adapter
  whose `matches()` is `True`. Raises `RuntimeError` if none match (only possible if even
  `generic` is unavailable, which shouldn't happen in a normal install).

Order matters: `available()` tries entry points, then the built-ins in the fixed order
`hf`, `pyg`, `generic`, with `generic` always pushed last no matter where it came from.
Write `matches()` narrowly enough that your adapter doesn't shadow one that should have
run first.

## Registering by entry point

```toml
[project.entry-points."downshift.adapters"]
myfamily = "my_pkg.adapter:MyAdapter"
```

The entry point names the adapter class, and the registry instantiates it with no
arguments - the same way it builds the built-ins, none of which keeps a module-level
instance. An entry point that names a ready-made instance instead is used
as-is, so plugins written against the older `ADAPTER = MyAdapter()` convention still load.

**The module stays cheap to import.** `available()` imports every registered entry
point's module just to build the adapter list, on every process that calls `check`,
`export`, or `serve` - even for a model the adapter will never end up handling. An
`ImportError` while loading an entry point is treated as "this adapter's optional
dependency isn't installed" and silently skipped, which means a *different* kind of
import failure (a typo, a genuinely broken dependency chain) is silently skipped too. Put
your model library's own import, and anything else heavy, inside `prepare()` - which only
runs once `matches()` has already said yes - not at the module scope that defines
`MyAdapter`. This is exactly why the built-in `hf` and `pyg` adapters are
*not* entry points: `HFAdapter`/`PyGAdapter` need `transformers`/`torch_geometric` already
imported just to define the class, so they can't follow their own rule. `registry.py`
loads them directly instead, gated on `"transformers" in sys.modules` /
`"torch_geometric" in sys.modules`, so a plain-PyTorch `check` never imports either.

## One-off adapters: a bare `.py` file

No install, no entry point, no `pyproject.toml` change:

```bash
downshift check my_model:model --adapter path/to/pointcloud_adapter.py
```

Point at the class with `--adapter path/to/pointcloud_adapter.py:MyAdapter` and
`registry.load_from_file` instantiates it with no arguments. A bare `.py` path looks up a
module-level `ADAPTER` instead, which may name the class (`ADAPTER = MyAdapter`) or an
instance. Either way, the loaded object is checked with
`isinstance(obj, Adapter)` and `load_from_file` raises `LoadError` with a specific message
if it doesn't implement all five attributes.

## Seeding a custom sampler

`--seed` only reaches a custom `vary_fn` if that function draws its randomness from
torch's own global RNG (`torch.rand`, `torch.randint`, `torch.randn`, ...), because
`verify()` seeds and calls every sample inside a single `torch.random.fork_rng()` block:

```python
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed)
    for i in range(k):
        sample = vary_fn(i)
        ...
```

`fork_rng` means this never disturbs the caller's own global RNG state outside the
`with` block, but it also means an adapter that keeps its own `random.Random()` instance,
or calls `numpy.random` directly, is invisible to `--seed` - its samples will differ
between runs even at the same seed. The built-in `hf` and `pyg` adapters' `vary_fn`s both
draw from `torch`'s global functions for exactly this reason; write yours the same way if
you want `-k`/`--samples` runs to be reproducible.

## A complete minimal adapter

A toy "point cloud" family: a model whose `forward` takes two co-indexed tensors, point
coordinates and per-point features - a case the built-in `generic` adapter gets wrong,
since it would mark axis 0 of each input dynamic independently instead of tying them
together.

```python
import torch
from torch import nn

from downshift.adapters.base import Prepared


class PointCloudNet(nn.Module):
    def __init__(self, coord_dim: int = 3, feature_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.encode = nn.Linear(coord_dim + feature_dim, out_dim)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.encode(torch.cat([points, features], dim=-1)).relu()


class PointCloudAdapter:
    name = "pointcloud"
    family = "pointcloud"

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool:
        return isinstance(model, PointCloudNet)

    def example_inputs(self, model: nn.Module) -> tuple | None:
        in_features = model.encode.in_features
        coord_dim = 3
        n = 50
        return torch.randn(n, coord_dim), torch.randn(n, in_features - coord_dim)

    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared:
        point_dim = torch.export.Dim("num_points", min=1, max=1 << 16)
        return Prepared(
            model=model,
            inputs=example_inputs,
            input_names=("points", "features"),
            dynamic_shapes=({0: point_dim}, {0: point_dim}),
            vary_fn=None,  # the default axis-0 sampler already ties both inputs together
            family=self.family,
        )
```

Both inputs' axis 0 share one `torch.export.Dim`, so `torch.export` knows `points` and
`features` must always have the same point count - `vary_fn=None` is enough here because
`make_shared_axis0_vary_fn` already varies every dynamic input's axis 0 together by
default; a custom `vary_fn` is only needed when inputs vary in a way that isn't "all
dynamic axes move to the same size."

Use it without registering anything:

```python
import downshift

model = PointCloudNet().eval()
# example_inputs() supplies the inputs
verdict = downshift.check(model, adapter=PointCloudAdapter())
```

or from the CLI, unregistered:

```bash
downshift check my_model:model --adapter path/to/pointcloud_adapter.py:PointCloudAdapter
```

or registered for auto-detection and `--adapter pointcloud`, via
`[project.entry-points."downshift.adapters"]` as shown above.
