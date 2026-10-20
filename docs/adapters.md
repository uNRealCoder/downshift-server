# Adapters

This page describes how downshift turns an `nn.Module` into a module that `torch.export` can trace, how downshift finds adapters, and how to write your own adapter.

## What an adapter does

An adapter knows one model family. It does two jobs:

1. It builds example inputs when the caller gave none.
2. It turns the model and the inputs into a flat tensor signature with a fixed number of arguments. `torch.export` can trace this signature. The adapter also gives the verification step what it needs to generate more samples.

`downshift.core.verdict.prepare_model` is the only caller. It does these steps:

1. It resolves an adapter.
2. If necessary, it synthesizes inputs (`downshift.core.inputs.synthesize`).
3. It calls `adapter.prepare(...)`.

## The `Adapter` protocol

```python
from downshift.adapters.base import Adapter, Prepared


@runtime_checkable
class Adapter(Protocol):
    name: str

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool: ...
    def example_inputs(self, model: nn.Module) -> tuple | None: ...
    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared: ...
```

The protocol is a `typing.Protocol` with the `@runtime_checkable` mark. `isinstance(obj, Adapter)` therefore checks the structure. A custom adapter does not inherit from anything. It needs these four attributes with matching signatures:

- **`name: str`** is the identifier that selects the adapter. It is used in `--adapter name`, in `adapter="name"` on `check()`, `export()`, `intake()` and `prepare_model()`, and in `registry.get()`. It is also the value that `ExportVerdict.model_family` and the `family` field of `/metadata` report. Set `Prepared.family` to this value.
- **`matches(model, example_inputs) -> bool`** is called by `registry.detect()`, in the order of registration (see below), when the caller gave no `--adapter` or `adapter=`. The first adapter that returns `True` wins. Check something structural. For example, the built-in `pyg` adapter checks for a `MessagePassing` layer. Do not check something incidental. A false positive takes a model away from an adapter that fits better, and nobody sees this.
- **`example_inputs(model) -> tuple | None`** is called only when the caller gave no `example_inputs`. Return a tuple of forward arguments that you can build from the model alone. For example, read the shapes from the first `Linear` or `Conv` layer, or use `hidden_size` from a Hugging Face config. If you cannot guess, return `None`. `synthesize()` then raises a `ValueError` that tells the caller how to pass inputs explicitly.
- **`prepare(model, example_inputs) -> Prepared`** is the main job of the adapter. It returns everything that is ready for export. `example_inputs` is always a tuple. The caller or `example_inputs()` already resolved it. It is never `None`.

## `Prepared`

```python
@dataclass
class Prepared:
    model: nn.Module  # export-ready module; forward takes flat tensors
    inputs: tuple  # flat example inputs, one per input_names entry
    input_names: tuple[str, ...]
    dynamic_shapes: tuple  # per input: {axis: torch.export.Dim} or None
    vary_fn: VaryFn | None  # sample i -> inputs; None means the shared-axis-0 default
    family: str  # the adapter's name
```

- `model` can be a different object from the model that the caller passed in. Wrap the model if its `forward` takes a dataclass, a dict, or keyword-only arguments that `torch.export` cannot trace. The `forward` of `model` must take exactly `len(input_names)` positional tensor arguments.
- `inputs` and `input_names` are parallel tuples. `inputs[i]` is the example value for `input_names[i]`.
- `dynamic_shapes` is also parallel to both. It has one entry for each input. The entry is `None` (fully static) or a `{axis_index: torch.export.Dim(...)}` dict. `downshift.core.shapes` has the helpers that the built-in adapters use to build these entries:
  - `apply_dynamic_override` applies the `--dynamic` CLI override.
  - `dim_bounds` and `alternative_sizes` sample inside the bounds.
- `vary_fn`, if it is not `None`, is called as `vary_fn(i)` for `i` in `range(k)` during verification. It must return a tuple with the same shape as `inputs`. `vary_fn(0)` must return exactly the baseline `inputs`. Sample 0 is never varied. It checks that the model works at the traced shape.
- If `vary_fn` is `None`, downshift uses the default of `make_shared_axis0_vary_fn`. Axis 0 of every dynamic input moves together. The samples stay close to the example. For floats, the sampler tiles rows and adds small noise. For integers, it takes values from the observed range. It does not generate unrelated noise.
- `dynamic_dims` is a derived property and not a field. It is `{input_name: sorted(axes)}` for each input that has a non-empty `dynamic_shapes` entry. This is the value in `ExportVerdict.dynamic_dims`.

## Resolution: `detect()`, `available()` and `get()`

All three are in `downshift.adapters.registry`.

- **`get(name)`** loads exactly one adapter by name. It does not import the optional dependency of another family. It tries three forms in this order:
  1. A `path/to/adapter.py[:attr]` spec. Downshift splits on the literal `.py:`. This way, the colon of a Windows drive letter is never the separator. `load_from_file` loads the adapter directly from the file.
  2. A built-in name (`generic`, `pyg`, `hf`). Downshift imports that one module directly.
  3. Any other name. Downshift looks for it among the registered entry points.

  If nothing matches, `get` raises `LoadError`. `LoadError` is a subclass of `ValueError`. It is defined in `downshift._imports` and `downshift.loading` re-exports it.
- **`available()`** returns every adapter that downshift can load now. It includes:
  - The entry points in the `downshift.adapters` group. It skips an entry point if an `ImportError` shows that an optional dependency is not installed.
  - The built-in adapters whose gating module is already in `sys.modules`. The gating module is `transformers` for `hf` and `torch_geometric` for `pyg`. `generic` has none.

  Because of this rule, the search for adapters for a plain PyTorch model never imports either module. `generic` always goes to the end, in all cases. Downshift caches the results for each process. The key of the cache is the set of optional families in use at that time. If a plugin imports its own optional dependency later, the cache can only gain entries. It never loses one.
- **`detect(model, example_inputs)`** calls `available()` and returns the first adapter whose `matches()` is `True`. If none match, it raises `RuntimeError`. This can happen only if `generic` is also not available. A normal installation does not have this fault.

The order is important. `available()` tries the entry points first. It then tries the built-in adapters in the fixed order `hf`, `pyg`, `generic`. `generic` is always last, wherever it came from. Make `matches()` narrow, so that your adapter does not take the place of an adapter that must run first.

## Registering by entry point

```toml
[project.entry-points."downshift.adapters"]
myfamily = "my_pkg.adapter:MyAdapter"
```

The entry point names the adapter class. The registry creates an instance with no arguments. It builds the built-in adapters in the same way, and none of them keeps a module-level instance. If an entry point names a ready-made instance, downshift uses it as it is. Plugins that use the older `ADAPTER = MyAdapter()` convention continue to load.

**Keep the import of the module cheap.** `available()` imports the module of each registered entry point to build the adapter list. It does this in each process that calls `check`, `export` or `serve`. It does this also for a model that the adapter will never handle.

An `ImportError` while loading an entry point means "the optional dependency of this adapter is not installed". Downshift skips the adapter without a message. A different kind of import failure is also skipped without a message. Examples are a typo and a broken dependency chain.

Put the import of your model library, and all other heavy imports, inside `prepare()`. `prepare()` runs only after `matches()` has returned `True`. Do not put them in the module scope that defines `MyAdapter`.

For this reason, the built-in `hf` and `pyg` adapters are not entry points. `HFAdapter` and `PyGAdapter` need `transformers` and `torch_geometric` to be imported before they can define the class. They cannot follow this rule. `registry.py` loads them directly. It loads them only if `"transformers" in sys.modules` or `"torch_geometric" in sys.modules`. A `check` on a plain PyTorch model never imports either library.

## One-off adapters: a bare `.py` file

You do not need an installation, an entry point or a change to `pyproject.toml`:

```bash
downshift check my_model:model --adapter path/to/pointcloud_adapter.py
```

To name the class, use `--adapter path/to/pointcloud_adapter.py:MyAdapter`. `registry.load_from_file` creates an instance with no arguments. For a bare `.py` path, downshift looks for a module-level `ADAPTER`. It can name the class (`ADAPTER = MyAdapter`) or be an instance.

In both cases, downshift checks the loaded object with `isinstance(obj, Adapter)`. If the object does not implement all four attributes, `load_from_file` raises `LoadError` with a specific message.

## Seeding a custom sampler

`--seed` reaches a custom `vary_fn` only if the function takes its random numbers from the global RNG of torch (`torch.rand`, `torch.randint`, `torch.randn`, and others). `verify()` seeds the RNG and calls every sample inside one `torch.random.fork_rng()` block:

```python
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed)
    for i in range(k):
        sample = vary_fn(i)
        ...
```

`fork_rng` makes sure that the global RNG state of the caller does not change outside the `with` block. The result is that `--seed` does not see some sources of random numbers. Examples are an adapter that keeps its own `random.Random()` instance and an adapter that calls `numpy.random` directly. The samples of such an adapter are different in each run, also with the same seed.

The `vary_fn` functions of the built-in `hf` and `pyg` adapters take their numbers from the global functions of torch for this reason. Write your function in the same way if you want `-k` and `--samples` runs to be reproducible.

## A complete minimal adapter

This example is a toy "point cloud" family. The `forward` of the model takes two tensors with the same first axis: point coordinates and features for each point. The built-in `generic` adapter handles this case in a wrong way. It marks axis 0 of each input as dynamic separately. It does not tie the axes together.

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
            family=self.name,
        )
```

Axis 0 of both inputs shares one `torch.export.Dim`. `torch.export` therefore knows that `points` and `features` always have the same number of points. `vary_fn=None` is enough here. `make_shared_axis0_vary_fn` already varies axis 0 of all dynamic inputs together. You need a custom `vary_fn` only when the inputs vary in a different way, not "all dynamic axes move to the same size".

To use the adapter without a registration:

```python
import downshift

model = PointCloudNet().eval()
# example_inputs() supplies the inputs
verdict = downshift.check(model, adapter=PointCloudAdapter())
```

To use it from the CLI without a registration:

```bash
downshift check my_model:model --adapter path/to/pointcloud_adapter.py:PointCloudAdapter
```

To use it with auto-detection and `--adapter pointcloud`, register it with `[project.entry-points."downshift.adapters"]`, as shown above.
