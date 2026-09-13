# %% [markdown]
# # Lesson 6: teaching downshift a new model family
#
# `generic`, `pyg`, and `hf` cover a lot, but not everything. An adapter is how you
# extend `downshift` to a model family it doesn't know about, without touching its code.
# It's a small object with four things:
#
# - `name`, `family` — identifiers. `family` shows up in the verdict.
# - `matches(model, example_inputs)` — should this adapter handle this model?
# - `example_inputs(model)` — build inputs when the caller didn't supply any, or return
#   `None` if it can't guess.
# - `prepare(model, example_inputs)` — return a `Prepared`: an export-ready module, flat
#   example inputs, their names, the dynamic-shape spec, and an optional sampler for
#   verification.
#
# This lesson writes one for a toy "point cloud" family: a model whose `forward` takes
# two tensors, point coordinates and per-point features.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

import downshift
from downshift.adapters.base import Prepared


class PointCloudNet(nn.Module):
    """A per-point MLP: no interaction between points, so any point count works."""

    def __init__(self, coord_dim: int = 3, feature_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.encode = nn.Linear(coord_dim + feature_dim, out_dim)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.encode(torch.cat([points, features], dim=-1)).relu()


model = PointCloudNet().eval()
points = torch.randn(100, 3)
features = torch.randn(100, 16)

# %% [markdown]
# Without an adapter, `check()` falls back to `generic`, which only marks axis 0 dynamic
# — the right choice here, since points and features co-vary (both indexed by "which
# point"). For a family where that heuristic is right, you don't need a custom adapter at
# all. Write one when the shapes need something the generic axis-0 rule gets wrong, or
# when you want inputs synthesized automatically — like lesson 4's `pyg` adapter reading
# `in_channels` off the first layer.

# %%
generic_verdict = downshift.check(model, (points, features))
print("generic adapter status:", generic_verdict.status, "| family:", generic_verdict.model_family)

# %% [markdown]
# ## The adapter
#
# `PointCloudNet.forward` already takes flat tensors — no dataclass or container to
# unwrap — so `prepare()` here is close to the minimum an adapter can do. `matches()`
# checks the model's type directly; a real adapter for a family with many model classes
# would check something more structural, the way `pyg`'s adapter looks for a
# `MessagePassing` layer.

# %%
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

    def prepare(self, model: nn.Module, example_inputs: tuple) -> Prepared:
        point_dim = torch.export.Dim("num_points", min=1, max=1 << 16)
        return Prepared(
            model=model,
            inputs=example_inputs,
            input_names=("points", "features"),
            dynamic_shapes=({0: point_dim}, {0: point_dim}),
            vary_fn=None,  # the default axis-0 sampler already does the right thing here
            family=self.family,
        )


adapter = PointCloudAdapter()

# %% [markdown]
# Pass an adapter instance straight to `check()` — no registration needed for local use.

# %%
verdict = downshift.check(model, (points, features), adapter=adapter)
print("status: ", verdict.status)
print("family: ", verdict.model_family)
print("inputs: ", verdict.input_names)
print("dynamic:", verdict.dynamic_dims)

# %% [markdown]
# And the input-synthesis path works too, since `example_inputs` builds its own tensors.

# %%
guessed = downshift.check(model, adapter=adapter)
print("status:", guessed.status, "| guessed a", guessed.input_names, "pair")

# %% [markdown]
# ## Shipping it
#
# For a one-off script, passing `adapter=` is enough. To make an adapter available to
# `downshift check`/`export`/`serve` on the command line — and to auto-detection, so
# users don't have to know it exists — register it under the `downshift.adapters`
# entry-point group in your own package's `pyproject.toml`:
#
# ```toml
# [project.entry-points."downshift.adapters"]
# pointcloud = "my_package.adapters:ADAPTER"
# ```
#
# where `my_package/adapters.py` exposes `ADAPTER = PointCloudAdapter()`. Adapters are
# tried most-specific first; `generic` always goes last, so a well-written `matches()`
# is what keeps two adapters from fighting over the same model.
#
# ## Or: skip packaging entirely
#
# For an adapter that'll never live in a package, `--adapter` (and `check()`'s `adapter=`)
# also accepts a bare `.py` file directly — no install, no entry point:
#
# ```
# downshift check my_model.py:model --adapter path/to/pointcloud_adapter.py
# ```
#
# The file just needs an `ADAPTER = PointCloudAdapter()` at module level (the same shape
# as `ADAPTER` above), or `--adapter path/to/pointcloud_adapter.py:PointCloudAdapter` to
# point at the class directly — it's instantiated with no arguments. Either way, whatever
# you load has to implement the same four things as `PointCloudAdapter` here (or the
# built-in `GenericAdapter`): `name`, `family`, `matches()`, `example_inputs()`, `prepare()`.

# %% [markdown]
# Next: [lesson 7](07_cli_walkthrough.py) covers the same ground from the command line.
