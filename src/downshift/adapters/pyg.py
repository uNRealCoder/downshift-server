"""PyG adapter: flattens torch_geometric.data.Data into (x, edge_index[, edge_attr])
tensors and provides GNN-aware dynamic shapes and shape-varying (IMPLEMENTATION_PLAN.md
§5.3/§5.4).

Node count N and edge count E are independent dynamic dimensions — x's axis 0 is N,
edge_index's axis 1 is E — deliberately NOT sharing a single Dim the way the generic
adapter's batch-dim heuristic does. IMPLEMENTATION_PLAN.md §5.3 calls this out as the
easiest way to get a GNN export subtly wrong: linking N and E (or reusing one Dim for
both) makes the graph work on the export-time example and throw INVALID_ARGUMENT the
moment node/edge counts diverge, which they always do in practice.

This module is only imported when a PyG Data input is actually seen (downshift.export.verdict
does a lazy import), so torch_geometric stays an optional dependency for everyone else.
"""

import random
from collections.abc import Callable

import torch
from torch import nn
from torch_geometric.data import Data

from downshift.adapters._flatten import build_shim_class

BASE_FIELD_NAMES = ("x", "edge_index")  # edge_attr appended when present on the input Data


def is_pyg_data(example_inputs: tuple) -> bool:
    return len(example_inputs) == 1 and isinstance(example_inputs[0], Data)


def prepare(model: nn.Module, example_inputs: tuple) -> tuple[nn.Module, tuple, tuple, tuple[str, ...]]:
    (data,) = example_inputs
    field_names = list(BASE_FIELD_NAMES)
    if getattr(data, "edge_attr", None) is not None:
        field_names.append("edge_attr")
    field_names_t = tuple(field_names)

    flat_inputs = tuple(getattr(data, name) for name in field_names_t)

    shim_class = build_shim_class(len(field_names_t))
    shim = shim_class(model, lambda fields: Data(**fields), list(field_names_t))

    n_dim = torch.export.Dim("num_nodes", min=1, max=1 << 16)
    e_dim = torch.export.Dim("num_edges", min=1, max=1 << 16)
    axis_by_field = {"x": (0, n_dim), "edge_index": (1, e_dim), "edge_attr": (0, e_dim)}
    dynamic_shapes = tuple({axis_by_field[name][0]: axis_by_field[name][1]} for name in field_names_t)

    return shim, flat_inputs, dynamic_shapes, field_names_t


def make_vary_fn(
    base_inputs: tuple, field_names: tuple[str, ...], seed: int = 0
) -> Callable[[int], tuple]:
    """Regenerate (x, edge_index[, edge_attr]) with independently varied N and E.

    edge_index values are redrawn against the *sample's own* resampled node count, not
    the original tensor's numeric range — reusing the original range would produce
    out-of-bounds node references the moment N shrinks.
    """
    x_idx = field_names.index("x")
    edge_index_idx = field_names.index("edge_index")
    edge_attr_idx = field_names.index("edge_attr") if "edge_attr" in field_names else None

    base_n = base_inputs[x_idx].shape[0]
    base_e = base_inputs[edge_index_idx].shape[1]
    in_channels = base_inputs[x_idx].shape[1]
    x_dtype = base_inputs[x_idx].dtype
    edge_index_dtype = base_inputs[edge_index_idx].dtype
    edge_attr_dim = base_inputs[edge_attr_idx].shape[1] if edge_attr_idx is not None else None
    edge_attr_dtype = base_inputs[edge_attr_idx].dtype if edge_attr_idx is not None else None

    rng = random.Random(seed)
    n_candidates = sorted({1, 2, 3, base_n + 1, base_n * 2} - {base_n})
    e_candidates = sorted({1, 2, 3, base_e + 1, base_e * 2} - {base_e})

    def vary(i: int) -> tuple:
        if i == 0:
            return base_inputs
        n = rng.choice(n_candidates) if n_candidates else base_n
        e = rng.choice(e_candidates) if e_candidates else base_e

        sample: list = [None] * len(field_names)
        sample[x_idx] = torch.randn(n, in_channels, dtype=x_dtype)
        sample[edge_index_idx] = torch.randint(0, n, (2, e), dtype=edge_index_dtype)
        if edge_attr_idx is not None:
            sample[edge_attr_idx] = torch.randn(e, edge_attr_dim, dtype=edge_attr_dtype)
        return tuple(sample)

    return vary
