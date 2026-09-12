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
from downshift.export.shapes import alternative_sizes

BASE_FIELD_NAMES = ("x", "edge_index")  # edge_attr appended when present on the input Data


def is_pyg_data(example_inputs: tuple) -> bool:
    return len(example_inputs) == 1 and isinstance(example_inputs[0], Data)


def prepare(model: nn.Module, example_inputs: tuple) -> tuple[nn.Module, tuple, tuple, tuple[str, ...]]:
    (data,) = example_inputs
    field_names: tuple[str, ...] = BASE_FIELD_NAMES
    if getattr(data, "edge_attr", None) is not None:
        field_names += ("edge_attr",)

    flat_inputs = tuple(getattr(data, name) for name in field_names)

    shim_class = build_shim_class(len(field_names))
    shim = shim_class(model, lambda fields: Data(**fields), field_names)

    n_dim = torch.export.Dim("num_nodes", min=1, max=1 << 16)
    e_dim = torch.export.Dim("num_edges", min=1, max=1 << 16)
    dynamic_axis_by_field = {"x": {0: n_dim}, "edge_index": {1: e_dim}, "edge_attr": {0: e_dim}}
    dynamic_shapes = tuple(dynamic_axis_by_field[name] for name in field_names)

    return shim, flat_inputs, dynamic_shapes, field_names


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

    base_x = base_inputs[x_idx]
    base_edge_index = base_inputs[edge_index_idx]
    # Pair the index with its tensor so one None check narrows both below.
    edge_attr = (edge_attr_idx, base_inputs[edge_attr_idx]) if edge_attr_idx is not None else None

    base_n, in_channels = base_x.shape[0], base_x.shape[1]
    base_e = base_edge_index.shape[1]

    rng = random.Random(seed)
    n_candidates = alternative_sizes(base_n)
    e_candidates = alternative_sizes(base_e)

    def vary(i: int) -> tuple:
        if i == 0:
            return base_inputs
        n = rng.choice(n_candidates) if n_candidates else base_n
        e = rng.choice(e_candidates) if e_candidates else base_e

        sample: list = [None] * len(field_names)
        sample[x_idx] = torch.randn(n, in_channels, dtype=base_x.dtype)
        sample[edge_index_idx] = torch.randint(0, n, (2, e), dtype=base_edge_index.dtype)
        if edge_attr is not None:
            idx, base_edge_attr = edge_attr
            sample[idx] = torch.randn(e, base_edge_attr.shape[1], dtype=base_edge_attr.dtype)
        return tuple(sample)

    return vary
