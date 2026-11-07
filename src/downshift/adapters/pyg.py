"""The PyG adapter: Data becomes (x, edge_index[, edge_attr]), with independent node and edge
counts. (One shared Dim works on the example graph and fails on the next.)
"""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from downshift.adapters._flatten import build_shim_class
from downshift.adapters.base import Family, Prepared, VaryFn
from downshift.core.shapes import (
    DEFAULT_DIM_MAX,
    alternative_sizes,
    dim_bounds,
    lower_axis_max,
    pick_size,
)

BASE_FIELD_NAMES = ("x", "edge_index")  # edge_attr is added if the input Data has it
_GUESS_NODES = 8
_GUESS_EDGES = 16


def is_pyg_data(example_inputs: tuple | None) -> bool:
    if example_inputs is None or len(example_inputs) != 1:
        return False
    return isinstance(example_inputs[0], Data)


def _first_in_channels(model: nn.Module) -> int | None:
    for layer in model.modules():
        if isinstance(layer, MessagePassing):
            in_channels = getattr(layer, "in_channels", None)
            if isinstance(in_channels, tuple):
                in_channels = in_channels[0]
            if isinstance(in_channels, int):
                return in_channels
    return None


class PyGAdapter:
    name = Family.pyg

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool:
        if is_pyg_data(example_inputs):
            return True
        if example_inputs is not None:
            return False
        return any(isinstance(m, MessagePassing) for m in model.modules())

    def example_inputs(self, model: nn.Module) -> tuple | None:
        in_channels = _first_in_channels(model)
        if in_channels is None:
            return None
        x = torch.randn(_GUESS_NODES, in_channels)
        edge_index = torch.randint(0, _GUESS_NODES, (2, _GUESS_EDGES))
        return (Data(x=x, edge_index=edge_index),)

    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared:
        (data,) = example_inputs
        names: tuple[str, ...] = BASE_FIELD_NAMES
        if getattr(data, "edge_attr", None) is not None:
            names += ("edge_attr",)

        inputs = tuple(getattr(data, n) for n in names)
        shim = build_shim_class(names)(model, lambda fields: Data(**fields), names)

        n_dim = torch.export.Dim("num_nodes", min=1, max=DEFAULT_DIM_MAX)
        e_dim = torch.export.Dim("num_edges", min=1, max=DEFAULT_DIM_MAX)
        axis_by_field = {"x": {0: n_dim}, "edge_index": {1: e_dim}, "edge_attr": {0: e_dim}}
        dynamic_shapes = lower_axis_max(tuple(axis_by_field[n] for n in names), axis_max)

        return Prepared(
            model=shim,
            inputs=inputs,
            input_names=names,
            dynamic_shapes=dynamic_shapes,
            vary_fn=make_vary_fn(inputs, names, dynamic_shapes, axis_max),
            family=self.name,
        )


def make_vary_fn(
    base_inputs: tuple,
    field_names: tuple[str, ...],
    dynamic_shapes: tuple | None = None,
    axis_max: dict[str, int] | None = None,
) -> VaryFn:
    """Samples with independent node and edge counts. edge_index is drawn for each sample's
    node count. With --axis-max, sample 1 is at the pinned num_nodes and num_edges."""
    x_idx = field_names.index("x")
    ei_idx = field_names.index("edge_index")
    ea_idx = field_names.index("edge_attr") if "edge_attr" in field_names else None

    base_x, base_ei = base_inputs[x_idx], base_inputs[ei_idx]
    base_n, in_channels = base_x.shape
    base_e = base_ei.shape[1]

    n_spec = dynamic_shapes[x_idx] if dynamic_shapes else None
    e_spec = dynamic_shapes[ei_idx] if dynamic_shapes else None
    n_lo, n_hi = dim_bounds(n_spec, 0)
    e_lo, e_hi = dim_bounds(e_spec, 1)
    n_candidates = alternative_sizes(base_n, n_lo, n_hi)
    e_candidates = alternative_sizes(base_e, e_lo, e_hi)

    pins = axis_max or {}
    pinned = "num_nodes" in pins or "num_edges" in pins

    def vary(i: int) -> tuple:
        if i == 0:
            return base_inputs
        if i == 1 and pinned:
            n = pins.get("num_nodes", base_n)
            e = pins.get("num_edges", base_e)
        else:
            n = pick_size(n_candidates) if n_candidates else base_n
            e = pick_size(e_candidates) if e_candidates else base_e
            # With N != E, verify() can tell outputs for nodes from outputs for edges.
            for _ in range(8):
                if e != n or not e_candidates:
                    break
                e = pick_size(e_candidates)

        sample: list = [None] * len(field_names)
        sample[x_idx] = torch.randn(n, in_channels, dtype=base_x.dtype)
        sample[ei_idx] = torch.randint(0, n, (2, e), dtype=base_ei.dtype)
        if ea_idx is not None:
            base_ea = base_inputs[ea_idx]
            sample[ea_idx] = torch.randn(e, base_ea.shape[1], dtype=base_ea.dtype)
        return tuple(sample)

    return vary
