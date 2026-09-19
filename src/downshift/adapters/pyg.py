"""PyG adapter: flattens torch_geometric.data.Data into (x, edge_index[, edge_attr]).

Node count N and edge count E are independent dynamic dims. Tying them to one Dim is the
classic way to get a GNN export that works on the example graph and throws
INVALID_ARGUMENT on the next one.

Only imported when a PyG Data input actually shows up, so torch_geometric stays optional.
"""

import random

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from downshift.adapters._flatten import build_shim_class
from downshift.adapters.base import Prepared, VaryFn
from downshift.core.shapes import alternative_sizes

BASE_FIELD_NAMES = ("x", "edge_index")  # edge_attr appended when present on the input Data
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
    name = "pyg"
    family = "pyg"

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

    def prepare(self, model: nn.Module, example_inputs: tuple) -> Prepared:
        (data,) = example_inputs
        names: tuple[str, ...] = BASE_FIELD_NAMES
        if getattr(data, "edge_attr", None) is not None:
            names += ("edge_attr",)

        inputs = tuple(getattr(data, n) for n in names)
        shim = build_shim_class(names)(model, lambda fields: Data(**fields), names)

        n_dim = torch.export.Dim("num_nodes", min=1, max=1 << 16)
        e_dim = torch.export.Dim("num_edges", min=1, max=1 << 16)
        axis_by_field = {"x": {0: n_dim}, "edge_index": {1: e_dim}, "edge_attr": {0: e_dim}}

        return Prepared(
            model=shim,
            inputs=inputs,
            input_names=names,
            dynamic_shapes=tuple(axis_by_field[n] for n in names),
            vary_fn=make_vary_fn(inputs, names),
            family=self.family,
        )


def make_vary_fn(base_inputs: tuple, field_names: tuple[str, ...], seed: int = 0) -> VaryFn:
    """Regenerate (x, edge_index[, edge_attr]) with independently varied N and E.

    edge_index is redrawn against the sample's own node count, not the original tensor's
    value range, so a shrunken graph never references nodes it doesn't have.
    """
    x_idx = field_names.index("x")
    ei_idx = field_names.index("edge_index")
    ea_idx = field_names.index("edge_attr") if "edge_attr" in field_names else None

    base_x, base_ei = base_inputs[x_idx], base_inputs[ei_idx]
    base_n, in_channels = base_x.shape
    base_e = base_ei.shape[1]

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
        sample[ei_idx] = torch.randint(0, n, (2, e), dtype=base_ei.dtype)
        if ea_idx is not None:
            base_ea = base_inputs[ea_idx]
            sample[ea_idx] = torch.randn(e, base_ea.shape[1], dtype=base_ea.dtype)
        return tuple(sample)

    return vary


ADAPTER = PyGAdapter()
