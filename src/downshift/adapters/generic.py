"""Generic adapter: handles an arbitrary nn.Module.

The only export hazard it actively works around today is a single dataclass positional
argument (IMPLEMENTATION_PLAN.md §5.4's container-flattening problem, in miniature).
torch.export only accepts pytree containers of basic types (Tensor, int, float, ...) —
a plain dataclass is rejected outright unless registered or flattened first. adapters/pyg.py
does the same thing for torch_geometric.data.Data via the same shim machinery.
"""

import dataclasses

import torch
from torch import nn

from downshift.adapters._flatten import build_shim_class


def prepare(model: nn.Module, example_inputs: tuple) -> tuple[nn.Module, tuple]:
    """Flatten a single dataclass positional argument into plain tensors, if present.

    Returns (model, example_inputs) unchanged when there's nothing to flatten.
    """
    if len(example_inputs) != 1:
        return model, example_inputs

    (arg,) = example_inputs
    if not dataclasses.is_dataclass(arg) or isinstance(arg, type):
        return model, example_inputs

    field_names = [f.name for f in dataclasses.fields(arg)]
    flat_inputs = tuple(getattr(arg, name) for name in field_names)
    if not all(isinstance(t, torch.Tensor) for t in flat_inputs):
        # A field isn't a tensor — flattening it into torch.export's arg tuple wouldn't
        # help, so leave the original model/inputs and let export fail with a real error.
        return model, example_inputs

    dataclass_type = type(arg)
    shim_class = build_shim_class(len(field_names))
    shim = shim_class(model, lambda fields: dataclass_type(**fields), field_names)
    return shim, flat_inputs
