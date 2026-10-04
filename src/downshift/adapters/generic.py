"""Generic adapter: any nn.Module.

Handles two things. A single dataclass positional argument gets flattened into plain
tensors (torch.export rejects unregistered dataclasses outright). And when no example
inputs are given, it guesses a shape from the first Linear/Conv layer, which is enough
for the torchvision-style single-tensor case.
"""

import dataclasses
import inspect

import torch
from torch import nn

from downshift.adapters._flatten import build_shim_class
from downshift.adapters.base import Family, Prepared
from downshift.core.shapes import infer_dynamic_shapes, lower_axis_max, pin_vary_fn

_GUESS_SPATIAL = 32


def _forward_param_names(model: nn.Module) -> tuple[str, ...]:
    params = inspect.signature(model.forward).parameters.values()
    return tuple(
        p.name
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"
    )


def _guess_single_tensor_input(model: nn.Module) -> torch.Tensor | None:
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            return torch.randn(1, layer.in_features)
        if isinstance(layer, nn.Conv1d):
            return torch.randn(1, layer.in_channels, _GUESS_SPATIAL)
        if isinstance(layer, nn.Conv2d):
            return torch.randn(1, layer.in_channels, _GUESS_SPATIAL, _GUESS_SPATIAL)
        if isinstance(layer, nn.Conv3d):
            return torch.randn(1, layer.in_channels, 8, 8, 8)
        if isinstance(layer, nn.Embedding):
            return torch.randint(0, layer.num_embeddings, (1, 8))
    return None


class GenericAdapter:
    name = Family.generic

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool:
        return True

    def example_inputs(self, model: nn.Module) -> tuple | None:
        if len(_forward_param_names(model)) != 1:
            return None
        guess = _guess_single_tensor_input(model)
        return (guess,) if guess is not None else None

    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared:
        flattened = _flatten_dataclass(model, example_inputs)
        if flattened is not None:
            model, inputs, names = flattened
        else:
            inputs = example_inputs
            param_names = _forward_param_names(model)
            names = tuple(
                param_names[i] if i < len(param_names) else f"input_{i}" for i in range(len(inputs))
            )
        dynamic_shapes = lower_axis_max(infer_dynamic_shapes(inputs), axis_max)
        return Prepared(
            model=model,
            inputs=inputs,
            input_names=names,
            dynamic_shapes=dynamic_shapes,
            vary_fn=pin_vary_fn(inputs, dynamic_shapes, axis_max) if axis_max else None,
            family=self.name,
        )


def _flatten_dataclass(
    model: nn.Module, example_inputs: tuple
) -> tuple[nn.Module, tuple, tuple[str, ...]] | None:
    if len(example_inputs) != 1:
        return None
    (arg,) = example_inputs
    if not dataclasses.is_dataclass(arg) or isinstance(arg, type):
        return None

    names = tuple(f.name for f in dataclasses.fields(arg))
    tensors = tuple(getattr(arg, n) for n in names)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return None  # can't flatten a non-tensor field; let export produce the real error

    dataclass_type = type(arg)
    shim = build_shim_class(names)(model, lambda fields: dataclass_type(**fields), names)
    return shim, tensors, names
