"""The Adapter protocol. An adapter knows one model family: it guesses example inputs, and it
turns the model into a module with a flat tensor signature, with dynamic shapes and a sampler
for verification.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from torch import nn

VaryFn = Callable[[int], tuple]


class Family(StrEnum):
    """The built-in adapter names. A custom adapter's `name` is a plain string."""

    hf = "hf"
    pyg = "pyg"
    generic = "generic"


@dataclass
class Prepared:
    model: nn.Module  # the module that is ready for export. forward takes flat tensors
    inputs: tuple  # flat example inputs, one for each entry of input_names
    input_names: tuple[str, ...]
    dynamic_shapes: tuple  # for each input: {axis: torch.export.Dim} or None
    vary_fn: VaryFn | None  # sample i -> inputs. None means: use the shared-axis-0 default
    family: str  # the `name` of the adapter. ExportVerdict.model_family reports it

    @property
    def dynamic_dims(self) -> dict[str, list[int]]:
        return {
            name: sorted(spec)
            for name, spec in zip(self.input_names, self.dynamic_shapes, strict=True)
            if spec
        }


@runtime_checkable
class Adapter(Protocol):
    name: str  # ExportVerdict.model_family also reports it

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool: ...

    def example_inputs(self, model: nn.Module) -> tuple | None:
        """Adapter-derived example inputs, used when the user gave none. None means no guess."""
        ...

    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared:
        """`axis_max` is --axis-max: {Dim name: largest size to serve}. Lower those Dims
        (core.shapes.lower_axis_max) and make verification sample 1 exactly those sizes
        (core.shapes.pin_vary_fn)."""
        ...
