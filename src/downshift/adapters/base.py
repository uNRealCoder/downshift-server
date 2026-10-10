"""What an adapter is.

An adapter knows one model family. It does two jobs. (a) It builds example inputs if the user
gave none. (b) It turns the model and the inputs into something that torch.export can trace: a
module with a flat tensor signature with a fixed number of arguments. It also gives the spec
for the dynamic shapes and a way to generate more samples for the verification.
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
        """`axis_max` is --axis-max: {Dim name: largest size to serve}. Lower the maximum of the
        named Dims (core.shapes.lower_axis_max validates the names and the ceilings). Make
        verification sample 1 exactly at those sizes (core.shapes.pin_vary_fn for the generic
        case)."""
        ...
