"""The Adapter protocol. An adapter knows one model family: it guesses example inputs, and it
turns the model into a module with a flat tensor signature, with dynamic shapes and a sampler
for verification.
"""

from typing import Protocol, runtime_checkable

from torch import nn

from downshift.entities.adapters import Prepared


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
