"""What an adapter is.

An adapter knows one model family well enough to (a) build example inputs when the user
didn't give any, and (b) turn the model + inputs into something torch.export can trace:
a module with a flat, fixed-arity tensor signature, plus the dynamic-shape spec and a way
to generate more samples for verification.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import nn

VaryFn = Callable[[int], tuple]


@dataclass
class Prepared:
    model: nn.Module  # export-ready module; forward takes flat tensors
    inputs: tuple  # flat example inputs, one per input_names entry
    input_names: tuple[str, ...]
    dynamic_shapes: tuple  # per input: {axis: torch.export.Dim} or None
    vary_fn: VaryFn | None  # sample i -> inputs; None means use the shared-axis-0 default
    family: str

    @property
    def dynamic_dims(self) -> dict[str, list[int]]:
        return {
            name: sorted(spec)
            for name, spec in zip(self.input_names, self.dynamic_shapes, strict=True)
            if spec
        }


@runtime_checkable
class Adapter(Protocol):
    name: str
    family: str

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool: ...

    def example_inputs(self, model: nn.Module) -> tuple | None:
        """Tier-2 input synthesis. Return None when this adapter can't guess."""
        ...

    def prepare(self, model: nn.Module, example_inputs: tuple) -> Prepared: ...
