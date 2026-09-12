"""Example-input synthesis ladder.

1. user-supplied      always wins
2. adapter-derived    the adapter knows its family (HF config, PyG in_channels, ...)
3. signature guess    the generic adapter's first-Linear/Conv heuristic
4. fail loudly        say exactly what to pass
"""

from torch import nn

from downshift.adapters.base import Adapter


def synthesize(model: nn.Module, adapter: Adapter, user_inputs: tuple | None) -> tuple:
    if user_inputs is not None:
        return user_inputs
    guessed = adapter.example_inputs(model)
    if guessed is not None:
        return guessed
    raise ValueError(
        f"Couldn't work out example inputs for {type(model).__name__} with the "
        f"{adapter.name!r} adapter. Pass them explicitly: from Python, "
        "check(model, example_inputs=(tensor, ...)); from the CLI, --inputs module:function "
        "where the function returns a tuple of forward() arguments."
    )
