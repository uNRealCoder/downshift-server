"""The ladder for the synthesis of example inputs.

1. user-supplied      always has priority
2. adapter-derived    the adapter knows its family (HF config, PyG in_channels, ...)
3. signature guess    the first-Linear or first-Conv rule of the generic adapter
4. fail with an error    the error says exactly what to pass
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
        f"Cannot find example inputs for {type(model).__name__} with the "
        f"{adapter.name!r} adapter. Pass them explicitly. From Python, use "
        "check(model, example_inputs=(tensor, ...)). From the CLI, use --inputs "
        "module:function. The function returns a tuple of forward() arguments."
    )
