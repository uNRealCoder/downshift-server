"""A shim that wraps a model that takes a container argument (a dataclass, a PyG Data, ...). The
shim lets torch.export see a plain tensor signature with a fixed number of arguments. Export the
shim and not the model.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


class FlattenShimBase(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        rebuild: Callable[[Mapping[str, torch.Tensor]], Any],
        field_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.model = model
        self._rebuild = rebuild
        self._field_names = field_names
        self.train(model.training)

    def _call(self, tensors: tuple[torch.Tensor, ...]) -> Any:
        return self.model(self._rebuild(dict(zip(self._field_names, tensors, strict=True))))


def build_shim_class(field_names: Sequence[str]) -> type[FlattenShimBase]:
    """Generate a subclass whose forward() has one named positional parameter for each field.

    A `def forward(self, *tensors)` would bind everything into one VAR_POSITIONAL argument.
    torch.export would then see one tuple input. This does not match a dynamic_shapes tuple
    with one entry for each input. The parameter names also become the input names of the ONNX
    graph.
    """
    params = ", ".join(n if n.isidentifier() else f"t{i}" for i, n in enumerate(field_names))
    src = f"def forward(self, {params}):\n    return self._call(({params},))\n"
    namespace: dict[str, Any] = {}
    exec(src, namespace)  # noqa: S102 - field names only. No text that a user controls
    return type("FlattenShim", (FlattenShimBase,), {"forward": namespace["forward"]})
