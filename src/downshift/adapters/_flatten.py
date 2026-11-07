"""A shim for a model that takes a container argument (a dataclass, a PyG Data): it gives
torch.export a flat tensor signature."""

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
    """A subclass whose forward() has one named parameter for each field. (*tensors would be
    one tuple input to torch.export.) The names become the ONNX input names."""
    params = ", ".join(n if n.isidentifier() else f"t{i}" for i, n in enumerate(field_names))
    src = f"def forward(self, {params}):\n    return self._call(({params},))\n"
    namespace: dict[str, Any] = {}
    exec(src, namespace)  # noqa: S102 - field names only. No text that a user controls
    return type("FlattenShim", (FlattenShimBase,), {"forward": namespace["forward"]})
