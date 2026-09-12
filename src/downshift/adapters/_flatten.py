"""Shared flatten-shim machinery (IMPLEMENTATION_PLAN.md §5.4): wrap a model that takes a
container argument (a dataclass, a torch_geometric.data.Data, ...) so torch.export sees a
plain, fixed-arity tensor signature instead. Export the shim, not the original model.

Used by adapters/generic.py (dataclass fields) and adapters/pyg.py (Data.x/edge_index/...).
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

    def _call(self, tensors: tuple[torch.Tensor, ...]) -> Any:
        container = self._rebuild(dict(zip(self._field_names, tensors, strict=True)))
        return self.model(container)


def build_shim_class(field_count: int) -> type[FlattenShimBase]:
    # A plain `def forward(self, *tensors)` binds every positional arg into a single
    # VAR_POSITIONAL parameter, so torch.export sees ONE top-level argument (a tuple),
    # not N — which fails structural matching against an N-element dynamic_shapes tuple
    # with "inputs has 1 elements, but dynamic_shapes has N elements" (confirmed
    # empirically). Give the shim a real, fixed-arity signature instead.
    params = ", ".join(f"t{i}" for i in range(field_count))
    src = f"def forward(self, {params}):\n    return self._call(({params},))\n"
    namespace: dict[str, Any] = {}
    exec(src, namespace)  # noqa: S102 - generates a plain, inspectable method, no user input
    return type("FlattenShim", (FlattenShimBase,), {"forward": namespace["forward"]})
