"""Dynamic-shape inference and the `--dynamic` override.

Default heuristic: axis 0 of every tensor input is dynamic and they all share one Dim
(the batch case). Adapters override this where it's wrong, e.g. PyG's independent N/E.
"""

from typing import Any

import torch


def alternative_sizes(base_size: int) -> list[int]:
    """Sizes to exercise a dynamic axis with, excluding the export-time size."""
    return sorted({1, 2, 3, base_size + 1, base_size * 2} - {base_size})


def infer_dynamic_shapes(inputs: tuple) -> tuple:
    dim0 = torch.export.Dim("dim0", min=1, max=1 << 16)
    return tuple({0: dim0} if isinstance(t, torch.Tensor) and t.ndim > 0 else None for t in inputs)


def parse_dynamic_spec(spec: str) -> dict[str, list[int]]:
    """Parse "x:0,edge_index:1" or "x:0:1" into {name: [axes]}."""
    result: dict[str, list[int]] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, axes = item.partition(":")
        if not name or not axes:
            raise ValueError(f"bad --dynamic entry {item!r}; expected name:axis[:axis...]")
        result.setdefault(name, []).extend(int(a) for a in axes.split(":"))
    return result


def apply_dynamic_override(
    input_names: tuple[str, ...], inputs: tuple, override: dict[str, list[int]]
) -> tuple:
    """Build a dynamic_shapes tuple from an explicit {name: [axes]} spec. Every
    (name, axis) pair gets its own independent Dim."""
    unknown = set(override) - set(input_names)
    if unknown:
        raise ValueError(f"--dynamic names {sorted(unknown)} not in inputs {list(input_names)}")
    shapes: list[dict[int, Any] | None] = []
    for name, tensor in zip(input_names, inputs, strict=True):
        axes = override.get(name)
        if not axes or not isinstance(tensor, torch.Tensor):
            shapes.append(None)
            continue
        shapes.append(
            {axis: torch.export.Dim(f"{name}_{axis}", min=1, max=1 << 16) for axis in axes}
        )
    return tuple(shapes)


def safe_capture_inputs(inputs: tuple, dynamic_shapes: tuple) -> tuple:
    """torch.export specialises a size-1 dim to a constant even when it's marked dynamic.
    Double any such axis for the trace only; verification still uses the real sizes."""
    safe = []
    for t, spec in zip(inputs, dynamic_shapes, strict=True):
        if isinstance(t, torch.Tensor) and spec:
            for axis in spec:
                if t.shape[axis] == 1:
                    t = torch.cat([t, t], dim=axis)
        safe.append(t)
    return tuple(safe)
