"""Dynamic shapes: the default rule, the `--dynamic` override, `--axis-max`, and the sample
resizing that verification uses.

Default rule: axis 0 of each tensor input is dynamic, and all inputs share one Dim (the batch
case). Adapters override this where it is wrong (PyG has independent node and edge counts).
"""

from typing import Any

import torch

from downshift.adapters.base import VaryFn
from downshift.core.axes import AxisBounds, DimBound

# The maximum of each Dim that downshift makes itself (the default rule, --dynamic, PyG).
DEFAULT_DIM_MAX = 1 << 16


def alternative_sizes(base_size: int, lo: int = 1, hi: int | None = None) -> list[int]:
    """The sizes to test a dynamic axis with, other than the export size, inside [lo, hi]."""
    candidates = {1, 2, 3, base_size + 1, base_size * 2} - {base_size}
    return sorted(c for c in candidates if c >= lo and (hi is None or c <= hi))


def pick_size(candidates: list[int]) -> int:
    """Draw one size from the global torch RNG. verify() seeds it inside fork_rng(), so this
    carries --seed into each sampler. Do not use `random` here."""
    return int(candidates[torch.randint(len(candidates), ())])


def dim_bounds(spec: dict[int, Any] | None, axis: int) -> tuple[int, int]:
    """(min, max) of one axis of a dynamic_shapes entry. (1, DEFAULT_DIM_MAX) if the axis is
    not dynamic or the Dim has no bounds."""
    dim = (spec or {}).get(axis)
    lo, hi = getattr(dim, "min", None), getattr(dim, "max", None)
    if lo is None or hi is None:
        return 1, DEFAULT_DIM_MAX
    return lo, hi


def dim_name(dim: Any, axis: int) -> str:
    return getattr(dim, "__name__", str(axis))


def dynamic_bounds(input_names: tuple[str, ...], dynamic_shapes: tuple) -> AxisBounds:
    """For each input and dynamic axis: the (Dim name, min, max) that the export traced."""
    return {
        name: {
            axis: DimBound(dim_name(dim, axis), *dim_bounds(spec, axis))
            for axis, dim in spec.items()
        }
        for name, spec in zip(input_names, dynamic_shapes, strict=True)
        if spec
    }


def infer_dynamic_shapes(inputs: tuple) -> tuple:
    dim0 = torch.export.Dim("dim0", min=1, max=DEFAULT_DIM_MAX)
    return tuple({0: dim0} if isinstance(t, torch.Tensor) and t.ndim > 0 else None for t in inputs)


def parse_dynamic_spec(spec: str) -> dict[str, list[int]]:
    """Parse "x:0,edge_index:1" or "x:0:1" into {name: [axes]}."""
    result: dict[str, list[int]] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, axes = item.partition(":")
        if not name or not axes:
            raise ValueError(f"bad --dynamic entry {item!r}. Expected name:axis[:axis...]")
        result.setdefault(name, []).extend(int(a) for a in axes.split(":"))
    return result


def apply_dynamic_override(
    input_names: tuple[str, ...], inputs: tuple, override: dict[str, list[int]]
) -> tuple:
    """A dynamic_shapes tuple from {name: [axes]}. Each (name, axis) gets its own Dim."""
    unknown = set(override) - set(input_names)
    if unknown:
        raise ValueError(f"--dynamic names {sorted(unknown)} not in inputs {list(input_names)}")
    return tuple(
        {axis: torch.export.Dim(f"{name}_{axis}", min=1, max=DEFAULT_DIM_MAX) for axis in axes}
        if (axes := override.get(name)) and isinstance(tensor, torch.Tensor)
        else None
        for name, tensor in zip(input_names, inputs, strict=True)
    )


def lower_axis_max(dynamic_shapes: tuple, axis_max: dict[str, int] | None) -> tuple:
    """Lower the maximum of each Dim named in `axis_max`. Shared Dims stay shared. An unknown
    name, or a value outside the [min, max] of the Dim, raises ValueError: --axis-max can only
    narrow a bound."""
    if not axis_max:
        return dynamic_shapes
    bounds = {
        dim_name(dim, axis): dim_bounds(spec, axis)
        for spec in dynamic_shapes
        for axis, dim in (spec or {}).items()
    }
    unknown = sorted(set(axis_max) - set(bounds))
    if unknown:
        raise ValueError(
            f"--axis-max names {unknown} are not axes of this model. Its axes are {sorted(bounds)}"
        )
    for name, n in axis_max.items():
        lo, hi = bounds[name]
        if n > hi:
            raise ValueError(f"--axis-max {name}={n} is above the limit of {hi} for that axis")
        if n < lo:
            raise ValueError(f"--axis-max {name}={n} is below the minimum of {lo} for that axis")
    lowered = {
        name: torch.export.Dim(name, min=bounds[name][0], max=n) for name, n in axis_max.items()
    }
    return tuple(
        None
        if spec is None
        else {axis: lowered.get(dim_name(dim, axis), dim) for axis, dim in spec.items()}
        for spec in dynamic_shapes
    )


def resize_axis(tensor: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    """Tile or slice `axis` to `size`, so the sample stays like the example. Floats get noise
    that follows their spread. Integers (usually indices) stay in the observed range."""
    if tensor.ndim == 0 or tensor.shape[axis] == size:
        return tensor
    moved = tensor.movedim(axis, 0)
    if moved.is_floating_point():
        reps = -(-size // moved.shape[0])
        tiled = moved.repeat(reps, *([1] * (moved.ndim - 1)))[:size]
        resized = tiled + 0.1 * moved.std(unbiased=False) * torch.randn(
            tiled.shape, dtype=moved.dtype
        )
    else:
        lo = int(moved.min().item())
        hi = max(int(moved.max().item()) + 1, lo + 1)
        resized = torch.randint(lo, hi, (size, *moved.shape[1:]), dtype=moved.dtype)
    return resized.movedim(0, axis)


def pin_vary_fn(
    inputs: tuple, dynamic_shapes: tuple, axis_max: dict[str, int], inner: VaryFn | None = None
) -> VaryFn:
    """A sampler whose sample 1 is exactly at the --axis-max sizes. The other samples come
    from `inner` (default: the shared axis-0 sampler)."""
    if inner is None:
        from downshift.core.verify import make_shared_axis0_vary_fn

        inner = make_shared_axis0_vary_fn(inputs, dynamic_shapes)

    def vary(i: int) -> tuple:
        if i != 1:
            return inner(i)
        sample = []
        for tensor, spec in zip(inputs, dynamic_shapes, strict=True):
            if isinstance(tensor, torch.Tensor) and spec:
                for axis, dim in spec.items():
                    pinned = axis_max.get(dim_name(dim, axis))
                    if pinned is not None:
                        tensor = resize_axis(tensor, axis, pinned)
            sample.append(tensor)
        return tuple(sample)

    return vary


def safe_capture_inputs(inputs: tuple, dynamic_shapes: tuple) -> tuple:
    """torch.export makes a dynamic axis of size 1 a constant. Double such axes for the trace
    only. Verification still uses the real sizes."""
    safe = []
    for t, spec in zip(inputs, dynamic_shapes, strict=True):
        if isinstance(t, torch.Tensor) and spec:
            for axis in spec:
                if t.shape[axis] == 1:
                    t = torch.cat([t, t], dim=axis)
        safe.append(t)
    return tuple(safe)
