"""The inference of dynamic shapes, and the `--dynamic` override.

Default rule: axis 0 of each tensor input is dynamic, and all of them share one Dim (the batch
case). Adapters override this where it is wrong. Example: the independent N and E of PyG.
"""

from typing import Any

import torch

from downshift.adapters.base import VaryFn
from downshift.core.axes import AxisBounds, DimBound

# The maximum of each Dim that downshift makes itself (the default rule, --dynamic, PyG).
DEFAULT_DIM_MAX = 1 << 16


def alternative_sizes(base_size: int, lo: int = 1, hi: int | None = None) -> list[int]:
    """The sizes to test a dynamic axis with, without the size at export time.

    Downshift drops candidates outside [lo, hi]. The bounds of a Dim (from dim_bounds) are
    therefore never violated. hi=None means no upper bound.
    """
    candidates = {1, 2, 3, base_size + 1, base_size * 2} - {base_size}
    candidates = {c for c in candidates if c >= lo}
    if hi is not None:
        candidates = {c for c in candidates if c <= hi}
    return sorted(candidates)


def pick_size(candidates: list[int]) -> int:
    """Draw one candidate size from the global RNG of torch.

    verify() runs each sample inside torch.random.fork_rng() after torch.manual_seed(seed). A
    draw from the global RNG therefore carries --seed into a sampler. The Adapter protocol does
    not need a new parameter. Each sampler uses this function and not `random`. `random` would
    get its seed one time, at prepare(), and would never see the seed of verify().
    """
    return int(candidates[torch.randint(len(candidates), ())])


def dim_bounds(spec: dict[int, Any] | None, axis: int) -> tuple[int, int]:
    """(min, max) for one axis of a dynamic_shapes entry, for example {0: Dim("n", min=1, max=64)}.

    If the axis is not dynamic, or the Dim does not expose the attributes in this torch
    version, it returns (1, DEFAULT_DIM_MAX). This is the one place where that can happen. It
    guards against that risk.
    """
    fallback = (1, DEFAULT_DIM_MAX)
    if not spec or axis not in spec:
        return fallback
    dim = spec[axis]
    lo = getattr(dim, "min", None)
    hi = getattr(dim, "max", None)
    if lo is None or hi is None:
        return fallback
    return lo, hi


def dynamic_bounds(input_names: tuple[str, ...], dynamic_shapes: tuple) -> AxisBounds:
    """For each input and each dynamic axis: (Dim name, min, max) as the export traced it. These
    are the served bounds that core/axes.py reports and that the request check enforces."""
    bounds: AxisBounds = {}
    for name, spec in zip(input_names, dynamic_shapes, strict=True):
        if spec:
            bounds[name] = {
                axis: DimBound(dim_name(dim, axis), *dim_bounds(spec, axis))
                for axis, dim in spec.items()
            }
    return bounds


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
    """Build a dynamic_shapes tuple from an explicit {name: [axes]} spec. Each (name, axis)
    pair gets its own independent Dim."""
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
            {axis: torch.export.Dim(f"{name}_{axis}", min=1, max=DEFAULT_DIM_MAX) for axis in axes}
        )
    return tuple(shapes)


def dim_name(dim: Any, axis: int) -> str:
    return getattr(dim, "__name__", str(axis))


def lower_axis_max(dynamic_shapes: tuple, axis_max: dict[str, int] | None) -> tuple:
    """Build each named Dim in `dynamic_shapes` again. The name and the minimum stay the same.
    The maximum becomes `axis_max[name]`. Dims that several inputs share stay shared.

    An unknown name, or a value outside the own [min, max] of the Dim, raises ValueError. The
    CLI reports this as a usage error. The ceiling is the value that the adapter set (for
    Hugging Face, the position limit of the model). --axis-max can therefore only narrow a
    bound. It can never widen it.
    """
    if not axis_max:
        return dynamic_shapes
    bounds: dict[str, tuple[int, int]] = {}
    for spec in dynamic_shapes:
        for axis, dim in (spec or {}).items():
            bounds[dim_name(dim, axis)] = dim_bounds(spec, axis)
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
    """Tile or slice `axis` to `size`. The result stays close to the example and is not pure
    noise. For floats, downshift tiles the own rows of the example and adds noise that follows
    its spread. A varied sample then looks like a plausible input. Integers (usually indices)
    stay inside the observed range. A 0-d tensor has no axis to resize, and it comes back
    unchanged."""
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
    """A verification sampler whose sample 1 is exactly at the --axis-max values. Each dynamic
    axis whose Dim is named in `axis_max` gets that size. The other axes keep the size of the
    example. The other samples come from `inner`, or from the default shared-axis-0 sampler."""
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
    """torch.export specializes a dimension of size 1 to a constant, also if it is marked as
    dynamic. Double each such axis for the trace only. Verification still uses the real sizes."""
    safe = []
    for t, spec in zip(inputs, dynamic_shapes, strict=True):
        if isinstance(t, torch.Tensor) and spec:
            for axis in spec:
                if t.shape[axis] == 1:
                    t = torch.cat([t, t], dim=axis)
        safe.append(t)
    return tuple(safe)
