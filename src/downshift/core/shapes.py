"""Dynamic-shape inference and the `--dynamic` override.

Default heuristic: axis 0 of every tensor input is dynamic and they all share one Dim
(the batch case). Adapters override this where it's wrong, e.g. PyG's independent N/E.
"""

from typing import Any

import torch

from downshift.adapters.base import VaryFn


def alternative_sizes(base_size: int, lo: int = 1, hi: int | None = None) -> list[int]:
    """Sizes to exercise a dynamic axis with, excluding the export-time size.

    Candidates outside [lo, hi] are dropped, so a Dim's bounds (from dim_bounds) are never
    violated; hi=None means no upper bound.
    """
    candidates = {1, 2, 3, base_size + 1, base_size * 2} - {base_size}
    candidates = {c for c in candidates if c >= lo}
    if hi is not None:
        candidates = {c for c in candidates if c <= hi}
    return sorted(candidates)


def pick_size(candidates: list[int]) -> int:
    """Draw one candidate size from torch's global RNG.

    verify() runs every sample inside torch.random.fork_rng() after torch.manual_seed(seed),
    so drawing from the global RNG is what carries --seed into a sampler without widening the
    Adapter protocol. Every sampler goes through here rather than through `random`, which
    would be seeded once at prepare() time and never see the seed at all.
    """
    return int(candidates[torch.randint(len(candidates), ())])


def dim_bounds(spec: dict[int, Any] | None, axis: int) -> tuple[int, int]:
    """(min, max) for one axis of a dynamic_shapes entry, e.g. {0: Dim("n", min=1, max=64)}.

    Falls back to (1, 1 << 16) when the axis isn't dynamic or the Dim doesn't expose the
    attributes on this torch version - the one place that can happen, per the risk it guards.
    """
    fallback = (1, 1 << 16)
    if not spec or axis not in spec:
        return fallback
    dim = spec[axis]
    lo = getattr(dim, "min", None)
    hi = getattr(dim, "max", None)
    if lo is None or hi is None:
        return fallback
    return lo, hi


def dynamic_bounds(
    input_names: tuple[str, ...], dynamic_shapes: tuple
) -> dict[str, dict[int, tuple[str, int, int]]]:
    """Per input, per dynamic axis: (Dim name, min, max) as the export traced it. The served
    bounds that core/axes.py reports and the request check enforces."""
    bounds: dict[str, dict[int, tuple[str, int, int]]] = {}
    for name, spec in zip(input_names, dynamic_shapes, strict=True):
        if spec:
            bounds[name] = {
                axis: (getattr(dim, "__name__", str(axis)), *dim_bounds(spec, axis))
                for axis, dim in spec.items()
            }
    return bounds


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


def _dim_name(dim: Any, axis: int) -> str:
    return getattr(dim, "__name__", str(axis))


def lower_axis_max(dynamic_shapes: tuple, axis_max: dict[str, int] | None) -> tuple:
    """Rebuild each named Dim in `dynamic_shapes` with the same name and min and the max
    lowered to `axis_max[name]`; Dims shared by several inputs stay shared.

    An unknown name or a value outside the Dim's own [min, max] raises ValueError, which the
    CLI reports as a usage error. The ceiling is whatever the adapter set (for Hugging Face
    the model's position limit), so --axis-max can only narrow a bound, never widen it.
    """
    if not axis_max:
        return dynamic_shapes
    bounds: dict[str, tuple[int, int]] = {}
    for spec in dynamic_shapes:
        for axis, dim in (spec or {}).items():
            bounds[_dim_name(dim, axis)] = dim_bounds(spec, axis)
    unknown = sorted(set(axis_max) - set(bounds))
    if unknown:
        raise ValueError(
            f"--axis-max names {unknown} are not axes of this model; its axes are {sorted(bounds)}"
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
        else {axis: lowered.get(_dim_name(dim, axis), dim) for axis, dim in spec.items()}
        for spec in dynamic_shapes
    )


def _resize_axis(tensor: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    """Tile or slice `axis` to `size`: floats get noise scaled to the example's spread,
    integers stay inside the observed value range."""
    if tensor.shape[axis] == size:
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
    """Verification sampler whose sample 1 sits exactly at the --axis-max values: every
    dynamic axis whose Dim is named in `axis_max` is resized to it, the rest keep the example's
    size. Other samples come from `inner`, or from the default shared-axis-0 sampler."""
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
                    pinned = axis_max.get(_dim_name(dim, axis))
                    if pinned is not None:
                        tensor = _resize_axis(tensor, axis, pinned)
            sample.append(tensor)
        return tuple(sample)

    return vary


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
