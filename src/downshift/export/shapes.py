"""Dynamic-shape inference (IMPLEMENTATION_PLAN.md §5.3).

v0.1 heuristic: mark axis 0 of every tensor input as dynamic, sharing a single
torch.export.Dim across all of them. This assumes axis-0 sizes co-vary together (the
common "batch" case, and also true of the scatter_include_self_false fixture where x's
and segment_ids's axis-0 are both the node count). GNN N/E as *independent* dynamic dims
is a PyG-adapter-specific override, not this default.
"""

import torch


def alternative_sizes(base_size: int) -> list[int]:
    """Sizes to exercise a dynamic axis with during verification, excluding the
    export-time size itself (which is always tested separately as sample 0).
    """
    return sorted({1, 2, 3, base_size + 1, base_size * 2} - {base_size})


def infer_dynamic_shapes(inputs: tuple) -> tuple:
    dim0 = torch.export.Dim("dim0", min=1, max=1 << 16)
    return tuple(
        {0: dim0} if isinstance(t, torch.Tensor) and t.ndim > 0 else None for t in inputs
    )


def safe_capture_inputs(inputs: tuple, dynamic_shapes: tuple) -> tuple:
    """Work around torch.export's 0/1 specialization: a dim of concrete size 1 gets baked
    in as a compile-time constant even when marked dynamic (confirmed empirically against
    torch 2.14 — IMPLEMENTATION_PLAN.md doesn't mention this pitfall). Duplicate along
    whichever axis is marked dynamic so the traced graph doesn't silently freeze on
    whatever example shape happened to be 1. Only used for the capture/tracing step —
    verify.py exercises the *real* example shapes, including size-1 ones, against the
    resulting graph.
    """
    safe = []
    for t, spec in zip(inputs, dynamic_shapes, strict=True):
        if isinstance(t, torch.Tensor) and spec:
            axis = next(iter(spec))
            if t.shape[axis] == 1:
                safe.append(torch.cat([t, t], dim=axis))
                continue
        safe.append(t)
    return tuple(safe)
