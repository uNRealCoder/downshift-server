"""Torch tensors as the numpy feeds a backend takes. In core rather than serve because the export
cache (core/memo.py) saves example feeds too."""

from collections.abc import Sequence

import numpy as np
import torch

# bf16/fp16 have no numpy dtype; the wire contract for both is float32 (B1).
WIDEN_DTYPES = (torch.bfloat16, torch.float16)


def widen_for_wire(t: torch.Tensor) -> torch.Tensor:
    """bf16/fp16 -> float32, everything else unchanged (B1)."""
    return t.float() if t.dtype in WIDEN_DTYPES else t


def example_feeds(input_names: Sequence[str], example_inputs: tuple) -> dict[str, np.ndarray]:
    """Example inputs as the numpy feeds a backend takes, keyed by forward-argument name.

    bf16/fp16 tensors have no numpy dtype, so they're widened to float32 first, same as the
    wire contract (B1).
    """
    return {
        name: widen_for_wire(t).numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        for name, t in zip(input_names, example_inputs, strict=True)
    }
