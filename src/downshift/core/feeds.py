"""Torch tensors as the numpy feeds that a backend takes. This module is in core and not in serve,
because the export cache (core/memo.py) also saves example feeds."""

from collections.abc import Sequence

import numpy as np
import torch

# bf16 and fp16 have no numpy dtype. The wire contract for both is float32 (B1).
WIDEN_DTYPES = (torch.bfloat16, torch.float16)


def widen_for_wire(t: torch.Tensor) -> torch.Tensor:
    """bf16/fp16 -> float32, everything else unchanged (B1)."""
    return t.float() if t.dtype in WIDEN_DTYPES else t


def example_feeds(input_names: Sequence[str], example_inputs: tuple) -> dict[str, np.ndarray]:
    """Example inputs as the numpy feeds that a backend takes. The key is the name of the
    forward argument.

    bf16 and fp16 tensors have no numpy dtype. Downshift widens them to float32 first. This is
    the same as the wire contract (B1).
    """
    return {
        name: widen_for_wire(t).numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        for name, t in zip(input_names, example_inputs, strict=True)
    }
