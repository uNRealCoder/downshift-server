"""Torch tensors as the numpy feeds that a backend takes."""

from collections.abc import Sequence

import numpy as np
import torch

# bf16 has no numpy dtype. On the wire, bf16 and fp16 are float32.
WIDEN_DTYPES = (torch.bfloat16, torch.float16)


def widen_for_wire(t: torch.Tensor) -> torch.Tensor:
    """bf16/fp16 -> float32, everything else unchanged."""
    return t.float() if t.dtype in WIDEN_DTYPES else t


def example_feeds(input_names: Sequence[str], example_inputs: tuple) -> dict[str, np.ndarray]:
    """{forward argument name: numpy array}, with bf16 and fp16 widened to float32."""
    return {
        name: widen_for_wire(t).numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        for name, t in zip(input_names, example_inputs, strict=True)
    }
