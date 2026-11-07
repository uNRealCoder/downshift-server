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
        name: widen_for_wire(t).detach().cpu().numpy()
        if isinstance(t, torch.Tensor)
        else np.asarray(t)
        for name, t in zip(input_names, example_inputs, strict=True)
    }


# The verification samples that the serving check replays, in bytes. A larger set is cut down.
MAX_SAMPLE_BYTES = 64 * 1024 * 1024


def bound_samples(
    samples: list[dict[str, np.ndarray]], limit: int = MAX_SAMPLE_BYTES
) -> list[dict[str, np.ndarray]]:
    """The samples that fit in `limit` bytes, in their first order. The smallest and the largest
    come first, then the others from small to large. The loop stops at the first one that does
    not fit."""
    sizes = [sum(a.nbytes for a in sample.values()) for sample in samples]
    order = sorted(range(len(samples)), key=sizes.__getitem__)
    if len(order) > 2:
        order = [order[0], order[-1], *order[1:-1]]
    kept: list[int] = []
    total = 0
    for index in order:
        if total + sizes[index] > limit:
            break
        kept.append(index)
        total += sizes[index]
    return [samples[i] for i in sorted(kept)]
