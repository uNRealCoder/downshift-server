"""The serving check: the backend that serves, on the device that serves, against the CPU
result of the same model. The verdict is made on the CPU, so this check covers the GPU."""

import numpy as np
import torch

from downshift.core.verify import compare_sample, fold_cosine
from downshift.entities.serving import ServingNumerics
from downshift.serve.backends import Backend
from downshift.settings import TOLERANCE_TF32


def serves_off_cpu(backend: Backend) -> bool:
    device = backend.metadata().device
    return not device.startswith(("cpu", "CPU"))


def check_serving(
    backend: Backend,
    references: list[dict[str, np.ndarray]],
    feeds: list[dict[str, np.ndarray]],
    atol: float,
    rtol: float,
    *,
    device: str,
    embedding: bool,
    tf32: bool,
    total: int,
) -> ServingNumerics:
    """Run each of `feeds` through `backend` and compare with `references`, the CPU outputs.
    `total` is the number of verification samples. With `tf32` on, the tolerance is at least
    the TF32 pair."""
    if tf32:
        atol, rtol = max(atol, TOLERANCE_TF32[0]), max(rtol, TOLERANCE_TF32[1])
    max_abs = 0.0
    cosine: float | None = None
    failed = False
    for reference, sample in zip(references, feeds, strict=True):
        got = backend.infer(sample)
        names = list(reference)
        expected = [torch.from_numpy(reference[name]) for name in names]
        actual = [got[name] for name in names if name in got]
        sample_abs, _, sample_failed, _, _ = compare_sample(expected, actual, atol, rtol, 0)
        max_abs = max(max_abs, sample_abs)
        failed = failed or sample_failed
        if embedding and actual:
            cosine, low = fold_cosine(reference[names[0]], actual[0], cosine)
            failed = failed or low
    meta = backend.metadata()
    return ServingNumerics(
        provider=getattr(backend, "provider", meta.name.value),
        device=device,
        tf32=tf32,
        samples=len(feeds),
        samples_total=max(total, len(feeds)),
        max_abs_err=max_abs,
        min_cosine=cosine,
        ok=not failed,
        atol=atol,
        rtol=rtol,
    )
