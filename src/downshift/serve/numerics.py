"""The serving check: the backend that serves, on the device that serves, against the CPU
result of the same model. The verdict is made on the CPU, so this check covers the GPU."""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from downshift.core.verify import compare_sample, min_row_cosine
from downshift.serve.backends import Backend
from downshift.settings import TOLERANCE_COSINE, TOLERANCE_TF32


@dataclass
class ServingNumerics:
    provider: str  # the execution provider, or "torch"
    device: str
    tf32: bool
    samples: int
    max_abs_err: float
    min_cosine: float | None  # for an embedding model only
    ok: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        cosine = "" if self.min_cosine is None else f", min cosine {self.min_cosine:.6f}"
        return f"max abs err {self.max_abs_err:.1e} over {self.samples} samples{cosine}"


def serves_off_cpu(backend: Backend) -> bool:
    device = backend.metadata().device
    return not device.startswith(("cpu", "CPU"))


def check_serving(
    backend: Backend,
    reference: dict[str, np.ndarray],
    feeds: dict[str, np.ndarray],
    atol: float,
    rtol: float,
    *,
    device: str,
    embedding: bool,
    tf32: bool,
) -> ServingNumerics:
    """Run `feeds` through `backend` once and compare with `reference`, the CPU outputs. With
    `tf32` on, the tolerance is at least the TF32 pair."""
    if tf32:
        atol, rtol = max(atol, TOLERANCE_TF32[0]), max(rtol, TOLERANCE_TF32[1])
    got = backend.infer(feeds)
    names = list(reference)
    expected = [torch.from_numpy(reference[name]) for name in names]
    actual = [got[name] for name in names if name in got]
    max_abs, _, failed, _, _ = compare_sample(expected, actual, atol, rtol, 0)
    cosine = None
    if embedding and actual and expected[0].shape == actual[0].shape:
        cosine = min_row_cosine(reference[names[0]], actual[0])
        failed = failed or cosine < TOLERANCE_COSINE
    meta = backend.metadata()
    return ServingNumerics(
        provider=getattr(backend, "provider", meta.name.value),
        device=device,
        tf32=tf32,
        samples=1,
        max_abs_err=max_abs,
        min_cosine=cosine,
        ok=not failed,
        atol=atol,
        rtol=rtol,
    )
