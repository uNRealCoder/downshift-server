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
    samples: int  # the samples that the check replayed
    samples_total: int  # the samples of the verification. Fewer replayed: a size limit
    max_abs_err: float
    min_cosine: float | None  # for an embedding model only
    ok: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        cosine = "" if self.min_cosine is None else f", min cosine {self.min_cosine:.6f}"
        of = "" if self.samples_total == self.samples else f" of {self.samples_total}"
        return f"max abs err {self.max_abs_err:.1e} over {self.samples}{of} samples{cosine}"


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
        if embedding and actual and expected[0].shape == actual[0].shape:
            row = min_row_cosine(reference[names[0]], actual[0])
            cosine = row if cosine is None else min(cosine, row)
            failed = failed or row < TOLERANCE_COSINE
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
