# %% [markdown]
# # Lesson 1: the four verdicts
#
# `downshift.check(model, example_inputs)` traces a model through `torch.export`,
# translates it to ONNX, then runs both the PyTorch model and the ONNX graph on several
# inputs and compares the numbers. The result is one of four verdicts:
#
# - **CLEAN** — exports, matches PyTorch on every sample, survives shapes it wasn't
#   traced on.
# - **DEGRADED** — exports without error, but the numbers drift past tolerance on at
#   least one sample. This is the state most tools don't have: a graph that "succeeded"
#   and is still wrong.
# - **FAILED** — doesn't export at all. Not an error condition, just a fact.
# - **UNVERIFIED** — only reachable when you start from a pre-built `.onnx` file with no
#   PyTorch model to check it against (lesson 2 shows this).
#
# This lesson builds three small models, one for each of the first three verdicts, and
# reads the verdict object each one produces.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

import downshift

# %% [markdown]
# ## A clean model
#
# Nothing unusual here: two linear layers and a ReLU. This should export and match
# exactly.


# %%
class CleanMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


clean_model = CleanMLP().eval()
clean_inputs = (torch.randn(1, 16),)

verdict = downshift.check(clean_model, clean_inputs)
print(f"status:   {verdict.status}")
print(f"backend:  {verdict.recommended_backend}")
print(f"reason:   {verdict.reason}")
print(
    f"numerics: max abs err {verdict.numerics.max_abs_err:.2e} over "
    f"{verdict.numerics.samples_tested} samples, {verdict.numerics.failures} failed"
)

# %% [markdown]
# `verdict.numerics` is a `NumericsReport`. Its samples aren't all the same shape as
# `clean_inputs` — `check()` varies the batch dimension across the K samples specifically
# to catch a graph that only works at the shape it was traced on. That's what
# `numerics.shape_generalization` reports.

# %%
print("shape_generalization:", verdict.numerics.shape_generalization)
print("dynamic dims:", verdict.dynamic_dims)

# %% [markdown]
# ## A model that exports cleanly and lies
#
# `scatter_reduce(..., reduce="mean", include_self=False)` has no faithful ONNX
# translation. The exporter doesn't refuse — it emits a plain scatter with no reduction,
# which type-checks, runs, and returns the wrong numbers. No exception anywhere in the
# pipeline. This is exactly why numerical verification isn't optional in `downshift`.


# %%
class SegmentMean(nn.Module):
    """Average the rows that share a segment id, ignoring the zero-initialized slots."""

    def __init__(self, num_segments: int = 4, features: int = 8):
        super().__init__()
        self.num_segments = num_segments
        self.linear = nn.Linear(features, features)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        out = torch.zeros(self.num_segments, x.shape[-1], dtype=x.dtype)
        index = segment_ids.unsqueeze(-1).expand_as(x)
        return out.scatter_reduce(0, index, x, reduce="mean", include_self=False)


lying_model = SegmentMean().eval()
lying_inputs = (torch.randn(6, 8), torch.randint(0, 4, (6,)))

verdict = downshift.check(lying_model, lying_inputs)
print(f"status:   {verdict.status}")
print(f"backend:  {verdict.recommended_backend}")
print(f"reason:   {verdict.reason}")
print(
    f"{verdict.numerics.failures}/{verdict.numerics.samples_tested} samples wrong, "
    f"max abs err {verdict.numerics.max_abs_err:.2f}"
)

# %% [markdown]
# The export produced a real `.onnx` graph — `verdict.onnx_program` is set, and you
# could force-serve it with `--force-onnx` — but the recommended backend is `torch`. The
# eager PyTorch path is correct; the ONNX graph is not.

# %% [markdown]
# ## A model that doesn't export
#
# Data-dependent control flow — branching on a value only known at runtime — is the
# other broad hazard class. Here `torch.export` can't produce a single graph that covers
# both branches.


# %%
class DataDependentBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Linear(8, 8)
        self.neg = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.pos(x)
        return self.neg(x)


branch_model = DataDependentBranch().eval()
branch_inputs = (torch.randn(1, 8),)

verdict = downshift.check(branch_model, branch_inputs)
print(f"status:  {verdict.status}")
print(f"backend: {verdict.recommended_backend}")
print(f"reason:  {verdict.reason}")

# %% [markdown]
# `verdict.numerics` is `None` here — there was no ONNX graph to compare against. The
# model is still fully usable: `recommended_backend` is `torch`, and serving this model
# (lesson 3) works through the exact same HTTP contract as the clean one. FAILED is a
# supported path, not an error you have to work around.

# %% [markdown]
# ## Exit codes
#
# Each status maps to an exit code, so `downshift check` can gate a CI pipeline:
# `CLEAN` 0, `FAILED` 1, `DEGRADED` 2, `UNVERIFIED` 3.

# %%
print("clean_model exit code:", downshift.check(clean_model, clean_inputs).exit_code)
print("lying_model exit code:", downshift.check(lying_model, lying_inputs).exit_code)
print("branch_model exit code:", downshift.check(branch_model, branch_inputs).exit_code)

# %% [markdown]
# Next: [lesson 2](02_export_and_manifest.py) writes the `.onnx` artifact for the clean
# model and reads back the provenance manifest that gets written alongside it.
