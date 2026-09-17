# %% [markdown]
# # Lesson 2: writing the artifact and reading its manifest
#
# `downshift.check()` from lesson 1 never touches disk. `downshift.export()` does the
# same work and then writes the `.onnx` file plus a `.manifest.json` sidecar next to it —
# the provenance record: source checksum, library versions, opset, observed dtype, and
# the full verdict.

# %%
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

import downshift


class CleanMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = CleanMLP().eval()
inputs = (torch.randn(1, 16),)

out_dir = Path(tempfile.mkdtemp(prefix="downshift-tutorial-"))
onnx_path = out_dir / "clean_mlp.onnx"

verdict = downshift.export(model, onnx_path, inputs)
print("status:", verdict.status)
print("wrote:", sorted(p.name for p in out_dir.iterdir()))

# %% [markdown]
# ## The manifest
#
# `export()` writes `<stem>.manifest.json` next to the `.onnx` file automatically. It's
# meant to answer "where did this file come from and what checked it" without needing
# the training code around.

# %%
manifest_path = onnx_path.with_suffix(".manifest.json")
manifest = json.loads(manifest_path.read_text())
print(json.dumps(manifest, indent=2))

# %% [markdown]
# A few fields worth knowing:
#
# - `onnx_sha256` / `source_sha256` — hash of the artifact, and of the source checkpoint
#   file when the model came from one (`None` here, since `model` was built in memory).
# - `observed_dtype` — what the export actually produced, read back out of the graph's
#   weights. Not a setting you chose; a fact about the file.
# - `verdict` — the same object from lesson 1, serialized.

# %% [markdown]
# ## `--fp16`
#
# `fp16=True` casts the model to half precision before tracing. It's a plain `.half()`
# call, not a quantization pass — `downshift` doesn't do quantization at all. Tolerances
# widen automatically for the lower precision.

# %%
fp16_path = out_dir / "clean_mlp_fp16.onnx"
fp16_verdict = downshift.export(model, fp16_path, inputs, fp16=True)
print("fp16 status:", fp16_verdict.status)
fp16_manifest = json.loads(fp16_path.with_suffix(".manifest.json").read_text())
print("observed dtype:", fp16_manifest["observed_dtype"])

# %% [markdown]
# ## `--no-verify`
#
# Skipping verification is an explicit escape hatch, not a default. The verdict says so:
# the status is `UNVERIFIED`, never `CLEAN`, because nobody checked.

# %%
unverified_path = out_dir / "clean_mlp_unverified.onnx"
unverified = downshift.export(model, unverified_path, inputs, verify_numerics=False)
print("status:", unverified.status)
print("reason:", unverified.reason)

# %% [markdown]
# ## A FAILED export writes nothing
#
# Going back to lesson 1's branching model: since there's no ONNX graph to save, `export`
# writes no `.onnx` and no manifest. Check the verdict before assuming a file exists.


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


before = sorted(out_dir.iterdir())
failed_verdict = downshift.export(
    DataDependentBranch().eval(), out_dir / "branch.onnx", (torch.randn(1, 8),)
)
after = sorted(out_dir.iterdir())
print("status:", failed_verdict.status)
print("onnx_path:", failed_verdict.onnx_path)
print("directory unchanged:", after == before)

# %% [markdown]
# Next: [lesson 3](03_serve_and_query.py) serves the clean model over HTTP and shows the
# same `/predict` contract working for a model that never produced an ONNX graph at all.
