# %% [markdown]
# # Lesson 4: graph neural networks
#
# Requires the `gnn` extra: `pip install -e ".[gnn]"`.
#
# A GNN's inputs don't have one batch dimension — they have two independent sizes, node
# count `N` and edge count `E`, that vary separately from graph to graph. The `pyg`
# adapter marks them as two independent `torch.export.Dim` objects rather than tying them
# to the same one, which is the detail that makes a naive GNN export work on the example
# graph and throw `INVALID_ARGUMENT` the moment node and edge counts diverge.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import downshift


class GCN(nn.Module):
    def __init__(self, in_channels: int = 8, hidden: int = 8, out_channels: int = 4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index).relu()
        return self.conv2(x, data.edge_index)


model = GCN().eval()

# %% [markdown]
# `torch_geometric.data.Data` bundles node features and edge indices into one object.
# `torch.export` can't trace a container argument directly, so the `pyg` adapter wraps
# the model in a shim that takes `x` and `edge_index` as separate flat tensors — you pass
# `Data` in, `check()` does the unwrapping.

# %%
x = torch.randn(6, 8)  # 6 nodes, 8 features each
edge_index = torch.randint(0, 6, (2, 10))  # 10 directed edges
graph = Data(x=x, edge_index=edge_index)

verdict = downshift.check(model, (graph,))
print("status:  ", verdict.status)
print("family:  ", verdict.model_family)
print("inputs:  ", verdict.input_names)
print("dynamic: ", verdict.dynamic_dims)

# %% [markdown]
# `dynamic_dims` shows `x`'s node axis and `edge_index`'s edge axis as separate entries —
# that's the independence. Verification exercises both: some of the K samples have more
# nodes than edges relative to the traced example, some have fewer, in combinations the
# export never saw.

# %%
print("shape_generalization:", verdict.numerics.shape_generalization)
print("max abs err:", verdict.numerics.max_abs_err)

# %% [markdown]
# ## Checking without an example graph
#
# The adapter can also build its own example graph, reading `in_channels` off the first
# `GCNConv`/`SAGEConv`/`GATConv` layer it finds. Handy for a quick check when you don't
# have a real graph handy yet.

# %%
verdict_no_input = downshift.check(model)
print("status:", verdict_no_input.status, "| guessed inputs:", verdict_no_input.input_names)

# %% [markdown]
# ## What actually breaks GNN exports
#
# It isn't the graph convolution itself — GCN, GraphSAGE, and GAT from PyTorch Geometric
# all export CLEAN in this environment (see `docs/compatibility.md` for the current
# numbers). The hazard is the aggregation primitive underneath hand-written
# message-passing: `scatter_reduce(..., include_self=False)`, which lesson 1's
# `SegmentMean` model demonstrates directly. A GNN built from a library layer is often
# fine; a GNN with a custom aggregation step is exactly where to run `check()` before you
# trust the export.

# %% [markdown]
# Next: [lesson 5](05_huggingface_encoder.py) checks a Hugging Face encoder model.
