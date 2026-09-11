"""Hazard: scatter_reduce(include_self=False) has no ONNX translation — known hard FAIL.

This is the exact op pattern GNN message-passing aggregation (PyG SAGEConv/GATConv) hits;
see design doc §5.5 / §5.8. Should fail loudly with a specific, actionable error, not silently.
"""

import torch
from torch import nn


class ScatterIncludeSelfFalse(nn.Module):
    def __init__(self, num_segments: int = 4, features: int = 8):
        super().__init__()
        self.num_segments = num_segments
        self.linear = nn.Linear(features, features)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        out = torch.zeros(self.num_segments, x.shape[-1], dtype=x.dtype)
        index = segment_ids.unsqueeze(-1).expand_as(x)
        return out.scatter_reduce(0, index, x, reduce="mean", include_self=False)


def make_model() -> ScatterIncludeSelfFalse:
    model = ScatterIncludeSelfFalse()
    model.eval()
    return model


def make_inputs(num_nodes: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(num_nodes, 8)
    segment_ids = torch.randint(0, 4, (num_nodes,))
    return x, segment_ids
