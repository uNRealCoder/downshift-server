"""Hazard: scatter_reduce(include_self=False) has no faithful ONNX translation.

This is the aggregation pattern PyG message passing is built on. On torch 2.14 it exports
without error under strict=False and returns wrong numbers, so the verdict is DEGRADED
rather than FAILED. Numerical verification is the only thing that catches it.
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
