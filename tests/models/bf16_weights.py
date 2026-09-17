"""Hazard: bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel.

torch.export/torch.onnx.export trace and translate a bfloat16 nn.Linear without complaint;
onnxruntime's CPU execution provider is the one that gives up, raising NOT_IMPLEMENTED for
Gemm(13) in bfloat16. verify() sees a real ONNX Runtime failure, not a numeric divergence.
"""

import torch
from torch import nn


class Bf16Weights(nn.Module):
    def __init__(self, features: int = 8):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_model() -> Bf16Weights:
    model = Bf16Weights()
    model.eval()
    return model.bfloat16()


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 8).to(torch.bfloat16),)
