"""Hazard: custom autograd.Function with no symbolic override.

On torch 2.14 torch.export traces straight through forward() (clamp and multiply are both
traceable), so this comes back CLEAN.
"""

import torch
from torch import nn


class _ClampSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        clamped = x.clamp(-1.0, 1.0)
        ctx.save_for_backward(clamped)
        return clamped * clamped

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (clamped,) = ctx.saved_tensors
        return grad_output * 2 * clamped


class CustomAutograd(nn.Module):
    def __init__(self, features: int = 8):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ClampSquare.apply(self.linear(x))


def make_model() -> CustomAutograd:
    model = CustomAutograd()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 8),)
