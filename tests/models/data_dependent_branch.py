"""Hazard: data-dependent control flow. A Python `if` on a tensor value. torch.export cannot
trace it, so on torch 2.14 the verdict is FAILED under both strict modes."""

import torch
from torch import nn


class DataDependentBranch(nn.Module):
    def __init__(self, features: int = 8):
        super().__init__()
        self.pos_branch = nn.Linear(features, features)
        self.neg_branch = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.pos_branch(x)
        return self.neg_branch(x)


def make_model() -> DataDependentBranch:
    model = DataDependentBranch()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 8),)
