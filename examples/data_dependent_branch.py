"""FAILED: a Python `if` on a tensor value. torch.export cannot trace it."""

import torch
from torch import nn


class DataDependentBranch(nn.Module):
    def __init__(self, features: int = 8):
        super().__init__()
        self.pos_branch = nn.Linear(features, features)
        self.neg_branch = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor
        if x.sum() > 0:
            out = self.pos_branch(x)
        else:
            out = self.neg_branch(x)
        return out


def make_model() -> DataDependentBranch:
    model = DataDependentBranch()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 8),)
