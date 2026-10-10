"""CLEAN: two linear layers and a ReLU. There are no export hazards."""

import torch
from torch import Tensor, nn


class CleanMLP(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


def make_model() -> CleanMLP:
    model = CleanMLP()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[Tensor]:
    return (torch.randn(batch, 16),)
