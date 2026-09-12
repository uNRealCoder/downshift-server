"""Control fixture: no export hazards."""

import torch
from torch import nn


class CleanMLP(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model() -> CleanMLP:
    model = CleanMLP()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 16),)
