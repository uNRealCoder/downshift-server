"""Compute-heavy counterpart to clean_mlp. Same shape of model, enough FLOPs to be visible.

The fixture corpus in tests/models is sized for export hazards, not for arithmetic: clean_mlp
is 16->32->4 and runs in ~50us, which is far below the cost of the HTTP round trip that carries
it. This one is wide enough that the backend actually shows up in the throughput number.
"""

import torch
from torch import nn


class WideMLP(nn.Module):
    def __init__(self, in_features: int = 512, hidden: int = 2048, out_features: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model() -> WideMLP:
    model = WideMLP()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 512),)
