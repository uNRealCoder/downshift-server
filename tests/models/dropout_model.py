"""Hazard: stochastic layer. Verification is meaningless unless the model is in eval()
at export time."""

import torch
from torch import nn


class DropoutModel(nn.Module):
    def __init__(self, features: int = 16, p: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(features, features)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(x))


def make_model() -> DropoutModel:
    model = DropoutModel()
    model.eval()  # the point of this fixture; drop it and verification goes random
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 16),)
