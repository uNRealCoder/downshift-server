"""Hazard: stochastic layer. Verification has no meaning unless the model is in eval() at the
time of the export."""

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
    model.eval()  # the purpose of this fixture. Without it, verification becomes random
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 16),)
