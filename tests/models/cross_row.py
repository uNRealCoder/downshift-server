"""A model whose row output depends on the other rows of the batch: the mean over axis 0."""

import torch
from torch import nn


class CrossRow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return h - h.mean(0, keepdim=True)


def make_model() -> CrossRow:
    model = CrossRow()
    model.eval()
    return model


def make_inputs(batch: int = 3) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 8),)
