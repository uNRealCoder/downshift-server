"""Hazard: dataclass container input. The same flattening problem PyG Data has, without
the PyG dependency."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Batch:
    x: torch.Tensor
    mask: torch.Tensor


class DictInput(nn.Module):
    def __init__(self, features: int = 8):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.linear(batch.x) * batch.mask


def make_model() -> DictInput:
    model = DictInput()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[Batch]:
    return (Batch(x=torch.randn(batch, 8), mask=torch.ones(batch, 8)),)
