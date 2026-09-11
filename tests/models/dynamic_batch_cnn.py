"""Hazard: batch-dim generalization. Export-time batch size must not get baked in."""

import torch
from torch import nn


class DynamicBatchCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def make_model() -> DynamicBatchCNN:
    model = DynamicBatchCNN()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 3, 16, 16),)
