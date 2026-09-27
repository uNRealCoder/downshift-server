"""Compute-heavy counterpart to dynamic_batch_cnn: a small conv stack on 32x32 inputs.

Also the case where the JSON wire format starts to hurt: a batch of 32 images is ~98k floats,
which is megabytes of text before a single convolution runs.
"""

import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(self.pool(x).flatten(1))


def make_model() -> ConvNet:
    model = ConvNet()
    model.eval()
    return model


def make_inputs(batch: int = 1) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 3, 32, 32),)
