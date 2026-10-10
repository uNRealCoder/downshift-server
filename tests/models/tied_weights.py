"""Hazard: tied embedding/output weight (GPT-2/OPT-style). It exports as CLEAN on torch 2.14. The
shared storage shows as a warning of the verdict."""

import torch
from torch import nn


class TiedWeights(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.output = nn.Linear(hidden, vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        return self.output(hidden)


def make_model() -> TiedWeights:
    model = TiedWeights()
    model.eval()
    return model


def make_inputs(batch: int = 1, seq_len: int = 5) -> tuple[torch.Tensor]:
    return (torch.randint(0, 32, (batch, seq_len)),)
