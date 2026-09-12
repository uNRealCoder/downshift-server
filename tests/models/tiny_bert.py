"""HF fixture: a randomly initialised two-layer BERT encoder. No download, exports CLEAN."""

import torch
from transformers import BertConfig, BertModel


def make_model() -> BertModel:
    config = BertConfig(
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    model = BertModel(config)
    model.eval()
    return model


def make_inputs(batch: int = 2, seq: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, 100, (batch, seq))
    return input_ids, torch.ones_like(input_ids)
