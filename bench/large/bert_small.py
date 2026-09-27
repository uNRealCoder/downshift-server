"""Compute-heavy counterpart to tiny_bert: a 4-layer, 256-hidden encoder over 128 tokens.

Still randomly initialised, so nothing is downloaded, but sized like an encoder somebody
would actually put behind an endpoint (sentence embeddings, classification, reranking).
"""

import torch
from transformers import BertConfig, BertModel


def make_model() -> BertModel:
    config = BertConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
    )
    model = BertModel(config)
    model.eval()
    return model


def make_inputs(batch: int = 1, seq: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, 30522, (batch, seq))
    return input_ids, torch.ones_like(input_ids)
