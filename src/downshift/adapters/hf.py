"""Hugging Face adapter, encoder-only models.

Builds input_ids / attention_mask straight from the model config rather than pulling in
optimum. The export shim unwraps the ModelOutput so torch.export sees a plain tensor
(last_hidden_state for base models, logits for heads).

Only imported when transformers is installed.
"""

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel

from downshift.adapters.base import Prepared, VaryFn
from downshift.core.shapes import alternative_sizes

INPUT_NAMES = ("input_ids", "attention_mask")
_GUESS_BATCH = 2
_GUESS_SEQ = 8


class _FirstOutputShim(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.train(model.training)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        first: torch.Tensor = out[0]
        return first


def load_pretrained(repo_id_or_path: str) -> PreTrainedModel:
    """repo_id_or_path is either a Hugging Face hub id or a local directory containing a
    previously downloaded repo (config.json, weights, etc.) -- from_pretrained handles both."""
    model: PreTrainedModel = AutoModel.from_pretrained(repo_id_or_path)
    return model


class HFAdapter:
    name = "hf"
    family = "hf-transformers"

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool:
        return isinstance(model, PreTrainedModel)

    def example_inputs(self, model: nn.Module) -> tuple | None:
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab is None:
            return None
        input_ids = torch.randint(0, vocab, (_GUESS_BATCH, _GUESS_SEQ))
        return input_ids, torch.ones_like(input_ids)

    def prepare(self, model: nn.Module, example_inputs: tuple) -> Prepared:
        input_ids, attention_mask = example_inputs
        # getattr, not model.config: nn.Module's typeshed makes attribute access resolve to
        # Tensor | Module, losing the actual PretrainedConfig type getattr(..., str) keeps as Any.
        config = getattr(model, "config")  # noqa: B009
        # Position embeddings cap the sequence length; a looser bound trips export's guards.
        max_seq = int(getattr(config, "max_position_embeddings", 1 << 12))
        batch = torch.export.Dim("batch", min=1, max=1 << 12)
        seq = torch.export.Dim("seq", min=1, max=max_seq)
        spec = {0: batch, 1: seq}
        vocab = int(config.vocab_size)
        inputs = (input_ids, attention_mask)
        return Prepared(
            model=_FirstOutputShim(model),
            inputs=inputs,
            input_names=INPUT_NAMES,
            dynamic_shapes=(spec, spec),
            vary_fn=make_vary_fn(inputs, vocab, max_seq),
            family=self.family,
        )


def make_vary_fn(base_inputs: tuple, vocab_size: int, max_seq: int = 1 << 12) -> VaryFn:
    """Verification samples after the first vary batch and sequence length, and pad: each
    row gets its own random length in [1, s] with the mask zeroed beyond it (and at least
    one attended position), so padding is actually exercised rather than always-full masks.
    """
    input_ids, _ = base_inputs
    base_batch, base_seq = input_ids.shape
    batch_candidates = alternative_sizes(base_batch)
    seq_candidates = alternative_sizes(base_seq, 1, max_seq)

    def vary(i: int) -> tuple:
        if i == 0:
            return base_inputs
        if batch_candidates:
            b = batch_candidates[torch.randint(len(batch_candidates), ())]
        else:
            b = base_batch
        if seq_candidates:
            s = seq_candidates[torch.randint(len(seq_candidates), ())]
        else:
            s = base_seq
        ids = torch.randint(0, vocab_size, (b, s), dtype=input_ids.dtype)
        lengths = torch.randint(1, s + 1, (b,))
        mask = (torch.arange(s).unsqueeze(0) < lengths.unsqueeze(1)).to(input_ids.dtype)
        return ids, mask

    return vary


ADAPTER = HFAdapter()
