"""The Hugging Face adapter: input_ids and attention_mask from the config, and a shim that
exports the first output (last_hidden_state, or a head's logits), pooled if the repo has a
recipe. Imported only if transformers is installed.
"""

import torch
from torch import nn
from transformers import PreTrainedModel

from downshift.adapters.base import Family, Prepared, VaryFn
from downshift.adapters.embedding import (
    EMBEDDING_ATTR,
    PADDING_SIDE_ATTR,
    EmbeddingRecipe,
    PoolingHead,
)
from downshift.core.shapes import alternative_sizes, lower_axis_max, pick_size
from downshift.hf_repo import position_limit

INPUT_NAMES = ("input_ids", "attention_mask")
_GUESS_BATCH = 2
_GUESS_SEQ = 8


class _FirstOutputShim(nn.Module):
    """The first output of the model, pooled into one vector per text if there is a recipe."""

    def __init__(self, model: nn.Module, recipe: EmbeddingRecipe | None = None) -> None:
        super().__init__()
        self.model = model
        self.head = PoolingHead(recipe) if recipe is not None else None
        self.train(model.training)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        first: torch.Tensor = out[0]
        return self.head(first, attention_mask) if self.head is not None else first


class HFAdapter:
    name = Family.hf

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool:
        return isinstance(model, PreTrainedModel)

    def example_inputs(self, model: nn.Module) -> tuple | None:
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab is None:
            return None
        input_ids = torch.randint(0, vocab, (_GUESS_BATCH, _GUESS_SEQ))
        return input_ids, torch.ones_like(input_ids)

    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared:
        input_ids, attention_mask = example_inputs
        # getattr keeps the type Any; nn.Module attribute access types it as Tensor | Module.
        config = getattr(model, "config")  # noqa: B009
        # The position embeddings bound the sequence; a looser bound trips export guards.
        longest = position_limit(config)
        max_seq = longest or 1 << 12
        batch = torch.export.Dim("batch", min=1, max=1 << 12)
        seq = torch.export.Dim("seq", min=1, max=max_seq)
        spec = {0: batch, 1: seq}
        dynamic_shapes = lower_axis_max((spec, spec), axis_max)
        vocab = int(config.vocab_size)
        inputs = (input_ids, attention_mask)
        return Prepared(
            model=_FirstOutputShim(model, getattr(model, EMBEDDING_ATTR, None)),
            inputs=inputs,
            input_names=INPUT_NAMES,
            dynamic_shapes=dynamic_shapes,
            vary_fn=make_vary_fn(
                inputs,
                vocab,
                max_seq,
                longest,
                axis_max,
                getattr(model, PADDING_SIDE_ATTR, "right"),
            ),
            family=self.name,
        )


def make_vary_fn(
    base_inputs: tuple,
    vocab_size: int,
    max_seq: int = 1 << 12,
    longest: int | None = None,
    axis_max: dict[str, int] | None = None,
    padding_side: str = "right",
) -> VaryFn:
    """Samples that vary batch and sequence length, with real padding: each row has its own
    length, padded on `padding_side`.

    Sample 1 is one row at `longest` (the position limit), so a graph that fails only on long
    sequences cannot pass. With --axis-max, sample 1 is at the pinned sizes instead, and the
    other samples stay within them.
    """
    input_ids, _ = base_inputs
    base_batch, base_seq = input_ids.shape
    pins = axis_max or {}
    batch_candidates = alternative_sizes(base_batch, 1, pins.get("batch"))
    seq_candidates = alternative_sizes(base_seq, 1, min(max_seq, pins.get("seq", max_seq)))
    pinned_seq = pins.get("seq", longest)
    pinned_batch = pins.get("batch")

    def vary(i: int) -> tuple:
        if i == 0:
            return base_inputs
        if i == 1 and (pinned_seq is not None or pinned_batch is not None):
            shape = (pinned_batch or 1, pinned_seq or base_seq)
            ids = torch.randint(0, vocab_size, shape, dtype=input_ids.dtype)
            return ids, torch.ones_like(ids)
        b = pick_size(batch_candidates) if batch_candidates else base_batch
        s = pick_size(seq_candidates) if seq_candidates else base_seq
        ids = torch.randint(0, vocab_size, (b, s), dtype=input_ids.dtype)
        lengths = torch.randint(1, s + 1, (b,))
        positions = torch.arange(s).unsqueeze(0)
        if padding_side == "left":
            attended = positions >= (s - lengths).unsqueeze(1)
        else:
            attended = positions < lengths.unsqueeze(1)
        return ids, attended.to(input_ids.dtype)

    return vary
