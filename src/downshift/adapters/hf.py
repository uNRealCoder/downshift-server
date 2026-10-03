"""Hugging Face adapter, encoder-only models.

Builds input_ids / attention_mask straight from the model config rather than pulling in
optimum. The export shim unwraps the ModelOutput so torch.export sees a plain tensor
(last_hidden_state for base models, logits for a sequence- or token-classification head,
which hf_repo.load_pretrained picks from config.architectures).

Only imported when transformers is installed.
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
    """First output of the model, pooled into one embedding per text when the repo has a
    recipe (see adapters/embedding.py); the pooling is part of the graph that gets exported."""

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
    family = "hf-transformers"

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
        # getattr, not model.config: nn.Module's typeshed makes attribute access resolve to
        # Tensor | Module, losing the actual PretrainedConfig type getattr(..., str) keeps as Any.
        config = getattr(model, "config")  # noqa: B009
        # Position embeddings cap the sequence length; a looser bound trips export's guards.
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
            family=self.family,
        )


def make_vary_fn(
    base_inputs: tuple,
    vocab_size: int,
    max_seq: int = 1 << 12,
    longest: int | None = None,
    axis_max: dict[str, int] | None = None,
    padding_side: str = "right",
) -> VaryFn:
    """Verification samples after the first vary batch and sequence length, and pad: each
    row gets its own random length in [1, s] with the mask zeroed beyond it (and at least
    one attended position), so padding is actually exercised rather than always-full masks.
    With `padding_side` "left" the zeros sit at the start of each row instead, as the
    tokenizer of a decoder embedder will pad at serve time.

    `longest` is the longest sequence the model itself declares (its position embeddings).
    When set, sample 1 is one full-length row at exactly that length, so a graph that only
    diverges at long sequences cannot pass on short samples alone. The other sizes stay near
    the example and cannot guarantee that.

    `axis_max` is --axis-max: it caps the sampled sizes, and a pinned `seq` (or `batch`) makes
    sample 1 sit exactly at that size instead, taking over the full-length sample's slot.
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
