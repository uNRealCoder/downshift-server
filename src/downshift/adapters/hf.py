"""The Hugging Face adapter, for encoder-only models.

It builds input_ids and attention_mask directly from the model config. It does not import
optimum. The export shim unwraps the ModelOutput, so torch.export sees a plain tensor. The
tensor is last_hidden_state for base models. For a sequence-classification or
token-classification head, it is the logits. hf_repo.load_pretrained selects this from
config.architectures.

Downshift imports it only if transformers is installed.
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
    """The first output of the model. If the repo has a recipe, it is pooled into one embedding
    for each text (see adapters/embedding.py). The pooling is part of the graph that downshift
    exports."""

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
        # Use getattr and not model.config. The typeshed of nn.Module makes attribute access
        # resolve to Tensor | Module. That loses the real PretrainedConfig type. getattr(..., str)
        # keeps it as Any.
        config = getattr(model, "config")  # noqa: B009
        # The position embeddings limit the sequence length. A looser bound trips the guards of the export.
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
    """The verification samples after the first one. They vary the batch size and the sequence
    length, and they pad. Each row gets its own random length in [1, s]. The mask is zero
    beyond that length (and at least one position is attended). Padding is then tested for real
    and not only with masks that are always full. With `padding_side` "left", the zeros are at
    the start of each row. The tokenizer of a decoder embedder pads in this way at serve time.

    `longest` is the longest sequence that the model itself declares (its position
    embeddings). If it is set, sample 1 is one row of full length, at exactly that length. A
    graph that diverges only at long sequences then cannot pass with short samples alone. The
    other sizes stay near the example and cannot guarantee that.

    `axis_max` is --axis-max. It limits the sampled sizes. A pinned `seq` (or `batch`) puts
    sample 1 exactly at that size instead. It then takes the slot of the full-length sample.
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
