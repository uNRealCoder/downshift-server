"""Hugging Face adapter, encoder-only models.

Loads from a repo directory already downloaded onto this machine, recognised by the
config.json in it; nothing here talks to the hub.

Builds input_ids / attention_mask straight from the model config rather than pulling in
optimum. The export shim unwraps the ModelOutput so torch.export sees a plain tensor
(last_hidden_state for base models, logits for a sequence- or token-classification head,
which load_pretrained picks from config.architectures).

Only imported when transformers is installed.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from downshift.adapters.base import Prepared, VaryFn
from downshift.adapters.embedding import (
    EMBEDDING_ATTR,
    EmbeddingRecipe,
    PoolingHead,
    resolve_recipe,
)
from downshift.adapters.registry import Family
from downshift.adapters.text import SIGMOID, SOFTMAX, TextIO
from downshift.core.shapes import alternative_sizes, pick_size

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


# Task heads whose first output is the logits: a repo that ships one is loaded with it, not
# as the bare encoder (AutoModel silently drops the head's weights). Keyed by the suffix of
# the class name in config.architectures.
_TASK_HEADS = {
    "ForSequenceClassification": AutoModelForSequenceClassification,
    "ForTokenClassification": AutoModelForTokenClassification,
}
_UNBOUNDED_LENGTH = 1 << 20  # tokenizers report "no limit" as a huge sentinel
# A tokenizer is only real if one of these is next to config.json.
_VOCAB_FILES = (
    "tokenizer.json",
    "vocab.txt",
    "vocab.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)


# Families whose position ids start at pad_token_id + 1 (padding_idx offset), so the position
# table holds that many fewer usable positions than max_position_embeddings says.
_PADDING_OFFSET_TYPES = frozenset({"roberta", "xlm-roberta", "xlm-roberta-xl", "camembert"})
_DEFAULT_PAD_TOKEN_ID = 1  # RobertaConfig's own default, for a config that leaves it unset
_MIN_USABLE_POSITIONS = 2  # torch.export.Dim needs max > min (min is 1)


def position_limit(config: PretrainedConfig) -> int | None:
    """The longest sequence the position embeddings can address, or None when the config
    declares no max_position_embeddings. The RoBERTa family reserves the first
    pad_token_id + 1 slots, so 514 positions serve 512 tokens."""
    declared = getattr(config, "max_position_embeddings", None)
    if declared is None:
        return None
    usable = int(declared)
    if getattr(config, "model_type", None) in _PADDING_OFFSET_TYPES:
        pad_token_id = getattr(config, "pad_token_id", None)
        usable -= (_DEFAULT_PAD_TOKEN_ID if pad_token_id is None else int(pad_token_id)) + 1
    return max(usable, _MIN_USABLE_POSITIONS)


def _task_head(config: PretrainedConfig) -> str | None:
    for architecture in getattr(config, "architectures", None) or ():
        for suffix in _TASK_HEADS:
            if architecture.endswith(suffix):
                return suffix
    return None


def embedding_recipe(
    path: str, pooling: str | None = None, normalize: bool | None = None
) -> EmbeddingRecipe | None:
    """How this repo turns token vectors into an embedding, or None to serve them as they
    are: the sentence-transformers recipe in the repo, changed by --pooling/--normalize.
    Raises RecipeError (a ValueError) for a recipe that cannot be applied faithfully.
    """
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    return resolve_recipe(path, pooling, normalize, has_head=_task_head(config) is not None)


def load_pretrained(
    path: str, pooling: str | None = None, normalize: bool | None = None
) -> PreTrainedModel:
    """path is an already-downloaded repo directory on this machine (config.json, weights).

    local_files_only pins from_pretrained to that directory: no hub id is resolved and no
    request leaves the process, so a missing or half-downloaded file fails loudly here
    instead of being silently fetched.

    A repo whose config.json declares a sequence- or token-classification architecture is
    loaded with that head, so the served output is its logits; anything else loads as the
    bare encoder. An encoder-only repo that declares a pooling recipe is served pooled: the
    recipe is left on the model for the adapter to put into the exported graph.
    """
    if not Path(path).is_dir():
        raise ValueError(
            f"{path} is not a directory. A Hugging Face model is only accepted as a repo "
            "directory already downloaded onto this machine; download it first "
            "(huggingface-cli download, git clone) and pass the directory it landed in."
        )
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    head = _task_head(config)
    # Before the weights: fail fast.
    recipe = resolve_recipe(path, pooling, normalize, has_head=head is not None)
    auto_class = _TASK_HEADS[head] if head is not None else AutoModel
    model: PreTrainedModel = auto_class.from_pretrained(path, local_files_only=True)
    setattr(model, EMBEDDING_ATTR, recipe)
    return model


def vocab_size(path: str) -> int | None:
    """The model's vocabulary size, for the server's input_ids range check (B3), or None
    when the config doesn't declare one. Same local-only rule as load_pretrained."""
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    size = getattr(config, "vocab_size", None)
    return int(size) if size is not None else None


def load_text_io(path: str, recipe: EmbeddingRecipe | None = None) -> TextIO | None:
    """The tokenizer and label metadata that let /predict take text, or None when the repo
    has no tokenizer files. Same local-only rule as load_pretrained.

    The longest text accepted is the smallest of the limits the model's own files declare:
    the tokenizer's, the position embeddings', and the length the author trained at
    (`recipe.max_seq_length`). None of them is downshift's to pick.
    """
    if not any((Path(path) / name).is_file() for name in _VOCAB_FILES):
        # AutoTokenizer does not raise here: it builds an empty tokenizer that maps every
        # word to [UNK], and a classifier fed that answers confidently about nothing.
        return None
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except (OSError, ValueError):
        return None
    declared = (
        tokenizer.model_max_length,
        position_limit(config),
        recipe.max_seq_length if recipe is not None else None,
    )
    limits = [
        int(limit) for limit in declared if limit is not None and int(limit) < _UNBOUNDED_LENGTH
    ]
    id2label: dict[int, str] | None = None
    activation: str | None = None
    if _task_head(config) == "ForSequenceClassification" and config.num_labels > 1:
        id2label = {int(i): str(label) for i, label in config.id2label.items()}
        multi_label = getattr(config, "problem_type", None) == "multi_label_classification"
        activation = SIGMOID if multi_label else SOFTMAX
    return TextIO(
        tokenizer=tokenizer,
        max_length=min(limits, default=_UNBOUNDED_LENGTH),
        id2label=id2label,
        activation=activation,
    )


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

    def prepare(self, model: nn.Module, example_inputs: tuple) -> Prepared:
        input_ids, attention_mask = example_inputs
        # getattr, not model.config: nn.Module's typeshed makes attribute access resolve to
        # Tensor | Module, losing the actual PretrainedConfig type getattr(..., str) keeps as Any.
        config = getattr(model, "config")  # noqa: B009
        # Position embeddings cap the sequence length; a looser bound trips export's guards.
        max_seq = position_limit(config) or 1 << 12
        batch = torch.export.Dim("batch", min=1, max=1 << 12)
        seq = torch.export.Dim("seq", min=1, max=max_seq)
        spec = {0: batch, 1: seq}
        vocab = int(config.vocab_size)
        inputs = (input_ids, attention_mask)
        return Prepared(
            model=_FirstOutputShim(model, getattr(model, EMBEDDING_ATTR, None)),
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
        b = pick_size(batch_candidates) if batch_candidates else base_batch
        s = pick_size(seq_candidates) if seq_candidates else base_seq
        ids = torch.randint(0, vocab_size, (b, s), dtype=input_ids.dtype)
        lengths = torch.randint(1, s + 1, (b,))
        mask = (torch.arange(s).unsqueeze(0) < lengths.unsqueeze(1)).to(input_ids.dtype)
        return ids, mask

    return vary


ADAPTER = HFAdapter()
