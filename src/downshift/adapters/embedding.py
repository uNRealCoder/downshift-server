"""Sentence embeddings from an encoder-only repo: the recipe, and the graph that applies it.

`config.json` says nothing about pooling; a sentence-transformers repo keeps its recipe in
sibling files: `modules.json` (Transformer -> Pooling -> optional Normalize),
`1_Pooling/config.json` (which pooling) and `sentence_bert_config.json` (the sequence length
the model was trained at, often shorter than its position embeddings). This module reads
those, and PoolingHead applies the result inside the exported graph, so verification
compares the finished embedding and ONNX Runtime runs the pooling too.

Anything the recipe asks for that is not applied here (a Dense module, several poolings
concatenated) is refused rather than approximated: a model served
with the wrong pooling still returns plausible vectors, just worse ones.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from downshift.adapters.pooling import PoolingChoice

# Set on the loaded model by hf_repo.load_pretrained; the hf adapter reads it in prepare().
EMBEDDING_ATTR = "downshift_embedding"

_POOLING_FLAGS = {
    "pooling_mode_cls_token": PoolingChoice.cls,
    "pooling_mode_mean_tokens": PoolingChoice.mean,
    "pooling_mode_max_tokens": PoolingChoice.maximum,
    "pooling_mode_mean_sqrt_len_tokens": PoolingChoice.mean_sqrt_len,
    "pooling_mode_lasttoken": PoolingChoice.lasttoken,
    "pooling_mode_weightedmean_tokens": PoolingChoice.weightedmean,
}
_MASKED_OUT = -1e9  # what sentence-transformers fills padding with before a max


class RecipeError(ValueError):
    """The embedding recipe can't be served: the repo declares one downshift does not apply,
    or --pooling/--normalize contradict what the repo is."""


@dataclass(frozen=True)
class EmbeddingRecipe:
    pooling: str
    normalize: bool
    max_seq_length: int | None  # what the model was trained at; None when the repo does not say
    origin: str  # "modules.json", or which flag overrode it

    def describe(self) -> str:
        return f"{self.pooling} pooling" + (", L2-normalised" if self.normalize else "")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RecipeError(f"can't read {path.name}: {exc}") from exc


def _max_seq_length(*dirs: Path) -> int | None:
    for directory in dirs:
        config = _read_json(directory / "sentence_bert_config.json")
        if isinstance(config, dict) and isinstance(config.get("max_seq_length"), int):
            return int(config["max_seq_length"])
    return None


def _pooling_mode(directory: Path) -> str:
    config = _read_json(directory / "config.json")
    if not isinstance(config, dict):
        raise RecipeError(f"modules.json names a Pooling module but {directory} has no config.json")
    modes = [mode for flag, mode in _POOLING_FLAGS.items() if config.get(flag)]
    if len(modes) != 1:
        raise RecipeError(
            f"{directory.name}/config.json enables {len(modes)} pooling modes; "
            "downshift applies exactly one"
        )
    return str(modes[0])


def read_recipe(path: Path) -> EmbeddingRecipe | None:
    """The recipe a sentence-transformers repo declares, or None when it has no modules.json.

    Raises RecipeError for one this module cannot apply faithfully.
    """
    modules = _read_json(path / "modules.json")
    if modules is None:
        return None
    if not isinstance(modules, list):
        raise RecipeError("modules.json is not a list of modules")
    pooling: str | None = None
    normalize = False
    transformer_dir = path
    for module in modules:
        kind = str(module.get("type", ""))
        short = kind.rsplit(".", 1)[-1]
        if short == "Transformer":
            transformer_dir = path / str(module.get("path", ""))
        elif short == "Pooling" and pooling is None:
            pooling = _pooling_mode(path / str(module.get("path", "")))
        elif short == "Normalize" and pooling is not None:
            normalize = True
        else:
            raise RecipeError(
                f"modules.json lists {kind or 'a module with no type'}, which downshift does not "
                "apply, so the embedding would not match the reference. Pass --pooling none to "
                "serve the encoder's token-level output instead."
            )
    if pooling is None:
        raise RecipeError("modules.json has no Pooling module")
    return EmbeddingRecipe(
        pooling, normalize, _max_seq_length(transformer_dir, path), "modules.json"
    )


def resolve_recipe(
    path: str | Path, pooling: str | None, normalize: bool | None, *, has_head: bool
) -> EmbeddingRecipe | None:
    """What to serve: the repo's own recipe, changed by --pooling/--normalize when given.

    `pooling` "none" means the encoder's token-level output, ignoring whatever the repo
    declares. A repo with a classification head has no embedding to pool, so a recipe is only
    ever an explicit mistake there.
    """
    if pooling == PoolingChoice.none:
        if normalize is not None:
            raise RecipeError(
                "--normalize needs a pooling; --pooling none serves the raw token vectors"
            )
        return None
    if has_head:
        if pooling is not None or normalize is not None:
            raise RecipeError(
                "--pooling/--normalize apply to encoder-only repos; "
                "this one has a classification head"
            )
        return None
    root = Path(path)
    detected = read_recipe(root)
    if pooling is None and normalize is None:
        return detected
    if pooling is None:
        if detected is None:
            raise RecipeError(
                "--normalize/--no-normalize needs a pooling and this repo declares none; "
                "also pass --pooling"
            )
        return replace(
            detected, normalize=bool(normalize), origin=f"{detected.origin}, --normalize changed"
        )
    base = detected or EmbeddingRecipe(pooling, False, _max_seq_length(root), "--pooling")
    return replace(
        base,
        pooling=str(pooling),
        normalize=base.normalize if normalize is None else normalize,
        origin="--pooling" if detected is None else "modules.json, --pooling changed",
    )


def pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """[batch, seq, hidden] token vectors -> [batch, hidden], padding excluded."""
    if mode == PoolingChoice.cls:
        return hidden[:, 0]
    if mode == PoolingChoice.lasttoken:
        # the last attended position, so left and right padding both work
        positions = torch.arange(mask.shape[1], device=mask.device)
        last = (mask.to(positions.dtype) * positions).argmax(dim=1)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    keep = mask.unsqueeze(-1).to(hidden.dtype)
    if mode == PoolingChoice.maximum:
        return hidden.masked_fill(keep == 0, _MASKED_OUT).max(dim=1).values
    if mode == PoolingChoice.weightedmean:
        # later tokens weigh more: position 1..S, as sentence-transformers does
        weights = torch.arange(1, hidden.shape[1] + 1, device=hidden.device).to(hidden.dtype)
        weights = weights.unsqueeze(0).unsqueeze(-1) * keep
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
    total = (hidden * keep).sum(dim=1)
    count = keep.sum(dim=1).clamp(min=1e-9)
    return total / count if mode == PoolingChoice.mean else total / count.sqrt()


class PoolingHead(nn.Module):
    def __init__(self, recipe: EmbeddingRecipe) -> None:
        super().__init__()
        self.mode = recipe.pooling
        self.normalize = recipe.normalize

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = pool(hidden, mask, self.mode)
        return F.normalize(pooled, p=2.0, dim=1) if self.normalize else pooled
