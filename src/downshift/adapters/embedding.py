"""Sentence embeddings from an encoder-only repo: the recipe, and the graph that applies it.

`config.json` has no information about pooling. A sentence-transformers repo keeps its recipe
in sibling files:

- `modules.json` (Transformer, then Pooling, then an optional Normalize).
- `1_Pooling/config.json` (the kind of pooling).
- `sentence_bert_config.json` (the sequence length that the model used for training. It is
  often shorter than its position embeddings).

This module reads these files. PoolingHead applies the result inside the exported graph.
Verification therefore compares the finished embedding, and ONNX Runtime also runs the pooling.

Downshift refuses a recipe that asks for something that this module does not apply (a Dense
module, or several poolings that are concatenated). It does not approximate. A model that is
served with the wrong pooling still returns vectors that look correct. They are only worse.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from downshift.adapters.pooling import PoolingChoice

# hf_repo.load_pretrained sets it on the loaded model. The hf adapter reads it in prepare().
EMBEDDING_ATTR = "downshift_embedding"
# The padding side of the tokenizer ("left" or "right"). Downshift sets it next to
# EMBEDDING_ATTR. The verification sampler pads in the same way as the tokenizer at serve time.
PADDING_SIDE_ATTR = "downshift_padding_side"

_POOLING_FLAGS = {
    "pooling_mode_cls_token": PoolingChoice.cls,
    "pooling_mode_mean_tokens": PoolingChoice.mean,
    "pooling_mode_max_tokens": PoolingChoice.maximum,
    "pooling_mode_mean_sqrt_len_tokens": PoolingChoice.mean_sqrt_len,
    "pooling_mode_lasttoken": PoolingChoice.lasttoken,
    "pooling_mode_weightedmean_tokens": PoolingChoice.weightedmean,
}
_MASKED_OUT = -1e9  # the value that sentence-transformers puts in padding before a max


class RecipeError(ValueError):
    """Downshift cannot serve the embedding recipe. The repo declares a recipe that downshift
    does not apply, or --pooling and --normalize contradict the kind of the repo."""


@dataclass(frozen=True)
class EmbeddingRecipe:
    pooling: str
    normalize: bool
    max_seq_length: int | None  # the training length of the model. None if the repo does not say
    origin: str  # "modules.json", or the flag that overrode it
    prompts: dict[str, str] = field(default_factory=dict)  # name -> text for the start of a row
    default_prompt: str | None = None  # the name that downshift applies if a request names none

    def describe(self) -> str:
        return f"{self.pooling} pooling" + (", L2-normalised" if self.normalize else "")

    def info(self, dimension: int | None) -> dict[str, Any]:
        """The recipe as `embedding` of /schema and the safetensors metadata `downshift.embedding`
        report it (without the prompts), for an output of `dimension` values for each row."""
        return {
            "pooling": self.pooling,
            "normalized": self.normalize,
            "dimension": dimension,
            "max_seq_length": self.max_seq_length,
            "from": self.origin,
        }


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


def _prompts(root: Path) -> tuple[dict[str, str], str | None]:
    """The named prompts and default_prompt_name from config_sentence_transformers.json."""
    config = _read_json(root / "config_sentence_transformers.json")
    if not isinstance(config, dict):
        return {}, None
    raw = config.get("prompts")
    prompts = (
        {str(k): v for k, v in raw.items() if isinstance(v, str)} if isinstance(raw, dict) else {}
    )
    default = config.get("default_prompt_name")
    return prompts, default if isinstance(default, str) and default in prompts else None


def _pooling_mode(directory: Path) -> str:
    config = _read_json(directory / "config.json")
    if not isinstance(config, dict):
        raise RecipeError(f"modules.json names a Pooling module but {directory} has no config.json")
    modes = [mode for flag, mode in _POOLING_FLAGS.items() if config.get(flag)]
    if len(modes) != 1:
        raise RecipeError(
            f"{directory.name}/config.json enables {len(modes)} pooling modes. "
            "Downshift applies exactly one"
        )
    return str(modes[0])


def read_recipe(path: Path) -> EmbeddingRecipe | None:
    """The recipe that a sentence-transformers repo declares. None if it has no modules.json.

    It raises RecipeError for a recipe that this module cannot apply faithfully.
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
    prompts, default_prompt = _prompts(path)
    return EmbeddingRecipe(
        pooling,
        normalize,
        _max_seq_length(transformer_dir, path),
        "modules.json",
        prompts,
        default_prompt,
    )


def resolve_recipe(
    path: str | Path, pooling: str | None, normalize: bool | None, *, has_head: bool
) -> EmbeddingRecipe | None:
    """What to serve: the recipe of the repo itself, changed by --pooling and --normalize if you
    give them.

    `pooling` "none" means the output of the encoder at token level. It ignores what the repo
    declares. A repo with a classification head has no embedding to pool. A recipe there can
    only be an explicit mistake.
    """
    if pooling == PoolingChoice.none:
        if normalize is not None:
            raise RecipeError(
                "--normalize needs a pooling. --pooling none serves the raw token vectors"
            )
        return None
    if has_head:
        if pooling is not None or normalize is not None:
            raise RecipeError(
                "--pooling/--normalize apply to encoder-only repos. "
                "This repo has a classification head"
            )
        return None
    root = Path(path)
    detected = read_recipe(root)
    if pooling is None and normalize is None:
        return detected
    if pooling is None:
        if detected is None:
            raise RecipeError(
                "--normalize/--no-normalize needs a pooling, and this repo declares none. "
                "Also pass --pooling"
            )
        return replace(
            detected, normalize=bool(normalize), origin=f"{detected.origin}, --normalize changed"
        )
    prompts, default_prompt = _prompts(root)
    base = detected or EmbeddingRecipe(
        pooling, False, _max_seq_length(root), "--pooling", prompts, default_prompt
    )
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
        # the last attended position. Left padding and right padding then both work
        positions = torch.arange(mask.shape[1], device=mask.device)
        last = (mask.to(positions.dtype) * positions).argmax(dim=1)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    keep = mask.unsqueeze(-1).to(hidden.dtype)
    if mode == PoolingChoice.maximum:
        return hidden.masked_fill(keep == 0, _MASKED_OUT).max(dim=1).values
    if mode == PoolingChoice.weightedmean:
        # later tokens have more weight: position 1..S, as sentence-transformers does
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
        # float32 on the wire, for all dtypes of the checkpoint. It does nothing for float32 models
        pooled = pool(hidden, mask, self.mode).float()
        return F.normalize(pooled, p=2.0, dim=1) if self.normalize else pooled
