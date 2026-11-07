"""Read a downloaded Hugging Face repo directory, never the hub: the model (with its task
head), the pooling recipe, and the tokenizer and labels for text input. Needs transformers.
"""

import json
import logging
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from downshift.adapters.embedding import (
    EMBEDDING_ATTR,
    PADDING_SIDE_ATTR,
    EmbeddingRecipe,
    resolve_recipe,
)
from downshift.adapters.pooling import PoolingChoice
from downshift.adapters.text import SIGMOID, SOFTMAX, TextIO

logger = logging.getLogger("downshift.hf_repo")

# Task heads (by config.architectures suffix). AutoModel would silently drop the head weights.
_TASK_HEADS = {
    "ForSequenceClassification": AutoModelForSequenceClassification,
    "ForTokenClassification": AutoModelForTokenClassification,
}
# Text generators, refused unless they have an embedding recipe.
_TEXT_GENERATING_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")
_UNBOUNDED_LENGTH = 1 << 20  # tokenizers report "no limit" as a very large value
# A tokenizer is real only if one of these files is next to config.json.
_VOCAB_FILES = (
    "tokenizer.json",
    "vocab.txt",
    "vocab.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)

# Families whose position ids start at pad_token_id + 1, so fewer positions are usable.
_PADDING_OFFSET_TYPES = frozenset({"roberta", "xlm-roberta", "xlm-roberta-xl", "camembert"})
_DEFAULT_PAD_TOKEN_ID = 1  # the default of RobertaConfig itself, for a config that does not set it
_MIN_USABLE_POSITIONS = 2  # torch.export.Dim needs max > min (min is 1)


def position_limit(config: PretrainedConfig) -> int | None:
    """The longest sequence the position embeddings address, or None. RoBERTa reserves
    pad_token_id + 1 slots: 514 positions serve 512 tokens."""
    declared = getattr(config, "max_position_embeddings", None)
    if declared is None:
        return None
    usable = int(declared)
    if getattr(config, "model_type", None) in _PADDING_OFFSET_TYPES:
        pad_token_id = getattr(config, "pad_token_id", None)
        usable -= (_DEFAULT_PAD_TOKEN_ID if pad_token_id is None else int(pad_token_id)) + 1
    return max(usable, _MIN_USABLE_POSITIONS)


class RemoteCodeError(ValueError):
    """The repo's config or tokenizer config names code to download and run (auto_map)."""


class TextGenerationModelError(ValueError):
    """The repo is a text generator with no embedding recipe; downshift serves embedders."""


def _json_dict(path: Path) -> dict:
    """A JSON object file of the repo, or {} (the loaders report a broken file themselves)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _refuse_remote_code(root: Path) -> None:
    for name in ("config.json", "tokenizer_config.json"):
        if _json_dict(root / name).get("auto_map"):
            raise RemoteCodeError(
                f"{name} has an auto_map, so loading it would run Python code from the repo. "
                "Downshift never runs the code of a repo (no trust_remote_code). Export the model "
                "to ONNX yourself and serve the .onnx file, or use a repo with no auto_map"
            )


def load_config(path: str) -> PretrainedConfig:
    """config.json, local files only. RemoteCodeError if the repo names its own code."""
    _refuse_remote_code(Path(path))
    config: PretrainedConfig = AutoConfig.from_pretrained(path, local_files_only=True)
    return config


def _text_generator(config: PretrainedConfig) -> str | None:
    for architecture in getattr(config, "architectures", None) or ():
        if str(architecture).endswith(_TEXT_GENERATING_SUFFIXES):
            return str(architecture)
    return None


def _task_head(config: PretrainedConfig) -> str | None:
    for architecture in getattr(config, "architectures", None) or ():
        for suffix in _TASK_HEADS:
            if architecture.endswith(suffix):
                return suffix
    return None


def embedding_recipe(
    path: str,
    config: PretrainedConfig,
    pooling: str | None = None,
    normalize: bool | None = None,
) -> EmbeddingRecipe | None:
    """The pooling recipe to serve (None: token vectors). See embedding.resolve_recipe."""
    return resolve_recipe(path, pooling, normalize, has_head=_task_head(config) is not None)


def load_pretrained(
    path: str, pooling: str | None = None, normalize: bool | None = None
) -> PreTrainedModel:
    """The model of a downloaded repo directory, local files only (a missing file fails here).
    A declared classification head loads with it; other repos load as the bare encoder. The
    recipe goes on the model for the hf adapter."""
    config = load_config(path)
    recipe = embedding_recipe(path, config, pooling, normalize)  # before the weights, to fail fast
    generator = _text_generator(config)
    if generator is not None and (recipe is None or pooling == PoolingChoice.none):
        raise TextGenerationModelError(
            f"{generator}: this model generates text; downshift serves text embedding models; "
            "pass --pooling lasttoken if this checkpoint is an embedder"
        )
    head = _task_head(config)
    auto_class = _TASK_HEADS[head] if head is not None else AutoModel
    # Prefer safetensors over a pickled pytorch_model.bin next to it.
    safetensors = any(Path(path).glob("*.safetensors"))
    kwargs = {"use_safetensors": True} if safetensors else {}
    model: PreTrainedModel
    if generator is None:
        model = auto_class.from_pretrained(path, local_files_only=True, **kwargs)
    else:
        model = _load_backbone(path, **kwargs)
    setattr(model, EMBEDDING_ATTR, recipe)
    setattr(model, PADDING_SIDE_ATTR, _padding_side(Path(path)))
    return model


def _load_backbone(path: str, **kwargs: bool) -> PreTrainedModel:
    """The backbone of a text-generating repo, without lm_head. The transformers load report
    would flag lm_head, so it is silenced and its other keys logged again."""
    hf_logger = logging.getLogger("transformers")
    previous = hf_logger.level
    hf_logger.setLevel(logging.ERROR)
    try:
        model, info = AutoModel.from_pretrained(
            path, local_files_only=True, output_loading_info=True, **kwargs
        )
    finally:
        hf_logger.setLevel(previous)
    for kind in ("unexpected_keys", "missing_keys"):
        keys = sorted(k for k in info.get(kind, ()) if not k.startswith("lm_head."))
        if keys:
            logger.warning("%s: %s %s", path, kind.replace("_", " "), keys)
    model.config.use_cache = False  # no KV cache in the exported graph
    loaded: PreTrainedModel = model
    return loaded


def _padding_side(root: Path) -> str:
    """The side the repo's tokenizer pads on; a decoder embedder usually pads left."""
    side = _json_dict(root / "tokenizer_config.json").get("padding_side")
    return "left" if side == "left" else "right"


def load_text_io(
    path: str, config: PretrainedConfig, recipe: EmbeddingRecipe | None = None
) -> TextIO | None:
    """The tokenizer and labels for text input, or None without tokenizer files. The length
    limit is the smallest that the model files declare: tokenizer, positions, or training."""
    if not any((Path(path) / name).is_file() for name in _VOCAB_FILES):
        # AutoTokenizer would build an empty tokenizer that maps every word to [UNK].
        return None
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
    labels = getattr(config, "id2label", None) or {}
    if _task_head(config) == "ForSequenceClassification" and len(labels) > 1:
        id2label = {int(i): str(label) for i, label in labels.items()}
        multi_label = getattr(config, "problem_type", None) == "multi_label_classification"
        activation = SIGMOID if multi_label else SOFTMAX
    return TextIO(
        tokenizer=tokenizer,
        max_length=min(limits, default=_UNBOUNDED_LENGTH),
        id2label=id2label,
        activation=activation,
    )
