"""Reading a Hugging Face repo directory already downloaded onto this machine: the model
(with its task head, if it ships one), its pooling recipe, and the tokenizer/label metadata
that lets /predict take text. Nothing here talks to the hub.

Exporting the loaded model is the hf adapter's job (adapters/hf.py); this module only reads
the repo. Only imported when transformers is installed.
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

# Task heads whose first output is the logits: a repo that ships one is loaded with it, not
# as the bare encoder (AutoModel silently drops the head's weights). Keyed by the suffix of
# the class name in config.architectures.
_TASK_HEADS = {
    "ForSequenceClassification": AutoModelForSequenceClassification,
    "ForTokenClassification": AutoModelForTokenClassification,
}
# Class-name suffixes of models that generate text; without an embedding recipe they are refused.
_TEXT_GENERATING_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")
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


class RemoteCodeError(ValueError):
    """The repo's config or tokenizer config names code to download and run (auto_map)."""


class TextGenerationModelError(ValueError):
    """The repo is a text generator with no embedding recipe; downshift serves embedders."""


def _json_dict(path: Path) -> dict:
    """A repo JSON file's top-level object; {} when it is missing, unreadable or not an object
    (the loaders report a broken file themselves)."""
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
                "downshift never runs a repo's own code (no trust_remote_code)"
            )


def load_config(path: str) -> PretrainedConfig:
    """The repo's config.json, read from this machine only (local_files_only). Raises
    RemoteCodeError for a repo that names its own code."""
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
    """How this repo turns token vectors into an embedding, or None to serve them as they
    are: the sentence-transformers recipe in the repo, changed by --pooling/--normalize.
    Raises RecipeError (a ValueError) for a recipe that cannot be applied faithfully.
    `config` is the repo's already-loaded AutoConfig, so a boot reads config.json once.
    """
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
    recipe is left on the model for the hf adapter to put into the exported graph.
    """
    config = load_config(path)
    recipe = embedding_recipe(path, config, pooling, normalize)  # before the weights: fail fast
    generator = _text_generator(config)
    if generator is not None and (recipe is None or pooling == PoolingChoice.none):
        raise TextGenerationModelError(
            f"{generator}: this model generates text; downshift serves text embedding models; "
            "pass --pooling lasttoken if this checkpoint is an embedder"
        )
    head = _task_head(config)
    auto_class = _TASK_HEADS[head] if head is not None else AutoModel
    # A safetensors file is plain tensors; without this, transformers may fall back to a
    # pickled pytorch_model.bin next to it.
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
    """The backbone of a text-generating repo, loaded to embed. Its lm_head weight is left
    unread on purpose, so transformers' own load report (which lists it as UNEXPECTED) is
    silenced for this call and anything else it says is re-logged at warning."""
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
    """The tokenizer and label metadata that let /predict take text, or None when the repo
    has no tokenizer files. Same local-only rule and `config` as embedding_recipe.

    The longest text accepted is the smallest of the limits the model's own files declare:
    the tokenizer's, the position embeddings', and the length the author trained at
    (`recipe.max_seq_length`). None of them is downshift's to pick.
    """
    if not any((Path(path) / name).is_file() for name in _VOCAB_FILES):
        # AutoTokenizer does not raise here: it builds an empty tokenizer that maps every
        # word to [UNK], and a classifier fed that answers confidently about nothing.
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
