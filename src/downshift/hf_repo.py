"""This module reads a Hugging Face repo directory that is already downloaded to this machine.
It reads the model (with its task head, if the repo has one), the pooling recipe, and the
tokenizer and label metadata that let /predict take text. It never talks to the hub.

The hf adapter exports the loaded model (adapters/hf.py). This module only reads the repo.
Downshift imports it only if transformers is installed.
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

# Task heads whose first output is the logits. A repo that has one loads with it and not as the
# bare encoder. (AutoModel drops the weights of the head without a message.) The key is the
# suffix of the class name in config.architectures.
_TASK_HEADS = {
    "ForSequenceClassification": AutoModelForSequenceClassification,
    "ForTokenClassification": AutoModelForTokenClassification,
}
# Class-name suffixes of models that generate text. Downshift refuses them if they have no embedding recipe.
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

# Families whose position ids start at pad_token_id + 1 (the padding_idx offset). The position
# table therefore has that many fewer usable positions than max_position_embeddings says.
_PADDING_OFFSET_TYPES = frozenset({"roberta", "xlm-roberta", "xlm-roberta-xl", "camembert"})
_DEFAULT_PAD_TOKEN_ID = 1  # the default of RobertaConfig itself, for a config that does not set it
_MIN_USABLE_POSITIONS = 2  # torch.export.Dim needs max > min (min is 1)


def position_limit(config: PretrainedConfig) -> int | None:
    """The longest sequence that the position embeddings can address. None if the config
    declares no max_position_embeddings. The RoBERTa family reserves the first
    pad_token_id + 1 slots. For example, 514 positions serve 512 tokens."""
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
    """The top-level object of a JSON file in the repo. {} if the file is missing, cannot be
    read, or is not an object. (The loaders report a broken file themselves.)"""
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
                "Downshift never runs the code of a repo (no trust_remote_code)"
            )


def load_config(path: str) -> PretrainedConfig:
    """The config.json of the repo, read from this machine only (local_files_only). It raises
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
    """How this repo turns token vectors into an embedding. None means that downshift serves
    the token vectors as they are. The recipe is the sentence-transformers recipe in the repo,
    changed by --pooling and --normalize. It raises RecipeError (a ValueError) for a recipe
    that downshift cannot apply faithfully. `config` is the AutoConfig of the repo that is
    already loaded, so a boot reads config.json one time.
    """
    return resolve_recipe(path, pooling, normalize, has_head=_task_head(config) is not None)


def load_pretrained(
    path: str, pooling: str | None = None, normalize: bool | None = None
) -> PreTrainedModel:
    """path is a repo directory that is already downloaded to this machine (config.json and the
    weights).

    local_files_only limits from_pretrained to that directory. Downshift resolves no hub id,
    and no request leaves the process. A missing or half-downloaded file therefore fails here
    with a clear error. Downshift does not fetch it without a message.

    If config.json of the repo declares a sequence-classification or token-classification
    architecture, downshift loads the repo with that head. The served output is then its logits.
    Any other repo loads as the bare encoder. If an encoder-only repo declares a pooling recipe,
    downshift serves it pooled. The recipe stays on the model, and the hf adapter puts it into
    the exported graph.
    """
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
    # A safetensors file has plain tensors. Without this, transformers can use a pickled
    # pytorch_model.bin file next to it.
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
    """The backbone of a text-generating repo, loaded to embed. Downshift does not read its
    lm_head weight, by design. The own load report of transformers lists it as UNEXPECTED, so
    downshift silences the report for this call. It logs anything else that the report says
    again at warning."""
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
    """The tokenizer and label metadata that let /predict take text. None if the repo has no
    tokenizer files. The local-only rule and `config` are the same as for embedding_recipe.

    The longest text that downshift accepts is the smallest of the limits that the files of
    the model declare: the limit of the tokenizer, the limit of the position embeddings, and the
    length that the author used for training (`recipe.max_seq_length`). Downshift does not
    choose any of them.
    """
    if not any((Path(path) / name).is_file() for name in _VOCAB_FILES):
        # AutoTokenizer does not raise an error here. It builds an empty tokenizer that maps
        # each word to [UNK]. A classifier that receives this gives a confident answer about
        # nothing.
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
