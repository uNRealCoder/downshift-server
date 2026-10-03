"""Fixtures for a downloaded Hugging Face repo directory: a tiny random-weight BERT with a
hand-built tokenizer, and the sentence-transformers files (modules.json, 1_Pooling/config.json,
sentence_bert_config.json) that make it an embedding model. Written to a tmp path; nothing is
downloaded."""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertModel,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

MAX_POSITIONS = 32
HIDDEN = 16
DECODER_HIDDEN = 32
_WORDS = ["hello", "world", "ignore", "previous", "instructions", "the", "capital"]

ST_MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]
MEAN_POOLING = {
    "word_embedding_dimension": HIDDEN,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
}


def tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({word: 4 + i for i, word in enumerate(_WORDS)})
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def config(**extra) -> BertConfig:
    return BertConfig(
        vocab_size=100,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=MAX_POSITIONS,
        **extra,
    )


def write_encoder_repo(
    path: Path,
    *,
    modules: list[dict] | None = ST_MODULES,
    pooling: dict | None = MEAN_POOLING,
    max_seq_length: int | None = 16,
) -> str:
    """An encoder-only repo. Each keyword drops or changes one sentence-transformers file:
    modules=None writes no modules.json, pooling=None writes no 1_Pooling/config.json."""
    torch.manual_seed(0)
    BertModel(config()).eval().save_pretrained(path)
    tokenizer().save_pretrained(path)
    if modules is not None:
        (path / "modules.json").write_text(json.dumps(modules))
    if pooling is not None:
        (path / "1_Pooling").mkdir(exist_ok=True)
        (path / "1_Pooling" / "config.json").write_text(json.dumps(pooling))
    if max_seq_length is not None:
        (path / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": max_seq_length})
        )
    return str(path)


LASTTOKEN_POOLING = {
    "word_embedding_dimension": DECODER_HIDDEN,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": False,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_lasttoken": True,
}


def write_decoder_repo(
    path: Path,
    *,
    recipe: str | None = None,
    padding_side: str = "left",
    prompts: dict[str, str] | None = None,
    auto_map: bool = False,
) -> str:
    """A tiny random Qwen2ForCausalLM repo (safetensors, lm_head.weight included). recipe is
    a sentence-transformers pooling file set: "lasttoken" writes modules.json with a
    lasttoken Pooling and a Normalize module; None writes none. prompts goes into
    config_sentence_transformers.json; auto_map adds an auto_map to config.json."""
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=DECODER_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        architectures=["Qwen2ForCausalLM"],
    )
    Qwen2ForCausalLM(cfg).eval().save_pretrained(path)
    tokenizer().save_pretrained(path)
    token_config = path / "tokenizer_config.json"
    data = json.loads(token_config.read_text())
    data["padding_side"] = padding_side
    token_config.write_text(json.dumps(data))
    if auto_map:
        config_file = path / "config.json"
        data = json.loads(config_file.read_text())
        data["auto_map"] = {"AutoModel": "modeling_custom.CustomModel"}
        config_file.write_text(json.dumps(data))
    if recipe is not None:
        assert recipe == "lasttoken", recipe
        (path / "modules.json").write_text(json.dumps(ST_MODULES))
        (path / "1_Pooling").mkdir(exist_ok=True)
        (path / "1_Pooling" / "config.json").write_text(json.dumps(LASTTOKEN_POOLING))
    if prompts is not None:
        (path / "config_sentence_transformers.json").write_text(json.dumps({"prompts": prompts}))
    return str(path)
