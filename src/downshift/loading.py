"""Turn a CLI model argument into something the export layer can use.

downshift serves models that are already downloaded onto this machine, and only those.
Three kinds of artifact on disk are accepted:

  model.onnx                a downloaded or already-exported ONNX file, served as-is
                            (UNVERIFIED without --reference)
  weights.pt                a downloaded PyTorch checkpoint (state dict); needs
                            --model-class pkg.module:Class. Loaded with weights_only=True;
                            a pickled full module needs --unsafe-load.
                            Also .pth, .bin, .ckpt.
  path/to/repo/dir          a downloaded Hugging Face repo: a directory with a config.json
                            in it, which is how one is recognised (needs the [hf] extra)

plus one form that names a model already importable in this process rather than a file:

  pkg.module:attr           import spec; attr is an nn.Module instance or a zero-arg factory.
                            A sibling `make_inputs` in the same module is picked up automatically.

Nothing here reaches the network. Hugging Face hub ids are not accepted: download the repo
yourself (huggingface-cli download, git clone) and pass the directory it landed in.
"""

import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Import-spec resolution and its error type are torch-free leaf helpers; re-exported here so
# existing `from downshift.loading import import_object, LoadError` keeps working.
from downshift._imports import (  # noqa: F401
    LoadError,
    import_object,
    is_import_spec,
)
from downshift.adapters.base import Family

# What a model argument turned out to be; defined torch-free in downshift.sources and
# re-exported here, where the kind is decided.
from downshift.sources import (  # noqa: F401
    HF_REPO_DIR,
    IMPORT_SPEC,
    IN_PROCESS_MODULE,
    ONNX_FILE,
    SOURCE_KIND_HELP,
    TORCH_CHECKPOINT,
    UNKNOWN_SOURCE,
)

_STATE_DICT_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt"}
# The shape of a hub id -- "bert-base-uncased" or "owner/name", never a path: a spec that
# looks like one but doesn't exist earns a pointed error rather than a bare "does not exist".
_HUB_ID = re.compile(r"^[\w-][\w.-]*(/[\w.-]+)?$")


# The one-line summary of what is accepted, reused by every error that has to say it.
ACCEPTED = (
    "downshift only serves models already downloaded onto this machine: a .onnx file, a "
    "PyTorch checkpoint (.pt/.pth/.bin/.ckpt), a Hugging Face repo directory (one holding "
    "a config.json), or an importable package.module:attr"
)


@dataclass(frozen=True)
class LoadSpec:
    """Everything load_model needs to resolve one MODEL argument. pooling and normalize
    only mean something for a Hugging Face encoder repo (see the module docstring)."""

    model: str
    inputs: str | None = None
    model_class: str | None = None
    unsafe_load: bool = False
    pooling: str | None = None
    normalize: bool | None = None


@dataclass
class LoadedModel:
    source: str
    model: nn.Module | None = None
    onnx_path: Path | None = None
    example_inputs: tuple | None = None
    adapter_hint: str | None = None
    # Which accepted form `source` turned out to be; see the module docstring.
    kind: str = UNKNOWN_SOURCE

    @property
    def source_path(self) -> Path | None:
        path = Path(self.source)
        return path if path.exists() else None


def _instantiate(obj: Any) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    if callable(obj):
        model = obj()
        if isinstance(model, nn.Module):
            return model
        raise LoadError(f"{obj!r}() returned {type(model).__name__}, not an nn.Module")
    raise LoadError(f"{obj!r} is neither an nn.Module nor a callable that builds one")


def load_inputs(spec: str) -> tuple:
    inputs = import_object(spec)
    if callable(inputs):
        inputs = inputs()
    return inputs if isinstance(inputs, tuple) else (inputs,)


def _load_from_import_spec(spec: str, inputs_spec: str | None) -> LoadedModel:
    model = _instantiate(import_object(spec))
    inputs: tuple | None = None
    if inputs_spec:
        inputs = load_inputs(inputs_spec)
    else:
        module_name = spec.partition(":")[0]
        if callable(getattr(import_module(module_name), "make_inputs", None)):
            inputs = load_inputs(f"{module_name}:make_inputs")
    return LoadedModel(source=spec, model=model, example_inputs=inputs, kind=IMPORT_SPEC)


def _load_checkpoint(path: Path, model_class: str | None, unsafe_load: bool) -> nn.Module:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=not unsafe_load)
    except Exception as exc:  # torch raises a few different types here
        if unsafe_load:
            raise LoadError(f"failed to load {path}: {exc}") from exc
        raise LoadError(
            f"{path} isn't loadable with weights_only=True ({type(exc).__name__}). If it "
            "holds a pickled nn.Module from a source you trust, re-run with --unsafe-load."
        ) from exc

    if isinstance(payload, nn.Module):
        return payload
    if isinstance(payload, dict):
        state = payload.get("state_dict", payload)
        if model_class is None:
            raise LoadError(
                f"{path} is a state dict; pass --model-class package.module:Class so it can "
                "be instantiated and the weights loaded into it."
            )
        model = _instantiate(import_object(model_class))
        model.load_state_dict(state)
        return model
    raise LoadError(f"{path} contained {type(payload).__name__}, expected a state dict")


def load_model(spec: LoadSpec) -> LoadedModel:
    path = Path(spec.model)
    if path.suffix == ".onnx":
        if not path.exists():
            raise LoadError(f"{path} is not on this machine; download or export it first")
        return LoadedModel(source=spec.model, onnx_path=path, kind=ONNX_FILE)

    if path.exists() and path.suffix in _STATE_DICT_SUFFIXES:
        model = _load_checkpoint(path, spec.model_class, spec.unsafe_load)
        return LoadedModel(
            source=spec.model,
            model=model,
            example_inputs=load_inputs(spec.inputs) if spec.inputs else None,
            kind=TORCH_CHECKPOINT,
        )

    if is_import_spec(spec.model):
        return _load_from_import_spec(spec.model, spec.inputs)

    if path.is_dir():
        if not (path / "config.json").exists():
            raise LoadError(
                f"{path} has no config.json, so it is not a downloaded Hugging Face repo. "
                "A directory is only accepted as a model when it holds the repo's config.json."
            )
        return _load_hf(spec.model, spec.inputs, spec.pooling, spec.normalize)

    if path.exists():
        raise LoadError(f"don't know how to load {path} (suffix {path.suffix!r}). {ACCEPTED}.")

    if not path.suffix and _HUB_ID.match(spec.model):
        raise LoadError(
            f"'{spec.model}' directory is not on this machine. A Hugging Face hub id is not accepted -- "
            f"downshift only serves models that are already downloaded. Download the repo "
            f"first (huggingface-cli download {spec.model} --local-dir "
            f"./{spec.model.rpartition('/')[2]}) and pass that directory, which must hold a "
            f"config.json."
        )
    raise LoadError(f"{path} is not on this machine. {ACCEPTED}.")


def resolve_tokenizer_source(spec: str) -> str:
    """Check --tokenizer-from names a downloaded Hugging Face repo directory and return it as
    a string; the repo itself is read later, by serve.engine.attach_hf_metadata."""
    path = Path(spec)
    if not (path / "config.json").is_file():
        raise LoadError(
            f"--tokenizer-from {path} is not a downloaded Hugging Face repo directory (one "
            "holding a config.json); same rule as a Hugging Face repo passed as MODEL."
        )
    return str(path)


def _load_hf(
    spec: str, inputs: str | None, pooling: str | None, normalize: bool | None
) -> LoadedModel:
    try:
        from downshift import hf_repo
        from downshift.adapters.embedding import RecipeError
    except ImportError as exc:
        raise LoadError(
            f"loading {spec!r} as a Hugging Face repo directory needs the [hf] extra: "
            "pip install 'downshift-server[hf]'"
        ) from exc
    try:
        model = hf_repo.load_pretrained(spec, pooling, normalize)
    except (
        RecipeError,
        hf_repo.RemoteCodeError,
        hf_repo.TextGenerationModelError,
    ) as exc:  # about the flags, the recipe or what the repo is, not the download
        raise LoadError(f"{spec}: {exc}") from exc
    except (OSError, ValueError) as exc:  # a bad or incomplete download surfaces as either
        raise LoadError(
            f"can't load {spec!r} as a downloaded Hugging Face repo: {exc}. Nothing is "
            "fetched to fill a gap, so an incomplete download fails here rather than later."
        ) from exc
    return LoadedModel(
        source=spec,
        model=model,
        example_inputs=load_inputs(inputs) if inputs else None,
        adapter_hint=Family.hf,
        kind=HF_REPO_DIR,
    )
