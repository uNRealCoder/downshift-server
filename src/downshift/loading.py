"""Turn a CLI model argument into something that the export layer can use.

Downshift serves only models that are already downloaded to this machine. It accepts three
kinds of artifact on disk:

  model.onnx                a downloaded or already exported ONNX file, served as it is
                            (UNVERIFIED without --reference)
  weights.pt                a downloaded PyTorch checkpoint (state dict). It needs
                            --model-class pkg.module:Class. Downshift loads it with
                            weights_only=True. A pickled full module needs --unsafe-load.
                            The extensions .pth, .bin and .ckpt also work.
  path/to/repo/dir          a downloaded Hugging Face repo: a directory with a config.json.
                            This file identifies the repo (needs the [hf] extra)

There is also one form that names a model that this process can already import. It is not a
file:

  pkg.module:attr           an import spec. attr is an nn.Module instance or a factory with no
                            arguments. If a `make_inputs` function is in the same module,
                            downshift uses it automatically.

This module never uses the network. It does not accept Hugging Face hub ids. Download the repo
yourself (huggingface-cli download, git clone) and pass the directory where it is.
"""

import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import nn

# The import-spec resolution and its error type are leaf helpers without torch.
from downshift._imports import (
    LoadError,
    import_object,
    is_import_spec,
)
from downshift.adapters.base import Family

# What a model argument is. It is defined without torch in downshift.sources.
from downshift.sources import (
    HF_REPO_DIR,
    IMPORT_SPEC,
    ONNX_FILE,
    TORCH_CHECKPOINT,
    UNKNOWN_SOURCE,
)

_STATE_DICT_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt"}
# The form of a hub id: "bert-base-uncased" or "owner/name", never a path. A spec that looks
# like one but does not exist gets a clear error and not only "does not exist".
_HUB_ID = re.compile(r"^[\w-][\w.-]*(/[\w.-]+)?$")


# The one-line summary of what downshift accepts. Each error that must say this reuses it.
ACCEPTED = (
    "downshift only serves models already downloaded onto this machine: a .onnx file, a "
    "PyTorch checkpoint (.pt/.pth/.bin/.ckpt), a Hugging Face repo directory (one holding "
    "a config.json), or an importable package.module:attr"
)


@dataclass(frozen=True)
class LoadSpec:
    """Everything that load_model needs to resolve one MODEL argument. pooling and normalize
    have a meaning only for a Hugging Face encoder repo (see the module docstring)."""

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
    # The accepted form that `source` is. See the module docstring.
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
    except Exception as exc:  # torch raises several different types here
        if unsafe_load:
            raise LoadError(f"failed to load {path}: {exc}") from exc
        raise LoadError(
            f"{path} is not loadable with weights_only=True ({type(exc).__name__}). If it "
            "holds a pickled nn.Module from a source that you trust, run again with --unsafe-load."
        ) from exc

    if isinstance(payload, nn.Module):
        return payload
    if isinstance(payload, dict):
        state = payload.get("state_dict", payload)
        if model_class is None:
            raise LoadError(
                f"{path} is a state dict. Pass --model-class package.module:Class. Downshift "
                "then creates the model and loads the weights into it."
            )
        model = _instantiate(import_object(model_class))
        model.load_state_dict(state)
        return model
    raise LoadError(f"{path} contained {type(payload).__name__}, expected a state dict")


def load_model(spec: LoadSpec) -> LoadedModel:
    path = Path(spec.model)
    if path.suffix == ".onnx":
        if not path.exists():
            raise LoadError(f"{path} is not on this machine. Download or export it first")
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
        raise LoadError(f"cannot load {path} (unknown suffix {path.suffix!r}). {ACCEPTED}.")

    if not path.suffix and _HUB_ID.match(spec.model):
        raise LoadError(
            f"'{spec.model}' directory is not on this machine. A Hugging Face hub id is not accepted. "
            "Downshift serves only models that are already downloaded. Download "
            f"the repo first (huggingface-cli download {spec.model} --local-dir "
            f"./{spec.model.rpartition('/')[2]}) and pass that directory. It must have a "
            "config.json."
        )
    raise LoadError(f"{path} is not on this machine. {ACCEPTED}.")


def hf_repo_dir(model: str) -> str | None:
    """`model` itself if load_model reads it as a Hugging Face repo directory (the same checks,
    in the same order, as load_model). Otherwise None. It does not touch the weights."""
    path = Path(model)
    if path.suffix == ".onnx" or (path.exists() and path.suffix in _STATE_DICT_SUFFIXES):
        return None
    if is_import_spec(model):
        return None
    return model if (path / "config.json").is_file() else None


def resolve_tokenizer_source(spec: str) -> str:
    """Check that --tokenizer-from names a downloaded Hugging Face repo directory. Return it as
    a string. serve.engine.attach_hf_metadata reads the repo later."""
    path = Path(spec)
    if not (path / "config.json").is_file():
        raise LoadError(
            f"--tokenizer-from {path} is not a downloaded Hugging Face repo directory (one "
            "holding a config.json). The rule is the same as for a Hugging Face repo that you "
            "pass as MODEL."
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
    ) as exc:  # about the flags, the recipe or the kind of repo, and not about the download
        raise LoadError(f"{spec}: {exc}") from exc
    except (OSError, ValueError) as exc:  # a bad or incomplete download shows as one of them
        raise LoadError(
            f"cannot load {spec!r} as a downloaded Hugging Face repo: {exc}. Downshift "
            "fetches nothing to fill a gap. An incomplete download therefore fails here and "
            "not later."
        ) from exc
    return LoadedModel(
        source=spec,
        model=model,
        example_inputs=load_inputs(inputs) if inputs else None,
        adapter_hint=Family.hf,
        kind=HF_REPO_DIR,
    )
