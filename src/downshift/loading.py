"""Turn a CLI model argument into something the export layer can use.

Accepted forms:
  model.onnx                pre-built ONNX, served as-is (UNVERIFIED without --reference)
  pkg.module:attr           import spec; attr is a module instance or a zero-arg factory.
                            A sibling `make_inputs` in the same module is picked up automatically.
  weights.pt                state dict; needs --model-class pkg.module:Class. Loaded with
                            weights_only=True. A pickled full module needs --unsafe-load.
  org/repo                  Hugging Face hub id (needs the [hf] extra)
"""

import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import nn

_IMPORT_SPEC = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
_STATE_DICT_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt"}


class LoadError(ValueError):
    pass


@dataclass
class LoadedModel:
    source: str
    model: nn.Module | None = None
    onnx_path: Path | None = None
    example_inputs: tuple | None = None
    adapter_hint: str | None = None

    @property
    def source_path(self) -> Path | None:
        path = Path(self.source)
        return path if path.exists() else None


def import_object(spec: str) -> Any:
    if not _IMPORT_SPEC.match(spec):
        raise LoadError(f"{spec!r} is not an import spec of the form package.module:attr")
    module_name, _, attr = spec.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise LoadError(f"can't import {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise LoadError(f"{module_name!r} has no attribute {attr!r}") from exc


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
    return LoadedModel(source=spec, model=model, example_inputs=inputs)


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


def load_model(
    spec: str,
    inputs: str | None = None,
    model_class: str | None = None,
    unsafe_load: bool = False,
) -> LoadedModel:
    path = Path(spec)
    if path.suffix == ".onnx":
        if not path.exists():
            raise LoadError(f"{path} does not exist")
        return LoadedModel(source=spec, onnx_path=path)

    if path.exists() and path.suffix in _STATE_DICT_SUFFIXES:
        model = _load_checkpoint(path, model_class, unsafe_load)
        return LoadedModel(
            source=spec, model=model, example_inputs=load_inputs(inputs) if inputs else None
        )

    if _IMPORT_SPEC.match(spec):
        return _load_from_import_spec(spec, inputs)

    if path.exists():
        raise LoadError(f"don't know how to load {path} (suffix {path.suffix!r})")
    if path.suffix or path.is_absolute():
        raise LoadError(f"{path} does not exist")

    try:
        from downshift.adapters import hf
    except ImportError as exc:
        raise LoadError(
            f"{spec!r} isn't a file or an import spec; treating it as a Hugging Face repo id "
            "needs the [hf] extra: pip install 'downshift-server[hf]'"
        ) from exc
    try:
        model = hf.load_pretrained(spec)
    except (OSError, ValueError) as exc:  # hub errors surface as either
        raise LoadError(f"can't load {spec!r} from the Hugging Face hub: {exc}") from exc
    return LoadedModel(
        source=spec,
        model=model,
        example_inputs=load_inputs(inputs) if inputs else None,
        adapter_hint="hf",
    )
