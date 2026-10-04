"""The in-process export memo: one process that exports or verifies the same model more than
once (check() then export() in a CI script, app_for() called repeatedly in a notebook or a test
suite) reuses the first result. Nothing here touches disk; core/export_cache.py is the opt-in
disk tier behind it (`--export-cache-dir`) and shares its key and entry types.

A key hashes everything that decides the exported graph and its verdict: the weights, the code
that defines the model, the trace and verify options, and the toolchain. HF repo directories are
keyed from their files before anything is loaded, so a hit never calls `from_pretrained`; every
other source is keyed from the loaded module.
"""

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from downshift import settings
from downshift._version import __version__

if TYPE_CHECKING:
    from downshift.core.verdict import ExportVerdict

logger = logging.getLogger("downshift.memo")

MAX_ENTRIES = 2
STORED_STATUSES = frozenset({"CLEAN", "DEGRADED"})

_TOOLS = ("torch", "onnx", "onnxscript", "onnxruntime", "transformers", "torch_geometric")
# Code that ships with these is covered by their version, so it isn't hashed per class.
_LIBRARY_PACKAGES = ("torch", "transformers", "torch_geometric")
_SENTENCE_TRANSFORMERS_DIR = re.compile(r"^\d+_\w+$")
_SCALAR_TYPES = (bool, int, float, str, bytes, type(None))


@dataclass
class ExportEntry:
    """What a boot needs to skip export and verify. `onnx_bytes` is the serialized graph; for
    an external-data graph it is empty and `onnx_path` names the .onnx beside its data file
    (a fresh export's temp copy, or a disk-tier file)."""

    verdict: dict
    input_names: list[str]
    axis_bounds: dict[str, list[list]]
    feeds: dict[str, np.ndarray] | None = None
    onnx_bytes: bytes = field(default=b"", repr=False)
    onnx_path: Path | None = None

    @property
    def status(self) -> str:
        return str(self.verdict["status"])

    @property
    def external(self) -> bool:
        return not self.onnx_bytes and self.onnx_path is not None

    def example_inputs(self) -> tuple | None:
        if self.feeds is None:
            return None
        return tuple(self.feeds[name] for name in self.input_names)


class ExportMemo:
    """A lock-protected LRU of exports. FAILED verdicts and graphs with external data are
    never kept: a failure is cheap to re-diagnose, and a second multi-GB copy in RAM for a
    rare reuse costs more than it saves."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, ExportEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> ExportEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: ExportEntry) -> None:
        if entry.status not in STORED_STATUSES or entry.external:
            return
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries


MEMO = ExportMemo(max_entries=MAX_ENTRIES)


def memo_key(parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def guarded[T](fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """A key that can't be computed (a file vanished, a tensor on the meta device) only costs
    the reuse: the caller exports as if there were no cache."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        logger.debug("export memo key unavailable", exc_info=True)
        return None


def file_identity(path: str | Path) -> list:
    """(abspath, size, mtime_ns, ctime_ns, inode): changes whenever the file is rewritten."""
    stat = Path(path).stat()
    return [os.path.abspath(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def sha256_file(path: str | Path) -> str:
    """A file's sha256, streamed in 1 MiB chunks so a multi-GB file is never held in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tool_versions() -> dict[str, str | None]:
    return {"downshift": __version__, **{name: _dist_version(name) for name in _TOOLS}}


def _code_ref(spec: object) -> list | None:
    """A `pkg.module:fn` / `path/to/file.py[:attr]` spec (or a callable) together with a digest
    of the file that defines it, so editing the code behind --vary or --adapter misses."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        name = getattr(spec, "name", None) or getattr(spec, "__qualname__", None)
        target = spec if inspect.isroutine(spec) or inspect.isclass(spec) else type(spec)
        return [str(name or target.__qualname__), _defining_file_digest(target)]
    path, sep, _ = spec.partition(".py:")
    candidate = Path(path + ".py") if sep else Path(spec)
    if candidate.suffix == ".py" and candidate.is_file():
        return [spec, sha256_file(candidate)]
    found = None
    if ":" in spec:
        try:
            found = importlib.util.find_spec(spec.partition(":")[0])
        except (ImportError, ValueError, AttributeError):
            pass
    origin = getattr(found, "origin", None)
    if origin and Path(origin).is_file():
        return [spec, sha256_file(origin)]
    return [spec, None]


def _defining_file_digest(obj: object) -> str | None:
    try:
        file = inspect.getfile(obj)  # type: ignore[arg-type]
        if Path(file).is_file():
            return sha256_file(file)
        return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()  # type: ignore[arg-type]
    except (TypeError, OSError):
        return None


def common_parts(
    *,
    adapter: object,
    dynamic: dict[str, list[int]] | None,
    k: int,
    seed: int,
    atol: float | None,
    rtol: float | None,
    vary: object,
    axis_max: dict[str, int] | None,
    pooling: str | None = None,
    normalize: bool | None = None,
    fp16: bool = False,
) -> dict:
    """Everything outside the model that decides the graph or the verdict. The opset isn't a
    downshift setting (torch picks it); it moves with the torch and onnxscript versions."""
    return {
        "adapter": _code_ref(adapter),
        "dynamic": dynamic,
        "k": k,
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "tolerances": {name: list(pair) for name, pair in settings.TOLERANCES.items()},
        "vary": _code_ref(vary),
        "axis_max": axis_max,
        "pooling": str(pooling) if pooling is not None else None,
        "normalize": normalize,
        "fp16": fp16,
        "versions": tool_versions(),
    }


def run_options(opts: Any, adapter: object) -> dict:
    """`common_parts` keywords from ServeOptions (or anything shaped like it)."""
    return {
        "adapter": adapter,
        "dynamic": opts.dynamic,
        "k": opts.k,
        "seed": opts.seed,
        "atol": opts.atol,
        "rtol": opts.rtol,
        "vary": opts.vary,
        "axis_max": opts.axis_max,
        "pooling": opts.pooling,
        "normalize": opts.normalize,
    }


def repo_files(root: str | Path) -> list[Path]:
    """The files of an HF repo directory that can shape the export: its top level (config,
    weights, tokenizer) and the sentence-transformers module directories (1_Pooling, ...).
    Subdirectories holding other copies of the model (onnx/, openvino/) are left out."""
    base = Path(root)
    entries = sorted(base.iterdir())
    files = [p for p in entries if p.is_file() and not p.name.startswith(".")]
    for sub in entries:
        if sub.is_dir() and _SENTENCE_TRANSFORMERS_DIR.match(sub.name):
            files += [
                p for p in sorted(sub.iterdir()) if p.is_file() and not p.name.startswith(".")
            ]
    return files


def repo_parts(root: str | Path, fingerprint: Callable[[Path], object]) -> list:
    base = Path(root)
    return [[p.relative_to(base).as_posix(), fingerprint(p)] for p in repo_files(base)]


def _tensor_part(t: Any) -> list:
    array = t.detach().cpu().contiguous().numpy() if isinstance(t, torch.Tensor) else t
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return [str(array.dtype), list(array.shape), digest]


def example_parts(example_inputs: tuple | None) -> list | None:
    if example_inputs is None:
        return None
    return [
        _tensor_part(t) if isinstance(t, (torch.Tensor, np.ndarray)) else repr(t)
        for t in example_inputs
    ]


def state_digest(model: torch.nn.Module) -> str:
    """The `state_dict` tensor bytes, in key order."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(f"{name}|".encode())
        if not isinstance(value, torch.Tensor):
            digest.update(repr(value).encode())
            continue
        digest.update(f"{value.dtype}|{tuple(value.shape)}|".encode())
        flat = value.detach().cpu().contiguous().reshape(-1)
        digest.update(flat.view(torch.uint8).numpy())
    return digest.hexdigest()


def _library_roots() -> list[Path]:
    roots = []
    for name in _LIBRARY_PACKAGES:
        file = getattr(sys.modules.get(name), "__file__", None)
        if file:
            roots.append(Path(file).resolve().parent)
    return roots


def code_parts(model: torch.nn.Module) -> list:
    """A digest of the source file of every distinct nn.Module subclass in the model that
    doesn't come from torch, transformers or torch_geometric: editing a `forward` misses."""
    roots = _library_roots()
    parts = []
    classes = {type(m) for m in model.modules()}
    for cls in sorted(classes, key=lambda c: f"{c.__module__}.{c.__qualname__}"):
        try:
            file = Path(inspect.getfile(cls)).resolve()
        except (TypeError, OSError):
            continue  # a builtin or dynamically built class: nothing to read
        if any(file.is_relative_to(root) for root in roots):
            continue
        parts.append([f"{cls.__module__}.{cls.__qualname__}", _defining_file_digest(cls)])
    return parts


def _simple(value: object) -> object | None:
    if isinstance(value, _SCALAR_TYPES):
        return repr(value)
    if isinstance(value, (tuple, list)) and len(value) <= 64:
        items = [_simple(v) for v in value]
        return items if all(i is not None for i in items) else None
    return None


def _attribute_parts(model: torch.nn.Module) -> list:
    """Plain attributes (sizes, flags, scales) of every submodule: a constructor argument can
    change the graph with no change to the weights or the source."""
    parts = []
    for name, module in model.named_modules():
        attrs = {
            key: simple
            for key, value in vars(module).items()
            if key != "training" and (simple := _simple(value)) is not None
        }
        parts.append([name, type(module).__qualname__, attrs])
    return parts


def module_parts(model: torch.nn.Module) -> dict:
    from downshift.adapters.embedding import EMBEDDING_ATTR, PADDING_SIDE_ATTR

    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    return {
        "state": state_digest(model),
        "code": code_parts(model),
        "attributes": _attribute_parts(model),
        "config": to_dict() if callable(to_dict) else None,
        "recipe": repr(getattr(model, EMBEDDING_ATTR, None)),
        "padding_side": repr(getattr(model, PADDING_SIDE_ATTR, None)),
    }


def model_key(model: torch.nn.Module, example_inputs: tuple | None, **run: Any) -> str:
    return memo_key(
        {
            "model": module_parts(model),
            "inputs": example_parts(example_inputs),
            "run": common_parts(**run),
        }
    )


def repo_key(
    repo: str | Path,
    inputs_spec: str | None,
    fingerprint: Callable[[Path], object],
    **run: Any,
) -> str:
    """The key of a Hugging Face repo directory, from its files and before it is loaded.
    `fingerprint` is file_identity for the memo and a content digest for the disk tier."""
    inputs = None
    if inputs_spec:
        from downshift.loading import load_inputs

        inputs = example_parts(load_inputs(inputs_spec))
    return memo_key(
        {
            "repo": repo_parts(repo, fingerprint),
            "inputs": inputs,
            "inputs_code": _code_ref(inputs_spec),
            "run": common_parts(**run),
        }
    )


def build_entry(
    verdict: "ExportVerdict",
    input_names: tuple[str, ...],
    example_inputs: tuple | None,
    axis_bounds: dict[str, list[list]],
) -> ExportEntry:
    from downshift.core.feeds import example_feeds

    feeds = example_feeds(input_names, example_inputs) if example_inputs is not None else None
    return ExportEntry(
        verdict=verdict.to_dict(),
        input_names=list(input_names),
        axis_bounds=axis_bounds,
        feeds=feeds,
        onnx_bytes=verdict.onnx_bytes,
        onnx_path=verdict.onnx_path,
    )


def entry_from_verdict(verdict: "ExportVerdict") -> ExportEntry | None:
    """The entry for a verdict straight out of build_verdict (it still has its Prepared)."""
    from downshift.core.axes import axis_bounds_to_json
    from downshift.core.shapes import dynamic_bounds

    prepared = verdict.prepared
    if prepared is None:
        return None
    bounds = dynamic_bounds(prepared.input_names, prepared.dynamic_shapes)
    return build_entry(verdict, prepared.input_names, prepared.inputs, axis_bounds_to_json(bounds))
