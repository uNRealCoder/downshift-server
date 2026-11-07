"""The kinds of model source (no torch, for the CLI), and how to name a source for clients and
manifests: the file or directory name, never the local directory layout."""

import os
import re
from collections.abc import Iterable

ONNX_FILE = "onnx-file"
TORCH_CHECKPOINT = "torch-checkpoint"
HF_REPO_DIR = "hf-repo-dir"
IMPORT_SPEC = "import-spec"
# app_for() only: an nn.Module that the host process built.
IN_PROCESS_MODULE = "in-process-module"
UNKNOWN_SOURCE = "unknown"

SOURCE_KIND_HELP = {
    ONNX_FILE: "a downloaded or already-exported ONNX file on this machine",
    TORCH_CHECKPOINT: "a downloaded PyTorch checkpoint on this machine",
    HF_REPO_DIR: "a downloaded Hugging Face repo directory on this machine (has config.json)",
    IMPORT_SPEC: "an nn.Module imported from a module installed in this process",
    IN_PROCESS_MODULE: "an nn.Module this process already had in memory",
    UNKNOWN_SOURCE: "not one of the accepted local forms",
}

# Kinds whose source is a name, not a path.
LABEL_KINDS = frozenset({IMPORT_SPEC, IN_PROCESS_MODULE})


def path_basename(path: str) -> str:
    """The last component of `path`, split on both separators. "." and ".." name the
    directory that they point to."""
    parts = [part for part in re.split(r"[\\/]+", path) if part]
    name = parts[-1] if parts else ""
    if name in ("", ".", ".."):
        name = os.path.basename(os.path.abspath(path))
    return name or "model"


def display_source(source: str, kind: str) -> str:
    """`source` as a client sees it: the base name of a path; a label as it is."""
    return source if kind in LABEL_KINDS else path_basename(source)


def hide_paths(text: str, paths: Iterable[str | os.PathLike[str] | None]) -> str:
    """`text` with each of `paths` (as given, absolute, both separator styles) reduced to its
    base name. For free text that quotes a path, such as an exception message."""
    spellings: set[str] = set()
    for path in paths:
        if not path:
            continue
        given = os.fspath(path)
        for spelling in (given, os.path.abspath(given)):
            spellings.update((spelling, spelling.replace("\\", "/"), spelling.replace("/", "\\")))
    # Longest first, so no path is half replaced through its prefix. Skip spellings that are
    # only dots and separators (they match everywhere) and bare names (nothing to hide).
    for spelling in sorted(spellings, key=len, reverse=True):
        if not spelling.strip("./\\") or path_basename(spelling) == spelling:
            continue
        text = text.replace(spelling, path_basename(spelling))
    return text
