"""What a model argument can be. This module uses only the standard library. The banner of the
CLI can therefore name the kinds without an import of torch. (downshift.loading decides the
kind, and it imports torch.)

It also gives the name of a source for people who are not the operator: the HTTP routes and
the manifest that goes with an exported file. They get the file name or directory name. They
never get the directory layout of the machine that loaded the model."""

import os
import re
from collections.abc import Iterable

# /schema reports these to clients. The first three are the forms of a downloaded artifact.
# "import-spec" names a model that this process can already import.
ONNX_FILE = "onnx-file"
TORCH_CHECKPOINT = "torch-checkpoint"
HF_REPO_DIR = "hf-repo-dir"
IMPORT_SPEC = "import-spec"
# The CLI cannot reach this kind. app_for() takes an nn.Module that the host process already built.
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

# These two are names and not locations. Every other kind (also an unknown one) is a path.
LABEL_KINDS = frozenset({IMPORT_SPEC, IN_PROCESS_MODULE})


def path_basename(path: str) -> str:
    """The last component of `path`. Downshift splits on both separators. A Windows path is
    therefore cut correctly when the server runs on POSIX (and the reverse). "." and ".." name
    the directory that they point to."""
    parts = [part for part in re.split(r"[\\/]+", path) if part]
    name = parts[-1] if parts else ""
    if name in ("", ".", ".."):
        name = os.path.basename(os.path.abspath(path))
    return name or "model"


def display_source(source: str, kind: str) -> str:
    """`source` as a client or a manifest reader must see it: the file name or directory name
    for a path. An import spec or an in-process label does not change."""
    return source if kind in LABEL_KINDS else path_basename(source)


def hide_paths(text: str, paths: Iterable[str | os.PathLike[str] | None]) -> str:
    """`text` with each of `paths` (as given, absolute, and in both separator styles) replaced
    by its basename. Use it for free-form messages that an exception or a note carries. They
    quote the path that they received."""
    spellings: set[str] = set()
    for path in paths:
        if not path:
            continue
        given = os.fspath(path)
        for spelling in (given, os.path.abspath(given)):
            spellings.update((spelling, spelling.replace("\\", "/"), spelling.replace("/", "\\")))
    # The longest first. A path is then never half replaced by one of its own prefixes. A
    # spelling that is only dots and separators ("." for the current directory) would match
    # everywhere in normal text. A spelling that is already a bare name has nothing to hide.
    for spelling in sorted(spellings, key=len, reverse=True):
        if not spelling.strip("./\\") or path_basename(spelling) == spelling:
            continue
        text = text.replace(spelling, path_basename(spelling))
    return text
