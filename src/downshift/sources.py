"""What a model argument can turn out to be. Stdlib only, so the CLI's banner can name the
kinds without importing torch (downshift.loading, which decides the kind, imports it).

Also how a source is named to anyone who is not the operator: the HTTP routes and the
manifest that travels with an exported file. They get the file or directory name, never the
directory layout of the machine it was loaded from."""

import os
import re
from collections.abc import Iterable

# Reported to clients by /schema. The first three are the downloaded-artifact forms;
# "import-spec" names a model already importable in this process.
ONNX_FILE = "onnx-file"
TORCH_CHECKPOINT = "torch-checkpoint"
HF_REPO_DIR = "hf-repo-dir"
IMPORT_SPEC = "import-spec"
# Not reachable from the CLI: app_for() takes an nn.Module the host process already built.
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

# These two are names, not locations; every other kind (including an unknown one) is a path.
LABEL_KINDS = frozenset({IMPORT_SPEC, IN_PROCESS_MODULE})


def path_basename(path: str) -> str:
    """The last component of `path`. Split on both separators so a Windows path is cut
    correctly when the server runs on POSIX (and the reverse); "." and ".." name the
    directory they point at."""
    parts = [part for part in re.split(r"[\\/]+", path) if part]
    name = parts[-1] if parts else ""
    if name in ("", ".", ".."):
        name = os.path.basename(os.path.abspath(path))
    return name or "model"


def display_source(source: str, kind: str) -> str:
    """`source` as a client or a manifest reader should see it: the file or directory name
    for a path, an import spec or an in-process label unchanged."""
    return source if kind in LABEL_KINDS else path_basename(source)


def hide_paths(text: str, paths: Iterable[str | os.PathLike[str] | None]) -> str:
    """`text` with each of `paths` (as given, absolute, and either separator style) replaced
    by its basename. For free-form messages an exception or a note carries, which quote the
    path they were handed."""
    spellings: set[str] = set()
    for path in paths:
        if not path:
            continue
        given = os.fspath(path)
        for spelling in (given, os.path.abspath(given)):
            spellings.update((spelling, spelling.replace("\\", "/"), spelling.replace("/", "\\")))
    # Longest first, so a path is never half-replaced by one of its own prefixes. A spelling
    # that is only dots and separators ("." for the current directory) would match all over
    # ordinary text, and one that is already a bare name has nothing to hide.
    for spelling in sorted(spellings, key=len, reverse=True):
        if not spelling.strip("./\\") or path_basename(spelling) == spelling:
            continue
        text = text.replace(spelling, path_basename(spelling))
    return text
