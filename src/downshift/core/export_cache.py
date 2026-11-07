"""The optional disk tier behind the export memo (`--export-cache-dir DIR`).

Each entry `DIR/<key>/` holds what `downshift export` writes (model.onnx, any external data,
model.manifest.json) plus verdict.json, serving.json and feeds.npz. Downshift never deletes
entries. To clear the cache, delete the directory.

The key is the memo key, except that HF repo files are identified by content (inodes differ
between pods on one volume). `DIR/index.json` keeps those digests by file identity.

Both the manifest of an entry and `index.json` have `format_version`. Downshift ignores a file
with another version (or none) and makes it again, with one warning.

A write builds the entry in a temporary directory and renames it into place. A read checks the
graph against the manifest hashes. A bad entry is a miss, with one warning.
"""

import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from downshift.core.memo import (
    FORMAT_VERSION,
    MEMO,
    STORED_STATUSES,
    ExportEntry,
    file_identity,
    sha256_file,
)

logger = logging.getLogger("downshift.export_cache")

MODEL = "model.onnx"
VERDICT = "verdict.json"
FEEDS = "feeds.npz"
SERVING = "serving.json"
INDEX = "index.json"

# numpy.savez takes `file` and `allow_pickle` as its own keywords: a prefix keeps an input of
# that name apart.
_FEED_PREFIX = "in."

# A writer that a signal killed leaves a temporary or an old directory. The cache removes it
# when it is older than this.
_STALE_SECONDS = 3600


def check_dir(path: str | Path) -> Path:
    """The cache directory, or a ValueError (a usage error) if it cannot hold entries."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(
            f"--export-cache-dir {str(root)!r} is not an existing directory. Create it first"
        )
    try:
        with tempfile.TemporaryFile(dir=root):
            pass
    except OSError as exc:
        raise ValueError(
            f"--export-cache-dir {str(root)!r} is not writable: {exc.strerror or exc}. "
            "Give a directory that this process can write to"
        ) from exc
    return root


def lookup(
    mem_key: str | None, disk: "ExportCache | None", disk_key: str | None
) -> tuple[ExportEntry | None, str]:
    """(entry, tier): the memo first, then the disk tier; tier is "memory" or "disk"."""
    entry = MEMO.get(mem_key) if mem_key is not None else None
    if entry is None and disk is not None and disk_key is not None:
        return disk.get(disk_key), "disk"
    return entry, "memory"


def store(
    entry: ExportEntry, mem_key: str | None, disk: "ExportCache | None", disk_key: str | None
) -> None:
    """Keep `entry` in each tier that has a key. A failed disk write is only a warning."""
    if mem_key is not None:
        MEMO.put(mem_key, entry)
    if disk is not None and disk_key is not None:
        try:
            disk.put(disk_key, entry)
        except OSError as exc:
            logger.warning("export cache write failed in %s: %s", disk.root, exc)


def _write_json_atomic(path: Path, data: object) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


class ExportCache:
    def __init__(self, root: str | Path) -> None:
        self.root = check_dir(root)
        self._index: dict[str, str] | None = None
        self._remove_stale()

    def _remove_stale(self) -> None:
        """Delete the `.tmp-*` and `.old-*` directories that are older than an hour."""
        cutoff = time.time() - _STALE_SECONDS
        with contextlib.suppress(OSError):
            for path in self.root.iterdir():
                if (".tmp-" in path.name or ".old-" in path.name) and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)

    def _load_index(self) -> dict[str, str]:
        if self._index is None:
            self._index = {}
            try:
                data = json.loads((self.root / INDEX).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return self._index
            if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
                logger.warning(
                    "export cache index %s has format version %r, and downshift reads %d. "
                    "Downshift makes the index again",
                    self.root / INDEX,
                    data.get("format_version") if isinstance(data, dict) else None,
                    FORMAT_VERSION,
                )
            elif isinstance(data.get("digests"), dict):
                self._index = data["digests"]
        return self._index

    def fingerprint(self, path: str | Path) -> str:
        """sha256 of a file's content, remembered by (abspath, size, mtime, ctime, inode)."""
        identity = "|".join(str(part) for part in file_identity(path))
        index = self._load_index()
        digest = index.get(identity)
        if digest is None:
            digest = sha256_file(path)
            index[identity] = digest
            try:
                _write_json_atomic(
                    self.root / INDEX, {"format_version": FORMAT_VERSION, "digests": index}
                )
            except OSError as exc:
                logger.warning("export cache index not saved: %s", exc)
        return digest

    def get(self, key: str) -> ExportEntry | None:
        entry_dir = self.root / key
        if not entry_dir.is_dir():
            return None
        try:
            return self._read(entry_dir)
        except Exception as exc:  # noqa: BLE001 - for each fault in the entry, it is a miss
            logger.warning(
                "export cache entry %s is unusable (%s: %s). Exporting again",
                key[:12],
                type(exc).__name__,
                exc,
            )
            return None

    def _read(self, entry_dir: Path) -> ExportEntry:
        manifest = json.loads((entry_dir / "model.manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"its format version is {manifest.get('format_version')!r}, "
                f"and downshift reads {FORMAT_VERSION}"
            )
        onnx_path = entry_dir / MODEL
        if sha256_file(onnx_path) != manifest["onnx_sha256"]:
            raise ValueError(f"{MODEL} does not match its manifest hash")
        for item in manifest["external_data"]:
            name = item["file"]
            if Path(name).name != name:
                raise ValueError(f"external data location {name!r} leaves the entry")
            if sha256_file(entry_dir / name) != item["sha256"]:
                raise ValueError(f"{name} does not match its manifest hash")
        verdict = json.loads((entry_dir / VERDICT).read_text(encoding="utf-8"))
        serving = json.loads((entry_dir / SERVING).read_text(encoding="utf-8"))
        feeds = None
        if (entry_dir / FEEDS).exists():
            with np.load(entry_dir / FEEDS, allow_pickle=False) as archive:
                feeds = {name.removeprefix(_FEED_PREFIX): archive[name] for name in archive.files}
        return ExportEntry(
            verdict=verdict,
            input_names=list(serving["input_names"]),
            axis_bounds=serving["axis_bounds"],
            feeds=feeds,
            onnx_path=onnx_path,
        )

    def put(self, key: str, entry: ExportEntry) -> None:
        """Save a CLEAN or DEGRADED entry. Raises OSError if the directory cannot be written."""
        from downshift._version import __version__
        from downshift.core.manifest import external_data_files, write_manifest
        from downshift.core.verdict import ExportVerdict

        if entry.status not in STORED_STATUSES:
            return
        tmp = self.root / f"{key}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            onnx_path = tmp / MODEL
            if entry.onnx_bytes:
                onnx_path.write_bytes(entry.onnx_bytes)
            else:
                assert entry.onnx_path is not None
                shutil.copyfile(entry.onnx_path, onnx_path)
                for name in external_data_files(entry.onnx_path):
                    shutil.copyfile(entry.onnx_path.parent / name, tmp / name)
            verdict = ExportVerdict.from_dict(entry.verdict)
            write_manifest(onnx_path, verdict, None, __version__)
            _write_json_atomic(tmp / VERDICT, entry.verdict | {"onnx_path": None})
            _write_json_atomic(
                tmp / SERVING,
                {"input_names": entry.input_names, "axis_bounds": entry.axis_bounds},
            )
            feeds = entry.feeds
            if feeds and all(array.dtype.kind != "O" for array in feeds.values()):
                np.savez(tmp / FEEDS, **{_FEED_PREFIX + n: a for n, a in feeds.items()})  # type: ignore[arg-type]  # the numpy stub misreads **kwds
            self._install(tmp, self.root / key)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @staticmethod
    def _install(tmp: Path, final: Path) -> None:
        """Rename `tmp` to `final`, replacing an existing entry. The loser of a race drops its
        copy."""
        stale = None
        if final.exists():
            stale = final.with_name(f"{final.name}.old-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            try:
                os.replace(final, stale)
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
                return
        try:
            os.replace(tmp, final)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not final.exists():
                raise
        finally:
            if stale is not None:
                shutil.rmtree(stale, ignore_errors=True)
