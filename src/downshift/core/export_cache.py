"""The opt-in disk tier behind the export memo (`--export-cache-dir DIR`).

Each entry is `DIR/<key>/`: what `downshift export` writes (model.onnx, its external data file
if any, model.manifest.json) plus verdict.json, feeds.npz and serving.json. The directory is the
operator's: downshift never evicts from it, polices its permissions or offers commands for it.
Deleting it clears the cache.

The key is core/memo.py's, except that an HF repo's files are identified by content (sha256),
because inodes and ctimes differ between pods sharing a volume. Digests are remembered in
`DIR/index.json` by file identity, so an unchanged file is hashed once per machine.

Writes build the entry in `DIR/<key>.tmp-<pid>-<rand>/` and rename it into place, so a reader
never sees half an entry and a crash leaves only a temp directory nobody reads. A read re-hashes
the graph and its data file against the manifest; an entry that fails any check is one warning
and a miss, and the re-export overwrites it.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from downshift.core.memo import ExportEntry, file_identity

logger = logging.getLogger("downshift.export_cache")

MODEL = "model.onnx"
VERDICT = "verdict.json"
FEEDS = "feeds.npz"
SERVING = "serving.json"
INDEX = "index.json"


def check_dir(path: str | Path) -> Path:
    """The cache directory, or a ValueError (a usage error) when it can't hold entries: the
    operator asked for persistence, so serving without it would hide the problem."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"--export-cache-dir {str(root)!r} is not an existing directory")
    try:
        with tempfile.TemporaryFile(dir=root):
            pass
    except OSError as exc:
        raise ValueError(
            f"--export-cache-dir {str(root)!r} is not writable: {exc.strerror or exc}"
        ) from exc
    return root


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: object) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


class ExportCache:
    def __init__(self, root: str | Path) -> None:
        self.root = check_dir(root)
        self._index: dict[str, str] | None = None

    def _load_index(self) -> dict[str, str]:
        if self._index is None:
            try:
                data = json.loads((self.root / INDEX).read_text(encoding="utf-8"))
                self._index = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self._index = {}
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
                _write_json_atomic(self.root / INDEX, index)
            except OSError as exc:
                logger.warning("export cache index not saved: %s", exc)
        return digest

    def get(self, key: str) -> ExportEntry | None:
        entry_dir = self.root / key
        if not entry_dir.is_dir():
            return None
        try:
            return self._read(entry_dir)
        except Exception as exc:  # noqa: BLE001 - whatever is wrong with it, it is a miss
            logger.warning(
                "export cache entry %s is unusable (%s: %s); exporting again",
                key[:12],
                type(exc).__name__,
                exc,
            )
            return None

    def _read(self, entry_dir: Path) -> ExportEntry:
        manifest = json.loads((entry_dir / "model.manifest.json").read_text(encoding="utf-8"))
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
                feeds = {name: archive[name] for name in archive.files}
        return ExportEntry(
            verdict=verdict,
            input_names=list(serving["input_names"]),
            axis_bounds=serving["axis_bounds"],
            feeds=feeds,
            onnx_path=onnx_path,
        )

    def put(self, key: str, entry: ExportEntry) -> None:
        """Save a CLEAN or DEGRADED entry (external-data ones too); anything else is ignored.
        Raises OSError when the directory can't be written; the caller warns and serves on."""
        from downshift._version import __version__
        from downshift.core.manifest import external_data_files, write_manifest
        from downshift.core.verdict import ExportVerdict

        if entry.status not in ("CLEAN", "DEGRADED"):
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
                np.savez(tmp / FEEDS, **feeds)  # type: ignore[arg-type]  # numpy's stub misreads **kwds
            self._install(tmp, self.root / key)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @staticmethod
    def _install(tmp: Path, final: Path) -> None:
        """Rename `tmp` to `final`, moving an existing (stale or corrupt) entry aside first.
        A writer that loses a race to another process just drops its own copy."""
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
