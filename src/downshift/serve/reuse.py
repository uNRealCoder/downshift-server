"""Boot a ServingState from an earlier export when there is one: the in-process memo
(core/memo.py), then the `--export-cache-dir` disk tier (core/export_cache.py). A miss is the
usual prepare_serving, whose verified export is then stored in both.

A Hugging Face repo directory is looked up before anything is loaded, so a hit never calls
`from_pretrained` (when the cached graph serves it: a torch-served hit still loads the model,
which torch needs to run, and only skips export and verify). Every other source is keyed from
the loaded module, so a hit there skips export and verify but not the load. A `.onnx` source
exports nothing, so it is never cached.
"""

import copy
import logging
from collections.abc import Callable
from pathlib import Path

from downshift.adapters.base import Family
from downshift.core import memo
from downshift.core.export_cache import ExportCache
from downshift.core.verdict import BackendName, ExportVerdict
from downshift.loading import HF_REPO_DIR, LoadedModel
from downshift.serve.engine import (
    ServingState,
    axis_bounds_from_json,
    axis_bounds_to_json,
    choose_backend,
    prepare_serving,
    serving_state_from_artifact,
    serving_state_from_torch_artifact,
)
from downshift.serve.options import ServeOptions

logger = logging.getLogger("downshift.reuse")

Loader = Callable[[], tuple[LoadedModel, LoadedModel | None]]


def prepare_serving_reusing(
    load: Loader,
    opts: ServeOptions,
    tokenizer_from: str | None = None,
    *,
    repo: str | None = None,
    inputs_spec: str | None = None,
    cache: bool = True,
) -> ServingState:
    """`load()` returns the loaded MODEL and its --reference model, if any; it is only called
    when something has to be loaded. `repo` is MODEL when it is a Hugging Face repo directory
    (and `inputs_spec` its --inputs). `cache=False` skips both tiers, in both directions."""
    if not cache:
        model, model_ref = load()
        return prepare_serving(model, opts, model_ref, tokenizer_from)

    disk = ExportCache(opts.export_cache_dir) if opts.export_cache_dir else None
    loaded: LoadedModel | None = None
    reference: LoadedModel | None = None
    mem_key: str | None = None
    disk_key: str | None = None
    if repo is not None:
        run = memo.run_options(opts, opts.adapter or Family.hf)
        mem_key = memo.guarded(memo.repo_key, repo, inputs_spec, memo.file_identity, **run)
        entry = memo.MEMO.get(mem_key) if mem_key is not None else None
        tier = "memory"
        if entry is None and disk is not None:
            disk_key = memo.guarded(memo.repo_key, repo, inputs_spec, disk.fingerprint, **run)
    else:
        loaded, reference = load()
        entry, tier = None, "memory"
        if loaded.model is not None and loaded.onnx_path is None:
            run = memo.run_options(opts, opts.adapter or loaded.adapter_hint)
            mem_key = memo.guarded(memo.model_key, loaded.model, loaded.example_inputs, **run)
            entry = memo.MEMO.get(mem_key) if mem_key is not None else None
            disk_key = mem_key if disk is not None else None
    if entry is None and disk is not None and disk_key is not None:
        entry, tier = disk.get(disk_key), "disk"

    if entry is not None:
        return _state_from_entry(entry, tier, load, loaded, reference, opts, tokenizer_from, repo)

    if loaded is None:
        loaded, reference = load()
    state = prepare_serving(loaded, opts, reference, tokenizer_from)
    _store(state, mem_key, disk, disk_key)
    return state


def _state_from_entry(
    entry: memo.ExportEntry,
    tier: str,
    load: Loader,
    loaded: LoadedModel | None,
    reference: LoadedModel | None,
    opts: ServeOptions,
    tokenizer_from: str | None,
    repo: str | None,
) -> ServingState:
    verdict = ExportVerdict.from_dict(copy.deepcopy(entry.verdict))
    verdict.onnx_bytes = entry.onnx_bytes
    verdict.onnx_path = entry.onnx_path
    name, notes = choose_backend(verdict, opts, has_torch=True)
    if loaded is None:
        source, kind = str(repo), HF_REPO_DIR
    else:
        source, kind = loaded.source, loaded.kind
    hf_source = source if kind == HF_REPO_DIR else tokenizer_from

    if name == BackendName.torch:
        # Eager serving needs the model itself: the cache only saves the export and verify.
        if loaded is None:
            loaded, reference = load()
        return serving_state_from_torch_artifact(
            loaded, verdict, opts, notes=notes, hf_source=hf_source, reused=tier
        )

    if loaded is not None:
        loaded.model = None
    if reference is not None:
        reference.model = None
    return serving_state_from_artifact(
        source,
        entry.onnx_path,
        verdict,
        opts,
        tuple(entry.input_names),
        notes,
        entry.example_inputs(),
        axis_bounds_from_json(entry.axis_bounds),
        kind=kind,
        hf_source=hf_source,
        reused=tier,
    )


def _store(
    state: ServingState, mem_key: str | None, disk: ExportCache | None, disk_key: str | None
) -> None:
    verdict = state.verdict
    if verdict.status not in memo.STORED_STATUSES:
        return
    if mem_key is None and (disk is None or disk_key is None):
        return
    entry = memo.build_entry(
        verdict, state.input_names, state.example_inputs, axis_bounds_to_json(state.axis_bounds)
    )
    if mem_key is not None:
        memo.MEMO.put(mem_key, entry)
    if disk is not None and disk_key is not None:
        try:
            disk.put(disk_key, entry)
        except OSError as exc:
            logger.warning("export cache write failed in %s: %s", Path(disk.root), exc)
