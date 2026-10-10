"""Boot a ServingState from an earlier export, if there is one. The sources are the in-process
memo (core/memo.py) and then the `--export-cache-dir` disk tier (core/export_cache.py). A miss is
the usual prepare_serving. Downshift then stores its verified export in both.

Downshift looks up a Hugging Face repo directory before it loads anything. A hit therefore never
calls `from_pretrained`, if the cached graph serves it. A hit that torch serves still loads the
model, because torch needs it to run. It skips only the export and the verification. For each
other source, downshift makes the key from the loaded module. A hit there skips the export and
the verification, but not the load. A `.onnx` source exports nothing, so downshift never caches
it.
"""

import copy
from collections.abc import Callable

from downshift.adapters.base import Family
from downshift.core import memo
from downshift.core.axes import axis_bounds_from_json, axis_bounds_to_json
from downshift.core.export_cache import ExportCache, lookup, store
from downshift.core.verdict import BackendName, ExportVerdict
from downshift.loading import LoadedModel
from downshift.serve.engine import (
    ServingState,
    choose_backend,
    prepare_serving,
    serving_state_from_artifact,
    serving_state_from_torch_artifact,
)
from downshift.serve.options import ServeOptions
from downshift.sources import HF_REPO_DIR

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
    """`load()` returns the loaded MODEL and its --reference model, if there is one. Downshift
    calls it only when it must load something. `repo` is MODEL if it is a Hugging Face repo
    directory (and `inputs_spec` is its --inputs). `cache=False` skips both tiers, for reads and
    for writes."""
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
        # The disk key of a repo hashes the file contents. Downshift therefore computes it only on a memo miss.
        if disk is not None and (mem_key is None or mem_key not in memo.MEMO):
            disk_key = memo.guarded(memo.repo_key, repo, inputs_spec, disk.fingerprint, **run)
    else:
        loaded, reference = load()
        if loaded.model is not None and loaded.onnx_path is None:
            run = memo.run_options(opts, opts.adapter or loaded.adapter_hint)
            mem_key = memo.guarded(memo.model_key, loaded.model, loaded.example_inputs, **run)
            disk_key = mem_key if disk is not None else None
    entry, tier = lookup(mem_key, disk, disk_key)

    if entry is not None:
        source, kind = (str(repo), HF_REPO_DIR) if loaded is None else (loaded.source, loaded.kind)
        return state_from_entry(
            entry,
            load,
            opts,
            tokenizer_from,
            source=source,
            kind=kind,
            reused=tier,
            loaded=loaded,
            reference=reference,
        )

    if loaded is None:
        loaded, reference = load()
    state = prepare_serving(loaded, opts, reference, tokenizer_from)
    _store(state, mem_key, disk, disk_key)
    return state


def state_from_entry(
    entry: memo.ExportEntry,
    load: Loader,
    opts: ServeOptions,
    tokenizer_from: str | None,
    *,
    source: str,
    kind: str,
    reused: str | None = None,
    loaded: LoadedModel | None = None,
    reference: LoadedModel | None = None,
) -> ServingState:
    """A ServingState from an export that already ran: a cache hit (`reused` names the tier), or
    a `serve --workers N` worker that takes the export of its parent. Downshift selects the
    backend again from the verdict, in the same way as the boot of the export itself. It calls
    `load()` only when the torch backend needs the model and `loaded` does not have it."""
    verdict = ExportVerdict.from_dict(copy.deepcopy(entry.verdict))
    verdict.onnx_bytes = entry.onnx_bytes
    verdict.onnx_path = entry.onnx_path
    name, notes = choose_backend(verdict, opts, has_torch=True)
    hf_source = source if kind == HF_REPO_DIR else tokenizer_from

    if name == BackendName.torch:
        # Eager serving needs the model itself. The export saves only the export and the verification.
        if loaded is None:
            loaded, reference = load()
        return serving_state_from_torch_artifact(
            loaded, verdict, opts, notes=notes, hf_source=hf_source, reused=reused
        )

    if loaded is not None:
        loaded.model = None
    if reference is not None:
        reference.model = None
    return serving_state_from_artifact(
        source,
        verdict,
        opts,
        tuple(entry.input_names),
        notes,
        entry.example_inputs(),
        axis_bounds_from_json(entry.axis_bounds),
        kind=kind,
        hf_source=hf_source,
        reused=reused,
    )


def _store(
    state: ServingState, mem_key: str | None, disk: ExportCache | None, disk_key: str | None
) -> None:
    verdict = state.verdict
    if verdict.status not in memo.STORED_STATUSES:
        return  # checked here and also by each tier, to skip the building of the entry
    if mem_key is None and (disk is None or disk_key is None):
        return
    entry = memo.build_entry(
        verdict, state.input_names, state.example_inputs, axis_bounds_to_json(state.axis_bounds)
    )
    store(entry, mem_key, disk, disk_key)
