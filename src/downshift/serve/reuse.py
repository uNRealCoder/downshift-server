"""Boot from an earlier export if there is one: the in-process memo, then the
`--export-cache-dir` disk tier. A miss runs prepare_serving and stores its export in both.

An HF repo directory is looked up before anything loads, so a hit served by ONNX Runtime
never calls `from_pretrained`. Other sources are keyed from the loaded module. A `.onnx`
source exports nothing and is never cached.
"""

from downshift.adapters.base import Family
from downshift.core import memo
from downshift.core.axes import axis_bounds_to_json
from downshift.core.export_cache import ExportCache, lookup, store
from downshift.serve.engine import Loader, ServingState, prepare_serving, state_from_entry
from downshift.serve.options import ServeOptions
from downshift.sources import HF_REPO_DIR


def prepare_serving_reusing(
    load: Loader,
    opts: ServeOptions,
    tokenizer_from: str | None = None,
    *,
    repo: str | None = None,
    inputs_spec: str | None = None,
    cache: bool = True,
) -> ServingState:
    """`repo` is MODEL if it is an HF repo directory (with `inputs_spec`, its --inputs).
    `cache=False` skips both tiers."""
    if not cache:
        model, model_ref = load()
        return prepare_serving(model, opts, model_ref, tokenizer_from)

    disk = ExportCache(opts.export_cache_dir) if opts.export_cache_dir else None
    loaded = reference = None
    mem_key = disk_key = None
    if repo is not None:
        run = memo.run_options(opts, opts.adapter or Family.hf)
        mem_key = memo.guarded(memo.repo_key, repo, inputs_spec, memo.file_identity, **run)
        # The disk key hashes file contents: compute it only on a memo miss.
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
    verdict = state.verdict
    if verdict.status in memo.STORED_STATUSES and (mem_key or disk_key):
        entry = memo.build_entry(
            verdict, state.input_names, state.example_inputs, axis_bounds_to_json(state.axis_bounds)
        )
        store(entry, mem_key, disk, disk_key)
    return state
