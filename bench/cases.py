"""The benchmark corpus: five fixtures, one per hazard class, plus payload scaling.

Every server variant in this harness is handed the *same* prepared module and the *same*
exported ONNX graph, so the only variable between them is the serving policy and the
server code around it. Nothing here is downshift-specific except `prepare`, which is
just the flatten-and-export step the naive baselines would otherwise have to hand-roll.
"""

from __future__ import annotations

import base64
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Every model is built under this seed in every process, so the reference output computed by
# the orchestrator and the weights inside a server subprocess are the same weights. Without it
# each process randomly initialises its own model and every correctness number is noise.
SEED = 1234

# Tier 1: the export-hazard fixtures, sized for tracing rather than for arithmetic.
# Tier 2: compute-heavy models of the same shapes, where the backend choice is observable
# above the cost of the HTTP round trip.
FIXTURE_CASES = (
    "clean_mlp",
    "dynamic_batch_cnn",
    "gnn_gcn",
    "tiny_bert",
    "scatter_include_self_false",
)
LARGE_CASES = ("mlp_large", "cnn_large", "bert_small")
CASE_NAMES = FIXTURE_CASES + LARGE_CASES

TIER = {**{n: "fixture" for n in FIXTURE_CASES}, **{n: "compute-heavy" for n in LARGE_CASES}}

# What one unit of "batch" means per model. For the graph model it is one more disjoint graph
# concatenated with offset edge indices, which is how you batch PyG without graph batching.
BATCH_KIND = {
    "clean_mlp": "rows",
    "dynamic_batch_cnn": "images",
    "gnn_gcn": "graphs",
    "tiny_bert": "sequences",
    "scatter_include_self_false": "6-node groups",
    "mlp_large": "rows",
    "cnn_large": "images",
    "bert_small": "sequences",
}


@dataclass
class Case:
    name: str
    family: str
    input_names: tuple[str, ...]
    dtypes: dict[str, str]  # numpy dtype name per input
    module: torch.nn.Module  # the flattened, export-ready module
    onnx_bytes: bytes | None
    verdict_status: str
    verdict_backend: str
    verdict_reason: str
    max_abs_err: float | None


def _import(name: str):
    package = "bench.large" if name in LARGE_CASES else "tests.models"
    return importlib.import_module(f"{package}.{name}")


def load_fixture(name: str) -> tuple[torch.nn.Module, tuple | None]:
    """Build the model and its example inputs deterministically. Every process must use this."""
    mod = _import(name)
    torch.manual_seed(SEED)
    model = mod.make_model()
    example = None
    if hasattr(mod, "make_inputs"):
        torch.manual_seed(SEED)
        example = mod.make_inputs()
    return model, example


def make_inputs(name: str, batch: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Numpy feeds for one request at the given batch size. Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    if name == "clean_mlp":
        return {"x": rng.standard_normal((batch, 16), dtype=np.float32)}
    if name == "dynamic_batch_cnn":
        return {"x": rng.standard_normal((batch, 3, 16, 16), dtype=np.float32)}
    if name == "tiny_bert":
        ids = rng.integers(0, 100, (batch, 8), dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    if name == "scatter_include_self_false":
        n = 6 * batch
        return {
            "x": rng.standard_normal((n, 8), dtype=np.float32),
            "segment_ids": rng.integers(0, 4, (n,), dtype=np.int64),
        }
    if name == "mlp_large":
        return {"x": rng.standard_normal((batch, 512), dtype=np.float32)}
    if name == "cnn_large":
        return {"x": rng.standard_normal((batch, 3, 32, 32), dtype=np.float32)}
    if name == "bert_small":
        ids = rng.integers(0, 30522, (batch, 128), dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    if name == "gnn_gcn":
        # batch disjoint 6-node/10-edge graphs, edge indices offset per graph
        nodes_per, edges_per = 6, 10
        x = rng.standard_normal((nodes_per * batch, 8), dtype=np.float32)
        blocks = [
            rng.integers(0, nodes_per, (2, edges_per), dtype=np.int64) + g * nodes_per
            for g in range(batch)
        ]
        return {"x": x, "edge_index": np.concatenate(blocks, axis=1)}
    raise KeyError(name)


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    """The base64 TypedArray form: raw little-endian C-contiguous bytes plus dtype and shape.

    Both are mandatory on the wire because the shape is not inferable from the bytes.
    """
    # np.require rather than np.ascontiguousarray: the latter promotes a 0-d scalar to (1,),
    # which would put the wrong shape on the wire.
    arr = np.require(arr, requirements="C")
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
    }


def decode_array(obj: Any) -> np.ndarray:
    """Inverse of `encode_array`, and also accepts the nested-list form, so a response can be
    scored against the reference regardless of which encoding the server was asked for."""
    if isinstance(obj, dict) and isinstance(obj.get("data"), str):
        raw = base64.b64decode(obj["data"])
        return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    if isinstance(obj, dict):
        return np.asarray(obj["data"], dtype=np.dtype(obj["dtype"]))
    return np.asarray(obj)


def to_payload(feeds: dict[str, np.ndarray], encoding: str = "json") -> dict[str, Any]:
    """The request body. Identical across all server variants so request size is not a variable.

    `encoding="json"` is the nested-list form every server accepts. `encoding="base64"` sends
    each input as `{"data": <base64>, "dtype", "shape"}` and asks for the outputs in the same
    form via the top-level `output_encoding` field.
    """
    if encoding == "json":
        return {"inputs": {k: v.tolist() for k, v in feeds.items()}}
    if encoding == "base64":
        return {
            "inputs": {k: encode_array(v) for k, v in feeds.items()},
            "output_encoding": "base64",
        }
    raise ValueError(f"unknown encoding {encoding!r}")


def prepare(name: str, export: bool = True) -> Case:
    """Flatten the fixture into export form and (optionally) capture the ONNX graph."""
    from downshift.core.verdict import build_verdict, prepare_model

    model, example = load_fixture(name)
    prepared = prepare_model(model, example)

    dtypes = {
        n: str(t.numpy().dtype) for n, t in zip(prepared.input_names, prepared.inputs, strict=True)
    }
    if not export:
        return Case(
            name=name,
            family=prepared.family,
            input_names=prepared.input_names,
            dtypes=dtypes,
            module=prepared.model,
            onnx_bytes=None,
            verdict_status="SKIPPED",
            verdict_backend="torch",
            verdict_reason="export skipped",
            max_abs_err=None,
        )

    verdict = build_verdict(prepared, k=8)
    onnx_bytes = None
    if verdict.onnx_program is not None:
        onnx_bytes = verdict.onnx_program.model_proto.SerializeToString()
    return Case(
        name=name,
        family=prepared.family,
        input_names=prepared.input_names,
        dtypes=dtypes,
        module=prepared.model,
        onnx_bytes=onnx_bytes,
        verdict_status=verdict.status,
        verdict_backend=str(verdict.recommended_backend),
        verdict_reason=verdict.reason,
        max_abs_err=verdict.numerics.max_abs_err if verdict.numerics else None,
    )


def artifact_dir(root: Path) -> Path:
    d = root / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def onnx_path(root: Path, name: str) -> Path:
    return artifact_dir(root) / f"{name}.onnx"


def torch_reference(case: Case, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Eager PyTorch output: the ground truth every variant's response is scored against."""
    args = [torch.from_numpy(np.ascontiguousarray(feeds[n])) for n in case.input_names]
    with torch.inference_mode():
        out = case.module(*args)
    tensors = (
        [out] if isinstance(out, torch.Tensor) else [t for t in out if isinstance(t, torch.Tensor)]
    )
    return [t.detach().cpu().numpy() for t in tensors]
