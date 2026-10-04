"""Request/response models for the HTTP layer, plus JSON <-> numpy conversion."""

import math
import re
from enum import StrEnum
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from downshift.serve.codec import b64decode
from downshift.settings import DEFAULT_MAX_INPUT_BYTES


class OutputEncoding(StrEnum):
    """How response tensors are written: nested lists, or {data, dtype, shape} base64 dicts."""

    json = "json"
    base64 = "base64"


class RequestOutputEncoding(StrEnum):
    """What a request's `output_encoding` may name: OutputEncoding plus safetensors, which is a
    whole-response format (an `Accept` header asks for it too) and so not a server default."""

    json = "json"
    base64 = "base64"
    safetensors = "safetensors"


OutputEncodingField = Annotated[
    RequestOutputEncoding | None,
    Field(
        None,
        description=(
            "How response tensors are encoded: 'json' for nested lists, 'base64' for "
            '{"data": <base64 little-endian bytes>, "dtype": ..., "shape": [...]} per output, '
            "'safetensors' for an application/vnd.safetensors response body (the same as "
            "sending Accept: application/vnd.safetensors). "
            "Omit to use the server default (--output-encoding / DOWNSHIFT_OUTPUT_ENCODING)."
        ),
    ),
]

# ORT reports dtypes as "tensor(float)"; map those names onto numpy ones. bfloat16 has no
# numpy dtype of its own; the wire contract for it is float32, same as the torch backend's
# (B1), so it maps there too.
_ORT_DTYPES = {
    "float": "float32",
    "float16": "float16",
    "bfloat16": "float32",
    "double": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
}
_TENSOR_RE = re.compile(r"^tensor\((\w+)\)$")


def normalize_dtype(dtype: str | None) -> str | None:
    """Turn "tensor(float)", "tensor(float32)" or "float32" into a numpy dtype name; None if
    unknown. The torch backend spells dtypes the numpy way inside "tensor(...)", ORT its own."""
    if dtype is None:
        return None
    match = _TENSOR_RE.match(dtype)
    name = match.group(1) if match else dtype
    if match and name in _ORT_DTYPES:
        return _ORT_DTYPES[name]
    try:
        return np.dtype(name).name
    except TypeError:
        return None


class TypedArray(BaseModel):
    """Explicit form of an input: {"data": [...], "dtype": "float32", "shape": [2, 3]}.

    `data` may instead be a base64 string of the raw little-endian buffer, in which case
    `dtype` and `shape` are required.
    """

    data: Any
    dtype: str | None = None
    shape: list[int] | None = None


class PredictRequest(BaseModel):
    """`inputs` maps input name -> nested list, or a TypedArray object for explicit typing.

    A model served from a Hugging Face repo directory that has tokenizer files also accepts
    `text` (one string or a list) instead of `inputs`; the server tokenizes. Send one or the
    other, not both.
    """

    inputs: dict[str, Any] = Field(default_factory=dict)
    text: str | list[str] | None = None
    # One of the repo's named prompts (config_sentence_transformers.json), put before each
    # text row; only meaningful with `text`.
    prompt_name: str | None = None
    output_encoding: OutputEncodingField = None

    @model_validator(mode="after")
    def _one_kind_of_input(self) -> "PredictRequest":
        if self.text is not None and self.inputs:
            raise ValueError("send either 'inputs' or 'text', not both")
        return self


class PredictResponse(BaseModel):
    outputs: dict[str, Any]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]
    # Present only for a sequence classifier: one {label, score, probabilities} per row.
    predictions: list[dict[str, Any]] | None = None


class GraphItem(BaseModel):
    """One graph of a batch: node features, COO edge index in the graph's own node ids,
    optional edge attributes."""

    x: list | dict
    edge_index: list | dict
    edge_attr: list | dict | None = None


class GraphPredictRequest(BaseModel):
    """One graph (top-level `x`, `edge_index`, optional `edge_attr`) or a batch (`graphs`, a
    list of the same three fields), never both.

    A batch runs as one inference over the graphs joined into one disjoint graph; each
    `edge_index` uses its own graph's local node ids and the server does the offsetting. The
    response is {"graphs": [{outputs, shapes, dtypes}, ...]} in request order; with
    `Accept: application/vnd.safetensors` each output is a tensor named
    `graphs.<i>.<output name>`, and `__metadata__` carries the graph count as
    `downshift.graphs`. Node-level and edge-level outputs are split per graph; a model whose
    output has a fixed size (a pooled readout) takes one graph per request.
    A safetensors request body may carry the batch as concatenated `x`, `edge_index`,
    `edge_attr` plus int64 `num_nodes` and `num_edges` vectors of length G.
    Each tensor is a nested list or a TypedArray dict (which is how base64 arrives).
    """

    x: list | dict | None = None
    edge_index: list | dict | None = None
    edge_attr: list | dict | None = None
    graphs: list[GraphItem] | None = None
    output_encoding: OutputEncodingField = None

    @model_validator(mode="after")
    def _one_graph_or_a_batch(self) -> "GraphPredictRequest":
        single = (self.x, self.edge_index, self.edge_attr)
        if self.graphs is not None:
            if any(v is not None for v in single):
                raise ValueError(
                    "send either 'graphs' or top-level x/edge_index/edge_attr, not both"
                )
            if not self.graphs:
                raise ValueError("'graphs' is empty")
        elif self.x is None or self.edge_index is None:
            raise ValueError("send 'x' and 'edge_index' for one graph, or 'graphs' for a batch")
        return self


class WorstMismatchInfo(BaseModel):
    sample: int
    output: int
    index: list[int]
    expected: float
    got: float
    input_shapes: list[list[int]]


class NumericsInfo(BaseModel):
    samples_tested: int
    max_abs_err: float
    max_rel_err: float
    failures: int
    shape_generalization: bool | None
    tolerance_abs: float
    tolerance_rel: float
    tolerance_dtype: str
    tolerance_overridden: bool
    baseline_failed: bool
    worst: WorstMismatchInfo | None = None
    sample_shapes: list[list[list[int]]] = Field(default_factory=list)
    seed: int
    notes: list[str] = Field(default_factory=list)
    passed: bool


class VerdictInfo(BaseModel):
    """ExportVerdict.to_dict(); see downshift.core.verdict."""

    status: str
    model_family: str
    capture_strategy: str | None
    opset: int | None
    op_types: dict[str, int]
    numerics: NumericsInfo | None
    shape_generalization: bool | None
    shape_generalization_reason: str | None
    recommended_backend: str
    reason: str
    input_names: list[str]
    dynamic_dims: dict[str, list[int]]
    unsupported_ops: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # No onnx_path: to_dict() carries one, but it is a location on the server's disk.


class IOSpecInfo(BaseModel):
    name: str
    dtype: str | None
    shape: list[int | str | None] | None


class BackendInfo(BaseModel):
    """BackendMeta.to_dict(); see downshift.serve.backends."""

    name: str
    device: str
    inputs: list[IOSpecInfo] = Field(default_factory=list)
    outputs: list[IOSpecInfo] = Field(default_factory=list)


class AxisInfo(BaseModel):
    """One dynamic axis: what the server accepts (`served_*`, the export Dim's bounds) next to
    what verification ran (`sampled_*`, None when verify never ran). `name` is the Dim name
    `--axis-max` takes. See downshift.core.axes."""

    input: str
    axis: int
    name: str
    served_min: int
    served_max: int
    sampled_min: int | None = None
    sampled_max: int | None = None


class Limits(BaseModel):
    """The serving limits a client runs into; the same block on /metadata and /schema."""

    max_body_bytes: int
    max_input_bytes: int
    max_concurrency: int
    max_queue: int
    request_timeout: float


class MetadataResponse(BaseModel):
    model: str
    family: str
    verdict: VerdictInfo
    axes: list[AxisInfo] = Field(default_factory=list)
    backend: BackendInfo
    input_names: list[str]
    notes: list[str] = Field(default_factory=list)
    version: str
    limits: Limits
    execution: str  # "threadpool" or "inline" (--execution)
    boot: dict[str, float] = Field(default_factory=dict)
    warmup: dict | None = None


class AxisBound(BaseModel):
    """min/max the adapter's export traced this axis for (torch.export.Dim's own bounds).

    Only present for axes an adapter made dynamic on a model downshift itself exported; a
    bare `.onnx` with no reference model carries none (TensorSchema.bounds is None then).
    """

    min: int
    max: int


class TensorSchema(BaseModel):
    """One input or output as /schema describes it.

    In `shape`, an int is a fixed size and a string is a dynamic axis: either a name the
    adapter chose ("batch", "seq") or the plain word "dynamic". `example_shape` is that
    shape with every dynamic axis pinned to 1, which is what `example_request` used.
    `bounds`, when known, is parallel to `shape`: one AxisBound per dynamic axis downshift's
    own export traced, None elsewhere.
    """

    name: str
    dtype: str | None = None
    shape: list[int | str] | None = None
    required: bool | None = None  # inputs only; every declared input is required
    example_shape: list[int] | None = None
    bounds: list[AxisBound | None] | None = None


class InputFormat(BaseModel):
    """One of the wire forms a tensor value may take in a request body."""

    name: str
    description: str


class SourceInfo(BaseModel):
    spec: str
    kind: str
    description: str
    fetched_at_runtime: bool


class TextInputInfo(BaseModel):
    """Present when /predict also takes {"text": ...}. See downshift/adapters/text.py."""

    field: str
    max_length: int
    over_length: str
    example_request: dict
    # The three below are set only for a sequence classifier.
    labels: list[str] | None = None
    activation: str | None = None
    response: str | None = None


class EmbeddingInfo(BaseModel):
    """Present when the graph ends in a pooling step. See downshift/adapters/embedding.py."""

    model_config = ConfigDict(populate_by_name=True)

    pooling: str
    normalized: bool
    dimension: int | None
    max_seq_length: int | None
    origin: str = Field(alias="from")
    # Named prompts a /predict `text` request may select with `prompt_name`, and the one that
    # applies when it names none.
    prompts: dict[str, str] = Field(default_factory=dict)
    default_prompt: str | None = None


class SchemaResponse(BaseModel):
    """The answer to "what do I POST?": see downshift/serve/describe.py."""

    model: str
    source: SourceInfo
    family: str
    backend: str
    device: str
    endpoint: str
    graph_endpoint: str | None = None
    # PyG models: output name -> "node" / "edge" (split per graph in a `graphs` batch),
    # "fixed" / "unknown" (one graph per request).
    graph_batching: dict[str, str] | None = None
    inputs: list[TensorSchema] = Field(default_factory=list)
    outputs: list[TensorSchema] = Field(default_factory=list)
    axes: list[AxisInfo] = Field(default_factory=list)
    example_request: dict | None = None
    example_curl: str | None = None
    text_input: TextInputInfo | None = None
    embedding: EmbeddingInfo | None = None
    input_formats: list[InputFormat] = Field(default_factory=list)
    output_encodings: list[str] = Field(default_factory=list)
    default_output_encoding: str
    limits: Limits
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    # The loader's current phase while not ready (U4); None once ready, or when the app was
    # built with state= directly (no loader, so no not-ready window to report on).
    phase: str | None = None


def _from_base64(
    name: str, typed: TypedArray, dtype_name: str | None, max_bytes: int
) -> np.ndarray:
    """A view over the decoded bytes: no cast, no copy. The caller's dtype wins."""
    if dtype_name is None or typed.shape is None:
        raise ValueError(
            f"input {name!r}: base64 input needs dtype and shape "
            "(the shape cannot be inferred from bytes)"
        )
    # np.dtype(">f4").name is plain "float32", so byte order is checked on the client's spec.
    if str(typed.dtype).startswith(">"):
        raise ValueError(
            f"input {name!r}: dtype {typed.dtype!r} is big-endian; base64 input must be little-endian"
        )
    dtype = np.dtype(dtype_name)
    if any(dim < 0 for dim in typed.shape):
        raise ValueError(f"input {name!r}: shape {typed.shape} has a negative dimension")
    expected = math.prod(typed.shape) * dtype.itemsize
    # Checked before decoding: the shape says how much would be handed to the backend.
    if expected > max_bytes:
        raise ValueError(
            f"input {name!r}: {expected} bytes exceeds the server limit of {max_bytes} bytes"
        )
    try:
        raw = b64decode(typed.data)
    except ValueError as exc:  # binascii.Error is a ValueError
        raise ValueError(f"input {name!r}: invalid base64 data: {exc}") from exc
    if len(raw) != expected:
        raise ValueError(
            f"input {name!r}: expected {expected} bytes for shape {typed.shape} and dtype "
            f"{dtype.name}, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(typed.shape)


def to_numpy(
    name: str,
    value: Any,
    expected_dtype: str | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> np.ndarray:
    """Convert a JSON input to an ndarray.

    `expected_dtype` is whatever the backend declared (an ORT "tensor(...)" name, a numpy
    name, or None). An explicit dtype in the payload wins. With nothing declared, integer
    lists become int64 and everything else float32, which is what torch models expect.

    A TypedArray whose `data` is a string is base64 of the raw little-endian buffer and
    must carry dtype and shape; it decodes to at most `max_bytes`.
    """
    explicit_dtype: str | None = None
    explicit_shape: list[int] | None = None
    if isinstance(value, dict):
        typed = TypedArray.model_validate(value)
        if typed.dtype:
            explicit_dtype = normalize_dtype(typed.dtype)
            if explicit_dtype is None:
                raise ValueError(f"input {name!r}: unknown dtype {typed.dtype!r}")
        if isinstance(typed.data, str):
            return _from_base64(name, typed, explicit_dtype, max_bytes)
        value = typed.data
        explicit_shape = typed.shape

    try:
        arr = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"input {name!r}: {exc}") from exc
    if arr.dtype == object:
        raise ValueError(f"input {name!r}: ragged or non-numeric array")

    dtype = explicit_dtype or normalize_dtype(expected_dtype)
    if dtype is None:
        dtype = "int64" if np.issubdtype(arr.dtype, np.integer) else "float32"
    try:
        arr = arr.astype(dtype, copy=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"input {name!r}: cannot cast to {dtype}: {exc}") from exc

    if explicit_shape is not None:
        try:
            arr = arr.reshape(explicit_shape)
        except ValueError as exc:
            raise ValueError(f"input {name!r}: {exc}") from exc
    return arr
