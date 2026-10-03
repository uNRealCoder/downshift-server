"""What `GET /schema` answers: the exact input format a client has to send.

`/metadata` reports how the model was *judged* -- the full export verdict, boot timings,
warmup stats -- which is the operator's view. This module answers the one question a caller
has before their first request: what goes in the body, with what dtypes and shapes, and a
worked example they can paste. Everything here is derived from the backend's own declared
IO, so it describes the graph that is actually running, not what the source model promised.

The `source` block repeats, per server, what `downshift` accepts at all: a model already
downloaded onto this machine -- an ONNX file, a PyTorch checkpoint, or a Hugging Face repo
directory holding a config.json -- or a module importable in this process. Nothing is ever
fetched, so a client reading this knows the server is pinned to that one local artifact.
"""

import math
import re
from typing import Any

import orjson

from downshift.loading import HF_REPO_DIR, SOURCE_KIND_HELP, UNKNOWN_SOURCE
from downshift.serve.backends import IOSpec, concrete_dim
from downshift.serve.engine import DimBound, ServingState
from downshift.serve.schemas import (
    GRAPH_INPUTS,
    AxisBound,
    AxisInfo,
    EmbeddingInfo,
    InputFormat,
    OutputEncoding,
    SchemaLimits,
    SchemaResponse,
    SourceInfo,
    TensorSchema,
    TextInputInfo,
    normalize_dtype,
)
from downshift.sources import display_source

# An inlined example stays a thing you can read and paste. Past this many elements across
# all inputs it is dropped and the per-input `example_shape` is left to speak for itself.
MAX_EXAMPLE_ELEMENTS = 256

# torch.export names the axes it made dynamic itself "s0", "s77", "u3"; ONNX Runtime then
# reports those verbatim. They carry no meaning outside the tracer, so this endpoint says
# "dynamic" instead and keeps only the names an adapter actually chose ("batch", "seq").
_ANONYMOUS_AXIS = re.compile(r"^[su]\d+$")
DYNAMIC_AXIS = "dynamic"

INPUT_FORMATS = {
    "nested list": 'a plain JSON array matching the tensor\'s rank, e.g. "x": [[1.0, 2.0]]',
    "typed object": (
        '{"data": [[1.0, 2.0]], "dtype": "float32", "shape": [1, 2]} -- dtype and shape are '
        "optional here and only cast/reshape what `data` already holds"
    ),
    "base64": (
        '{"data": "<base64>", "dtype": "float32", "shape": [1, 2]} -- raw little-endian '
        "buffer; dtype and shape are required, since neither can be read out of the bytes. "
        "Recommended for large tensors: it skips the float-to-text-to-float round trip and "
        "is roughly 4x cheaper for this server to parse than the same tensor as nested lists"
    ),
}


def _axis(dim: Any) -> int | str:
    """One axis as the client should read it: the fixed size, a meaningful dynamic-axis
    name, or the plain word "dynamic" for an unnamed or tracer-generated one."""
    if isinstance(dim, int) and dim > 0:
        return dim
    if isinstance(dim, str) and dim and not _ANONYMOUS_AXIS.match(dim):
        return dim
    return DYNAMIC_AXIS


def _fill_value(name: str, dtype: str | None) -> Any:
    """The single value an example tensor is filled with.

    Zero for everything, except an attention mask, where zero means "attend to nothing" and
    would hand back NaNs from a request that is otherwise perfectly well-formed.
    """
    if name.endswith("_mask"):
        return 1
    if dtype == "bool":
        return False
    if dtype is not None and dtype.startswith(("int", "uint")):
        return 0
    return 0.0


def _nested(shape: list[int], value: Any) -> Any:
    """A nested list of `value` with the given shape; a scalar for rank 0."""
    out = value
    for dim in reversed(shape):
        out = [out] * dim
    return out


def _tensor_schema(
    spec: IOSpec, *, required: bool | None, axis_info: dict[int, DimBound] | None = None
) -> TensorSchema:
    """One input or output, with the backend's dtype name normalised to the one a client
    would put in a typed/base64 body ("tensor(float)" -> "float32").

    `axis_info` (U2), when given, is the adapter's own axis names and (min, max) bounds
    (state.axis_bounds): the ONNX graph has lost the names (torch.export's own "s0"/"s1"),
    so a bounded axis reports the adapter's name and its bound instead of "dynamic".
    """
    raw = spec.shape
    shape: list[int | str] | None = None
    bounds: list[AxisBound | None] | None = None
    if raw is not None:
        infos = [axis_info.get(i) if axis_info else None for i in range(len(raw))]
        shape = [
            _axis(dim) if info is None else info.name for dim, info in zip(raw, infos, strict=True)
        ]
        if axis_info:
            bounds = [None if i is None else AxisBound(min=i.min, max=i.max) for i in infos]
    return TensorSchema(
        name=spec.name,
        dtype=normalize_dtype(spec.dtype),
        shape=shape,
        required=required,
        example_shape=[concrete_dim(dim) for dim in raw] if raw is not None else None,
        bounds=bounds,
    )


def _example_inputs(inputs: list[TensorSchema]) -> dict[str, Any] | None:
    """A complete, well-formed `inputs` body, or None when it would be too big to read.

    The values are filler; only the names, dtypes and shapes are meant to be copied.
    """
    sized: list[tuple[TensorSchema, list[int]]] = []
    for i in inputs:
        if i.example_shape is None:
            return None
        sized.append((i, i.example_shape))
    if not sized or sum(math.prod(shape) for _, shape in sized) > MAX_EXAMPLE_ELEMENTS:
        return None
    return {i.name: _nested(shape, _fill_value(i.name, i.dtype)) for i, shape in sized}


def _curl(url: str, body: dict[str, Any]) -> str:
    payload = orjson.dumps(body).decode()
    return f"curl -s {url} -H 'content-type: application/json' -d '{payload}'"


def _notes(
    outputs: list[TensorSchema], inputs: list[TensorSchema], example: dict | None
) -> list[str]:
    notes: list[str] = []
    if example is None and inputs:
        shapes = ", ".join(f"{i.name} {i.example_shape}" for i in inputs if i.example_shape)
        notes.append(
            "no example body inlined: it would run past "
            f"{MAX_EXAMPLE_ELEMENTS} elements. Build one of shape {shapes} yourself, and "
            "send it base64 rather than as nested lists if it is large."
        )
    if not outputs:
        notes.append(
            "output names and dtypes are not known yet: this model is served on the torch "
            "backend with no example inputs, so nothing has been run through it. They are "
            "output_0, output_1, ... in the model's own output order."
        )
    if any(i.dtype is None for i in inputs):
        notes.append(
            "some inputs declare no dtype: integer JSON lists are read as int64 and "
            "everything else as float32. Send a typed object to pin a dtype yourself."
        )
    return notes


def _token_level(outputs: list[TensorSchema]) -> bool:
    return bool(outputs) and outputs[0].shape is not None and len(outputs[0].shape) == 3


def _text_input(state: ServingState) -> TextInputInfo | None:
    text = state.text
    if text is None:
        return None
    labels = [text.id2label[i] for i in sorted(text.id2label)] if text.id2label else None
    return TextInputInfo(
        field="text",
        max_length=text.max_length,
        over_length="refused with 400; nothing is truncated",
        example_request={"text": ["your text here"]},
        labels=labels,
        activation=text.activation if text.id2label else None,
        response="`predictions`: one {label, score, probabilities} per row"
        if text.id2label
        else None,
    )


def _embedding(state: ServingState, outputs: list[TensorSchema]) -> EmbeddingInfo | None:
    recipe = state.embedding
    if recipe is None:
        return None
    last = outputs[0].shape[-1] if outputs and outputs[0].shape else None
    return EmbeddingInfo.model_validate(
        {
            "pooling": recipe.pooling,
            "normalized": recipe.normalize,
            "dimension": last if isinstance(last, int) else None,
            "max_seq_length": recipe.max_seq_length,
            "from": recipe.origin,
            "prompts": recipe.prompts,
            "default_prompt": recipe.default_prompt,
        }
    )


def axes_info(state: ServingState) -> list[AxisInfo]:
    return [AxisInfo.model_validate(fact.to_dict()) for fact in state.verdict.axes]


def describe(state: ServingState, predict_url: str) -> SchemaResponse:
    """Build the /schema body. `predict_url` is this server's own /predict URL, taken from
    the request, so the example curl is one the caller can actually run."""
    meta = state.backend.metadata()
    declared = {spec.name: spec for spec in meta.inputs}
    # Driven by state.input_names, not the backend's list: that is the forward-argument
    # order the predict routes check against, and the torch backend may declare less.
    inputs = [
        _tensor_schema(
            declared.get(name) or IOSpec(name, None, None),
            required=True,
            axis_info=state.axis_bounds.get(name),
        )
        for name in state.input_names
    ]
    outputs = [_tensor_schema(spec, required=None) for spec in meta.outputs]
    example = _example_inputs(inputs)
    body: dict[str, Any] | None = {"inputs": example} if example is not None else None
    notes = _notes(outputs, inputs, example)
    if state.hf_source is not None and state.embedding is None and _token_level(outputs):
        if state.source_kind == HF_REPO_DIR:
            notes.append(
                "output_0 is token-level ([batch, seq, hidden]): this repo declares no pooling "
                "recipe, so nothing was pooled. Serve with --pooling to get one vector per text."
            )
        else:
            notes.append(
                "output_0 is token-level ([batch, seq, hidden]): the served graph does no "
                "pooling, and --tokenizer-from only supplies the tokenizer. Pool the vectors "
                "yourself, or serve the Hugging Face repo directory with --pooling instead."
            )

    name = display_source(state.source, state.source_kind)
    return SchemaResponse(
        model=name,
        source=SourceInfo(
            spec=name,
            kind=state.source_kind,
            description=SOURCE_KIND_HELP.get(state.source_kind, SOURCE_KIND_HELP[UNKNOWN_SOURCE]),
            fetched_at_runtime=False,
        ),
        family=state.verdict.model_family,
        backend=meta.name,
        device=meta.device,
        endpoint="/predict",
        graph_endpoint="/predict/graph" if GRAPH_INPUTS <= set(state.input_names) else None,
        inputs=inputs,
        outputs=outputs,
        axes=axes_info(state),
        example_request=body,
        example_curl=_curl(predict_url, body) if body is not None else None,
        text_input=_text_input(state),
        embedding=_embedding(state, outputs),
        input_formats=[InputFormat(name=k, description=v) for k, v in INPUT_FORMATS.items()],
        output_encodings=[e.value for e in OutputEncoding],
        default_output_encoding=state.options.output_encoding.value,
        limits=SchemaLimits(
            max_body_bytes=state.options.max_body_bytes,
            max_input_bytes=state.options.max_input_bytes,
        ),
        notes=notes,
    )
