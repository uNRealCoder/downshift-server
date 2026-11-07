"""`GET /schema`: what a client must POST, with an example body to paste. It is read from the
IO that the running backend declares. (`/metadata` is the operator's view: the verdict, boot
timings and warmup.)
"""

import math
import re
from typing import Any

import orjson

from downshift.core.axes import DimBound
from downshift.serve.backends import IOSpec, concrete_dim
from downshift.serve.engine import ServingState
from downshift.serve.graphs import GRAPH_INPUTS
from downshift.serve.schemas import (
    AxisBound,
    AxisInfo,
    EmbeddingInfo,
    InputFormat,
    Limits,
    OutputEncoding,
    SchemaResponse,
    SourceInfo,
    TensorSchema,
    TextInputInfo,
    normalize_dtype,
)
from downshift.sources import HF_REPO_DIR, SOURCE_KIND_HELP, UNKNOWN_SOURCE, display_source

# Above this many elements across all inputs, the example is dropped; example_shape remains.
MAX_EXAMPLE_ELEMENTS = 256

# Axis names that torch.export invents ("s0", "u3"). They show as "dynamic"; adapter names
# ("batch", "seq") stay.
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
    """A fixed size, a meaningful dynamic-axis name, or "dynamic"."""
    if isinstance(dim, int) and dim > 0:
        return dim
    if isinstance(dim, str) and dim and not _ANONYMOUS_AXIS.match(dim):
        return dim
    return DYNAMIC_AXIS


def _fill_value(name: str, dtype: str | None) -> Any:
    """Zero, except 1 for a mask: an all-zero attention mask gives NaN."""
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
    """One input or output, with the dtype as a client writes it ("float32"). `axis_info`
    (state.axis_bounds) gives the adapter's axis names and bounds, which the graph lost."""
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
    """A complete, valid `inputs` body of filler values, or None if it is too large."""
    shapes = [i.example_shape for i in inputs]
    if not inputs or any(s is None for s in shapes):
        return None
    if sum(math.prod(s) for s in shapes if s is not None) > MAX_EXAMPLE_ELEMENTS:
        return None
    return {i.name: _nested(i.example_shape or [], _fill_value(i.name, i.dtype)) for i in inputs}


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
        recipe.info(last if isinstance(last, int) else None)
        | {"prompts": recipe.prompts, "default_prompt": recipe.default_prompt}
    )


def _graph_batching(state: ServingState, outputs: list[TensorSchema]) -> dict[str, str] | None:
    """The axis-0 kind of each output of a PyG model, or None if nothing was classified."""
    kinds = state.verdict.output_axes
    if not kinds:
        return None
    names = [o.name for o in outputs] or [f"output_{i}" for i in range(len(kinds))]
    return {name: kinds[i] if i < len(kinds) else "unknown" for i, name in enumerate(names)}


def limits_info(state: ServingState) -> Limits:
    opts = state.options
    return Limits(
        max_body_bytes=opts.max_body_bytes,
        max_input_bytes=opts.max_input_bytes,
        max_concurrency=opts.max_concurrency,
        max_queue=opts.max_queue,
        request_timeout=opts.request_timeout,
    )


def axes_info(state: ServingState) -> list[AxisInfo]:
    return [AxisInfo.model_validate(fact.to_dict()) for fact in state.verdict.axes]


def describe(state: ServingState, predict_url: str) -> SchemaResponse:
    """The /schema body. `predict_url` comes from the request, so the example curl runs."""
    meta = state.backend.metadata()
    declared = {spec.name: spec for spec in meta.inputs}
    # state.input_names is what /predict checks; the torch backend can declare less.
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
        graph_batching=_graph_batching(state, outputs),
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
        limits=limits_info(state),
        notes=notes,
    )
