"""The input feeds of a request: prompts, tokenizing, graph batching and the input checks (a 400)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from downshift.entities.adapters import TextTooLongError
from downshift.entities.axes import DimBound
from downshift.entities.requests import ParsedRequest
from downshift.entities.serving import ServingState
from downshift.entities.signatures import GraphLayout
from downshift.serve.errors import ApiError
from downshift.serve.graphs import (
    GRAPH_TENSORS,
    batch_graphs,
    binary_batch,
    edge_index_violation,
    index_range_violation,
    split_refusal,
)
from downshift.serve.schemas import normalize_dtype, to_numpy


def resolve_prompt(state: ServingState, prompt_name: str | None, has_text: bool) -> str:
    """The prompt to put before each text row: the named one, else the repo default, else ""."""
    recipe = state.embedding
    prompts = recipe.prompts if recipe is not None else {}
    if prompt_name is None:
        default = recipe.default_prompt if recipe is not None else None
        return prompts[default] if has_text and default is not None else ""
    if not has_text:
        raise ApiError(
            400, "invalid_input", "'prompt_name' applies to a 'text' request. None was sent"
        )
    if prompt_name not in prompts:
        available = sorted(prompts)
        raise ApiError(
            400,
            "unknown_prompt",
            f"unknown prompt_name {prompt_name!r:.64}. "
            + (f"this model has: {available}" if available else "this model has no named prompts"),
        )
    return prompts[prompt_name]


def text_feeds(
    state: ServingState,
    text: list[str],
    declared: dict[str, str | None],
    prompt: str = "",
    count_tokens: bool = True,
) -> tuple[dict[str, np.ndarray], list[int] | None]:
    """Tokenize `text` (each row after `prompt`) into the graph inputs. With `count_tokens`,
    also returns the token count of each row, from the tokenizer output before it is cut to the
    model inputs."""
    assert state.text is not None  # run_predict refuses a text request without it
    encoded = state.text.encode([prompt + row for row in text] if prompt else text)
    missing = [n for n in state.input_names if n not in encoded]
    if missing:
        raise ValueError(f"the tokenizer does not produce the model's inputs {missing}")
    feeds = {n: to_numpy(n, encoded[n], declared.get(n)) for n in state.input_names}
    return feeds, _row_tokens(encoded) if count_tokens else None


def _vocab_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """ORT wraps a negative input_ids index: check the range before the inference."""
    if state.vocab_size is None:
        return None
    return index_range_violation(
        "input_ids", feeds.get("input_ids"), state.vocab_size, "vocabulary"
    )


def bound_message(name: str, axis: int, size: int, bound: DimBound) -> str:
    return f"{name} axis {axis} is {size}; this model accepts {bound.min} to {bound.max}"


def shape_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """The first input whose rank or fixed axis differs from the backend's declaration, so a
    shape error inside the model is never blamed on the client."""
    for name, arr in feeds.items():
        spec = state.input_specs.get(name)
        if spec is None or spec.shape is None:
            continue
        if arr.ndim != len(spec.shape):
            return f"{name} has {arr.ndim} dimensions; this model takes {len(spec.shape)}"
        for axis, dim in enumerate(spec.shape):
            if isinstance(dim, int) and arr.shape[axis] != dim:
                return f"{name} axis {axis} is {arr.shape[axis]}; this model takes {dim}"
    return None


def _bound_violation(
    state: ServingState, feeds: dict[str, np.ndarray], skip_batch_axis: bool = False
) -> str | None:
    """The first input outside the axis bounds that the export traced. `skip_batch_axis` is
    for cache-miss rows: the caller already checked the whole batch."""
    for name, arr in feeds.items():
        bounds = state.axis_bounds.get(name)
        if not bounds:
            continue
        for axis, bound in bounds.items():
            if axis >= arr.ndim or (skip_batch_axis and axis == 0):
                continue
            size = arr.shape[axis]
            if not (bound.min <= size <= bound.max):
                return bound_message(name, axis, size, bound)
    return None


def input_violation(
    state: ServingState,
    feeds: dict[str, np.ndarray],
    *,
    edges: bool = True,
    skip_batch_axis: bool = False,
) -> str | None:
    """The first client error in `feeds`: a token ID outside the vocabulary, an edge index
    outside the nodes, a wrong rank or fixed axis, or an axis outside its bounds."""
    return (
        _vocab_violation(state, feeds)
        or (edge_index_violation(feeds) if edges else None)
        or shape_violation(state, feeds)
        or _bound_violation(state, feeds, skip_batch_axis)
    )


def _binary_feeds(
    state: ServingState, tensors: dict[str, np.ndarray], declared: dict[str, str | None]
) -> dict[str, np.ndarray]:
    """The arrays of a safetensors body, never cast: a wrong dtype is a 400, not a copy."""
    feeds = {n: tensors[n] for n in state.input_names}
    for name, arr in feeds.items():
        expected = normalize_dtype(declared.get(name))
        if expected is not None and arr.dtype.name != expected:
            raise ValueError(f"input {name!r} is {arr.dtype.name}. This model takes {expected}")
    return feeds


def tensor_feeds(state: ServingState, request: ParsedRequest) -> dict[str, np.ndarray]:
    """The arrays of an `inputs` or safetensors request."""
    declared = state.declared_dtypes
    if request.tensors is not None:
        return _binary_feeds(state, request.tensors, declared)
    max_bytes = state.options.max_input_bytes
    return {
        n: to_numpy(n, request.inputs[n], declared.get(n), max_bytes=max_bytes)
        for n in state.input_names
    }


def _graph_item(
    state: ServingState, index: int, graph: dict[str, Any], max_bytes: int
) -> dict[str, np.ndarray]:
    """One `graphs` entry (JSON values) to arrays; an error names the graph."""
    declared = state.declared_dtypes
    try:
        return {
            n: to_numpy(n, graph[n], declared.get(n), max_bytes=max_bytes)
            for n in GRAPH_TENSORS
            if n in state.input_names and graph.get(n) is not None
        }
    except (ValueError, TypeError) as exc:
        raise ValueError(f"graphs[{index}]: {exc}") from exc


def _batch_feeds(
    state: ServingState,
    graphs: list[dict[str, Any]] | None,
    tensors: dict[str, np.ndarray] | None,
    max_bytes: int,
) -> tuple[dict[str, np.ndarray], GraphLayout]:
    """One batched feed set, and the counts for each graph that split the response, from a
    `graphs` list or a safetensors body with num_nodes and num_edges."""
    names = [spec.name for spec in state.backend.metadata().outputs]
    if tensors is not None:
        _binary_feeds(state, tensors, state.declared_dtypes)
        feeds, node_counts, edge_counts = binary_batch(tensors)
    else:
        assert graphs is not None
        items = [_graph_item(state, i, graph, max_bytes) for i, graph in enumerate(graphs)]
        feeds, node_counts, edge_counts = batch_graphs(items)
    refusal = split_refusal(state.verdict.output_axes, names, len(node_counts))
    if refusal is not None:
        raise ValueError(refusal)
    missing = [n for n in state.input_names if n not in feeds]
    if missing:
        raise ValueError(f"missing inputs: {missing}")
    return {n: feeds[n] for n in state.input_names}, (node_counts, edge_counts)


def _row_tokens(feeds: dict[str, np.ndarray]) -> list[int]:
    """The tokens that are not padding, for each row of text feeds."""
    mask = feeds.get("attention_mask")
    if mask is not None:
        return mask.reshape(mask.shape[0], -1).sum(axis=1).tolist()  # type: ignore[no-any-return]
    ids = feeds["input_ids"] if "input_ids" in feeds else next(iter(feeds.values()))
    return [ids.shape[1]] * ids.shape[0]


def input_error(exc: Exception) -> ApiError:
    code = "text_too_long" if isinstance(exc, TextTooLongError) else "invalid_input"
    return ApiError(400, code, str(exc))


def prepare_feeds(
    state: ServingState,
    request: ParsedRequest,
    skip_batch_axis: bool = False,
    count_tokens: bool = False,
) -> tuple[dict[str, np.ndarray], float, GraphLayout | None]:
    """The graph inputs of a validated request (decode, convert, tokenize, join graphs), after
    the checks that need them. Returns (feeds, milliseconds, graph layout or None). A text
    request gets `request.tokens` only with `count_tokens`. Raises a 400 for each client
    error."""
    tensors = request.tensors
    start = time.perf_counter()
    layout: GraphLayout | None = None
    try:
        if request.graphs is not None or (
            tensors is not None and ("num_nodes" in tensors or "num_edges" in tensors)
        ):
            feeds, layout = _batch_feeds(
                state, request.graphs, tensors, state.options.max_input_bytes
            )
        elif request.text is not None:
            feeds, request.tokens = text_feeds(
                state, request.text, state.declared_dtypes, request.prompt, count_tokens
            )
        else:
            feeds = tensor_feeds(state, request)
    except (ValueError, TypeError, KeyError) as exc:
        raise input_error(exc) from exc

    # A graph batch was already checked graph by graph.
    violation = input_violation(state, feeds, edges=layout is None, skip_batch_axis=skip_batch_axis)
    if violation is not None:
        raise ApiError(400, "invalid_input", violation)
    return feeds, (time.perf_counter() - start) * 1000, layout


def as_batch(value: str | list[str]) -> list[str]:
    """A single string is a batch of one, so the response shape is the same for both forms."""
    return [value] if isinstance(value, str) else list(value)
