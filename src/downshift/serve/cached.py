"""The result-cache path of /predict: look up the rows, compute the missing ones, join."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future

import numpy as np
from fastapi import Response

from downshift import settings
from downshift.entities.cache import Lookup, Row
from downshift.entities.requests import CachePlan, ParsedRequest
from downshift.entities.serving import ServingState
from downshift.entities.signatures import Encode, Parse
from downshift.serve.encode import encode_response
from downshift.serve.errors import ApiError
from downshift.serve.feeds import (
    bound_message,
    input_error,
    input_violation,
    prepare_feeds,
    tensor_feeds,
)
from downshift.serve.parse import parse_request
from downshift.serve.result_cache import ResultCache, tensor_keys, text_keys
from downshift.serve.stages import check_timeout, run_in_pool, run_inference

# The key of the token count in a cached text row. The encoders never see it as an output.
TOKENS_KEY = "__tokens__"


def _row_count_violation(state: ServingState, rows: int) -> str | None:
    """The batch-axis bound for a text request, whose hit rows are never tokenized."""
    for name, bounds in state.axis_bounds.items():
        bound = bounds.get(0)
        if bound is not None and not (bound.min <= rows <= bound.max):
            return bound_message(name, 0, rows, bound)
    return None


def _subset(
    request: ParsedRequest, arrays: dict[str, np.ndarray] | None, rows: list[int]
) -> ParsedRequest:
    """A request with only `rows` of the batch."""
    if request.text is not None:
        return ParsedRequest(text=[request.text[i] for i in rows], prompt=request.prompt)
    assert arrays is not None
    return ParsedRequest(tensors={n: a[rows] for n, a in arrays.items()})


def cache_prepare(
    state: ServingState,
    cache: ResultCache,
    request: ParsedRequest,
    timings_ms: dict[str, float],
    use_lookup: bool = True,
    deadline: float | None = None,
) -> CachePlan:
    """The prep hop of a cached request. A tensor request is checked whole before the lookup,
    so a hit never hides a client error. Only text rows that missed are tokenized. The lookup
    claims the missed rows; a later failure in this hop releases them. A request that is past
    its `deadline` claims nothing."""
    start = time.perf_counter()
    arrays: dict[str, np.ndarray] | None = None
    try:
        if request.text is not None:
            keys = text_keys(request.prompt, request.text)
            violation = _row_count_violation(state, len(keys))
        else:
            arrays = tensor_feeds(state, request)
            counts = {a.shape[0] if a.ndim else -1 for a in arrays.values()}
            if len(counts) != 1 or -1 in counts:
                raise ValueError("each input needs the same number of rows on axis 0")
            keys = tensor_keys(arrays, counts.pop())
            violation = input_violation(state, arrays)
    except (ValueError, TypeError, KeyError) as exc:
        raise input_error(exc) from exc
    if violation is not None:
        raise ApiError(400, "invalid_input", violation)

    if not keys:
        whole, _, _ = prepare_feeds(state, request)
        return CachePlan(request, arrays, keys, Lookup(), whole, passthrough=True)
    check_timeout(state, deadline)
    cache_start = time.perf_counter()
    # Cache-Control: no-cache reads and claims nothing. The rows are still stored.
    found = cache.lookup(keys) if use_lookup else Lookup(missing=list(range(len(keys))))
    claimed = [keys[i] for i in found.missing] if use_lookup else []
    timings_ms["cache"] = round((time.perf_counter() - cache_start) * 1000, 2)
    timings_ms["cache_hits"] = len(found.hits) + len(found.pending)
    timings_ms["cache_rows"] = len(keys)
    tokens: list[int] | None = None
    try:
        feeds: dict[str, np.ndarray] | None = None
        if found.missing:
            if arrays is not None and len(found.missing) == len(keys):
                feeds = arrays
            else:
                subset = _subset(request, arrays, found.missing)
                feeds, _, _ = prepare_feeds(state, subset, skip_batch_axis=True, count_tokens=True)
                tokens = subset.tokens
    except BaseException:
        cache.fail(claimed)
        raise
    timings_ms["prep"] = round((time.perf_counter() - start) * 1000, 2)
    return CachePlan(request, arrays, keys, found, feeds, claimed, tokens=tokens)


def _release_claims(cache: ResultCache, prep: asyncio.Future[CachePlan]) -> None:
    """The request left before its plan arrived: fail its claimed rows, so the waiters compute
    them themselves."""
    if not prep.cancelled() and prep.exception() is None:
        cache.fail(prep.result().owned)


async def await_row(future: Future[Row]) -> Row:
    """The row that another request computes. asyncio.shield (unlike awaiting the wrapped
    future) does not cancel it if this request leaves, so the other waiters still get the row."""
    wrapped = asyncio.wrap_future(future)
    wrapped.add_done_callback(lambda f: f.cancelled() or f.exception())  # no unread errors
    return await asyncio.shield(wrapped)


async def run_cached(
    state: ServingState,
    cache: ResultCache,
    parsed: ParsedRequest | None,
    *,
    body: bytes,
    graph: bool,
    binary: bool,
    accept_safetensors: bool,
    parse: Parse | None,
    encode: Encode | None,
    with_tokens: bool,
    run_here: bool,
    deadline: float | None,
    timings_ms: dict[str, float],
    use_lookup: bool = True,
) -> Response:
    """run_predict with the result cache. The model sees only the missed rows. Hits, rows from
    other requests and new rows are joined in request order and encoded on the prep pool. A
    text row keeps its token count under TOKENS_KEY."""
    pool = None if run_here else state.prep_executor

    def prepare() -> CachePlan:
        request = parsed or (
            parse() if parse is not None else parse_request(state, body, graph, binary, timings_ms)
        )
        return cache_prepare(state, cache, request, timings_ms, use_lookup, deadline)

    # If the client leaves during prep, the thread still finishes and claims rows: release them.
    prep = asyncio.ensure_future(run_in_pool(pool, "prep_wait", timings_ms, prepare))
    try:
        plan: CachePlan = await asyncio.shield(prep)
    except asyncio.CancelledError:
        prep.add_done_callback(lambda t: _release_claims(cache, t))
        raise
    owned = plan.owned

    # `rows` maps a row index of the request to its output. The cache hits are in it already.
    # The steps below add the rows that this request computes, then the rows that other
    # requests compute. `finish` joins them in the order of the request.
    rows: dict[int, Row] = dict(plan.found.hits)
    served = len(plan.found.hits)  # the rows that this request did not compute

    async def compute(
        feeds: dict[str, np.ndarray], indexes: list[int], tokens: list[int] | None
    ) -> Response | None:
        """Run the model on `feeds` (the rows `indexes`) and give each row to the cache."""

        def infer_rows() -> Response | None:
            check_timeout(state, deadline)
            outputs, infer_ms = run_inference(state, feeds)
            timings_ms["infer"] = round(infer_ms, 2)
            # True if each output has one entry for each row that we sent. Only then can we
            # cache and join the rows one by one.
            by_row = all(o.ndim and o.shape[0] == len(indexes) for o in outputs.values())
            if plan.passthrough or (not by_row and len(indexes) == len(plan.keys)):
                # An empty batch, or outputs without a row axis: nothing to cache or join. The
                # waiters of the claimed rows compute them themselves.
                cache.fail(owned)
                owned.clear()
                if encode is not None:
                    return encode(outputs, plan.request, None, None)
                return encode_response(
                    state, outputs, plan.request.encoding, timings_ms, accept_safetensors, None
                )
            if not by_row:
                raise RuntimeError("an output of the model does not follow the batch size")
            for n, i in enumerate(indexes):
                rows[i] = {name: out[n] for name, out in outputs.items()}
                if tokens is not None:
                    rows[i][TOKENS_KEY] = np.asarray(tokens[n])
                cache.complete(plan.keys[i], rows[i])
                if plan.keys[i] in owned:  # a later fail() must not touch a new owner's claim
                    owned.remove(plan.keys[i])
            return None

        done: Response | None = await run_in_pool(
            None if run_here else state.executor, "infer_wait", timings_ms, infer_rows
        )
        return done

    # Step 1: compute the rows that no request has claimed yet (the rows that this one owns).
    # Step 2: wait for the rows that other requests compute.
    # Step 3: if the owner of a row failed, compute that row here.
    # A failure at any step releases the claims of this request, so no waiter hangs.
    try:
        check_timeout(state, deadline)
        if plan.feeds is not None:
            done = await compute(plan.feeds, plan.found.missing, plan.tokens)
            if done is not None:
                return done
        # A waiter whose owner failed computes the row itself. It never sees the owner's error.
        retry: list[int] = []
        for i, future in plan.found.pending.items():
            try:
                rows[i] = await await_row(future)
                served += 1
            except Exception:  # noqa: BLE001 - any failure of the owner means "compute it here"
                retry.append(i)
        if retry:
            subset = _subset(plan.request, plan.arrays, retry)
            feeds = await run_in_pool(
                pool,
                "prep_wait",
                timings_ms,
                lambda: prepare_feeds(state, subset, skip_batch_axis=True, count_tokens=True)[0],
            )
            done = await compute(feeds, retry, subset.tokens)
            if done is not None:
                return done
    except BaseException:
        cache.fail(owned)
        raise

    def finish() -> Response:
        names = [n for n in rows[0] if n != TOKENS_KEY]
        arrays = {n: np.stack([rows[i][n] for i in range(len(plan.keys))]) for n in names}
        tokens = None
        if with_tokens and plan.request.text is not None:
            tokens = [int(rows[i][TOKENS_KEY]) for i in range(len(plan.keys))]
        if encode is not None:
            return encode(arrays, plan.request, tokens, None)
        return encode_response(
            state, arrays, plan.request.encoding, timings_ms, accept_safetensors, None
        )

    response: Response = await run_in_pool(pool, "encode_wait", timings_ms, finish)
    computed = len(plan.keys) - served
    response.headers[settings.CACHE_HEADER] = (
        "hit" if computed == 0 else "miss" if served == 0 else "partial"
    )
    return response
