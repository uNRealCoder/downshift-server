"""The request path of /predict and /predict/graph, from the raw body to the response: parse,
convert and check (prep pool), infer and encode (inference pool), and the result cache.

To read this file, go in this order:

1. `run_predict`: the entry point. It picks one of three paths: inline (all on the event
   loop), cached (`run_cached`), or the plain two-hop path.
2. `parse_request`, `prepare_feeds`, `input_violation`: decode the body, make the arrays, check them.
   Each client error is a 400 here, so the model never sees a bad input.
3. `run_inference`, `encode_response`: the backend call, and the JSON, base64 or safetensors body.
4. `run_cached`, `cache_prepare`, `CachePlan`: the same path with the result cache.
"""

import logging

import numpy as np
from fastapi import Response

from downshift.entities.requests import ParsedRequest
from downshift.entities.serving import ServingState
from downshift.entities.signatures import Encode, GraphLayout, Parse
from downshift.serve.cached import run_cached
from downshift.serve.encode import encode_response
from downshift.serve.feeds import prepare_feeds
from downshift.serve.parse import parse_request
from downshift.serve.stages import check_timeout, run_in_pool, run_inference

logger = logging.getLogger("downshift.serve")


async def run_predict(
    state: ServingState,
    body: bytes,
    *,
    graph: bool,
    binary: bool,
    inline: bool,
    deadline: float | None,
    timings_ms: dict[str, float],
    accept_safetensors: bool,
    parse: Parse | None,
    encode: Encode | None,
    with_tokens: bool,
    use_cache: bool = True,
) -> Response:
    """One admitted predict, from its raw body, in two hops: parse and prepare on the prep
    pool, then infer and encode on an inference thread. The event loop does neither, so
    /health and /ready never wait.

    `binary` is a safetensors body. `inline` (--execution inline, a small JSON body) does all
    of it on the event loop, except tokenization. `timings_ms` collects the milliseconds of
    each stage for the access log and the optional Server-Timing header.

    `parse` (None: decode `body` as a /predict body) builds the ParsedRequest, and `encode` (None: the
    /predict response) builds the response. /v1/embeddings sets both and has the same admission,
    pools, timeout, cache and row checks. `with_tokens` gives `encode` the token count of each
    text row. `deadline` is a time.monotonic value (None for no limit).
    """
    # Path 1, inline: parse here, on the event loop (a small JSON body only).
    parsed: ParsedRequest | None = None
    if inline:
        parsed = (
            parse() if parse is not None else parse_request(state, body, graph, binary, timings_ms)
        )
    # A text request still goes to the prep pool, because the tokenizer can be slow.
    run_here = parsed is not None and parsed.text is None
    # Path 2, cached: the result cache never serves /predict/graph.
    if state.result_cache is not None and not graph:
        return await run_cached(
            state,
            state.result_cache,
            parsed,
            body=body,
            graph=graph,
            binary=binary,
            accept_safetensors=accept_safetensors,
            parse=parse,
            encode=encode,
            with_tokens=with_tokens,
            run_here=run_here,
            deadline=deadline,
            timings_ms=timings_ms,
            use_lookup=use_cache,
        )

    # Path 3, plain: hop 1 (prep pool) parses and checks, hop 2 (inference pool) infers and
    # encodes. The event loop does neither, so /health and /ready never wait.
    def prepare() -> tuple[ParsedRequest, dict[str, np.ndarray], GraphLayout | None]:
        request = parsed or (
            parse() if parse is not None else parse_request(state, body, graph, binary, timings_ms)
        )
        feeds, prep_ms, layout = prepare_feeds(state, request, count_tokens=with_tokens)
        timings_ms["prep"] = round(prep_ms, 2)
        return request, feeds, layout

    request, feeds, layout = await run_in_pool(
        None if run_here else state.prep_executor, "prep_wait", timings_ms, prepare
    )

    check_timeout(state, deadline)

    def infer_and_encode() -> Response:
        # Again: the request may have waited behind a running inference.
        check_timeout(state, deadline)
        outputs, infer_ms = run_inference(state, feeds)
        timings_ms["infer"] = round(infer_ms, 2)
        tokens = request.tokens if with_tokens else None
        if encode is not None:
            return encode(outputs, request, tokens, layout)
        return encode_response(
            state, outputs, request.encoding, timings_ms, accept_safetensors, layout
        )

    response: Response = await run_in_pool(
        None if run_here else state.executor, "infer_wait", timings_ms, infer_and_encode
    )
    return response


__all__ = ["run_predict"]
