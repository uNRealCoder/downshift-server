"""The pool hops of a predict: the time limit, the inference and the thread handoff."""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any

import numpy as np

from downshift.entities.serving import ServingState
from downshift.serve.backends import InferenceInputError
from downshift.serve.errors import ApiError


def check_timeout(state: ServingState, deadline: float | None) -> None:
    """A 503 if `deadline` (a time.monotonic value, None for no limit) has passed."""
    if deadline is not None and (now := time.monotonic()) > deadline:
        request_timeout = state.options.request_timeout
        waited = now - deadline + request_timeout
        raise ApiError(
            503,
            "request_timeout",
            f"request waited {waited:.1f}s in queue, past the {request_timeout:.1f}s "
            "--request-timeout",
        )


def run_inference(
    state: ServingState, feeds: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    """The backend call: (outputs, milliseconds). Only InferenceInputError is a 400. Any
    other error propagates to the 500 handler, which hides its text from the client."""
    start = time.perf_counter()
    try:
        outputs = state.backend.infer(feeds)
    except InferenceInputError as exc:
        raise ApiError(400, "invalid_input", str(exc)) from exc
    return outputs, (time.perf_counter() - start) * 1000


async def run_in_pool(
    executor: Executor | None,
    wait_key: str,
    timings_ms: dict[str, float],
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """fn(*args) in `executor` with the contextvars (so log lines keep the request ID), or on
    the event loop if `executor` is None. timings_ms[wait_key] is the queue time."""
    if executor is None:
        timings_ms[wait_key] = 0.0
        return fn(*args)
    submitted = time.perf_counter()

    def call() -> Any:
        timings_ms[wait_key] = round((time.perf_counter() - submitted) * 1000, 2)
        return fn(*args)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, contextvars.copy_context().run, call)
