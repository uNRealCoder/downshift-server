"""The type aliases of the request path."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from fastapi import Response

from downshift.entities.requests import ParsedRequest

# The (node_counts, edge_counts) of each graph in a graph batch. The response is split by them.
GraphLayout = tuple[list[int], list[int]]


Parse = Callable[[], ParsedRequest]


# (outputs, request, tokens, layout) -> the response. `tokens` is the count of tokens for each
# row, only for a text request and only if run_predict was asked for it. `layout` is for a graph
# batch.
Encode = Callable[
    [dict[str, np.ndarray], ParsedRequest, list[int] | None, GraphLayout | None], Response
]
