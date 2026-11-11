"""The function types of the request path: how a route gives `run_predict` its own parser and
response encoder. Used by serve/predict.py, serve/cached.py, serve/feeds.py and serve/encode.py.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from fastapi import Response

from downshift.entities.requests import ParsedRequest

# The (node_counts, edge_counts) of each graph in a graph batch.
GraphLayout = tuple[list[int], list[int]]


# A function that builds the ParsedRequest. /v1/embeddings gives its own to `run_predict`.
Parse = Callable[[], ParsedRequest]


# A function that builds the response from the outputs. /v1/embeddings gives its own.
Encode = Callable[
    [dict[str, np.ndarray], ParsedRequest, list[int] | None, GraphLayout | None], Response
]
