"""Entities: requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from downshift.entities.cache import Lookup
from downshift.entities.options import OutputEncoding, RequestOutputEncoding


@dataclass
class ParsedRequest:
    """A predict body, decoded and validated: what prepare_feeds takes, plus the encoding."""

    inputs: dict[str, Any] = field(default_factory=dict)
    text: list[str] | None = None
    prompt: str = ""
    tensors: dict[str, np.ndarray] | None = None
    graphs: list[dict[str, Any]] | None = None
    encoding: OutputEncoding | RequestOutputEncoding | None = None
    dimensions: int | None = None  # /v1/embeddings: keep the first values of each vector
    tokens: list[int] | None = None  # set by prepare_feeds for a text request


@dataclass
class CachePlan:
    """The prep result of a cached request: its row keys, the lookup, and the feeds of the rows
    that this request must compute."""

    request: ParsedRequest
    arrays: dict[str, np.ndarray] | None  # the full input arrays of a tensor request
    keys: list[bytes]
    found: Lookup
    feeds: dict[str, np.ndarray] | None  # for the missing rows only. None if there are none
    owned: list[bytes] = field(default_factory=list)  # the keys that this request claimed
    passthrough: bool = False  # an empty batch: run the model once and skip the cache
    tokens: list[int] | None = None  # the token count of each row of `feeds`, for a text request
