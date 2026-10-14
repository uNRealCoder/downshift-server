"""The result cache: the output of one row of the batch axis, by the key of its input.

It keeps at most `max_rows` rows. It does not count bytes. The memory of one row is the size of
the output of one row, and the user sets the row count to fit it.

Three rules keep it safe and useful:

- A row is stored on its second sighting (a "doorkeeper", as in TinyLFU). A bulk load of rows
  that come one time cannot push the hot rows out.
- Two requests never compute the same row. The first request claims it ("single-flight"). The
  others wait for its result.
- It holds only plain numpy arrays. A cached row is a copy, so it never keeps the buffer of a
  whole batch alive.
"""

import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

Row = dict[str, np.ndarray]  # output name -> the array of this row, with no batch axis

_MIN_DOORKEEPER = 1024


@dataclass
class Lookup:
    """The answer for the rows of one request. `hits` and `pending` map a row index to its
    result (or to a Future of it). `missing` lists the rows that this request must compute. It
    has claimed them: it must call complete() or fail() for each of them."""

    hits: dict[int, Row] = field(default_factory=dict)
    pending: dict[int, "Future[Row]"] = field(default_factory=dict)
    missing: list[int] = field(default_factory=list)


class ResultCache:
    def __init__(self, max_rows: int) -> None:
        self.max_rows = max_rows
        self._rows: OrderedDict[bytes, Row] = OrderedDict()
        self._seen: set[bytes] = set()  # the doorkeeper
        self._seen_limit = max(_MIN_DOORKEEPER, 10 * max_rows)
        self._flying: dict[bytes, Future[Row]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def lookup(self, keys: list[bytes]) -> Lookup:
        """Sort the rows of a request into hits, rows in flight and rows to compute. All of it
        happens in one locked step, so a row has at most one owner."""
        found = Lookup()
        with self._lock:
            for i, key in enumerate(keys):
                row = self._rows.get(key)
                if row is not None:
                    self._rows.move_to_end(key)
                    found.hits[i] = row
                elif key in self._flying:
                    found.pending[i] = self._flying[key]
                else:
                    self._flying[key] = Future()
                    found.missing.append(i)
            self.hits += len(found.hits) + len(found.pending)
            self.misses += len(found.missing)
        return found

    def complete(self, key: bytes, row: Row) -> None:
        """The owner has computed a row. Give it to the waiters, and store it if this is its
        second sighting. `key` may have no owner (a request that skipped the lookup)."""
        row = {name: arr.copy() for name, arr in row.items()}
        with self._lock:
            future = self._flying.pop(key, None)
            if key in self._seen or key in self._rows:
                self._seen.discard(key)
                self._rows[key] = row
                self._rows.move_to_end(key)
                while len(self._rows) > self.max_rows:
                    self._rows.popitem(last=False)
            else:
                if len(self._seen) >= self._seen_limit:
                    self._seen.clear()
                self._seen.add(key)
        if future is not None:
            future.set_result(row)

    def fail(self, keys: list[bytes], exc: BaseException) -> None:
        """The owner could not compute these rows. The waiters get the same error."""
        with self._lock:
            futures = [f for f in (self._flying.pop(key, None) for key in keys) if f is not None]
        for future in futures:
            future.set_exception(exc)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_rows": self.max_rows,
                "entries": len(self._rows),
                "hits": self.hits,
                "misses": self.misses,
            }


def _digest(kind: bytes) -> hashlib.blake2b:
    return hashlib.blake2b(kind, digest_size=16)


def text_keys(prompt: str, rows: list[str]) -> list[bytes]:
    """One key for each text row. The key covers the prompt and the row, and it is made before
    the tokenizer runs. A hit therefore also skips the tokenization."""
    head = prompt.encode()
    keys = []
    for row in rows:
        h = _digest(b"T")
        h.update(len(head).to_bytes(4, "big") + head)
        h.update(row.encode())
        keys.append(h.digest())
    return keys


def tensor_keys(arrays: dict[str, np.ndarray], rows: int) -> list[bytes]:
    """One key for each row of the batch axis. The key covers the name, the dtype, the shape and
    the bytes of each input row. It comes from the arrays, so the spaces and the number format of
    the JSON body do not change it."""
    keys = []
    for i in range(rows):
        h = _digest(b"N")
        for name, arr in arrays.items():
            row = np.ascontiguousarray(arr[i])
            h.update(f"{name}|{row.dtype.name}|{row.shape}|".encode())
            h.update(row.data)
        keys.append(h.digest())
    return keys
