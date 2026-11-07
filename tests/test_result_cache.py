"""The result cache (serve/result_cache.py): count limit, doorkeeper, single-flight."""

import threading

import numpy as np
import pytest

from downshift.serve.result_cache import ResultCache


def _row(value: float, size: int = 4):
    return {"output_0": np.full(size, value, dtype=np.float32)}


def _put(cache: ResultCache, key: bytes, row) -> None:
    cache.lookup([key])  # the owner claims the row
    cache.complete(key, row)


def _key(i: int) -> bytes:
    return i.to_bytes(8, "big")


def test_a_row_is_stored_on_its_second_sighting():
    cache = ResultCache(10)

    _put(cache, b"k", _row(1))
    assert len(cache) == 0, "the first sighting only enters the doorkeeper"
    _put(cache, b"k", _row(1))

    found = cache.lookup([b"k"])
    assert list(found.hits) == [0]
    assert found.hits[0]["output_0"].tolist() == [1.0] * 4
    assert found.missing == []


def test_the_limit_counts_rows_and_the_oldest_row_goes_first():
    cache = ResultCache(2)
    for i in range(3):
        _put(cache, _key(i), _row(i))
        _put(cache, _key(i), _row(i))

    found = cache.lookup([_key(0), _key(1), _key(2)])

    assert sorted(found.hits) == [1, 2]
    assert found.missing == [0]


def test_a_hit_moves_the_row_to_the_back_of_the_queue():
    cache = ResultCache(2)
    for i in (0, 1):
        _put(cache, _key(i), _row(i))
        _put(cache, _key(i), _row(i))
    cache.lookup([_key(0)])  # key 0 is now the newest
    _put(cache, _key(2), _row(2))
    _put(cache, _key(2), _row(2))

    found = cache.lookup([_key(0), _key(1)])

    assert sorted(found.hits) == [0]


def test_a_bulk_load_of_single_rows_does_not_push_out_the_hot_rows():
    cache = ResultCache(1000)
    for i in range(1000):
        _put(cache, _key(i), _row(i))
        _put(cache, _key(i), _row(i))
    for i in range(1000, 101_000):
        _put(cache, _key(i), _row(i))  # each of these comes one time

    found = cache.lookup([_key(i) for i in range(1000)])

    assert len(found.hits) == 1000
    assert len(cache) == 1000


def test_the_doorkeeper_has_a_limit_and_resets():
    cache = ResultCache(1)
    limit = cache._seen_limit
    for i in range(limit + 5):
        _put(cache, _key(i), _row(i))

    assert len(cache._seen) <= limit


def test_a_stored_row_is_a_copy():
    cache = ResultCache(4)
    batch = np.arange(8, dtype=np.float32).reshape(2, 4)
    for _ in range(2):
        cache.lookup([b"k"])
        cache.complete(b"k", {"output_0": batch[0]})
    batch[:] = -1

    assert cache.lookup([b"k"]).hits[0]["output_0"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_two_requests_never_compute_the_same_row():
    cache = ResultCache(4)

    first = cache.lookup([b"k"])
    second = cache.lookup([b"k"])

    assert first.missing == [0]
    assert second.missing == []
    cache.complete(b"k", _row(7))
    assert second.pending[0].result()["output_0"].tolist() == [7.0] * 4


def test_the_same_row_twice_in_one_request_has_one_owner():
    cache = ResultCache(4)

    found = cache.lookup([b"k", b"k"])

    assert found.missing == [0]
    assert list(found.pending) == [1]


def test_single_flight_across_threads():
    cache = ResultCache(4)
    owners = []
    results = []
    gate = threading.Barrier(8)
    looked = threading.Barrier(8)

    def worker():
        gate.wait()
        found = cache.lookup([b"k"])
        looked.wait()  # every thread has looked before the owner completes
        if found.missing:
            owners.append(1)
            cache.complete(b"k", _row(3))
            results.append(3.0)
        elif found.pending:
            results.append(float(found.pending[0].result(timeout=5)["output_0"][0]))
        else:
            results.append(float(found.hits[0]["output_0"][0]))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(owners) == 1
    assert results == [3.0] * 8


def test_a_failure_goes_to_the_waiters_and_frees_the_row():
    cache = ResultCache(4)
    cache.lookup([b"k"])
    waiter = cache.lookup([b"k"]).pending[0]

    cache.fail([b"k"], RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        waiter.result()
    assert cache.lookup([b"k"]).missing == [0], "the next request may try again"


def test_complete_without_an_owner_still_counts_for_the_doorkeeper():
    cache = ResultCache(4)
    cache.complete(b"k", _row(1))
    cache.complete(b"k", _row(1))

    assert list(cache.lookup([b"k"]).hits) == [0]


def test_the_counters():
    cache = ResultCache(4)
    _put(cache, b"k", _row(1))
    _put(cache, b"k", _row(1))
    cache.lookup([b"k", b"other"])

    assert cache.stats() == {"max_rows": 4, "entries": 1, "hits": 1, "misses": 3}


def test_a_text_key_covers_the_prompt_and_the_row():
    from downshift.serve.result_cache import text_keys

    same = text_keys("", ["a", "b"])
    assert text_keys("", ["a", "b"]) == same
    assert len(set(same)) == 2
    assert text_keys("query: ", ["a"]) != text_keys("", ["a"])
    # The prompt and the row must not blur into each other.
    assert text_keys("ab", ["c"]) != text_keys("a", ["bc"])


def test_a_tensor_key_comes_from_the_array_and_not_from_the_json():
    from downshift.serve.result_cache import tensor_keys
    from downshift.serve.schemas import to_numpy

    spaced = to_numpy("x", [[1, 2.5], [3, 4]], "float32")
    compact = to_numpy("x", [[1.0, 2.50], [3.0, 4.0]], "float32")

    assert tensor_keys({"x": spaced}, 2) == tensor_keys({"x": compact}, 2)
    assert len(set(tensor_keys({"x": spaced}, 2))) == 2
    assert tensor_keys({"x": spaced.astype(np.float64)}, 2) != tensor_keys({"x": spaced}, 2)
    assert tensor_keys({"y": spaced}, 2) != tensor_keys({"x": spaced}, 2)


def test_a_text_key_and_a_tensor_key_never_meet():
    from downshift.serve.result_cache import tensor_keys, text_keys

    assert set(text_keys("", ["x"])).isdisjoint(tensor_keys({"x": np.zeros((1, 1))}, 1))
