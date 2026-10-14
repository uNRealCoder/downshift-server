"""The result cache on the serving path: boot rules, then the request path."""

import pytest

from downshift.loading import LoadSpec, load_model
from downshift.serve.engine import prepare_serving
from downshift.serve.options import ServeOptions
from downshift.settings import DEFAULT_RESULT_CACHE_ROWS


def _state(name: str, **opts):
    loaded = load_model(LoadSpec(f"tests.models.{name}:make_model"))
    return prepare_serving(loaded, ServeOptions(warmup=1, **opts))


def test_the_cache_is_on_by_default_for_independent_rows():
    state = _state("clean_mlp")

    assert state.result_cache is not None
    assert state.result_cache.max_rows == DEFAULT_RESULT_CACHE_ROWS == 100


def test_zero_rows_turns_the_cache_off():
    state = _state("clean_mlp", result_cache_rows=0)

    assert state.result_cache is None
    assert state.cache_off_reason is None


def test_the_default_is_off_for_a_model_with_dependent_rows():
    state = _state("cross_row")

    assert state.result_cache is None
    assert state.cache_off_reason is not None
    assert "when it runs with other rows" in state.cache_off_reason


def test_a_set_size_on_a_model_with_dependent_rows_is_a_usage_error():
    with pytest.raises(ValueError, match=r"Remove --result-cache-rows, or set it to 0"):
        _state("cross_row", result_cache_rows=50)


def test_the_size_is_free_for_independent_rows():
    assert _state("clean_mlp", result_cache_rows=7).result_cache.max_rows == 7  # type: ignore[union-attr]


def test_metadata_reports_the_cache(mlp_client, mlp_state):
    cache = mlp_client.get("/metadata").json()["cache"]

    assert cache["max_rows"] == 100
    assert set(cache) == {"max_rows", "entries", "hits", "misses"}
    assert mlp_state.result_cache is not None


def test_metadata_has_no_cache_when_it_is_off():
    from fastapi.testclient import TestClient

    from downshift.serve.app import build_app

    client = TestClient(build_app(_state("clean_mlp", result_cache_rows=0)))

    assert client.get("/metadata").json()["cache"] is None
