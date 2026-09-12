"""Adapter registry, detection, and the per-family prepare()/vary_fn behaviour."""

import inspect

import pytest
import torch

from downshift.adapters import _flatten, generic, hf, pyg, registry
from downshift.export.verdict import prepare_model
from tests.models import (
    clean_mlp,
    dict_input,
    dynamic_batch_cnn,
    gnn_gcn,
    scatter_include_self_false,
    tiny_bert,
)

# --- registry ---------------------------------------------------------------------------


def test_available_lists_builtins_with_generic_last():
    names = list(registry.available())
    assert {"generic", "pyg", "hf"} <= set(names)
    assert names[-1] == "generic"


def test_get_unknown_adapter_lists_available_names():
    with pytest.raises(KeyError, match="nope") as excinfo:
        registry.get("nope")
    assert "generic" in str(excinfo.value)


class _FakeAdapter:
    name = "fake"
    family = "fake"

    def matches(self, model, example_inputs):
        return True

    def example_inputs(self, model):
        return None

    def prepare(self, model, example_inputs):
        raise NotImplementedError


class _FakeEntryPoint:
    def __init__(self, loader):
        self._loader = loader

    def load(self):
        return self._loader()


def test_entry_point_adapter_is_discovered_and_wins_over_generic(monkeypatch):
    fake = _FakeAdapter()
    monkeypatch.setattr(registry, "entry_points", lambda group: [_FakeEntryPoint(lambda: fake)])

    available = registry.available()
    assert available["fake"] is fake
    assert list(available)[-1] == "generic"
    assert registry.detect(clean_mlp.make_model(), clean_mlp.make_inputs()) is fake


def test_entry_point_with_missing_dependency_is_skipped(monkeypatch):
    def broken():
        raise ImportError("optional dependency not installed")

    monkeypatch.setattr(registry, "entry_points", lambda group: [_FakeEntryPoint(broken)])

    names = list(registry.available())
    assert "fake" not in names
    assert names[-1] == "generic"


@pytest.mark.parametrize(
    ("make_model", "make_inputs", "expected"),
    [
        (clean_mlp.make_model, clean_mlp.make_inputs, "generic"),
        (gnn_gcn.make_model, gnn_gcn.make_inputs, "pyg"),
        (tiny_bert.make_model, tiny_bert.make_inputs, "hf"),
        (tiny_bert.make_model, lambda: None, "hf"),
        # No inputs at all: PyG is detected structurally from MessagePassing layers.
        (gnn_gcn.make_model, lambda: None, "pyg"),
    ],
    ids=["mlp", "pyg-data", "bert", "bert-no-inputs", "gcn-no-inputs"],
)
def test_detect_picks_the_family_adapter(make_model, make_inputs, expected):
    assert registry.detect(make_model(), make_inputs()).name == expected


# --- generic adapter --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_model", "expected_shape"),
    [(clean_mlp.make_model, (1, 16)), (dynamic_batch_cnn.make_model, (1, 3, 32, 32))],
    ids=["linear", "conv2d"],
)
def test_generic_guesses_single_tensor_from_first_layer(make_model, expected_shape):
    guess = generic.ADAPTER.example_inputs(make_model())
    assert guess is not None
    assert len(guess) == 1
    assert tuple(guess[0].shape) == expected_shape


def test_generic_does_not_guess_for_multi_argument_forward():
    assert generic.ADAPTER.example_inputs(scatter_include_self_false.make_model()) is None


def test_generic_flattens_dataclass_input_into_named_tensors():
    prepared = generic.ADAPTER.prepare(dict_input.make_model(), dict_input.make_inputs())

    assert prepared.input_names == ("x", "mask")
    assert all(isinstance(t, torch.Tensor) for t in prepared.inputs)
    assert list(inspect.signature(prepared.model.forward).parameters) == ["x", "mask"]


@pytest.mark.parametrize(
    ("module", "expected"),
    [(gnn_gcn, {"x": [0], "edge_index": [1]}), (clean_mlp, {"x": [0]})],
    ids=["gcn", "mlp"],
)
def test_prepared_dynamic_dims(module, expected):
    prepared = prepare_model(module.make_model(), module.make_inputs())
    assert prepared.dynamic_dims == expected


# --- flatten shim -----------------------------------------------------------------------


def test_shim_replaces_non_identifier_field_names_positionally():
    shim_class = _flatten.build_shim_class(("x", "edge index"))
    assert list(inspect.signature(shim_class.forward).parameters) == ["self", "x", "t1"]


@pytest.mark.parametrize("training", [False, True])
def test_shim_inherits_train_eval_mode(training):
    model = clean_mlp.make_model().train(training)
    shim = _flatten.build_shim_class(("x",))(model, lambda fields: fields["x"], ("x",))
    assert shim.training is training


# --- vary functions ---------------------------------------------------------------------


def test_pyg_vary_fn_never_references_missing_nodes():
    (data,) = gnn_gcn.make_inputs(num_nodes=6, num_edges=10)
    base = (data.x, data.edge_index)
    vary = pyg.make_vary_fn(base, ("x", "edge_index"))

    assert vary(0) is base
    for i in range(1, 21):
        x, edge_index = vary(i)
        assert edge_index.shape[0] == 2
        assert x.shape[1] == data.x.shape[1]
        assert int(edge_index.max()) < x.shape[0]


def test_hf_vary_fn_keeps_ids_in_vocab_and_mask_aligned():
    base = tiny_bert.make_inputs()
    vary = hf.make_vary_fn(base, vocab_size=100)

    assert vary(0) is base
    for i in range(1, 21):
        ids, mask = vary(i)
        assert ids.dtype == base[0].dtype
        assert 0 <= int(ids.min()) and int(ids.max()) < 100
        assert mask.shape == ids.shape
        assert bool((mask == 1).all())
