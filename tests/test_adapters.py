"""Adapter registry, detection, and the per-family prepare()/vary_fn behaviour."""

import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import SAGEConv

from downshift.adapters import _flatten, generic, hf, pyg, registry
from downshift.export.verdict import prepare_model
from downshift.loading import LoadError
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


def test_load_spec_returns_none_when_module_missing():
    assert registry._load_spec("no.such.module:ADAPTER") is None


def test_get_returns_the_matching_adapter():
    assert registry.get("generic") is generic.ADAPTER


# --- custom adapters loaded from a .py file ----------------------------------------------

_CUSTOM_ADAPTER_INSTANCE = '''
from downshift.adapters.base import Prepared

class MyAdapter:
    name = "custom"
    family = "custom-family"

    def matches(self, model, example_inputs):
        return True

    def example_inputs(self, model):
        return None

    def prepare(self, model, example_inputs):
        return Prepared(
            model=model,
            inputs=example_inputs,
            input_names=("x",),
            dynamic_shapes=(None,),
            vary_fn=None,
            family=self.family,
        )

ADAPTER = MyAdapter()
'''

_CUSTOM_ADAPTER_CLASS_ONLY = _CUSTOM_ADAPTER_INSTANCE.replace('ADAPTER = MyAdapter()\n', "")

_NOT_AN_ADAPTER = "NOT_AN_ADAPTER = object()\n"


def test_get_loads_custom_adapter_from_py_file_default_attr(tmp_path):
    path = tmp_path / "my_adapter.py"
    path.write_text(_CUSTOM_ADAPTER_INSTANCE)

    adapter = registry.get(str(path))
    assert adapter.name == "custom"
    assert adapter.family == "custom-family"


def test_get_loads_custom_adapter_from_py_file_with_explicit_attr(tmp_path):
    path = tmp_path / "my_adapter.py"
    path.write_text(_CUSTOM_ADAPTER_INSTANCE)

    adapter = registry.get(f"{path}:ADAPTER")
    assert adapter.name == "custom"


def test_get_loads_custom_adapter_class_and_instantiates_it(tmp_path):
    path = tmp_path / "my_adapter.py"
    path.write_text(_CUSTOM_ADAPTER_CLASS_ONLY)

    adapter = registry.get(f"{path}:MyAdapter")
    assert adapter.name == "custom"


def test_get_custom_adapter_missing_file_raises_load_error(tmp_path):
    missing = tmp_path / "nope.py"
    with pytest.raises(LoadError, match="does not exist"):
        registry.get(str(missing))


def test_get_custom_adapter_missing_attr_raises_load_error(tmp_path):
    path = tmp_path / "my_adapter.py"
    path.write_text(_CUSTOM_ADAPTER_INSTANCE)

    with pytest.raises(LoadError, match="no attribute"):
        registry.get(f"{path}:NOPE")


def test_get_custom_adapter_wrong_shape_raises_load_error(tmp_path):
    path = tmp_path / "my_adapter.py"
    path.write_text(_NOT_AN_ADAPTER)

    with pytest.raises(LoadError, match="not an Adapter"):
        registry.get(f"{path}:NOT_AN_ADAPTER")


def test_prepare_model_accepts_custom_adapter_file_path(tmp_path):
    path = tmp_path / "my_adapter.py"
    path.write_text(_CUSTOM_ADAPTER_INSTANCE)

    prepared = prepare_model(clean_mlp.make_model(), clean_mlp.make_inputs(), adapter=str(path))
    assert prepared.family == "custom-family"


def test_detect_raises_when_no_adapter_matches(monkeypatch):
    monkeypatch.setattr(registry, "available", lambda: {})
    with pytest.raises(RuntimeError, match="no adapter matched"):
        registry.detect(clean_mlp.make_model(), clean_mlp.make_inputs())


def test_prepare_model_accepts_adapter_name_as_string():
    prepared = prepare_model(clean_mlp.make_model(), clean_mlp.make_inputs(), adapter="generic")
    assert prepared.family == "generic-torch"


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


class _Conv1dOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 4, 3)

    def forward(self, x):
        return self.conv(x)


class _Conv3dOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 4, 3)

    def forward(self, x):
        return self.conv(x)


class _EmbeddingOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 4)

    def forward(self, x):
        return self.emb(x)


class _NoGuessableLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = torch.nn.LSTM(4, 4)

    def forward(self, x):
        return self.rnn(x)


@pytest.mark.parametrize(
    ("model_cls", "expected_shape"),
    [(_Conv1dOnly, (1, 3, generic._GUESS_SPATIAL)), (_Conv3dOnly, (1, 3, 8, 8, 8))],
    ids=["conv1d", "conv3d"],
)
def test_generic_guesses_conv1d_and_conv3d(model_cls, expected_shape):
    guess = generic.ADAPTER.example_inputs(model_cls())
    assert guess is not None
    assert tuple(guess[0].shape) == expected_shape


def test_generic_guesses_embedding_input():
    guess = generic.ADAPTER.example_inputs(_EmbeddingOnly())
    assert guess is not None
    assert tuple(guess[0].shape) == (1, 8)
    assert guess[0].dtype == torch.int64


def test_generic_guess_returns_none_for_unrecognized_layer():
    assert generic.ADAPTER.example_inputs(_NoGuessableLayer()) is None


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


@dataclasses.dataclass
class _MixedFields:
    x: torch.Tensor
    flag: bool


def test_flatten_dataclass_declines_when_a_field_is_not_a_tensor():
    model = clean_mlp.make_model()
    data = _MixedFields(x=torch.randn(1, 16), flag=True)
    assert generic._flatten_dataclass(model, (data,)) is None


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


def test_pyg_example_inputs_guesses_from_message_passing_layer():
    guess = pyg.ADAPTER.example_inputs(gnn_gcn.make_model())
    assert guess is not None
    assert isinstance(guess[0], PyGData)
    assert guess[0].x.shape[1] == 8


def test_pyg_example_inputs_none_without_message_passing_layer():
    assert pyg.ADAPTER.example_inputs(clean_mlp.make_model()) is None


class _BipartiteSAGE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = SAGEConv((8, 8), 4)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def test_pyg_first_in_channels_unwraps_a_tuple_in_channels():
    guess = pyg.ADAPTER.example_inputs(_BipartiteSAGE())
    assert guess is not None
    assert guess[0].x.shape[1] == 8


def test_pyg_prepare_includes_edge_attr_when_present_on_the_input():
    model = gnn_gcn.make_model()
    data = PyGData(
        x=torch.randn(6, 8), edge_index=torch.randint(0, 6, (2, 10)), edge_attr=torch.randn(10, 3)
    )
    prepared = pyg.ADAPTER.prepare(model, (data,))
    assert prepared.input_names == ("x", "edge_index", "edge_attr")
    assert len(prepared.inputs) == 3


def test_pyg_vary_fn_resizes_edge_attr_with_the_edge_count():
    x = torch.randn(6, 8)
    edge_index = torch.randint(0, 6, (2, 10))
    edge_attr = torch.randn(10, 3)
    base = (x, edge_index, edge_attr)
    vary = pyg.make_vary_fn(base, ("x", "edge_index", "edge_attr"))

    assert vary(0) is base
    for i in range(1, 11):
        _, sample_edge_index, sample_edge_attr = vary(i)
        assert sample_edge_attr.shape == (sample_edge_index.shape[1], edge_attr.shape[1])


def test_hf_example_inputs_none_without_vocab_size():
    assert hf.HFAdapter().example_inputs(SimpleNamespace()) is None


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
