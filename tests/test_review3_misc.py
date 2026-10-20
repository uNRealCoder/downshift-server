"""Small parts that we added in the 0.4.0 review round: the JSON handoffs of axis bounds and
ServeArgs, the RoBERTa position limits, the load-phase reporter, the capture of stderr off the
main thread, the new limit defaults and the plain-text banner."""

import pytest

from downshift.core.axes import DimBound
from downshift.loading import LoadSpec, load_model
from downshift.serve.engine import ServeOptions, prepare_serving

BOUNDS = {
    "input_ids": {0: DimBound("batch", 1, 4096), 1: DimBound("seq", 1, 510)},
    "attention_mask": {0: DimBound("batch", 1, 4096), 1: DimBound("seq", 1, 510)},
}


# --- axis bounds and ServeArgs hand-off ------------------------------------------------------


# --- position_limit --------------------------------------------------------------------------


# --- load-phase reporting --------------------------------------------------------------------


# --- capture off the main thread -------------------------------------------------------------


# --- limit defaults --------------------------------------------------------------------------


# --- the report logger's plain text ----------------------------------------------------------


@pytest.fixture(scope="module")
def mlp_state():
    return prepare_serving(
        load_model(LoadSpec("tests.models.clean_mlp:make_model")), ServeOptions(warmup=1)
    )
