"""The last step of the example-input ladder: an error when nobody can guess the inputs."""

import pytest

from downshift.core.verdict import prepare_model
from tests.models import scatter_include_self_false


def test_prepare_model_raises_when_nothing_can_guess_inputs():
    model = scatter_include_self_false.make_model()
    with pytest.raises(ValueError, match="Cannot find example inputs"):
        prepare_model(model, None, "generic")
