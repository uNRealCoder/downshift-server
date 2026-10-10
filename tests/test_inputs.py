"""synthesize(): the last step of the example-input ladder. Downshift reaches it when nobody can guess."""

import pytest

from downshift.adapters import generic
from downshift.core.inputs import synthesize
from tests.models import scatter_include_self_false


def test_synthesize_raises_when_nothing_can_guess_inputs():
    model = scatter_include_self_false.make_model()
    with pytest.raises(ValueError, match="Cannot find example inputs"):
        synthesize(model, generic.GenericAdapter(), None)
