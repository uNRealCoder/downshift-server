"""Throwaway: import every fixture and run one forward pass. Not a committed pytest test."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES = [
    "clean_mlp",
    "dynamic_batch_cnn",
    "data_dependent_branch",
    "custom_autograd",
    "tied_weights",
    "dict_input",
    "dropout_model",
    "scatter_include_self_false",
]

for name in FIXTURES:
    try:
        module = __import__(f"tests.models.{name}", fromlist=["make_model", "make_inputs"])
        model = module.make_model()
        inputs = module.make_inputs()
        out = model(*inputs)
        shape = out.shape if hasattr(out, "shape") else type(out)
        print(f"OK   {name:32s} output shape={shape}")
    except Exception as e:
        print(f"FAIL {name:32s} {type(e).__name__}: {e}")
        traceback.print_exc()
