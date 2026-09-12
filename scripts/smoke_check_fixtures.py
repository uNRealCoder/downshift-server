"""Throwaway: import every fixture and run one forward pass. Not a committed pytest test."""

import importlib
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


def main() -> None:
    for name in FIXTURES:
        try:
            module = importlib.import_module(f"tests.models.{name}")
            out = module.make_model()(*module.make_inputs())
            shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"OK   {name:32s} output shape={shape}")
        except Exception as e:
            print(f"FAIL {name:32s} {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
