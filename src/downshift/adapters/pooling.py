"""The `--pooling` choices. Stdlib only, so the CLI can parse the flag without importing torch,
and the adapters can use the names without depending on the serve package."""

from enum import StrEnum


class PoolingChoice(StrEnum):
    """--pooling: how an encoder-only repo's token vectors become one embedding per text.
    "none" serves the token vectors as they are, ignoring any recipe the repo declares."""

    mean = "mean"
    cls = "cls"
    maximum = "max"  # not `max`: that would shadow the builtin inside the class body
    mean_sqrt_len = "mean_sqrt_len"
    lasttoken = "lasttoken"
    weightedmean = "weightedmean"
    none = "none"
