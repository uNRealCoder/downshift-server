"""The `--pooling` choices. This module uses only the standard library. The CLI can therefore
parse the flag without an import of torch. The adapters can use the names without a dependency
on the serve package."""

from enum import StrEnum


class PoolingChoice(StrEnum):
    """--pooling: how the token vectors of an encoder-only repo become one embedding for each
    text. "none" serves the token vectors as they are. It ignores each recipe that the repo
    declares."""

    mean = "mean"
    cls = "cls"
    maximum = "max"  # not `max`, because that would hide the builtin inside the class body
    mean_sqrt_len = "mean_sqrt_len"
    lasttoken = "lasttoken"
    weightedmean = "weightedmean"
    none = "none"
