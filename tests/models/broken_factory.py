"""Not an export hazard fixture: raises as soon as it is instantiated. CLI tests use it when they
need a real, unexpected crash (exit code 5) and not a usage error (exit code 4).
"""


def make_model():
    raise RuntimeError("boom: this factory always raises")
