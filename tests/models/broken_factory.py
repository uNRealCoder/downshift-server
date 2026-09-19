"""Not an export hazard fixture: raises as soon as it's instantiated. Used by CLI tests that
need a genuine, unexpected crash (exit code 5) rather than a usage error (exit code 4).
"""


def make_model():
    raise RuntimeError("boom: this factory always raises")
