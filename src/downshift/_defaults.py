"""Defaults that the library functions need at import time. This module reads no environment
variable, so `import downshift` never fails on a bad DOWNSHIFT_* value."""

# The verification samples for check(), export(), intake(), build_verdict() and ServeOptions.samples.
DEFAULT_SAMPLES = 8
