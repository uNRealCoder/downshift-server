"""The metrics of one worker process, and their Prometheus text (format 0.0.4).

`Metrics` holds only what the request path must count. The event loop writes it, so it has no
lock. The gauges and the cache rows are not counted for each request: `render` reads them from
the ServingState. Each label has a fixed set of values. A request value is never a label.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from downshift._version import __version__

if TYPE_CHECKING:
    from downshift.serve.engine import ServingState

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Seconds. The last bucket is +Inf.
BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

# The stage label -> the timings_ms keys of predict that it adds up.
_STAGE_KEYS = {
    "queue": ("prep_wait", "infer_wait", "encode_wait"),
    "prep": ("parse", "prep"),
    "infer": ("infer",),
    "encode": ("encode",),
}
# The error codes (serve/errors.py) that count as a rejected request.
_REJECT_REASONS = {
    "at_capacity": "capacity",
    "request_timeout": "timeout",
    "not_ready": "not_ready",
}
_CACHE_RESULTS = ("hit", "partial", "miss")


@dataclass
class Histogram:
    counts: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS) + 1))
    total: float = 0.0

    def observe(self, seconds: float) -> None:
        self.counts[bisect_left(BUCKETS, seconds)] += 1
        self.total += seconds


@dataclass
class Metrics:
    worker: str = "0"
    requests: dict[tuple[str, str], int] = field(default_factory=dict)  # (route, code)
    request_seconds: dict[str, Histogram] = field(default_factory=dict)  # by route
    stage_seconds: dict[str, Histogram] = field(default_factory=dict)  # by stage
    rejected: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in _REJECT_REASONS.values()}
    )
    cache_requests: dict[str, int] = field(
        default_factory=lambda: {result: 0 for result in _CACHE_RESULTS}
    )

    def record_request(self, route: str, status: int, seconds: float) -> None:
        key = (route, str(status))
        self.requests[key] = self.requests.get(key, 0) + 1
        _histogram(self.request_seconds, route).observe(seconds)

    def record_stages(self, timings_ms: dict[str, float]) -> None:
        for stage, keys in _STAGE_KEYS.items():
            milliseconds = 0.0
            seen = False
            for key in keys:
                if key in timings_ms:
                    milliseconds += timings_ms[key]
                    seen = True
            if seen:
                _histogram(self.stage_seconds, stage).observe(milliseconds / 1000)

    def record_rejected(self, code: str | None) -> None:
        reason = _REJECT_REASONS.get(code or "")
        if reason is not None:
            self.rejected[reason] += 1

    def record_cache_result(self, result: str | None) -> None:
        if result in _CACHE_RESULTS:
            self.cache_requests[result] += 1


def _histogram(series: dict[str, Histogram], key: str) -> Histogram:
    hist = series.get(key)
    if hist is None:
        hist = series[key] = Histogram()
    return hist


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: dict[str, object]) -> str:
    return "{" + ",".join(f'{name}="{_escape(value)}"' for name, value in pairs.items()) + "}"


def render(metrics: Metrics, state: ServingState | None) -> str:
    """The Prometheus text of `metrics`, with the gauges of `state`. `state` is None while the
    model loads."""
    worker = {"worker": metrics.worker}
    lines: list[str] = []

    def head(name: str, kind: str, text: str) -> None:
        lines.append(f"# HELP {name} {text}")
        lines.append(f"# TYPE {name} {kind}")

    def sample(name: str, value: float, **labels: object) -> None:
        lines.append(f"{name}{_labels({**labels, **worker})} {value}")

    def counter(name: str, text: str, label_names: tuple[str, ...], values: dict) -> None:
        head(name, "counter", text)
        for key, count in values.items():
            keys = key if isinstance(key, tuple) else (key,)
            sample(name, count, **dict(zip(label_names, keys, strict=True)))

    def histogram(name: str, text: str, label: str, values: dict[str, Histogram]) -> None:
        head(name, "histogram", text)
        for value, hist in values.items():
            labels = {label: value}
            running = 0
            for bound, count in zip((*BUCKETS, "+Inf"), hist.counts, strict=True):
                running += count
                sample(f"{name}_bucket", running, **labels, le=str(bound))
            sample(f"{name}_sum", hist.total, **labels)
            sample(f"{name}_count", running, **labels)

    counter(
        "downshift_requests_total",
        "HTTP requests, by route template and status code.",
        ("route", "code"),
        metrics.requests,
    )
    histogram(
        "downshift_request_seconds",
        "Seconds from the start of a request to the end of its response.",
        "route",
        metrics.request_seconds,
    )
    histogram(
        "downshift_stage_seconds",
        "Seconds of each stage of a predict: queue, prep, infer, encode.",
        "stage",
        metrics.stage_seconds,
    )
    counter(
        "downshift_rejected_total",
        "Requests that the server refused with a 503, by reason.",
        ("reason",),
        metrics.rejected,
    )
    counter(
        "downshift_cache_requests_total",
        "Predict requests, by result cache outcome.",
        ("result",),
        metrics.cache_requests,
    )

    cache = state.result_cache.stats() if state is not None and state.result_cache else None
    head("downshift_cache_rows_total", "counter", "Rows that the result cache found or missed.")
    if cache is not None:
        sample("downshift_cache_rows_total", cache["hits"], result="hit")
        sample("downshift_cache_rows_total", cache["misses"], result="miss")

    head("downshift_ready", "gauge", "1 when the model is loaded and the server is ready.")
    sample("downshift_ready", int(state is not None and state.ready))
    head("downshift_inflight", "gauge", "Admitted predicts that have not finished.")
    head("downshift_queue_depth", "gauge", "Admitted predicts that wait for an inference slot.")
    head("downshift_queue_limit", "gauge", "The --max-queue value.")
    head("downshift_model_info", "gauge", "The model that this worker serves. The value is 1.")
    head("downshift_boot_seconds", "gauge", "Seconds of each boot phase.")
    if state is not None:
        sample("downshift_inflight", state.in_flight)
        waiting = max(0, state.in_flight - state.options.max_concurrency)
        sample("downshift_queue_depth", waiting)
        sample("downshift_queue_limit", state.options.max_queue)
        sample(
            "downshift_model_info",
            1,
            downshift_version=__version__,
            verdict=state.verdict.status,
            backend=state.backend.name,
            family=state.verdict.model_family,
        )
        for phase, seconds in state.timings.items():
            sample("downshift_boot_seconds", seconds, phase=phase)
    return "\n".join(lines) + "\n"
