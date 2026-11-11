"""The metrics of one worker process, and their Prometheus text (format 0.0.4).

`Metrics` holds only what the request path must count. The event loop writes it, so it has no
lock. The gauges and the cache rows are not counted for each request: `render` reads them from
the ServingState. Each label has a fixed set of values. A request value is never a label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from downshift._version import __version__
from downshift.entities.metrics import BUCKETS, Histogram, Metrics

if TYPE_CHECKING:
    from downshift.entities.serving import ServingState

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


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
