"""Prometheus series for `/metrics`, on a registry the app owns.

Never prometheus_client's global REGISTRY: it raises "Duplicated timeseries" the second time an
app is built in one process. Under `--workers N` the parent sets PROMETHEUS_MULTIPROC_DIR, the
values live in mmap files shared by the workers, and `render` aggregates them through a fresh
registry per scrape.
"""

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector

from downshift.sources import path_basename

LATENCY_BUCKETS = tuple(sorted((0.0005, 0.001, *Histogram.DEFAULT_BUCKETS)))
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)
STAGES = ("parse", "prep_wait", "prep", "infer_wait", "infer", "encode")
REJECT_REASONS = ("capacity", "timeout", "body_too_large", "not_ready", "auth")


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        reg = self.registry
        self.info = Gauge(
            "downshift_info",
            "Build and model facts, always 1",
            ["version", "model", "backend", "verdict", "execution", "adapter"],
            registry=reg,
        )
        self.ready = Gauge(
            "downshift_ready",
            "1 once the model is loaded and verified",
            registry=reg,
            multiprocess_mode="liveall",
        )
        self.boot_seconds = Gauge(
            "downshift_boot_seconds",
            "Seconds spent in each boot phase",
            ["phase"],
            registry=reg,
            multiprocess_mode="max",
        )
        self.requests = Counter(
            "downshift_requests_total",
            "Requests by matched route template and status code",
            ["route", "status"],
            registry=reg,
        )
        self.request_duration = Histogram(
            "downshift_request_duration_seconds",
            "Request wall time by matched route template",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=reg,
        )
        self.stage_duration = Histogram(
            "downshift_stage_duration_seconds",
            "Time per request stage, the split Server-Timing reports",
            ["stage"],
            buckets=LATENCY_BUCKETS,
            registry=reg,
        )
        self.in_flight = Gauge(
            "downshift_in_flight",
            "Requests being served",
            registry=reg,
            multiprocess_mode="livesum",
        )
        self.queued = Gauge(
            "downshift_queued",
            "Requests waiting for a slot",
            registry=reg,
            multiprocess_mode="livesum",
        )
        self.concurrency = Gauge(
            "downshift_concurrency",
            "Configured --max-concurrency",
            registry=reg,
            multiprocess_mode="liveall",
        )
        self.rejected = Counter(
            "downshift_rejected_total", "Rejected requests by reason", ["reason"], registry=reg
        )
        self.batch_size = Histogram(
            "downshift_batch_size",
            "Rows per predict call",
            buckets=BATCH_BUCKETS,
            registry=reg,
        )
        self.graphs_per_request = Histogram(
            "downshift_graphs_per_request",
            "Graphs per /predict/graph request",
            buckets=BATCH_BUCKETS,
            registry=reg,
        )

    def set_info(
        self,
        *,
        version: str,
        source: str,
        backend: str,
        verdict: str,
        execution: str,
        adapter: str,
    ) -> None:
        self.info.labels(version, path_basename(source), backend, verdict, execution, adapter).set(
            1
        )

    def observe_request(self, route: str, status: int, seconds: float) -> None:
        self.requests.labels(route, str(status)).inc()
        self.request_duration.labels(route).observe(seconds)

    def observe_stages(self, timings_ms: dict[str, float]) -> None:
        """Observe the known stages only, so a stray timing key can't mint a new series."""
        for stage in STAGES:
            ms = timings_ms.get(stage)
            if ms is not None:
                self.stage_duration.labels(stage).observe(ms / 1000.0)

    def reject(self, reason: str) -> None:
        self.rejected.labels(reason).inc()

    def render(self) -> tuple[bytes, str]:
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            registry = CollectorRegistry()
            MultiProcessCollector(registry)
            return generate_latest(registry), CONTENT_TYPE_LATEST
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
