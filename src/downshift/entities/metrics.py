"""Entities: metrics."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

# Seconds. The last bucket is +Inf.
BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


# The stage label -> the timings_ms keys of predict that it adds up.
STAGE_KEYS = {
    "queue": ("prep_wait", "infer_wait", "encode_wait"),
    "prep": ("parse", "prep"),
    "infer": ("infer",),
    "encode": ("encode",),
}


# The error codes (serve/errors.py) that count as a rejected request.
REJECT_REASONS = {
    "at_capacity": "capacity",
    "request_timeout": "timeout",
    "not_ready": "not_ready",
}


CACHE_RESULTS = ("hit", "partial", "miss")


@dataclass
class Histogram:
    counts: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS) + 1))
    total: float = 0.0

    def observe(self, seconds: float) -> None:
        self.counts[bisect_left(BUCKETS, seconds)] += 1
        self.total += seconds


def histogram(series: dict[str, Histogram], key: str) -> Histogram:
    hist = series.get(key)
    if hist is None:
        hist = series[key] = Histogram()
    return hist


@dataclass
class Metrics:
    worker: str = "0"
    requests: dict[tuple[str, str], int] = field(default_factory=dict)  # (route, code)
    request_seconds: dict[str, Histogram] = field(default_factory=dict)  # by route
    stage_seconds: dict[str, Histogram] = field(default_factory=dict)  # by stage
    rejected: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in REJECT_REASONS.values()}
    )
    cache_requests: dict[str, int] = field(
        default_factory=lambda: {result: 0 for result in CACHE_RESULTS}
    )

    def record_request(self, route: str, status: int, seconds: float) -> None:
        key = (route, str(status))
        self.requests[key] = self.requests.get(key, 0) + 1
        histogram(self.request_seconds, route).observe(seconds)

    def record_stages(self, timings_ms: dict[str, float]) -> None:
        for stage, keys in STAGE_KEYS.items():
            milliseconds = 0.0
            seen = False
            for key in keys:
                if key in timings_ms:
                    milliseconds += timings_ms[key]
                    seen = True
            if seen:
                histogram(self.stage_seconds, stage).observe(milliseconds / 1000)

    def record_rejected(self, code: str | None) -> None:
        reason = REJECT_REASONS.get(code or "")
        if reason is not None:
            self.rejected[reason] += 1

    def record_cache_result(self, result: str | None) -> None:
        if result in CACHE_RESULTS:
            self.cache_requests[result] += 1
