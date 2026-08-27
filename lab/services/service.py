"""Shared runtime for the three lab services: logging, metrics, health, serving.

These services are the thing being observed, so they import nothing from
`causalops`. They learn which incident is active from a file the scenario
controller writes, which is why a container started before a scenario still
labels its samples correctly once one begins.
"""

import json
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

RUNS_ROOT = Path(os.environ.get("CAUSALOPS_RUNS", "/runs"))
ACTIVE_INCIDENT_FILE = RUNS_ROOT / "active-incident.txt"
NO_INCIDENT = "none"

MAX_LOG_FIELDS = 8
MAX_FIELD_LENGTH = 120

# Lab-defect-fix Unit 2, W6. `prometheus_client`'s own default buckets
# (`.005` to `10.0`, doubling roughly every 2-3 steps) put this lab's
# entire real dynamic range inside one bucket -- confirmed measured
# (§8.6): a real ~1.21s mean latency was reported by `histogram_quantile`
# as `peak 2.425`, a bucket-interpolation artifact roughly double reality,
# not a query defect. Deriving a replacement ladder from the histogram's
# OWN `histogram_quantile` output during a live run would be circular --
# the histogram's coarseness IS the defect being fixed, so its own output
# cannot be trusted to reveal where the boundaries should go. This ladder
# is derived instead from the lab's known, source-level timing mechanics:
#   - the timeout path: `orders.py`'s `INVENTORY_MAX_ATTEMPTS` (3) ×
#     `INVENTORY_ATTEMPT_TIMEOUT_SECONDS` (0.4s) ~= 1.2s before
#     `upstream_timeout` is logged;
#   - the delayed-success path: `inventory.py`'s configured
#     `response_delay_seconds`, `1.5` or `2.0` across the four scenario
#     definitions.
# Boundaries cluster tightly around both landmarks (1.1-1.3s, 1.5-2.5s) so
# `histogram_quantile`'s interpolation has real cut points to work with in
# exactly the range this lab's own fault traffic actually populates,
# instead of one 5-9.995-wide gap swallowing it whole. `+Inf` is not
# hand-appended: `prometheus_client`'s own `Histogram` adds it automatically
# when the given sequence does not already end there (`_prepare_buckets`),
# so no infinity literal needs copying from this comment into code.
LATENCY_BUCKETS_SECONDS = (
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    5.0,
)

# What a route hands back: an HTTP status, a small JSON body, and the outcome label
# the metrics carry.
RouteResult = tuple[int, dict[str, Any], str]
Route = Callable[[str], RouteResult]


class LabService:
    """One lab service: its metrics, its JSONL log, and its HTTP server."""

    def __init__(self, name: str, port: int, path: str) -> None:
        self.name = name
        self.port = port
        self.path = path
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "causalops_requests_total",
            "Requests handled by a lab service.",
            ["service", "incident", "outcome"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "causalops_request_latency_seconds",
            "Latency of requests handled by a lab service.",
            ["service", "incident"],
            registry=self.registry,
            buckets=LATENCY_BUCKETS_SECONDS,
        )
        self.pool_attempts_per_capacity = Gauge(
            "causalops_pool_attempts_per_capacity",
            "Cumulative pool slot acquisition attempts for the incident "
            "divided by configured capacity. Exceeds 1 once attempts "
            "outstrip the pool; this is not an occupancy or utilization "
            "fraction.",
            ["service", "incident"],
            registry=self.registry,
        )
        self.write_lock = threading.Lock()

    def incident_id(self) -> str:
        try:
            named = ACTIVE_INCIDENT_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return NO_INCIDENT
        return named or NO_INCIDENT

    def log(self, severity: str, event: str, request_id: str, **fields: Any) -> None:
        """Append one bounded structured record for the active incident."""
        incident = self.incident_id()
        if incident == NO_INCIDENT:
            return
        record = {
            "at": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "service": self.name,
            "severity": severity,
            "event": event,
            "fields": bounded_fields(fields),
        }
        directory = RUNS_ROOT / incident / "logs"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with self.write_lock:
                with (directory / f"{self.name}.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as log_file:
                    log_file.write(json.dumps(record) + "\n")
        except OSError:
            # A lab service must keep serving even when its log volume is unhappy.
            return

    def observe(self, outcome: str, seconds: float) -> None:
        incident = self.incident_id()
        self.requests.labels(self.name, incident, outcome).inc()
        self.latency.labels(self.name, incident).observe(seconds)

    def set_pool_attempts_per_capacity(self, attempts_per_capacity: float) -> None:
        self.pool_attempts_per_capacity.labels(self.name, self.incident_id()).set(
            attempts_per_capacity
        )

    def serve(self, route: Route) -> None:
        server = ThreadingHTTPServer(("", self.port), self.handler_class(route))
        server.serve_forever()

    def handler_class(self, route: Route) -> type[BaseHTTPRequestHandler]:
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802  (http.server names it)
                if self.path == "/healthz":
                    self.reply(200, {"status": "ok", "service": service.name})
                    return
                if self.path == "/metrics":
                    self.reply_metrics()
                    return
                if self.path != service.path:
                    self.reply(404, {"error": "unknown path"})
                    return
                request_id = uuid4().hex[:12]
                started = time.monotonic()
                status, body, outcome = route(request_id)
                service.observe(outcome, time.monotonic() - started)
                self.reply(status, body)

            def reply(self, status: int, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def reply_metrics(self) -> None:
                payload = generate_latest(service.registry)
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                # The JSONL log is the record that matters; stderr noise is not.
                return

        return Handler


def bounded_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Keep a log record small enough that a bounded tool result stays useful."""
    bounded: dict[str, Any] = {}
    for key, value in list(fields.items())[:MAX_LOG_FIELDS]:
        bounded[key] = value[:MAX_FIELD_LENGTH] if isinstance(value, str) else value
    return bounded


def read_lab_config(incident_id: str) -> dict[str, Any]:
    """Controller-written service configuration for the active incident."""
    try:
        text = (RUNS_ROOT / incident_id / "lab" / "config.json").read_text(
            encoding="utf-8"
        )
        loaded: dict[str, Any] = json.loads(text)
    except (OSError, ValueError):
        # `JSONDecodeError` and `UnicodeDecodeError` both subclass ValueError,
        # so this also covers a torn or non-UTF-8 read, not just malformed JSON.
        return {}
    return loaded
