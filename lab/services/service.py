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
        )
        self.pool_in_use = Gauge(
            "causalops_pool_in_use",
            "Slots in use in the bounded resource pool.",
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
        self.pool_in_use.labels(self.name, incident).set(0)

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
    except OSError:
        return {}
    loaded: dict[str, Any] = json.loads(text)
    return loaded
