import json
import threading
import time
import urllib.parse
from collections.abc import Iterator
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

import pytest
from fake_incident import (
    INCIDENT_ID,
    SERVICES,
    WINDOW_END,
    WINDOW_START,
    incident_scope,
)

from causalops.domain import (
    EvidenceKind,
    IncidentScope,
    ReasonCode,
    ToolOutcome,
    ToolProposal,
)
from causalops.evidence import MAX_RESULT_BYTES
from causalops.prometheus import MAX_METRIC_SAMPLES, parse_samples, run_metric_check
from causalops.telemetry import (
    MAX_LOG_ROWS,
    RunPaths,
    registered_check_runner,
    run_changes_check,
    run_logs_check,
    run_topology_check,
)
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
)

SAMPLE_COUNT = MAX_METRIC_SAMPLES + 5


class RecordingPrometheus(NamedTuple):
    """The loopback server's address, plus every PromQL string it received."""

    url: str
    queries: list[str]


def prometheus_body() -> bytes:
    values = [[float(index), str(index * 0.5)] for index in range(SAMPLE_COUNT)]
    return json.dumps(
        {"status": "success", "data": {"result": [{"values": values}]}}
    ).encode("utf-8")


@pytest.fixture
def fake_prometheus() -> Iterator[RecordingPrometheus]:
    """A loopback stand-in for Prometheus, which section 12 allows in tests."""
    queries: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            received = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            queries.append(received.get("query", [""])[0])
            payload = prometheus_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield RecordingPrometheus(f"http://127.0.0.1:{server.server_port}", queries)
    server.shutdown()
    server.server_close()


@pytest.fixture
def stalled_prometheus() -> Iterator[str]:
    """A loopback server whose answer never arrives inside a short client timeout."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            time.sleep(2)
            payload = prometheus_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def metric_arguments(
    service: str = "gateway",
    template: MetricTemplate = MetricTemplate.GATEWAY_ERROR_RATE,
) -> QueryMetricArguments:
    return QueryMetricArguments(
        template=template,
        service=service,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )


def logs_arguments(
    row_limit: int = 20, log_filter: LogFilter = LogFilter.ERRORS_ONLY
) -> QueryLogsArguments:
    return QueryLogsArguments(
        log_filter=log_filter,
        service="orders",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        row_limit=row_limit,
    )


def write_log(paths: RunPaths, rows: list[dict[str, object]]) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(row) + "\n" for row in rows)
    (paths.logs / "orders.jsonl").write_text(lines, encoding="utf-8")


def log_row(
    offset: int,
    severity: str = "error",
    event: str = "config_rejected_request",
    detail: str = "x",
) -> dict[str, object]:
    return {
        "at": (WINDOW_START + timedelta(seconds=offset)).isoformat(),
        "request_id": f"r{offset}",
        "service": "orders",
        "severity": severity,
        "event": event,
        "fields": {"config_key": "require_order_token", "detail": detail},
    }


def test_a_metric_query_returns_bounded_samples(
    tmp_path: Path, fake_prometheus: RecordingPrometheus
) -> None:
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), fake_prometheus.url, 5
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.kind is EvidenceKind.METRIC
    assert outcome.payload["sample_count"] == MAX_METRIC_SAMPLES
    assert outcome.payload["truncated"] is True
    assert outcome.payload["max_value"] == (MAX_METRIC_SAMPLES - 1) * 0.5


def test_a_metric_query_against_nothing_is_unavailable() -> None:
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), "http://127.0.0.1:1", 1
    )

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_a_metric_query_that_outlasts_its_timeout_times_out(
    stalled_prometheus: str,
) -> None:
    """A genuine `urllib.request.urlopen` timeout, not a stand-in for one."""
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), stalled_prometheus, 1
    )

    assert outcome.outcome is ToolOutcome.TIMEOUT
    assert outcome.reason_code is ReasonCode.TOOL_TIMEOUT


def test_a_service_name_that_could_reach_promql_is_refused() -> None:
    outcome = run_metric_check(
        metric_arguments(service='gateway"} or up{'), incident_scope(), "http://x", 1
    )

    assert outcome.outcome is ToolOutcome.ERROR
    assert outcome.reason_code is ReasonCode.TOOL_ERROR


def test_an_unreadable_prometheus_answer_is_an_error() -> None:
    assert parse_samples({"status": "error"}) is None
    assert parse_samples({"status": "success", "data": {"result": []}}) == []
    assert parse_samples("nonsense") is None


def test_a_log_query_returns_only_matching_rows_in_the_window(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1),
            log_row(2, severity="info", event="request_served"),
            log_row(99_999),
        ],
    )

    outcome = run_logs_check(logs_arguments(), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "config_rejected_request"


def test_a_log_query_clamps_the_row_limit_to_the_budget(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(paths, [log_row(minute) for minute in range(MAX_LOG_ROWS + 10)])

    outcome = run_logs_check(logs_arguments(row_limit=200), paths)

    assert outcome.payload["row_count"] == MAX_LOG_ROWS
    assert outcome.payload["truncated"] is True


def test_a_log_result_stays_inside_the_byte_bound(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(paths, [log_row(minute, detail="x" * 2000) for minute in range(30)])

    outcome = run_logs_check(logs_arguments(row_limit=30), paths)

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True


def test_a_log_query_for_a_run_without_that_log_is_unavailable(tmp_path: Path) -> None:
    outcome = run_logs_check(logs_arguments(), RunPaths(root=tmp_path))

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_recent_changes_are_filtered_by_service_and_window(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": "configuration update: require_order_token enabled",
                },
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "inventory",
                    "summary": "image rebuild",
                },
                {
                    "at": (WINDOW_START - timedelta(days=1)).isoformat(),
                    "service": "orders",
                    "summary": "older change",
                },
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.payload["change_count"] == 1
    assert "require_order_token" in str(outcome.payload["summaries"])


def test_topology_reads_the_run_manifest(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    paths.topology_file.write_text(
        json.dumps({"services": list(SERVICES), "edges": ["gateway>orders"]}),
        encoding="utf-8",
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["edge_count"] == 1


def test_a_missing_manifest_is_unavailable_rather_than_a_crash(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)

    topology = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)
    changes = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert topology.outcome is ToolOutcome.UNAVAILABLE
    assert changes.outcome is ToolOutcome.UNAVAILABLE


def test_the_runner_sends_each_tool_to_its_own_backend(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(paths, [log_row(1)])
    paths.topology_file.write_text(json.dumps({"edges": []}), encoding="utf-8")
    run = registered_check_runner(paths, "http://127.0.0.1:1", 1)

    logs = run(
        ToolProposal(
            arguments=logs_arguments(), evidence_gap="gap", expected_observation="rows"
        ),
        incident_scope(),
    )
    topology = run(
        ToolProposal(
            arguments=GetTopologyArguments(incident_id=INCIDENT_ID),
            evidence_gap="gap",
            expected_observation="edges",
        ),
        incident_scope(),
    )
    metric = run(
        ToolProposal(
            arguments=metric_arguments(),
            evidence_gap="gap",
            expected_observation="samples",
        ),
        incident_scope(),
    )

    assert logs.kind is EvidenceKind.LOG
    assert topology.kind is EvidenceKind.TOPOLOGY
    assert metric.kind is EvidenceKind.METRIC


# Cross-incident isolation
#
# A run backend never receives an incident_id argument for logs or changes; it
# only ever sees the RunPaths of the run it was handed. These tests build two
# separate incident directories with distinguishable content and confirm a
# check pointed at one run's paths never surfaces the other run's content.

OTHER_INCIDENT_ID = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"


def other_incident_scope() -> IncidentScope:
    return incident_scope().model_copy(update={"incident_id": OTHER_INCIDENT_ID})


def test_a_log_check_never_surfaces_another_incidents_rows(tmp_path: Path) -> None:
    paths_a = RunPaths(root=tmp_path / "incident-a")
    paths_b = RunPaths(root=tmp_path / "incident-b")
    write_log(paths_a, [log_row(1, detail="incident-a-only-marker")])
    write_log(paths_b, [log_row(1, detail="incident-b-only-marker")])

    outcome = run_logs_check(logs_arguments(), paths_a)

    dumped = json.dumps(outcome.payload)
    assert "incident-a-only-marker" in dumped
    assert "incident-b-only-marker" not in dumped


def write_changes(paths: RunPaths, summary: str) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": summary,
                }
            ]
        ),
        encoding="utf-8",
    )


def test_a_changes_check_never_surfaces_another_incidents_entries(
    tmp_path: Path,
) -> None:
    paths_a = RunPaths(root=tmp_path / "incident-a")
    paths_b = RunPaths(root=tmp_path / "incident-b")
    write_changes(paths_a, "incident-a-only-change")
    write_changes(paths_b, "incident-b-only-change")

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths_a,
    )

    dumped = json.dumps(outcome.payload)
    assert "incident-a-only-change" in dumped
    assert "incident-b-only-change" not in dumped


def test_the_metric_query_label_is_derived_from_the_scope_not_an_argument(
    fake_prometheus: RecordingPrometheus,
) -> None:
    run_metric_check(metric_arguments(), incident_scope(), fake_prometheus.url, 5)
    run_metric_check(metric_arguments(), other_incident_scope(), fake_prometheus.url, 5)

    query_a, query_b = fake_prometheus.queries[-2], fake_prometheus.queries[-1]
    assert f'incident="{incident_scope().incident_id}"' in query_a
    assert f'incident="{other_incident_scope().incident_id}"' in query_b


def test_topology_decides_from_paths_not_the_argument_incident_id(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.topology_file.write_text(
        json.dumps({"services": list(SERVICES), "edges": ["gateway>orders"]}),
        encoding="utf-8",
    )

    outcome = run_topology_check(
        GetTopologyArguments(incident_id="not-the-real-one"), paths
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["edge_count"] == 1


# Template and filter coverage
#
# The tests above only exercise one metric template and one log filter. These
# confirm the remaining registered templates and filters run end to end.


def test_the_downstream_timeout_rate_template_executes(
    fake_prometheus: RecordingPrometheus,
) -> None:
    arguments = metric_arguments(template=MetricTemplate.DOWNSTREAM_TIMEOUT_RATE)

    outcome = run_metric_check(arguments, incident_scope(), fake_prometheus.url, 5)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["template"] == MetricTemplate.DOWNSTREAM_TIMEOUT_RATE.value


def test_the_resource_pool_in_use_template_executes(
    fake_prometheus: RecordingPrometheus,
) -> None:
    arguments = metric_arguments(template=MetricTemplate.RESOURCE_POOL_IN_USE)

    outcome = run_metric_check(arguments, incident_scope(), fake_prometheus.url, 5)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["template"] == MetricTemplate.RESOURCE_POOL_IN_USE.value


def test_the_gateway_latency_p95_template_executes(
    fake_prometheus: RecordingPrometheus,
) -> None:
    arguments = metric_arguments(template=MetricTemplate.GATEWAY_LATENCY_P95)

    outcome = run_metric_check(arguments, incident_scope(), fake_prometheus.url, 5)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["template"] == MetricTemplate.GATEWAY_LATENCY_P95.value


def test_the_timeouts_only_filter_matches_timeout_events(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1, event="upstream_timeout"),
            log_row(2, event="config_rejected_request"),
        ],
    )

    outcome = run_logs_check(logs_arguments(log_filter=LogFilter.TIMEOUTS_ONLY), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "upstream_timeout"


def test_the_pool_exhaustion_filter_matches_pool_exhausted_events(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1, event="pool_exhausted"),
            log_row(2, event="upstream_timeout"),
        ],
    )

    outcome = run_logs_check(
        logs_arguments(log_filter=LogFilter.POOL_EXHAUSTION), paths
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "pool_exhausted"


def test_the_config_reload_filter_matches_config_loaded_events(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1, event="config_loaded"),
            log_row(2, event="upstream_timeout"),
        ],
    )

    outcome = run_logs_check(logs_arguments(log_filter=LogFilter.CONFIG_RELOAD), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "config_loaded"
