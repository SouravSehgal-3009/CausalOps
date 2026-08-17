import json
import threading
from collections.abc import Iterator
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fake_incident import (
    INCIDENT_ID,
    SERVICES,
    WINDOW_END,
    WINDOW_START,
    incident_scope,
)

from causalops.domain import EvidenceKind, ReasonCode, ToolOutcome, ToolProposal
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


def prometheus_body() -> bytes:
    values = [[float(index), str(index * 0.5)] for index in range(SAMPLE_COUNT)]
    return json.dumps(
        {"status": "success", "data": {"result": [{"values": values}]}}
    ).encode("utf-8")


@pytest.fixture
def fake_prometheus() -> Iterator[str]:
    """A loopback stand-in for Prometheus, which section 12 allows in tests."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
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


def metric_arguments(service: str = "gateway") -> QueryMetricArguments:
    return QueryMetricArguments(
        template=MetricTemplate.GATEWAY_ERROR_RATE,
        service=service,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )


def logs_arguments(row_limit: int = 20) -> QueryLogsArguments:
    return QueryLogsArguments(
        log_filter=LogFilter.ERRORS_ONLY,
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
    tmp_path: Path, fake_prometheus: str
) -> None:
    outcome = run_metric_check(metric_arguments(), incident_scope(), fake_prometheus, 5)

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
