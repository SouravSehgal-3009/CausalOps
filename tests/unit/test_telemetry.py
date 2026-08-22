import json
import threading
import time
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
    RecordingPrometheus,
    incident_scope,
    log_row,
    prometheus_body,
    write_log,
)
from pydantic import JsonValue

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
    _registered_check_runner,
    run_changes_check,
    run_logs_check,
    run_topology_check,
    within_window,
)
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
    RunbookTopic,
    SearchRunbooksArguments,
)


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


def test_a_metric_query_returns_bounded_samples(
    tmp_path: Path,
    fake_prometheus: RecordingPrometheus,
) -> None:
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), fake_prometheus.url, 5
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.kind is EvidenceKind.METRIC
    assert outcome.payload["sample_count"] == MAX_METRIC_SAMPLES
    assert outcome.payload["truncated"] is True
    assert outcome.payload["max_value"] == (MAX_METRIC_SAMPLES - 1) * 0.5
    assert f"{MAX_METRIC_SAMPLES} samples" in outcome.summary


def test_a_metric_summary_reports_the_post_trim_count_not_the_pre_trim_one(
    monkeypatch: pytest.MonkeyPatch, fake_prometheus: RecordingPrometheus
) -> None:
    """Post-freeze review. `run_metric_check`'s summary string used to read
    `len(kept)` (the PRE-trim count) instead of `payload['sample_count']`
    (kept honest post-trim by `trim_to_bytes`, Unit 3b-4 addendum's C3)
    four lines below where the payload was already fixed -- reproduced
    live by correctness (payload said `sample_count: 371`, the summary
    still said "900 samples"). Unreachable through this function's own
    REAL data shape in this test suite: `MetricSample.at`/`.value` are
    both floats, always small, so `trim_to_bytes`'s byte-level popping
    never actually fires below the `MAX_METRIC_SAMPLES` count cap already
    applied earlier in the function. Monkeypatching `trim_to_bytes` to
    additionally drop one sample (simulating what a future, larger row
    shape would trigger) forces `payload['sample_count']` and the
    pre-trim `len(kept)` to genuinely differ, proving the summary string
    reads the field the payload itself reports, not a stale local
    variable, regardless of whether today's data can reach the
    difference."""
    import causalops.prometheus as prometheus_module

    real_trim_to_bytes = prometheus_module.trim_to_bytes

    def shrinking_trim_to_bytes(
        payload: dict[str, JsonValue],
        rows_key: str,
        rows: list[JsonValue],
        count_key: str,
    ) -> dict[str, JsonValue]:
        result = real_trim_to_bytes(payload, rows_key, rows, count_key)
        current_count = result[count_key]
        assert isinstance(current_count, int)
        result[count_key] = current_count - 1
        return result

    monkeypatch.setattr(prometheus_module, "trim_to_bytes", shrinking_trim_to_bytes)

    outcome = run_metric_check(
        metric_arguments(), incident_scope(), fake_prometheus.url, 5
    )

    assert outcome.payload["sample_count"] == MAX_METRIC_SAMPLES - 1
    assert f"{MAX_METRIC_SAMPLES - 1} samples" in outcome.summary
    assert f"{MAX_METRIC_SAMPLES} samples" not in outcome.summary


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
    # Post-freeze review, P3-1. 30 rows were matched, but byte-trimming
    # (not just the `row_limit` cap) drops some of them -- `outcome.summary`
    # must report the SAME, smaller, post-trim count `payload["row_count"]`
    # holds, not the pre-trim 30 `len(rows)` used to report regardless of
    # what trimming did.
    assert outcome.payload["row_count"] < 30
    assert f"{outcome.payload['row_count']} rows" in outcome.summary
    assert outcome.summary.endswith("(truncated)")


def test_a_log_query_for_a_run_without_that_log_is_unavailable(tmp_path: Path) -> None:
    outcome = run_logs_check(logs_arguments(), RunPaths(root=tmp_path))

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_within_window_rejects_a_naive_timestamp_instead_of_raising() -> None:
    """Unit 3b-4 addendum, C2. `datetime.fromisoformat` returns a naive
    `datetime` for a string carrying no UTC offset; comparing it against
    the aware `WINDOW_START`/`WINDOW_END` this function is always called
    with used to raise `TypeError`, uncaught (only `ValueError`, the parse
    failure, was caught) -- a naive timestamp is now excluded the same way
    an unparseable one already was, not raised."""
    naive = WINDOW_START.replace(tzinfo=None).isoformat()

    assert within_window(naive, WINDOW_START, WINDOW_END) is False


def test_a_log_row_with_a_naive_timestamp_is_excluded_not_crashed(
    tmp_path: Path,
) -> None:
    """The end-to-end sibling of `test_within_window_rejects_a_naive_
    timestamp_instead_of_raising`: before this fix, this call raised
    `TypeError` out of `within_window`, uncaught by `run_logs_check`,
    turning one malformed row into a crash for the whole check rather than
    excluding just that row."""
    paths = RunPaths(root=tmp_path)
    naive_row = log_row(5)
    naive_row["at"] = WINDOW_START.replace(tzinfo=None).isoformat()
    write_log(paths, [naive_row, log_row(10)])

    outcome = run_logs_check(logs_arguments(), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1


def test_a_change_row_with_a_naive_timestamp_is_excluded_not_crashed(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": WINDOW_START.replace(tzinfo=None).isoformat(),
                    "service": "orders",
                    "summary": "naive-timestamp change",
                },
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": "aware change",
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

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["change_count"] == 1


def test_an_oversized_single_summary_still_forces_the_payload_under_budget(
    tmp_path: Path,
) -> None:
    """Unit 3b-4 addendum, C3. `run_changes_check` joins every matched
    change's `summary` into one scalar `summaries` string before calling
    `trim_to_bytes` -- a single change with an oversized summary makes that
    scalar bigger than `MAX_RESULT_BYTES` all by itself, which no amount of
    popping the `changes` list could ever fix (there is only one change to
    begin with). Before this fix, `trim_to_bytes` would try popping it,
    empty the list, and still return an over-budget payload silently."""
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": "x" * (MAX_RESULT_BYTES + 4000),
                }
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

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    # `changes` (which embeds the raw entry, oversized `summary` field and
    # all) is popped before the scalar fallback ever runs -- see
    # `trim_to_bytes`'s own docstring for why that ordering matters here:
    # the single change's OWN `summary` field is a second copy of the same
    # oversized text `summaries` holds, invisible to a fallback that only
    # inspects top-level scalars. `change_count` still honestly matches
    # what `changes` actually holds, whatever that ends up being.
    assert outcome.payload["change_count"] == len(outcome.payload["changes"])  # type: ignore[arg-type]
    # Post-freeze review, P3-1. Before that fix, `outcome.summary` was
    # built from `len(changes)` (the PRE-trim count, always 1 here) --
    # "1 recent changes on orders" while the payload actually held zero.
    # Reproduced by correctness directly against this exact scenario.
    assert outcome.payload["change_count"] == 0
    assert outcome.summary == "0 recent changes on orders (truncated)"
    assert isinstance(outcome.payload["summaries"], str)
    assert len(outcome.payload["summaries"]) < MAX_RESULT_BYTES


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
    assert "1 service edges" in outcome.summary


def test_topology_summary_reports_the_post_trim_edge_count(tmp_path: Path) -> None:
    """Post-freeze review. `run_topology_check`'s summary string used to
    read `len(edge_list)` (the PRE-trim count) instead of
    `payload['edge_count']` (kept honest post-trim by `trim_to_bytes`,
    Unit 3b-4 addendum's C3) -- the same bug already fixed this round in
    `run_logs_check`/`run_changes_check`/`run_metric_check`. Unlike the
    metric case, this one is genuinely reachable with real data: many
    long edge strings really do exceed the byte budget without needing a
    monkeypatch."""
    paths = RunPaths(root=tmp_path)
    edges = [f"service-{i}>service-{i + 1}" for i in range(600)]
    paths.topology_file.write_text(
        json.dumps({"services": [], "edges": edges}), encoding="utf-8"
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert outcome.payload["truncated"] is True
    kept_edge_count = outcome.payload["edge_count"]
    assert isinstance(kept_edge_count, int)
    assert kept_edge_count < len(edges)
    assert f"{kept_edge_count} service edges" in outcome.summary
    assert f"{len(edges)} service edges" not in outcome.summary


def test_an_oversized_services_list_still_fits_the_byte_bound(tmp_path: Path) -> None:
    """The services-list fix. `services` is a LIST, not the string-valued
    scalar `trim_to_bytes`'s own fallback (C3) can shrink, and it is not
    `edges`, the row list `run_topology_check`'s other `trim_to_bytes`
    call already bounds -- before this fix, an oversized `services` list
    would pass through both mechanisms untouched and blow the byte
    budget. Reachable with real data, same as the edges case above: many
    long service names really do exceed it. Not reachable through any of
    the four shipped lab topologies (all under 100 bytes total) -- this
    is a defensive bound, not a scenario this project's own lab can
    trigger, per the owner's P3 severity ruling."""
    paths = RunPaths(root=tmp_path)
    services = [f"service-with-a-long-descriptive-name-{i}" for i in range(400)]
    paths.topology_file.write_text(
        json.dumps({"services": services, "edges": []}), encoding="utf-8"
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    kept_service_count = outcome.payload["service_count"]
    assert isinstance(kept_service_count, int)
    assert kept_service_count < len(services)
    assert kept_service_count == len(outcome.payload["services"])  # type: ignore[arg-type]


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
    run = _registered_check_runner(paths, "http://127.0.0.1:1", 1)

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


def test_the_runner_raises_loudly_for_search_runbooks(tmp_path: Path) -> None:
    """This seam predates `search_runbooks` and is superseded by
    `tool_wrappers.dispatch_registry` -- nothing in `cli.py` calls it. Its
    own `RunCheck` return type is `CheckOutcome` only, which cannot express
    `RunbookCheckOutcome`, so a `search_runbooks` proposal must raise here
    rather than silently falling through to `run_topology_check` the way an
    unconditional last branch would have."""
    paths = RunPaths(root=tmp_path)
    run = _registered_check_runner(paths, "http://127.0.0.1:1", 1)

    with pytest.raises(ValueError, match="SearchRunbooksArguments"):
        run(
            ToolProposal(
                arguments=SearchRunbooksArguments(
                    topic=RunbookTopic.GATEWAY_ERRORS, limit=3
                ),
                evidence_gap="gap",
                expected_observation="a passage",
            ),
            incident_scope(),
        )


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
