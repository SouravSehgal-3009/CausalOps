"""The `service` field descriptions, against the real lab.

The restriction documented there (a metric template or log category recorded by
one lab service and not another) is architectural, not scenario-specific --
it comes from which `lab/services/*.py` module publishes which gauge/counter
or writes which log category, not from any one incident's fault injection.
So one real scenario start is sufficient to prove it; the point of these
tests is that the registered tool backends genuinely return the documented
empty result for a service that never produces the data, not that any
particular family exercises it.

Marked `docker` for the same reason as `test_pool_metric_gauge_name.py`:
every non-docker test in this codebase uses `RecordingPrometheus`/canned
fixture data, which would return whatever a test wires it to return
regardless of whether the real lab actually has no data for `inventory` --
only a call against the live lab and Prometheus proves the restriction is
real, not merely documented.
"""

from pathlib import Path

import pytest

from causalops.domain import MetricSampleStatus, StoredIncident, ToolOutcome
from causalops.prometheus import DEFAULT_PROMETHEUS_URL, run_metric_check
from causalops.scenario_control import reset_scenario, runs_root, start_scenario
from causalops.telemetry import RunPaths, run_logs_check
from causalops.tools import (
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
)

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILY = "configuration_change"


def stored_incident(root: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(root) / incident_id / "incident.json").read_text(encoding="utf-8")
    return StoredIncident.model_validate_json(text)


def test_inventory_has_no_resource_pool_metric_in_the_real_lab() -> None:
    """`QueryMetricArguments.service`'s new description states
    `resource_pool_attempts_per_capacity` is recorded only by `orders` --
    `lab/services/inventory.py` never publishes that gauge. Against the
    real lab, querying it for `inventory` must come back as a real,
    executed check with a genuinely empty result (`NO_RETURNED_SERIES`,
    `sample_count == 0`), not `TOOL_UNAVAILABLE` or an error -- the tool
    runs fine, there is simply no series to return."""
    incident_id = start_scenario(REPOSITORY, FAMILY, "evaluation")

    try:
        incident = stored_incident(REPOSITORY, incident_id)
        scope = incident.scope

        arguments = QueryMetricArguments(
            template=MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_CAPACITY,
            service="inventory",
            window_start=scope.started_at,
            window_end=scope.ended_at,
        )
        outcome = run_metric_check(arguments, scope, DEFAULT_PROMETHEUS_URL, timeout=10)

        assert outcome.outcome is ToolOutcome.EXECUTED
        assert outcome.payload["status"] == MetricSampleStatus.NO_RETURNED_SERIES.value
        assert outcome.payload["sample_count"] == 0
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()


def test_inventory_has_no_matching_rows_for_any_log_category_in_the_real_lab() -> None:
    """`QueryLogsArguments.service`'s new description states `inventory`
    logs none of the four `LogFilter` categories -- `lab/services/
    inventory.py` never writes an `errors_only`/`timeouts_only`/
    `pool_exhaustion`/`config_reload`-matching row. Against the real lab,
    each of the four filters must come back as a real, executed check with
    zero matching rows, not `TOOL_UNAVAILABLE` -- the log file exists (the
    service runs and logs *something*), only its filtered content is
    always empty."""
    incident_id = start_scenario(REPOSITORY, FAMILY, "evaluation")

    try:
        incident = stored_incident(REPOSITORY, incident_id)
        scope = incident.scope
        paths = RunPaths(root=runs_root(REPOSITORY) / incident_id)

        for log_filter in LogFilter:
            arguments = QueryLogsArguments(
                log_filter=log_filter,
                service="inventory",
                window_start=scope.started_at,
                window_end=scope.ended_at,
                row_limit=40,
            )
            outcome = run_logs_check(arguments, paths)

            assert outcome.outcome is ToolOutcome.EXECUTED, log_filter
            assert outcome.payload["row_count"] == 0, log_filter
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
