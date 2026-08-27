"""The renamed pool gauge, against the real lab -- Lab-defect-fix Unit A (F1,
revised).

Marked `docker` for the same reason as `test_configuration_change.py`. Every
non-docker test in this codebase uses `RecordingPrometheus`
(`tests/unit/fake_incident.py`), which records the query string it was given
and returns canned data regardless of what that string says -- it never
validates the query against what a real Prometheus would actually have data
for. That means a mismatch between the gauge name `lab/services/orders.py`
publishes and the PromQL string `causalops/prometheus.py` queries for
(exactly the class of bug F1's rename was at risk of introducing: one side of
a two-file rename updated, the other not) would pass the entire non-docker
suite. This test calls `run_metric_check` directly against the running lab
and is the only thing that would catch that class of drift.
"""

from pathlib import Path

import pytest

from causalops.domain import MetricSampleStatus, StoredIncident, ToolOutcome
from causalops.prometheus import (
    DEFAULT_PROMETHEUS_URL,
    METRIC_QUERIES,
    run_metric_check,
)
from causalops.scenario_control import reset_scenario, runs_root, start_scenario
from causalops.tools import MetricTemplate, QueryMetricArguments

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILY = "resource_pool_saturation"


def stored_incident(root: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(root) / incident_id / "incident.json").read_text(encoding="utf-8")
    return StoredIncident.model_validate_json(text)


def test_the_renamed_gauge_returns_real_samples_from_the_running_lab() -> None:
    """`lab/services/orders.py` publishes `causalops_pool_attempts_per_
    capacity`; `causalops/prometheus.py`'s `METRIC_QUERIES` entry for
    `MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_CAPACITY` must query that
    exact name. `status == SAMPLED` and `sample_count > 0` alone are not
    enough: this test runs against an already-running lab, never rebuilt
    per-test, so a stale-on-BOTH-sides scenario (an old PromQL string still
    querying an old gauge name, neither side renamed) would also return
    real samples and pass those two checks -- they only prove *some* query
    the app and the lab still agree on returned data, not that the app
    queried the *new* name specifically. The `payload["promql"]` assertion
    below is what actually proves that: it pins the exact query string sent
    to Prometheus, built from the checked-in `METRIC_QUERIES` template
    rather than a hardcoded literal, so it breaks loudly (not silently) if
    the template is ever reformatted without the gauge name itself
    changing."""
    incident_id = start_scenario(REPOSITORY, FAMILY, "evaluation")

    try:
        incident = stored_incident(REPOSITORY, incident_id)
        scope = incident.scope

        arguments = QueryMetricArguments(
            template=MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_CAPACITY,
            service="orders",
            window_start=scope.started_at,
            window_end=scope.ended_at,
        )
        outcome = run_metric_check(arguments, scope, DEFAULT_PROMETHEUS_URL, timeout=10)

        assert outcome.outcome is ToolOutcome.EXECUTED
        template = METRIC_QUERIES[MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_CAPACITY]
        expected_promql = template.format(service="orders", incident=scope.incident_id)
        assert outcome.payload["promql"] == expected_promql
        assert outcome.payload["status"] == MetricSampleStatus.SAMPLED.value
        sample_count = outcome.payload["sample_count"]
        assert isinstance(sample_count, int)
        assert sample_count > 0
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
