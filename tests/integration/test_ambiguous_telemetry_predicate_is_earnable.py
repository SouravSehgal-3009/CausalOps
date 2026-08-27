"""F5's new `ambiguous_telemetry` required-evidence predicates, against the
real lab -- Lab-defect-fix Unit B, follow-up to `LAB_DEFECTS_FIX_PLAN.md`.

`tests/integration/test_incident_manifest_fidelity.py::
test_ambiguous_telemetry_shows_both_fault_signals_under_the_evaluation_seed`
already proves the underlying 5/5 `pool_exhausted`/`upstream_timeout` split
exists in `logs/orders.jsonl` -- but it reads that file directly off disk,
bypassing the `query_logs` tool backend entirely. F5's own predicates are
satisfied only if an investigator can reach both event codes *through the
registered tool call* (`run_logs_check`, the same backend the policy-wrapped
`query_logs` tool ultimately dispatches to). This test proves that -- a
distinct, necessary claim the manifest-fidelity test does not make.
"""

from pathlib import Path

import pytest

from causalops.domain import StoredIncident, ToolOutcome
from causalops.scenario_control import reset_scenario, runs_root, start_scenario
from causalops.telemetry import RunPaths, run_logs_check
from causalops.tools import LogFilter, QueryLogsArguments

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILY = "ambiguous_telemetry"


def stored_incident(root: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(root) / incident_id / "incident.json").read_text(encoding="utf-8")
    return StoredIncident.model_validate_json(text)


def test_both_required_event_codes_are_reachable_through_the_query_logs_backend() -> (
    None
):
    """`lab/scenarios/ambiguous_telemetry.json`'s `expected.predicates` now
    requires `event_codes` to CONTAIN both `pool_exhausted` and
    `upstream_timeout`. Both are logged with `severity="error"` in
    `logs/orders.jsonl` (`lab/services/orders.py`), so `LogFilter.
    ERRORS_ONLY` against service `orders` over the incident's own scope
    window must surface both -- proving the predicate is earnable by an
    investigator using the registered tool, not just present in the raw
    fixture."""
    incident_id = start_scenario(REPOSITORY, FAMILY, "evaluation")

    try:
        incident = stored_incident(REPOSITORY, incident_id)
        scope = incident.scope
        paths = RunPaths(root=runs_root(REPOSITORY) / incident_id)

        arguments = QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY,
            service="orders",
            window_start=scope.started_at,
            window_end=scope.ended_at,
            row_limit=200,
        )
        outcome = run_logs_check(arguments, paths)

        assert outcome.outcome is ToolOutcome.EXECUTED
        event_codes = outcome.payload["event_codes"]
        assert isinstance(event_codes, str)
        assert "pool_exhausted" in event_codes
        assert "upstream_timeout" in event_codes
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
