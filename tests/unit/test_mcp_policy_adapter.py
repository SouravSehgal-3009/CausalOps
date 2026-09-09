"""The candidate MCP executor must retain the established policy boundary."""

import json

import pytest
from fake_incident import (
    INCIDENT_ID,
    RecordingChangesBackend,
    RecordingLogsBackend,
    RecordingMetricBackend,
    RecordingRunbooksBackend,
    RecordingTopologyBackend,
    StepClock,
    incident_scope,
)

from causalops.domain import Budgets, PolicyResult, ReasonCode
from causalops.mcp_policy_adapter import (
    McpDispatchApprovalError,
    PolicyWrappedMcpExecutor,
    policy_approved_mcp_server,
)
from causalops.tool_wrappers import ReservationLedger, dispatch_registry
from causalops.tools import GetTopologyArguments


def test_policy_wrapped_executor_refuses_cross_incident_before_backend_call() -> None:
    topology = RecordingTopologyBackend()
    executor = PolicyWrappedMcpExecutor(
        dispatch_registry(
            run_metric=RecordingMetricBackend(),
            run_logs=RecordingLogsBackend(),
            run_changes=RecordingChangesBackend(),
            run_topology=topology,
            run_search=RecordingRunbooksBackend(),
        ),
        incident_scope(),
        set(),
        Budgets(),
        ReservationLedger(Budgets().executed_tools),
        StepClock(),
    )

    response = executor.call(GetTopologyArguments(incident_id="other-incident"))
    payload = json.loads(response.content[0].text)

    assert topology.calls == []
    assert payload["receipt"]["policy_result"] == PolicyResult.DENIED.value
    assert payload["receipt"]["reason_code"] == ReasonCode.CROSS_INCIDENT_REQUEST.value
    assert payload["receipt"]["incident_id"] == INCIDENT_ID


def test_policy_approved_server_fails_closed_without_a_reviewed_record() -> None:
    executor = PolicyWrappedMcpExecutor(
        dispatch_registry(
            run_metric=RecordingMetricBackend(),
            run_logs=RecordingLogsBackend(),
            run_changes=RecordingChangesBackend(),
            run_topology=RecordingTopologyBackend(),
            run_search=RecordingRunbooksBackend(),
        ),
        incident_scope(),
        set(),
        Budgets(),
        ReservationLedger(Budgets().executed_tools),
        StepClock(),
    )

    with pytest.raises(McpDispatchApprovalError, match="remains disabled"):
        policy_approved_mcp_server(executor)
