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

import causalops.mcp_policy_adapter as mcp_policy_adapter
from causalops.domain import Budgets, PolicyResult, ReasonCode
from causalops.mcp_policy_adapter import (
    McpDispatchApprovalError,
    PolicyWrappedMcpExecutor,
    policy_approved_mcp_server,
)
from causalops.mcp_stdio import McpObservabilityServer
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


def _executor() -> PolicyWrappedMcpExecutor:
    return PolicyWrappedMcpExecutor(
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


def test_policy_approved_server_fails_closed_without_a_reviewed_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mechanism itself, proven in isolation: even though a real
    reviewed record now exists at module load (see
    `mcp_policy_adapter._APPROVED_MCP_DISPATCH` and
    `infra/phase3/VALIDATION.md`), the gate must still refuse cleanly
    whenever that record is absent -- monkeypatched here rather than
    relying on the module's own current value, so this test keeps proving
    the invariant regardless of whether approval exists today."""
    monkeypatch.setattr(mcp_policy_adapter, "_APPROVED_MCP_DISPATCH", None)

    with pytest.raises(McpDispatchApprovalError, match="remains disabled"):
        policy_approved_mcp_server(_executor())


def test_policy_approved_server_succeeds_with_todays_reviewed_record() -> None:
    """`_APPROVED_MCP_DISPATCH` is a real, reviewed record as of this
    commit (see `infra/phase3/VALIDATION.md`'s "Real local-stdio MCP
    transport" entry for the full 5-condition evidence) -- proves the
    approved path actually builds a working, policy-wrapped server, not
    just that the disabled path refuses."""
    server = policy_approved_mcp_server(_executor())

    assert isinstance(server, McpObservabilityServer)
