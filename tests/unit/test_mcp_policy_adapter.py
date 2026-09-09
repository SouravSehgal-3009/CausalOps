"""The candidate MCP executor must retain the established policy boundary."""

import json
from collections.abc import Callable

import pytest
from fake_incident import (
    INCIDENT_ID,
    RecordingChangesBackend,
    RecordingLogsBackend,
    RecordingMetricBackend,
    RecordingTopologyBackend,
    StepClock,
    incident_scope,
)

import causalops.mcp_policy_adapter as mcp_policy_adapter
from causalops.domain import (
    Budgets,
    CheckOutcome,
    IncidentScope,
    PolicyResult,
    ReasonCode,
)
from causalops.mcp_policy_adapter import (
    McpDispatchApprovalError,
    PolicyWrappedMcpExecutor,
    policy_approved_mcp_server,
)
from causalops.mcp_stdio import McpObservabilityServer
from causalops.tool_wrappers import (
    ReservationLedger,
    ToolWrapper,
    get_topology_wrapper,
    list_recent_changes_wrapper,
    query_logs_wrapper,
    query_metric_wrapper,
)
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    QueryLogsArguments,
    QueryMetricArguments,
    ToolName,
)


def _mcp_registry(
    *,
    run_metric: Callable[[QueryMetricArguments, IncidentScope], CheckOutcome]
    | None = None,
    run_logs: Callable[[QueryLogsArguments, IncidentScope], CheckOutcome] | None = None,
    run_changes: Callable[[ListRecentChangesArguments, IncidentScope], CheckOutcome]
    | None = None,
    run_topology: Callable[[GetTopologyArguments, IncidentScope], CheckOutcome]
    | None = None,
) -> dict[ToolName, ToolWrapper]:
    """A direct 4-tool registry, matching what `mcp_server_main._build_
    registry` actually builds -- `search_runbooks` is not one of the tools
    MCP is approved to serve (see `mcp_manifest.py`), so `PolicyWrappedMcp
    Executor`'s registry never carries it either."""
    return {
        ToolName.QUERY_METRIC: query_metric_wrapper(
            run_metric or RecordingMetricBackend()
        ),
        ToolName.QUERY_LOGS: query_logs_wrapper(run_logs or RecordingLogsBackend()),
        ToolName.LIST_RECENT_CHANGES: list_recent_changes_wrapper(
            run_changes or RecordingChangesBackend()
        ),
        ToolName.GET_TOPOLOGY: get_topology_wrapper(
            run_topology or RecordingTopologyBackend()
        ),
    }


def test_policy_wrapped_executor_refuses_cross_incident_before_backend_call() -> None:
    topology = RecordingTopologyBackend()
    executor = PolicyWrappedMcpExecutor(
        _mcp_registry(run_topology=topology),
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
        _mcp_registry(),
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
