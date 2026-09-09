"""Direct-backend vs MCP-backed dispatch, compared on the same proposals.

`_make_wrapper.dispatch()` (`tool_wrappers.py:465-511`) always calls
`authorize()` first and returns an early denial before `run_check` --
i.e. before the MCP path ever touches the child process -- so a
cross-incident or duplicate proposal is denied identically by both
registries by construction, not by anything MCP-specific. These tests
prove that reuse holds for real, against a real spawned subprocess.

Full outcome-content equivalence (condition 2's "same evidence, citations,
outcome for an EXECUTED check") cannot be proven yet:
`mcp_policy_adapter._APPROVED_MCP_DISPATCH` is genuinely `None` today, so
every real MCP dispatch is correctly refused by the safe default before
touching a backend at all (see `test_mcp_child_process.py`). The last test
here pins that honest current behavior; re-point it at a real EXECUTED
comparison once a reviewed approval record exists.
"""

from pathlib import Path

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

from causalops.domain import (
    Budgets,
    PolicyResult,
    ReasonCode,
    ToolOutcome,
    ToolProposal,
)
from causalops.mcp_child_process import McpChildProcess
from causalops.mcp_client_registry import build_mcp_tool_registry
from causalops.tool_wrappers import ReservationLedger, ToolWrapper, dispatch_registry
from causalops.tools import GetTopologyArguments, ToolName


def _ledger(budgets: Budgets) -> ReservationLedger:
    return ReservationLedger(budgets.executed_tools)


def _direct_registry() -> dict[ToolName, ToolWrapper]:
    return dispatch_registry(
        run_metric=RecordingMetricBackend(),
        run_logs=RecordingLogsBackend(),
        run_changes=RecordingChangesBackend(),
        run_topology=RecordingTopologyBackend(),
        run_search=RecordingRunbooksBackend(),
    )


def test_cross_incident_topology_is_denied_identically_by_both_registries(
    tmp_path: Path,
) -> None:
    proposal = ToolProposal(
        arguments=GetTopologyArguments(incident_id="other-incident"),
        evidence_gap="prove isolation",
        expected_observation="a denial",
    )
    budgets = Budgets()

    direct_registry = _direct_registry()
    direct_result = direct_registry[proposal.arguments.tool].dispatch(
        proposal, incident_scope(), set(), budgets, _ledger(budgets), StepClock()
    )

    child = McpChildProcess()
    child.start(tmp_path, incident_scope(), budgets)
    try:
        mcp_registry = build_mcp_tool_registry(child, budgets)
        mcp_result = mcp_registry[proposal.arguments.tool].dispatch(
            proposal, incident_scope(), set(), budgets, _ledger(budgets), StepClock()
        )
    finally:
        child.close()

    for result in (direct_result, mcp_result):
        assert result.receipt.policy_result is PolicyResult.DENIED
        assert result.receipt.reason_code is ReasonCode.CROSS_INCIDENT_REQUEST
        assert result.receipt.incident_id == INCIDENT_ID
        assert result.evidence is None


def test_duplicate_proposal_is_denied_identically_by_both_registries(
    tmp_path: Path,
) -> None:
    proposal = ToolProposal(
        arguments=GetTopologyArguments(incident_id=INCIDENT_ID),
        evidence_gap="confirm topology",
        expected_observation="the same edges twice",
    )
    budgets = Budgets()

    direct_registry = _direct_registry()
    direct_seen: set[str] = set()
    direct_registry[proposal.arguments.tool].dispatch(
        proposal, incident_scope(), direct_seen, budgets, _ledger(budgets), StepClock()
    )
    direct_second = direct_registry[proposal.arguments.tool].dispatch(
        proposal, incident_scope(), direct_seen, budgets, _ledger(budgets), StepClock()
    )

    child = McpChildProcess()
    child.start(tmp_path, incident_scope(), budgets)
    try:
        mcp_registry = build_mcp_tool_registry(child, budgets)
        mcp_seen: set[str] = set()
        mcp_registry[proposal.arguments.tool].dispatch(
            proposal, incident_scope(), mcp_seen, budgets, _ledger(budgets), StepClock()
        )
        mcp_second = mcp_registry[proposal.arguments.tool].dispatch(
            proposal, incident_scope(), mcp_seen, budgets, _ledger(budgets), StepClock()
        )
    finally:
        child.close()

    for second in (direct_second, mcp_second):
        assert second.receipt.policy_result is PolicyResult.DENIED
        assert second.receipt.reason_code is ReasonCode.DUPLICATE_PROPOSAL


def test_mcp_dispatch_today_is_a_clean_refusal_not_an_executed_result(
    tmp_path: Path,
) -> None:
    """Pins the honest current state: with no reviewed approval record,
    an otherwise-allowed MCP proposal is refused safely, not executed.
    Replace this with a real EXECUTED-outcome equivalence assertion once
    `mcp_policy_adapter._APPROVED_MCP_DISPATCH` is a real record."""
    proposal = ToolProposal(
        arguments=GetTopologyArguments(incident_id=INCIDENT_ID),
        evidence_gap="confirm topology",
        expected_observation="edges",
    )
    budgets = Budgets()

    child = McpChildProcess()
    child.start(tmp_path, incident_scope(), budgets)
    try:
        mcp_registry = build_mcp_tool_registry(child, budgets)
        result = mcp_registry[proposal.arguments.tool].dispatch(
            proposal, incident_scope(), set(), budgets, _ledger(budgets), StepClock()
        )
    finally:
        child.close()

    assert result.receipt.policy_result is PolicyResult.ALLOWED
    assert result.receipt.outcome is ToolOutcome.ERROR
    assert result.evidence is None
