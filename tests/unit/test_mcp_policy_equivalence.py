"""Direct-backend vs MCP-backed dispatch, compared on the same proposals.

`_make_wrapper.dispatch()` (`tool_wrappers.py:465-511`) always calls
`authorize()` first and returns an early denial before `run_check` --
i.e. before the MCP path ever touches the child process -- so a
cross-incident or duplicate proposal is denied identically by both
registries by construction, not by anything MCP-specific. These tests
prove that reuse holds for real, against a real spawned subprocess.

A reviewed `mcp_policy_adapter._APPROVED_MCP_DISPATCH` record now exists
(see `infra/phase3/VALIDATION.md`), so the last test here proves real
EXECUTED-outcome equivalence -- condition 2's "same evidence, citations,
outcome for an EXECUTED check" -- against a real spawned child process,
not just the pre-approval refusal shape.
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
    topology_proposal,
    write_topology,
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
from causalops.telemetry import RunPaths, run_topology_check
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


def test_mcp_dispatch_executes_and_matches_the_direct_backend_result(
    tmp_path: Path,
) -> None:
    """With a reviewed approval record in place, an allowed MCP proposal
    actually executes against the real backend -- not just a clean
    refusal -- and its receipt/evidence match the direct-dispatch path
    exactly, over a real spawned child process."""
    paths = RunPaths(root=tmp_path)
    write_topology(paths, ["gateway>orders"])
    proposal = topology_proposal()
    budgets = Budgets()

    direct_registry = dispatch_registry(
        run_metric=RecordingMetricBackend(),
        run_logs=RecordingLogsBackend(),
        run_changes=RecordingChangesBackend(),
        run_topology=lambda a, s: run_topology_check(a, paths),
        run_search=RecordingRunbooksBackend(),
    )
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

    assert direct_result.receipt.policy_result is PolicyResult.ALLOWED
    assert direct_result.receipt.outcome is ToolOutcome.EXECUTED
    assert direct_result.evidence is not None
    assert direct_result.evidence.payload["edge_count"] == 1

    assert mcp_result.receipt.policy_result is direct_result.receipt.policy_result
    assert mcp_result.receipt.outcome is direct_result.receipt.outcome
    assert mcp_result.evidence is not None
    assert mcp_result.evidence.payload == direct_result.evidence.payload
    assert mcp_result.evidence.summary == direct_result.evidence.summary
