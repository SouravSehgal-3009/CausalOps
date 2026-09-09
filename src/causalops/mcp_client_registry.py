"""Client-side MCP-backed tool registry and runtime wiring.

Builds a `Mapping[ToolName, ToolWrapper]` via the *existing*
`tool_wrappers.dispatch_registry` factory -- the same one
`live_setup._build_tool_registry` uses for direct backends -- except each
`run_check` closure forwards the call to a real MCP child process instead
of calling a backend function directly. Because `_make_wrapper.dispatch()`
always calls `authorize()` and returns an early denial before `run_check`
is ever invoked, malformed/unknown-tool/cross-incident/out-of-window/
duplicate-proposal refusal happens here for free, by reuse -- see
`tool_wrappers.py:465-511`.

The durable RESERVED/SETTLED receipt that graph state and `receipts.jsonl`
see is always the one this module's own `dispatch_registry` call produces
-- the MCP round trip is what happens *inside* that existing lifecycle,
translating the child's response back into a `CheckOutcome`/
`RunbookCheckOutcome`, never a replacement for it.
"""

from collections.abc import Callable, Mapping

from causalops.domain import (
    Budgets,
    CheckOutcome,
    EvidenceKind,
    IncidentScope,
    ReasonCode,
    RetrievalMode,
    RunbookCheckOutcome,
    StoredIncident,
    ToolOutcome,
)
from causalops.evidence import failed_check
from causalops.live_setup import REPLAY_FIXTURE
from causalops.mcp_child_process import (
    McpChildProcess,
    McpChildTimeoutError,
    McpChildUnavailableError,
)
from causalops.mcp_stdio import McpProtocolError, McpToolCallResult
from causalops.models import (
    ReplayReasoningModel,
    ReplayToolCallingModel,
    ToolCallingModel,
)
from causalops.telemetry import RunPaths
from causalops.tool_wrappers import DispatchResult, ToolWrapper, dispatch_registry
from causalops.tools import SearchRunbooksArguments, ToolArguments, ToolName

REPLAY_MODEL_NAME = "replay"

_KIND_BY_TOOL: Mapping[ToolName, EvidenceKind] = {
    ToolName.QUERY_METRIC: EvidenceKind.METRIC,
    ToolName.QUERY_LOGS: EvidenceKind.LOG,
    ToolName.LIST_RECENT_CHANGES: EvidenceKind.CHANGE,
    ToolName.GET_TOPOLOGY: EvidenceKind.TOPOLOGY,
}


def _parse_dispatch_result(result: McpToolCallResult) -> DispatchResult:
    if not result.content:
        raise McpProtocolError("MCP tool result carried no content")
    try:
        return DispatchResult.model_validate_json(result.content[0].text)
    except ValueError as error:
        raise McpProtocolError(
            "MCP tool result content is not a valid DispatchResult"
        ) from error


def _forwarding_run_check(
    tool: ToolName, child: McpChildProcess, timeout_seconds: float
) -> Callable[[ToolArguments, IncidentScope], CheckOutcome]:
    kind = _KIND_BY_TOOL[tool]

    def run_check(arguments: ToolArguments, scope: IncidentScope) -> CheckOutcome:
        del scope  # the server independently derives its own scope at start()
        try:
            result = child.call(
                tool, arguments.model_dump(mode="json"), timeout_seconds
            )
        except McpChildTimeoutError:
            return failed_check(
                kind,
                tool.value,
                ToolOutcome.TIMEOUT,
                ReasonCode.TOOL_TIMEOUT,
                "MCP child did not respond in time",
            )
        except McpChildUnavailableError:
            return failed_check(
                kind,
                tool.value,
                ToolOutcome.UNAVAILABLE,
                ReasonCode.TOOL_UNAVAILABLE,
                "MCP child is unavailable",
            )
        except McpProtocolError:
            return failed_check(
                kind,
                tool.value,
                ToolOutcome.ERROR,
                ReasonCode.TOOL_ERROR,
                "MCP child returned a malformed response",
            )
        # McpChildDiedMidCallError, and anything else, propagates uncaught --
        # see mcp_child_process.py's module docstring.
        dispatch_result = _parse_dispatch_result(result)
        receipt = dispatch_result.receipt
        if (
            receipt.outcome is ToolOutcome.EXECUTED
            and dispatch_result.evidence is not None
        ):
            evidence = dispatch_result.evidence
            return CheckOutcome(
                outcome=ToolOutcome.EXECUTED,
                kind=evidence.kind,
                source=evidence.source,
                summary=evidence.summary,
                payload=evidence.payload,
                duration_ms=receipt.duration_ms,
            )
        return failed_check(
            kind,
            tool.value,
            receipt.outcome or ToolOutcome.ERROR,
            receipt.reason_code or ReasonCode.TOOL_ERROR,
            "MCP child reported a non-executed outcome",
        )

    return run_check


def _forwarding_run_search(
    child: McpChildProcess, timeout_seconds: float
) -> Callable[[SearchRunbooksArguments, IncidentScope], RunbookCheckOutcome]:
    def run_search(
        arguments: SearchRunbooksArguments, scope: IncidentScope
    ) -> RunbookCheckOutcome:
        del scope
        try:
            result = child.call(
                ToolName.SEARCH_RUNBOOKS,
                arguments.model_dump(mode="json"),
                timeout_seconds,
            )
        except McpChildTimeoutError:
            return RunbookCheckOutcome(
                outcome=ToolOutcome.TIMEOUT,
                retrieval_mode=RetrievalMode.FTS5_LEXICAL,
                reason_code=ReasonCode.TOOL_TIMEOUT,
            )
        except McpChildUnavailableError:
            return RunbookCheckOutcome(
                outcome=ToolOutcome.UNAVAILABLE,
                retrieval_mode=RetrievalMode.FTS5_LEXICAL,
                reason_code=ReasonCode.TOOL_UNAVAILABLE,
            )
        except McpProtocolError:
            return RunbookCheckOutcome(
                outcome=ToolOutcome.ERROR,
                retrieval_mode=RetrievalMode.FTS5_LEXICAL,
                reason_code=ReasonCode.TOOL_ERROR,
            )
        dispatch_result = _parse_dispatch_result(result)
        return RunbookCheckOutcome(
            outcome=dispatch_result.receipt.outcome or ToolOutcome.ERROR,
            passages=dispatch_result.passages,
            retrieval_mode=dispatch_result.retrieval_mode or RetrievalMode.FTS5_LEXICAL,
            reason_code=dispatch_result.receipt.reason_code,
            duration_ms=dispatch_result.receipt.duration_ms,
        )

    return run_search


def build_mcp_tool_registry(
    child: McpChildProcess, budgets: Budgets
) -> Mapping[ToolName, ToolWrapper]:
    """Real reuse of `dispatch_registry` -- no new dispatch logic here."""
    timeout_seconds = float(budgets.tool_timeout_seconds)
    return dispatch_registry(
        run_metric=_forwarding_run_check(ToolName.QUERY_METRIC, child, timeout_seconds),
        run_logs=_forwarding_run_check(ToolName.QUERY_LOGS, child, timeout_seconds),
        run_changes=_forwarding_run_check(
            ToolName.LIST_RECENT_CHANGES, child, timeout_seconds
        ),
        run_topology=_forwarding_run_check(
            ToolName.GET_TOPOLOGY, child, timeout_seconds
        ),
        run_search=_forwarding_run_search(child, timeout_seconds),
    )


class McpBackedReplayRuntimeWiring:
    """MCP-backed alternative to `live_setup.HostedReplayRuntimeWiring`.

    Same replay reasoning model, same fixture -- only the tool registry's
    transport differs. Spawns one child process per investigation; the
    returned teardown callback must be invoked on every exit path (see
    `live_setup.ReplayRuntimeWiring`'s widened 4-tuple contract)."""

    def build(
        self, incident: StoredIncident, paths: RunPaths, budgets: Budgets
    ) -> tuple[
        ToolCallingModel, Mapping[ToolName, ToolWrapper], str, Callable[[], None]
    ]:
        child = McpChildProcess()
        child.start(paths.root, incident.scope, budgets)
        registry = build_mcp_tool_registry(child, budgets)
        replay_model = ReplayToolCallingModel(
            ReplayReasoningModel(
                REPLAY_FIXTURE,
                substitutions={
                    "incident_id": incident.scope.incident_id,
                    "window_start": incident.scope.started_at.isoformat(),
                    "window_end": incident.scope.ended_at.isoformat(),
                    "symptom_evidence_id": incident.packet.symptom_evidence_id,
                },
            )
        )
        return replay_model, registry, REPLAY_MODEL_NAME, child.close
