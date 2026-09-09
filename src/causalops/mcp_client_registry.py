"""Client-side MCP-backed tool registry and runtime wiring.

Builds a `Mapping[ToolName, ToolWrapper]` via the *existing*
`tool_wrappers.dispatch_registry` factory -- the same one
`live_setup._build_tool_registry` uses for direct backends -- except the 4
observability `run_check` closures forward each call to a real MCP child
process instead of calling a backend function directly. Because
`_make_wrapper.dispatch()` always calls `authorize()` and returns an early
denial before `run_check` is ever invoked, malformed/unknown-tool/
cross-incident/out-of-window/duplicate-proposal refusal happens here for
free, by reuse -- see `tool_wrappers.py:465-511`.

`search_runbooks` is deliberately NOT one of the MCP-forwarded tools: it
returns advisory guidance, not an incident observation, and the pinned MCP
manifest (`mcp_manifest.py`) does not approve it as an MCP-callable tool at
all -- the MCP child process cannot serve it even in principle. This
registry answers it the same way `live_setup._build_tool_registry` already
does for the non-MCP path: a direct, local `run_runbook_search` call, never
routed through the child.

The durable RESERVED/SETTLED receipt that graph state and `receipts.jsonl`
see is always the one this module's own `dispatch_registry` call produces
-- the MCP round trip (for the 4 observability tools) is what happens
*inside* that existing lifecycle, translating the child's response back
into a `CheckOutcome`, never a replacement for it.
"""

import os
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path

from causalops.cost_ledger import ensure_cost_ledger_table
from causalops.doctor import API_KEY_VARIABLE
from causalops.domain import (
    Budgets,
    CheckOutcome,
    EvidenceKind,
    IncidentScope,
    ReasonCode,
    StoredIncident,
    ToolOutcome,
)
from causalops.evidence import failed_check
from causalops.live_model import MODEL_NAME as LIVE_MODEL_NAME
from causalops.live_model import LiveClaudeModel
from causalops.live_setup import (
    ENABLE_CLAUDE_VARIABLE,
    REPLAY_FIXTURE,
    ProviderDisabledError,
    claude_enabled,
    live_evaluation_ceiling_usd,
)
from causalops.mcp_child_process import (
    McpChildProcess,
    McpChildTimeoutError,
    McpChildUnavailableError,
)
from causalops.mcp_stdio import McpProtocolError, McpToolCallResult
from causalops.model_profiles import CLAUDE_LEGACY_DISABLED
from causalops.models import (
    ReplayReasoningModel,
    ReplayToolCallingModel,
    ToolCallingModel,
)
from causalops.runbooks import RunbookIndex, run_runbook_search
from causalops.telemetry import RunPaths
from causalops.tool_wrappers import DispatchResult, ToolWrapper, dispatch_registry
from causalops.tools import ToolArguments, ToolName

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


def build_mcp_tool_registry(
    child: McpChildProcess, budgets: Budgets
) -> Mapping[ToolName, ToolWrapper]:
    """Real reuse of `dispatch_registry` -- no new dispatch logic here. The 4
    observability tools forward over the real MCP child; `search_runbooks`
    is answered locally (a fresh `RunbookIndex()` per call, matching
    `live_setup._build_tool_registry`'s own "cheap to rebuild" reasoning),
    never routed through MCP -- see this module's own docstring."""
    timeout_seconds = float(budgets.tool_timeout_seconds)
    runbook_index = RunbookIndex()
    return dispatch_registry(
        run_metric=_forwarding_run_check(ToolName.QUERY_METRIC, child, timeout_seconds),
        run_logs=_forwarding_run_check(ToolName.QUERY_LOGS, child, timeout_seconds),
        run_changes=_forwarding_run_check(
            ToolName.LIST_RECENT_CHANGES, child, timeout_seconds
        ),
        run_topology=_forwarding_run_check(
            ToolName.GET_TOPOLOGY, child, timeout_seconds
        ),
        run_search=lambda arguments, scope: run_runbook_search(
            arguments, runbook_index
        ),
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


def build_claude_model_and_mcp_registry(
    incident: StoredIncident,
    paths: RunPaths,
    budgets: Budgets,
    db_path: Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[
    ToolCallingModel,
    Mapping[ToolName, ToolWrapper],
    str,
    sqlite3.Connection,
    Callable[[], None],
]:
    """Claude, dispatched over the real MCP local-stdio transport.

    `evaluate_cli.py`'s composition root once the roadmap's Phase 3 exit
    criterion is met: the Claude reference evaluation runs "on final
    transport," not the direct in-process registry `live_setup
    .build_model_and_registry` still builds for `causalops investigate`
    (that legacy CLI's own transport choice is out of scope here). Mirrors
    `live_setup.build_model_and_registry`'s "claude" branch line for line
    -- same credential-presence-only check, same cost ledger connection,
    same `LiveClaudeModel` construction -- swapping only `_build_tool_
    registry`'s direct dispatch for `build_mcp_tool_registry`'s real
    spawned child. The caller must invoke the returned teardown
    (`McpChildProcess.close`) on every exit path, exactly like `live_setup
    .ReplayRuntimeWiring`'s widened 4-tuple contract already requires of
    its own callers.
    """
    process_environment = environment if environment is not None else os.environ
    if not claude_enabled(process_environment):
        raise ProviderDisabledError(
            f"{CLAUDE_LEGACY_DISABLED.kind.value} is disabled by "
            f"{ENABLE_CLAUDE_VARIABLE}=false"
        )
    child = McpChildProcess()
    child.start(paths.root, incident.scope, budgets)
    registry = build_mcp_tool_registry(child, budgets)
    ledger_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    ensure_cost_ledger_table(ledger_conn)
    credential_present = bool(process_environment.get(API_KEY_VARIABLE, "").strip())
    live_model = LiveClaudeModel(
        ledger_conn,
        ceiling_usd=live_evaluation_ceiling_usd(process_environment),
        credential_present=credential_present,
    )
    return live_model, registry, LIVE_MODEL_NAME, ledger_conn, child.close
