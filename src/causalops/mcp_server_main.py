"""Child-process entry point for the local-stdio MCP observability server.

Invoked only as ``python -m causalops.mcp_server_main <root> <scope-json>
<budgets-json>`` by ``mcp_child_process.McpChildProcess``, never run
directly by an operator. Bootstrap config arrives only via argv (never
stdio, which must stay pure JSON-RPC; never inherited environment, which
must never carry a credential into this process). All logging goes to
stderr; stdout carries JSON-RPC responses only.

Without a reviewed ``mcp_policy_adapter._APPROVED_MCP_DISPATCH`` record,
this process still starts and serves ``initialize``/``tools/list``
normally (both are pure protocol/manifest operations, no backend access) --
`mcp_policy_adapter`'s own module comment: "``PolicyApprovalRequiredExecutor``
is the required safe default." Every ``tools/call`` then gets that
executor's clean ``isError: true`` refusal, never a crash or an early
process exit. There is no environment-variable or argv way to override
this -- the only thing that ever changes it is the reviewed source-code
edit to ``_APPROVED_MCP_DISPATCH`` itself.
"""

import sys
from collections.abc import Mapping
from pathlib import Path

from causalops.domain import Budgets, IncidentScope
from causalops.domain import utc_now as _utc_now
from causalops.mcp_policy_adapter import (
    McpDispatchApprovalError,
    PolicyWrappedMcpExecutor,
    policy_approved_mcp_server,
)
from causalops.mcp_stdio import (
    McpObservabilityServer,
    McpProtocolError,
    decode_jsonrpc_line,
    encode_jsonrpc_line,
)
from causalops.prometheus import DEFAULT_PROMETHEUS_URL, run_metric_check
from causalops.telemetry import (
    RunPaths,
    run_changes_check,
    run_logs_check,
    run_topology_check,
)
from causalops.tool_wrappers import (
    ReservationLedger,
    ToolWrapper,
    get_topology_wrapper,
    list_recent_changes_wrapper,
    query_logs_wrapper,
    query_metric_wrapper,
)
from causalops.tools import ToolName


def _build_registry(
    paths: RunPaths, budgets: Budgets
) -> Mapping[ToolName, ToolWrapper]:
    """Mirrors `live_setup._build_tool_registry`'s real-backend wiring for the
    4 tools MCP is approved to serve (same functions, same lambdas) -- the
    only difference is this registry runs inside the MCP child, not the
    graph process. Built as a direct partial dict, not through
    `dispatch_registry` (which requires all 5 tools by design): `search_
    runbooks` is deliberately absent here -- see `mcp_manifest.py`'s own
    comment -- so `approved_tool_names()` and this registry's key set stay
    exactly equal, which `PolicyWrappedMcpExecutor.__init__` enforces."""
    return {
        ToolName.QUERY_METRIC: query_metric_wrapper(
            lambda arguments, scope: run_metric_check(
                arguments, scope, DEFAULT_PROMETHEUS_URL, budgets.tool_timeout_seconds
            )
        ),
        ToolName.QUERY_LOGS: query_logs_wrapper(
            lambda arguments, scope: run_logs_check(arguments, paths)
        ),
        ToolName.LIST_RECENT_CHANGES: list_recent_changes_wrapper(
            lambda arguments, scope: run_changes_check(arguments, paths)
        ),
        ToolName.GET_TOPOLOGY: get_topology_wrapper(
            lambda arguments, scope: run_topology_check(arguments, paths)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        print(
            "usage: python -m causalops.mcp_server_main "
            "<root> <scope-json> <budgets-json>",
            file=sys.stderr,
        )
        return 2
    root_argument, scope_json, budgets_json = arguments
    scope = IncidentScope.model_validate_json(scope_json)
    budgets = Budgets.model_validate_json(budgets_json)
    paths = RunPaths(root=Path(root_argument))

    registry = _build_registry(paths, budgets)
    ledger = ReservationLedger(budgets.executed_tools)
    executor = PolicyWrappedMcpExecutor(
        registry,
        scope,
        seen_fingerprints=set(),
        budgets=budgets,
        ledger=ledger,
        clock=_utc_now,
    )
    try:
        server = policy_approved_mcp_server(executor)
    except McpDispatchApprovalError as error:
        print(
            f"MCP dispatch not approved, serving the safe default "
            f"(every tools/call will be refused): {error}",
            file=sys.stderr,
        )
        server = McpObservabilityServer()

    print("MCP observability server ready", file=sys.stderr)
    for raw_line in sys.stdin:
        try:
            message = decode_jsonrpc_line(raw_line)
        except McpProtocolError as error:
            print(f"MCP protocol error decoding request: {error}", file=sys.stderr)
            continue
        try:
            response = server.handle(message)
        except McpProtocolError as error:
            print(f"MCP protocol error handling request: {error}", file=sys.stderr)
            continue
        if response is not None:
            sys.stdout.write(encode_jsonrpc_line(response))
            sys.stdout.flush()
    print("MCP observability server exiting (stdin closed)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
