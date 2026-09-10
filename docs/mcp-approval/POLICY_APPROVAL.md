# MCP Dispatch Approval Gate

Do not replace direct telemetry backends with MCP transport merely because an
MCP server starts or passes discovery. The approved manifest validates a
capability boundary; it does not grant execution authority.

A reviewer may approve an MCP-backed `ToolWrapper` composition only when all
of the following are recorded against one immutable server image and manifest
hash:

1. `tools/list` exactly passes `verify_discovered_tools` at the pinned MCP
   protocol revision.
2. Each of the five approved requests produces the same normalized outcome,
   receipt lifecycle, evidence/runbook separation, and reason code as the
   existing direct backend over the deterministic replay fixtures.
3. Malformed input, an unknown tool, a cross-incident topology request, an
   out-of-window request, and a duplicate proposal are refused by the existing
   policy wrapper before any MCP invocation.
4. The stdio child writes only JSON-RPC to stdout, uses no network endpoint,
   inherits no provider credentials, and cannot discover or invoke a tool not
   in the checked-in manifest.
5. Crash, timeout, and reconnect behavior preserves the existing durable
   `RESERVED`/`SETTLED` receipt semantics and fails safe when the child is
   unavailable.

The approval commit must name the server image digest, MCP protocol revision,
manifest hash, fixture results, and reviewer. It must also add a reviewed,
non-null `_APPROVED_MCP_DISPATCH` record in `mcp_policy_adapter.py` that
matches the pinned manifest. Until then, `PolicyApprovalRequiredExecutor` is
the required safe default and `policy_approved_mcp_server()` fails closed.
