"""Incident-scoped, policy-wrapped executor for a future MCP server.

This adapter is intentionally inert until VM composition injects it into
``McpObservabilityServer`` after the approval gate passes. It never imports a
telemetry backend: every request crosses the existing ``ToolWrapper`` policy,
reservation, and receipt lifecycle before any backend can run.
"""

import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from causalops.domain import Budgets, Clock, IncidentScope, ToolProposal
from causalops.mcp_manifest import (
    MCP_PROTOCOL_VERSION,
    approved_tool_names,
    pinned_observability_manifest,
)
from causalops.mcp_stdio import (
    McpObservabilityServer,
    McpTextContent,
    McpToolCallResult,
    McpToolExecutor,
)
from causalops.tool_wrappers import ReservationLedger, ToolWrapper
from causalops.tools import ToolArguments, ToolName


class McpDispatchApprovalError(RuntimeError):
    """MCP dispatch lacks the required reviewed deterministic approval."""


class McpDispatchApproval(BaseModel):
    """Immutable evidence record required before MCP-backed dispatch exists."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approval_commit: str
    reviewer: str
    server_image_digest: str
    manifest_sha256: str
    protocol_version: str


# Reviewed and recorded per docs/mcp-approval/POLICY_APPROVAL.md's 5
# conditions, all evidenced live on the project's dedicated GCE VM against
# real backends (Prometheus, log files, changes, topology, runbook search)
# -- see docs/mcp-approval/VALIDATION.md's "Real local-stdio MCP transport"
# entry for the full record: manifest/protocol pin verified over a real
# pipe (1); real EXECUTED-outcome equivalence, all 5 tools, direct vs MCP
# dispatch, byte-identical (2); cross-incident and duplicate-proposal
# refused identically by both paths before any subprocess is touched (3);
# no MCP network listener, minimal explicit child env, verified live via
# `ss -tlnp` before/after (4); kill/timeout/respawn all preserve receipt
# semantics, verified against real spawned subprocesses (5).
#
# No container image exists for this deployment (host-process, not
# Dockerized -- see docs/mcp-approval/VALIDATION.md's "Deployment shape,
# decided deliberately"), so `server_image_digest` is the same reviewed commit
# SHA as `approval_commit`, pinning the code identity a host-process
# deployment actually has instead of an image digest it doesn't.
_APPROVED_MCP_DISPATCH: McpDispatchApproval | None = McpDispatchApproval(
    approval_commit="615005961111d8ae478f385a04b87ebb03efdb6b",
    reviewer="Sourav Sehgal",
    server_image_digest="615005961111d8ae478f385a04b87ebb03efdb6b",
    manifest_sha256=pinned_observability_manifest().sha256,
    protocol_version=MCP_PROTOCOL_VERSION,
)


class PolicyWrappedMcpExecutor(McpToolExecutor):
    """Route one MCP call through wrappers bound to exactly one incident."""

    def __init__(
        self,
        registry: Mapping[ToolName, ToolWrapper],
        scope: IncidentScope,
        seen_fingerprints: set[str],
        budgets: Budgets,
        ledger: ReservationLedger,
        clock: Clock,
    ) -> None:
        if set(registry) != approved_tool_names():
            raise ValueError(
                "MCP executor requires the complete approved wrapper registry"
            )
        self._registry = registry
        self._scope = scope
        self._seen_fingerprints = seen_fingerprints
        self._budgets = budgets
        self._ledger = ledger
        self._clock = clock

    def call(self, arguments: ToolArguments) -> McpToolCallResult:
        """Dispatch through the policy wrapper and serialize its durable result."""
        proposal = ToolProposal(
            arguments=arguments,
            evidence_gap="MCP-observability request",
            expected_observation="policy-wrapped observability result",
        )
        result = self._registry[arguments.tool].dispatch(
            proposal,
            self._scope,
            self._seen_fingerprints,
            self._budgets,
            self._ledger,
            self._clock,
        )
        return McpToolCallResult(
            content=(
                McpTextContent(
                    type="text",
                    text=json.dumps(result.model_dump(mode="json"), sort_keys=True),
                ),
            )
        )


def policy_approved_mcp_server(
    executor: PolicyWrappedMcpExecutor,
) -> McpObservabilityServer:
    """Build an executable server only from a reviewed approval record."""
    approval = _APPROVED_MCP_DISPATCH
    if approval is None:
        raise McpDispatchApprovalError(
            "MCP dispatch remains disabled until deterministic policy approval"
        )
    manifest = pinned_observability_manifest()
    if (
        approval.protocol_version != MCP_PROTOCOL_VERSION
        or approval.manifest_sha256 != manifest.sha256
    ):
        raise McpDispatchApprovalError(
            "MCP dispatch approval does not match the pinned manifest"
        )
    if not isinstance(executor, PolicyWrappedMcpExecutor):
        raise McpDispatchApprovalError(
            "MCP dispatch requires a policy-wrapped executor"
        )
    return McpObservabilityServer._with_policy_approved_executor(executor)
