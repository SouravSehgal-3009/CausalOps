"""Pinned capability manifest for Phase 3's local observability MCP server.

MCP discovery is useful for detecting a server/configuration mismatch, but it
is not an authorization mechanism. The graph continues to receive its
existing policy-wrapped registry until a separately approved composition root
injects an MCP-backed implementation. This module gives that future boundary
one immutable, reviewable set of read-only tool schemas to verify first.
"""

import hashlib
import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    QueryLogsArguments,
    QueryMetricArguments,
    SearchRunbooksArguments,
    ToolName,
)

MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_SERVER_NAME = "causalops-observability"
MCP_SERVER_VERSION = "1"


class McpManifestError(RuntimeError):
    """A local MCP server does not match the reviewed capability manifest."""


class McpToolDefinition(BaseModel):
    """The complete approved shape of one model-callable MCP tool."""

    model_config = ConfigDict(frozen=True)

    name: ToolName
    input_schema: dict[str, JsonValue]


class McpToolDiscovery(BaseModel):
    """Untrusted shape returned by MCP ``tools/list`` discovery."""

    # MCP descriptions and annotations are deliberately ignored: they are
    # server-provided hints, not capability grants. Name and input schema are
    # the only fields this manifest verifies.
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    name: str
    input_schema: dict[str, JsonValue] = Field(alias="inputSchema")


class McpToolManifest(BaseModel):
    """The closed, read-only capability grant for the observability server."""

    model_config = ConfigDict(frozen=True)

    protocol_version: str
    server_name: str
    server_version: str
    tools: tuple[McpToolDefinition, ...]

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _canonical_json(value: JsonValue) -> str:
    """Encode JSON for exact, type-preserving trust-boundary comparison."""
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise McpManifestError("MCP discovery schema is not finite JSON") from error


_TOOL_ARGUMENT_MODELS: tuple[tuple[ToolName, type[BaseModel]], ...] = (
    (ToolName.QUERY_METRIC, QueryMetricArguments),
    (ToolName.QUERY_LOGS, QueryLogsArguments),
    (ToolName.LIST_RECENT_CHANGES, ListRecentChangesArguments),
    (ToolName.GET_TOPOLOGY, GetTopologyArguments),
    (ToolName.SEARCH_RUNBOOKS, SearchRunbooksArguments),
)

# A schema change must deliberately update this reviewed value and the
# corresponding manifest regression test. It prevents an edit to a Pydantic
# arguments model from silently expanding the MCP capability grant.
PINNED_MANIFEST_SHA256 = (
    "7886e3705cee64ca76e2a79cb72a44bfe0ac703a01465f8f483461c3b164074a"
)


def _build_manifest() -> McpToolManifest:
    return McpToolManifest(
        protocol_version=MCP_PROTOCOL_VERSION,
        server_name=MCP_SERVER_NAME,
        server_version=MCP_SERVER_VERSION,
        tools=tuple(
            McpToolDefinition(name=name, input_schema=model.model_json_schema())
            for name, model in _TOOL_ARGUMENT_MODELS
        ),
    )


def pinned_observability_manifest() -> McpToolManifest:
    """Return the approved manifest only if source schemas still match its pin."""
    manifest = _build_manifest()
    if manifest.sha256 != PINNED_MANIFEST_SHA256:
        raise McpManifestError(
            "observability MCP schemas changed; review and update the manifest pin"
        )
    return manifest


def verify_discovered_tools(discovered: Sequence[McpToolDiscovery]) -> McpToolManifest:
    """Verify discovery exactly, without granting any discovered capability.

    Server-provided names, schemas, descriptions, and annotations are all
    untrusted. A successful verification returns the separately pinned
    manifest, never the discovered values, so a server cannot use discovery
    to add, remove, rename, or alter a tool the model may call.
    """
    manifest = pinned_observability_manifest()
    expected = {tool.name.value: tool.input_schema for tool in manifest.tools}
    actual: dict[str, dict[str, JsonValue]] = {}
    for tool in discovered:
        if tool.name in actual:
            raise McpManifestError(f"MCP discovery repeats tool {tool.name!r}")
        actual[tool.name] = tool.input_schema
    if set(actual) != set(expected):
        raise McpManifestError("MCP discovery does not match the approved tool set")
    for name, expected_schema in expected.items():
        if _canonical_json(actual[name]) != _canonical_json(expected_schema):
            raise McpManifestError(f"MCP schema differs for approved tool {name!r}")
    return manifest


def approved_tool_names() -> frozenset[ToolName]:
    """Expose the closed capability set without exposing discovery output."""
    return frozenset(tool.name for tool in pinned_observability_manifest().tools)
