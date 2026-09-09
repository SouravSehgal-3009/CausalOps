"""The Phase 3 MCP capability pin is a trust boundary, not convenience data."""

from copy import deepcopy

import pytest

from causalops.mcp_manifest import (
    MCP_PROTOCOL_VERSION,
    McpManifestError,
    McpToolDiscovery,
    approved_tool_names,
    pinned_observability_manifest,
    verify_discovered_tools,
)
from causalops.tools import ToolName


def discovered_manifest() -> list[McpToolDiscovery]:
    manifest = pinned_observability_manifest()
    return [
        McpToolDiscovery(name=tool.name.value, inputSchema=tool.input_schema)
        for tool in manifest.tools
    ]


def test_manifest_pins_the_complete_read_only_tool_set() -> None:
    manifest = pinned_observability_manifest()

    assert manifest.protocol_version == MCP_PROTOCOL_VERSION
    assert approved_tool_names() == frozenset(ToolName)
    assert {tool.name for tool in manifest.tools} == set(ToolName)


def test_matching_discovery_returns_pinned_values_not_server_values() -> None:
    discovered = discovered_manifest()

    verified = verify_discovered_tools(discovered)

    assert verified == pinned_observability_manifest()
    assert verified.tools[0].input_schema is not discovered[0].input_schema


def test_discovery_cannot_add_or_remove_model_capabilities() -> None:
    discovered = discovered_manifest()
    discovered.pop()
    discovered.append(
        McpToolDiscovery(name="delete_everything", inputSchema={"type": "object"})
    )

    with pytest.raises(McpManifestError, match="approved tool set"):
        verify_discovered_tools(discovered)


def test_discovery_cannot_change_an_approved_tool_schema() -> None:
    discovered = discovered_manifest()
    first = discovered[0]
    discovered[0] = McpToolDiscovery(
        name=first.name,
        inputSchema={"type": "object", "additionalProperties": True},
    )

    with pytest.raises(McpManifestError, match="schema differs"):
        verify_discovered_tools(discovered)


def test_discovery_rejects_a_boolean_in_place_of_a_numeric_schema_value() -> None:
    discovered = discovered_manifest()
    index = next(
        index
        for index, tool in enumerate(discovered)
        if tool.name == ToolName.QUERY_LOGS
    )
    schema = deepcopy(discovered[index].input_schema)
    schema["properties"]["row_limit"]["minimum"] = True  # type: ignore[index]
    discovered[index] = McpToolDiscovery(
        name=ToolName.QUERY_LOGS.value,
        inputSchema=schema,
    )

    with pytest.raises(McpManifestError, match="schema differs"):
        verify_discovered_tools(discovered)


def test_discovery_cannot_repeat_a_tool_to_mask_a_missing_one() -> None:
    discovered = discovered_manifest()
    discovered[-1] = discovered[0]

    with pytest.raises(McpManifestError, match="repeats tool"):
        verify_discovered_tools(discovered)
