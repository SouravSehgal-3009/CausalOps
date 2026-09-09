"""Hermetic protocol tests for the unconnected Phase 3 stdio client boundary."""

import pytest

from causalops.mcp_manifest import MCP_PROTOCOL_VERSION, pinned_observability_manifest
from causalops.mcp_stdio import (
    McpObservabilityServer,
    McpProtocolError,
    McpSessionState,
    McpStdioSession,
    McpTextContent,
    McpToolCallResult,
    decode_jsonrpc_line,
    encode_jsonrpc_line,
)
from causalops.tools import ToolName


def response(
    request: dict[str, object], result: dict[str, object]
) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


def initialize_result(
    *, name: str | None = None, version: str | None = None
) -> dict[str, object]:
    manifest = pinned_observability_manifest()
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": {
            "name": manifest.server_name if name is None else name,
            "version": manifest.server_version if version is None else version,
        },
    }


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def call(self, arguments: object) -> McpToolCallResult:
        self.calls.append(arguments)
        return McpToolCallResult(
            content=(McpTextContent(type="text", text="unexpected dispatch"),)
        )


def ready_session() -> McpStdioSession:
    session = McpStdioSession()
    initialize = session.initialize_request()
    session.accept_initialize_response(
        response(initialize, initialize_result())  # type: ignore[arg-type]
    )
    session.initialized_notification()
    listing = session.tools_list_request()
    manifest = pinned_observability_manifest()
    session.accept_tools_list_response(
        response(
            listing,
            {
                "tools": [
                    {"name": tool.name.value, "inputSchema": tool.input_schema}
                    for tool in manifest.tools
                ]
            },
        )  # type: ignore[arg-type]
    )
    return session


def test_stdio_codec_requires_one_json_rpc_message_per_line() -> None:
    encoded = encode_jsonrpc_line({"jsonrpc": "2.0", "method": "tools/list"})

    assert decode_jsonrpc_line(encoded) == {
        "jsonrpc": "2.0",
        "method": "tools/list",
    }
    assert decode_jsonrpc_line('{"jsonrpc":"2.0"}\n') == {"jsonrpc": "2.0"}
    with pytest.raises(McpProtocolError, match="one newline-free"):
        decode_jsonrpc_line('{"jsonrpc":"2.0"}\nnot-json')
    with pytest.raises(McpProtocolError, match="not valid JSON"):
        decode_jsonrpc_line('{"jsonrpc":"2.0","value":NaN}')
    with pytest.raises(McpProtocolError, match="not valid JSON"):
        decode_jsonrpc_line('{"jsonrpc":"2.0","value":1e400}')


def test_session_requires_initialize_then_verified_tools_list() -> None:
    session = McpStdioSession()
    with pytest.raises(McpProtocolError, match="initialized session"):
        session.tools_list_request()

    initialize = session.initialize_request()
    session.accept_initialize_response(
        response(initialize, initialize_result())  # type: ignore[arg-type]
    )
    assert session.state is McpSessionState.INITIALIZED
    assert session.initialized_notification()["method"] == "notifications/initialized"
    assert session.state is McpSessionState.READY


def test_session_refuses_an_unpinned_protocol_or_manifest() -> None:
    session = McpStdioSession()
    initialize = session.initialize_request()
    with pytest.raises(McpProtocolError, match="unpinned protocol"):
        session.accept_initialize_response(
            response(initialize, {"protocolVersion": "future"})  # type: ignore[arg-type]
        )

    session = McpStdioSession()
    initialize = session.initialize_request()
    session.accept_initialize_response(
        response(initialize, initialize_result())  # type: ignore[arg-type]
    )
    session.initialized_notification()
    listing = session.tools_list_request()
    with pytest.raises(McpProtocolError, match="capability pin"):
        session.accept_tools_list_response(
            response(listing, {"tools": [{"name": "surprise", "inputSchema": {}}]})  # type: ignore[arg-type]
        )


def test_session_refuses_an_unpinned_server_identity() -> None:
    session = McpStdioSession()
    initialize = session.initialize_request()

    with pytest.raises(McpProtocolError, match="identity does not match"):
        session.accept_initialize_response(
            response(initialize, initialize_result(name="unexpected-server"))  # type: ignore[arg-type]
        )


def test_session_refuses_a_paginated_tools_list() -> None:
    session = McpStdioSession()
    initialize = session.initialize_request()
    session.accept_initialize_response(
        response(initialize, initialize_result())  # type: ignore[arg-type]
    )
    session.initialized_notification()
    listing = session.tools_list_request()
    manifest = pinned_observability_manifest()

    with pytest.raises(McpProtocolError, match="pagination is not permitted"):
        session.accept_tools_list_response(
            response(
                listing,
                {
                    "tools": [
                        {"name": tool.name.value, "inputSchema": tool.input_schema}
                        for tool in manifest.tools
                    ],
                    "nextCursor": "more-tools",
                },
            )  # type: ignore[arg-type]
        )


def test_tool_call_requires_verified_schema_and_strict_response() -> None:
    session = ready_session()
    with pytest.raises(McpProtocolError, match="approved schema"):
        session.tool_call_request(
            ToolName.GET_TOPOLOGY, {"incident_id": "one", "extra": True}
        )

    request = session.tool_call_request(ToolName.GET_TOPOLOGY, {"incident_id": "one"})
    with pytest.raises(McpProtocolError, match="tools/call result is invalid"):
        session.accept_tool_call_response(
            response(request, {"content": [{"type": "image", "text": "no"}]})  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("row_limit", [True, "1"])
def test_client_refuses_coerced_wire_tool_arguments(row_limit: object) -> None:
    session = ready_session()

    with pytest.raises(McpProtocolError, match="approved schema"):
        session.tool_call_request(
            ToolName.QUERY_LOGS,
            {
                "log_filter": "errors_only",
                "service": "gateway",
                "row_limit": row_limit,
            },  # type: ignore[arg-type]
        )


def test_tool_call_accepts_only_a_successful_text_result() -> None:
    session = ready_session()
    request = session.tool_call_request(ToolName.GET_TOPOLOGY, {"incident_id": "one"})

    result = session.accept_tool_call_response(
        response(request, {"content": [{"type": "text", "text": "ok"}]})  # type: ignore[arg-type]
    )

    assert result.content[0].text == "ok"


def test_server_exposes_only_the_pinned_catalog_and_refuses_unapproved_calls() -> None:
    server = McpObservabilityServer()
    initialize = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
        }
    )
    assert initialize is not None
    assert initialize["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert (
        server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    )
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert listed is not None
    assert {tool["name"] for tool in listed["result"]["tools"]} == {
        tool.value for tool in ToolName
    }
    refused = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": ToolName.GET_TOPOLOGY.value,
                "arguments": {"incident_id": "one"},
            },
        }
    )
    assert refused is not None
    assert refused["result"]["isError"] is True


@pytest.mark.parametrize("row_limit", [True, "1"])
def test_server_refuses_coerced_wire_tool_arguments_before_dispatch(
    row_limit: object,
) -> None:
    server = McpObservabilityServer()
    server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
        }
    )
    server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})

    with pytest.raises(McpProtocolError, match="approved schema"):
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": ToolName.QUERY_LOGS.value,
                    "arguments": {
                        "log_filter": "errors_only",
                        "service": "gateway",
                        "row_limit": row_limit,
                    },
                },
            }  # type: ignore[arg-type]
        )


def test_server_constructor_does_not_accept_an_executor() -> None:
    with pytest.raises(TypeError):
        McpObservabilityServer(RecordingExecutor())  # type: ignore[call-arg]


def test_client_ignores_server_annotations_but_verifies_the_schema() -> None:
    server = McpObservabilityServer()
    session = McpStdioSession()
    initialize = session.initialize_request()
    server_initialize = server.handle(initialize)
    assert server_initialize is not None
    session.accept_initialize_response(server_initialize)
    server.handle(session.initialized_notification())
    listing = session.tools_list_request()
    server_listing = server.handle(listing)
    assert server_listing is not None

    assert (
        session.accept_tools_list_response(server_listing)
        == pinned_observability_manifest()
    )


def test_server_refuses_unknown_and_out_of_order_requests() -> None:
    server = McpObservabilityServer()
    with pytest.raises(McpProtocolError, match="initialized session"):
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
        }
    )
    server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    with pytest.raises(McpProtocolError, match="unsupported MCP request"):
        server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/delete"})
    with pytest.raises(McpProtocolError, match="request id"):
        server.handle({"jsonrpc": "2.0", "id": {}, "method": "tools/list"})


def test_client_refuses_a_response_that_skips_json_rpc_versioning() -> None:
    session = McpStdioSession()
    session.initialize_request()

    with pytest.raises(McpProtocolError, match="response must use JSON-RPC"):
        session.accept_initialize_response(
            {"id": 1, "result": {"protocolVersion": MCP_PROTOCOL_VERSION}}
        )


def test_client_refuses_a_boolean_response_id_for_its_numeric_request() -> None:
    session = McpStdioSession()
    session.initialize_request()

    with pytest.raises(McpProtocolError, match="response id does not match"):
        session.accept_initialize_response(
            {
                "jsonrpc": "2.0",
                "id": True,
                "result": initialize_result(),
            }  # type: ignore[arg-type]
        )
