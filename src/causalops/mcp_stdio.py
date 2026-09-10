"""Pure JSON-RPC state machine for the future local-stdio MCP client.

This module deliberately owns no subprocess, socket, backend, or model
composition. It makes the protocol boundary testable without launching MCP
locally, and it cannot replace the policy-wrapped telemetry registry by
itself. A VM-only transport may later feed its newline-delimited messages
through this state machine after deterministic policy approval.
"""

import json
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from causalops.mcp_manifest import (
    MCP_PROTOCOL_VERSION,
    McpManifestError,
    McpToolDiscovery,
    McpToolManifest,
    pinned_observability_manifest,
    verify_discovered_tools,
)
from causalops.tools import ToolArguments, ToolName

_TOOL_ARGUMENTS: TypeAdapter[ToolArguments] = TypeAdapter(ToolArguments)


class McpProtocolError(RuntimeError):
    """A peer sent an invalid, out-of-order, or unsafe MCP JSON-RPC message."""


class McpSessionState(StrEnum):
    NEW = "new"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    READY = "ready"


class McpTextContent(BaseModel):
    """The only tool-result content accepted by the initial observability client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["text"]
    text: str


class McpToolCallResult(BaseModel):
    """Validated result from a successful, model-visible ``tools/call``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: tuple[McpTextContent, ...]
    is_error: bool = Field(default=False, alias="isError")


class McpToolExecutor(Protocol):
    """VM composition seam for an executor that has already passed policy."""

    def call(self, arguments: ToolArguments) -> McpToolCallResult: ...


class PolicyApprovalRequiredExecutor:
    """Safe default: an MCP server cannot reach a backend by default."""

    def call(self, arguments: ToolArguments) -> McpToolCallResult:
        del arguments
        return McpToolCallResult(
            content=(
                McpTextContent(
                    type="text",
                    text=(
                        "MCP dispatch is unavailable until deterministic "
                        "policy approval"
                    ),
                ),
            ),
            isError=True,
        )


def _parse_finite_float(text: str) -> float:
    """Parse a JSON number without silently converting overflow to infinity."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {text!r}")
    return value


def _validate_strict_tool_arguments(payload: dict[str, JsonValue]) -> ToolArguments:
    """Validate JSON wire values without Pydantic's Python-value coercion."""
    try:
        serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        return _TOOL_ARGUMENTS.validate_json(serialized, strict=True)
    except (TypeError, ValueError) as error:
        raise McpProtocolError(
            "MCP tool arguments do not match the approved schema"
        ) from error


def decode_jsonrpc_line(line: str) -> dict[str, JsonValue]:
    """Parse exactly one newline-free JSON-RPC object from stdio.

    Stdio permits one JSON-RPC message per line. Refusing embedded newlines
    prevents log text or framing ambiguities from being interpreted as a
    protocol message, and refusing non-finite JSON keeps the same boundary as
    the incident-manifest readers.
    """
    payload = line.removesuffix("\n").removesuffix("\r")
    if not payload or "\n" in payload or "\r" in payload:
        raise McpProtocolError("MCP stdio message must be one newline-free line")
    try:
        parsed = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token!r}")
            ),
            parse_float=_parse_finite_float,
        )
    except (TypeError, ValueError) as error:
        raise McpProtocolError("MCP stdio message is not valid JSON") from error
    if not isinstance(parsed, dict) or parsed.get("jsonrpc") != "2.0":
        raise McpProtocolError("MCP message must be a JSON-RPC 2.0 object")
    return parsed


def encode_jsonrpc_line(message: dict[str, JsonValue]) -> str:
    """Encode one JSON-RPC object without allowing stdout contamination."""
    if message.get("jsonrpc") != "2.0":
        raise McpProtocolError("outbound MCP message must use JSON-RPC 2.0")
    try:
        encoded = json.dumps(message, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise McpProtocolError(
            "outbound MCP message is not JSON serializable"
        ) from error
    if "\n" in encoded or "\r" in encoded:
        raise McpProtocolError("outbound MCP message must be one line")
    return f"{encoded}\n"


class McpStdioSession:
    """Strict lifecycle and capability gate for one future stdio connection."""

    def __init__(self) -> None:
        self._state = McpSessionState.NEW
        self._request_id = 0
        self._pending: tuple[int, str] | None = None
        self._manifest: McpToolManifest | None = None

    @property
    def state(self) -> McpSessionState:
        return self._state

    @property
    def manifest(self) -> McpToolManifest:
        if self._manifest is None:
            raise McpProtocolError(
                "MCP manifest is unavailable before verified discovery"
            )
        return self._manifest

    def _request(
        self, method: str, params: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if self._pending is not None:
            raise McpProtocolError("MCP session already has an outstanding request")
        self._request_id += 1
        self._pending = (self._request_id, method)
        return {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

    def initialize_request(self) -> dict[str, JsonValue]:
        if self._state is not McpSessionState.NEW:
            raise McpProtocolError("MCP initialization may occur only once")
        self._state = McpSessionState.INITIALIZING
        return self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "causalops", "version": "1"},
            },
        )

    def _response(
        self, message: dict[str, JsonValue], method: str
    ) -> dict[str, JsonValue]:
        pending = self._pending
        if pending is None or pending[1] != method:
            raise McpProtocolError(f"unexpected MCP response for {method!r}")
        if message.get("jsonrpc") != "2.0":
            raise McpProtocolError("MCP response must use JSON-RPC 2.0")
        response_id = message.get("id")
        if type(response_id) is not int or response_id != pending[0]:
            raise McpProtocolError("MCP response id does not match its request")
        self._pending = None
        error = message.get("error")
        result = message.get("result")
        if error is not None or not isinstance(result, dict):
            raise McpProtocolError(
                f"MCP {method} request did not return an object result"
            )
        return result

    def accept_initialize_response(self, message: dict[str, JsonValue]) -> None:
        result = self._response(message, "initialize")
        if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise McpProtocolError("MCP server negotiated an unpinned protocol version")
        server_info = result.get("serverInfo")
        manifest = pinned_observability_manifest()
        if not isinstance(server_info, dict) or (
            server_info.get("name") != manifest.server_name
            or server_info.get("version") != manifest.server_version
        ):
            raise McpProtocolError(
                "MCP server identity does not match the pinned manifest"
            )
        self._state = McpSessionState.INITIALIZED

    def initialized_notification(self) -> dict[str, JsonValue]:
        if self._state is not McpSessionState.INITIALIZED:
            raise McpProtocolError("MCP initialized notification is out of order")
        self._state = McpSessionState.READY
        return {"jsonrpc": "2.0", "method": "notifications/initialized"}

    def tools_list_request(self) -> dict[str, JsonValue]:
        if self._state is not McpSessionState.READY:
            raise McpProtocolError("MCP tools/list requires an initialized session")
        return self._request("tools/list", {})

    def accept_tools_list_response(
        self, message: dict[str, JsonValue]
    ) -> McpToolManifest:
        result = self._response(message, "tools/list")
        if result.get("nextCursor") is not None:
            raise McpProtocolError(
                "MCP tools/list pagination is not permitted for a pinned manifest"
            )
        try:
            raw_tools = result["tools"]
            if not isinstance(raw_tools, list):
                raise TypeError("tools must be an array")
            tools = [McpToolDiscovery.model_validate(tool) for tool in raw_tools]
        except (KeyError, TypeError, ValidationError) as error:
            raise McpProtocolError("MCP tools/list result is invalid") from error
        try:
            self._manifest = verify_discovered_tools(tools)
        except McpManifestError as error:
            raise McpProtocolError("MCP discovery failed the capability pin") from error
        return self._manifest

    def tool_call_request(
        self, name: ToolName, arguments: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if self._state is not McpSessionState.READY:
            raise McpProtocolError("MCP tools/call requires an initialized session")
        if self._manifest is None:
            raise McpProtocolError("MCP tools/call requires verified discovery")
        payload = {"tool": name.value, **arguments}
        parsed = _validate_strict_tool_arguments(payload)
        if parsed.tool is not name:
            raise McpProtocolError("MCP tool argument name does not match request name")
        return self._request(
            "tools/call",
            {"name": name.value, "arguments": parsed.model_dump(mode="json")},
        )

    def accept_tool_call_response(
        self, message: dict[str, JsonValue]
    ) -> McpToolCallResult:
        result = self._response(message, "tools/call")
        try:
            parsed = McpToolCallResult.model_validate(result)
        except ValidationError as error:
            raise McpProtocolError("MCP tools/call result is invalid") from error
        if parsed.is_error:
            raise McpProtocolError("MCP tool reported an error result")
        return parsed


class McpObservabilityServer:
    """Pure request handler for the future loopback-only stdio subprocess.

    The server handles protocol framing and approved schemas, and its public
    constructor always refuses calls. Only the policy adapter's reviewed
    composition factory may install an executor after deterministic approval.
    """

    def __init__(self) -> None:
        self._executor: McpToolExecutor = PolicyApprovalRequiredExecutor()
        self._state = McpSessionState.NEW
        self._manifest = verify_discovered_tools(
            [
                McpToolDiscovery(name=tool.name.value, inputSchema=tool.input_schema)
                for tool in pinned_observability_manifest().tools
            ]
        )

    @classmethod
    def _with_policy_approved_executor(
        cls, executor: McpToolExecutor
    ) -> "McpObservabilityServer":
        """Internal seam used only by the reviewed policy composition factory."""
        server = cls()
        server._executor = executor
        return server

    @staticmethod
    def _response(
        request_id: JsonValue, result: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _request_id(message: Mapping[str, JsonValue]) -> JsonValue:
        request_id = message.get("id")
        if isinstance(request_id, bool) or not isinstance(
            request_id, (str, int, float)
        ):
            raise McpProtocolError("MCP request id must be a string or number")
        return request_id

    @staticmethod
    def _params(message: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        params = message.get("params", {})
        if not isinstance(params, dict):
            raise McpProtocolError("MCP request params must be an object")
        return params

    def handle(self, message: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
        """Handle one decoded request/notification; callers own stdio I/O."""
        if message.get("jsonrpc") != "2.0":
            raise McpProtocolError("MCP request must use JSON-RPC 2.0")
        method = message.get("method")
        if not isinstance(method, str):
            raise McpProtocolError("MCP request must name a method")
        if method == "notifications/initialized":
            if "id" in message or self._state is not McpSessionState.INITIALIZED:
                raise McpProtocolError("MCP initialized notification is out of order")
            self._state = McpSessionState.READY
            return None
        if "id" not in message:
            raise McpProtocolError("unsupported MCP notification")
        request_id = self._request_id(message)
        if method == "initialize":
            if self._state is not McpSessionState.NEW:
                raise McpProtocolError("MCP initialization may occur only once")
            if self._params(message).get("protocolVersion") != MCP_PROTOCOL_VERSION:
                raise McpProtocolError(
                    "MCP client requested an unpinned protocol version"
                )
            self._state = McpSessionState.INITIALIZED
            return self._response(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": self._manifest.server_name,
                        "version": self._manifest.server_version,
                    },
                },
            )
        if self._state is not McpSessionState.READY:
            raise McpProtocolError("MCP request requires an initialized session")
        if method == "tools/list":
            return self._response(
                request_id,
                {
                    "tools": [
                        {
                            "name": tool.name.value,
                            "inputSchema": tool.input_schema,
                            "annotations": {
                                "readOnlyHint": True,
                                "destructiveHint": False,
                                "idempotentHint": True,
                                "openWorldHint": False,
                            },
                        }
                        for tool in self._manifest.tools
                    ]
                },
            )
        if method != "tools/call":
            raise McpProtocolError("unsupported MCP request method")
        params = self._params(message)
        name = params.get("name")
        raw_arguments = params.get("arguments")
        if not isinstance(name, str) or not isinstance(raw_arguments, dict):
            raise McpProtocolError("MCP tools/call requires name and object arguments")
        try:
            tool_name = ToolName(name)
            arguments = _validate_strict_tool_arguments(
                {"tool": tool_name.value, **raw_arguments}
            )
        except ValueError as error:
            raise McpProtocolError(
                "MCP tools/call arguments are not approved"
            ) from error
        if arguments.tool is not tool_name:
            raise McpProtocolError("MCP tools/call name and arguments disagree")
        result = self._executor.call(arguments)
        return self._response(request_id, result.model_dump(mode="json", by_alias=True))
