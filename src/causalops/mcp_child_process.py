"""Client-side subprocess launcher and supervisor for the local-stdio MCP
observability child.

No code in this repository previously supervised a long-lived child
process; this module is that supervision, kept deliberately small. It
drives the already-built `mcp_stdio.McpStdioSession` state machine over
real OS pipes -- it adds no new protocol logic of its own.

Fail-safe classification (matched against existing `tool_wrappers.py`
receipt contracts, not invented here) lives one layer up, in
`mcp_client_registry.py`'s `run_check` forwarders -- this module only
raises distinctly-typed exceptions so that layer can tell the failure
modes apart:

- `McpChildTimeoutError`: no response within the allotted time. The
  session is poisoned; the next `call()` respawns.
- `McpChildUnavailableError`: the child was dead (or poisoned) and a
  respawn attempt itself failed.
- `McpChildDiedMidCallError`: the child was alive when a request was sent
  but closed its stdout before a response line arrived. Deliberately a
  plain, uncaught-by-design failure one layer up -- see that module.
- `McpProtocolError` (from `mcp_stdio`): the child responded, but the
  response violated the wire protocol. Not a process-supervision failure,
  so it is left to propagate from `mcp_stdio` unchanged rather than
  wrapped here.
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from pydantic import JsonValue

from causalops.domain import Budgets, IncidentScope
from causalops.mcp_stdio import (
    McpStdioSession,
    McpToolCallResult,
    decode_jsonrpc_line,
    encode_jsonrpc_line,
)
from causalops.tools import ToolName

_EOF = object()


class McpChildProcessError(RuntimeError):
    """Base class for MCP child process supervision failures."""


class McpChildTimeoutError(McpChildProcessError):
    """No response arrived within the allotted time."""


class McpChildUnavailableError(McpChildProcessError):
    """The child was dead and a respawn attempt itself failed."""


class McpChildDiedMidCallError(McpChildProcessError):
    """The child closed stdout while a request was outstanding.

    Deliberately not caught by `mcp_client_registry.py`'s forwarders --
    it is meant to propagate the same way a raising direct backend does,
    per `tool_wrappers.py`'s own "a raising backend leaves a visible
    RESERVED receipt" contract.
    """


class McpChildProcess:
    """Owns one MCP server child process for exactly one investigation."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._session: McpStdioSession | None = None
        self._queue: queue.Queue[object] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._poisoned = True
        self._root: Path | None = None
        self._scope: IncidentScope | None = None
        self._budgets: Budgets | None = None

    def start(self, root: Path, scope: IncidentScope, budgets: Budgets) -> None:
        """Spawn a fresh child and complete its MCP handshake.

        Any failure here is a genuine start-time refusal -- it must
        propagate out of the composition root's `build()`, never be
        swallowed per-call. See `mcp_client_registry.McpBackedReplayRuntimeWiring`.
        """
        self._root, self._scope, self._budgets = root, scope, budgets
        self._queue = queue.Queue()
        self._poisoned = False
        self._process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-m",
                "causalops.mcp_server_main",
                str(root),
                scope.model_dump_json(),
                budgets.model_dump_json(),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={"PATH": os.environ.get("PATH", "")},
        )
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._session = McpStdioSession()
        self._handshake()

    def _drain_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._queue.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._queue.put(_EOF)

    def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        try:
            for line in self._process.stderr:
                print(f"[mcp child] {line.rstrip()}", file=sys.stderr)
        except (OSError, ValueError):
            pass

    def _write(self, message: dict[str, JsonValue]) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(encode_jsonrpc_line(message))
        self._process.stdin.flush()

    def _read_one(self, timeout_seconds: float) -> dict[str, JsonValue]:
        try:
            item = self._queue.get(timeout=timeout_seconds)
        except queue.Empty as error:
            self._poisoned = True
            raise McpChildTimeoutError(
                "MCP child did not respond within the allotted time"
            ) from error
        if item is _EOF:
            self._poisoned = True
            raise McpChildDiedMidCallError("MCP child closed stdout before responding")
        assert isinstance(item, str)
        return decode_jsonrpc_line(item)

    def _handshake(self, timeout_seconds: float = 30.0) -> None:
        assert self._session is not None
        self._write(self._session.initialize_request())
        self._session.accept_initialize_response(self._read_one(timeout_seconds))
        self._write(self._session.initialized_notification())
        self._write(self._session.tools_list_request())
        self._session.accept_tools_list_response(self._read_one(timeout_seconds))

    def respawn_if_dead(self) -> None:
        """Called at the top of every `call()`. Relaunches and re-verifies
        the handshake if the child exited or a prior call poisoned it --
        the entire meaning of "reconnect" here: no persistent network
        session to resume, only a local child to relaunch."""
        if (
            not self._poisoned
            and self._process is not None
            and self._process.poll() is None
        ):
            return
        assert self._root is not None
        assert self._scope is not None
        assert self._budgets is not None
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup before respawn
            pass
        try:
            self.start(self._root, self._scope, self._budgets)
        except Exception as error:
            raise McpChildUnavailableError(
                "MCP child could not be respawned"
            ) from error

    def call(
        self, tool: ToolName, arguments: dict[str, JsonValue], timeout_seconds: float
    ) -> McpToolCallResult:
        self.respawn_if_dead()
        assert self._session is not None
        request = self._session.tool_call_request(tool, arguments)
        self._write(request)
        response = self._read_one(timeout_seconds)
        return self._session.accept_tool_call_response(response)

    def close(self) -> None:
        process = self._process
        self._process = None
        self._session = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=5.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5.0)
