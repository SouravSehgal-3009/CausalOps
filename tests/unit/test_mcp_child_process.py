"""Real-subprocess tests for the local-stdio MCP transport.

Spawns the actual `causalops.mcp_server_main` entry point as a real OS
process against `tmp_path` fixtures -- no Docker, no VM, laptop-safe.
`mcp_policy_adapter._APPROVED_MCP_DISPATCH` is genuinely `None` today, so
every `tools/call` here is correctly refused by the safe default
(`PolicyApprovalRequiredExecutor`) -- these tests prove the transport
itself (framing, handshake, manifest verification, crash/timeout/respawn),
not real backend dispatch. Real dispatch becomes testable once a reviewed
approval record exists -- see `mcp_policy_adapter.py`'s own comment.
"""

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_incident import incident_scope

from causalops.domain import Budgets
from causalops.mcp_child_process import (
    McpChildDiedMidCallError,
    McpChildProcess,
    McpChildTimeoutError,
)
from causalops.tools import GetTopologyArguments, ToolName


@pytest.fixture
def child(tmp_path: Path) -> Iterator[McpChildProcess]:
    process = McpChildProcess()
    process.start(tmp_path, incident_scope(), Budgets())
    yield process
    process.close()


def test_real_subprocess_completes_the_full_handshake(child: McpChildProcess) -> None:
    """`start()` already drove initialize -> initialized -> tools/list to
    completion (or raised) -- reaching this line at all is the proof."""
    assert child._process is not None  # noqa: SLF001 - white-box process check
    assert child._process.poll() is None  # noqa: SLF001 - still alive


def test_unapproved_dispatch_is_refused_safely_not_crashed(
    child: McpChildProcess,
) -> None:
    """Today `_APPROVED_MCP_DISPATCH` is None, so every real call must come
    back as a clean MCP protocol error (`is_error`), never a subprocess
    crash or a raw exception leaking through."""
    from causalops.mcp_stdio import McpProtocolError

    with pytest.raises(McpProtocolError, match="error result"):
        child.call(
            ToolName.GET_TOPOLOGY,
            GetTopologyArguments(incident_id=incident_scope().incident_id).model_dump(
                mode="json"
            ),
            timeout_seconds=10.0,
        )
    # The process is still alive and usable after a clean refusal -- a
    # policy refusal is not a process-level failure.
    assert child._process is not None  # noqa: SLF001
    assert child._process.poll() is None  # noqa: SLF001


def test_killed_child_raises_died_mid_call_not_a_hang(tmp_path: Path) -> None:
    child = McpChildProcess()
    child.start(tmp_path, incident_scope(), Budgets())
    try:
        assert child._process is not None  # noqa: SLF001
        child._process.kill()  # noqa: SLF001 - simulate a real crash
        with pytest.raises(McpChildDiedMidCallError):
            child.call(
                ToolName.GET_TOPOLOGY,
                GetTopologyArguments(
                    incident_id=incident_scope().incident_id
                ).model_dump(mode="json"),
                timeout_seconds=10.0,
            )
    finally:
        child.close()


def test_call_after_death_respawns_and_reverifies_the_handshake(
    tmp_path: Path,
) -> None:
    child = McpChildProcess()
    child.start(tmp_path, incident_scope(), Budgets())
    try:
        assert child._process is not None  # noqa: SLF001
        first_pid = child._process.pid  # noqa: SLF001
        child._process.kill()  # noqa: SLF001
        child._process.wait(timeout=5.0)  # noqa: SLF001
        from causalops.mcp_stdio import McpProtocolError

        # respawn_if_dead() runs at the top of call(); the refusal below
        # (not a crash) proves the respawn + re-handshake both succeeded.
        with pytest.raises(McpProtocolError):
            child.call(
                ToolName.GET_TOPOLOGY,
                GetTopologyArguments(
                    incident_id=incident_scope().incident_id
                ).model_dump(mode="json"),
                timeout_seconds=10.0,
            )
        assert child._process is not None  # noqa: SLF001
        assert child._process.pid != first_pid  # noqa: SLF001 - a new process
        assert child._process.poll() is None  # noqa: SLF001
    finally:
        child.close()


def test_a_hung_child_times_out_and_poisons_the_session(tmp_path: Path) -> None:
    child = McpChildProcess()
    child.start(tmp_path, incident_scope(), Budgets())
    try:
        assert child._process is not None and child._process.stdin is not None  # noqa: SLF001
        # Write a syntactically-invalid partial line so the child's
        # readline() blocks forever waiting for a newline that never
        # comes, instead of ever producing a response -- a real hang, not
        # a simulated one.
        child._process.stdin.write("{")  # noqa: SLF001
        child._process.stdin.flush()  # noqa: SLF001
        start = time.monotonic()
        with pytest.raises(McpChildTimeoutError):
            child._read_one(timeout_seconds=1.0)  # noqa: SLF001
        assert time.monotonic() - start < 5.0
    finally:
        child.close()
