"""Real-subprocess tests for the local-stdio MCP transport.

Spawns the actual `causalops.mcp_server_main` entry point as a real OS
process against `tmp_path` fixtures -- no Docker, no VM, laptop-safe. A
reviewed `mcp_policy_adapter._APPROVED_MCP_DISPATCH` record now exists (see
`infra/phase3/VALIDATION.md`), so `tools/call` here actually executes
against the real backend rather than being refused by the safe default.
`tmp_path` starts with no telemetry/topology data written, so a call comes
back as a clean `UNAVAILABLE` outcome rather than `EXECUTED` -- these tests
prove the transport itself (framing, handshake, manifest verification,
crash/timeout/respawn) survives real dispatch cleanly, not the outcome
content; real EXECUTED-outcome equivalence lives in
`test_mcp_policy_equivalence.py`.
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_incident import incident_scope

from causalops.domain import Budgets, PolicyResult, ToolOutcome
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


def test_real_dispatch_with_no_backend_data_returns_clean_unavailable_not_a_crash(
    child: McpChildProcess,
) -> None:
    """The reviewed approval record is real now, so this call actually
    executes against `tmp_path`'s real backend. With no topology manifest
    written there, the real backend correctly answers `UNAVAILABLE` -- a
    normal MCP tool-call result, not a protocol error or a subprocess
    crash."""
    response = child.call(
        ToolName.GET_TOPOLOGY,
        GetTopologyArguments(incident_id=incident_scope().incident_id).model_dump(
            mode="json"
        ),
        timeout_seconds=10.0,
    )
    payload = json.loads(response.content[0].text)

    assert payload["receipt"]["policy_result"] == PolicyResult.ALLOWED.value
    assert payload["receipt"]["outcome"] == ToolOutcome.UNAVAILABLE.value
    assert payload["evidence"] is None
    # The process is still alive and usable after a clean result.
    assert child._process is not None  # noqa: SLF001
    assert child._process.poll() is None  # noqa: SLF001


def test_killed_child_raises_died_mid_call_not_a_hang(tmp_path: Path) -> None:
    """Confirmed live in CI: real OS process-kill timing is not a reliable
    way to exercise this path on every platform -- `Popen.kill()` (POSIX
    SIGKILL vs Windows `TerminateProcess`) does not reach the child at a
    consistent point relative to its own read-handle-write round trip, so
    a write-then-kill version still sometimes got a real answer back.
    Closing the process's own stdout pipe from this thread was tried next
    and does not reliably work either: closing a file object does not
    interrupt a *different* thread already blocked inside a read on it
    (a well-known POSIX/CPython gotcha, not platform-specific this time).

    `_read_one`'s actual job here is simple and independent of how a dead
    child gets discovered: given the `_EOF` sentinel `_drain_stdout`'s own
    `finally` unconditionally queues once its read loop exits for any
    reason, raise `McpChildDiedMidCallError`. Testing that directly, by
    injecting the sentinel, is deterministic on every platform and tests
    the actual unit of behavior that matters -- `_drain_stdout` producing
    that sentinel on a real death is separately relied on by
    `test_call_after_death_respawns_and_reverifies_the_handshake`, which
    already needs a real dead process's threads to wind down cleanly for
    `close()` to succeed."""
    from causalops.mcp_child_process import _EOF

    child = McpChildProcess()
    child.start(tmp_path, incident_scope(), Budgets())
    try:
        child._queue.put(_EOF)  # noqa: SLF001 - simulate the reader thread's own EOF
        with pytest.raises(McpChildDiedMidCallError):
            child._read_one(timeout_seconds=10.0)  # noqa: SLF001
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

        # respawn_if_dead() runs at the top of call(); a clean result
        # below (not a crash) proves the respawn + re-handshake both
        # succeeded.
        response = child.call(
            ToolName.GET_TOPOLOGY,
            GetTopologyArguments(incident_id=incident_scope().incident_id).model_dump(
                mode="json"
            ),
            timeout_seconds=10.0,
        )
        payload = json.loads(response.content[0].text)
        assert payload["receipt"]["policy_result"] == PolicyResult.ALLOWED.value

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
