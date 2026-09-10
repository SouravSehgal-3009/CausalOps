"""`McpBackedLiveRuntimeWiring` -- the hosted API's opt-in live-model wiring
(`CAUSALOPS_HOSTED_LIVE_MODEL`, `api_runtime.app()`). Every safety property
of the real `build_claude_model_and_mcp_registry` it delegates to
(credential gate, cost ceiling, the MCP approval gate the spawned child
itself enforces) is that function's own concern, already covered by its
own callers' tests (`test_evaluate_cli.py`) -- these tests only prove this
thin wrapper reshapes its 5-tuple into the `ReplayRuntimeWiring` Protocol's
4-tuple correctly, and that its `release()` closes both the child process
and the ledger connection.
"""

import sqlite3
from pathlib import Path

import pytest
from fake_incident import alert_packet, incident_scope, packet_evidence

from causalops import mcp_client_registry
from causalops.api import ScenarioFamily
from causalops.domain import Budgets, StoredIncident
from causalops.mcp_client_registry import McpBackedLiveRuntimeWiring
from causalops.telemetry import RunPaths


def _stored_incident() -> StoredIncident:
    return StoredIncident(
        scope=incident_scope(), packet=alert_packet(), evidence=packet_evidence()
    )


def test_build_reshapes_the_five_tuple_into_the_protocols_four_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_conn = sqlite3.connect(":memory:")
    child_closed = []

    def fake_build_claude_model_and_mcp_registry(
        incident: StoredIncident,
        paths: RunPaths,
        budgets: Budgets,
        db_path: Path,
        environment: object = None,
    ) -> tuple[object, object, str, sqlite3.Connection, object]:
        return (
            "the-model",
            "the-registry",
            "claude-sonnet-5",
            ledger_conn,
            lambda: child_closed.append(True),
        )

    monkeypatch.setattr(
        mcp_client_registry,
        "build_claude_model_and_mcp_registry",
        fake_build_claude_model_and_mcp_registry,
    )
    wiring = McpBackedLiveRuntimeWiring(tmp_path / "checkpoints.db")

    model, registry, model_name, release = wiring.build(
        _stored_incident(),
        RunPaths(root=tmp_path / "runs" / "incident-1"),
        Budgets(),
        family=ScenarioFamily.CONFIGURATION_CHANGE,
    )

    assert model == "the-model"
    assert registry == "the-registry"
    assert model_name == "claude-sonnet-5"

    release()

    assert child_closed == [True]
    with pytest.raises(sqlite3.ProgrammingError):
        ledger_conn.execute("SELECT 1")


def test_build_passes_through_db_path_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_build_claude_model_and_mcp_registry(
        incident: StoredIncident,
        paths: RunPaths,
        budgets: Budgets,
        db_path: Path,
        environment: object = None,
    ) -> tuple[object, object, str, sqlite3.Connection, object]:
        captured["db_path"] = db_path
        captured["environment"] = environment
        return ("model", "registry", "name", sqlite3.connect(":memory:"), lambda: None)

    monkeypatch.setattr(
        mcp_client_registry,
        "build_claude_model_and_mcp_registry",
        fake_build_claude_model_and_mcp_registry,
    )
    db_path = tmp_path / "checkpoints.db"
    env = {"ANTHROPIC_API_KEY": "test-key"}
    wiring = McpBackedLiveRuntimeWiring(db_path, env)

    _, _, _, release = wiring.build(
        _stored_incident(),
        RunPaths(root=tmp_path / "runs" / "incident-1"),
        Budgets(),
        family=ScenarioFamily.AMBIGUOUS_TELEMETRY,
    )
    release()

    assert captured["db_path"] == db_path
    assert captured["environment"] == env


def test_build_accepts_any_family_without_using_it_for_fixture_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike the replay wirings, live mode does real diagnostic work --
    `family` is part of the `ReplayRuntimeWiring` Protocol's signature but
    never reaches the faked call at all."""
    calls = []

    def fake_build_claude_model_and_mcp_registry(
        incident: StoredIncident,
        paths: RunPaths,
        budgets: Budgets,
        db_path: Path,
        environment: object = None,
    ) -> tuple[object, object, str, sqlite3.Connection, object]:
        calls.append((incident, paths, budgets, db_path, environment))
        return ("model", "registry", "name", sqlite3.connect(":memory:"), lambda: None)

    monkeypatch.setattr(
        mcp_client_registry,
        "build_claude_model_and_mcp_registry",
        fake_build_claude_model_and_mcp_registry,
    )
    wiring = McpBackedLiveRuntimeWiring(tmp_path / "checkpoints.db")

    for family in ScenarioFamily:
        _, _, _, release = wiring.build(
            _stored_incident(),
            RunPaths(root=tmp_path / "runs" / "incident-1"),
            Budgets(),
            family=family,
        )
        release()

    assert len(calls) == len(ScenarioFamily)
