"""`TECHNICAL_SPEC.md` section 11's second half: "a test proves no tracing
client is constructed and no tracing request is attempted." Unit 1a's
`test_tracing_disabled.py` proved the force-disable mechanism (both env
variables are set to `"false"` at import time); it could not prove the rest,
because nothing imported `langchain-core` yet. `graph.py` is the first thing
that does, so this is where the second half becomes testable.

The obvious version of this test is wrong. Verified against the installed
packages: `import langgraph.graph` pulls in `langsmith` transitively --
about 892 modules, including `langsmith.client`, `httpx`, and `requests` --
through `langgraph -> langchain_core.runnables.base ->
...tracers.schemas -> langsmith`. That import chain comes from
`langchain-core`, not from any LangGraph code of its own, and it happens
whether or not tracing is enabled. So a test asserting
`"langsmith" not in sys.modules` fails, and would be asserting something
`TECHNICAL_SPEC.md` never actually requires. What section 11 asks for is
narrower and is what this file checks: after a full graph run, no
`langsmith.client.Client` was ever instantiated, no HTTP request left the
process through the two client libraries LangSmith uses to send traces, and
both tracing environment variables are still `"false"`.
"""

import os
from pathlib import Path

import httpx
import langsmith.client
import pytest
import requests.sessions
from fake_incident import (
    RecordingLogsBackend,
    StepClock,
    alert_packet,
    incident_scope,
    logs_only_registry,
    packet_evidence,
)

from causalops.graph import run_graph_investigation
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunRecorder

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "causalops"
    / "replay_fixtures"
    / "graph_single_check.json"
)


def run_one_graph_investigation() -> None:
    scope = incident_scope()
    packet = alert_packet()
    substitutions = {
        "incident_id": scope.incident_id,
        "window_start": scope.started_at.isoformat(),
        "window_end": scope.ended_at.isoformat(),
        "symptom_evidence_id": packet.symptom_evidence_id,
    }
    model = ReplayToolCallingModel(
        ReplayReasoningModel(FIXTURE, substitutions=substitutions)
    )
    registry = logs_only_registry(RecordingLogsBackend())
    recorder = RunRecorder(StepClock())
    run_graph_investigation(
        scope, packet, packet_evidence(), model, registry, recorder, clock=StepClock()
    )


def test_no_tracing_client_is_constructed_during_a_graph_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "langsmith.client.Client was constructed during a graph run"
        )

    monkeypatch.setattr(langsmith.client.Client, "__init__", refuse_construction)

    run_one_graph_investigation()


def test_no_network_request_is_attempted_during_a_graph_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_send(*args: object, **kwargs: object) -> None:
        raise AssertionError("an HTTP request was sent during a graph run")

    # Both libraries LangSmith's client can use to actually put bytes on a
    # socket -- patching `Client.__init__` alone would not catch a tracer
    # that reused an existing client instance.
    monkeypatch.setattr(httpx.Client, "send", refuse_send)
    monkeypatch.setattr(requests.sessions.Session, "send", refuse_send)

    run_one_graph_investigation()


def test_tracing_env_vars_stay_disabled_after_a_graph_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regression guard distinct from `test_tracing_disabled.py`: this
    proves nothing inside a live graph run flips either variable back on,
    not just that `causalops/__init__.py` sets them at import time."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    run_one_graph_investigation()

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
