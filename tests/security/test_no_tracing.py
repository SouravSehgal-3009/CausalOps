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

Unit 3b-2, P2-3: the two `..._through_the_cli_entry_point` tests below
prove that *running* a replay investigation through `cli.main` sends
nothing -- a real, separate claim `pyproject.toml`'s `langchain-anthropic`
dependency comment also cites, since that import path (`causalops.cli` now
imports `causalops.live_model` unconditionally) is not exercised by this
file's other tests. What those two tests do *not* prove, despite an earlier
version of this file's own claim to the contrary: that merely *importing*
`causalops.cli` is safe. `from causalops import cli` sits at this file's
module scope, so it completes during pytest *collection* -- before either
test's `monkeypatch.setattr` installs. Injecting a module-level `httpx` GET
into `cli.py` and re-running leaves both tests passing (mutation-verified):
the refusal patches are never in place while the import actually runs.

The import-time property genuinely holds today, but for a different reason
than either test claims: `tests/conftest.py` installs a loopback-only
network guard for the whole pytest session before collection starts, so a
real network attempt during this file's own `from causalops import cli`
would already have surfaced as a collection error, independent of anything
in this file. `test_importing_causalops_cli_never_sends_a_tracing_request`
below is what actually proves the import-time claim on its own terms: a
fresh subprocess with the refusal monkeypatches installed *first*, then the
import, immune to collection-order timing and to whether the guard happens
to be active.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import langsmith.client
import pytest
import requests.sessions
from fake_incident import (
    RecordingLogsBackend,
    StepClock,
    alert_packet,
    change_row,
    incident_scope,
    logs_only_registry,
    packet_evidence,
    write_changes,
)

from causalops import cli
from causalops.domain import StoredIncident
from causalops.graph import run_graph_investigation
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunRecorder
from causalops.telemetry import RunPaths

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


def run_one_cli_replay_investigation(tmp_path: Path) -> None:
    """Unit 3b-2 audit: `causalops.cli` now imports `causalops.live_model`
    at module level unconditionally (`--model replay` or `--model claude`
    alike), which pulls in `langchain_anthropic` -- and, transitively,
    `anthropic`, `httpx`, and `requests` all over again through a second
    import path this file's other tests never exercise, since they call
    `causalops.graph`/`causalops.models` directly and never import
    `causalops.cli` at all. This drives a real replay investigation through
    `cli.main`, the actual console-script entry point, so the refusal
    monkeypatches below cover the import graph a real `causalops` invocation
    loads, not just the one `run_one_graph_investigation` happens to need.
    """
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    scope = incident_scope()
    packet = alert_packet()
    incident = StoredIncident(scope=scope, packet=packet, evidence=packet_evidence())
    paths = RunPaths(root=tmp_path / "runs" / scope.incident_id)
    paths.root.mkdir(parents=True)
    paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
    paths.logs.mkdir(parents=True)
    (paths.logs / "orders.jsonl").write_text(
        json.dumps(
            {
                "at": scope.started_at.isoformat(),
                "request_id": "r1",
                "service": "orders",
                "severity": "error",
                "event": "config_rejected_request",
                "fields": {"config_key": "require_order_token", "detail": "x"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_changes(paths, [change_row(offset=60)])

    exit_status = cli.main(["investigate", scope.incident_id, "--model", "replay"])

    assert exit_status == 0


def test_no_tracing_client_is_constructed_through_the_cli_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError("langsmith.client.Client was constructed through cli.main")

    monkeypatch.setattr(langsmith.client.Client, "__init__", refuse_construction)
    monkeypatch.chdir(tmp_path)

    run_one_cli_replay_investigation(tmp_path)


def test_no_network_request_is_attempted_through_the_cli_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_send(*args: object, **kwargs: object) -> None:
        raise AssertionError("an HTTP request was sent through cli.main")

    monkeypatch.setattr(httpx.Client, "send", refuse_send)
    monkeypatch.setattr(requests.sessions.Session, "send", refuse_send)
    monkeypatch.chdir(tmp_path)

    run_one_cli_replay_investigation(tmp_path)


def test_importing_causalops_cli_never_sends_a_tracing_request() -> None:
    """Unit 3b-2, P2-3's actual import-time proof, distinct from the two
    tests above. Installs the same three refusal monkeypatches those tests
    use, but in a *fresh subprocess*, before `causalops.cli` is ever
    imported -- `import causalops.cli` runs strictly after the patches are
    live, unlike this file's own module-scope `from causalops import cli`,
    which the module docstring above explains cannot make the same claim.
    `test_importing_the_investigator_never_loads_the_evaluator`
    (`tests/security/test_ground_truth_isolation.py`) already spawns a
    child interpreter for the same reason: a property about *what happens
    during an import* cannot be proven by patching after the import has
    already run."""
    script = (
        "import langsmith.client, httpx, requests.sessions\n"
        "def refuse(*args, **kwargs):\n"
        "    raise AssertionError('network activity during import')\n"
        "langsmith.client.Client.__init__ = refuse\n"
        "httpx.Client.send = refuse\n"
        "requests.sessions.Session.send = refuse\n"
        "import causalops.cli\n"
        "print('import-ok')\n"
    )

    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "import-ok"
