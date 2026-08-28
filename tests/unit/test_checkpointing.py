"""The SQLite checkpointer swap, the `investigation_id` inversion,
events durable in `GraphState`, and the msgpack hardening on the real
checkpoint database.

`test_graph_frozen_reports.py` is untouched by this file and stays that way:
every test below either drives `run_graph_investigation` with the new
`checkpointer`/`investigation_id` parameters directly and checks what those
two `None`-defaulted arguments exist to prove, or exercises `cli.py`'s own
wiring end to end. None of it asserts a value frozen anywhere else in this
project. `test_graph.py`'s own
`test_events_stay_continuous_across_a_second_dispatch_and_normalize_pass`
covers the events-in-state design's own correctness (the exhaustive
"every node return path carries its events forward" property); this file
covers the checkpointer and identity plumbing built on top of it.

`_model()` must substitute `graph_single_check.json`'s `{{...}}` tokens the
same way `test_graph.py`'s own `graph_replay_model()` does -- a first
implementation of this file skipped that, so every test below silently ran a
model response that failed to parse, crashed to `FAILED_SAFE`/`INTERNAL_ERROR`
before any tool ever dispatched, and "the checkpointer swap doesn't change
the investigation" was being proven by comparing two identical crashes. Every
test that depends on a check actually executing now asserts
`disposition is not Disposition.FAILED_SAFE` as a direct guard against that
regression recurring silently.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fake_incident import (
    FIXTURE_DIR,
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    RecordingLogsBackend,
    StepClock,
    alert_packet,
    assessment_json,
    change_row,
    incident_scope,
    logs_only_registry,
    packet_evidence,
    plan_json,
    replay_model,
    resume_graph_run,
    write_changes,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from causalops import cli
from causalops.domain import (
    Budgets,
    Disposition,
    EscalatedInvestigation,
    EscalationReason,
    ReceiptState,
    StoredIncident,
)
from causalops.graph import build_graph, run_graph_investigation
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunRecorder
from causalops.telemetry import RunPaths
from causalops.tool_wrappers import ToolWrapper
from causalops.tools import ToolName

GRAPH_FIXTURE = FIXTURE_DIR / "graph_single_check.json"


def _model() -> ReplayToolCallingModel:
    """`graph_single_check.json` scripts `{{...}}` placeholders the same way
    `lab_diagnosis.json` does, matching `test_graph.py::graph_replay_model`'s
    own substitutions exactly -- see this module's docstring for why a
    missing substitution here is not a cosmetic gap."""
    substitutions = {
        "incident_id": incident_scope().incident_id,
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "symptom_evidence_id": SYMPTOM_EVIDENCE_ID,
    }
    return ReplayToolCallingModel(
        ReplayReasoningModel(GRAPH_FIXTURE, substitutions=substitutions)
    )


def _registry() -> dict[ToolName, ToolWrapper]:
    return logs_only_registry(RecordingLogsBackend())


def test_a_file_backed_checkpointer_reproduces_the_in_memory_run(
    tmp_path: Path,
) -> None:
    """Swapping `InMemorySaver()` for a real `SqliteSaver` file is supposed
    to change durability, never the investigation itself. Exact evidence/
    receipt IDs are random per call (`uuid4()`, not the deterministic
    counting generator `test_graph_frozen_reports.py` installs), so this
    compares everything that must still agree: the disposition, the shape of
    the run, the full event vocabulary in order, and the evidence kinds --
    not the specific opaque IDs each independent run mints."""
    default_recorder = RunRecorder(StepClock())
    default_result = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        _model(),
        _registry(),
        default_recorder,
        Budgets(),
        StepClock(),
    )
    assert default_result.report.disposition is not Disposition.FAILED_SAFE

    db_path = tmp_path / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        sqlite_recorder = RunRecorder(StepClock())
        sqlite_result = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            _model(),
            _registry(),
            sqlite_recorder,
            Budgets(),
            StepClock(),
            checkpointer=SqliteSaver(conn),
        )

    assert sqlite_result.report.disposition is not Disposition.FAILED_SAFE
    assert sqlite_result.report.disposition == default_result.report.disposition
    assert sqlite_result.report.root_cause == default_result.report.root_cause
    assert sqlite_result.report.tools_executed == default_result.report.tools_executed
    assert (
        sqlite_result.report.model_calls_used == default_result.report.model_calls_used
    )
    assert len(sqlite_result.report.evidence_ids) == len(
        default_result.report.evidence_ids
    )
    assert len(sqlite_result.report.receipt_ids) == len(
        default_result.report.receipt_ids
    )
    assert [record.kind for record in sqlite_result.evidence] == [
        record.kind for record in default_result.evidence
    ]
    assert [event.name for event in sqlite_recorder.events] == [
        event.name for event in default_recorder.events
    ]
    assert db_path.is_file()


def test_a_second_connection_reads_back_the_finished_run(tmp_path: Path) -> None:
    """Precursor to a real two-process resume: a completed run's
    final checkpoint must be readable from an independent `SqliteSaver`
    instance opened fresh against the same file, not merely from the
    in-process object that wrote it."""
    db_path = tmp_path / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        result = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            _model(),
            _registry(),
            RunRecorder(StepClock()),
            Budgets(),
            StepClock(),
            investigation_id="durability-check",
            checkpointer=SqliteSaver(conn),
        )
    assert result.report.disposition is not Disposition.FAILED_SAFE

    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        reopened = SqliteSaver(conn)
        config: RunnableConfig = {"configurable": {"thread_id": "durability-check"}}
        checkpoint = reopened.get_tuple(config)

    assert checkpoint is not None
    assert checkpoint.checkpoint["channel_values"]["report"] is not None


def test_a_raising_backend_leaves_a_durable_reserved_receipt(tmp_path: Path) -> None:
    """The durable half of `test_graph.py`'s
    `test_a_raising_backend_leaves_a_visible_reserved_receipt_in_the_graph_report`,
    which only ever drives `InMemorySaver()` and so would pass identically
    whether or not a `RESERVED` receipt actually survives a real
    `SqliteSaver` file -- `tool_wrappers.py:24-28`'s own promise that a
    crash between reserving and settling stays visible "after the fact
    too, not just to a caller still holding the ledger."

    The window between the pure `authorize()` decision and
    `ReservationLedger.reserve()` recording it is not separately testable:
    both happen inside one synchronous call, inside one node invocation
    (`tool_wrappers.py:130-153`), with no I/O in between -- the only
    durable boundary is the node's return, which this test is already on
    the far side of. There is no narrower crash window to aim at.
    """
    db_path = tmp_path / "checkpoints.db"
    backend = RecordingLogsBackend(raises=RuntimeError("lab unreachable"))
    registry = logs_only_registry(backend)

    with cli._sqlite_checkpointer(db_path) as checkpointer:
        result = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            _model(),
            registry,
            RunRecorder(StepClock()),
            Budgets(),
            StepClock(),
            investigation_id="durable-reserved-receipt",
            checkpointer=checkpointer,
        )
    assert result.report.disposition is Disposition.FAILED_SAFE
    (only_receipt,) = result.receipts
    assert only_receipt.state is ReceiptState.RESERVED

    # A fresh connection, opened only against the file path -- the run's own
    # `checkpointer` is out of scope by now, closed at the end of the `with`
    # block above. This is the actual claim: not that the in-process ledger
    # held the receipt (already proven above and in `test_graph.py`), but
    # that it is still there after the process that wrote it is gone.
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        reopened = SqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None)
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": "durable-reserved-receipt"}
        }
        checkpoint = reopened.get_tuple(config)

    assert checkpoint is not None
    receipts = checkpoint.checkpoint["channel_values"]["receipts"]
    assert len(receipts) == 1
    assert receipts[0]["state"] == "RESERVED"
    assert receipts[0]["outcome"] is None


def test_a_two_process_pause_and_resume_settles_over_a_real_sqlite_file(
    tmp_path: Path,
) -> None:
    """The named proof, not the in-process convenience every other
    escalation-interrupt test in this project uses: every one of them drives
    `InMemorySaver()`, which would pass identically whether or not the
    real target -- resuming through the hardened `SqliteSaver` this
    project actually runs, `cli._sqlite_checkpointer` -- works at all. This
    test closes the connection between pause and resume and reopens the
    checkpoint file fresh, standing in for the second process
    `causalops approve`/`reject` runs in: it never touches the
    `checkpointer` the pause used, only the database path and the thread id
    a real second process would actually have."""
    db_path = tmp_path / "cp.db"
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    registry = logs_only_registry(RecordingLogsBackend())
    clock = StepClock()

    with cli._sqlite_checkpointer(db_path) as checkpointer:
        paused = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            model,
            registry,
            RunRecorder(StepClock()),
            Budgets(),
            clock,
            investigation_id="two-process-probe",
            checkpointer=checkpointer,
        )
    assert isinstance(paused, EscalatedInvestigation)
    assert paused.reason is EscalationReason.CONFLICTING_EVIDENCE
    calls_before_resume = len(model.requests)

    # A fresh connection and a fresh `SqliteSaver`, opened only against the
    # file path -- the pause's own `checkpointer` object is out of scope by
    # now, closed at the end of the `with` block above.
    with cli._sqlite_checkpointer(db_path) as reopened:
        compiled = build_graph(
            incident_scope(),
            alert_packet(),
            Budgets(),
            clock,
            model,
            registry,
            reopened,
            event_clock=StepClock(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": "two-process-probe"}}
        snapshot = compiled.get_state(config)
        assert snapshot.interrupts
        assert snapshot.interrupts[0].value["reason"] == "CONFLICTING_EVIDENCE"

        settled = resume_graph_run(compiled, config, "accept")

    assert settled.report.disposition is Disposition.DIAGNOSED
    assert settled.report.escalation is not None
    assert settled.report.escalation.decision == "accept"
    # Stronger than the in-process purity proxy elsewhere: zero additional
    # model calls after reopening the checkpoint from a fresh connection,
    # not just within one process's live object graph.
    assert len(model.requests) == calls_before_resume


def test_an_explicit_investigation_id_becomes_the_report_id_and_the_thread_id(
    tmp_path: Path,
) -> None:
    """`investigation_id` inverts from an output every caller used to
    receive to an optional input a resume path can supply -- it
    must land in both the finished report and LangGraph's own `thread_id`,
    which is where a resumed run would look it up."""
    db_path = tmp_path / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        checkpointer = SqliteSaver(conn)
        result = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            _model(),
            _registry(),
            RunRecorder(StepClock()),
            Budgets(),
            StepClock(),
            investigation_id="explicit-thread-id",
            checkpointer=checkpointer,
        )
        config: RunnableConfig = {"configurable": {"thread_id": "explicit-thread-id"}}
        checkpoint = checkpointer.get_tuple(config)

    assert result.report.disposition is not Disposition.FAILED_SAFE
    assert result.report.investigation_id == "explicit-thread-id"
    assert checkpoint is not None
    assert checkpoint.checkpoint["channel_values"]["investigation_id"] == (
        "explicit-thread-id"
    )


def test_run_id_is_present_and_distinct_from_investigation_id(tmp_path: Path) -> None:
    """`run_id` is a distinct, required `GraphState` field and is not
    exposed on `InvestigationReport` -- the only way to observe it is through
    the checkpointed `GraphState` a real checkpointer records. This does not
    (and, before a resume path exists with something to diverge
    across, cannot) prove `run_id` behaves differently from
    `investigation_id`; it proves the field exists, is populated, and is not
    silently aliased to the value `investigation_id` was given."""
    db_path = tmp_path / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        checkpointer = SqliteSaver(conn)
        result = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            _model(),
            _registry(),
            RunRecorder(StepClock()),
            Budgets(),
            StepClock(),
            investigation_id="run-id-check",
            checkpointer=checkpointer,
        )
        config: RunnableConfig = {"configurable": {"thread_id": "run-id-check"}}
        checkpoint = checkpointer.get_tuple(config)

    assert result.report.disposition is not Disposition.FAILED_SAFE
    assert checkpoint is not None
    run_id = checkpoint.checkpoint["channel_values"]["run_id"]
    assert isinstance(run_id, str)
    assert run_id
    assert run_id != "run-id-check"


def test_omitting_investigation_id_still_mints_a_fresh_one_each_call() -> None:
    """The default (`None`) branch is what every existing caller relies on --
    a regression here would silently collide every investigation onto one
    `thread_id`."""
    first = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        _model(),
        _registry(),
        RunRecorder(StepClock()),
        Budgets(),
        StepClock(),
    )
    second = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        _model(),
        _registry(),
        RunRecorder(StepClock()),
        Budgets(),
        StepClock(),
    )

    assert first.report.disposition is not Disposition.FAILED_SAFE
    assert second.report.disposition is not Disposition.FAILED_SAFE
    assert first.report.investigation_id != second.report.investigation_id


def test_the_hardened_serializer_refuses_an_unregistered_type() -> None:
    """`cli.py`'s `_sqlite_checkpointer` passes `allowed_msgpack_modules=None`
    to `JsonPlusSerializer`, the same restriction `LANGGRAPH_STRICT_MSGPACK=true`
    applies. Without it, a permissive serializer imports and instantiates
    whatever class name a checkpoint blob claims -- an attacker who could
    write to `checkpoints.db` could use that for code execution on the next
    read. `causalops.domain.Budgets` stands in for "some real application
    type nobody added to LangGraph's own safe-type allowlist": a permissive
    serializer reconstructs it; the hardened one returns the raw field dict
    instead, refusing to run its constructor at all.

    This proves the library flag itself is non-vacuous, using the exact
    construction `_sqlite_checkpointer` uses. It does not, on its own, prove
    `_sqlite_checkpointer` actually wires that construction into the object
    it hands the caller --
    `test_sqlite_checkpointer_yields_a_saver_with_the_hardened_serializer`
    below closes that gap by inspecting `_sqlite_checkpointer`'s own output
    directly."""
    permissive = JsonPlusSerializer()
    hardened = JsonPlusSerializer(allowed_msgpack_modules=None)
    blob = permissive.dumps_typed(Budgets())

    assert isinstance(permissive.loads_typed(blob), Budgets)
    restored = hardened.loads_typed(blob)
    assert not isinstance(restored, Budgets)
    assert isinstance(restored, dict)


def test_sqlite_checkpointer_yields_a_saver_with_the_hardened_serializer(
    tmp_path: Path,
) -> None:
    """Exercises `cli._sqlite_checkpointer` itself, not a `JsonPlusSerializer`
    built separately in the test -- a real gap once found: swapping
    `_sqlite_checkpointer`'s `allowed_msgpack_modules=None` for the
    permissive default made every test in this file pass, because nothing
    reached into the object it actually yields. `checkpointer.serde` is the
    exact instance `SqliteSaver.__init__` stored (`BaseCheckpointSaver`
    keeps whatever `serde` it is given), so asserting on it here is
    assertion on the wired object, not a stand-in for it."""
    with cli._sqlite_checkpointer(tmp_path / "checkpoints.db") as checkpointer:
        blob = JsonPlusSerializer().dumps_typed(Budgets())
        restored = checkpointer.serde.loads_typed(blob)

    assert not isinstance(restored, Budgets)
    assert isinstance(restored, dict)


def test_investigate_leaves_a_checkpoint_database_in_a_fresh_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-to-end acceptance check: a real `causalops
    investigate` run, in a project that has never had a `results/` directory
    before, must both succeed and leave `results/checkpoints.db` behind --
    proving `cli.py`'s own `mkdir(parents=True, exist_ok=True)` handles the
    first-ever run in a clean checkout, not just a pre-existing tree.
    Fabricates the incident directly, the same way
    `test_cli.py::test_investigate_runs_an_investigation_end_to_end` does,
    since the real scenario controller's docker-marked path is deselected
    from this fast suite."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "results").exists()

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
    db_path = tmp_path / "results" / "checkpoints.db"
    assert db_path.is_file()
    assert db_path.stat().st_size > 0

    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        saver = SqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None)
        )
        checkpoints = list(saver.list(None))
    assert len(checkpoints) >= 1
