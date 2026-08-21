"""Unit 2c: `OwnerDecision`'s validation, the append-only `owner_decisions`
store, and `causalops approve`/`reject` end to end through `cli.py`.

The CLI-level tests below follow `test_checkpointing.py`'s own idiom: a
pause is driven directly through `cli._sqlite_checkpointer` and a *test*
model script (never through `causalops investigate`, which is hard-wired to
`lab_diagnosis.json` -- a fixture that never escalates), then the command
under test is called against the bare database *path*, in a fresh
connection, standing in for the second process a real `causalops approve`/
`reject` would run in. `run_decision_command` rebuilds its own model and
tool registry from `runs/<incident_id>/incident.json` -- production
wiring, bound to `lab_diagnosis.json` and the real tool backends -- which
is intentionally *not* the model/registry that produced the pause: a plain
accept/reject resume never calls the model or a tool again
(`escalation_interrupt` routes straight to `final_report`), so the mismatch
is inert and exercises the real production code path rather than a stub.
"""

import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest
from fake_incident import (
    SYMPTOM_EVIDENCE_ID,
    RecordingLogsBackend,
    StepClock,
    alert_packet,
    assessment_json,
    incident_scope,
    logs_only_registry,
    packet_evidence,
    plan_json,
    replay_model,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, StateSnapshot
from pydantic import ValidationError

from causalops import cli
from causalops.approvals import (
    CheckpointStoreError,
    DecisionRow,
    OwnerDecision,
    ensure_decisions_table,
    read_decision_for_thread,
    record_decision_before_resume,
)
from causalops.domain import (
    Budgets,
    Disposition,
    EscalatedInvestigation,
    InvestigationReport,
    StoredIncident,
    utc_now,
)
from causalops.graph import build_graph, run_graph_investigation
from causalops.models import ReplayToolCallingModel
from causalops.run_records import RunRecorder

# ---------------------------------------------------------------------------
# OwnerDecision -- validation and normalization at the CLI boundary
# ---------------------------------------------------------------------------


def test_accept_forbids_a_rejection_note() -> None:
    with pytest.raises(ValidationError):
        OwnerDecision(decision="accept", rejection_note="not allowed")


def test_reject_requires_a_rejection_note() -> None:
    with pytest.raises(ValidationError):
        OwnerDecision(decision="reject")


def test_a_blank_rejection_note_is_treated_as_missing() -> None:
    """`reject "   "` must fail exactly like omitting the reason does --
    whitespace is not content."""
    with pytest.raises(ValidationError):
        OwnerDecision(decision="reject", rejection_note="   ")


def test_a_rejection_note_is_stripped_not_left_with_surrounding_whitespace() -> None:
    decision = OwnerDecision(decision="reject", rejection_note="  a real reason  ")

    assert decision.rejection_note == "a real reason"


def test_an_overlong_rejection_note_is_refused_not_truncated() -> None:
    """Silent truncation would lose whatever the owner wrote past the
    bound with nothing left to catch it -- overflow must be a loud
    `ValidationError`, and the accepted boundary (300 chars, matching this
    project's other bounded free-text fields) must still be `None`
    -truncated to be considered defensible."""
    at_the_limit = OwnerDecision(decision="reject", rejection_note="x" * 300)
    assert at_the_limit.rejection_note == "x" * 300

    with pytest.raises(ValidationError):
        OwnerDecision(decision="reject", rejection_note="x" * 301)


def test_resume_value_carries_decision_and_note_as_a_plain_mapping() -> None:
    decision = OwnerDecision(decision="reject", rejection_note="a real reason")

    assert decision.resume_value() == {
        "decision": "reject",
        "rejection_note": "a real reason",
    }


# ---------------------------------------------------------------------------
# The owner_decisions store
# ---------------------------------------------------------------------------


def test_ensure_decisions_table_is_idempotent(tmp_path: Path) -> None:
    with closing(sqlite3.connect(str(tmp_path / "cp.db"))) as conn:
        ensure_decisions_table(conn)
        ensure_decisions_table(conn)  # must not raise the second time


def test_a_recorded_decision_reads_back_identically(tmp_path: Path) -> None:
    with closing(sqlite3.connect(str(tmp_path / "cp.db"))) as conn:
        ensure_decisions_table(conn)
        decision = OwnerDecision(decision="reject", rejection_note="a real reason")
        record_decision_before_resume(conn, "thread-1", "cp-1", decision, utc_now())

        row = read_decision_for_thread(conn, "thread-1")

    assert row is not None
    assert row.matches(decision)
    assert row.checkpoint_id == "cp-1"


def test_an_unknown_thread_reads_back_no_decision(tmp_path: Path) -> None:
    with closing(sqlite3.connect(str(tmp_path / "cp.db"))) as conn:
        ensure_decisions_table(conn)
        assert read_decision_for_thread(conn, "nobody-paused-this") is None


def test_a_conflicting_composite_key_write_is_refused_by_the_database(
    tmp_path: Path,
) -> None:
    """`record_decision_before_resume`'s own `IntegrityError` handling,
    exercised directly against the composite `(thread_id, checkpoint_id)`
    primary key -- the guard `causalops.cli.run_decision_command` relies on
    when two callers race to record the first decision for the same paused
    checkpoint."""
    with closing(sqlite3.connect(str(tmp_path / "cp.db"))) as conn:
        ensure_decisions_table(conn)
        record_decision_before_resume(
            conn, "thread-1", "cp-1", OwnerDecision(decision="accept"), utc_now()
        )

        with pytest.raises(CheckpointStoreError) as excinfo:
            record_decision_before_resume(
                conn,
                "thread-1",
                "cp-1",
                OwnerDecision(decision="reject", rejection_note="changed my mind"),
                utc_now(),
            )

    assert excinfo.value.reason_code.value == "CONFLICTING_DECISION"


def test_a_mispaired_row_is_refused_as_a_store_problem_not_a_traceback(
    tmp_path: Path,
) -> None:
    """A row this module wrote can never be mis-paired -- `record_decision_
    before_resume` only ever inserts an already-validated `OwnerDecision`.
    A hand-corrupted row (raw SQL, bypassing this module entirely, standing
    in for disk corruption or manual tampering) is a different story:
    `sqlite3` enforces the composite primary key but nothing about
    "non-null iff decision='reject'". `read_decision_for_thread` must
    refuse this as `STORE_UNAVAILABLE`, the same contract a `sqlite3.Error`
    already gets, not let a `pydantic.ValidationError` escape as an
    unhandled traceback."""
    with closing(sqlite3.connect(str(tmp_path / "cp.db"))) as conn:
        ensure_decisions_table(conn)
        conn.execute(
            "INSERT INTO owner_decisions "
            "(thread_id, checkpoint_id, decision, rejection_note, decided_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("thread-1", "cp-1", "accept", "should not be here", utc_now().isoformat()),
        )
        conn.commit()

        with pytest.raises(CheckpointStoreError) as excinfo:
            read_decision_for_thread(conn, "thread-1")

    assert excinfo.value.reason_code.value == "STORE_UNAVAILABLE"


def test_an_unparseable_decided_at_is_refused_as_a_store_problem(
    tmp_path: Path,
) -> None:
    """Same contract, the other unparseable field: `decided_at` is stored
    as plain text (`ensure_decisions_table` has no datetime column type),
    so a corrupted value is only caught on the way back out, by
    `DecisionRow`'s own `UtcDatetime` field."""
    with closing(sqlite3.connect(str(tmp_path / "cp.db"))) as conn:
        ensure_decisions_table(conn)
        conn.execute(
            "INSERT INTO owner_decisions "
            "(thread_id, checkpoint_id, decision, rejection_note, decided_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("thread-1", "cp-1", "accept", None, "not-a-timestamp"),
        )
        conn.commit()

        with pytest.raises(CheckpointStoreError) as excinfo:
            read_decision_for_thread(conn, "thread-1")

    assert excinfo.value.reason_code.value == "STORE_UNAVAILABLE"


def test_approve_refuses_cleanly_when_the_decision_row_is_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-to-end proof through `cli.main`: a corrupted `owner_decisions`
    row must surface as `FAIL STORE_UNAVAILABLE ...`, never an unhandled
    traceback, since `CheckpointStoreError` is in `main`'s `except` tuple."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "corrupted-row-thread")

    db_path = tmp_path / "results" / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        ensure_decisions_table(conn)
        conn.execute(
            "INSERT INTO owner_decisions "
            "(thread_id, checkpoint_id, decision, rejection_note, decided_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "corrupted-row-thread",
                "some-checkpoint",
                "APPROVED",
                None,
                utc_now().isoformat(),
            ),
        )
        conn.commit()

    exit_status = cli.main(["approve", "corrupted-row-thread"])

    assert exit_status == 1
    assert "FAIL STORE_UNAVAILABLE" in capsys.readouterr().out


def test_approve_refuses_cleanly_when_the_finalized_report_is_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The identical-retry short-circuit reads `report.json` back off disk
    without ever touching the graph -- a corrupted artifact there must also
    refuse cleanly rather than raising an unhandled `ValidationError`."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "corrupted-report-thread")
    cli.main(["approve", "corrupted-report-thread"])

    report_path = (
        tmp_path
        / "results"
        / "investigations"
        / "corrupted-report-thread"
        / "report.json"
    )
    report_path.write_text("{}", encoding="utf-8")

    exit_status = cli.main(["approve", "corrupted-report-thread"])

    assert exit_status == 1
    assert "FAIL STORE_UNAVAILABLE" in capsys.readouterr().out


def test_read_decision_for_thread_returns_the_most_recent_row(tmp_path: Path) -> None:
    """Not reachable through today's graph (there is no path back to a
    second pause on the same thread yet), but the SQL itself must already
    prefer the latest row -- `ORDER BY decided_at DESC` is what makes a
    thread-scoped lookup meaningful once a second pause becomes possible."""
    with closing(sqlite3.connect(str(tmp_path / "cp.db"))) as conn:
        ensure_decisions_table(conn)
        # An explicit gap, not two back-to-back `utc_now()` calls -- two
        # calls close enough together could land on the same microsecond
        # and make "most recent" ambiguous by construction.
        earlier = utc_now()
        later = earlier + timedelta(seconds=1)
        record_decision_before_resume(
            conn, "thread-1", "cp-1", OwnerDecision(decision="accept"), earlier
        )
        record_decision_before_resume(
            conn,
            "thread-1",
            "cp-2",
            OwnerDecision(decision="reject", rejection_note="a second look"),
            later,
        )

        row = read_decision_for_thread(conn, "thread-1")

    assert row is not None
    assert row.checkpoint_id == "cp-2"


def test_decision_row_matches_compares_the_note_not_just_the_decision() -> None:
    """The note is the field most likely to legitimately change on a human
    retry -- a decision-only comparison would silently discard a corrected
    reason instead of flagging the retry as conflicting."""
    row = DecisionRow(
        thread_id="t",
        checkpoint_id="c",
        decision="reject",
        rejection_note="first reason",
        decided_at=utc_now(),
    )

    assert not row.matches(
        OwnerDecision(decision="reject", rejection_note="a different reason")
    )
    assert row.matches(OwnerDecision(decision="reject", rejection_note="first reason"))


# ---------------------------------------------------------------------------
# `causalops approve`/`reject` end to end through `cli.py`
# ---------------------------------------------------------------------------


def _write_incident(root: Path) -> None:
    scope = incident_scope()
    incident = StoredIncident(
        scope=scope, packet=alert_packet(), evidence=packet_evidence()
    )
    paths = cli.RunPaths(root=root / "runs" / scope.incident_id)
    paths.root.mkdir(parents=True)
    paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")


def _pause_a_thread(root: Path, thread_id: str) -> EscalatedInvestigation:
    """Pauses `thread_id` directly against `root/results/checkpoints.db`,
    standing in for the process that ran `causalops investigate` and
    escalated. Uses a test-scripted model (never `lab_diagnosis.json`) so
    the pause is deterministic and does not depend on the production
    fixture ever escalating, which it does not."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    model = ReplayToolCallingModel(replay_model(root, script))
    registry = logs_only_registry(RecordingLogsBackend())
    db_path = root / "results" / "checkpoints.db"
    with cli._sqlite_checkpointer(db_path) as checkpointer:
        paused = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            model,
            registry,
            RunRecorder(StepClock()),
            checkpointer=checkpointer,
            investigation_id=thread_id,
        )
    assert isinstance(paused, EscalatedInvestigation)
    return paused


def _report_json(root: Path, thread_id: str) -> InvestigationReport:
    path = root / "results" / "investigations" / thread_id / "report.json"
    return InvestigationReport.model_validate_json(path.read_text(encoding="utf-8"))


def _checkpoint_snapshot(root: Path, thread_id: str) -> StateSnapshot:
    """The same read `run_decision_command` itself performs before deciding
    whether a thread is still pending -- `.next`/`.interrupts`, never a bare
    `SqliteSaver.get_tuple`, which exposes neither. Used by the crash-
    recovery tests below to confirm each retry actually starts from the
    window its name claims: paused (interrupted, no report) for one test,
    settled (no interrupt, report present) for the other -- not merely
    "whichever function got monkeypatched," which a future `DISPATCH_TOOL`
    route could make ambiguous (a re-paused settle looks identical to a
    still-pending one by function-patched alone)."""
    db_path = root / "results" / "checkpoints.db"
    with cli._sqlite_checkpointer(db_path) as checkpointer:
        compiled = build_graph(
            incident_scope(),
            alert_packet(),
            Budgets(),
            utc_now,
            ReplayToolCallingModel(replay_model(root, {})),
            logs_only_registry(RecordingLogsBackend()),
            checkpointer,
            event_clock=utc_now,
        )
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        return compiled.get_state(config)


def _feed_one_bad_resume(root: Path, thread_id: str) -> None:
    """Puts a paused thread into the *re-paused* state from Unit 2b's own
    measured table -- `.next == ()`, `.interrupts` still non-empty --
    standing in for a stray malformed `Command(resume=...)` from outside
    this CLI (a test, or a future caller). Guard #1 must still recognize
    this thread as pending through `.interrupts`; using `.next` instead
    would misreport it as settled and refuse a legitimate approve."""
    incident = incident_scope()
    db_path = root / "results" / "checkpoints.db"
    with cli._sqlite_checkpointer(db_path) as checkpointer:
        compiled = build_graph(
            incident,
            alert_packet(),
            Budgets(),
            utc_now,
            ReplayToolCallingModel(replay_model(root, {})),
            logs_only_registry(RecordingLogsBackend()),
            checkpointer,
            event_clock=utc_now,
        )
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        result = compiled.invoke(Command(resume="not a valid decision"), config)
    assert "__interrupt__" in result


def test_a_re_paused_thread_is_still_approvable_through_dot_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard for Unit 2b's own measured trap: a re-paused run
    (after a bad resume value) shows `.next == ()`, exactly like a settled
    run does -- only `.interrupts` tells them apart. Mutating guard #1 to
    read `.next` instead would make this test fail by wrongly refusing a
    real pending approval."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "re-paused-thread")
    _feed_one_bad_resume(tmp_path, "re-paused-thread")

    exit_status = cli.main(["approve", "re-paused-thread"])

    assert exit_status == 0
    report = _report_json(tmp_path, "re-paused-thread")
    assert report.escalation is not None
    assert report.escalation.decision == "accept"


def test_approve_reports_a_locked_checkpoint_store_instead_of_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unit 2d: `run_decision_command`'s own `mkdir`/`sqlite3.connect` pair
    (`cli.py`, the connection `owner_decisions` lives behind) was the other
    untranslated connection open -- reached before any thread lookup, so no
    incident or paused thread is needed to exercise it, only a database
    path this process cannot open as a file.

    A read-only `results/` directory (via `os.chmod`) proved the same
    connect-time wrap in an earlier version of this test, but only on
    POSIX -- Windows ignores `os.chmod` for directories entirely, which is
    why that version passed CI on Ubuntu and went uncaught on Windows.
    Occupying `checkpoints.db`'s own path with a directory instead reaches
    the identical `sqlite3.connect()` call on every platform: SQLite's own
    file-open logic refuses a directory the same way everywhere, unlike an
    OS permission bit."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "checkpoints.db").mkdir()
    assert (results_dir / "checkpoints.db").is_dir(), (
        "setup did not leave a directory at checkpoints.db's path -- "
        "this test cannot prove anything here"
    )

    exit_status = cli.main(["approve", "any-thread-id"])

    assert exit_status == 1
    assert "FAIL STORE_UNAVAILABLE" in capsys.readouterr().out


def test_approve_on_an_unknown_thread_fails_thread_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asserts the specific `FAIL THREAD_NOT_FOUND` line, not just a nonzero
    exit code -- `exit_status == 1` alone cannot tell this refusal apart
    from `test_approve_on_a_never_paused_thread_fails_no_pending_interrupt`'s
    `NO_PENDING_INTERRUPT`; swapping the two reason codes in `cli.py` would
    pass both tests unless each one names its own code."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_status = cli.main(["approve", "no-such-thread"])

    assert exit_status == 1
    assert "FAIL THREAD_NOT_FOUND" in capsys.readouterr().out


def test_approve_on_a_never_paused_thread_fails_no_pending_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A checkpoint exists (the investigation ran and settled normally),
    but it never escalated -- `.interrupts` is empty, and Unit 2b's own
    measured trap says `.next` cannot be trusted to tell this apart from a
    re-paused run. Guard #1 must refuse this, not silently resume nothing
    and report success."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)

    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json()],  # DIAGNOSED, no contrary evidence
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    registry = logs_only_registry(RecordingLogsBackend())
    db_path = tmp_path / "results" / "checkpoints.db"
    with cli._sqlite_checkpointer(db_path) as checkpointer:
        settled = run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            model,
            registry,
            RunRecorder(StepClock()),
            checkpointer=checkpointer,
            investigation_id="settled-thread",
        )
    assert not isinstance(settled, EscalatedInvestigation)

    exit_status = cli.main(["approve", "settled-thread"])

    assert exit_status == 1
    assert "FAIL NO_PENDING_INTERRUPT" in capsys.readouterr().out


def test_the_decision_is_recorded_before_the_graph_is_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`TECHNICAL_SPEC.md:170-172`'s record-before-resume rule, pinned
    behaviourally rather than by call order: `resume_graph_investigation`
    is monkeypatched to crash before it can do anything, standing in for a
    real crash mid-resume. If the decision were recorded *after* resuming
    -- exactly the regression this test exists to catch -- the row would
    never be written, and this test would find nothing.

    A crash here is not a benign test artifact: 2c's own design note is
    that the *retry* path (`existing is not None`, artifacts not yet on
    disk) is what turns a mid-resume crash into a recoverable second
    attempt rather than a thread stuck behind a record claiming it settled
    with no way to reach `NO_PENDING_INTERRUPT` recovery, since
    `.interrupts` would by then be empty were no row on record at all."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "crash-before-settle-thread")

    def _crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash mid-resume")

    monkeypatch.setattr(cli, "resume_graph_investigation", _crash)

    with pytest.raises(RuntimeError, match="simulated crash mid-resume"):
        cli.main(["approve", "crash-before-settle-thread"])

    db_path = tmp_path / "results" / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        row = read_decision_for_thread(conn, "crash-before-settle-thread")
    assert row is not None
    assert row.decision == "accept"
    assert not (
        tmp_path / "results" / "investigations" / "crash-before-settle-thread"
    ).exists()


def test_a_retry_after_a_crash_before_resume_still_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit 2d's own crash/idempotency suite. The branch the test above
    stops short of: after the identical crash (decision row written,
    `Command(resume=...)` never called at all -- this thread is still
    genuinely paused, not merely reported as such), a *retry* must find the
    existing row, skip the redundant write, and actually resume this time.
    Before this unit this was 2c's own design note ("the retry path...is
    what turns a mid-resume crash into a recoverable second attempt"),
    reasoned about but never exercised."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "crash-before-resume-thread")

    real_resume = cli.resume_graph_investigation

    def _crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash mid-resume")

    monkeypatch.setattr(cli, "resume_graph_investigation", _crash)
    with pytest.raises(RuntimeError, match="simulated crash mid-resume"):
        cli.main(["approve", "crash-before-resume-thread"])
    # `monkeypatch.undo()` would also revert `monkeypatch.chdir` above --
    # restore only the one attribute this test actually broke.
    monkeypatch.setattr(cli, "resume_graph_investigation", real_resume)

    db_path = tmp_path / "results" / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        row_before_retry = read_decision_for_thread(conn, "crash-before-resume-thread")
    assert row_before_retry is not None
    assert not (
        tmp_path / "results" / "investigations" / "crash-before-resume-thread"
    ).exists()
    # The property this test's name claims and the earlier test does not
    # distinguish itself from: the thread is still genuinely paused, not
    # merely reported as such by which function got monkeypatched.
    snapshot_before_retry = _checkpoint_snapshot(tmp_path, "crash-before-resume-thread")
    assert snapshot_before_retry.interrupts
    assert snapshot_before_retry.next == ("escalation_interrupt",)
    assert snapshot_before_retry.values.get("report") is None

    exit_status = cli.main(["approve", "crash-before-resume-thread"])

    assert exit_status == 0
    report = _report_json(tmp_path, "crash-before-resume-thread")
    assert report.escalation is not None
    assert report.escalation.decision == "accept"


def test_a_retry_after_a_crash_before_finalize_still_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other untested transition: a crash *after* the graph genuinely
    resumes and settles (the decision row exists, the checkpoint is
    settled) but *before* `finalize_investigation` ever writes
    `report.json`. `run_decision_command`'s fall-through path (`existing is
    not None` and `investigations_dir` absent) must rebuild the graph, call
    `resume_graph_investigation` again against the now-settled thread, and
    finish the write -- resting on `Command(resume=...)` being a genuine
    no-op against a settled thread, confirmed by direct measurement before
    this suite was written (not merely asserted by a docstring)."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "crash-before-finalize-thread")

    real_write_artifacts = cli._write_investigation_artifacts

    def _crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash before finalize")

    monkeypatch.setattr(cli, "_write_investigation_artifacts", _crash)
    with pytest.raises(RuntimeError, match="simulated crash before finalize"):
        cli.main(["approve", "crash-before-finalize-thread"])
    # `monkeypatch.undo()` would also revert `monkeypatch.chdir` above --
    # restore only the one attribute this test actually broke.
    monkeypatch.setattr(cli, "_write_investigation_artifacts", real_write_artifacts)

    db_path = tmp_path / "results" / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        row_before_retry = read_decision_for_thread(
            conn, "crash-before-finalize-thread"
        )
    assert row_before_retry is not None
    assert not (
        tmp_path / "results" / "investigations" / "crash-before-finalize-thread"
    ).exists()
    # The property this test's name claims: the graph genuinely settled --
    # no pending interrupt, a real report already in the checkpoint -- not
    # merely "the function that writes artifacts happened to be the one
    # patched." Without this, a future `DISPATCH_TOOL` route that lets
    # `escalation_interrupt` re-pause could make this test a silent
    # duplicate of the one above and stay green.
    snapshot_before_retry = _checkpoint_snapshot(
        tmp_path, "crash-before-finalize-thread"
    )
    assert not snapshot_before_retry.interrupts
    assert snapshot_before_retry.next == ()
    assert snapshot_before_retry.values.get("report") is not None

    exit_status = cli.main(["approve", "crash-before-finalize-thread"])

    assert exit_status == 0
    report = _report_json(tmp_path, "crash-before-finalize-thread")
    assert report.escalation is not None
    assert report.escalation.decision == "accept"


def test_a_two_process_approve_settles_and_writes_full_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "approve-thread")

    exit_status = cli.main(["approve", "approve-thread"])

    assert exit_status == 0
    report = _report_json(tmp_path, "approve-thread")
    assert report.escalation is not None
    assert report.escalation.decision == "accept"
    assert report.escalation.rejection_note is None

    events_path = (
        tmp_path / "results" / "investigations" / "approve-thread" / "events.jsonl"
    )
    event_names = [
        json.loads(line)["name"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    # The pause happened mid-run (turn 0 stops before any dispatch) --
    # `investigation_started` and the turn-0 `stage_started`/`stage_finished`
    # trio must all survive the resume, not just the post-resume
    # `escalation_decided` event a truncated `events.jsonl` would still show.
    assert "investigation_started" in event_names
    assert "stage_started" in event_names
    assert "escalation_decided" in event_names


def test_a_two_process_reject_records_the_note_in_the_report_and_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "reject-thread")

    exit_status = cli.main(["reject", "reject-thread", "the citation looks wrong"])

    assert exit_status == 0
    report = _report_json(tmp_path, "reject-thread")
    assert report.escalation is not None
    assert report.escalation.decision == "reject"
    assert report.escalation.rejection_note == "the citation looks wrong"

    db_path = tmp_path / "results" / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        row = read_decision_for_thread(conn, "reject-thread")
    assert row is not None
    assert row.rejection_note == "the citation looks wrong"


def test_an_identical_retry_does_not_resume_the_graph_a_second_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken retry check looks like it worked -- resuming a settled
    thread is a silent no-op, and `finalize_investigation` refuses a
    second write for the same investigation id
    (`RESULT_ALREADY_FINALIZED`), which would surface as a `FAIL` on the
    *second* call if the short-circuit never fired. Asserting exit code 0
    on both calls is therefore already a strong check; the row count and
    unchanged artifact mtime are the direct ones."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "retry-thread")

    first_exit = cli.main(["approve", "retry-thread"])
    report_path = (
        tmp_path / "results" / "investigations" / "retry-thread" / "report.json"
    )
    first_mtime = report_path.stat().st_mtime_ns

    second_exit = cli.main(["approve", "retry-thread"])

    assert first_exit == 0
    assert second_exit == 0
    assert report_path.stat().st_mtime_ns == first_mtime

    db_path = tmp_path / "results" / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM owner_decisions WHERE thread_id = ?",
            ("retry-thread",),
        ).fetchone()
    assert rows[0] == 1


def test_a_conflicting_retry_is_refused_and_leaves_the_original_decision_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "conflict-thread")

    first_exit = cli.main(["approve", "conflict-thread"])
    second_exit = cli.main(["reject", "conflict-thread", "actually no"])

    assert first_exit == 0
    assert second_exit == 1
    report = _report_json(tmp_path, "conflict-thread")
    assert report.escalation is not None
    assert report.escalation.decision == "accept"


def test_a_malformed_rejection_reason_fails_before_anything_durable_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "malformed-thread")

    exit_status = cli.main(["reject", "malformed-thread", "   "])

    assert exit_status == 1
    db_path = tmp_path / "results" / "checkpoints.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        ensure_decisions_table(conn)
        assert read_decision_for_thread(conn, "malformed-thread") is None
    assert not (tmp_path / "results" / "investigations" / "malformed-thread").exists()


def test_approve_and_reject_never_fall_through_to_investigate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the explicit `main` dispatch branch: before this unit, `main`
    fell through to `run_investigate_command` for any unrecognized command
    value, which would have made `approve`/`reject` silently run an
    investigation instead of refusing on an unknown thread."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def _must_not_be_called(*args: object, **kwargs: object) -> int:
        raise AssertionError("run_investigate_command must not run for approve/reject")

    monkeypatch.setattr(cli, "run_investigate_command", _must_not_be_called)

    assert cli.main(["approve", "no-such-thread"]) == 1
    assert cli.main(["reject", "no-such-thread", "a reason"]) == 1


def test_approve_returns_a_zero_exit_code_for_a_diagnosed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_report_exit`'s convention (exit 1 for `FAILED_SAFE`, 0 otherwise)
    for a settled decision, not only for a fresh `investigate`. This checks
    only the reachable branch: `_pause_a_thread`'s scripted `DIAGNOSED`
    assessment settles to a non-`FAILED_SAFE` report on accept, and nothing
    in `escalation_interrupt`'s resume path (no model call, no tool call)
    gives this milestone's fixed replay fixtures a way to force the
    `FAILED_SAFE` branch after a valid decision -- so this test captures
    the real exit code and asserts it, rather than promising coverage of a
    branch it cannot actually reach."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _write_incident(tmp_path)
    _pause_a_thread(tmp_path, "diagnosed-thread")

    exit_status = cli.main(["approve", "diagnosed-thread"])

    assert exit_status == 0
    report = _report_json(tmp_path, "diagnosed-thread")
    assert report.disposition is not Disposition.FAILED_SAFE
