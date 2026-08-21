"""Command line entry point for CausalOps."""

import argparse
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from importlib.metadata import version
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from causalops.approvals import (
    ApprovalError,
    ApprovalReasonCode,
    OwnerDecision,
    ensure_decisions_table,
    read_decision_for_thread,
    record_decision_before_resume,
)
from causalops.doctor import (
    DoctorReport,
    ProjectPaths,
    find_project_root,
    project_root_not_found,
    run_doctor,
)
from causalops.domain import (
    Budgets,
    Disposition,
    EscalatedInvestigation,
    InvestigationReport,
    InvestigationResult,
    StoredIncident,
    utc_now,
)
from causalops.graph import (
    build_graph,
    resume_graph_investigation,
    run_graph_investigation,
)
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.prometheus import DEFAULT_PROMETHEUS_URL, run_metric_check
from causalops.report import render_report as render_markdown_report
from causalops.run_records import RunRecorder, RunRecordError, finalize_investigation
from causalops.scenario_control import (
    LabError,
    LabReasonCode,
    lab_down,
    lab_up,
    reset_scenario,
    start_scenario,
)
from causalops.system_probe import SystemProbe
from causalops.telemetry import (
    RunPaths,
    run_changes_check,
    run_logs_check,
    run_topology_check,
)
from causalops.tool_wrappers import ToolWrapper, dispatch_registry
from causalops.tools import ToolName

# Every caller of `run_investigation`/`resume_graph_investigation` in this
# file supplies this literally -- `--model` only ever accepts `"replay"`
# (`build_parser` below), and `approve`/`reject` never took a `--model` flag
# at all, since resuming re-opens the same investigation the original
# `investigate` call started. Named here so both call sites read the same
# constant instead of two independent string literals that could drift.
REPLAY_MODEL_NAME = "replay"

# Phase 1 step 1 implements only the local checks. TECHNICAL_OVERVIEW.md's
# Tests specified for the live Claude adapter section describes the
# authenticated model-metadata request this still lacks; it arrives with the
# live Claude adapter, not yet scheduled to a specific v2 unit.
MODEL_CHECK_NOTE = (
    "Not checked yet: the authenticated claude-sonnet-5 metadata request "
    "arrives in a later step."
)

REPLAY_FIXTURE_DIR = Path(__file__).parent / "replay_fixtures"
# `dispatch_registry` wraps all four tools as of Unit 1c, so the graph
# orchestrator runs `lab_diagnosis.json` -- two executed checks across two
# tools -- exactly as the retired loop orchestrator did; parity between the
# two was established in Unit 1d-1 before the loop was retired.
REPLAY_FIXTURE = REPLAY_FIXTURE_DIR / "lab_diagnosis.json"

# Unit 2b. `0` already means success and `1` already means `FAILED_SAFE`/a
# `LabError`/`RunRecordError` refusal (see `main`'s `except` clause below);
# `argparse` itself exits `2` on a usage error before `main`'s own body ever
# runs. An escalated, paused investigation is none of those three, so it
# gets its own code rather than overloading one of them.
EXIT_ESCALATED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalops",
        description="Evidence-grounded incident investigator for a local lab.",
    )
    parser.add_argument(
        "--version", action="version", version=f"causalops {version('causalops')}"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="Check that this machine can run CausalOps.")

    lab = subcommands.add_parser("lab", help="Start or stop the synthetic lab.")
    lab_actions = lab.add_subparsers(dest="action", required=True)
    lab_actions.add_parser("up", help="Start the lab and wait for it to be healthy.")
    lab_actions.add_parser("down", help="Stop the lab.")

    scenario = subcommands.add_parser("scenario", help="Create or clear an incident.")
    scenario_actions = scenario.add_subparsers(dest="action", required=True)
    start = scenario_actions.add_parser("start", help="Create one incident.")
    start.add_argument("family", help="Owner-facing scenario family name.")
    start.add_argument("--seed", choices=("development", "evaluation"), required=True)
    reset = scenario_actions.add_parser("reset", help="Delete one incident's state.")
    reset.add_argument("incident_id")

    investigation = subcommands.add_parser(
        "investigate", help="Investigate one opaque incident ID."
    )
    investigation.add_argument("incident_id")
    investigation.add_argument("--model", choices=("replay",), required=True)

    approve = subcommands.add_parser(
        "approve", help="Accept a paused investigation's diagnosis or abstention."
    )
    approve.add_argument("thread_id")

    reject = subcommands.add_parser(
        "reject", help="Reject a paused investigation and record why."
    )
    reject.add_argument("thread_id")
    reject.add_argument("reason")

    return parser


def render_report(report: DoctorReport) -> str:
    name_width = max((len(check.name) for check in report.checks), default=0)
    codes = [
        check.reason_code.value if check.reason_code else "-" for check in report.checks
    ]
    code_width = max((len(code) for code in codes), default=0)
    lines = [
        f"{check.status.value:<4} {check.name:<{name_width}} "
        f"{code:<{code_width}} {check.message}"
        for check, code in zip(report.checks, codes, strict=True)
    ]
    lines.append("")
    lines.append(MODEL_CHECK_NOTE)
    failures = report.failures
    if not failures:
        lines.append("doctor: OK")
        return "\n".join(lines)
    noun = "check" if len(failures) == 1 else "checks"
    lines.append(f"doctor: FAILED ({len(failures)} {noun})")
    return "\n".join(lines)


def exit_code(report: DoctorReport) -> int:
    return 1 if report.failures else 0


def run_doctor_command(root: Path) -> int:
    report = run_doctor(ProjectPaths(root=root), SystemProbe(), os.environ)
    print(render_report(report))
    return exit_code(report)


def run_lab_command(root: Path, action: str) -> int:
    if action == "up":
        lab_up(root)
        print("lab: up")
        return 0
    lab_down(root)
    print("lab: down")
    return 0


def run_scenario_command(root: Path, arguments: argparse.Namespace) -> int:
    if arguments.action == "start":
        incident_id = start_scenario(root, arguments.family, arguments.seed)
        print(incident_id)
        return 0
    reset_scenario(root, arguments.incident_id)
    print(f"scenario: reset {arguments.incident_id}")
    return 0


@contextmanager
def _sqlite_checkpointer(db_path: Path) -> Iterator[SqliteSaver]:
    """A `SqliteSaver` connected to `db_path`, open for the caller's
    `with` block.

    Deserialization is restricted to LangGraph's built-in safe-type allowlist
    -- the same restriction `LANGGRAPH_STRICT_MSGPACK=true` applies, set here
    as a constructor argument instead of an environment variable so the one
    place this database connection is opened is also the one place this
    policy is decided, rather than adding another variable to
    `.env.example`. `checkpoints.db` is a real file on disk for the life of
    an investigation; a permissive serializer would import and instantiate
    whatever class name a checkpoint blob claims, which turns write access to
    that file into code execution on the next read. `SqliteSaver.from_conn_string`
    doesn't expose a `serde` parameter, so the connection is opened directly,
    matching what that classmethod does internally.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=serde)


def _build_model_and_registry(
    incident: StoredIncident, paths: RunPaths, budgets: Budgets
) -> tuple[ReplayToolCallingModel, Mapping[ToolName, ToolWrapper]]:
    """The model and tool registry one incident needs, built the same way
    for a fresh `investigate` and for Unit 2c's `approve`/`reject` resume --
    both need the exact same wiring `build_graph` requires, even though a
    plain accept/reject resume never actually calls the model or a tool
    again (`escalation_interrupt` routes straight to `final_report`)."""
    model = ReplayToolCallingModel(
        ReplayReasoningModel(
            REPLAY_FIXTURE,
            substitutions={
                "incident_id": incident.scope.incident_id,
                "window_start": incident.scope.started_at.isoformat(),
                "window_end": incident.scope.ended_at.isoformat(),
                "symptom_evidence_id": incident.packet.symptom_evidence_id,
            },
        )
    )
    registry = dispatch_registry(
        run_metric=lambda arguments, scope: run_metric_check(
            arguments, scope, DEFAULT_PROMETHEUS_URL, budgets.tool_timeout_seconds
        ),
        run_logs=lambda arguments, scope: run_logs_check(arguments, paths),
        run_changes=lambda arguments, scope: run_changes_check(arguments, paths),
        run_topology=lambda arguments, scope: run_topology_check(arguments, paths),
    )
    return model, registry


def run_investigation(
    incident: StoredIncident,
    paths: RunPaths,
    budgets: Budgets,
    recorder: RunRecorder,
    checkpointer: BaseCheckpointSaver[str],
) -> InvestigationResult | EscalatedInvestigation:
    model, registry = _build_model_and_registry(incident, paths, budgets)
    return run_graph_investigation(
        incident.scope,
        incident.packet,
        incident.evidence,
        model,
        registry,
        recorder,
        budgets,
        utc_now,
        checkpointer=checkpointer,
    )


def _write_investigation_artifacts(
    root: Path, result: InvestigationResult, recorder: RunRecorder, model_name: str
) -> Path:
    """Finalizes one settled investigation's artifacts -- shared by
    `run_investigate_command` (a fresh settle) and `run_decision_command`
    (Unit 2c's approve/reject settle), so this write happens in exactly one
    place regardless of which command produced the terminal result."""
    return finalize_investigation(
        root / "results",
        result.report,
        recorder.events,
        result.evidence,
        result.receipts,
        render_markdown_report(
            result.report, result.evidence, result.receipts, model_name
        ),
    )


def _report_exit(report: InvestigationReport, artifact_path: Path) -> int:
    print(f"{report.disposition.value} {report.root_cause.value}")
    print(report.investigation_id)
    print(f"artifacts: {artifact_path}")
    return 1 if report.disposition is Disposition.FAILED_SAFE else 0


# Loads the incident and writes the investigation's result -- the CLI's one
# job for `investigate`.
def run_investigate_command(root: Path, incident_id: str, model_name: str) -> int:
    paths = RunPaths(root=root / "runs" / incident_id)
    if not paths.incident_file.is_file():
        raise LabError(
            LabReasonCode.INCIDENT_NOT_FOUND, f"no run directory for {incident_id}"
        )
    incident = StoredIncident.model_validate_json(
        paths.incident_file.read_text(encoding="utf-8")
    )
    budgets = Budgets()
    recorder = RunRecorder(utc_now)
    db_path = root / "results" / "checkpoints.db"
    with _sqlite_checkpointer(db_path) as checkpointer:
        result = run_investigation(incident, paths, budgets, recorder, checkpointer)
    if isinstance(result, EscalatedInvestigation):
        # No terminal report exists yet for a paused run -- print what the
        # owner needs to resume it (`causalops approve`/`reject`, Unit 2c)
        # and stop, without ever calling `finalize_investigation`. That call
        # refuses a second write for an already-finalized investigation id
        # (`RESULT_ALREADY_FINALIZED`); calling it now, on a run that has not
        # produced a report yet, would instead poison the real terminal
        # write this thread gets once it is actually resumed and accepted
        # or rejected.
        print(f"ESCALATED {result.reason.value} {result.thread_id}")
        print(f"remaining checks: {result.remaining_check_count}")
        return EXIT_ESCALATED
    written = _write_investigation_artifacts(root, result, recorder, model_name)
    return _report_exit(result.report, written)


def _resolve_thread_incident_id(checkpointer: SqliteSaver, thread_id: str) -> str:
    """The only durable link from a bare thread id to its incident: nothing
    on disk maps one to the other outside the checkpoint itself --
    `investigation_id` is a fresh `uuid4().hex` and `results/investigations/
    <id>/` does not exist yet for a paused run. Reads with no graph built,
    the same pattern `test_a_second_connection_reads_back_the_finished_run`
    established in Unit 2a."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    checkpoint = checkpointer.get_tuple(config)
    if checkpoint is None:
        raise ApprovalError(
            ApprovalReasonCode.THREAD_NOT_FOUND, f"no checkpoint for thread {thread_id}"
        )
    incident_id = checkpoint.checkpoint["channel_values"].get("incident_id")
    if not isinstance(incident_id, str):
        raise ApprovalError(
            ApprovalReasonCode.THREAD_NOT_FOUND,
            f"thread {thread_id} has no recorded incident_id",
        )
    return incident_id


def run_decision_command(
    root: Path, thread_id: str, owner_decision: OwnerDecision
) -> int:
    """Resumes a paused investigation from a second process, behind the
    append-only `owner_decisions` record.

    Ordering, forced by what each step needs and what each guard protects
    against (a resume that silently does nothing, and a resume that
    silently repeats):

    1. Look up any existing decision for `thread_id` -- by thread alone, not
       the `(thread_id, checkpoint_id)` composite key the write below uses,
       because a settled thread's *current* checkpoint id is not the one
       its decision was recorded against (`escalation_interrupt` and
       `final_report` each commit a checkpoint after the resume). A
       mismatched existing decision is refused immediately, before the
       checkpoint database or the incident file is even opened.
    2. If a matching decision already exists *and* the artifacts are
       already on disk, this is an identical retry: report success from the
       finalized `report.json` directly. The graph is never touched.
    3. Otherwise -- a first decision, or a matching decision whose resume
       never finished (a crash between the record write and finalize) --
       resolve the incident, build the model/registry/graph, and check for
       a pending interrupt (Unit 2b's measured trap: `.interrupts`, never
       `.next`, since a re-paused run and a settled run both show `.next
       == ()`).
    4. A first decision with no pending interrupt is refused
       (`NO_PENDING_INTERRUPT`): this thread never paused through this
       mechanism, or was resumed outside it. A first decision with a
       pending interrupt is recorded -- before resume, per
       `TECHNICAL_SPEC.md:162-164` -- against the checkpoint id the pending
       snapshot names. A matching-but-unfinished decision skips this write
       (it already exists) and goes straight to resume; whether the graph
       is still pending or already settled from a prior attempt,
       `resume_graph_investigation`'s `Command(resume=...)` call handles
       both -- LangGraph resumes a still-pending interrupt normally and is
       a documented silent no-op, returning the finished state unchanged,
       on an already-settled one.
    """
    db_path = root / "results" / "checkpoints.db"
    investigations_dir = root / "results" / "investigations" / thread_id
    # `_sqlite_checkpointer` below does the same `mkdir` for its own
    # connection; this one is opened first (guard #4's lookup needs no
    # graph, no incident, and ideally no checkpointer either), so it needs
    # its own -- otherwise a project that has never run `investigate` has
    # no `results/` directory yet, and `sqlite3.connect` raises a bare
    # `OperationalError` instead of the `THREAD_NOT_FOUND` this really is.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    decisions_connection = sqlite3.connect(str(db_path), check_same_thread=False)
    with closing(decisions_connection) as decisions_conn:
        ensure_decisions_table(decisions_conn)
        existing = read_decision_for_thread(decisions_conn, thread_id)

        if existing is not None and not existing.matches(owner_decision):
            raise ApprovalError(
                ApprovalReasonCode.CONFLICTING_DECISION,
                f"{thread_id} already recorded decision={existing.decision!r} "
                f"rejection_note={existing.rejection_note!r}; this request "
                f"asked for decision={owner_decision.decision!r} "
                f"rejection_note={owner_decision.rejection_note!r}",
            )
        if existing is not None and investigations_dir.is_dir():
            # An identical retry reads the already-finalized artifact back
            # rather than touching the graph again. That artifact is this
            # store's own prior output, but it is still a file on disk a
            # corrupted write or a hand-edited byte could make unreadable
            # -- refused the same way a corrupt `owner_decisions` row is
            # above, not surfaced as an uncaught traceback.
            try:
                report = InvestigationReport.model_validate_json(
                    (investigations_dir / "report.json").read_text(encoding="utf-8")
                )
            except ValidationError as error:
                raise ApprovalError(
                    ApprovalReasonCode.STORE_UNAVAILABLE,
                    f"{investigations_dir / 'report.json'} is unreadable: {error}",
                ) from error
            return _report_exit(report, investigations_dir)

        with _sqlite_checkpointer(db_path) as checkpointer:
            incident_id = _resolve_thread_incident_id(checkpointer, thread_id)
            paths = RunPaths(root=root / "runs" / incident_id)
            if not paths.incident_file.is_file():
                raise LabError(
                    LabReasonCode.INCIDENT_NOT_FOUND,
                    f"no run directory for {incident_id}",
                )
            incident = StoredIncident.model_validate_json(
                paths.incident_file.read_text(encoding="utf-8")
            )
            budgets = Budgets()
            model, registry = _build_model_and_registry(incident, paths, budgets)
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            snapshot = build_graph(
                incident.scope,
                incident.packet,
                budgets,
                utc_now,
                model,
                registry,
                checkpointer,
                event_clock=utc_now,
            ).get_state(config)

            if existing is None:
                if not snapshot.interrupts:
                    raise ApprovalError(
                        ApprovalReasonCode.NO_PENDING_INTERRUPT,
                        f"{thread_id} has no pending approval to decide",
                    )
                checkpoint_id = str(snapshot.config["configurable"]["checkpoint_id"])
                record_decision_before_resume(
                    decisions_conn, thread_id, checkpoint_id, owner_decision, utc_now()
                )

            recorder = RunRecorder(utc_now)
            result = resume_graph_investigation(
                thread_id,
                checkpointer,
                incident.scope,
                incident.packet,
                model,
                registry,
                recorder,
                owner_decision.decision,
                owner_decision.rejection_note,
                budgets,
                utc_now,
            )

        written = _write_investigation_artifacts(
            root, result, recorder, REPLAY_MODEL_NAME
        )
        return _report_exit(result.report, written)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    start = Path.cwd()
    root = find_project_root(start)
    if root is None:
        print(render_report(DoctorReport(checks=(project_root_not_found(start),))))
        return 1
    if arguments.command == "doctor":
        return run_doctor_command(root)
    try:
        if arguments.command == "lab":
            return run_lab_command(root, arguments.action)
        if arguments.command == "scenario":
            return run_scenario_command(root, arguments)
        if arguments.command == "investigate":
            return run_investigate_command(root, arguments.incident_id, arguments.model)
        if arguments.command == "approve":
            return run_decision_command(
                root, arguments.thread_id, OwnerDecision(decision="accept")
            )
        if arguments.command == "reject":
            try:
                owner_decision = OwnerDecision(
                    decision="reject", rejection_note=arguments.reason
                )
            except ValidationError as error:
                raise ApprovalError(
                    ApprovalReasonCode.INVALID_REJECTION_NOTE, str(error)
                ) from error
            return run_decision_command(root, arguments.thread_id, owner_decision)
        # `argparse`'s `required=True` on `subcommands` (see `build_parser`)
        # makes every other value impossible -- this is not a fall-through
        # default, it is the explicit branch this project's own history
        # flags: leaving the old implicit fall-through here would have made
        # `approve`/`reject` silently run `run_investigate_command` instead.
        raise AssertionError(f"unhandled command {arguments.command!r}")
    except (LabError, RunRecordError, ApprovalError) as refusal:
        print(f"FAIL {refusal.reason_code.value} {refusal}")
        return 1
