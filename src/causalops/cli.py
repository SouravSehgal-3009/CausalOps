"""Command line entry point for CausalOps."""

import argparse
import os
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from causalops.doctor import (
    DoctorReport,
    ProjectPaths,
    find_project_root,
    project_root_not_found,
    run_doctor,
)
from causalops.domain import Budgets, Disposition, StoredIncident
from causalops.models import ReplayReasoningModel
from causalops.prometheus import DEFAULT_PROMETHEUS_URL
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
from causalops.telemetry import RunPaths, registered_check_runner
from causalops.workflow import run_investigation, utc_now

# Phase 1 step 1 implements only the local checks. TECHNICAL_OVERVIEW.md section 9
# also requires an authenticated model-metadata request, which arrives in Phase 3.
MODEL_CHECK_NOTE = (
    "Not checked yet: the authenticated claude-sonnet-5 metadata request "
    "arrives in a later step."
)

REPLAY_FIXTURE = Path(__file__).parent / "replay_fixtures" / "lab_diagnosis.json"


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
    model = ReplayReasoningModel(
        REPLAY_FIXTURE,
        substitutions={
            "incident_id": incident.scope.incident_id,
            "window_start": incident.scope.started_at.isoformat(),
            "window_end": incident.scope.ended_at.isoformat(),
            "symptom_evidence_id": incident.packet.symptom_evidence_id,
        },
    )
    recorder = RunRecorder(utc_now)
    result = run_investigation(
        incident.scope,
        incident.packet,
        incident.evidence,
        model,
        registered_check_runner(
            paths, DEFAULT_PROMETHEUS_URL, budgets.tool_timeout_seconds
        ),
        recorder,
        budgets,
        utc_now,
    )
    written = finalize_investigation(
        root / "results",
        result.report,
        recorder.events,
        result.evidence,
        result.receipts,
        render_markdown_report(
            result.report, result.evidence, result.receipts, model_name
        ),
    )
    print(f"{result.report.disposition.value} {result.report.root_cause.value}")
    print(f"artifacts: {written}")
    return 1 if result.report.disposition is Disposition.FAILED_SAFE else 0


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
        return run_investigate_command(root, arguments.incident_id, arguments.model)
    except (LabError, RunRecordError) as refusal:
        print(f"FAIL {refusal.reason_code.value} {refusal}")
        return 1
