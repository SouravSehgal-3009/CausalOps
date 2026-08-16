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
from causalops.system_probe import SystemProbe

# Phase 1 step 1 implements only the local checks. TECHNICAL_OVERVIEW.md section 9
# also requires an authenticated model-metadata request, which arrives in Phase 3.
MODEL_CHECK_NOTE = (
    "Not checked yet: the authenticated claude-sonnet-5 metadata request "
    "arrives in a later step."
)


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


def main(argv: Sequence[str] | None = None) -> int:
    # `doctor` is the only subcommand, so parsing exists to reject anything else.
    build_parser().parse_args(argv)
    start = Path.cwd()
    root = find_project_root(start)
    if root is None:
        report = DoctorReport(checks=(project_root_not_found(start),))
    else:
        report = run_doctor(ProjectPaths(root=root), SystemProbe(), os.environ)
    print(render_report(report))
    return exit_code(report)
