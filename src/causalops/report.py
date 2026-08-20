"""The Markdown report that sits beside the finalized run record."""

from collections.abc import Sequence

from causalops.domain import Disposition, Evidence, InvestigationReport, ToolReceipt

REPLAY_CAVEAT = (
    "This run used scripted replay fixtures. It shows that the workflow, policy, "
    "and tools behave as specified. It is not evidence of diagnostic accuracy."
)


def render_report(
    report: InvestigationReport,
    evidence: Sequence[Evidence],
    receipts: Sequence[ToolReceipt],
    model_name: str,
) -> str:
    lines = [
        f"# Investigation {report.investigation_id}",
        "",
        f"- Incident: `{report.incident_id}`",
        f"- Disposition: **{report.disposition.value}**",
        f"- Root cause: **{report.root_cause.value}**",
        f"- Model: {model_name}",
        f"- Started: {report.started_at.isoformat()}",
        f"- Latency: {report.latency_ms} ms",
    ]
    if report.reason_code is not None:
        lines.append(f"- Stopped because: `{report.reason_code.value}`")
    lines.extend(["", *assessment_section(report)])
    lines.extend(["", *evidence_section(report, evidence)])
    lines.extend(["", *checks_section(receipts)])
    lines.extend(["", *budget_section(report)])
    lines.extend(["", *limitations_section(report, model_name)])
    return "\n".join(lines) + "\n"


def assessment_section(report: InvestigationReport) -> list[str]:
    lines = ["## What it concluded", ""]
    if report.assessment is None:
        lines.append(
            "The workflow stopped before a valid assessment, so application code "
            "recorded a safe failure instead of an answer."
        )
        return lines
    lines.append(f"{report.assessment.uncertainty}")
    lines.extend(["", f"Proposed next step: {report.assessment.next_step}"])
    return lines


def evidence_section(
    report: InvestigationReport, evidence: Sequence[Evidence]
) -> list[str]:
    lines = ["## Evidence it cited", ""]
    cited = set(
        report.assessment.supporting_evidence_ids
        + report.assessment.contrary_evidence_ids
        if report.assessment is not None
        else ()
    )
    if not cited:
        lines.append("No evidence was cited.")
        return lines
    for record in evidence:
        if record.evidence_id in cited:
            lines.append(
                f"- `{record.evidence_id}` [{record.kind.value}] from "
                f"{record.source}: {record.summary}"
            )
    lines.extend(["", f"{len(evidence)} evidence records were collected in total."])
    return lines


def checks_section(receipts: Sequence[ToolReceipt]) -> list[str]:
    lines = ["## Checks it asked for", ""]
    if not receipts:
        lines.append("No checks were proposed.")
        return lines
    lines.extend(
        [
            "| Tool | Policy | Outcome | Reason | Duration |",
            "|---|---|---|---|---:|",
        ]
    )
    for receipt in receipts:
        reason = receipt.reason_code.value if receipt.reason_code else "-"
        # A receipt reaching a finished report should already be settled; a
        # reserved one here means the run stopped mid-dispatch, and the report
        # says so plainly instead of crashing on a missing outcome.
        outcome = (
            receipt.outcome.value
            if receipt.outcome is not None
            else f"unsettled ({receipt.state.value.lower()})"
        )
        lines.append(
            f"| `{receipt.tool.value}` | {receipt.policy_result.value} | "
            f"{outcome} | `{reason}` | {receipt.duration_ms} ms |"
        )
    return lines


def budget_section(report: InvestigationReport) -> list[str]:
    return [
        "## What it spent",
        "",
        f"- Model calls: {report.model_calls_used} of {report.budgets.model_calls}",
        f"- Repairs: {report.repairs_used} of {report.budgets.repairs}",
        f"- Checks executed: {report.tools_executed} of "
        f"{report.budgets.executed_tools}",
        f"- Invalid responses: {report.invalid_responses}",
        f"- Token usage: {usage_line(report)}",
        f"- Final context digest: `{report.final_context_digest[:16]}`",
    ]


def usage_line(report: InvestigationReport) -> str:
    if report.usage is None:
        return "not reported by this model"
    return f"{report.usage.input_tokens} in, {report.usage.output_tokens} out"


def limitations_section(report: InvestigationReport, model_name: str) -> list[str]:
    lines = ["## Limitations", ""]
    if model_name == "replay":
        lines.append(f"- {REPLAY_CAVEAT}")
    for limitation in report.limitations:
        lines.append(f"- {limitation}")
    if report.disposition is Disposition.FAILED_SAFE:
        lines.append(
            "- A safe failure means the workflow protected itself, not that the "
            "incident was understood."
        )
    lines.extend(
        [
            "",
            f"Versions: schema {report.versions.schema_version}, prompt "
            f"{report.versions.prompt_version}, policy "
            f"{report.versions.policy_version}, tools "
            f"{report.versions.tool_registry_version}.",
        ]
    )
    return lines
