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
    lines.extend(["", *guidance_section(report)])
    lines.extend(["", *checks_section(receipts)])
    lines.extend(["", *budget_section(report)])
    if report.escalation is not None:
        lines.extend(["", *escalation_section(report)])
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


def guidance_section(report: InvestigationReport) -> list[str]:
    """Incident evidence gets a full section above; guidance gets its own
    section here too, not just a `retrieval_mode` line in `budget_section`.
    `report.runbook_passage_ids` is every passage this run retrieved --
    guidance alone can never prove an incident's cause, which is why this
    reads ids, never the assessment's `supporting_evidence_ids`: guidance
    and evidence citations stay in their own separate fields, the way
    `FinalAssessment`'s own docstring already keeps them. `(cited)` marks a
    passage the model actually named in `runbook_citations`; an unmarked
    one was retrieved but not used."""
    lines = ["## Guidance it consulted", ""]
    if not report.runbook_passage_ids:
        lines.append("No runbook guidance was retrieved.")
        return lines
    cited = (
        set(report.assessment.runbook_citations)
        if report.assessment is not None
        else set()
    )
    for passage_id in report.runbook_passage_ids:
        marker = " (cited)" if passage_id in cited else ""
        lines.append(f"- `{passage_id}`{marker}")
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
        # `TECHNICAL_SPEC.md` §7: "The CLI report ... must surface this
        # value." Printed even when `disabled` -- an owner should be able to
        # tell "retrieval never ran" from "retrieval ran but this section
        # was silently dropped" by reading the report, not by knowing the
        # default.
        f"- Runbook retrieval mode: `{report.retrieval_mode.value}`",
        f"- Final context digest: `{report.final_context_digest[:16]}`",
    ]


def usage_line(report: InvestigationReport) -> str:
    if report.usage is None:
        return "not reported by this model"
    return f"{report.usage.input_tokens} in, {report.usage.output_tokens} out"


def escalation_section(report: InvestigationReport) -> list[str]:
    """Only called when `report.escalation` is set -- the caller checks, not
    this function, the same pattern every other optional section in this
    file leaves to `render_report`. `rejection_note` only ever holds text on
    a reject (`EscalationRecord.check_rejection_note_pairing` enforces the
    pairing), so the line is omitted entirely on an accept rather than
    printed empty."""
    assert report.escalation is not None
    lines = [
        "## Owner escalation",
        "",
        f"- Reason: `{report.escalation.reason.value}`",
        f"- Decision: **{report.escalation.decision}**",
    ]
    if report.escalation.rejection_note is not None:
        lines.append(f"- Owner's note: {report.escalation.rejection_note}")
    return lines


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
