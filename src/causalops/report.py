"""The Markdown report that sits beside the finalized run record."""

from collections.abc import Sequence

from causalops.domain import (
    Disposition,
    Evidence,
    InvestigationReport,
    PolicyResult,
    ReceiptState,
    ToolReceipt,
)
from causalops.tools import ToolName

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
    lines.extend(["", *budget_section(report, receipts)])
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


def budget_section(
    report: InvestigationReport, receipts: Sequence[ToolReceipt]
) -> list[str]:
    # `report.tools_executed` (`InvestigationReport`'s own schema field)
    # counts every settled, allowed receipt across all five tools --
    # correct as a total, but `search_runbooks` spends from its own
    # `budgets.runbook_searches` pool, separate from `budgets.executed_tools`
    # the other four tools share (see `Budgets.runbook_searches`'s own
    # docstring). Comparing that combined total against the single
    # `executed_tools` denominator can read as "3 of 2" -- over budget --
    # for a run that used both pools correctly and fully. Split here, the
    # same fix `model_calls`/`repairs` above already needed for the same
    # reason (see that line's own comment).
    #
    # Subtracted from the trusted schema total, not independently recounted
    # from `receipts` alone: `receipts` is a second parameter a caller could
    # in principle pass out of sync with the report that produced
    # `tools_executed` (every real caller passes the matching list, but
    # nothing enforces that structurally). Subtraction means a caller that
    # passes no/mismatched receipts still gets the report's own honest
    # total on the diagnostic line, rather than silently under-reporting it
    # as zero.
    runbook_searches_executed = sum(
        1
        for receipt in receipts
        if receipt.policy_result is PolicyResult.ALLOWED
        and receipt.state is ReceiptState.SETTLED
        and receipt.tool is ToolName.SEARCH_RUNBOOKS
    )
    diagnostic_executed = report.tools_executed - runbook_searches_executed
    return [
        "## What it spent",
        "",
        # `model_calls_used` counts every attempt, including repairs, which
        # are now funded from the separate `budgets.repairs` pool -- the
        # true ceiling a run can reach is the two pools added together, not
        # `budgets.model_calls` alone (that would misreport, e.g., "7 of 6"
        # after a repair).
        f"- Model calls: {report.model_calls_used} of "
        f"{report.budgets.model_calls + report.budgets.repairs}",
        f"- Repairs: {report.repairs_used} of {report.budgets.repairs}",
        f"- Diagnostic checks executed: {diagnostic_executed} of "
        f"{report.budgets.executed_tools}",
        f"- Runbook searches: {runbook_searches_executed} of "
        f"{report.budgets.runbook_searches}",
        f"- Invalid responses: {report.invalid_responses}",
        f"- Token usage: {usage_line(report)}",
        # The CLI report must surface this value.
        # Printed even when `disabled` -- an owner should be able to
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
