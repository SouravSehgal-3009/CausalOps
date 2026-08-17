"""Deterministic authorization of a model-proposed check.

Unknown tools, templates, and filters cannot reach this module: they have no
representation in the typed arguments, so schema validation rejects them first.
What remains are the scope, ordering, and budget rules from section 7.
"""

from collections.abc import Container

from pydantic import BaseModel, ConfigDict

from causalops.domain import (
    Budgets,
    IncidentScope,
    PolicyResult,
    ReasonCode,
    ToolProposal,
)
from causalops.tools import GetTopologyArguments, QueryLogsArguments, fingerprint

POLICY_VERSION = "1"


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: PolicyResult
    fingerprint: str
    reason_code: ReasonCode | None = None
    message: str = ""


def authorize(
    proposal: ToolProposal,
    scope: IncidentScope,
    seen_fingerprints: Container[str],
    budgets: Budgets,
    tools_remaining: int,
) -> PolicyDecision:
    mark = fingerprint(proposal.arguments)
    # The loop skips a check it cannot afford, so this denial is the boundary
    # guarantee section 7 asks for rather than a path the workflow normally takes.
    if tools_remaining <= 0:
        return deny(
            mark, ReasonCode.BUDGET_EXHAUSTED, "no diagnostic check budget left"
        )
    if mark in seen_fingerprints:
        return deny(
            mark, ReasonCode.DUPLICATE_PROPOSAL, "this check was proposed already"
        )

    arguments = proposal.arguments
    if isinstance(arguments, GetTopologyArguments):
        if arguments.incident_id != scope.incident_id:
            return deny(
                mark, ReasonCode.CROSS_INCIDENT_REQUEST, "that is another incident"
            )
        return PolicyDecision(result=PolicyResult.ALLOWED, fingerprint=mark)

    if arguments.service not in scope.services:
        return deny(mark, ReasonCode.UNKNOWN_SERVICE, "that service is out of scope")
    if (
        arguments.window_start < scope.started_at
        or arguments.window_end > scope.ended_at
    ):
        return deny(
            mark, ReasonCode.OUTSIDE_INCIDENT_WINDOW, "that window leaves the incident"
        )
    if arguments.window_end <= arguments.window_start:
        return deny(
            mark,
            ReasonCode.OUTSIDE_INCIDENT_WINDOW,
            "that window ends before it starts",
        )
    if (
        isinstance(arguments, QueryLogsArguments)
        and arguments.row_limit > budgets.log_rows
    ):
        return deny(
            mark, ReasonCode.RESULT_LIMIT_EXCEEDED, "that row limit is above the budget"
        )
    return PolicyDecision(result=PolicyResult.ALLOWED, fingerprint=mark)


def deny(mark: str, reason_code: ReasonCode, message: str) -> PolicyDecision:
    return PolicyDecision(
        result=PolicyResult.DENIED,
        fingerprint=mark,
        reason_code=reason_code,
        message=message,
    )
