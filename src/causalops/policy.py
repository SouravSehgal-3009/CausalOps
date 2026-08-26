"""Deterministic authorization of a model-proposed check.

Unknown tools, templates, and filters cannot reach this module: they have no
representation in the typed arguments, so schema validation rejects them first.
What remains are the scope, ordering, and budget rules from the Investigator
tools and policy section.
"""

from collections.abc import Container
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from causalops.domain import (
    Budgets,
    IncidentScope,
    PolicyResult,
    ReasonCode,
    ToolProposal,
)
from causalops.tools import (
    GetTopologyArguments,
    QueryLogsArguments,
    SearchRunbooksArguments,
    fingerprint,
)

POLICY_VERSION = "3"


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: PolicyResult
    fingerprint: str
    reason_code: ReasonCode | None = None
    message: str = ""

    @model_validator(mode="after")
    def check_reason_code(self) -> Self:
        if self.result is PolicyResult.DENIED and self.reason_code is None:
            raise ValueError("a denial must carry a reason code")
        if self.result is PolicyResult.ALLOWED and self.reason_code is not None:
            raise ValueError("an allowed decision must not carry a reason code")
        return self


def authorize(
    proposal: ToolProposal,
    scope: IncidentScope,
    seen_fingerprints: Container[str],
    budgets: Budgets,
    tools_remaining: int,
) -> PolicyDecision:
    """Decide whether a proposed check may run.

    Lab-defect-fix Unit 3, W1: the ordinary path here is always
    `tool_wrappers.ToolWrapper.dispatch`, which resolves an omitted or
    out-of-scope window into the incident's own bounds
    (`resolve_effective_window`) before ever calling this function -- no
    call reached through that path can hand this function a `None` window.
    A caller that bypasses the wrapper (today, only this module's own
    direct-call unit tests) can still hand this function an unresolved
    window; that case is refused explicitly with `UNRESOLVED_WINDOW` below,
    as a denial rather than a crash, so a direct or future caller that skips
    normalization fails as a policy decision, not a `TypeError`.
    """
    mark = fingerprint(proposal.arguments)
    # The loop skips a check it cannot afford, so this denial is the boundary
    # guarantee the Investigator tools and policy section asks for, rather than a
    # path the workflow normally takes.
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

    if isinstance(arguments, SearchRunbooksArguments):
        # Not incident-scoped: no service, no window, so this must return
        # before the `arguments.service` read below -- `SearchRunbooksArguments`
        # has no such field and would raise `AttributeError` there. The
        # denial shape mirrors `QueryLogsArguments.row_limit` above rather
        # than inventing a new reason code: an oversized `limit` is a
        # resource-limit denial, the same category `RESULT_LIMIT_EXCEEDED`
        # already names, not a scope-escape one -- guidance has no incident
        # scope to escape.
        if arguments.limit > budgets.runbook_passages:
            return deny(
                mark,
                ReasonCode.RESULT_LIMIT_EXCEEDED,
                "that passage limit is above the budget",
            )
        return PolicyDecision(result=PolicyResult.ALLOWED, fingerprint=mark)

    if arguments.service not in scope.services:
        return deny(mark, ReasonCode.UNKNOWN_SERVICE, "that service is out of scope")
    window_start = arguments.window_start
    window_end = arguments.window_end
    # Lab-defect-fix Unit 3, W1. See this function's own docstring above --
    # the ordinary (wrapper) path never reaches here with either bound
    # unresolved; only a direct or future caller that skips normalization
    # can.
    if window_start is None or window_end is None:
        return deny(
            mark,
            ReasonCode.UNRESOLVED_WINDOW,
            "a direct caller must resolve the window before calling authorize",
        )
    if window_start < scope.started_at or window_end > scope.ended_at:
        return deny(
            mark, ReasonCode.OUTSIDE_INCIDENT_WINDOW, "that window leaves the incident"
        )
    if window_end <= window_start:
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
