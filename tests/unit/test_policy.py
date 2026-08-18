from datetime import timedelta

import pytest
from fake_incident import (
    WINDOW_END,
    WINDOW_START,
    incident_scope,
    logs_proposal,
    metric_proposal,
)
from pydantic import ValidationError

from causalops.domain import Budgets, PolicyResult, ReasonCode, ToolProposal
from causalops.policy import PolicyDecision, authorize
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    QueryLogsArguments,
    QueryMetricArguments,
)
from causalops.tools import fingerprint as tool_fingerprint

BUDGETS = Budgets()


def decide(
    proposal: ToolProposal, seen: set[str] | None = None, tools_remaining: int = 2
) -> tuple[PolicyResult, ReasonCode | None]:
    decision = authorize(
        proposal, incident_scope(), seen or set(), BUDGETS, tools_remaining
    )
    return decision.result, decision.reason_code


def topology_proposal(incident_id: str) -> ToolProposal:
    return ToolProposal(
        arguments=GetTopologyArguments(incident_id=incident_id),
        evidence_gap="what calls what",
        expected_observation="the service graph",
    )


def test_a_scoped_check_is_allowed() -> None:
    assert decide(metric_proposal()) == (PolicyResult.ALLOWED, None)


def test_a_service_outside_the_incident_is_denied() -> None:
    assert decide(metric_proposal(service="billing")) == (
        PolicyResult.DENIED,
        ReasonCode.UNKNOWN_SERVICE,
    )


def test_another_incident_is_denied() -> None:
    assert decide(topology_proposal("some-other-incident")) == (
        PolicyResult.DENIED,
        ReasonCode.CROSS_INCIDENT_REQUEST,
    )
    assert decide(topology_proposal(incident_scope().incident_id))[0] is (
        PolicyResult.ALLOWED
    )


def test_a_window_outside_the_incident_is_denied() -> None:
    before = ToolProposal(
        arguments=QueryMetricArguments(
            template=metric_proposal().arguments.template,  # type: ignore[union-attr]
            service="gateway",
            window_start=WINDOW_START - timedelta(minutes=5),
            window_end=WINDOW_END,
        ),
        evidence_gap="latency before the incident",
        expected_observation="a baseline",
    )

    assert decide(before) == (
        PolicyResult.DENIED,
        ReasonCode.OUTSIDE_INCIDENT_WINDOW,
    )


def test_a_window_that_extends_past_the_incident_is_denied() -> None:
    past_the_end = ToolProposal(
        arguments=QueryMetricArguments(
            template=metric_proposal().arguments.template,  # type: ignore[union-attr]
            service="gateway",
            window_start=WINDOW_START,
            window_end=WINDOW_END + timedelta(minutes=5),
        ),
        evidence_gap="latency after the incident window closed",
        expected_observation="a drop back to baseline",
    )

    assert decide(past_the_end) == (
        PolicyResult.DENIED,
        ReasonCode.OUTSIDE_INCIDENT_WINDOW,
    )


def test_a_backwards_window_is_denied() -> None:
    backwards = ToolProposal(
        arguments=QueryMetricArguments(
            template=metric_proposal().arguments.template,  # type: ignore[union-attr]
            service="gateway",
            window_start=WINDOW_END,
            window_end=WINDOW_END,
        ),
        evidence_gap="an empty window",
        expected_observation="nothing",
    )

    assert decide(backwards) == (
        PolicyResult.DENIED,
        ReasonCode.OUTSIDE_INCIDENT_WINDOW,
    )


def test_a_row_limit_above_the_budget_is_denied() -> None:
    assert decide(logs_proposal(row_limit=BUDGETS.log_rows + 1)) == (
        PolicyResult.DENIED,
        ReasonCode.RESULT_LIMIT_EXCEEDED,
    )
    assert decide(logs_proposal(row_limit=BUDGETS.log_rows))[0] is PolicyResult.ALLOWED


def test_the_same_proposal_twice_is_denied() -> None:
    seen = {tool_fingerprint(metric_proposal().arguments)}

    assert decide(metric_proposal(), seen=seen) == (
        PolicyResult.DENIED,
        ReasonCode.DUPLICATE_PROPOSAL,
    )


def test_a_proposal_without_check_budget_is_denied() -> None:
    assert decide(metric_proposal(), tools_remaining=0) == (
        PolicyResult.DENIED,
        ReasonCode.BUDGET_EXHAUSTED,
    )


def test_a_denial_carries_a_fingerprint_so_the_receipt_can_record_it() -> None:
    decision = authorize(
        metric_proposal(service="billing"), incident_scope(), set(), BUDGETS, 2
    )

    assert decision.fingerprint == tool_fingerprint(
        metric_proposal(service="billing").arguments
    )
    assert decision.message


def test_a_denial_without_a_reason_code_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyDecision(result=PolicyResult.DENIED, fingerprint="f")


def test_an_allowed_decision_with_a_reason_code_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyDecision(
            result=PolicyResult.ALLOWED,
            fingerprint="f",
            reason_code=ReasonCode.UNKNOWN_SERVICE,
        )


@pytest.mark.parametrize("service", ["../../etc/passwd", "orders/../../secret"])
def test_a_path_traversal_shaped_service_is_denied_before_any_backend_path_join(
    service: str,
) -> None:
    """The allowlist check runs before a service name could reach a file path.

    Proves the block happens in `authorize`, ahead of the tool backends that
    later join a service name onto a run directory path.
    """
    logs = ToolProposal(
        arguments=QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY,
            service=service,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            row_limit=20,
        ),
        evidence_gap="whether that path holds relevant logs",
        expected_observation="nothing, because the service is not registered",
    )
    changes = ToolProposal(
        arguments=ListRecentChangesArguments(
            service=service, window_start=WINDOW_START, window_end=WINDOW_END
        ),
        evidence_gap="whether that path holds relevant changes",
        expected_observation="nothing, because the service is not registered",
    )

    assert decide(logs) == (PolicyResult.DENIED, ReasonCode.UNKNOWN_SERVICE)
    assert decide(changes) == (PolicyResult.DENIED, ReasonCode.UNKNOWN_SERVICE)
