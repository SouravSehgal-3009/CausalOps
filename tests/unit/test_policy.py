from datetime import timedelta

import pytest
from fake_incident import (
    WINDOW_END,
    WINDOW_START,
    incident_scope,
    logs_proposal,
    metric_proposal,
    runbooks_proposal,
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
    """Lab-defect-fix Unit 3, W1: this calls `authorize` directly, the way
    only this module's own tests and a future caller that skips
    `tool_wrappers.resolve_effective_window` would -- the ordinary wrapper
    path always resolves/clamps a window first, so a raw window entirely
    outside scope like this one never reaches `authorize` that way. This
    tests `authorize`'s own direct-call contract: a raw, unresolved window
    it is handed is still denied on its own merits, not a forgery/replay
    scenario."""
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
    """Lab-defect-fix Unit 3, W1: same direct-call contract as the test
    above -- reached only by bypassing `tool_wrappers.resolve_effective_
    window`, which would have clamped this window to the incident before
    `authorize` ever saw it."""
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
    """Lab-defect-fix Unit 3, W1: same direct-call contract as the two
    tests above. Unlike them, a backwards window stays reachable through
    the ordinary wrapper path too -- clamping narrows but cannot un-invert
    a window, see `test_a_window_entirely_outside_scope_still_denies_
    after_clamping` in `test_tool_wrappers.py` -- so this
    denial is not exclusively a direct-call concern, just tested directly
    here like its neighbours."""
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


def test_an_unresolved_start_is_denied_not_raised() -> None:
    """Lab-defect-fix Unit 3, W1/Q16(i). A direct caller (today, only this
    module's own tests) can hand `authorize` a proposal whose window was
    never resolved -- the ordinary `tool_wrappers.ToolWrapper.dispatch`
    path always resolves both bounds first. `authorize` must refuse this
    with `UNRESOLVED_WINDOW`, a denial, never a `TypeError` from comparing
    `None` against a real datetime."""
    unresolved_start = ToolProposal(
        arguments=QueryMetricArguments(
            template=metric_proposal().arguments.template,  # type: ignore[union-attr]
            service="gateway",
            window_start=None,
            window_end=WINDOW_END,
        ),
        evidence_gap="latency during the window",
        expected_observation="a latency rise",
    )

    assert decide(unresolved_start) == (
        PolicyResult.DENIED,
        ReasonCode.UNRESOLVED_WINDOW,
    )


def test_an_unresolved_end_is_denied_not_raised() -> None:
    """Same contract as the test above, the other bound."""
    unresolved_end = ToolProposal(
        arguments=QueryMetricArguments(
            template=metric_proposal().arguments.template,  # type: ignore[union-attr]
            service="gateway",
            window_start=WINDOW_START,
            window_end=None,
        ),
        evidence_gap="latency during the window",
        expected_observation="a latency rise",
    )

    assert decide(unresolved_end) == (
        PolicyResult.DENIED,
        ReasonCode.UNRESOLVED_WINDOW,
    )


def test_a_row_limit_above_the_budget_is_denied() -> None:
    assert decide(logs_proposal(row_limit=BUDGETS.log_rows + 1)) == (
        PolicyResult.DENIED,
        ReasonCode.RESULT_LIMIT_EXCEEDED,
    )
    assert decide(logs_proposal(row_limit=BUDGETS.log_rows))[0] is PolicyResult.ALLOWED


def test_a_passage_limit_above_the_budget_is_denied() -> None:
    """`SearchRunbooksArguments`'s own branch, mirroring
    `test_a_row_limit_above_the_budget_is_denied` above: allowed at exactly
    `budgets.runbook_passages`, denied one above it -- the boundary case
    that distinguishes `>` from `>=` in `policy.py`'s new branch, which a
    denial proposal built only from `budget + 1` cannot distinguish on its
    own (both operators deny `budget + 1`; only the exact-budget case tells
    them apart)."""
    assert decide(runbooks_proposal(limit=BUDGETS.runbook_passages + 1)) == (
        PolicyResult.DENIED,
        ReasonCode.RESULT_LIMIT_EXCEEDED,
    )
    assert (
        decide(runbooks_proposal(limit=BUDGETS.runbook_passages))[0]
        is PolicyResult.ALLOWED
    )


def test_a_search_runbooks_proposal_returns_before_the_service_fallthrough() -> None:
    """`SearchRunbooksArguments` has no `service` field -- `policy.authorize`
    falls through to `arguments.service` after its `GetTopologyArguments`
    branch, so a `search_runbooks` proposal that reached that line would
    raise `AttributeError` instead of returning a policy decision. This
    proves the new branch returns first: a proposal within budget is
    `ALLOWED` without ever raising, and the out-of-budget case above already
    proves the denial path returns too, rather than falling through and
    crashing on the way to a decision."""
    assert decide(runbooks_proposal())[0] is PolicyResult.ALLOWED


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


def test_a_row_limit_denial_states_the_requested_and_budget_numbers() -> None:
    """Fix F3. `PolicyDecision.message` reaches `events.jsonl`'s
    `proposal_denied.message` verbatim -- an owner reading that audit log
    needs to see the real numbers, not a bare "above the budget," without
    cross-referencing `Budgets` source. Asserted on the numbers themselves,
    not exact sentence wording, so a future rewording does not break this
    test for no reason."""
    over_limit = BUDGETS.log_rows + 7
    decision = authorize(
        logs_proposal(row_limit=over_limit), incident_scope(), set(), BUDGETS, 2
    )

    assert decision.reason_code is ReasonCode.RESULT_LIMIT_EXCEEDED
    assert str(over_limit) in decision.message
    assert str(BUDGETS.log_rows) in decision.message


def test_a_passage_limit_denial_states_the_requested_and_budget_numbers() -> None:
    """Fix F3, the `search_runbooks` sibling of the test above."""
    over_limit = BUDGETS.runbook_passages + 3
    decision = authorize(
        runbooks_proposal(limit=over_limit), incident_scope(), set(), BUDGETS, 2
    )

    assert decision.reason_code is ReasonCode.RESULT_LIMIT_EXCEEDED
    assert str(over_limit) in decision.message
    assert str(BUDGETS.runbook_passages) in decision.message


def test_a_corrected_row_limit_retry_is_allowed_and_the_original_stays_denied() -> None:
    """Fix F3's corrected-retry proof: proposing an over-limit `row_limit`
    is denied; proposing the corrected limit is `ALLOWED`; proposing the
    original over-limit value again afterward is still `DUPLICATE_
    PROPOSAL`, not re-evaluated against the budget -- proving F3's
    message-text fix left the fingerprint invariant `tool_wrappers.py`
    depends on (a denied proposal's fingerprint is marked seen too, so it
    cannot be silently retried) undisturbed."""
    over_limit = BUDGETS.log_rows + 5
    seen: set[str] = set()

    over = logs_proposal(row_limit=over_limit)
    first = authorize(over, incident_scope(), seen, BUDGETS, 2)
    assert first.result is PolicyResult.DENIED
    # Mirrors `tool_wrappers.ToolWrapper.dispatch`'s own rule: a
    # fingerprint is marked seen whether the decision allows or denies it.
    seen.add(first.fingerprint)

    corrected = logs_proposal(row_limit=BUDGETS.log_rows)
    second = authorize(corrected, incident_scope(), seen, BUDGETS, 2)
    assert second.result is PolicyResult.ALLOWED
    seen.add(second.fingerprint)

    third = authorize(over, incident_scope(), seen, BUDGETS, 2)
    assert third.result is PolicyResult.DENIED
    assert third.reason_code is ReasonCode.DUPLICATE_PROPOSAL


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
