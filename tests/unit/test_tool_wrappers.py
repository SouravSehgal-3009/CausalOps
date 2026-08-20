from pathlib import Path

import pytest
from fake_incident import (
    WINDOW_END,
    WINDOW_START,
    RecordingLogsBackend,
    StepClock,
    incident_scope,
    log_row,
    logs_proposal,
    write_log,
)

from causalops.domain import (
    Budgets,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    ToolOutcome,
    ToolProposal,
    ToolReceipt,
)
from causalops.telemetry import RunPaths, run_logs_check
from causalops.tool_wrappers import (
    ReceiptAlreadySettled,
    ReservationLedger,
    ToolWrapper,
    dispatch_registry,
    query_logs_wrapper,
)
from causalops.tools import (
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
    ToolName,
)


def out_of_scope_proposal() -> ToolProposal:
    return ToolProposal(
        arguments=QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY,
            service="billing",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            row_limit=20,
        ),
        evidence_gap="whether billing logged anything",
        expected_observation="nothing, billing is out of scope",
    )


def another_logs_proposal() -> ToolProposal:
    return ToolProposal(
        arguments=QueryLogsArguments(
            log_filter=LogFilter.POOL_EXHAUSTION,
            service="orders",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            row_limit=20,
        ),
        evidence_gap="whether orders exhausted a pool",
        expected_observation="pool exhaustion rows",
    )


def test_an_allowed_check_executes_and_settles_with_evidence() -> None:
    backend = RecordingLogsBackend()
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        logs_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.state is ReceiptState.SETTLED
    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.receipt.policy_result is PolicyResult.ALLOWED
    assert result.evidence is not None
    assert result.evidence.receipt_id == result.receipt.receipt_id
    assert backend.calls == [logs_proposal().arguments]


def test_a_denied_proposal_never_reaches_the_backend() -> None:
    backend = RecordingLogsBackend()
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        out_of_scope_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.policy_result is PolicyResult.DENIED
    assert result.receipt.reason_code is ReasonCode.UNKNOWN_SERVICE
    assert result.receipt.state is ReceiptState.SETTLED
    assert result.evidence is None
    assert backend.calls == []
    # A denial is recorded (so ledger.receipts() is the complete list, denials
    # included) but never spends a slot -- the same rule the legacy loop
    # already enforces for a denied proposal (test_workflow.py's
    # test_a_denied_proposal_costs_a_model_call_but_no_check_slot).
    assert ledger.receipts() == (result.receipt,)
    assert ledger.slots_left() == 2


def test_a_duplicate_fingerprint_is_denied_without_a_second_backend_call() -> None:
    backend = RecordingLogsBackend()
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)
    seen: set[str] = set()

    first = wrapper.dispatch(
        logs_proposal(), incident_scope(), seen, Budgets(), ledger, StepClock()
    )
    second = wrapper.dispatch(
        logs_proposal(), incident_scope(), seen, Budgets(), ledger, StepClock()
    )

    assert first.receipt.policy_result is PolicyResult.ALLOWED
    assert second.receipt.policy_result is PolicyResult.DENIED
    assert second.receipt.reason_code is ReasonCode.DUPLICATE_PROPOSAL
    assert len(backend.calls) == 1


def test_a_second_dispatch_past_budget_is_denied_by_authorize_before_reservation() -> (
    None
):
    backend = RecordingLogsBackend()
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=1)
    seen: set[str] = set()

    first = wrapper.dispatch(
        logs_proposal(), incident_scope(), seen, Budgets(), ledger, StepClock()
    )
    second = wrapper.dispatch(
        another_logs_proposal(), incident_scope(), seen, Budgets(), ledger, StepClock()
    )

    assert first.receipt.policy_result is PolicyResult.ALLOWED
    assert second.receipt.policy_result is PolicyResult.DENIED
    assert second.receipt.reason_code is ReasonCode.BUDGET_EXHAUSTED
    assert len(backend.calls) == 1


def test_slots_left_counts_only_allowed_receipts_not_denials() -> None:
    """`ledger.slots_left() == budget - len(ledger.receipts())` held under the
    old accounting but stopped being true in general once denials started
    being recorded too (a denial is in `receipts()` but must not spend a
    slot). This proves the real rule against an independently computed
    count, with a denial actually present, rather than a fixture that
    happens to have none -- which is exactly how the old, narrower version
    of this test kept passing after the rule it asserted stopped holding."""
    backend = RecordingLogsBackend()
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)
    seen: set[str] = set()

    wrapper.dispatch(
        logs_proposal(), incident_scope(), seen, Budgets(), ledger, StepClock()
    )
    wrapper.dispatch(
        another_logs_proposal(), incident_scope(), seen, Budgets(), ledger, StepClock()
    )
    wrapper.dispatch(
        out_of_scope_proposal(), incident_scope(), seen, Budgets(), ledger, StepClock()
    )

    allowed_receipt_count = sum(
        1 for r in ledger.receipts() if r.policy_result is PolicyResult.ALLOWED
    )

    assert len(ledger.receipts()) == 3
    assert allowed_receipt_count == 2
    assert ledger.slots_left() == 2 - allowed_receipt_count
    assert ledger.slots_left() == 0


def test_record_refuses_an_allowed_receipt() -> None:
    """The guard `record()` needs beside its other two: it exists to record a
    denial, which never spends a slot. An `ALLOWED` receipt must go through
    `reserve()`/`settle()` instead, or it would spend a slot that was never
    reserved -- the exact shape this unit exists to eliminate."""
    ledger = ReservationLedger(executed_tools_budget=2)
    allowed_receipt = ToolReceipt(
        receipt_id="receipt-allowed",
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        policy_result=PolicyResult.ALLOWED,
        outcome=ToolOutcome.EXECUTED,
        requested_at=WINDOW_START,
        duration_ms=5,
    )

    with pytest.raises(ValueError, match="not a denial"):
        ledger.record(allowed_receipt)

    assert ledger.slots_left() == 2


def test_a_receipt_settles_exactly_once() -> None:
    ledger = ReservationLedger(executed_tools_budget=1)
    reserved = ledger.reserve(
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        requested_at=WINDOW_START,
    )
    assert reserved is not None

    ledger.settle(
        receipt_id=reserved.receipt_id,
        outcome=ToolOutcome.EXECUTED,
        reason_code=None,
        duration_ms=5,
        result_digest="digest",
        evidence_id="evidence-1",
    )

    with pytest.raises(ReceiptAlreadySettled):
        ledger.settle(
            receipt_id=reserved.receipt_id,
            outcome=ToolOutcome.EXECUTED,
            reason_code=None,
            duration_ms=5,
            result_digest="digest",
            evidence_id="evidence-1",
        )


def test_settling_a_receipt_that_was_never_reserved_is_refused() -> None:
    ledger = ReservationLedger(executed_tools_budget=1)

    with pytest.raises(ReceiptAlreadySettled):
        ledger.settle(
            receipt_id="never-reserved",
            outcome=ToolOutcome.EXECUTED,
            reason_code=None,
            duration_ms=5,
            result_digest=None,
            evidence_id=None,
        )


def test_a_raising_backend_leaves_a_visible_reserved_receipt() -> None:
    backend = RecordingLogsBackend(raises=RuntimeError("lab unreachable"))
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    with pytest.raises(RuntimeError, match="lab unreachable"):
        wrapper.dispatch(
            logs_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
        )

    (only_receipt,) = ledger.receipts()
    assert only_receipt.state is ReceiptState.RESERVED
    assert only_receipt.outcome is None
    assert backend.calls == [logs_proposal().arguments]


def test_a_ledger_rebuilt_from_receipts_reports_the_same_slots_left() -> None:
    """`graph.py`'s dispatch node has no live ledger to hold between calls --
    only the receipt list graph state already carries. This is the property
    that makes rebuilding safe: a ledger built fresh from the same receipts
    a prior ledger produced must count the same remaining slots, or budget
    accounting would drift silently across a dispatch."""
    original = ReservationLedger(executed_tools_budget=2)
    wrapper = query_logs_wrapper(RecordingLogsBackend())
    seen: set[str] = set()
    wrapper.dispatch(
        logs_proposal(), incident_scope(), seen, Budgets(), original, StepClock()
    )
    wrapper.dispatch(
        another_logs_proposal(),
        incident_scope(),
        seen,
        Budgets(),
        original,
        StepClock(),
    )

    rebuilt = ReservationLedger.from_receipts(
        original.receipts(), executed_tools_budget=2
    )

    assert rebuilt.slots_left() == original.slots_left()
    assert rebuilt.receipts() == original.receipts()


def test_from_receipts_refuses_a_duplicate_receipt_id() -> None:
    """`record()` already raises on a duplicate `receipt_id`
    (`test_record_refuses_an_allowed_receipt` covers a different one of its
    guards); `from_receipts` is a second way to populate the same internal
    dict and needs the same protection, or a colliding `receipt_id` would
    silently keep only the last entry instead of surfacing the corruption."""
    receipt = ToolReceipt(
        receipt_id="receipt-1",
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        policy_result=PolicyResult.ALLOWED,
        state=ReceiptState.RESERVED,
        requested_at=WINDOW_START,
        duration_ms=0,
    )

    with pytest.raises(ValueError, match="duplicate receipt_id"):
        ReservationLedger.from_receipts([receipt, receipt], executed_tools_budget=2)


def test_a_receipt_round_trips_through_json_without_losing_fidelity() -> None:
    """Graph state stores receipts as `model_dump(mode="json")` dicts, not
    live `ToolReceipt` objects. If that round trip lost anything -- a
    timestamp's timezone, an enum's value -- budget accounting would corrupt
    silently, since `slots_left()` reads `policy_result` off the
    reconstructed object."""
    ledger = ReservationLedger(executed_tools_budget=1)
    reserved = ledger.reserve(
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        requested_at=WINDOW_START,
    )
    assert reserved is not None

    dumped = reserved.model_dump(mode="json")
    reloaded = ToolReceipt.model_validate(dumped)

    assert reloaded == reserved


def test_a_fully_populated_settled_receipt_round_trips_too() -> None:
    """The `RESERVED` case above is the least-populated shape a receipt can
    take -- `outcome`, `reason_code`, `result_digest`, and `evidence_id` are
    all still `None`. A `SETTLED` receipt with every optional field filled
    in is the harder case to prove lossless, and the more representative one
    for what Milestone 2's checkpoint resume actually has to survive."""
    ledger = ReservationLedger(executed_tools_budget=1)
    reserved = ledger.reserve(
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        requested_at=WINDOW_START,
    )
    assert reserved is not None
    settled = ledger.settle(
        receipt_id=reserved.receipt_id,
        outcome=ToolOutcome.EXECUTED,
        reason_code=None,
        duration_ms=12,
        result_digest="digest",
        evidence_id="evidence-1",
    )

    dumped = settled.model_dump(mode="json")
    reloaded = ToolReceipt.model_validate(dumped)

    assert reloaded == settled


def test_the_wrapper_refuses_arguments_for_a_different_tool() -> None:
    wrapper = query_logs_wrapper(RecordingLogsBackend())
    wrong_tool_proposal = ToolProposal(
        arguments=QueryMetricArguments(
            template=MetricTemplate.GATEWAY_ERROR_RATE,
            service="gateway",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        ),
        evidence_gap="gap",
        expected_observation="observation",
    )

    with pytest.raises(ValueError, match="query_logs wrapper received"):
        wrapper.dispatch(
            wrong_tool_proposal,
            incident_scope(),
            set(),
            Budgets(),
            ReservationLedger(executed_tools_budget=2),
            StepClock(),
        )


def test_the_registry_holds_only_wrapper_produced_entries() -> None:
    registry = dispatch_registry(RecordingLogsBackend())

    assert set(registry) == {ToolName.QUERY_LOGS}
    assert isinstance(registry[ToolName.QUERY_LOGS], ToolWrapper)
    assert registry[ToolName.QUERY_LOGS].tool is ToolName.QUERY_LOGS


def test_the_real_backend_executes_against_a_written_log_file(tmp_path: Path) -> None:
    """The one test that wires `run_logs_check` for real rather than the spy --
    with a `tmp_path` and no log written, it would return `UNAVAILABLE`, which
    would make every other wrapper test pass for the wrong reason."""
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [log_row(1, service="inventory", event="upstream_timeout")],
        service="inventory",
    )
    wrapper = query_logs_wrapper(lambda args: run_logs_check(args, paths))
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        logs_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    assert result.evidence.payload["row_count"] == 1
