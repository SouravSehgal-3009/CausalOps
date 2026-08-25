from pathlib import Path

import pytest
from fake_incident import (
    INCIDENT_ID,
    WINDOW_END,
    WINDOW_START,
    RecordingChangesBackend,
    RecordingLogsBackend,
    RecordingMetricBackend,
    RecordingPrometheus,
    RecordingRunbooksBackend,
    RecordingTopologyBackend,
    StepClock,
    change_row,
    changes_proposal,
    incident_scope,
    log_row,
    logs_proposal,
    metric_proposal,
    runbooks_proposal,
    topology_proposal,
    write_changes,
    write_log,
    write_topology,
)

import causalops.tool_wrappers as tool_wrappers_module
from causalops.domain import (
    Budgets,
    CheckOutcome,
    EvidenceKind,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    ToolOutcome,
    ToolProposal,
    ToolReceipt,
)
from causalops.prometheus import run_metric_check
from causalops.telemetry import (
    RunPaths,
    run_changes_check,
    run_logs_check,
    run_topology_check,
)
from causalops.tool_wrappers import (
    ReceiptAlreadySettled,
    ReservationLedger,
    ToolWrapper,
    dispatch_registry,
    get_topology_wrapper,
    list_recent_changes_wrapper,
    query_logs_wrapper,
    query_metric_wrapper,
)
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
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
    assert backend.calls == [(logs_proposal().arguments, incident_scope())]


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
    # included) but never spends a slot -- the same rule the now-retired loop
    # already enforced for a denied proposal (test_graph.py's
    # test_a_denied_proposal_costs_a_model_call_but_no_check_slot, ported
    # from the loop's own test of the same name).
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
        arguments=logs_proposal().arguments,
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
    assert backend.calls == [(logs_proposal().arguments, incident_scope())]


def test_a_crash_after_settle_still_leaves_evidence_recoverable_from_the_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settle-then-crash window, targeted precisely: `ledger.settle()`
    inside `_make_wrapper`'s `dispatch()` already durably recorded the
    receipt's `evidence_id`/`result_digest` before this test's monkeypatch
    makes the very next statement -- constructing the `DispatchResult` that
    hands the `Evidence` object back to the caller -- raise instead. This is
    a narrower, later crash than
    `test_a_raising_backend_leaves_a_visible_reserved_receipt` above: that one
    crashes inside the backend, before `settle()` ever runs, and leaves a
    `RESERVED` receipt. This one crashes after `settle()` already succeeded,
    and the fix under test is that the `Evidence` object itself -- not just
    the digest/id already on the receipt -- survives in the ledger despite
    the crash."""
    original_init = tool_wrappers_module.DispatchResult.__init__

    def crashing_init(self: object, **data: object) -> None:
        if data.get("evidence") is not None:
            raise RuntimeError("crash after settle, before handoff")
        original_init(self, **data)  # type: ignore[misc]

    monkeypatch.setattr(tool_wrappers_module.DispatchResult, "__init__", crashing_init)
    backend = RecordingLogsBackend()
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    with pytest.raises(RuntimeError, match="crash after settle, before handoff"):
        wrapper.dispatch(
            logs_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
        )

    (only_receipt,) = ledger.receipts()
    assert only_receipt.state is ReceiptState.SETTLED
    assert only_receipt.evidence_id is not None
    (only_evidence,) = ledger.evidence()
    assert only_evidence.evidence_id == only_receipt.evidence_id
    assert only_evidence.content_hash == only_receipt.result_digest


def test_an_unavailable_outcome_settles_with_no_evidence_but_still_spends_a_slot() -> (
    None
):
    """`_make_wrapper.dispatch` builds evidence only when the backend reports
    `EXECUTED` (`tool_wrappers.py:374-393`) -- untested until now for any of
    the three non-`EXECUTED` outcomes a backend can report. The now-retired
    `workflow.py`'s loop already proved this for `UNAVAILABLE`
    (`test_workflow.py::test_an_unavailable_check_is_recorded_without_evidence`);
    this is the same property against the wrapper's independent
    implementation of the same rule."""
    outcome = CheckOutcome(
        outcome=ToolOutcome.UNAVAILABLE,
        kind=EvidenceKind.LOG,
        source="query_logs",
        summary="the log service did not respond",
        reason_code=ReasonCode.TOOL_UNAVAILABLE,
        duration_ms=8,
    )
    backend = RecordingLogsBackend(outcome=outcome)
    wrapper = query_logs_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        logs_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.UNAVAILABLE
    assert result.receipt.reason_code is ReasonCode.TOOL_UNAVAILABLE
    assert result.receipt.state is ReceiptState.SETTLED
    assert result.receipt.evidence_id is None
    assert result.receipt.result_digest is None
    assert result.evidence is None
    assert ledger.evidence() == ()
    # A slot is still spent: an attempt was made, whether or not it produced
    # evidence.
    assert ledger.slots_left() == 1


def test_a_timed_out_outcome_settles_with_no_evidence_but_still_spends_a_slot() -> None:
    """Same property as the `UNAVAILABLE` case above, for `TIMEOUT` --
    `test_workflow.py::test_a_check_that_times_out_still_spends_its_slot`'s
    loop-level counterpart."""
    outcome = CheckOutcome(
        outcome=ToolOutcome.TIMEOUT,
        kind=EvidenceKind.METRIC,
        source="query_metric",
        summary="the metric backend did not respond in time",
        reason_code=ReasonCode.TOOL_TIMEOUT,
        duration_ms=10000,
    )
    backend = RecordingMetricBackend(outcome=outcome)
    wrapper = query_metric_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        metric_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.TIMEOUT
    assert result.receipt.reason_code is ReasonCode.TOOL_TIMEOUT
    assert result.receipt.evidence_id is None
    assert result.evidence is None
    assert ledger.evidence() == ()
    assert ledger.slots_left() == 1


def test_an_error_outcome_settles_with_no_evidence_but_still_spends_a_slot() -> None:
    """Same property again, for `ERROR` --
    `test_workflow.py::test_a_failing_check_records_the_error_without_evidence`'s
    loop-level counterpart. Distinct from
    `test_a_raising_backend_leaves_a_visible_reserved_receipt` above: this
    backend does not raise, it returns a settled `ERROR` outcome, which is a
    different code path through `dispatch()` (past `run_check(...)`, into
    `settle()`) than a raising backend ever reaches."""
    outcome = CheckOutcome(
        outcome=ToolOutcome.ERROR,
        kind=EvidenceKind.CHANGE,
        source="list_recent_changes",
        summary="the change log could not be read",
        reason_code=ReasonCode.TOOL_ERROR,
        duration_ms=4,
    )
    backend = RecordingChangesBackend(outcome=outcome)
    wrapper = list_recent_changes_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        changes_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.ERROR
    assert result.receipt.reason_code is ReasonCode.TOOL_ERROR
    assert result.receipt.evidence_id is None
    assert result.evidence is None
    assert ledger.evidence() == ()
    assert ledger.slots_left() == 1


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
    reconstructed object. `arguments` (lab-defect-fix Unit 1) round-trips
    through the same `model_dump`/`model_validate` pair, via `ToolArguments`'s
    own `tool`-discriminated union, exactly like every other typed field
    here."""
    ledger = ReservationLedger(executed_tools_budget=1)
    reserved = ledger.reserve(
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        requested_at=WINDOW_START,
        arguments=logs_proposal().arguments,
    )
    assert reserved is not None

    dumped = reserved.model_dump(mode="json")
    reloaded = ToolReceipt.model_validate(dumped)

    assert reloaded == reserved
    assert reloaded.arguments == logs_proposal().arguments


def test_a_fully_populated_settled_receipt_round_trips_too() -> None:
    """The `RESERVED` case above is the least-populated shape a receipt can
    take -- `outcome`, `reason_code`, `result_digest`, and `evidence_id` are
    all still `None`. A `SETTLED` receipt with every optional field filled
    in is the harder case to prove lossless, and the more representative one
    for what Milestone 2's checkpoint resume actually has to survive.
    `arguments` set at reserve time must still be present after `settle()`
    replaces the receipt -- `settle()` copies it from the reserved receipt
    rather than re-deriving it, since it never sees the original proposal."""
    ledger = ReservationLedger(executed_tools_budget=1)
    reserved = ledger.reserve(
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        requested_at=WINDOW_START,
        arguments=logs_proposal().arguments,
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

    assert settled.arguments == logs_proposal().arguments

    dumped = settled.model_dump(mode="json")
    reloaded = ToolReceipt.model_validate(dumped)

    assert reloaded == settled


def test_a_receipt_dict_written_before_this_unit_still_validates() -> None:
    """Backward compatibility, lab-defect-fix Unit 1: a `receipts.jsonl` line
    or checkpoint dump written before this unit's `arguments` field existed
    has no `arguments` key at all -- `_rebuild_receipts` in `graph.py` runs
    `ToolReceipt.model_validate` over every persisted receipt on every node
    call, so a dict missing this key must still validate, with `arguments`
    defaulting to `None`, or every pre-existing checkpoint would fail to
    resume. `None` means "written before this unit," never "ran with no
    arguments" -- see `ToolReceipt.arguments`'s own docstring."""
    pre_unit_dump = {
        "receipt_id": "receipt-pre-unit-1",
        "incident_id": incident_scope().incident_id,
        "tool": ToolName.QUERY_LOGS.value,
        "fingerprint": "f" * 8,
        "policy_result": PolicyResult.ALLOWED.value,
        "state": ReceiptState.SETTLED.value,
        "outcome": ToolOutcome.EXECUTED.value,
        "requested_at": WINDOW_START.isoformat(),
        "duration_ms": 5,
        # No "arguments" key -- exactly what a pre-Unit-1 dump looks like.
    }

    reloaded = ToolReceipt.model_validate(pre_unit_dump)

    assert reloaded.arguments is None


def test_every_fresh_receipt_construction_site_sets_arguments() -> None:
    """The forward guarantee lab-defect-fix Unit 1 pins with a test rather
    than a validator (per the plan's own reasoning: a `schema_version`-tied
    `model_validator` would encode a migration policy this project does not
    otherwise have, for a field nothing reads yet): every receipt built
    *fresh* by this module -- `ledger.reserve`, `ledger.settle` (which
    carries the reserved receipt's `arguments` forward), and
    `_denied_receipt` -- always sets `arguments` to a real value, never
    `None`. Only a receipt round-tripped from a pre-Unit-1 artifact should
    ever carry `None`, which the test above covers separately."""
    ledger = ReservationLedger(executed_tools_budget=2)
    reserved = ledger.reserve(
        incident_id=incident_scope().incident_id,
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        requested_at=WINDOW_START,
        arguments=logs_proposal().arguments,
    )
    assert reserved is not None
    assert reserved.arguments is not None

    settled = ledger.settle(
        receipt_id=reserved.receipt_id,
        outcome=ToolOutcome.EXECUTED,
        reason_code=None,
        duration_ms=5,
        result_digest="digest",
        evidence_id="evidence-1",
    )
    assert settled.arguments is not None

    wrapper = query_logs_wrapper(RecordingLogsBackend())
    denied = wrapper.dispatch(
        out_of_scope_proposal(),
        incident_scope(),
        set(),
        Budgets(),
        ReservationLedger(executed_tools_budget=2),
        StepClock(),
    )
    assert denied.receipt.policy_result is PolicyResult.DENIED
    assert denied.receipt.arguments is not None


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

    with pytest.raises(
        ValueError, match="query_logs wrapper expects QueryLogsArguments, got Query"
    ):
        wrapper.dispatch(
            wrong_tool_proposal,
            incident_scope(),
            set(),
            Budgets(),
            ReservationLedger(executed_tools_budget=2),
            StepClock(),
        )


def test_the_registry_holds_only_wrapper_produced_entries() -> None:
    """All five tools now, not just `query_logs` -- `dispatch_registry` always
    builds the full registry as of Unit 3a."""
    registry = dispatch_registry(
        run_metric=RecordingMetricBackend(),
        run_logs=RecordingLogsBackend(),
        run_changes=RecordingChangesBackend(),
        run_topology=RecordingTopologyBackend(),
        run_search=RecordingRunbooksBackend(),
    )

    assert set(registry) == set(ToolName)
    for tool in ToolName:
        assert isinstance(registry[tool], ToolWrapper)
        assert registry[tool].tool is tool


def test_only_search_runbooks_ever_yields_passages_and_never_evidence() -> None:
    """The registry-level proof that `_make_wrapper`'s branch is tied to
    tool identity, not just result type in the abstract: dispatch one
    allowed, executing proposal through every one of the five registered
    wrappers and check `DispatchResult.passages`/`.evidence` per tool.
    `_make_wrapper`'s own docstring already concedes `mypy` cannot bind
    `tool` to `arguments_type` -- this is the runtime check that closes
    that gap empirically, the same role the wrapper-identity test plays for
    construction and the spy-backend test plays for denial."""
    registry = dispatch_registry(
        run_metric=RecordingMetricBackend(),
        run_logs=RecordingLogsBackend(),
        run_changes=RecordingChangesBackend(),
        run_topology=RecordingTopologyBackend(),
        run_search=RecordingRunbooksBackend(),
    )
    proposal_by_tool = {
        ToolName.QUERY_METRIC: metric_proposal(),
        ToolName.QUERY_LOGS: logs_proposal(),
        ToolName.LIST_RECENT_CHANGES: changes_proposal(),
        ToolName.GET_TOPOLOGY: topology_proposal(),
        ToolName.SEARCH_RUNBOOKS: runbooks_proposal(),
    }
    assert set(registry) == set(proposal_by_tool)

    for tool, wrapper in registry.items():
        result = wrapper.dispatch(
            proposal_by_tool[tool],
            incident_scope(),
            set(),
            Budgets(),
            ReservationLedger(executed_tools_budget=2),
            StepClock(),
        )
        assert result.receipt.policy_result is PolicyResult.ALLOWED
        if tool is ToolName.SEARCH_RUNBOOKS:
            assert result.passages != ()
            assert result.evidence is None
        else:
            assert result.passages == ()
            assert result.evidence is not None


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
    wrapper = query_logs_wrapper(lambda args, scope: run_logs_check(args, paths))
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        logs_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    assert result.evidence.payload["row_count"] == 1


# -- Unit 1c: the three tools query_logs's wrapper proved the shape against --
#
# One allowed-executes test and one denied-untouched test per tool, mirroring
# the query_logs coverage above at a lighter weight: the dispatch body
# (authorize -> reserve -> dispatch -> settle) is the same generic factory for
# all four, already proven exhaustively against query_logs, so what is new
# per tool here is only "does this tool's own wrapper reach this tool's own
# backend with this tool's own arguments" -- plus one real-backend test per
# tool, for the same reason `test_the_real_backend_executes_against_a_written_log_file`
# exists: a spy that is never wired to the real backend would make every
# other test in this section pass for the wrong reason.


def out_of_scope_metric_proposal() -> ToolProposal:
    return ToolProposal(
        arguments=QueryMetricArguments(
            template=MetricTemplate.GATEWAY_ERROR_RATE,
            service="billing",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        ),
        evidence_gap="whether billing's error rate moved",
        expected_observation="nothing, billing is out of scope",
    )


def test_query_metric_wrapper_executes_and_settles_with_evidence() -> None:
    backend = RecordingMetricBackend()
    wrapper = query_metric_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        metric_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    assert backend.calls == [(metric_proposal().arguments, incident_scope())]


def test_query_metric_wrapper_denies_an_out_of_scope_proposal_untouched() -> None:
    backend = RecordingMetricBackend()
    wrapper = query_metric_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        out_of_scope_metric_proposal(),
        incident_scope(),
        set(),
        Budgets(),
        ledger,
        StepClock(),
    )

    assert result.receipt.policy_result is PolicyResult.DENIED
    assert result.receipt.reason_code is ReasonCode.UNKNOWN_SERVICE
    assert backend.calls == []


def test_the_real_metric_backend_executes_against_a_loopback_prometheus(
    fake_prometheus: RecordingPrometheus,
) -> None:
    """`run_metric_check` needs the `IncidentScope` for the PromQL `incident`
    label -- the one backend seam that actually reads the scope argument
    every wrapper is now given."""
    wrapper = query_metric_wrapper(
        lambda args, scope: run_metric_check(args, scope, fake_prometheus.url, 5)
    )
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        metric_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    # The wrapper forwarded `dispatch()`'s own `incident_scope()` through to
    # `run_metric_check`, not some other scope: the PromQL the fake server
    # actually received names this dispatch's incident in the `incident`
    # label -- the cross-incident isolation this backend seam exists for.
    assert f'incident="{incident_scope().incident_id}"' in fake_prometheus.queries[-1]


def out_of_scope_changes_proposal() -> ToolProposal:
    return ToolProposal(
        arguments=ListRecentChangesArguments(
            service="billing", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        evidence_gap="whether billing changed recently",
        expected_observation="nothing, billing is out of scope",
    )


def test_list_recent_changes_wrapper_executes_and_settles_with_evidence() -> None:
    backend = RecordingChangesBackend()
    wrapper = list_recent_changes_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        changes_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    assert backend.calls == [(changes_proposal().arguments, incident_scope())]


def test_list_recent_changes_wrapper_denies_an_out_of_scope_proposal_untouched() -> (
    None
):
    backend = RecordingChangesBackend()
    wrapper = list_recent_changes_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        out_of_scope_changes_proposal(),
        incident_scope(),
        set(),
        Budgets(),
        ledger,
        StepClock(),
    )

    assert result.receipt.policy_result is PolicyResult.DENIED
    assert result.receipt.reason_code is ReasonCode.UNKNOWN_SERVICE
    assert backend.calls == []


def test_the_real_changes_backend_executes_against_a_written_changes_file(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    write_changes(paths, [change_row(1)])
    wrapper = list_recent_changes_wrapper(
        lambda args, scope: run_changes_check(args, paths)
    )
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        changes_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    assert result.evidence.payload["change_count"] == 1


def out_of_scope_topology_proposal() -> ToolProposal:
    """`get_topology`'s only field beyond `tool` is `incident_id` -- its one
    refusable shape is a cross-incident id (`policy.authorize`'s
    `CROSS_INCIDENT_REQUEST` branch), not a service or window."""
    return ToolProposal(
        arguments=GetTopologyArguments(incident_id="a" * len(INCIDENT_ID)),
        evidence_gap="another incident's topology",
        expected_observation="nothing, that incident is out of scope",
    )


def test_get_topology_wrapper_executes_and_settles_with_evidence() -> None:
    backend = RecordingTopologyBackend()
    wrapper = get_topology_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        topology_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    assert backend.calls == [(topology_proposal().arguments, incident_scope())]


def test_get_topology_wrapper_denies_a_cross_incident_proposal_untouched() -> None:
    backend = RecordingTopologyBackend()
    wrapper = get_topology_wrapper(backend)
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        out_of_scope_topology_proposal(),
        incident_scope(),
        set(),
        Budgets(),
        ledger,
        StepClock(),
    )

    assert result.receipt.policy_result is PolicyResult.DENIED
    assert result.receipt.reason_code is ReasonCode.CROSS_INCIDENT_REQUEST
    assert backend.calls == []


def test_the_real_topology_backend_executes_against_a_written_manifest(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    write_topology(paths, ["gateway>orders"])
    wrapper = get_topology_wrapper(lambda args, scope: run_topology_check(args, paths))
    ledger = ReservationLedger(executed_tools_budget=2)

    result = wrapper.dispatch(
        topology_proposal(), incident_scope(), set(), Budgets(), ledger, StepClock()
    )

    assert result.receipt.outcome is ToolOutcome.EXECUTED
    assert result.evidence is not None
    assert result.evidence.payload["edge_count"] == 1
