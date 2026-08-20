from pathlib import Path

import pytest
from fake_incident import (
    FIXTURE_DIR,
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    RecordingLogsBackend,
    StepClock,
    alert_packet,
    assessment_json,
    incident_scope,
    logs_only_registry,
    logs_proposal,
    metric_proposal,
    packet_evidence,
    plan_json,
    replay_model,
    update_json,
)

import causalops.graph as graph_module
from causalops.domain import (
    Budgets,
    Disposition,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RootCauseCode,
    ToolProposal,
)
from causalops.graph import GRAPH_RECURSION_LIMIT, run_graph_investigation
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunRecorder
from causalops.tool_wrappers import ToolWrapper, query_logs_wrapper
from causalops.tools import LogFilter, QueryLogsArguments, ToolName
from causalops.workflow import InvestigationResult

GRAPH_FIXTURE = FIXTURE_DIR / "graph_single_check.json"


def graph_replay_model(fixture: Path = GRAPH_FIXTURE) -> ReplayToolCallingModel:
    """`graph_single_check.json` scripts `{{...}}` placeholders the same way
    `lab_diagnosis.json` does -- it is the fixture `cli.py` runs against a
    real, opaque incident -- so a test using it has to substitute exactly
    what a real run would, matching `incident_scope()`/`alert_packet()`."""
    substitutions = {
        "incident_id": incident_scope().incident_id,
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "symptom_evidence_id": SYMPTOM_EVIDENCE_ID,
    }
    return ReplayToolCallingModel(
        ReplayReasoningModel(fixture, substitutions=substitutions)
    )


def investigate_via_graph(
    model: ReplayToolCallingModel,
    registry: dict[ToolName, ToolWrapper] | None = None,
    budgets: Budgets | None = None,
    clock: StepClock | None = None,
) -> tuple[InvestigationResult, RunRecorder]:
    ticking = clock or StepClock()
    recorder = RunRecorder(ticking)
    resolved_registry = (
        registry if registry is not None else logs_only_registry(RecordingLogsBackend())
    )
    result = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        model,
        resolved_registry,
        recorder,
        budgets or Budgets(),
        ticking,
    )
    return result, recorder


def another_logs_proposal() -> ToolProposal:
    """A second `query_logs` proposal that fingerprints differently from
    `fake_incident.logs_proposal()`, so scripting both in one throwaway
    fixture does not trip the duplicate-proposal denial."""
    return ToolProposal(
        arguments=QueryLogsArguments(
            log_filter=LogFilter.POOL_EXHAUSTION,
            service="orders",
            window_start=incident_scope().started_at,
            window_end=incident_scope().ended_at,
            row_limit=20,
        ),
        evidence_gap="whether orders exhausted a pool",
        expected_observation="pool exhaustion rows",
    )


def test_a_scripted_diagnosis_runs_one_check_and_diagnoses() -> None:
    result, _ = investigate_via_graph(graph_replay_model())
    report = result.report

    assert report.disposition is Disposition.DIAGNOSED
    assert report.root_cause is RootCauseCode.CONFIG_CHANGE
    assert report.model_calls_used == 3
    assert report.tools_executed == 1
    assert len(result.receipts) == 1
    assert result.receipts[0].state is ReceiptState.SETTLED


def test_the_graph_loops_back_for_a_second_check_when_budget_allows(
    tmp_path: Path,
) -> None:
    """Reproduces `workflow.py:150`'s guard at the graph's `normalize_evidence`
    conditional edge: with the default budget (two executed checks, four
    model calls), a second `INVESTIGATE` turn is asked and its proposal is
    dispatched too."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "hypothesis_update": [plan_json(proposal=another_logs_proposal())],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))

    result, _ = investigate_via_graph(model)

    assert result.report.tools_executed == 2
    assert len(result.receipts) == 2
    assert [r.policy_result for r in result.receipts] == [
        PolicyResult.ALLOWED,
        PolicyResult.ALLOWED,
    ]
    assert model.requests[0].stage.value == "initial_plan"
    assert model.requests[1].stage.value == "hypothesis_update"
    assert model.requests[2].stage.value == "final_assessment"


def test_the_loop_guard_skips_a_second_turn_once_the_check_budget_is_spent(
    tmp_path: Path,
) -> None:
    """`tools_left() > 0` is false after one executed check when the budget
    only allows one, so `normalize_evidence` must route straight to
    `final_assessment` -- the graph never asks `HYPOTHESIS_UPDATE` at all."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    budgets = Budgets(executed_tools=1)

    result, _ = investigate_via_graph(model, budgets=budgets)

    assert result.report.tools_executed == 1
    assert [r.stage.value for r in model.requests] == [
        "initial_plan",
        "final_assessment",
    ]


def test_a_raising_backend_leaves_a_visible_reserved_receipt_in_the_graph_report() -> (
    None
):
    """The graph-level demonstration of the fix for the hazard the pre-edit
    report raised: a crash inside `dispatch_tool` must not lose the
    `RESERVED` receipt `ReservationLedger.reserve()` already wrote, the way
    it would if the crash were left to propagate out of `invoke()` instead
    of being caught inside the node. Mirrors
    `test_tool_wrappers.py::test_a_raising_backend_leaves_a_visible_reserved_receipt`
    at the orchestrator level."""
    backend = RecordingLogsBackend(raises=RuntimeError("lab unreachable"))
    registry = logs_only_registry(backend)

    result, recorder = investigate_via_graph(graph_replay_model(), registry=registry)

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.INTERNAL_ERROR
    assert result.report.assessment is None
    (only_receipt,) = result.receipts
    assert only_receipt.state is ReceiptState.RESERVED
    assert only_receipt.outcome is None
    # `graph_single_check.json` proposes `query_logs` on the `orders`
    # service -- the backend was reached exactly once, with that proposal's
    # arguments, before it raised.
    assert len(backend.calls) == 1
    assert backend.calls[0][0].service == "orders"
    names = [event.name for event in recorder.events]
    assert "backend_crashed" in names
    # The crash is a modeled transition (see `graph.py`'s `dispatch_tool`):
    # the graph still finishes through `normalize_evidence`, it does not
    # escape `invoke()` and land in `run_graph_investigation`'s outer
    # containment.
    assert "internal_error" not in names


def test_an_unwrapped_tool_proposal_is_refused_before_a_backend_is_reached(
    tmp_path: Path,
) -> None:
    """`dispatch_registry` always builds the full four-tool registry now, so
    a genuinely partial registry -- the shape this test needs -- has to be
    built by hand instead: a plain `dict[ToolName, ToolWrapper]` holding only
    `query_logs`, the same type `dispatch_registry` itself returns. The
    property this guards outlives the tool count: a proposal for a
    registered-but-unwrapped tool must not reach any backend --
    `dispatch_tool` looks the tool up inside the same `try` that calls the
    wrapper, so a missing entry is contained exactly like a backend crash.
    Milestone 2's `search_runbooks` will be exactly this shape again before
    it, too, gets a wrapper."""
    script = {
        "initial_plan": [plan_json(proposal=metric_proposal())],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    backend = RecordingLogsBackend()
    registry: dict[ToolName, ToolWrapper] = {
        ToolName.QUERY_LOGS: query_logs_wrapper(backend)
    }

    result, _ = investigate_via_graph(model, registry=registry)

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.INTERNAL_ERROR
    assert result.receipts == ()
    assert backend.calls == []


def test_a_broken_tool_call_round_trip_is_an_internal_error_not_a_model_mistake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`to_tool_call`/`parse_tool_call` are proven lossless in
    `test_tool_calls.py`; if that ever broke, `investigate()` raises
    `AssertionError` rather than silently reporting `MODEL_OUTPUT_INVALID`,
    which would misattribute an internal bug to the model. That
    `AssertionError` then reaches `run_graph_investigation`'s own outer
    containment the same way any other unmodeled node exception would --
    exactly how `workflow.py`'s `run_investigation` already turns an
    `AssertionError` deep in the loop into `internal_error()` rather than
    letting it crash the caller -- so the observable outcome here is a safe
    `FAILED_SAFE`/`INTERNAL_ERROR` report, not a raised exception."""
    monkeypatch.setattr(graph_module, "parse_tool_call", lambda call: None)

    result, recorder = investigate_via_graph(graph_replay_model())

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.INTERNAL_ERROR
    assert result.receipts == ()
    # The one model call that succeeded (`graph_single_check.json`'s
    # `initial_plan` response parses fine; only the round trip after it
    # fails) must still be counted -- `_StageCounters.record_call` runs
    # before the risky call it is counting, and this node's `try` covers
    # everything after it too, so a crash past the model call does not
    # erase the call that already happened.
    assert result.report.model_calls_used == 1
    names = [event.name for event in recorder.events]
    assert "internal_error" in names


def test_the_recursion_limit_is_far_below_the_library_default() -> None:
    """The library default is 10007 (`_internal/_config.py:32`); this graph's
    longest real path is eight supersteps. A regression that removed the
    explicit limit would let a looping bug spin for a very long time before
    failing instead of failing fast."""
    assert GRAPH_RECURSION_LIMIT < 100


def test_a_denied_proposal_never_reaches_the_backend_through_the_graph(
    tmp_path: Path,
) -> None:
    out_of_scope = ToolProposal(
        arguments=QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY,
            service="billing",
            window_start=incident_scope().started_at,
            window_end=incident_scope().ended_at,
            row_limit=20,
        ),
        evidence_gap="whether billing logged anything",
        expected_observation="nothing, billing is out of scope",
    )
    script = {
        "initial_plan": [plan_json(proposal=out_of_scope)],
        "hypothesis_update": [update_json(stop_reason="nothing safe left to check")],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    backend = RecordingLogsBackend()
    registry = logs_only_registry(backend)

    result, _ = investigate_via_graph(model, registry=registry)

    (only_receipt,) = result.receipts
    assert only_receipt.policy_result is PolicyResult.DENIED
    assert only_receipt.reason_code is ReasonCode.UNKNOWN_SERVICE
    assert backend.calls == []


def test_the_graph_does_not_ask_a_third_investigate_turn_after_a_denial(
    tmp_path: Path,
) -> None:
    """P1-1's regression test. `workflow.py`'s loop calls `plan_second_check()`
    at most once, from `run()`, regardless of whether the second proposal is
    allowed or denied -- there is no third ask, because `investigate()`'s own
    stage mapping has no third stage to ask (turn >= 1 always means
    `HYPOTHESIS_UPDATE`). A denial does not spend a slot
    (`ReservationLedger.slots_left()`), so a router bounded only by
    `tools_left()`/`model_calls_left()` would loop for a phantom third turn
    here, exhausting this fixture's single scripted `hypothesis_update`
    response. `model_turn < 2` in `route_after_normalize` is what prevents
    that."""
    repeated = logs_proposal()
    script = {
        "initial_plan": [plan_json(proposal=repeated)],
        "hypothesis_update": [plan_json(proposal=repeated)],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))

    result, _ = investigate_via_graph(model)

    assert [request.stage.value for request in model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert result.report.disposition is Disposition.DIAGNOSED
    first, second = result.receipts
    assert first.policy_result is PolicyResult.ALLOWED
    assert second.policy_result is PolicyResult.DENIED
    assert second.reason_code is ReasonCode.DUPLICATE_PROPOSAL


class _RaisingModel:
    """Raises on every `propose`/`respond` call -- the model-side analogue of
    `RecordingLogsBackend(raises=...)`, used to prove a spent model call
    survives the crash that immediately follows it."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.requests: list[object] = []

    def propose(self, request: object, schema: object) -> object:
        raise self.error

    def respond(self, request: object) -> object:
        raise self.error


def test_a_crashing_model_still_reports_the_spent_call() -> None:
    """P1-2's regression test. `investigate`'s `ask_once` calls
    `counters.record_call` *before* `model.propose`, so a raising model must
    still leave `model_calls_used == 1` in the final report, not 0 -- the
    call was spent even though it never returned. Before the fix, this
    node-local counter died with the crashed frame and the outer
    containment reported the pre-node checkpoint's `model_calls_used`
    (`0`) instead."""
    model = _RaisingModel(RuntimeError("provider timeout"))

    result, recorder = investigate_via_graph(model)  # type: ignore[arg-type]

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.INTERNAL_ERROR
    assert result.report.model_calls_used == 1
    names = [event.name for event in recorder.events]
    assert "internal_error" in names
