from pathlib import Path
from typing import Any

import pytest
from fake_incident import (
    FIXTURE_DIR,
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    RecordingLogsBackend,
    RecordingMetricBackend,
    StepClock,
    UsageReportingModel,
    alert_packet,
    assessment_json,
    incident_scope,
    logs_only_registry,
    logs_proposal,
    metric_proposal,
    packet_evidence,
    plan_json,
    registry_with,
    replay_model,
    resume_graph_run,
    update_json,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

import causalops.graph as graph_module
import causalops.tool_wrappers as tool_wrappers_module
from causalops.domain import (
    Budgets,
    CheckOutcome,
    Disposition,
    EscalatedInvestigation,
    EscalationReason,
    EvidenceKind,
    FinalAssessment,
    GraphPhase,
    Hypothesis,
    InitialPlan,
    InvestigationResult,
    ModelDisposition,
    ModelUsage,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RootCauseCode,
    ToolOutcome,
    ToolProposal,
)
from causalops.evidence import build_evidence
from causalops.graph import GRAPH_RECURSION_LIMIT, build_graph, run_graph_investigation
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunRecorder
from causalops.tool_wrappers import ToolWrapper, query_logs_wrapper
from causalops.tools import LogFilter, QueryLogsArguments, ToolName

GRAPH_FIXTURE = FIXTURE_DIR / "graph_single_check.json"
OTHER_INCIDENT_ID = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
FORGED_HYPOTHESIS_EVIDENCE_ID = "forged-hypothesis-evidence-id"


def fixture_model(name: str) -> ReplayToolCallingModel:
    """`workflow.py`'s own `fixture_model()` helper, ported: the checked-in
    fixtures under `FIXTURE_DIR` are stage-response scripts with no `{{...}}`
    placeholders, orchestrator-independent, so the same file works for both
    the loop and the graph -- only the wrapping model type differs."""
    return ReplayToolCallingModel(ReplayReasoningModel(FIXTURE_DIR / name))


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
) -> tuple[InvestigationResult | EscalatedInvestigation, RunRecorder]:
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


def test_the_run_records_its_states_in_order() -> None:
    """`test_workflow.py::test_the_run_records_its_states_in_order`, ported.

    `RunRecorder`'s own sequence numbering is already proven
    orchestrator-independently (`test_run_records.py`), so this pins two
    things specific to the graph: its own event vocabulary --
    "investigation_started" first, "check_finished" present for the one
    executed check -- and that a recorded event carries a `GraphPhase` tag at
    all. The loop original asserted its own state name
    (`EXECUTE_SECOND_CHECK`), which has no graph equivalent; the property
    that survives the port is "some event names a phase", checked here
    against `DISPATCH_TOOL`, the phase the one executed check runs under.
    `events.jsonl` is an artifact the completion definition has the owner
    inspect directly -- this is the only test proving it carries a phase tag
    at all.
    """
    _, recorder = investigate_via_graph(graph_replay_model())

    names = [event.name for event in recorder.events]
    assert names[0] == "investigation_started"
    assert "check_finished" in names
    assert [event.sequence for event in recorder.events] == list(
        range(1, len(recorder.events) + 1)
    )
    assert GraphPhase.DISPATCH_TOOL.value in {event.state for event in recorder.events}


def test_the_graph_loops_back_for_a_second_check_when_budget_allows(
    tmp_path: Path,
) -> None:
    """Reproduces the retired loop's own guard at the graph's
    `normalize_evidence` conditional edge: with the default budget (two
    executed checks, four model calls), a second `INVESTIGATE` turn is asked
    and its proposal is dispatched too."""
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

    # `test_workflow.py::test_an_unexpected_failure_becomes_a_terminal_state`,
    # ported: only the exception's class name may reach the event log or the
    # report, never its message text -- `graph.py`'s `dispatch_tool` except
    # handler records `error=type(error).__name__` only, the same redaction
    # rule the now-retired `workflow.py`'s `internal_error()` already
    # enforced.
    recorded = "".join(event.model_dump_json() for event in recorder.events)
    assert "RuntimeError" in recorded
    assert "lab unreachable" not in recorded
    assert "lab unreachable" not in result.report.model_dump_json()


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
    exactly how the now-retired `workflow.py`'s own `run_investigation` (a
    different function from `cli.py`'s dispatcher of the same name today)
    already turned an `AssertionError` deep in the loop into
    `internal_error()` rather than letting it crash the caller -- so the
    observable outcome here is a safe `FAILED_SAFE`/`INTERNAL_ERROR` report,
    not a raised exception."""
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
    """P1-1's regression test. The now-retired `workflow.py`'s loop called
    `plan_second_check()` at most once, from `run()`, regardless of whether
    the second proposal was allowed or denied -- there was no third ask,
    because `investigate()`'s own stage mapping has no third stage to ask
    (turn >= 1 always means `HYPOTHESIS_UPDATE`). A denial does not spend a
    slot (`ReservationLedger.slots_left()`), so a router bounded only by
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

    # Same redaction rule as the dispatch-crash path above
    # (`test_a_raising_backend_leaves_a_visible_reserved_receipt_in_the_graph_report`):
    # only the exception's class name may reach the event log or the report.
    recorded = "".join(event.model_dump_json() for event in recorder.events)
    assert "RuntimeError" in recorded
    assert "provider timeout" not in recorded
    assert "provider timeout" not in result.report.model_dump_json()


def test_a_graphbubbleup_escape_still_syncs_the_callers_recorder() -> None:
    """`GraphBubbleUp` -- `GraphInterrupt` (raised by `interrupt()`),
    `GraphDrained`, `ParentCommand` -- is a control-flow signal, not a
    failure, so `run_graph_investigation` re-raises it rather than
    converting it to a safe report. But a `raise` skips the function's
    normal return, which is the only place the caller's `recorder` used to
    get synced from state. Before this fix, the caller's `recorder` was left
    holding nothing at all: the seed event goes to `seed_recorder`, not
    `recorder`, and every node's own events live only in state.

    This branch is **not** where Milestone 2b's pause lands, and this test
    must not be read as proving it is -- confirmed twice, independently
    (once while planning Milestone 2, once by this unit's correctness
    review), against the installed LangGraph: a real `interrupt()` call does
    not make `.invoke()` raise. The Pregel loop catches `GraphInterrupt`
    itself and `.invoke()` returns *normally*, with an `__interrupt__` key
    in the output, so 2b's pause handling has to be written against that
    normal return, not this `except`. What this branch actually guards is a
    `GraphBubbleUp` that genuinely escapes `.invoke()` uncaught -- this
    codebase does not build anything that raises one today, so it is
    presently unreached in production, but the three `except GraphBubbleUp:
    raise` guards inside `investigate`/`dispatch_tool`/`final_assessment`
    (one per node, not counting this function's own outer handler below,
    which is the branch this test exercises) exist so that if one ever
    does, it is not swallowed by those nodes' own blanket
    `except Exception:` and misreported as a crash.

    A raw `GraphBubbleUp`, raised directly from a node below rather than via
    `interrupt()`, is a faithful stand-in for that escape: confirmed against
    the installed LangGraph to actually propagate out of `.invoke()` (unlike
    a real `interrupt()`, which does not), so it is enough to prove the
    sync-before-`raise` fix without needing a scenario that makes this
    branch reachable in production to exist yet."""
    model = _RaisingModel(GraphBubbleUp("interrupt-like signal"))
    registry = logs_only_registry(RecordingLogsBackend())
    recorder = RunRecorder(StepClock())

    with pytest.raises(GraphBubbleUp):
        run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            model,  # type: ignore[arg-type]
            registry,
            recorder,
            Budgets(),
            StepClock(),
        )

    # The crash happens on turn 0's first model call, before `investigate`
    # ever returns, so the only committed checkpoint is the initial state
    # `run_graph_investigation` seeds before `.invoke()` -- one event. The
    # point is not the count, it is that this is no longer empty.
    names = [event.name for event in recorder.events]
    assert names == ["investigation_started"]


def test_a_checkpoint_read_failure_does_not_replace_the_graphbubbleup_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The event-recovery read the test above exercises is I/O against the
    same database `checkpointer` uses, so it can itself fail --
    `compiled.get_state(config)` can raise `sqlite3.OperationalError` against
    a locked, full, or corrupt database, or `KeyError`/`ValidationError`
    against a malformed or partial checkpoint. Any of those replacing the
    original `GraphBubbleUp` would turn a pause or a parent command into an
    unhandled database error `main` cannot format into `FAIL <CODE>
    <message>` -- confirmed missing by a reviewer's mutation that narrowed
    the guard to `except ZeroDivisionError:` and found the exact same 316
    tests green either way, since nothing before this test could see the
    difference.

    `caught.value.__context__ is None` is the assertion that actually proves
    it, not merely `pytest.raises(GraphBubbleUp)` matching: Python chains a
    new exception raised while another is being handled onto that new
    exception's `__context__` automatically. If the read's `OSError` reached
    the top of this `except GraphBubbleUp:` block unswallowed, it would be
    *that* exception propagating, chained onto the `GraphBubbleUp` being
    handled -- and `pytest.raises(GraphBubbleUp)` would not match an
    `OSError` at all, so this test would fail two different ways depending
    on exactly how the guard broke, not just one."""
    model = _RaisingModel(GraphBubbleUp("interrupt-like signal"))
    registry = logs_only_registry(RecordingLogsBackend())
    recorder = RunRecorder(StepClock())

    def broken_get_state(self: object, config: object, **kwargs: object) -> object:
        raise OSError("simulated: database is locked")

    monkeypatch.setattr(CompiledStateGraph, "get_state", broken_get_state)

    with pytest.raises(GraphBubbleUp) as caught:
        run_graph_investigation(
            incident_scope(),
            alert_packet(),
            packet_evidence(),
            model,  # type: ignore[arg-type]
            registry,
            recorder,
            Budgets(),
            StepClock(),
        )

    assert caught.value.__context__ is None


# --- Unit 1d-1/1d-2: behaviours `graph.py` already implements but that, until
# now, only `test_workflow.py` proved. Each port below asserts the same
# *property* its loop original does, not a copied-over literal, since the two
# orchestrators' numbers can legitimately differ (see `_build_report`'s two
# documented differences from `Investigation.report()`). Every fixture reused
# here (`correct_abstention.json`, `repair_then_valid.json`,
# `malformed_output.json`, `forged_citation.json`, `valid_diagnosis.json`,
# `service_out_of_scope.json`, `duplicate_proposal.json`) is
# orchestrator-independent: a stage-response script with no `{{...}}`
# placeholders, so the exact same checked-in file `test_workflow.py` used
# works for the graph too. `workflow.py` and `test_workflow.py` were deleted
# in Unit 1d-2, once every behaviour they alone proved had a port here.


def test_a_denied_proposal_costs_a_model_call_but_no_check_slot() -> None:
    """`test_workflow.py::test_a_denied_proposal_costs_a_model_call_but_no_check_slot`,
    ported: `service_out_of_scope.json` proposes a `query_metric` check
    against `billing`, which `incident_scope()` does not name, so the wrapper
    denies it before any backend call and the model has no safe check left to
    propose -- an abstention that still spent the model call the denied
    proposal used.

    Unit 2b: an abstention with an unspent check slot (nothing was executed,
    so the full budget remains) is exactly `INSUFFICIENT_EVIDENCE_WITH_
    CHECK_REMAINING`, so this scenario now pauses rather than reaching
    `final_report` directly. The receipt/backend assertions below are
    unaffected -- `dispatch_tool` already settled that receipt before
    `final_assessment`, let alone `escalation_interrupt`, ever ran."""
    backend = RecordingMetricBackend()
    registry = registry_with(run_metric=backend)

    result, _ = investigate_via_graph(
        fixture_model("service_out_of_scope.json"), registry=registry
    )

    receipt = result.receipts[0]
    assert receipt.policy_result is PolicyResult.DENIED
    assert receipt.reason_code is ReasonCode.UNKNOWN_SERVICE
    assert receipt.outcome is ToolOutcome.NOT_EXECUTED
    assert backend.calls == []
    assert len(result.evidence) == 2
    # A denial is not terminal: the run reached a valid abstention, with a
    # check slot still open, and paused for the owner rather than finalizing.
    assert isinstance(result, EscalatedInvestigation)
    assert result.reason is EscalationReason.INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING
    assert result.remaining_check_count > 0


def test_the_same_proposal_twice_is_denied_as_a_duplicate() -> None:
    """`test_workflow.py::test_the_same_proposal_twice_is_denied_as_a_duplicate`,
    ported: `duplicate_proposal.json` scripts the identical `query_metric`
    proposal for both `initial_plan` and `hypothesis_update`; the second
    fingerprints the same as the first and is denied without reaching the
    backend a second time."""
    backend = RecordingMetricBackend()
    registry = registry_with(run_metric=backend)

    result, _ = investigate_via_graph(
        fixture_model("duplicate_proposal.json"), registry=registry
    )

    assert result.receipts[0].policy_result is PolicyResult.ALLOWED
    assert result.receipts[1].reason_code is ReasonCode.DUPLICATE_PROPOSAL
    assert result.report.tools_executed == 1
    assert len(backend.calls) == 1


def test_a_scripted_abstention_stops_early_and_abstains() -> None:
    """`test_workflow.py::test_a_scripted_abstention_stops_early_and_abstains`,
    ported: `correct_abstention.json` proposes one `query_metric` check, then
    stops and abstains rather than diagnosing.

    Unit 2b: one executed check out of a two-check budget still leaves a
    slot open, so this abstention is also `INSUFFICIENT_EVIDENCE_WITH_
    CHECK_REMAINING` and now pauses instead of finalizing -- see the sibling
    test above for the same change on a zero-executed-checks abstention."""
    registry = registry_with(run_metric=RecordingMetricBackend())

    result, _ = investigate_via_graph(
        fixture_model("correct_abstention.json"), registry=registry
    )

    assert len(result.receipts) == 1
    assert result.receipts[0].outcome is ToolOutcome.EXECUTED
    assert isinstance(result, EscalatedInvestigation)
    assert result.reason is EscalationReason.INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING
    assert result.remaining_check_count == 1


def test_one_invalid_response_is_repaired_and_the_run_continues() -> None:
    """`test_workflow.py::test_one_invalid_response_is_repaired_and_the_run_continues`,
    ported: `repair_then_valid.json` scripts an invalid `initial_plan`
    response first, then a valid one."""
    registry = registry_with(run_metric=RecordingMetricBackend())

    result, _ = investigate_via_graph(
        fixture_model("repair_then_valid.json"), registry=registry
    )
    report = result.report

    assert report.disposition is Disposition.DIAGNOSED
    assert report.repairs_used == 1
    assert report.invalid_responses == 1
    # The repair spends an ordinary model call: plan, repair, update, assessment.
    assert report.model_calls_used == 4


def test_a_repair_tells_the_model_what_was_wrong() -> None:
    """`test_workflow.py::test_a_repair_tells_the_model_what_was_wrong`, ported."""
    model = fixture_model("repair_then_valid.json")
    registry = registry_with(run_metric=RecordingMetricBackend())

    investigate_via_graph(model, registry=registry)

    first, repair = model.requests[0], model.requests[1]
    assert first.repair_errors is None
    assert repair.stage is first.stage
    assert repair.repair_errors and "hypotheses" in repair.repair_errors


def test_a_run_with_no_repair_budget_stops_at_the_first_invalid_response() -> None:
    """`test_workflow.py::test_a_run_with_no_repair_budget_stops_at_the_first_invalid_response`,
    ported. Also covers Unit 2a's `investigate`'s `stopped_state` -- its
    return is the one that carries `stage_stopped`/`invalid_response` into
    state, and no other test in this file asserts on events recorded along
    that specific return path, so a missing `"events"` key there would not
    otherwise be noticed."""
    result, recorder = investigate_via_graph(
        fixture_model("malformed_output.json"), budgets=Budgets(repairs=0)
    )

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.REPAIR_EXHAUSTED
    assert result.report.repairs_used == 0
    assert result.report.model_calls_used == 1
    names = [event.name for event in recorder.events]
    assert "invalid_response" in names
    assert "stage_stopped" in names


def test_a_response_that_stays_invalid_fails_safe() -> None:
    """`test_workflow.py::test_a_response_that_stays_invalid_fails_safe`, ported."""
    result, _ = investigate_via_graph(fixture_model("malformed_output.json"))
    report = result.report

    assert report.disposition is Disposition.FAILED_SAFE
    assert report.root_cause is RootCauseCode.UNDETERMINED
    assert report.reason_code is ReasonCode.MODEL_OUTPUT_INVALID
    assert report.assessment is None
    assert report.invalid_responses == 2
    assert report.tools_executed == 0


def test_a_naive_window_from_the_model_still_produces_a_report(tmp_path: Path) -> None:
    """`test_workflow.py::test_a_naive_window_from_the_model_still_produces_a_report`,
    ported.

    Correction to this port's own original justification: I had argued the
    graph reaches this failure through the `to_tool_call`/`parse_tool_call`
    round trip, a different layer than the loop's direct schema validation.
    That is wrong -- `ReplayToolCallingModel.propose()` calls
    `parse_response(schema, ...)` *before* it ever encodes a tool call
    (`models.py`'s `propose()`: `if parsed is None: return ...` precedes the
    `to_tool_call` line entirely). A naive `window_start` fails `UtcDatetime`
    validation inside `InitialPlan.proposal.arguments` at that same
    `parse_response` step, identically to the loop, so `parsed` is `None`
    before the tool-call round trip is ever reached: this is the same code
    path as `test_a_response_that_stays_invalid_fails_safe` above, not a
    distinct one. Kept as its own test only to pin the specific
    naive-datetime trigger, not because it proves anything that test does
    not."""
    naive = plan_json(metric_proposal())
    naive["proposal"]["arguments"]["window_start"] = "2026-08-16T10:00:00"
    model = ReplayToolCallingModel(
        replay_model(tmp_path, {"initial_plan": [naive, naive]})
    )

    result, _ = investigate_via_graph(model)

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.MODEL_OUTPUT_INVALID
    assert result.report.invalid_responses == 2


def test_a_cited_evidence_id_that_does_not_exist_fails_safe() -> None:
    """`test_workflow.py::test_a_cited_evidence_id_that_does_not_exist_fails_safe`,
    ported. Also covers Unit 2a's `final_assessment`'s `failed_state` -- its
    return is the one that carries `forged_citation` into state, and no
    other test in this file asserts on events recorded along that specific
    return path, so a missing `"events"` key there would not otherwise be
    noticed."""
    result, recorder = investigate_via_graph(fixture_model("forged_citation.json"))

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.FORGED_EVIDENCE_REFERENCE
    names = [event.name for event in recorder.events]
    assert "forged_citation" in names


def test_citing_another_incidents_real_evidence_id_fails_safe(tmp_path: Path) -> None:
    """`test_workflow.py::test_citing_another_incidents_real_evidence_id_fails_safe`,
    ported."""
    other_evidence = build_evidence(
        incident_id=OTHER_INCIDENT_ID,
        kind=EvidenceKind.METRIC,
        source="query_metric",
        observed_at=WINDOW_START,
        summary="a real observation, but recorded against a different incident",
        payload={"p95_ms": 500},
    )
    model = ReplayToolCallingModel(
        replay_model(
            tmp_path,
            {
                "initial_plan": [plan_json(stop_reason="the alert is enough")],
                "final_assessment": [
                    FinalAssessment(
                        disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
                        root_cause=RootCauseCode.UNDETERMINED,
                        contrary_evidence_ids=(other_evidence.evidence_id,),
                        uncertainty="a contrary reading needs to be checked",
                        next_step="verify against the other incident's record",
                    ).model_dump(mode="json")
                ],
            },
        )
    )

    result, _ = investigate_via_graph(model)

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.FORGED_EVIDENCE_REFERENCE


def test_citing_a_real_same_incident_id_as_contrary_reaches_its_terminal_disposition(
    tmp_path: Path,
) -> None:
    """`test_workflow.py::test_citing_a_real_same_incident_id_as_contrary_reaches_its_terminal_disposition`,
    ported -- the control case paired with the two forged-citation tests
    above: a real, same-incident evidence id cited as contrary is not a
    forgery.

    Unit 2b: this scenario also happens to satisfy two escalation triggers
    at once (an abstention with a full, unspent check budget, and a
    non-empty `contrary_evidence_ids`), so it now pauses rather than
    reaching `final_report`. That pause is itself still the proof this test
    exists for: `final_assessment`'s forged-citation check runs before the
    escalation router does, and a forged citation would have produced a
    `FAILED_SAFE` `InvestigationResult` there, bypassing escalation
    entirely (`route_after_final_assessment` only reaches the router when
    `final_assessment` produced a real assessment). Reaching
    `EscalatedInvestigation` at all is therefore still evidence the
    citation was accepted as genuine, not a forgery."""
    model = ReplayToolCallingModel(
        replay_model(
            tmp_path,
            {
                "initial_plan": [plan_json(stop_reason="the alert is enough")],
                "final_assessment": [
                    FinalAssessment(
                        disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
                        root_cause=RootCauseCode.UNDETERMINED,
                        contrary_evidence_ids=(SYMPTOM_EVIDENCE_ID,),
                        uncertainty="a contrary reading needs to be checked",
                        next_step="verify the symptom evidence again",
                    ).model_dump(mode="json")
                ],
            },
        )
    )

    result, _ = investigate_via_graph(model)

    assert isinstance(result, EscalatedInvestigation)
    assert result.reason is EscalationReason.INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING


def test_a_forged_id_in_a_hypothesis_citation_never_reaches_later_output(
    tmp_path: Path,
) -> None:
    """`test_workflow.py::test_a_forged_id_in_a_hypothesis_citation_never_reaches_later_output`,
    ported. Pins the same actual behavior: a hypothesis citation is never
    validated against the evidence store, so a forged id inside a
    hypothesis's own `supporting_evidence_ids`/`contrary_evidence_ids` lives
    only in that one parsed response and never reaches later context, the
    report, or a recorded event."""
    plan_hypotheses = (
        Hypothesis(
            root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
            rank=1,
            supporting_evidence_ids=(FORGED_HYPOTHESIS_EVIDENCE_ID,),
            missing_evidence="inventory timeout rate in the window",
        ),
        Hypothesis(
            root_cause=RootCauseCode.RESOURCE_POOL_SATURATION,
            rank=2,
            contrary_evidence_ids=(FORGED_HYPOTHESIS_EVIDENCE_ID,),
            missing_evidence="orders pool usage in the window",
        ),
    )
    model = ReplayToolCallingModel(
        replay_model(
            tmp_path,
            {
                "initial_plan": [
                    InitialPlan(
                        hypotheses=plan_hypotheses,
                        stop_reason="the alert is enough",
                    ).model_dump(mode="json")
                ],
                "final_assessment": [assessment_json()],
            },
        )
    )

    result, recorder = investigate_via_graph(model)

    assert result.report.disposition is not Disposition.FAILED_SAFE

    for request in model.requests[1:]:
        assert FORGED_HYPOTHESIS_EVIDENCE_ID not in request.context_text
    assert FORGED_HYPOTHESIS_EVIDENCE_ID not in result.report.model_dump_json()
    recorded = "".join(event.model_dump_json() for event in recorder.events)
    assert FORGED_HYPOTHESIS_EVIDENCE_ID not in recorded


def test_running_out_of_model_calls_fails_safe() -> None:
    """`test_workflow.py::test_running_out_of_model_calls_fails_safe`, ported."""
    registry = registry_with(run_metric=RecordingMetricBackend())

    result, _ = investigate_via_graph(
        fixture_model("valid_diagnosis.json"),
        registry=registry,
        budgets=Budgets(model_calls=1),
    )

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.MODEL_CALL_BUDGET_EXHAUSTED
    assert result.report.model_calls_used == 1


def test_the_second_check_is_skipped_when_the_model_call_budget_would_not_fit(
    tmp_path: Path,
) -> None:
    """`test_workflow.py::test_the_second_check_is_skipped_when_the_assessment_would_not_fit`,
    ported -- and a genuine gap found during this audit, not merely a port:
    `route_after_normalize`'s `_model_calls_left(...) >= 2` branch
    (`graph.py`) was untested until now.
    `test_the_loop_guard_skips_a_second_turn_once_the_check_budget_is_spent`
    above exercises only the sibling `tools_left() > 0` branch (via
    `executed_tools=1`); this is the other one, via `model_calls=2`: one
    spent call and one still owed to `FINAL_ASSESSMENT` leaves no room for
    `HYPOTHESIS_UPDATE`."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    budgets = Budgets(model_calls=2)

    result, _ = investigate_via_graph(model, budgets=budgets)

    assert result.report.disposition is Disposition.DIAGNOSED
    assert result.report.model_calls_used == 2
    assert result.report.tools_executed == 1
    assert [request.stage.value for request in model.requests] == [
        "initial_plan",
        "final_assessment",
    ]


def test_a_run_past_its_wall_clock_fails_safe() -> None:
    """`test_workflow.py::test_a_run_past_its_wall_clock_fails_safe`, ported."""
    result, _ = investigate_via_graph(
        graph_replay_model(), clock=StepClock(step_seconds=400)
    )

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.WALL_CLOCK_EXPIRED
    assert result.report.model_calls_used == 0


def test_token_usage_adds_up_across_every_model_call() -> None:
    """`test_workflow.py::test_token_usage_adds_up_across_every_model_call`,
    ported. `UsageReportingModel` wraps the inner `ReplayReasoningModel`
    *before* `ReplayToolCallingModel` wraps that -- one more layer than the
    loop needs, since the graph always talks to a tool-calling adapter."""
    substitutions = {
        "incident_id": incident_scope().incident_id,
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "symptom_evidence_id": SYMPTOM_EVIDENCE_ID,
    }
    model = ReplayToolCallingModel(
        UsageReportingModel(
            ReplayReasoningModel(GRAPH_FIXTURE, substitutions=substitutions),
            ModelUsage(input_tokens=1000, output_tokens=200),
        )
    )

    result, _ = investigate_via_graph(model)

    assert result.report.model_calls_used == 3
    assert result.report.usage == ModelUsage(input_tokens=3000, output_tokens=600)
    assert result.report.limitations == ()


def test_the_report_names_the_evidence_and_receipts_it_rests_on() -> None:
    """`test_workflow.py::test_the_report_names_the_evidence_and_receipts_it_rests_on`,
    ported -- the no-usage-reported half of the `limitations` property, paired
    with `test_token_usage_adds_up_across_every_model_call` above."""
    result, _ = investigate_via_graph(graph_replay_model())

    assert SYMPTOM_EVIDENCE_ID in result.report.evidence_ids
    assert set(result.report.receipt_ids) == {
        receipt.receipt_id for receipt in result.receipts
    }
    assert result.report.usage is None
    assert result.report.limitations == ("this model reports no token usage",)


def test_the_same_run_produces_the_same_context_digest(tmp_path: Path) -> None:
    """`test_workflow.py::test_the_same_run_produces_the_same_context_digest`,
    ported."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(supporting=(SYMPTOM_EVIDENCE_ID,))],
    }

    first, _ = investigate_via_graph(
        ReplayToolCallingModel(replay_model(tmp_path, script))
    )
    second, _ = investigate_via_graph(
        ReplayToolCallingModel(replay_model(tmp_path, script))
    )

    assert first.report.final_context_digest == second.report.final_context_digest
    assert first.report.final_context_digest


def test_a_settle_then_crash_still_carries_evidence_into_the_final_graph_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph-level proof of the `tool_wrappers.py` evidence-carry fix,
    driven through the full graph so `dispatch_tool`'s own recovery path
    (`ledger.evidence()`, read inside its `except` handler) is what's under
    test here, not just the ledger's durability in isolation --
    `test_tool_wrappers.py::test_a_crash_after_settle_still_leaves_evidence_recoverable_from_the_ledger`
    proves that half. Same settle-then-crash window: `ledger.settle()`
    inside `wrapper.dispatch` already durably recorded the `Evidence` object
    before this monkeypatch makes the very next statement -- constructing
    the `DispatchResult` that used to be the only way that record reached
    its caller -- raise instead."""
    original_init = tool_wrappers_module.DispatchResult.__init__

    def crashing_init(self: object, **data: object) -> None:
        if data.get("evidence") is not None:
            raise RuntimeError("crash after settle, before handoff")
        original_init(self, **data)  # type: ignore[misc]

    monkeypatch.setattr(tool_wrappers_module.DispatchResult, "__init__", crashing_init)

    result, recorder = investigate_via_graph(graph_replay_model())

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.INTERNAL_ERROR
    (only_receipt,) = result.receipts
    assert only_receipt.state is ReceiptState.SETTLED
    assert only_receipt.evidence_id is not None
    # The point of the fix: the settled evidence id is not just on the
    # receipt, it is in the report's own `evidence_ids` and in the evidence
    # actually returned -- recovered from the ledger despite the crash,
    # not lost with the wrapper's frame the way it would be pre-fix.
    assert only_receipt.evidence_id in result.report.evidence_ids
    recovered = [
        record
        for record in result.evidence
        if record.evidence_id == only_receipt.evidence_id
    ]
    assert len(recovered) == 1
    assert recovered[0].content_hash == only_receipt.result_digest
    names = [event.name for event in recorder.events]
    assert "backend_crashed" in names


def test_events_stay_continuous_across_a_second_dispatch_and_normalize_pass(
    tmp_path: Path,
) -> None:
    """Unit 2a moved every node's events into `state["events"]`: each node
    rebuilds a local recorder from that list, records its own events into
    the copy, and must return the full extended list on every one of its
    return paths, or that node's events vanish from state the moment the
    next node reads it. Every single-check fixture this file otherwise uses
    only reaches `dispatch_tool`/`normalize_evidence` once, so none of them
    can expose a gap that only shows up on a *second* pass through either
    node. This reuses the same two-check script
    `test_the_graph_loops_back_for_a_second_check_when_budget_allows` above
    already uses, and checks the one thing a dropped node-local event list
    would break: every event from both passes reaching the caller's
    `recorder`, numbered without a gap."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "hypothesis_update": [plan_json(proposal=another_logs_proposal())],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))

    result, recorder = investigate_via_graph(model)

    assert result.report.tools_executed == 2
    names = [event.name for event in recorder.events]
    assert names.count("proposal_received") == 2
    assert names.count("check_started") == 2
    assert names.count("check_finished") == 2
    assert names.count("evidence_normalized") == 2
    assert names.count("stage_started") == 3
    assert [event.sequence for event in recorder.events] == list(
        range(1, len(recorder.events) + 1)
    )


# --- Unit 2b: the escalation interrupt. `_escalate` runs a scripted scenario
# to its first pause and hands back everything a test needs to resume it
# directly against the raw LangGraph API -- `run_graph_investigation` has no
# resume parameter of its own (see its own docstring for why: 2b's approved
# boundary is graph-level resume driven from tests, not a second production
# entry point), so every resume below goes through `Command(resume=...)`
# against a `compiled` graph and `config` this helper already built with the
# same `checkpointer`/`investigation_id` the pause used.


def _escalate(
    tmp_path: Path,
    script: dict[str, list[dict[str, Any]]],
    *,
    registry: dict[ToolName, ToolWrapper] | None = None,
    investigation_id: str = "escalation-probe",
) -> tuple[
    EscalatedInvestigation,
    CompiledStateGraph[Any, Any, Any, Any],
    RunnableConfig,
    ReplayToolCallingModel,
]:
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    resolved_registry = (
        registry if registry is not None else logs_only_registry(RecordingLogsBackend())
    )
    checkpointer = InMemorySaver()
    clock = StepClock()
    result = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        model,
        resolved_registry,
        RunRecorder(StepClock()),
        Budgets(),
        clock,
        investigation_id=investigation_id,
        checkpointer=checkpointer,
    )
    assert isinstance(result, EscalatedInvestigation)
    compiled = build_graph(
        incident_scope(),
        alert_packet(),
        Budgets(),
        clock,
        model,
        resolved_registry,
        checkpointer,
        event_clock=StepClock(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": investigation_id}}
    return result, compiled, config, model


def _unavailable_logs_backend() -> RecordingLogsBackend:
    return RecordingLogsBackend(
        outcome=CheckOutcome(
            outcome=ToolOutcome.UNAVAILABLE,
            kind=EvidenceKind.LOG,
            source="query_logs",
            summary="log backend unavailable",
            reason_code=ReasonCode.TOOL_UNAVAILABLE,
        )
    )


def test_a_tool_unavailable_receipt_triggers_escalation(tmp_path: Path) -> None:
    """A `query_logs` check that comes back `UNAVAILABLE` pauses the run
    regardless of what the model concludes from the checks that did run --
    `_escalation_reason` checks this trigger first, independent of
    disposition.

    Resumed here, not just paused: every other test asserting the recorded
    `escalation_decided` event's `reason` field uses this file's
    `CONFLICTING_EVIDENCE` scenario, so an assertion of
    `reason == "CONFLICTING_EVIDENCE"` there cannot tell a correct event
    from one that always records that same literal regardless of the real
    trigger. This is the one place a different reason value is available to
    check it against."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "hypothesis_update": [update_json(stop_reason="one check was enough")],
        "final_assessment": [assessment_json()],
    }

    paused, compiled, config, _ = _escalate(
        tmp_path, script, registry=logs_only_registry(_unavailable_logs_backend())
    )

    assert paused.reason is EscalationReason.TOOL_UNAVAILABLE

    resume_graph_run(compiled, config, "accept")

    events = compiled.get_state(config).values["events"]
    decided = [event for event in events if event["name"] == "escalation_decided"]
    assert len(decided) == 1
    assert decided[0]["fields"]["reason"] == "TOOL_UNAVAILABLE"


def test_conflicting_evidence_triggers_escalation(tmp_path: Path) -> None:
    """A diagnosis that cites its own supporting evidence as contrary too is
    still a real diagnosis (`FinalAssessment` has no rule forbidding an id
    appearing in both lists) -- it is `_escalation_reason`'s
    `CONFLICTING_EVIDENCE` check that catches it, isolated from the other
    two triggers: zero checks are proposed, so no receipt exists to go
    `UNAVAILABLE`, and the disposition is `DIAGNOSED`, not
    `INSUFFICIENT_EVIDENCE`."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }

    paused, _, _, _ = _escalate(tmp_path, script)

    assert paused.reason is EscalationReason.CONFLICTING_EVIDENCE
    assert paused.receipts == ()


def test_tool_unavailable_outranks_a_second_trigger_when_both_apply(
    tmp_path: Path,
) -> None:
    """`_escalation_reason` checks `TOOL_UNAVAILABLE` before
    `INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING` -- this scenario satisfies
    both at once (an `UNAVAILABLE` receipt that still leaves a slot open,
    and an abstention), so which one comes back proves the order is
    enforced, not incidental."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "hypothesis_update": [update_json(stop_reason="no second check available")],
        "final_assessment": [
            assessment_json(
                disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
                root_cause=RootCauseCode.UNDETERMINED,
                supporting=(),
            )
        ],
    }

    paused, _, _, _ = _escalate(
        tmp_path, script, registry=logs_only_registry(_unavailable_logs_backend())
    )

    assert paused.reason is EscalationReason.TOOL_UNAVAILABLE
    # Confirms the second trigger's own condition genuinely held too --
    # this is not passing merely because the second trigger was absent.
    assert paused.remaining_check_count > 0


def test_resuming_does_not_repeat_the_model_call_the_pause_already_spent(
    tmp_path: Path,
) -> None:
    """A probe against the installed LangGraph (recorded in this unit's
    pre-edit report) showed only the interrupted node re-running on
    resume, not the whole graph. This is the operationally meaningful form
    of that claim: `final_assessment` -- the node whose model call produced
    the assessment `escalation_interrupt` is now pausing on -- must not run
    a second time and spend a second model call when the pause resumes."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, model = _escalate(tmp_path, script)
    calls_before_resume = len(model.requests)

    resume_graph_run(compiled, config, "accept")

    assert len(model.requests) == calls_before_resume


def test_accepting_an_escalation_keeps_the_assessment_and_records_accept(
    tmp_path: Path,
) -> None:
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    settled = resume_graph_run(compiled, config, "accept")

    assert settled.report.disposition is Disposition.DIAGNOSED
    assert (
        settled.report.root_cause
        is RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION
    )
    assert settled.report.escalation is not None
    assert settled.report.escalation.reason is EscalationReason.CONFLICTING_EVIDENCE
    assert settled.report.escalation.decision == "accept"

    # The node's own event, not just its effect on the report -- deletable
    # with the suite green until this asserted it directly.
    events = compiled.get_state(config).values["events"]
    decided = [event for event in events if event["name"] == "escalation_decided"]
    assert len(decided) == 1
    assert decided[0]["fields"] == {
        "reason": "CONFLICTING_EVIDENCE",
        "decision": "accept",
    }


def test_rejecting_an_escalation_keeps_the_assessment_and_records_reject(
    tmp_path: Path,
) -> None:
    """`reject` stops an investigation without erasing what it concluded:
    the disposition and root cause the model reached still stand, only
    `escalation.decision` differs from the accept case above."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    settled = resume_graph_run(compiled, config, "reject")

    assert settled.report.disposition is Disposition.DIAGNOSED
    assert (
        settled.report.root_cause
        is RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION
    )
    assert settled.report.escalation is not None
    assert settled.report.escalation.reason is EscalationReason.CONFLICTING_EVIDENCE
    assert settled.report.escalation.decision == "reject"

    # The node's own event, not just its effect on the report -- paired
    # with the accept test's identical check, this is what actually pins
    # `decision` to the real resume value rather than to whichever literal
    # a single scenario happened to use.
    events = compiled.get_state(config).values["events"]
    decided = [event for event in events if event["name"] == "escalation_decided"]
    assert len(decided) == 1
    assert decided[0]["fields"] == {
        "reason": "CONFLICTING_EVIDENCE",
        "decision": "reject",
    }


def test_an_unrecognised_resume_decision_re_pauses_instead_of_bricking_the_run(
    tmp_path: Path,
) -> None:
    """A reviewer reproduced this against a real `SqliteSaver`: an earlier
    version of this node raised on a bad decision, and because LangGraph
    persists a resume value against the interrupt id and replays it on
    every later resume of that same interrupt, the raise recurred on every
    subsequent attempt -- a typo permanently bricked the thread, with no
    finalized artifact either, since 2b never finalizes on pause. The fix
    is to re-interrupt instead of raising, so the recovery this test
    proves is the property that actually matters: a bad decision re-pauses
    the same thread, and a later valid decision still settles it.

    Two consecutive bad resumes, not one: a single bad resume cannot tell
    `while decision not in (...): decision = interrupt(...)` apart from
    `if decision not in (...): decision = interrupt(...)` -- both handle
    exactly one retry. Only a second consecutive bad resume distinguishes
    them, and a reviewer measured the `if` variant failing that second
    resume with a bare `AssertionError` out of `_build_report`, no report,
    no recoverable artifact -- the exact bricking class this fix exists to
    prevent, and the class 2c's real owner input will hit routinely, since
    two typos in a row is ordinary."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    first_bad = compiled.invoke(Command(resume="maybe"), config)
    assert "__interrupt__" in first_bad
    assert first_bad["__interrupt__"][0].value["retry"] is True

    second_bad = compiled.invoke(Command(resume="also-bad"), config)
    assert "__interrupt__" in second_bad
    assert second_bad["__interrupt__"][0].value["retry"] is True

    settled = resume_graph_run(compiled, config, "accept")

    assert settled.report.escalation is not None
    assert settled.report.escalation.decision == "accept"


def test_a_bare_accept_string_re_pauses_under_the_unit_2c_resume_contract(
    tmp_path: Path,
) -> None:
    """Unit 2c changes what a *valid* resume value looks like: a mapping
    with `decision`/`rejection_note` keys, not a bare string. A plain
    `Command(resume="accept")` -- exactly what settled every escalation
    test before this unit -- must now re-pause instead of settling, the
    same way a typo does. `resume_graph_run` (this file's own helper)
    already sends the compound shape; this test proves the *old* shape is
    rejected, which nothing else in this suite checks directly."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    bare_string = compiled.invoke(Command(resume="accept"), config)

    assert "__interrupt__" in bare_string
    assert bare_string["__interrupt__"][0].value["retry"] is True

    settled = resume_graph_run(compiled, config, "accept")
    assert settled.report.escalation is not None
    assert settled.report.escalation.decision == "accept"


def test_an_accept_carrying_a_rejection_note_re_pauses(tmp_path: Path) -> None:
    """`_parse_resume_decision`'s accept-side pairing check, exercised
    directly through `Command(resume=...)` -- a mis-paired mapping
    (`decision="accept"` with a non-`None` `rejection_note`) must re-pause
    the same way a malformed value does, never settle with a note attached
    to an acceptance. A mutation dropping this check would let such a
    mapping reach `EscalationRecord`'s own validator instead of the node's,
    turning a recoverable re-pause into `_build_report` raising and
    `_settle_invocation` converting that into `FAILED_SAFE` -- the real
    diagnosis destroyed rather than the run pausing again."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    mispaired = compiled.invoke(
        Command(resume={"decision": "accept", "rejection_note": "should not be here"}),
        config,
    )

    assert "__interrupt__" in mispaired
    assert mispaired["__interrupt__"][0].value["retry"] is True

    settled = resume_graph_run(compiled, config, "accept")
    assert settled.report.escalation is not None
    assert settled.report.escalation.decision == "accept"


def test_a_whitespace_only_rejection_note_re_pauses(tmp_path: Path) -> None:
    """The exact reproduction a reviewer measured: `{"decision": "reject",
    "rejection_note": "   "}` must not settle with a whitespace-only note
    -- `_parse_resume_decision` strips before the emptiness check, the same
    normalization `causalops.approvals.OwnerDecision` already applies at
    the CLI boundary, so a caller that bypasses the CLI entirely still
    cannot land a blank-looking `- Owner's note:` line in the report."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    mispaired = compiled.invoke(
        Command(resume={"decision": "reject", "rejection_note": "   "}), config
    )

    assert "__interrupt__" in mispaired
    assert mispaired["__interrupt__"][0].value["retry"] is True

    settled = resume_graph_run(compiled, config, "accept")
    assert settled.report.escalation is not None
    assert settled.report.escalation.decision == "accept"


def test_a_reject_with_no_note_re_pauses(tmp_path: Path) -> None:
    """`_parse_resume_decision`'s reject-side pairing check: a mis-paired
    mapping (`decision="reject"` with `rejection_note=None`) must re-pause
    rather than reach `EscalationRecord`'s constructor with an invalid
    pairing -- the same destroyed-diagnosis failure mode
    `test_an_accept_carrying_a_rejection_note_re_pauses` guards on the
    other side of the pairing."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    mispaired = compiled.invoke(
        Command(resume={"decision": "reject", "rejection_note": None}), config
    )

    assert "__interrupt__" in mispaired
    assert mispaired["__interrupt__"][0].value["retry"] is True

    settled = resume_graph_run(compiled, config, "accept")
    assert settled.report.escalation is not None
    assert settled.report.escalation.decision == "accept"


def test_an_abstention_with_the_check_budget_spent_does_not_escalate(
    tmp_path: Path,
) -> None:
    """`INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING` is named for the case
    where a slot is still open -- an abstention that already spent the
    whole two-check budget is not that case, and must reach a real,
    finalized `InvestigationResult` rather than pausing. Mutating
    `_escalation_reason`'s `_tools_left(receipts, budgets) > 0` to `>= 0`
    would escalate this scenario too; this is the test that catches it."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "hypothesis_update": [update_json(proposal=another_logs_proposal())],
        "final_assessment": [
            assessment_json(
                disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
                root_cause=RootCauseCode.UNDETERMINED,
                supporting=(),
            )
        ],
    }
    registry = logs_only_registry(RecordingLogsBackend())

    result, _ = investigate_via_graph(
        ReplayToolCallingModel(replay_model(tmp_path, script)), registry=registry
    )

    assert isinstance(result, InvestigationResult)
    assert result.report.disposition is Disposition.INSUFFICIENT_EVIDENCE
    assert result.report.tools_executed == 2


def test_the_escalation_interrupt_node_advances_the_phase_before_final_report(
    tmp_path: Path,
) -> None:
    """`escalation_interrupt`'s own `"phase"` write is unobservable through
    any of this file's other resume tests -- `final_report` always runs
    immediately after and overwrites it, so a terminal `InvestigationResult`
    can never show `ESCALATION_INTERRUPT` regardless of whether this node's
    own return carried that key. `compiled.stream(..., stream_mode="values")`
    is the one way to see it: it yields the committed state after each
    superstep, so the chunk between `final_assessment` and `final_report`
    is `escalation_interrupt`'s own commit, still holding its phase."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    _, compiled, config, _ = _escalate(tmp_path, script)

    phases = [
        chunk["phase"]
        for chunk in compiled.stream(
            Command(resume={"decision": "accept", "rejection_note": None}),
            config,
            stream_mode="values",
        )
    ]

    assert phases == [
        GraphPhase.FINAL_ASSESSMENT.value,
        GraphPhase.ESCALATION_INTERRUPT.value,
        GraphPhase.FINAL_REPORT.value,
    ]


def test_a_paused_runs_recorder_is_not_empty_and_ends_at_the_pre_pause_stage(
    tmp_path: Path,
) -> None:
    """The recorder sync inside `run_graph_investigation`'s tail runs before
    the pause branch is checked, precisely so a caller's own `RunRecorder`
    still reflects everything recorded before the pause. Moving that sync
    below the pause check passes the whole suite green with an empty
    recorder handed back on a paused run, since nothing else in this
    project reads a paused caller's `recorder.events` -- this does."""
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(contrary=(SYMPTOM_EVIDENCE_ID,))],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    recorder = RunRecorder(StepClock())

    result = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        model,
        logs_only_registry(RecordingLogsBackend()),
        recorder,
        Budgets(),
        StepClock(),
        investigation_id="paused-recorder-probe",
        checkpointer=InMemorySaver(),
    )

    assert isinstance(result, EscalatedInvestigation)
    names = [event.name for event in recorder.events]
    assert names
    assert names[0] == "investigation_started"
    # The last event recorded is `final_assessment`'s own `stage_started`
    # -- the stage that pauses next, `escalation_interrupt`, never gets a
    # chance to record anything into this recorder at all, since its own
    # event only lands in state on the resumed attempt.
    assert names[-1] == "stage_started"


def test_an_escalated_investigation_carries_the_same_evidence_and_receipts_so_far(
    tmp_path: Path,
) -> None:
    """`EscalatedInvestigation` mirrors `InvestigationResult`'s own
    `evidence`/`receipts` shape, so an owner inspecting a paused run sees
    the same policy-authorized tool evidence a finished one would show."""
    script = {
        "initial_plan": [plan_json(proposal=logs_proposal())],
        "hypothesis_update": [update_json(stop_reason="one check was enough")],
        "final_assessment": [assessment_json()],
    }

    paused, _, _, _ = _escalate(
        tmp_path, script, registry=logs_only_registry(_unavailable_logs_backend())
    )

    assert paused.thread_id == "escalation-probe"
    assert paused.checkpoint_id
    assert paused.proposal_fingerprint is None
    assert len(paused.receipts) == 1
    assert paused.receipts[0].outcome is ToolOutcome.UNAVAILABLE
    assert len(paused.evidence) == 2
