"""The loop and the graph agree on one replay incident.

`TECHNICAL_SPEC.md` §12 calls this bounded tool-graph parity: Milestone 1
only retires `workflow.py`'s loop (Unit 1d) once the two orchestrators are
shown to agree on the same script. This file is that proof --
`graph_single_check.json` for the single-check happy path, plus three
throwaway scripts for the branch cases a happy-path fixture cannot reach
(a first-turn denial that still permits a second turn, a second-turn denial
that must not -- P1-1's regression at the orchestrator-comparison level --
and a model that crashes mid-run, P1-2's), plus `lab_diagnosis.json` for the
two-executed-check, two-tool case none of the single-check scenarios reach.
As of Unit 1c, `dispatch_registry` wraps all four tools, so every scenario
here runs against the same real backends the CLI wires, not spies.

**Every dispatch attempt mints receipt/evidence ids through `new_opaque_id()`
(`evidence.py:37`), and each orchestrator calls it independently** -- the
loop and the graph do not share one call sequence. Left alone, that makes
receipt ids, evidence ids, and `final_context_digest` (which quotes evidence
ids in the rendered prompt) incomparable by construction: any two independent
`uuid4()` sequences differ, so equality there would prove nothing. `run_both`
closes that gap instead of excluding it: `_install_counting_ids` replaces
`new_opaque_id` with a plain call-counter, reset to zero immediately before
each orchestrator's own run. If the two orchestrators mint ids in the same
order for the same operations -- receipt before evidence, once per dispatch,
regardless of which tool -- their two id sequences are then byte-identical,
not merely equal in shape. That turns three of the four things the docstring
below used to list as excluded into hard equalities; only wall-clock fields
remain excluded, because nothing here makes two independent `StepClock`
instances tick in lockstep, or needs to.

What is compared, and what is deliberately not:

- The stage sequence each model was asked (`ReplayReasoningModel.requests`
  already records this for the loop; `ReplayToolCallingModel.requests`
  delegates to the same list for the graph).
- Report-level, exactly: `disposition`, `root_cause`, `tools_executed`,
  `model_calls_used`, `repairs_used`, `invalid_responses`, `usage`,
  `final_context_digest`, `evidence_ids`, and `receipt_ids` -- the last two
  now compared as literal id sequences, not merely by kind or shape, thanks
  to the id-normalisation harness above. Each receipt's full
  `(policy_result, state, outcome, reason_code)` tuple is still compared too,
  in order, alongside its now-exact `receipt_id`.
- The dispatch-vocabulary event field dicts (`proposal_received`,
  `proposal_denied`, `check_started`, `check_finished`) recorded by each
  orchestrator's `RunRecorder`, compared as ordered `(name, fields)` pairs --
  see `dispatch_events`/`assert_dispatch_events_agree` below for exactly
  what is excluded from a `fields` dict and why.
- Wall-clock fields (`latency_ms`, `started_at`, `finished_at`) are the one
  thing still excluded: both orchestrators use independent `StepClock`
  instances that tick a different number of times, and nothing about this
  unit's id-normalisation touches that.
"""

from pathlib import Path

import pytest
from fake_incident import (
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    StepClock,
    alert_packet,
    assessment_json,
    change_row,
    incident_scope,
    logs_proposal,
    packet_evidence,
    plan_json,
    replay_model,
    write_changes,
)

import causalops.evidence as evidence_module
import causalops.graph as graph_module
import causalops.tool_wrappers as tool_wrappers_module
import causalops.workflow as workflow_module
from causalops.domain import (
    Budgets,
    IncidentScope,
    InitialAlertPacket,
    ToolProposal,
    ToolReceipt,
)
from causalops.evidence import EvidenceKind
from causalops.graph import run_graph_investigation
from causalops.models import (
    ModelRequest,
    ModelResponse,
    ReasoningModel,
    ReplayReasoningModel,
    ReplayToolCallingModel,
)
from causalops.prometheus import run_metric_check
from causalops.run_records import RunRecorder
from causalops.telemetry import (
    RunPaths,
    registered_check_runner,
    run_changes_check,
    run_logs_check,
    run_topology_check,
)
from causalops.tool_wrappers import dispatch_registry
from causalops.tools import LogFilter, QueryLogsArguments, ToolName
from causalops.workflow import InvestigationResult
from causalops.workflow import run_investigation as run_loop_investigation

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "causalops"
    / "replay_fixtures"
    / "graph_single_check.json"
)


def substitutions(scope: IncidentScope, packet: InitialAlertPacket) -> dict[str, str]:
    return {
        "incident_id": scope.incident_id,
        "window_start": scope.started_at.isoformat(),
        "window_end": scope.ended_at.isoformat(),
        "symptom_evidence_id": packet.symptom_evidence_id,
    }


def write_orders_error_row(paths: RunPaths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    row = (
        '{"at": "'
        + WINDOW_START.isoformat()
        + '", "request_id": "r1", "service": "orders", "severity": "error", '
        '"event": "config_rejected_request", '
        '"fields": {"config_key": "require_order_token", "detail": "x"}}\n'
    )
    (paths.logs / "orders.jsonl").write_text(row, encoding="utf-8")


def receipt_shape(
    receipt: ToolReceipt,
) -> tuple[ToolName, str, str, str | None, str | None]:
    return (
        receipt.tool,
        receipt.policy_result.value,
        receipt.state.value,
        receipt.outcome.value if receipt.outcome is not None else None,
        receipt.reason_code.value if receipt.reason_code is not None else None,
    )


def out_of_scope_logs_proposal() -> ToolProposal:
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


class RaisesOnSecondCall:
    """Wraps a `ReplayReasoningModel` so its *second* `.respond()` call
    raises, whichever orchestrator is driving it. `ReplayToolCallingModel`'s
    `propose()` and `respond()` both delegate to `self.inner.respond(...)`,
    so wrapping the inner model here (rather than the tool-calling adapter)
    counts every underlying call regardless of which of the graph's two
    entry points made it -- and the same wrapper drives the loop directly,
    since it satisfies `ReasoningModel` itself.
    """

    def __init__(self, inner: ReplayReasoningModel) -> None:
        self.inner = inner
        self.calls = 0

    @property
    def requests(self) -> list[ModelRequest]:
        return self.inner.requests

    def respond(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("provider timeout")
        return self.inner.respond(request)


class _CountingIdGenerator:
    """A deterministic stand-in for `new_opaque_id()`: increasing integers
    formatted to the same 32-hex-character shape a real opaque id has, so a
    receipt/evidence id comparison across orchestrators is exact instead of
    merely equal in kind."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"{self.count:032x}"


def _install_counting_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every module that did `from causalops.evidence import new_opaque_id`
    holds its own binding of that name -- patching
    `causalops.evidence.new_opaque_id` alone would leave `tool_wrappers.py`'s
    and `workflow.py`'s own copies pointing at the real, `uuid4()`-backed
    function. All four share one fresh counter, so a run's receipt id and
    its evidence id fall on the same incrementing sequence, exactly as they
    would if one real `new_opaque_id()` produced both."""
    generator = _CountingIdGenerator()
    for module in (
        evidence_module,
        graph_module,
        tool_wrappers_module,
        workflow_module,
    ):
        monkeypatch.setattr(module, "new_opaque_id", generator)


DISPATCH_EVENT_NAMES = {
    "proposal_received",
    "proposal_denied",
    "check_started",
    "check_finished",
}


def dispatch_events(recorder: RunRecorder) -> list[tuple[str, dict[str, object]]]:
    """The dispatch-vocabulary events, as ordered `(name, fields)` pairs.

    Three things a whole-event comparison would need to strip are excluded
    by construction instead, because none of them live inside `fields`:
    `at` (each recorder owns an independent `StepClock`, read a different
    number of times by each orchestrator), `sequence` (positional within one
    recorder's own list), and `state` (the loop tags each event with the
    stage it happened during; the graph tags every dispatch event
    `DISPATCH_TOOL`; the two vocabularies do not overlap, and that is
    inherent to the two orchestrators' shapes, not a defect in either)."""
    return [
        (event.name, dict(event.fields))
        for event in recorder.events
        if event.name in DISPATCH_EVENT_NAMES
    ]


def assert_dispatch_events_agree(
    loop_recorder: RunRecorder, graph_recorder: RunRecorder
) -> None:
    loop_events = dispatch_events(loop_recorder)
    graph_events = dispatch_events(graph_recorder)
    assert len(loop_events) == len(graph_events)
    for (loop_name, loop_fields), (graph_name, graph_fields) in zip(
        loop_events, graph_events, strict=True
    ):
        assert loop_name == graph_name
        if loop_name == "check_finished":
            # The graph carries `duration_ms` as a documented superset (see
            # this module's docstring); the loop's `check_finished` never
            # had that field to begin with.
            graph_fields = {
                key: value
                for key, value in graph_fields.items()
                if key != "duration_ms"
            }
        assert loop_fields == graph_fields


def run_both(
    loop_model: ReasoningModel,
    graph_model: ReplayToolCallingModel,
    paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
    budgets: Budgets | None = None,
) -> tuple[InvestigationResult, InvestigationResult, RunRecorder, RunRecorder]:
    scope = incident_scope()
    packet = alert_packet()
    evidence_records = packet_evidence()
    resolved_budgets = budgets or Budgets()

    _install_counting_ids(monkeypatch)
    loop_recorder = RunRecorder(StepClock())
    loop_result = run_loop_investigation(
        scope,
        packet,
        evidence_records,
        loop_model,
        registered_check_runner(paths, "http://unused", 10),
        loop_recorder,
        resolved_budgets,
        StepClock(),
    )

    _install_counting_ids(monkeypatch)
    graph_recorder = RunRecorder(StepClock())
    # All four real backends, the same ones `cli.py` wires -- a parity claim
    # should be proven against what actually runs, not a spy. None of this
    # file's scenarios reach `query_metric`, so its unreachable URL is never
    # dialled; it is wired for real all the same, not stubbed out.
    registry = dispatch_registry(
        run_metric=lambda arguments, scope: run_metric_check(
            arguments, scope, "http://unused", 1
        ),
        run_logs=lambda arguments, scope: run_logs_check(arguments, paths),
        run_changes=lambda arguments, scope: run_changes_check(arguments, paths),
        run_topology=lambda arguments, scope: run_topology_check(arguments, paths),
    )
    graph_result = run_graph_investigation(
        scope,
        packet,
        evidence_records,
        graph_model,
        registry,
        graph_recorder,
        resolved_budgets,
        StepClock(),
    )
    return loop_result, graph_result, loop_recorder, graph_recorder


def assert_reports_agree(
    loop_result: InvestigationResult, graph_result: InvestigationResult
) -> None:
    loop_report = loop_result.report
    graph_report = graph_result.report
    assert loop_report.disposition == graph_report.disposition
    assert loop_report.root_cause == graph_report.root_cause
    assert loop_report.tools_executed == graph_report.tools_executed
    assert loop_report.model_calls_used == graph_report.model_calls_used
    assert loop_report.repairs_used == graph_report.repairs_used
    assert loop_report.invalid_responses == graph_report.invalid_responses
    assert loop_report.usage == graph_report.usage
    # Exact now, not merely equal-in-shape, thanks to the id-normalisation
    # harness above: both orchestrators minted the same ids in the same
    # order, so the digest that quotes them and the id sequences themselves
    # are directly comparable.
    assert loop_report.final_context_digest == graph_report.final_context_digest
    assert loop_report.evidence_ids == graph_report.evidence_ids
    assert loop_report.receipt_ids == graph_report.receipt_ids
    assert [receipt_shape(r) for r in loop_result.receipts] == [
        receipt_shape(r) for r in graph_result.receipts
    ]


def test_the_loop_and_the_graph_agree_on_one_replay_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RunPaths(root=tmp_path)
    write_orders_error_row(paths)
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)

    loop_model = ReplayReasoningModel(FIXTURE, substitutions=subs)
    graph_model = ReplayToolCallingModel(
        ReplayReasoningModel(FIXTURE, substitutions=subs)
    )

    loop_result, graph_result, loop_recorder, graph_recorder = run_both(
        loop_model, graph_model, paths, monkeypatch
    )

    assert [request.stage for request in loop_model.requests] == [
        request.stage for request in graph_model.requests
    ]
    assert_reports_agree(loop_result, graph_result)
    assert_dispatch_events_agree(loop_recorder, graph_recorder)

    # Both orchestrators collected the same evidence kinds, in the same
    # order and now with the same ids too: the packet's symptom/topology
    # records, then the one executed check's log evidence.
    assert [record.kind for record in loop_result.evidence] == [
        record.kind for record in graph_result.evidence
    ]
    assert [record.evidence_id for record in loop_result.evidence] == [
        record.evidence_id for record in graph_result.evidence
    ]
    assert EvidenceKind.LOG in [record.kind for record in graph_result.evidence]
    assert SYMPTOM_EVIDENCE_ID in [
        record.evidence_id for record in graph_result.evidence
    ]


def test_a_first_turn_denial_still_permits_a_second_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario (a): an out-of-scope proposal on turn 0, denied, followed by
    a scripted second proposal on turn 1. Both orchestrators must still ask
    `HYPOTHESIS_UPDATE` -- `plan_second_check()` gates on whether turn 0
    *proposed* something, not on whether it was *allowed* -- so this also
    checks the `model_turn < 2` bound does not overreach into cutting off a
    legitimate second turn.

    This is a second, independent P1-1 regression, not only an overreach
    guard: turn 0's denial does not spend a slot, so after turn 1 executes
    (spending one of two), `tools_left()` is still 1 -- without
    `model_turn < 2`, the router would loop for a phantom third turn here
    too, on a script that never denies the second proposal at all. The
    guard is what stops it, in both this scenario and (b)'s below."""
    paths = RunPaths(root=tmp_path)
    script = {
        "initial_plan": [plan_json(proposal=out_of_scope_logs_proposal())],
        "hypothesis_update": [plan_json(proposal=logs_proposal())],
        "final_assessment": [assessment_json()],
    }
    loop_model = replay_model(tmp_path, script)
    graph_model = ReplayToolCallingModel(replay_model(tmp_path, script))

    loop_result, graph_result, loop_recorder, graph_recorder = run_both(
        loop_model, graph_model, paths, monkeypatch
    )

    assert [request.stage.value for request in loop_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_reports_agree(loop_result, graph_result)
    assert_dispatch_events_agree(loop_recorder, graph_recorder)


def test_a_repeated_proposal_after_one_executed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario (b), P1-1's regression at the orchestrator-comparison level:
    the same proposal scripted verbatim for both turns. Turn 0 executes;
    turn 1's proposal fingerprints identically and is denied as a duplicate.
    Before the fix, `route_after_normalize` had nothing stopping it from
    looping for a phantom third turn (a denial does not spend a slot), which
    would have asked `HYPOTHESIS_UPDATE` again and exhausted this script's
    single scripted response for that stage -- diverging sharply from the
    loop, which never asks a third time at all."""
    paths = RunPaths(root=tmp_path)
    repeated = logs_proposal()
    script = {
        "initial_plan": [plan_json(proposal=repeated)],
        "hypothesis_update": [plan_json(proposal=repeated)],
        "final_assessment": [assessment_json()],
    }
    loop_model = replay_model(tmp_path, script)
    graph_model = ReplayToolCallingModel(replay_model(tmp_path, script))

    loop_result, graph_result, loop_recorder, graph_recorder = run_both(
        loop_model, graph_model, paths, monkeypatch
    )

    assert [request.stage.value for request in loop_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_reports_agree(loop_result, graph_result)
    assert_dispatch_events_agree(loop_recorder, graph_recorder)


def test_a_model_that_raises_on_its_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario (c), P1-2's regression at the orchestrator-comparison level.
    `BudgetLedger.record_model_call()` runs before `self.model.respond()` in
    the loop, so a crashed second call is still counted when
    `internal_error()` builds the report -- that has always been true of
    `workflow.py`. Before the graph's own fix, its node-local counters died
    with the crashed node's frame instead, so the graph reported one fewer
    call than the loop for the identical script."""
    paths = RunPaths(root=tmp_path)
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)

    loop_model = RaisesOnSecondCall(ReplayReasoningModel(FIXTURE, substitutions=subs))
    graph_model = ReplayToolCallingModel(
        RaisesOnSecondCall(ReplayReasoningModel(FIXTURE, substitutions=subs))
    )

    loop_result, graph_result, loop_recorder, graph_recorder = run_both(
        loop_model, graph_model, paths, monkeypatch
    )

    assert loop_result.report.model_calls_used == 2
    assert_reports_agree(loop_result, graph_result)
    assert_dispatch_events_agree(loop_recorder, graph_recorder)


LAB_DIAGNOSIS_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "causalops"
    / "replay_fixtures"
    / "lab_diagnosis.json"
)


def test_the_loop_and_the_graph_agree_on_two_executed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lab_diagnosis.json` -- the loop's own default fixture -- proposes
    `query_logs` then `list_recent_changes`: two executed checks across two
    tools, new ground for this file. Every scenario above ran a single
    `query_logs` check; this is the first proof that the three tools Unit 1c
    wrapped agree with the loop too, not just that the shape holds when
    `query_logs` is the only tool involved."""
    paths = RunPaths(root=tmp_path)
    write_orders_error_row(paths)
    write_changes(paths, [change_row(1)])
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)

    loop_model = ReplayReasoningModel(LAB_DIAGNOSIS_FIXTURE, substitutions=subs)
    graph_model = ReplayToolCallingModel(
        ReplayReasoningModel(LAB_DIAGNOSIS_FIXTURE, substitutions=subs)
    )

    loop_result, graph_result, loop_recorder, graph_recorder = run_both(
        loop_model, graph_model, paths, monkeypatch
    )

    assert [request.stage for request in loop_model.requests] == [
        request.stage for request in graph_model.requests
    ]
    assert loop_result.report.tools_executed == 2
    assert_reports_agree(loop_result, graph_result)
    assert_dispatch_events_agree(loop_recorder, graph_recorder)
    assert [record.kind for record in loop_result.evidence] == [
        record.kind for record in graph_result.evidence
    ]
    assert [record.evidence_id for record in loop_result.evidence] == [
        record.evidence_id for record in graph_result.evidence
    ]
    assert EvidenceKind.CHANGE in [record.kind for record in graph_result.evidence]
