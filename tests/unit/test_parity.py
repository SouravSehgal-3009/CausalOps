"""The loop and the graph agree on one replay incident.

`TECHNICAL_SPEC.md` §12 calls this bounded tool-graph parity: Milestone 1
only retires `workflow.py`'s loop (Unit 1d) once the two orchestrators are
shown to agree on the same script. This file is that proof --
`graph_single_check.json` for the single-check happy path, plus three
throwaway scripts for the branch cases a happy-path fixture cannot reach:
a first-turn denial that still permits a second turn, a second-turn denial
that must not (P1-1's regression at the orchestrator-comparison level), and
a model that crashes mid-run (P1-2's). `dispatch_registry` wraps only
`query_logs` until Unit 1c, so every scenario here uses that one tool.

What is compared, and what is deliberately not:

- The stage sequence each model was asked (`ReplayReasoningModel.requests`
  already records this for the loop; `ReplayToolCallingModel.requests`
  delegates to the same list for the graph).
- Report-level: `disposition`, `root_cause`, `tools_executed`,
  `model_calls_used`, and each receipt's `(policy_result, state, outcome,
  reason_code)` tuple, in order.
- `final_context_digest` is **not** compared. Evidence IDs are `uuid4().hex`,
  minted independently by each orchestrator's own `new_opaque_id()` calls,
  so the digest of context that quotes a check result is stable only when no
  check evidence enters context -- not the case for every scenario here.
  Comparing it would be asserting a coincidence, not parity.
- Wall-clock fields (`latency_ms`, `started_at`, `finished_at`) and `usage`
  are not compared either: both orchestrators use independent `StepClock`
  instances, and the replay model never reports usage.
"""

from pathlib import Path

from fake_incident import (
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    StepClock,
    alert_packet,
    assessment_json,
    incident_scope,
    logs_proposal,
    packet_evidence,
    plan_json,
    replay_model,
)

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
from causalops.run_records import RunRecorder
from causalops.telemetry import RunPaths, registered_check_runner, run_logs_check
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


def run_both(
    loop_model: ReasoningModel,
    graph_model: ReplayToolCallingModel,
    paths: RunPaths,
    budgets: Budgets | None = None,
) -> tuple[InvestigationResult, InvestigationResult]:
    scope = incident_scope()
    packet = alert_packet()
    evidence_records = packet_evidence()
    resolved_budgets = budgets or Budgets()

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

    graph_recorder = RunRecorder(StepClock())
    registry = dispatch_registry(lambda arguments: run_logs_check(arguments, paths))
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
    return loop_result, graph_result


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
    assert [receipt_shape(r) for r in loop_result.receipts] == [
        receipt_shape(r) for r in graph_result.receipts
    ]


def test_the_loop_and_the_graph_agree_on_one_replay_incident(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_orders_error_row(paths)
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)

    loop_model = ReplayReasoningModel(FIXTURE, substitutions=subs)
    graph_model = ReplayToolCallingModel(
        ReplayReasoningModel(FIXTURE, substitutions=subs)
    )

    loop_result, graph_result = run_both(loop_model, graph_model, paths)

    assert [request.stage for request in loop_model.requests] == [
        request.stage for request in graph_model.requests
    ]
    assert_reports_agree(loop_result, graph_result)

    # Both orchestrators collected the same evidence kinds in the same
    # order: the packet's symptom/topology records, then the one executed
    # check's log evidence. IDs differ (independent `uuid4()` calls), so the
    # comparison is by kind, not by identity.
    assert [record.kind for record in loop_result.evidence] == [
        record.kind for record in graph_result.evidence
    ]
    assert EvidenceKind.LOG in [record.kind for record in graph_result.evidence]
    assert SYMPTOM_EVIDENCE_ID in [
        record.evidence_id for record in graph_result.evidence
    ]


def test_a_first_turn_denial_still_permits_a_second_turn(tmp_path: Path) -> None:
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

    loop_result, graph_result = run_both(loop_model, graph_model, paths)

    assert [request.stage.value for request in loop_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_reports_agree(loop_result, graph_result)


def test_a_repeated_proposal_after_one_executed_check(tmp_path: Path) -> None:
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

    loop_result, graph_result = run_both(loop_model, graph_model, paths)

    assert [request.stage.value for request in loop_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_reports_agree(loop_result, graph_result)


def test_a_model_that_raises_on_its_second_call(tmp_path: Path) -> None:
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

    loop_result, graph_result = run_both(loop_model, graph_model, paths)

    assert loop_result.report.model_calls_used == 2
    assert_reports_agree(loop_result, graph_result)
