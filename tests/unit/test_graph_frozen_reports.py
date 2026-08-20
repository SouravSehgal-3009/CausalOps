"""The graph reproduces the loop's own recorded outcome on five scripts.

Through Unit 1c, this file ran the loop and the graph side by side and
compared their live output (`TECHNICAL_SPEC.md` §12's bounded tool-graph
parity gate for retiring `workflow.py`). That gate is now met -- a reviewer's
144-pair differential sweep found zero divergence across 13 dimensions -- and
Unit 1d-1 converts this file accordingly: each scenario below now runs the
graph alone, against literal values captured from the loop's actual last-known
output while both orchestrators were still callable side by side. Unit 1d-2
deletes `workflow.py` entirely; this file's job from that point on is a
regression pin on the graph's own behaviour, not a comparison.

**One property has no successor and is dropped outright, not reworded**: "the
loop and the graph asked the same stages in the same order" was a genuine
two-orchestrator comparison. With one orchestrator left, there is nothing left
to compare it against -- each test below still asserts the graph's own stage
sequence against a frozen expected list, which is a different, weaker claim
than the original, and is named as such rather than presented as equivalent.

`graph_single_check.json` covers the single-check happy path; three throwaway
scripts cover branch cases it cannot reach (a first-turn denial that still
permits a second turn, a second-turn denial that must not -- P1-1's
regression, pinned here at the orchestrator level -- and a model that crashes
mid-run, P1-2's); `lab_diagnosis.json` covers the two-executed-check, two-tool
case none of the single-check scripts reach. Every scenario runs against the
same real backends `cli.py` wires, not spies, matching Unit 1c's coverage.

**Every dispatch attempt mints receipt/evidence ids through `new_opaque_id()`
(`evidence.py:37`).** The id-normalisation harness below
(`_install_counting_ids`) still runs before each test, for a different reason
than it used to: it no longer aligns two independent orchestrators' id
sequences, it makes *this one* orchestrator's ids deterministic, which is what
lets the literals frozen in each test below match on every run rather than
only the run that produced them.

What each test pins, and what it does not:

- The graph's own stage sequence (`ReplayToolCallingModel.requests`), where a
  scenario's script depends on it.
- Report-level, exactly: `disposition`, `root_cause`, `tools_executed`,
  `model_calls_used`, `repairs_used`, `invalid_responses`, `usage`,
  `final_context_digest`, `evidence_ids`, and `receipt_ids`. Each receipt's
  full `(tool, policy_result, state, outcome, reason_code)` tuple is pinned
  too, in order.
- The dispatch-vocabulary event field dicts (`proposal_received`,
  `proposal_denied`, `check_started`, `check_finished`) the graph's own
  `RunRecorder` produced, as ordered `(name, fields)` pairs.
- Wall-clock fields (`latency_ms`, `started_at`, `finished_at`) are excluded,
  as they always were: nothing about a frozen literal makes `StepClock`
  readings meaningful to pin.
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
from causalops.domain import (
    Budgets,
    Disposition,
    IncidentScope,
    InitialAlertPacket,
    InvestigationReport,
    InvestigationResult,
    RootCauseCode,
    ToolProposal,
    ToolReceipt,
)
from causalops.evidence import EvidenceKind
from causalops.graph import run_graph_investigation
from causalops.models import (
    ModelRequest,
    ModelResponse,
    ReplayReasoningModel,
    ReplayToolCallingModel,
)
from causalops.prometheus import run_metric_check
from causalops.run_records import RunRecorder
from causalops.telemetry import (
    RunPaths,
    run_changes_check,
    run_logs_check,
    run_topology_check,
)
from causalops.tool_wrappers import dispatch_registry
from causalops.tools import LogFilter, QueryLogsArguments, ToolName

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "causalops"
    / "replay_fixtures"
    / "graph_single_check.json"
)

LAB_DIAGNOSIS_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "causalops"
    / "replay_fixtures"
    / "lab_diagnosis.json"
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
    raises. `ReplayToolCallingModel`'s `propose()` and `respond()` both
    delegate to `self.inner.respond(...)`, so wrapping the inner model here
    counts every underlying call regardless of which of the graph's two entry
    points made it."""

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
    formatted to the same 32-hex-character shape a real opaque id has, so the
    literals frozen below match on every run, not only the run that produced
    them."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"{self.count:032x}"


def _install_counting_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every module that did `from causalops.evidence import new_opaque_id`
    holds its own binding of that name -- patching
    `causalops.evidence.new_opaque_id` alone would leave `tool_wrappers.py`'s
    own copy pointing at the real, `uuid4()`-backed function."""
    generator = _CountingIdGenerator()
    for module in (evidence_module, graph_module, tool_wrappers_module):
        monkeypatch.setattr(module, "new_opaque_id", generator)


DISPATCH_EVENT_NAMES = {
    "proposal_received",
    "proposal_denied",
    "check_started",
    "check_finished",
}


def dispatch_events(recorder: RunRecorder) -> list[tuple[str, dict[str, object]]]:
    """The dispatch-vocabulary events, as ordered `(name, fields)` pairs."""
    return [
        (event.name, dict(event.fields))
        for event in recorder.events
        if event.name in DISPATCH_EVENT_NAMES
    ]


def run_once(
    graph_model: ReplayToolCallingModel,
    paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
    budgets: Budgets | None = None,
) -> tuple[InvestigationResult, RunRecorder]:
    """Runs the graph orchestrator alone. `run_both`'s old name for this half
    is gone along with the loop half it used to run beside -- kept as one
    function, not inlined per test, since all five scenarios wire the same
    real four-tool registry the same way."""
    scope = incident_scope()
    packet = alert_packet()
    evidence_records = packet_evidence()
    resolved_budgets = budgets or Budgets()

    _install_counting_ids(monkeypatch)
    graph_recorder = RunRecorder(StepClock())
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
    return graph_result, graph_recorder


def assert_report_matches_frozen(
    report: InvestigationReport,
    *,
    disposition: Disposition,
    root_cause: RootCauseCode,
    tools_executed: int,
    model_calls_used: int,
    repairs_used: int,
    invalid_responses: int,
    final_context_digest: str,
    evidence_ids: tuple[str, ...],
    receipt_ids: tuple[str, ...],
) -> None:
    """Field-by-field comparison against literals frozen from the loop's
    actual last-known output, not against another live run -- the same field
    list `assert_reports_agree` used to compare between two orchestrators,
    now compared against one orchestrator and a constant."""
    assert report.disposition is disposition
    assert report.root_cause is root_cause
    assert report.tools_executed == tools_executed
    assert report.model_calls_used == model_calls_used
    assert report.repairs_used == repairs_used
    assert report.invalid_responses == invalid_responses
    assert report.usage is None
    assert report.final_context_digest == final_context_digest
    assert report.evidence_ids == evidence_ids
    assert report.receipt_ids == receipt_ids


def test_the_graph_reproduces_the_frozen_report_for_one_replay_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through Unit 1c this test ran the loop beside the graph on
    `graph_single_check.json` and compared their live output; the values
    below are that comparison's last agreement, frozen."""
    paths = RunPaths(root=tmp_path)
    write_orders_error_row(paths)
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)
    graph_model = ReplayToolCallingModel(
        ReplayReasoningModel(FIXTURE, substitutions=subs)
    )

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert [request.stage.value for request in graph_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.CONFIG_CHANGE,
        tools_executed=1,
        model_calls_used=3,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "ed690a1ae427badf456f7e4cb50ac532b980cbb601074e572de7fc78ed83dbfe"
        ),
        evidence_ids=(
            SYMPTOM_EVIDENCE_ID,
            "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d",
            "00000000000000000000000000000003",
        ),
        receipt_ids=("00000000000000000000000000000002",),
    )
    assert [receipt_shape(r) for r in result.receipts] == [
        (ToolName.QUERY_LOGS, "ALLOWED", "SETTLED", "EXECUTED", None)
    ]
    assert dispatch_events(recorder) == [
        ("proposal_received", {"tool": "query_logs"}),
        ("check_started", {"tool": "query_logs"}),
        ("check_finished", {"outcome": "EXECUTED", "duration_ms": 0}),
    ]
    assert [record.kind for record in result.evidence] == [
        EvidenceKind.SYMPTOM,
        EvidenceKind.TOPOLOGY,
        EvidenceKind.LOG,
    ]


def test_the_graph_reproduces_the_frozen_report_after_a_first_turn_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-1's regression, pinned at the orchestrator level: an out-of-scope
    proposal on turn 0, denied, still permits a second turn --
    `plan_second_check()`'s graph equivalent gates on whether turn 0
    *proposed* something, not on whether it was *allowed*."""
    paths = RunPaths(root=tmp_path)
    script = {
        "initial_plan": [plan_json(proposal=out_of_scope_logs_proposal())],
        "hypothesis_update": [plan_json(proposal=logs_proposal())],
        "final_assessment": [assessment_json()],
    }
    graph_model = ReplayToolCallingModel(replay_model(tmp_path, script))

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert [request.stage.value for request in graph_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
        tools_executed=1,
        model_calls_used=3,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "0bc37313ccbaa1bc89ad576dfb200988ca959d2673de0dcd34edd2f35ce62ae2"
        ),
        evidence_ids=(SYMPTOM_EVIDENCE_ID, "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"),
        receipt_ids=(
            "00000000000000000000000000000002",
            "00000000000000000000000000000003",
        ),
    )
    assert [receipt_shape(r) for r in result.receipts] == [
        (ToolName.QUERY_LOGS, "DENIED", "SETTLED", "NOT_EXECUTED", "UNKNOWN_SERVICE"),
        (ToolName.QUERY_LOGS, "ALLOWED", "SETTLED", "UNAVAILABLE", "TOOL_UNAVAILABLE"),
    ]
    assert dispatch_events(recorder) == [
        ("proposal_received", {"tool": "query_logs"}),
        (
            "proposal_denied",
            {"reason": "UNKNOWN_SERVICE", "message": "that service is out of scope"},
        ),
        ("proposal_received", {"tool": "query_logs"}),
        ("check_started", {"tool": "query_logs"}),
        ("check_finished", {"outcome": "UNAVAILABLE", "duration_ms": 0}),
    ]


def test_the_graph_reproduces_the_frozen_report_after_a_repeated_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-1's other regression, pinned at the orchestrator level: the same
    proposal scripted verbatim for both turns. Turn 0 executes; turn 1's
    proposal fingerprints identically and is denied as a duplicate rather
    than the router asking a phantom third turn."""
    paths = RunPaths(root=tmp_path)
    repeated = logs_proposal()
    script = {
        "initial_plan": [plan_json(proposal=repeated)],
        "hypothesis_update": [plan_json(proposal=repeated)],
        "final_assessment": [assessment_json()],
    }
    graph_model = ReplayToolCallingModel(replay_model(tmp_path, script))

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert [request.stage.value for request in graph_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
        tools_executed=1,
        model_calls_used=3,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "0bc37313ccbaa1bc89ad576dfb200988ca959d2673de0dcd34edd2f35ce62ae2"
        ),
        evidence_ids=(SYMPTOM_EVIDENCE_ID, "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"),
        receipt_ids=(
            "00000000000000000000000000000002",
            "00000000000000000000000000000003",
        ),
    )
    assert [receipt_shape(r) for r in result.receipts] == [
        (ToolName.QUERY_LOGS, "ALLOWED", "SETTLED", "UNAVAILABLE", "TOOL_UNAVAILABLE"),
        (
            ToolName.QUERY_LOGS,
            "DENIED",
            "SETTLED",
            "NOT_EXECUTED",
            "DUPLICATE_PROPOSAL",
        ),
    ]
    assert dispatch_events(recorder) == [
        ("proposal_received", {"tool": "query_logs"}),
        ("check_started", {"tool": "query_logs"}),
        ("check_finished", {"outcome": "UNAVAILABLE", "duration_ms": 0}),
        ("proposal_received", {"tool": "query_logs"}),
        (
            "proposal_denied",
            {
                "reason": "DUPLICATE_PROPOSAL",
                "message": "this check was proposed already",
            },
        ),
    ]


def test_the_graph_reproduces_the_frozen_report_when_the_second_call_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-2's regression, pinned at the orchestrator level: `investigate`'s
    `ask_once` calls `counters.record_call` *before* `model.propose`, so a
    raising model must still leave `model_calls_used == 2` in the final
    report, not 1 -- the second call was spent even though it never
    returned."""
    paths = RunPaths(root=tmp_path)
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)
    graph_model = ReplayToolCallingModel(
        RaisesOnSecondCall(ReplayReasoningModel(FIXTURE, substitutions=subs))
    )

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.FAILED_SAFE,
        root_cause=RootCauseCode.UNDETERMINED,
        tools_executed=1,
        model_calls_used=2,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "67b92dc34438566a373fb89780377510388271e973c993209fd6e35ac63cab5d"
        ),
        evidence_ids=(SYMPTOM_EVIDENCE_ID, "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"),
        receipt_ids=("00000000000000000000000000000002",),
    )
    assert [receipt_shape(r) for r in result.receipts] == [
        (ToolName.QUERY_LOGS, "ALLOWED", "SETTLED", "UNAVAILABLE", "TOOL_UNAVAILABLE")
    ]
    assert dispatch_events(recorder) == [
        ("proposal_received", {"tool": "query_logs"}),
        ("check_started", {"tool": "query_logs"}),
        ("check_finished", {"outcome": "UNAVAILABLE", "duration_ms": 0}),
    ]


def test_the_graph_reproduces_the_frozen_report_for_two_executed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lab_diagnosis.json` proposes `query_logs` then `list_recent_changes`:
    two executed checks across two tools, the only scenario here that is not
    a single-`query_logs` script -- proof that the three tools Unit 1c
    wrapped agreed with the loop too, not just that the shape held when
    `query_logs` was the only tool involved."""
    paths = RunPaths(root=tmp_path)
    write_orders_error_row(paths)
    write_changes(paths, [change_row(1)])
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)
    graph_model = ReplayToolCallingModel(
        ReplayReasoningModel(LAB_DIAGNOSIS_FIXTURE, substitutions=subs)
    )

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert [request.stage.value for request in graph_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.CONFIG_CHANGE,
        tools_executed=2,
        model_calls_used=3,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "44e5043842b3e3701b183c4b995d8d7e1935021daaba017c8321d0fff4fc802b"
        ),
        evidence_ids=(
            SYMPTOM_EVIDENCE_ID,
            "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d",
            "00000000000000000000000000000003",
            "00000000000000000000000000000005",
        ),
        receipt_ids=(
            "00000000000000000000000000000002",
            "00000000000000000000000000000004",
        ),
    )
    assert [receipt_shape(r) for r in result.receipts] == [
        (ToolName.QUERY_LOGS, "ALLOWED", "SETTLED", "EXECUTED", None),
        (ToolName.LIST_RECENT_CHANGES, "ALLOWED", "SETTLED", "EXECUTED", None),
    ]
    assert dispatch_events(recorder) == [
        ("proposal_received", {"tool": "query_logs"}),
        ("check_started", {"tool": "query_logs"}),
        ("check_finished", {"outcome": "EXECUTED", "duration_ms": 0}),
        ("proposal_received", {"tool": "list_recent_changes"}),
        ("check_started", {"tool": "list_recent_changes"}),
        ("check_finished", {"outcome": "EXECUTED", "duration_ms": 0}),
    ]
    assert [record.kind for record in result.evidence] == [
        EvidenceKind.SYMPTOM,
        EvidenceKind.TOPOLOGY,
        EvidenceKind.LOG,
        EvidenceKind.CHANGE,
    ]
