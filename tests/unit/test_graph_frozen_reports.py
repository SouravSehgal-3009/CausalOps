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
  `RunRecorder` produced, as ordered `(name, fields)` pairs -- except
  `check_finished`'s `duration_ms`, dropped for the reason below.
  `proposal_denied`/`check_finished` also carry `proposal_turn` and
  `receipt_id` as of lab-defect-fix Unit 1 (W11, see the note below);
  both are pinned like every other field here, not excluded, since the
  counting id generator two paragraphs down makes `receipt_id`
  deterministic the same way it already does for `evidence_ids`/
  `receipt_ids` above.
- Wall-clock fields (`latency_ms`, `started_at`, `finished_at`) are excluded,
  as they always were: nothing about a frozen literal makes `StepClock`
  readings meaningful to pin. `check_finished`'s `duration_ms` joined that
  exclusion after this docstring already claimed it, not before: this
  paragraph said "wall-clock fields ... are excluded" while six literals a
  few hundred lines below pinned `duration_ms` to an exact `0`, which is a
  wall-clock field by the same definition -- `executed_check`
  (`evidence.py:92`) measures it with `time.monotonic()` around the real
  backend call, not with the injected `StepClock`. CI caught the gap: the
  branch run for `e6eb574` measured 15 ms on Windows and failed, the merge
  run for `d6f06cd`, an identical tree, measured 0 ms on Linux and passed.
  `0` was never a frozen fact about the graph; it was an assertion that the
  test machine is fast. This file's own stated contract is why fixing it
  here is legitimate rather than a quiet literal update -- see
  `TECHNICAL_OVERVIEW.md`'s Milestone 2 section for the full record. No
  other literal moved at that unit: ids, digests, disposition, receipt
  shapes and evidence kinds were unaffected, and
  `test_a_simulated_slow_machine_still_matches_the_frozen_report` below
  reproduces the Windows measurement directly so the fix is provable on any
  machine, not just a slow one.

**Unit 3a moves every `final_context_digest` literal below, and nothing
else.** `SYSTEM_TEXT` (`prompts.py`) gained one sentence distinguishing
runbook guidance from incident evidence, so the model correctly cites
retrieved passages separately once `search_runbooks` exists. The digest is
`digest_text(system_text + context_text + repair_errors)`
(`_render_stage_request`), and `system_text` is identical across every
stage in every scenario here, so that one sentence shifts every digest in
this file -- a real, named reason, not drift. None of the five scenarios
below ever proposes `search_runbooks` (confirmed: no fixture script
references it), so ids, disposition, receipt shapes, evidence kinds, event
vocabulary and `duration_ms` are all unaffected -- only the digest moved.

**Post-review strict-schema round moves every `final_context_digest`
literal below a second time, and nothing else** -- the same mechanism as
Unit 3a's own note above. `SYSTEM_TEXT` (`prompts.py`) gained one sentence
telling the model to answer a tool call with no accompanying narrative
text, closing a real risk `live_model.py`'s `_has_visible_content` guard
otherwise leaves open (a stray sentence beside a genuine tool call would
burn the run's one repair slot). `system_text` is identical across every
stage in every scenario here and feeds directly into
`_render_stage_request`'s digest, so that one sentence shifts every digest
in this file again, confirmed by running the suite before and after and
updating only the six `final_context_digest` literals to match -- no other
field in this file changed.

**A later round moves every `final_context_digest` literal below a third
time, and nothing else** -- the same mechanism as the two notes above.
The sentence the previous round added was ambiguous: "respond with the
tool call alone" reads naturally either as "no narrative text alongside
the tool call" (the intended meaning) or as "make only one tool call"
(wrong -- `live_model.py` requires exactly one native call on every
INITIAL_PLAN/HYPOTHESIS_UPDATE turn). Reworded to "do not add
narrative text, explanation, or commentary outside the tool call's own
fields," which keeps the original intent and removes the cardinality
reading. `system_text` is identical across every stage in every scenario
here and feeds directly into `_render_stage_request`'s digest, so the
reworded sentence shifts every digest in this file again, confirmed by
running the suite before and after and updating only the six
`final_context_digest` literals to match -- no other field in this file
changed.

**Lab-defect-fix Unit 1 (W11) adds two fields to `proposal_denied` and
`check_finished`, and moves no digest.** `dispatch_tool` (`graph.py`) now
stamps both events with `proposal_turn` (the zero-based `investigate` turn
that produced the dispatched proposal) and `receipt_id` (the settled or
denied receipt's own id) -- the join key back to `investigate`'s own new
`proposal_recorded` event and to `receipts.jsonl`, added so a saved run can
answer "which window did this query use, and what did the model expect to
see" from `events.jsonl`/`receipts.jsonl` alone. Neither `SYSTEM_TEXT` nor
any tool schema changed, so `final_context_digest` is unaffected in every
scenario below -- confirmed by running the suite before and after: only the
six `dispatch_events(...)` literals moved, gaining `proposal_turn`/
`receipt_id` on their `proposal_denied`/`check_finished` entries, with
values read directly off each scenario's own already-deterministic id
sequence (`_install_counting_ids` below) rather than guessed.

**Lab-defect-fix Unit 3 (W1/Q2) moves every `final_context_digest` literal
below a fourth time, moves the hardcoded `Versions` literal in
`assert_report_matches_frozen`, and changes the stage sequence and
`model_calls_used` for two of the five scenarios.** `SYSTEM_TEXT`
(`prompts.py`) gained one sentence on the window-clamp contract, so
`system_text` -- identical across every stage in every scenario here and
feeding directly into `_render_stage_request`'s digest -- shifts every
digest in this file again, the same mechanism as the three notes above.
`TOOL_REGISTRY_VERSION` and `POLICY_VERSION` both bumped too (`tools.py`'s
window fields became optional; `policy.py` gained the `UNRESOLVED_WINDOW`
guard), so `assert_report_matches_frozen`'s hardcoded `Versions(...)`
literal now reads `"3"`/`"3"`/`"3"` instead of `"2"`/`"2"`/`"2"`.

Separately, and not a digest effect: `route_after_normalize` dropped its
`model_turn < 2` term (Q2), so a denial that spends no check slot no longer
caps an investigation at two `INVESTIGATE` turns by itself -- only running
out of check slots, model-call headroom, wall clock, or `budgets.
model_calls` turns does. Two of the five scenarios below (`after_a_first_
turn_denial`, `after_a_repeated_proposal`) reach turn 1 with a check slot
and model-call headroom still free (turn 1's own proposal is `ALLOWED` but
comes back `TOOL_UNAVAILABLE`/is a duplicate denial -- neither spends the
one slot the OTHER of the pair already used), so the graph now asks a real
third `HYPOTHESIS_UPDATE` turn where it previously stopped; both scripts
gained a second `hypothesis_update` response (a stop reason, so no new
dispatch event) to answer it, `model_calls_used` moved from 3 to 4 for
both, and their `[request.stage.value for ...]` literal gained a second
`"hypothesis_update"` entry. Confirmed by running the suite before and
after: `receipt_ids`/`evidence_ids`/`dispatch_events(...)` are unaffected
for both -- the added turn produces no proposal, so it mints no receipt or
evidence id and stamps no new dispatch event.

**Lab-defect-fix Unit 4 (W18) moves every `final_context_digest` literal
below a fifth time, and moves the hardcoded `Versions` literal's
`prompt_version` field, and nothing else.** `SYSTEM_TEXT` (`prompts.py`)
gained one sentence stating the `CONFIG_CHANGE` label convention
(proximate mechanism, not trigger), the same mechanism as every digest
move above -- confirmed by running the suite before and after: only the
six `final_context_digest` literals and `assert_report_matches_frozen`'s
`prompt_version="3"` (now `"4"`) changed; `POLICY_VERSION`/
`TOOL_REGISTRY_VERSION` are untouched by this unit, so their two fields
in that same `Versions(...)` literal stay `"3"`/`"3"`. No scenario below
proposes `list_recent_changes` or reaches `FINAL_ASSESSMENT` with a
`CONFIG_CHANGE` root cause other than the two already-`CONFIG_CHANGE`
scenarios' existing frozen output, so stage sequence, ids, disposition,
receipt shapes, evidence kinds, and event vocabulary are all unaffected.

**A Codex review round on Unit 4 found the sentence above still let a
resource-pool or downstream-timeout condition triggered by a
configuration change be mislabeled `CONFIG_CHANGE`, since "proximate
reason requests are failing" alone did not say which of two competing
readings wins when a config change caused a still-live pool/timeout
condition.** `SYSTEM_TEXT` was rewritten to state the priority
explicitly: the more specific label wins whenever its own evidence is
present, and `CONFIG_CHANGE` applies only when no more specific label
fits. This moves every `final_context_digest` literal below a sixth
time and `assert_report_matches_frozen`'s `prompt_version="4"` (now
`"5"`), and nothing else -- confirmed the same way as every prior round:
`POLICY_VERSION`/`TOOL_REGISTRY_VERSION` stay `"3"`/`"3"`, and no
scenario below changes stage sequence, ids, disposition, receipt shapes,
evidence kinds, or event vocabulary.
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
    resume_graph_run,
    write_changes,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

import causalops.evidence as evidence_module
import causalops.graph as graph_module
import causalops.tool_wrappers as tool_wrappers_module
from causalops.domain import (
    Budgets,
    Disposition,
    EscalatedInvestigation,
    IncidentScope,
    InitialAlertPacket,
    InvestigationReport,
    InvestigationResult,
    RetrievalMode,
    RootCauseCode,
    ToolProposal,
    ToolReceipt,
    Versions,
)
from causalops.evidence import EvidenceKind
from causalops.graph import build_graph, run_graph_investigation
from causalops.models import (
    ModelRequest,
    ModelResponse,
    ReplayReasoningModel,
    ReplayToolCallingModel,
)
from causalops.prometheus import run_metric_check
from causalops.run_records import RunEvent, RunRecorder
from causalops.runbooks import RunbookIndex, run_runbook_search
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
    """The dispatch-vocabulary events, as ordered `(name, fields)` pairs --
    except `check_finished`'s `duration_ms`, dropped before comparison. See
    the module docstring above for why.
    `assert_check_finished_durations_are_measured` below checks the field's
    shape instead of pinning its value."""
    return [
        (event.name, _drop_duration(dict(event.fields)))
        for event in recorder.events
        if event.name in DISPATCH_EVENT_NAMES
    ]


def _drop_duration(fields: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in fields.items() if key != "duration_ms"}


def assert_check_finished_durations_are_measured(recorder: RunRecorder) -> None:
    """`dispatch_events` drops `check_finished`'s `duration_ms` before
    comparison (see its own docstring). This checks separately that every
    `check_finished` event still carries one: present, an `int`, and
    non-negative -- the same invariant `ToolReceipt.duration_ms` already
    enforces (`domain.py:293`), asserted again here because the event's
    `fields` dict is a plain, unvalidated `dict[str, JsonValue]`."""
    for event in recorder.events:
        if event.name != "check_finished":
            continue
        assert "duration_ms" in event.fields
        duration = event.fields["duration_ms"]
        assert isinstance(duration, int)
        assert duration >= 0


def run_once(
    graph_model: ReplayToolCallingModel,
    paths: RunPaths,
    monkeypatch: pytest.MonkeyPatch,
    budgets: Budgets | None = None,
) -> tuple[InvestigationResult, RunRecorder]:
    """Runs the graph orchestrator alone. `run_both`'s old name for this half
    is gone along with the loop half it used to run beside -- kept as one
    function, not inlined per test, since all five scenarios wire the same
    real four-tool registry the same way.

    Unit 2b: two of the five scenarios this function serves escalate (a
    `query_logs` receipt that comes back `UNAVAILABLE` because those two
    scripts never call `write_orders_error_row`, reaching a diagnosis
    anyway). Rather than weaken this file's frozen-literal pin for those
    two, this function resumes a pause with "accept" and returns the
    settled report -- proving an accepted escalation is report-preserving,
    which is a real claim worth the extra step, not a workaround. An
    explicit `checkpointer` is passed to `run_graph_investigation` only so
    this function can hold onto it to resume; `investigation_id` is left
    for `run_graph_investigation` to auto-mint exactly as before, since the
    id-counting harness below depends on that one call to `new_opaque_id()`
    still happening before any evidence/receipt id is minted -- an explicit
    `investigation_id` here would shift every id after it by one and break
    all five scenarios' frozen id literals, not just the two that escalate.
    """
    scope = incident_scope()
    packet = alert_packet()
    evidence_records = packet_evidence()
    resolved_budgets = budgets or Budgets()

    _install_counting_ids(monkeypatch)
    domain_clock = StepClock()
    graph_recorder = RunRecorder(StepClock())
    runbook_index = RunbookIndex()
    registry = dispatch_registry(
        run_metric=lambda arguments, scope: run_metric_check(
            arguments, scope, "http://unused", 1
        ),
        run_logs=lambda arguments, scope: run_logs_check(arguments, paths),
        run_changes=lambda arguments, scope: run_changes_check(arguments, paths),
        run_topology=lambda arguments, scope: run_topology_check(arguments, paths),
        # None of this file's five scenarios ever scripts `search_runbooks`
        # (confirmed: no fixture references it), so this is wired to the
        # same real backend `cli.py` uses -- this file's own module
        # docstring's claim, "the same real backends `cli.py` wires, not
        # spies" -- even though it is never actually invoked.
        run_search=lambda arguments, scope: run_runbook_search(
            arguments, runbook_index
        ),
    )
    checkpointer = InMemorySaver()
    graph_result = run_graph_investigation(
        scope,
        packet,
        evidence_records,
        graph_model,
        registry,
        graph_recorder,
        resolved_budgets,
        domain_clock,
        checkpointer=checkpointer,
    )
    if isinstance(graph_result, EscalatedInvestigation):
        compiled = build_graph(
            scope,
            packet,
            resolved_budgets,
            domain_clock,
            graph_model,
            registry,
            checkpointer,
            event_clock=graph_recorder.clock,
        )
        config: RunnableConfig = {"configurable": {"thread_id": graph_result.thread_id}}
        graph_result = resume_graph_run(compiled, config, "accept")
        graph_recorder.recorded = [
            RunEvent.model_validate(dump)
            for dump in compiled.get_state(config).values["events"]
        ]
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
    # Unit 3a, added on review: no scenario in this file ever dispatches
    # `search_runbooks`, so every one of these five reports must show
    # retrieval never ran. Correctness's M15 proved these were unguarded --
    # seeding `retrieval_mode` as `FTS5_LEXICAL` at `graph.py`'s
    # `initial_state` construction left the full suite green with no
    # assertion anywhere catching it, even though the resulting report would
    # then falsely claim retrieval that never happened on every run in the
    # project.
    assert report.retrieval_mode is RetrievalMode.DISABLED
    assert report.runbook_passage_ids == ()
    # Unit 3a, added on review: nothing else in this file pinned `versions`
    # at all, so a future prompt/policy/tool-registry edit could ship
    # without its version bump and nothing here would notice -- the same
    # defect class as the `retrieval_mode` seed above, on a §10
    # reproducibility stamp. Deliberately hardcoded string literals, not a
    # comparison against the live `PROMPT_VERSION`/`POLICY_VERSION`/
    # `TOOL_REGISTRY_VERSION` constants: importing and comparing against
    # those would make this assertion move in lockstep with any future
    # change to them, which proves nothing -- confirmed by reverting
    # `PROMPT_VERSION` to `"1"` alone and getting 420 passed with no
    # constant-comparison version of this line to catch it.
    assert report.versions == Versions(
        schema_version="1",
        prompt_version="5",
        policy_version="3",
        tool_registry_version="3",
    )


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
            "bc0d229be4f04e14a6a735879f0c257882b72fe799ba34d16c48ab3ee5da255f"
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
        (
            "check_finished",
            {
                "proposal_turn": 0,
                "receipt_id": "00000000000000000000000000000002",
                "outcome": "EXECUTED",
            },
        ),
    ]
    assert_check_finished_durations_are_measured(recorder)
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
    *proposed* something, not on whether it was *allowed*.

    Lab-defect-fix Unit 3/Q2: turn 1's proposal is `ALLOWED` (it spends a
    slot even though its outcome comes back `TOOL_UNAVAILABLE`, since
    `ReservationLedger.slots_left()` counts by `policy_result`, not
    `outcome`), but a slot and model-call headroom both still remain
    afterward with `model_turn < 2` gone, so the graph now asks a third
    `HYPOTHESIS_UPDATE` turn. The added response gives a stop reason, which
    adds a fourth model call but no new dispatch event (no proposal to
    dispatch that turn)."""
    paths = RunPaths(root=tmp_path)
    script = {
        "initial_plan": [plan_json(proposal=out_of_scope_logs_proposal())],
        "hypothesis_update": [
            plan_json(proposal=logs_proposal()),
            plan_json(stop_reason="nothing safe left to check"),
        ],
        "final_assessment": [assessment_json()],
    }
    graph_model = ReplayToolCallingModel(replay_model(tmp_path, script))

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert [request.stage.value for request in graph_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
        tools_executed=1,
        model_calls_used=4,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "f057cde86d48a70e53d182d17cb9b5c7e6b40e5371637e80bf55d53bb86e38d6"
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
            {
                "proposal_turn": 0,
                "receipt_id": "00000000000000000000000000000002",
                "reason": "UNKNOWN_SERVICE",
                "message": "that service is out of scope",
            },
        ),
        ("proposal_received", {"tool": "query_logs"}),
        ("check_started", {"tool": "query_logs"}),
        (
            "check_finished",
            {
                "proposal_turn": 1,
                "receipt_id": "00000000000000000000000000000003",
                "outcome": "UNAVAILABLE",
            },
        ),
    ]
    assert_check_finished_durations_are_measured(recorder)


def test_the_graph_reproduces_the_frozen_report_after_a_repeated_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-1's other regression, pinned at the orchestrator level: the same
    proposal scripted verbatim for both turns. Turn 0 executes; turn 1's
    proposal fingerprints identically and is denied as a duplicate.

    Lab-defect-fix Unit 3/Q2: with `model_turn < 2` gone, one duplicate
    denial (no slot spent) no longer stops the run at two turns as long as
    a slot and model-call headroom remain -- the router asks a third
    `HYPOTHESIS_UPDATE` turn rather than the phantom-third-turn concern this
    test's own name once referred to (`route_after_normalize`'s job now is
    to bound genuine additional turns, not to forbid a third ask outright).
    The added response gives a stop reason, adding a fourth model call but
    no new dispatch event."""
    paths = RunPaths(root=tmp_path)
    repeated = logs_proposal()
    script = {
        "initial_plan": [plan_json(proposal=repeated)],
        "hypothesis_update": [
            plan_json(proposal=repeated),
            plan_json(stop_reason="nothing safe left to check"),
        ],
        "final_assessment": [assessment_json()],
    }
    graph_model = ReplayToolCallingModel(replay_model(tmp_path, script))

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert [request.stage.value for request in graph_model.requests] == [
        "initial_plan",
        "hypothesis_update",
        "hypothesis_update",
        "final_assessment",
    ]
    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
        tools_executed=1,
        model_calls_used=4,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "f057cde86d48a70e53d182d17cb9b5c7e6b40e5371637e80bf55d53bb86e38d6"
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
        (
            "check_finished",
            {
                "proposal_turn": 0,
                "receipt_id": "00000000000000000000000000000002",
                "outcome": "UNAVAILABLE",
            },
        ),
        ("proposal_received", {"tool": "query_logs"}),
        (
            "proposal_denied",
            {
                "proposal_turn": 1,
                "receipt_id": "00000000000000000000000000000003",
                "reason": "DUPLICATE_PROPOSAL",
                "message": "this check was proposed already",
            },
        ),
    ]
    assert_check_finished_durations_are_measured(recorder)


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
            "66751c77d0ff62b0480ad3857e4bf5d8382dd27195c46e7e9f7735f0d5df96a3"
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
        (
            "check_finished",
            {
                "proposal_turn": 0,
                "receipt_id": "00000000000000000000000000000002",
                "outcome": "UNAVAILABLE",
            },
        ),
    ]
    assert_check_finished_durations_are_measured(recorder)


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
            "cd113fafea822d8e09caca9f141ddb92758a2bf80cf2e523d3a3243a5cb951b8"
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
        (
            "check_finished",
            {
                "proposal_turn": 0,
                "receipt_id": "00000000000000000000000000000002",
                "outcome": "EXECUTED",
            },
        ),
        ("proposal_received", {"tool": "list_recent_changes"}),
        ("check_started", {"tool": "list_recent_changes"}),
        (
            "check_finished",
            {
                "proposal_turn": 1,
                "receipt_id": "00000000000000000000000000000004",
                "outcome": "EXECUTED",
            },
        ),
    ]
    assert_check_finished_durations_are_measured(recorder)
    assert [record.kind for record in result.evidence] == [
        EvidenceKind.SYMPTOM,
        EvidenceKind.TOPOLOGY,
        EvidenceKind.LOG,
        EvidenceKind.CHANGE,
    ]


def test_a_simulated_slow_machine_still_matches_the_frozen_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows CI measured 15 ms for the same backend call that reads back
    0 ms on a fast Linux disk (`e6eb574`'s branch run vs `d6f06cd`'s merge
    run, an identical tree). Nothing in this file forced a slow measurement
    before this test, which is why the flake stayed latent through every
    unit since 1c: every run happened to be fast enough to truncate to `0`.

    This monkeypatches `time.monotonic` to advance 15 ms on every read --
    reproducing Windows' exact measurement on whatever machine runs this
    suite -- and reruns the two-executed-check scenario. The frozen
    comparison must still pass, and each `check_finished` event must still
    carry a real (non-zero, here) duration: proof the fix does not depend on
    the backend being fast, not just an observation that it happened to be
    fast so far."""
    paths = RunPaths(root=tmp_path)
    write_orders_error_row(paths)
    write_changes(paths, [change_row(1)])
    scope = incident_scope()
    packet = alert_packet()
    subs = substitutions(scope, packet)
    graph_model = ReplayToolCallingModel(
        ReplayReasoningModel(LAB_DIAGNOSIS_FIXTURE, substitutions=subs)
    )

    reading = 0.0

    def slow_machine_monotonic() -> float:
        nonlocal reading
        reading += 0.015
        return reading

    # Patched on the `time` module itself, not on a `causalops.evidence`-only
    # binding: `evidence_module.time is time`, the same module object
    # `telemetry.py:106/161/205` reads `started = time.monotonic()` from
    # before handing it to `executed_check` here. Patching a narrower target
    # (e.g. an attribute on `evidence_module` alone) would leave those
    # `started` reads on the real clock while `executed_check`'s own call
    # used the patched one, producing a negative delta instead of 15 ms.
    monkeypatch.setattr(evidence_module.time, "monotonic", slow_machine_monotonic)

    result, recorder = run_once(graph_model, paths, monkeypatch)

    assert_report_matches_frozen(
        result.report,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.CONFIG_CHANGE,
        tools_executed=2,
        model_calls_used=3,
        repairs_used=0,
        invalid_responses=0,
        final_context_digest=(
            "cd113fafea822d8e09caca9f141ddb92758a2bf80cf2e523d3a3243a5cb951b8"
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
    assert dispatch_events(recorder) == [
        ("proposal_received", {"tool": "query_logs"}),
        ("check_started", {"tool": "query_logs"}),
        (
            "check_finished",
            {
                "proposal_turn": 0,
                "receipt_id": "00000000000000000000000000000002",
                "outcome": "EXECUTED",
            },
        ),
        ("proposal_received", {"tool": "list_recent_changes"}),
        ("check_started", {"tool": "list_recent_changes"}),
        (
            "check_finished",
            {
                "proposal_turn": 1,
                "receipt_id": "00000000000000000000000000000004",
                "outcome": "EXECUTED",
            },
        ),
    ]
    assert_check_finished_durations_are_measured(recorder)
    measured = [
        event.fields["duration_ms"]
        for event in recorder.events
        if event.name == "check_finished"
    ]
    assert measured == [15, 15]
