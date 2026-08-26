"""The LangGraph orchestrator: one replay incident run end to end through the
policy-wrapped dispatch node `tool_wrappers.py` built and nothing else consumed
before this module.

Unit 1a built this beside `workflow.py`'s `Investigation` loop, unchanged, so
an owner could run either path against the same incident and compare --
`TECHNICAL_SPEC.md` §12 calls this bounded tool-graph parity. Unit 1d-1
demonstrated that parity with a 144-pair differential sweep across 13
dimensions; Unit 1d-2 then retired the loop, `workflow.py` and `cli.py`'s
`--orchestrator` flag included. This file is now the only orchestrator.

Graph state is a JSON-only `TypedDict`: nothing here lives off-state.
`tool_wrappers.py`'s `ReservationLedger` and `evidence.py`'s `EvidenceStore`
are both rebuilt fresh, inside a node, from state's `receipts`/`evidence`
lists on every call that needs them -- there is no live object surviving
between graph turns, so Milestone 2's SQLite-checkpointed resume needs no
redesign of anything in this file. Unit 2a closes the one exception to that
claim: `RunRecorder` used to be a factory-closure object shared and mutated
across every node call, which would have lost every recorded event at a
process boundary. Events are now a `state["events"]` list, rebuilt into a
local `RunRecorder` by `_rebuild_recorder` at the top of each node exactly as
`_rebuild_receipts` already did for receipts, and written back whole in that
node's return -- the same "read the full field, extend it, return the full
field" pattern `dispatch_tool` already used for `evidence`. Every node
factory that touches the recorder takes two `Clock` parameters, not one:
`clock` times domain data and `event_clock` times `RunEvent.at` only, kept
apart so recording an event never perturbs a domain-timing read -- see
`build_graph`'s docstring for the full reasoning.

Two nodes decide whether the run continues, stops safely, or asks for a
diagnosis; `final_report` never makes that decision, it only serializes
whatever `investigate`/`dispatch_tool`/`final_assessment` already decided,
the same separation `workflow.py`'s `Investigation.report()` kept between
deciding an outcome and writing it down.

Every node that can raise mid-attempt (`investigate`, `dispatch_tool`,
`final_assessment`) wraps its own body in `try`/`except`: LangGraph gives a
crashed node no way to hand back the values it accumulated before the raise,
so a node that lets an exception escape loses them, not just the exception's
caller. `dispatch_tool` closes this for a reserved tool receipt;
`investigate`/`final_assessment` close the same gap for the model-call
budget, via `_StageCounters`/`_ask_with_repair` below. `GraphBubbleUp` is
re-raised first in all three, since it is a control-flow signal (interrupt,
drain, parent command), not a failure -- Milestone 2 adds `interrupt()`.

This file binds `ToolCallingModel` -- the `propose()`/`respond()` protocol in
`models.py` -- not the plain `ReasoningModel` protocol `workflow.py` used.
`ReplayToolCallingModel` is its first implementation; a live Claude adapter is
the second, which is what makes this a protocol worth naming rather than a
concrete type these nodes bind directly.
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, TypedDict, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import JsonValue

from causalops.cost_ledger import AmbiguousReservationNotResent, CostCeilingExceeded
from causalops.domain import (
    DEFAULT_BUDGETS,
    REPLAY_MODEL_NAME,
    Budgets,
    Clock,
    Disposition,
    EscalatedInvestigation,
    EscalationReason,
    EscalationRecord,
    Evidence,
    FinalAssessment,
    GraphPhase,
    HypothesisUpdate,
    IncidentScope,
    InitialAlertPacket,
    InitialPlan,
    InvestigationReport,
    InvestigationResult,
    ModelDisposition,
    ModelUsage,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RetrievalMode,
    RootCauseCode,
    RunbookPassage,
    ToolOutcome,
    ToolProposal,
    ToolReceipt,
    Versions,
    utc_now,
)
from causalops.evidence import EvidenceStore, digest_text, new_opaque_id
from causalops.models import ModelRequest, Stage, ToolCallingModel, parse_response
from causalops.policy import POLICY_VERSION
from causalops.pricing import InputTooLarge
from causalops.prompts import (
    PROMPT_VERSION,
    STAGE_INSTRUCTIONS,
    SYSTEM_TEXT,
    DeniedCheckNote,
    denial_guidance,
    render_context,
)
from causalops.run_records import RunEvent, RunRecorder
from causalops.tool_calls import (
    parse_tool_call,
    select_single_tool_call,
)
from causalops.tool_wrappers import ReservationLedger, ToolWrapper
from causalops.tools import TOOL_REGISTRY_VERSION, ToolName

# The library default is 10007 (`_internal/_config.py:32`), meant for graphs
# whose shape isn't known ahead of time. This graph's longest real path is
# nine supersteps (investigate, dispatch, normalize, investigate, dispatch,
# normalize, final_assessment, escalation_interrupt, final_report); 25 leaves
# headroom for a guard bug without letting a real one spin for thousands of
# steps first. A resumed run does not add to this count: it is a second,
# separate `.invoke()` call with its own fresh recursion budget, not more
# steps stacked onto the first.
GRAPH_RECURSION_LIMIT = 25


class GraphState(TypedDict):
    """A JSON-only projection of one investigation. Every field round-trips
    through `model_dump(mode="json")`/`model_validate` losslessly, so nothing
    is lost by holding domain records as plain dicts here instead of live
    objects -- see `test_a_receipt_round_trips_through_json_without_losing_fidelity`
    in `test_tool_wrappers.py` for the same claim proven at the receipt level.
    """

    investigation_id: str
    # `TECHNICAL_SPEC.md:140-142` requires this as a distinct, immutable field
    # alongside `investigation_id`/`thread_id`. In Unit 2a they are minted
    # together and never diverge -- there is no resume path yet for them to
    # diverge across. The distinction becomes load-bearing in Unit 2b, where
    # the model-request idempotency key is `run_id + graph_phase +
    # model_turn` (`TECHNICAL_SPEC.md:155-158`); this field exists now so
    # that key has something stable to name.
    run_id: str
    incident_id: str
    # Unit 3b-2. Set once, in `run_graph_investigation`'s `initial_state`,
    # from a caller-supplied name (or `REPLAY_MODEL_NAME` if the caller does
    # not pass one -- every pre-3b-2 test call site). Never written by any
    # node after that. This is what fixes `cli.py`'s resume path: a resumed
    # thread used to relabel its own artifact `REPLAY_MODEL_NAME`
    # unconditionally, because nothing durable said which model actually
    # produced the original investigation. `cli.py` reads this back from the
    # checkpoint's `channel_values` the same way it already reads
    # `incident_id` back (`_resolve_thread_incident_id`), with a
    # `REPLAY_MODEL_NAME` default so a checkpoint written before this field
    # existed still resumes and is labelled correctly (it can only ever have
    # been a replay run).
    model_name: str
    phase: str
    model_turn: int
    model_calls_used: int
    repairs_used: int
    invalid_responses: int
    started_at: str
    context_digest: str
    seen_fingerprints: list[str]
    receipts: list[dict[str, JsonValue]]
    evidence: list[dict[str, JsonValue]]
    # Milestone 3, Unit 3a. `runbook_passages` mirrors `evidence`'s own
    # "full list, rebuilt and extended, written back whole" pattern
    # (`_rebuild_passages` below), but a `RunbookPassage` dump, never an
    # `Evidence` one -- retrieved guidance never enters `evidence`, the
    # structural claim this whole unit exists to keep true. `retrieval_mode`
    # starts `disabled` and is set once a `search_runbooks` proposal is
    # actually reserved and dispatched, from the backend's own
    # configuration -- never inferred from whether `runbook_passages` ends
    # up non-empty. See `RetrievalMode`'s own docstring for why.
    runbook_passages: list[dict[str, JsonValue]]
    retrieval_mode: str
    # Every event any node has recorded so far, in order. A node rebuilds a
    # local `RunRecorder` from this list (`_rebuild_recorder`), records its
    # own events into that local copy, and returns the full extended list --
    # the same pattern `receipts`/`evidence` already use. See the module
    # docstring for why this replaced a shared closure-captured recorder.
    events: list[dict[str, JsonValue]]
    pending_proposal: dict[str, JsonValue] | None
    assessment: dict[str, JsonValue] | None
    usage: dict[str, JsonValue] | None
    failure_reason: str | None
    # Unit 2b, extended in 2c. All three are `None` together on every run
    # that never reached `escalation_interrupt`, and all three are set
    # together once the node returns -- `rejection_note` only actually
    # holds text when `escalation_decision == "reject"`; it is `None`
    # alongside the other two on every accept, matching
    # `EscalationRecord.check_rejection_note_pairing`. A node's return
    # value only lands in state when the node returns -- never on the
    # interrupted attempt, which raises instead of returning -- so these
    # cannot go through state as "reason known, decision pending": by the
    # time any of the three is visible here, the run has already resumed
    # and settled. `_build_report` reads them together for that reason.
    escalation_reason: str | None
    escalation_decision: str | None
    rejection_note: str | None
    report: dict[str, JsonValue] | None


class GraphNode(Protocol):
    """The shape `StateGraph.add_node` actually wants. A plain
    `Callable[[GraphState], dict[str, Any]]` type alias does not structurally
    match `add_node`'s generic `_Node[NodeInputT]` protocol under strict
    mypy -- a known limitation of `Callable[[X], Y]` aliases against a
    generic `Protocol.__call__` -- so the five node factories below return
    this named protocol instead."""

    def __call__(self, state: GraphState) -> dict[str, Any]: ...


def _rebuild_store(
    incident_id: str, evidence_dumps: Sequence[dict[str, JsonValue]]
) -> EvidenceStore:
    store = EvidenceStore(incident_id)
    for dump in evidence_dumps:
        store.add(Evidence.model_validate(dump))
    return store


def _rebuild_receipts(state: GraphState) -> list[ToolReceipt]:
    """Rebuild every receipt this thread has recorded so far.

    Lab-defect-fix Unit 1, W16: asserts every reconstructed receipt's
    `incident_id` agrees with this thread's own `state["incident_id"]`.
    `thread_id` *is* `investigation_id` (`run_graph_investigation` passes
    `investigation_id` as LangGraph's own `thread_id`) -- a distinct field
    from `incident_id`, not the same identifier under two names. What
    actually makes the invariant hold: `state["incident_id"]` is set once,
    at investigation creation, from the `IncidentScope` of the one incident
    this investigation was built against (`initial_state`, this module), and
    every receipt this codebase can produce is stamped
    `incident_id=scope.incident_id` at reserve/deny time (`tool_wrappers.py`)
    from that same state -- so within one investigation this can never
    actually disagree. `cli.py`'s `_load_verified_incident` is what verifies
    that scope before an investigation is ever started or resumed, closing
    the one place an operator could otherwise smuggle in a mismatched scope.
    This is defence-in-depth against a hand-edited or otherwise corrupted
    checkpoint DB, not a fix for a reachable cross-incident leak -- see
    `LAB_DEFECTS_FIX_PLAN.md` §2.2 for the full trace. Raises loudly rather
    than silently dropping the offending receipt: a dropped receipt would
    hand back a check slot that was actually spent, which is a worse failure
    than refusing to proceed.
    """
    receipts = [ToolReceipt.model_validate(dump) for dump in state["receipts"]]
    incident_id = state["incident_id"]
    for receipt in receipts:
        if receipt.incident_id != incident_id:
            raise AssertionError(
                f"receipt {receipt.receipt_id} has incident_id "
                f"{receipt.incident_id!r}, but this thread's own state "
                f"incident_id is {incident_id!r} -- a corrupted checkpoint, "
                "not a reachable cross-incident leak (see "
                "LAB_DEFECTS_FIX_PLAN.md §2.2)"
            )
    return receipts


def _proposal_turn(state: GraphState) -> int:
    """The zero-based `investigate` turn that produced `state["pending_
    proposal"]`, canonical name `proposal_turn` (lab-defect-fix Unit 1).

    `investigate` reads `turn_index = state["model_turn"]` (zero-based, and
    the value that selects `Stage`/schema) and returns `"model_turn":
    turn_index + 1` in the same update that sets `pending_proposal` --
    `dispatch_tool` always runs after that return has landed, so by the time
    this is called `state["model_turn"]` is one turn ahead of the turn that
    actually produced the pending proposal. Subtracting 1 recovers
    `investigate`'s own `turn_index`, so `dispatch_tool`'s events agree with
    `investigate`'s own `proposal_recorded` event on the same number for the
    same turn -- the join key evaluation needs to answer "which proposal
    produced this receipt" from `events.jsonl` alone. Safe because
    `model_turn` is written in exactly one place (`investigate`'s return) and
    `stopped_state` deliberately omits it -- a stopped turn neither
    increments `model_turn` nor leaves a `pending_proposal`, so this is only
    ever called when the invariant `state["model_turn"] == producing
    turn_index + 1` holds.
    """
    return state["model_turn"] - 1


def _rebuild_passages(state: GraphState) -> list[RunbookPassage]:
    """Milestone 3, Unit 3a. `state["runbook_passages"]`'s live-object
    counterpart, the same "rebuild from state's own dump list" pattern
    `_rebuild_receipts` already uses one line up."""
    return [RunbookPassage.model_validate(dump) for dump in state["runbook_passages"]]


def _rebuild_recorder(state: GraphState, clock: Clock) -> RunRecorder:
    """A `RunRecorder` seeded with every event already in state, so the next
    `.event(...)` call continues the same `sequence` numbering instead of
    starting over. Takes whole state, the same shape `_rebuild_receipts`
    takes, plus the one thing state itself cannot carry: a clock is a
    runtime dependency, not domain data. `RunRecorder.recorded` is a plain
    public list -- assigning it here is the same kind of reconstruction
    `_rebuild_receipts` does through `ToolReceipt.model_validate`, just
    without needing a dedicated "from events" constructor on `RunRecorder`
    itself."""
    recorder = RunRecorder(clock)
    recorder.recorded = [RunEvent.model_validate(dump) for dump in state["events"]]
    return recorder


def _dump_events(recorder: RunRecorder) -> list[dict[str, JsonValue]]:
    return [event.model_dump(mode="json") for event in recorder.events]


def _tools_left(receipts: Sequence[ToolReceipt], budgets: Budgets) -> int:
    return ReservationLedger.from_receipts(
        receipts, budgets.executed_tools
    ).slots_left()


def _model_calls_left(model_calls_used: int, budgets: Budgets) -> int:
    return budgets.model_calls - model_calls_used


def _denied_check_notes(
    receipts: Sequence[ToolReceipt], budgets: Budgets
) -> tuple[DeniedCheckNote, ...]:
    """Fix F2. Every denied receipt accumulated so far, rendered into a
    `DeniedCheckNote` the model can actually read -- filtered over *all*
    receipts, not just the most recent one, because each context render is
    a stateless re-render with no conversation history: a turn-0 denial
    must still appear in `final_assessment`'s own context if the model
    never got a later turn to address it."""
    notes = []
    for receipt in receipts:
        if receipt.policy_result is not PolicyResult.DENIED:
            continue
        assert receipt.reason_code is not None, (
            "a DENIED receipt always carries a reason_code -- "
            "tool_wrappers._denied_receipt is the only production site that "
            "builds one, and it copies reason_code straight from a "
            "PolicyDecision whose own check_reason_code validator already "
            "refused to construct a DENIED decision without one. ToolReceipt "
            "has no such validator of its own, and _rebuild_receipts "
            "rehydrates receipts from checkpoint state, so this stays a real "
            "tripwire against a corrupted checkpoint rather than a "
            "restatement of a type guarantee -- same posture as W16's own "
            "identity check."
        )
        notes.append(
            DeniedCheckNote(
                tool=receipt.tool,
                reason_code=receipt.reason_code,
                guidance=denial_guidance(receipt.tool, receipt.reason_code, budgets),
            )
        )
    return tuple(notes)


def _money_refusal_reason_code(
    refusal: CostCeilingExceeded | InputTooLarge | AmbiguousReservationNotResent,
) -> ReasonCode:
    """The three ways `_send` (`live_model.py`) can refuse a request
    *before* sending it, mapped to the specific `ReasonCode` an owner reads
    in a report -- shared by `investigate`'s and `final_assessment`'s
    identical except-blocks below, rather than repeating a three-way
    `isinstance` chain in both (the exact "same fix, two places, one
    missed" shape this unit's own investigation kept finding elsewhere).

    Post-freeze review, P3-5: the third branch used to be a bare
    fall-through (`return ReasonCode.AMBIGUOUS_MODEL_REQUEST` with no
    check at all) -- correct today, since the parameter type is a
    three-member union and every member is covered above, but a future
    FOURTH refusal type added to that union without also updating this
    function would silently misreport as `AMBIGUOUS_MODEL_REQUEST`,
    with no error telling anyone the mapping is now wrong. The explicit
    `isinstance` below plus the `raise` after it turns that into a loud
    failure instead of a silent misreport."""
    if isinstance(refusal, CostCeilingExceeded):
        return ReasonCode.COST_CEILING_EXCEEDED
    if isinstance(refusal, InputTooLarge):
        return ReasonCode.INPUT_TOKEN_CAP_EXCEEDED
    if isinstance(refusal, AmbiguousReservationNotResent):
        return ReasonCode.AMBIGUOUS_MODEL_REQUEST
    raise AssertionError(f"unhandled refusal: {type(refusal).__name__}")


def _expired(started_at: str, budgets: Budgets, clock: Clock) -> bool:
    started = datetime.fromisoformat(started_at)
    return (clock() - started).total_seconds() > budgets.wall_clock_seconds


def _accumulate_usage(
    existing: dict[str, JsonValue] | None, latest: ModelUsage | None
) -> dict[str, JsonValue] | None:
    """Mirrored `workflow.py`'s `add_usage`: the report publishes the total
    across every model call, not the last one."""
    if latest is None:
        return existing
    if existing is None:
        return latest.model_dump(mode="json")
    total = ModelUsage.model_validate(existing)
    combined = ModelUsage(
        input_tokens=total.input_tokens + latest.input_tokens,
        output_tokens=total.output_tokens + latest.output_tokens,
    )
    return combined.model_dump(mode="json")


def _unresolved_runbook_citations(
    assessment: FinalAssessment | None, passages: Sequence[RunbookPassage]
) -> tuple[str, ...]:
    """Milestone 3, Unit 3a. Cited `passage_id`s the model wrote that this
    run never actually retrieved -- a forged runbook citation, the same
    threat §9's "Forged citations" row names, applied to guidance instead
    of evidence.

    Deliberately non-fatal, and *not* `ReasonCode.FORGED_EVIDENCE_REFERENCE`
    -- both reviewers rejected that shape during the owner's review of this
    unit's pre-edit report, and the reasoning is load-bearing enough to
    repeat here: `FORGED_EVIDENCE_REFERENCE` nulls the assessment
    (`final_assessment`'s own `failed_state`), which turns a correct,
    fully evidence-backed `DIAGNOSED` result into `FAILED_SAFE` /
    `UNDETERMINED` over a citation that provably cannot affect whether the
    diagnosis is right -- `evaluation.py`'s `diagnosis_correct` reads
    `report.root_cause`, and a `RunbookPassage` can never support or
    contradict a root cause (§6). Failing the run would be a systematic
    confound in exactly the retrieval-vs-no-retrieval comparison Milestone
    3 exists to measure, introduced by the unit creating the treatment arm.
    Called from `_build_report`, not `final_assessment`: `final_assessment`
    needs no change at all, and the assessment's own bytes stay exactly
    what the model wrote (`_build_report` never reconstructs or strips
    `FinalAssessment` -- see its own call site below).
    """
    if assessment is None or not assessment.runbook_citations:
        return ()
    known = {passage.passage_id for passage in passages}
    return tuple(
        passage_id
        for passage_id in assessment.runbook_citations
        if passage_id not in known
    )


def _build_report(
    state: GraphState,
    budgets: Budgets,
    clock: Clock,
    *,
    force_failure_reason: ReasonCode | None = None,
) -> InvestigationReport:
    """The one place an `InvestigationReport` is assembled from graph state --
    used by the `final_report` node for the normal path and, with
    `force_failure_reason` set, by `run_graph_investigation`'s outer
    containment when `invoke()` raised and no node ever reached
    `final_report` at all.
    """
    receipts = _rebuild_receipts(state)
    store = _rebuild_store(state["incident_id"], state["evidence"])
    passages = _rebuild_passages(state)
    started_at = datetime.fromisoformat(state["started_at"])
    finished_at = clock()
    usage = (
        ModelUsage.model_validate(state["usage"])
        if state["usage"] is not None
        else None
    )
    limitations: list[str] = (
        [] if usage is not None else ["this model reports no token usage"]
    )
    # Matches `ReservationLedger.slots_left()`'s own definition of "spent":
    # an ALLOWED receipt that is still RESERVED (a crash mid-dispatch, see
    # `dispatch_tool` below) already spent its slot but never completed, so
    # it is excluded here -- the report's "executed" count means "finished,"
    # not merely "attempted."
    tools_executed = sum(
        1
        for receipt in receipts
        if receipt.policy_result is PolicyResult.ALLOWED
        and receipt.state is ReceiptState.SETTLED
    )
    versions = Versions(
        prompt_version=PROMPT_VERSION,
        policy_version=POLICY_VERSION,
        tool_registry_version=TOOL_REGISTRY_VERSION,
    )

    assessment_dump = None if force_failure_reason is not None else state["assessment"]
    assessment: FinalAssessment | None = None
    if assessment_dump is not None:
        assessment = FinalAssessment.model_validate(assessment_dump)
        disposition = (
            Disposition.DIAGNOSED
            if assessment.disposition is ModelDisposition.DIAGNOSED
            else Disposition.INSUFFICIENT_EVIDENCE
        )
        root_cause = assessment.root_cause
        reason_code = None
    else:
        disposition = Disposition.FAILED_SAFE
        root_cause = RootCauseCode.UNDETERMINED
        reason_value = (
            force_failure_reason.value
            if force_failure_reason
            else state["failure_reason"]
        )
        reason_code = (
            ReasonCode(reason_value) if reason_value else ReasonCode.INTERNAL_ERROR
        )

    unresolved_runbook_citations = _unresolved_runbook_citations(assessment, passages)
    if unresolved_runbook_citations:
        limitations.append(
            f"{len(unresolved_runbook_citations)} cited runbook passage id(s) "
            "could not be resolved against what this run actually retrieved"
        )

    # Unit 2b. Both keys are `None` together on a run that never reached
    # `escalation_interrupt`, and both set together once it has (see
    # `GraphState`'s own comment on these two fields for why they cannot be
    # observed half-set). This is not the same claim as "a failed-safe
    # report never carries an escalation record" -- it can: if
    # `escalation_interrupt` committed a real decision and `final_report`
    # then crashes, `run_graph_investigation`'s outer crash containment
    # rebuilds this function's `state` from that last committed checkpoint
    # and calls it with `force_failure_reason` set, so `escalation_reason`/
    # `escalation_decision` are both still non-`None` even though
    # `disposition` ends up `FAILED_SAFE` below. `InvestigationReport`'s own
    # validator does not forbid that combination, and it is the honest
    # answer: the owner's decision was real and durable; the crash that
    # follows it is a separate fact.
    escalation: EscalationRecord | None = None
    escalation_reason_value = state["escalation_reason"]
    if escalation_reason_value is not None:
        decision_value = state["escalation_decision"]
        assert decision_value in ("accept", "reject")
        escalation = EscalationRecord(
            reason=EscalationReason(escalation_reason_value),
            decision=cast(Literal["accept", "reject"], decision_value),
            rejection_note=state["rejection_note"],
        )

    return InvestigationReport(
        investigation_id=state["investigation_id"],
        incident_id=state["incident_id"],
        disposition=disposition,
        root_cause=root_cause,
        assessment=assessment,
        reason_code=reason_code,
        budgets=budgets,
        versions=versions,
        started_at=started_at,
        finished_at=finished_at,
        latency_ms=int((finished_at - started_at).total_seconds() * 1000),
        model_calls_used=state["model_calls_used"],
        repairs_used=state["repairs_used"],
        tools_executed=tools_executed,
        invalid_responses=state["invalid_responses"],
        usage=usage,
        final_context_digest=state["context_digest"],
        evidence_ids=tuple(record.evidence_id for record in store.ordered()),
        receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
        retrieval_mode=RetrievalMode(state["retrieval_mode"]),
        runbook_passage_ids=tuple(passage.passage_id for passage in passages),
        limitations=tuple(limitations),
        escalation=escalation,
    )


class _StageCounters:
    """Mutable budget bookkeeping shared by `investigate` and
    `final_assessment` while asking one stage, with at most one repair.

    A plain stateful object, not `nonlocal` split across two near-identical
    node bodies, because a top-level helper has no way to mutate a caller's
    locals -- the same reason `BudgetLedger` existed in `workflow.py`.
    `record_call` must run *before* the model call it is counting, not
    after, so that a node's `except` handler still reports an attempt that
    raised -- the same "reserve before the risky call" ordering
    `tool_wrappers.py`'s `ReservationLedger` uses for tool budget, applied
    here to model-call budget instead.
    """

    def __init__(
        self,
        model_calls_used: int,
        repairs_used: int,
        invalid_responses: int,
        usage: dict[str, JsonValue] | None,
        context_digest: str,
    ) -> None:
        self.model_calls_used = model_calls_used
        self.repairs_used = repairs_used
        self.invalid_responses = invalid_responses
        self.usage = usage
        self.context_digest = context_digest
        self.stop_reason: ReasonCode | None = None

    def record_call(self, context_digest: str) -> None:
        self.context_digest = context_digest
        self.model_calls_used += 1

    def record_usage(self, latest: ModelUsage | None) -> None:
        self.usage = _accumulate_usage(self.usage, latest)

    def may_repair(self, budgets: Budgets) -> bool:
        return (
            self.repairs_used < budgets.repairs
            and _model_calls_left(self.model_calls_used, budgets) > 0
        )


def _render_stage_request(
    packet: InitialAlertPacket,
    scope: IncidentScope,
    store: EvidenceStore,
    receipts: Sequence[ToolReceipt],
    budgets: Budgets,
    model_calls_used: int,
    stage: Stage,
    repair_errors: str | None,
    passages: Sequence[RunbookPassage] = (),
    *,
    run_id: str,
    graph_phase: str,
    model_turn: int,
) -> tuple[ModelRequest, str]:
    """The context-render-and-digest step every model call needs, identical
    for `INVESTIGATE` and `FINAL_ASSESSMENT`. While `workflow.py` still ran,
    its `call_model` kept its own copy of this same logic rather than
    importing this function -- the two orchestrators ran side by side and
    reaching across that boundary for one shared helper would have coupled
    their lifecycles for no present benefit, with `test_parity.py` guarding
    the two copies from drifting apart in the meantime. Unit 1d-2 retired
    `workflow.py` and that duplication with it; this function now has no
    copy to drift from, and `test_graph_frozen_reports.py` (`test_parity.py`,
    renamed) is a plain regression pin on this file's own behaviour, not a
    two-orchestrator drift guard.

    `run_id`/`graph_phase`/`model_turn` (Unit 3b-2) are keyword-only and
    required, not defaulted: every caller already has all three in scope
    (`state["run_id"]`, the node's own `GraphPhase`, and either `turn_index`
    or `state["model_turn"]`), and a silent default here would let a future
    call site mint a `ModelRequest` with a wrong or stale idempotency key
    without mypy or a test ever noticing. The digest is computed *before*
    `ModelRequest` is constructed, not after as before this unit, because
    `context_digest` is now one of the object's own frozen fields -- the
    formula itself (system text + context text + repair errors) is
    unchanged, so replay's requests and every existing digest-based
    assertion are unaffected.
    """
    evidence, markers = store.context_evidence()
    context = render_context(
        packet,
        scope,
        evidence,
        markers,
        _model_calls_left(model_calls_used, budgets),
        _tools_left(receipts, budgets),
        passages,
        _denied_check_notes(receipts, budgets),
    )
    system_text = SYSTEM_TEXT
    context_text = f"{context}\n\n## Task\n{STAGE_INSTRUCTIONS[stage]}"
    digest = digest_text(system_text + context_text + (repair_errors or ""))
    request = ModelRequest(
        stage=stage,
        system_text=system_text,
        context_text=context_text,
        repair_errors=repair_errors,
        run_id=run_id,
        graph_phase=graph_phase,
        model_turn=model_turn,
        context_digest=digest,
    )
    return request, digest


def _ask_with_repair[T](
    counters: _StageCounters,
    budgets: Budgets,
    clock: Clock,
    started_at: str,
    ask_once: Callable[[str | None], tuple[T | None, str]],
    on_stop: Callable[[ReasonCode], None],
    on_invalid: Callable[[str], None],
) -> T | None:
    """Ask once, repair once on invalid output, stop with a reason
    otherwise -- the control flow `investigate` and `final_assessment` both
    need before every model call.

    `ask_once(repair_errors)` owns its own risky call and must call
    `counters.record_call(...)` before making it (see `_StageCounters`).
    This function never calls the model directly, so it cannot itself lose
    a spent call to an exception; the caller's `try`/`except` around the
    whole call to this function is what makes that guarantee end to end.
    Returns the parsed value, or `None` with `counters.stop_reason` set.

    `on_invalid(reason)` carries `ask_once`'s own account of what was wrong
    -- a schema-validation summary, or (`investigate` only) why a proposed
    tool call was refused -- into the durable event log, so `events.jsonl`
    can distinguish "the provider returned junk" from "the model proposed
    two checks in one turn" instead of recording only that a turn failed.
    """
    if _expired(started_at, budgets, clock):
        counters.stop_reason = ReasonCode.WALL_CLOCK_EXPIRED
        on_stop(counters.stop_reason)
        return None
    if _model_calls_left(counters.model_calls_used, budgets) <= 0:
        counters.stop_reason = ReasonCode.MODEL_CALL_BUDGET_EXHAUSTED
        on_stop(counters.stop_reason)
        return None

    parsed, errors = ask_once(None)
    if parsed is not None:
        return parsed
    counters.invalid_responses += 1
    on_invalid(errors)
    if not counters.may_repair(budgets):
        counters.stop_reason = ReasonCode.REPAIR_EXHAUSTED
        on_stop(counters.stop_reason)
        return None
    counters.repairs_used += 1
    parsed, second_errors = ask_once(errors)
    if parsed is None:
        counters.invalid_responses += 1
        on_invalid(second_errors)
        counters.stop_reason = ReasonCode.MODEL_OUTPUT_INVALID
        on_stop(counters.stop_reason)
        return None
    return parsed


def _make_investigate(
    scope: IncidentScope,
    packet: InitialAlertPacket,
    budgets: Budgets,
    *,
    clock: Clock,
    # Deliberately not `clock`: see `build_graph`'s docstring for why event
    # timestamps and domain-timing reads must not share ticks.
    event_clock: Clock,
    model: ToolCallingModel,
) -> GraphNode:
    def investigate(state: GraphState) -> dict[str, Any]:
        turn_index = state["model_turn"]
        stage = Stage.INITIAL_PLAN if turn_index == 0 else Stage.HYPOTHESIS_UPDATE
        schema = InitialPlan if turn_index == 0 else HypothesisUpdate
        recorder = _rebuild_recorder(state, event_clock)
        recorder.event(GraphPhase.INVESTIGATE.value, "stage_started", stage=stage.value)

        receipts = _rebuild_receipts(state)
        store = _rebuild_store(state["incident_id"], state["evidence"])
        passages = _rebuild_passages(state)
        counters = _StageCounters(
            state["model_calls_used"],
            state["repairs_used"],
            state["invalid_responses"],
            state["usage"],
            state["context_digest"],
        )
        last_proposal: ToolProposal | None = None

        def ask_once(repair_errors: str | None) -> tuple[Any, str]:
            nonlocal last_proposal
            request, digest = _render_stage_request(
                packet,
                scope,
                store,
                receipts,
                budgets,
                counters.model_calls_used,
                stage,
                repair_errors,
                passages,
                run_id=state["run_id"],
                graph_phase=GraphPhase.INVESTIGATE.value,
                model_turn=turn_index,
            )
            counters.record_call(digest)
            turn = model.propose(request, schema)
            counters.record_usage(turn.usage)
            if turn.parsed is None:
                return None, turn.errors
            calls = turn.tool_call
            if not calls:
                last_proposal = None
                return turn.parsed, ""
            selected = select_single_tool_call(calls)
            if selected is None:
                return None, (
                    f"the model proposed {len(calls)} checks in one turn; "
                    "exactly one is allowed"
                )
            proposal, reason = parse_tool_call(selected)
            if proposal is None:
                return None, reason
            last_proposal = proposal
            return turn.parsed, ""

        def log_invalid(reason: str) -> None:
            recorder.event(
                GraphPhase.INVESTIGATE.value,
                "invalid_response",
                stage=stage.value,
                reason=reason,
            )

        def log_stop(reason: ReasonCode) -> None:
            recorder.event(
                GraphPhase.INVESTIGATE.value, "stage_stopped", reason=reason.value
            )

        def stopped_state(reason: ReasonCode | None) -> dict[str, Any]:
            return {
                "phase": GraphPhase.INVESTIGATE.value,
                "model_calls_used": counters.model_calls_used,
                "repairs_used": counters.repairs_used,
                "invalid_responses": counters.invalid_responses,
                "context_digest": counters.context_digest,
                "usage": counters.usage,
                "pending_proposal": None,
                # Turn 0 failing mirrors `run()`'s `if plan is None: return
                # self.failed_safe()`. Turn >=1 failing mirrors
                # `plan_second_check()` discarding a failed second stage and
                # letting the run continue to FINAL_ASSESSMENT unchanged --
                # `reason` only becomes a terminal failure on the first turn.
                "failure_reason": (
                    reason.value if reason is not None and turn_index == 0 else None
                ),
                "events": _dump_events(recorder),
            }

        try:
            parsed = _ask_with_repair(
                counters,
                budgets,
                clock,
                state["started_at"],
                ask_once,
                log_stop,
                log_invalid,
            )
            if parsed is None:
                return stopped_state(counters.stop_reason)
            proposal = last_proposal

            recorder.event(
                GraphPhase.INVESTIGATE.value,
                "stage_finished",
                proposed=proposal is not None,
            )
            # Lab-defect-fix Unit 1, W11. `parsed` (the whole `InitialPlan`/
            # `HypothesisUpdate`) and `proposal` (`ToolProposal | None`) are
            # both still in hand here, right where `stage_finished` is
            # already emitted -- the one place this turn's ranked hypotheses,
            # evidence gap, expected observation, and as-requested tool
            # arguments exist before they are otherwise discarded (only
            # `pending_proposal`, which carries no hypotheses, survives past
            # this return). `hypotheses` is always present (both schemas
            # require it whether the turn proposes a check or gives a stop
            # reason); the other four fields are `None` on a stop-reason turn,
            # since there is no proposal to describe. `proposal_turn` is the
            # zero-based `turn_index`, the same convention `_proposal_turn`
            # recovers for `dispatch_tool`'s own events, so a reader can join
            # this record to the receipt it produced (if any) without
            # re-deriving the off-by-one between the two nodes.
            recorder.event(
                GraphPhase.INVESTIGATE.value,
                "proposal_recorded",
                proposal_turn=turn_index,
                stage=stage.value,
                hypotheses=[h.model_dump(mode="json") for h in parsed.hypotheses],
                tool=proposal.tool.value if proposal else None,
                evidence_gap=proposal.evidence_gap if proposal else None,
                expected_observation=(
                    proposal.expected_observation if proposal else None
                ),
                arguments=(
                    proposal.arguments.model_dump(mode="json") if proposal else None
                ),
            )
            return {
                "phase": GraphPhase.INVESTIGATE.value,
                "model_turn": turn_index + 1,
                "model_calls_used": counters.model_calls_used,
                "repairs_used": counters.repairs_used,
                "invalid_responses": counters.invalid_responses,
                "context_digest": counters.context_digest,
                "usage": counters.usage,
                "pending_proposal": (
                    proposal.model_dump(mode="json") if proposal else None
                ),
                "failure_reason": None,
                "events": _dump_events(recorder),
            }
        except GraphBubbleUp:
            raise
        except (
            CostCeilingExceeded,
            InputTooLarge,
            AmbiguousReservationNotResent,
        ) as refusal:
            # Unit 3b-2, extended by the Unit 3b-4 addendum's Group B.
            # `LiveClaudeModel` raises one of these *before sending* (or, for
            # `AmbiguousReservationNotResent`, *instead of* sending) -- the
            # cost gate, the input-token cap, or a pre-existing reservation
            # for this exact key refused the request, so no NEW money was
            # spent by this attempt. Caught ahead of the blanket `except
            # Exception` below on purpose: that handler exists for a crash
            # mid-attempt, and this is not one -- it is a policy refusal
            # with a specific, actionable reason, the same "name the actual
            # outcome" reasoning `_ask_with_repair`'s other stop reasons
            # already get. None of the three is retried as a repair:
            # appending a correction message to an over-large or
            # already-ambiguous request cannot fix any of them.
            # `stopped_state` reused as-is, including its turn-0-vs-turn>=1
            # rule: a refusal on the second INVESTIGATE turn still lets the
            # run reach FINAL_ASSESSMENT with whatever evidence it has,
            # exactly like `MODEL_CALL_BUDGET_EXHAUSTED` does today.
            reason_code = _money_refusal_reason_code(refusal)
            log_stop(reason_code)
            return stopped_state(reason_code)
        except Exception as error:
            # A model call already counted by `counters.record_call` before
            # this point must not vanish with this node's frame -- the same
            # hazard `dispatch_tool` closes for a reserved tool receipt,
            # applied here to the model-call budget. Unlike the turn-0-only
            # rule above, a crash always ends the run: `workflow.py`'s
            # `plan_second_check()` only ever swallowed a stage that returned
            # `None` normally, never one that raised -- a raise there
            # propagated out of `run()` to the loop's own top-level entry
            # point (`workflow.py`'s own `run_investigation`, a different
            # function from `cli.py`'s dispatcher of the same name today)
            # regardless of which stage crashed.
            recorder.event(
                GraphPhase.INVESTIGATE.value,
                "internal_error",
                error=type(error).__name__,
            )
            return {
                "phase": GraphPhase.INVESTIGATE.value,
                "model_calls_used": counters.model_calls_used,
                "repairs_used": counters.repairs_used,
                "invalid_responses": counters.invalid_responses,
                "context_digest": counters.context_digest,
                "usage": counters.usage,
                "pending_proposal": None,
                "failure_reason": ReasonCode.INTERNAL_ERROR.value,
                "events": _dump_events(recorder),
            }

    return investigate


def _make_dispatch_tool(
    scope: IncidentScope,
    budgets: Budgets,
    *,
    clock: Clock,
    # Deliberately not `clock`: see `build_graph`'s docstring for why event
    # timestamps and domain-timing reads must not share ticks.
    event_clock: Clock,
    registry: Mapping[ToolName, ToolWrapper],
) -> GraphNode:
    def dispatch_tool(state: GraphState) -> dict[str, Any]:
        assert state["pending_proposal"] is not None
        proposal = ToolProposal.model_validate(state["pending_proposal"])
        receipts = _rebuild_receipts(state)
        ledger = ReservationLedger.from_receipts(receipts, budgets.executed_tools)
        seen = set(state["seen_fingerprints"])
        recorder = _rebuild_recorder(state, event_clock)
        recorder.event(
            GraphPhase.DISPATCH_TOOL.value,
            "proposal_received",
            tool=proposal.tool.value,
        )

        try:
            wrapper = registry[proposal.tool]
            result = wrapper.dispatch(proposal, scope, seen, budgets, ledger, clock)

            # `authorize()` runs inside `wrapper.dispatch`, invisible from
            # here, so this node cannot emit an event at the exact moment
            # authorization passes the way `workflow.py`'s `check_started`
            # once did. What it can restore is the retired loop's event
            # *vocabulary*:
            # a denial is `proposal_denied`, never `check_finished`, and an
            # executed check gets both `check_started` and `check_finished`
            # rather than one event carrying `policy_result` for both cases.
            #
            # `check_started` and `check_finished` are both emitted here,
            # after `wrapper.dispatch` has already returned -- the backend
            # call already happened in the gap *before* `check_started`, so
            # this pair is not a timing bracket the way the retired loop's
            # was. The
            # receipt's own `duration_ms` (measured inside the wrapper,
            # around the real call) is the authoritative figure;
            # `check_finished` carries it explicitly so nothing has to
            # subtract these two timestamps to get zero.
            # Lab-defect-fix Unit 1, W11: `proposal_turn` and `receipt_id`
            # are the join key back to `investigate`'s own `proposal_recorded`
            # event and to the receipt in `receipts.jsonl` -- added to these
            # two existing events only (never `proposal_received`/
            # `check_started`, which fire before `result.receipt` exists).
            # No new event: the outcome is already fully recorded here, in
            # two mutually exclusive branches; what was missing was a way to
            # correlate this record back to the turn and receipt that
            # produced it.
            proposal_turn = _proposal_turn(state)
            if result.receipt.policy_result is PolicyResult.DENIED:
                recorder.event(
                    GraphPhase.DISPATCH_TOOL.value,
                    "proposal_denied",
                    proposal_turn=proposal_turn,
                    receipt_id=result.receipt.receipt_id,
                    reason=(
                        result.receipt.reason_code.value
                        if result.receipt.reason_code
                        else ""
                    ),
                    message=result.message,
                )
            else:
                recorder.event(
                    GraphPhase.DISPATCH_TOOL.value,
                    "check_started",
                    tool=proposal.tool.value,
                )
                recorder.event(
                    GraphPhase.DISPATCH_TOOL.value,
                    "check_finished",
                    proposal_turn=proposal_turn,
                    receipt_id=result.receipt.receipt_id,
                    outcome=(
                        result.receipt.outcome.value if result.receipt.outcome else ""
                    ),
                    duration_ms=result.receipt.duration_ms,
                )

            evidence = state["evidence"]
            if result.evidence is not None:
                evidence = [*evidence, result.evidence.model_dump(mode="json")]
            # Milestone 3, Unit 3a. `result.passages`/`result.retrieval_mode`
            # are only ever non-empty/non-`None` when `proposal.tool` is
            # `search_runbooks` and the dispatch was allowed (see
            # `DispatchResult`'s own docstring) -- a denial never reaches
            # this far, and the other four tools' wrappers never populate
            # either field. `retrieval_mode` is taken from the result, not
            # inferred from whether `passages` is empty: a search that ran
            # in `fts5_lexical` mode and found nothing still ran in that
            # mode, which is exactly the condition `_escalation_reason`'s
            # new `RETRIEVAL_COVERAGE_INSUFFICIENT` branch below exists to
            # catch -- `disabled` must stay reserved for "never attempted."
            runbook_passages = state["runbook_passages"]
            if result.passages:
                runbook_passages = [
                    *runbook_passages,
                    *(passage.model_dump(mode="json") for passage in result.passages),
                ]
            retrieval_mode = (
                result.retrieval_mode.value
                if result.retrieval_mode is not None
                else state["retrieval_mode"]
            )
            return {
                "phase": GraphPhase.DISPATCH_TOOL.value,
                "receipts": [r.model_dump(mode="json") for r in ledger.receipts()],
                "seen_fingerprints": sorted(seen),
                "pending_proposal": None,
                "evidence": evidence,
                "runbook_passages": runbook_passages,
                "retrieval_mode": retrieval_mode,
                "events": _dump_events(recorder),
            }
        except GraphBubbleUp:
            raise
        except Exception as error:
            # The reservation already happened -- `reserve()` runs before the
            # backend call -- and is sitting in `ledger`, which is a local
            # about to go out of scope with this node's frame. Writing it
            # into the state update now is the only way it survives; letting
            # this exception propagate out of the node would lose it, the
            # exact gap `tool_wrappers.py`'s reservation exists to close.
            # A crash before `ledger.settle()` ever ran leaves the receipt
            # `RESERVED` and `ledger.evidence()` empty -- nothing to carry,
            # matching this handler's pre-Unit-1d behaviour exactly. The one
            # window that differs is settle-then-crash: a crash *inside*
            # `wrapper.dispatch`, between `ledger.settle()` succeeding and
            # `DispatchResult` being constructed and returned, where a
            # `SETTLED` receipt carries a real `evidence_id`/`result_digest`
            # for an `Evidence` record that would otherwise never enter
            # state. `ledger.settle()` now durably stores that record in the
            # ledger the instant it runs, keyed by `receipt_id` -- the same
            # durability the receipt itself already had -- so
            # `ledger.evidence()` recovers it here for exactly that window.
            # This does not cover every crash after `wrapper.dispatch`
            # returns: if this handler's own `recorder.event` call below
            # raises, that exception propagates out of this node too, and
            # the fallback in `run_graph_investigation` reads the last
            # committed checkpoint -- losing this dispatch's receipt as well,
            # not only its evidence. That gap is pre-existing and unrelated
            # to the fix here; closing it would mean this handler catching
            # its own `recorder.event` failure, which is not implemented.
            recorder.event(
                GraphPhase.DISPATCH_TOOL.value,
                "backend_crashed",
                error=type(error).__name__,
            )
            recovered_evidence = [
                record.model_dump(mode="json") for record in ledger.evidence()
            ]
            return {
                "phase": GraphPhase.DISPATCH_TOOL.value,
                "receipts": [r.model_dump(mode="json") for r in ledger.receipts()],
                "seen_fingerprints": sorted(seen),
                "pending_proposal": None,
                "evidence": [*state["evidence"], *recovered_evidence],
                "failure_reason": ReasonCode.INTERNAL_ERROR.value,
                "events": _dump_events(recorder),
            }

    return dispatch_tool


def _make_normalize_evidence(
    # Deliberately not `clock`: see `build_graph`'s docstring for why event
    # timestamps and domain-timing reads must not share ticks. This node has
    # no domain timing of its own, so it takes only this one.
    event_clock: Clock,
) -> GraphNode:
    def normalize_evidence(state: GraphState) -> dict[str, Any]:
        recorder = _rebuild_recorder(state, event_clock)
        recorder.event(
            GraphPhase.NORMALIZE_EVIDENCE.value,
            "evidence_normalized",
            count=len(state["evidence"]),
        )
        return {
            "phase": GraphPhase.NORMALIZE_EVIDENCE.value,
            "events": _dump_events(recorder),
        }

    return normalize_evidence


def _make_final_assessment(
    scope: IncidentScope,
    packet: InitialAlertPacket,
    budgets: Budgets,
    *,
    clock: Clock,
    # Deliberately not `clock`: see `build_graph`'s docstring for why event
    # timestamps and domain-timing reads must not share ticks.
    event_clock: Clock,
    model: ToolCallingModel,
) -> GraphNode:
    def final_assessment(state: GraphState) -> dict[str, Any]:
        recorder = _rebuild_recorder(state, event_clock)
        recorder.event(
            GraphPhase.FINAL_ASSESSMENT.value,
            "stage_started",
            stage=Stage.FINAL_ASSESSMENT.value,
        )
        receipts = _rebuild_receipts(state)
        store = _rebuild_store(state["incident_id"], state["evidence"])
        passages = _rebuild_passages(state)
        counters = _StageCounters(
            state["model_calls_used"],
            state["repairs_used"],
            state["invalid_responses"],
            state["usage"],
            state["context_digest"],
        )

        def ask_once(repair_errors: str | None) -> tuple[FinalAssessment | None, str]:
            request, digest = _render_stage_request(
                packet,
                scope,
                store,
                receipts,
                budgets,
                counters.model_calls_used,
                Stage.FINAL_ASSESSMENT,
                repair_errors,
                passages,
                run_id=state["run_id"],
                graph_phase=GraphPhase.FINAL_ASSESSMENT.value,
                model_turn=state["model_turn"],
            )
            counters.record_call(digest)
            response = model.respond(request)
            counters.record_usage(response.usage)
            return parse_response(FinalAssessment, response.content)

        def log_invalid(reason: str) -> None:
            recorder.event(
                GraphPhase.FINAL_ASSESSMENT.value,
                "invalid_response",
                stage=Stage.FINAL_ASSESSMENT.value,
                reason=reason,
            )

        def log_stop(reason: ReasonCode) -> None:
            recorder.event(
                GraphPhase.FINAL_ASSESSMENT.value, "stage_stopped", reason=reason.value
            )

        def failed_state(reason_code: ReasonCode) -> dict[str, Any]:
            return {
                "phase": GraphPhase.FINAL_ASSESSMENT.value,
                "model_calls_used": counters.model_calls_used,
                "repairs_used": counters.repairs_used,
                "invalid_responses": counters.invalid_responses,
                "context_digest": counters.context_digest,
                "usage": counters.usage,
                "assessment": None,
                "failure_reason": reason_code.value,
                "events": _dump_events(recorder),
            }

        try:
            parsed = _ask_with_repair(
                counters,
                budgets,
                clock,
                state["started_at"],
                ask_once,
                log_stop,
                log_invalid,
            )
            if parsed is None:
                assert counters.stop_reason is not None
                return failed_state(counters.stop_reason)

            cited = parsed.supporting_evidence_ids + parsed.contrary_evidence_ids
            forged = store.unknown_ids(cited)
            if forged:
                recorder.event(
                    GraphPhase.FINAL_ASSESSMENT.value,
                    "forged_citation",
                    cited=len(forged),
                )
                return failed_state(ReasonCode.FORGED_EVIDENCE_REFERENCE)

            return {
                "phase": GraphPhase.FINAL_ASSESSMENT.value,
                "model_calls_used": counters.model_calls_used,
                "repairs_used": counters.repairs_used,
                "invalid_responses": counters.invalid_responses,
                "context_digest": counters.context_digest,
                "usage": counters.usage,
                "assessment": parsed.model_dump(mode="json"),
                "failure_reason": None,
                "events": _dump_events(recorder),
            }
        except GraphBubbleUp:
            raise
        except (
            CostCeilingExceeded,
            InputTooLarge,
            AmbiguousReservationNotResent,
        ) as refusal:
            # Unit 3b-2, extended by the Unit 3b-4 addendum's Group B. Same
            # reasoning as `investigate`'s handler: a refused-before-sending
            # (or refused-instead-of-resending) request is not a crash, so
            # it is caught ahead of the blanket handler below and reported
            # with its own actionable reason code rather than
            # `INTERNAL_ERROR`.
            reason_code = _money_refusal_reason_code(refusal)
            log_stop(reason_code)
            return failed_state(reason_code)
        except Exception as error:
            # Same hazard, same fix as `investigate`'s handler above: a
            # crash here always ends the run (unlike a normal `None` return,
            # `final_assessment` has no turn-dependent silent-swallow rule
            # to begin with -- every stop here is already terminal).
            recorder.event(
                GraphPhase.FINAL_ASSESSMENT.value,
                "internal_error",
                error=type(error).__name__,
            )
            return failed_state(ReasonCode.INTERNAL_ERROR)

    return final_assessment


def _escalation_reason(
    receipts: Sequence[ToolReceipt],
    assessment: FinalAssessment,
    budgets: Budgets,
    retrieved_passage_count: int,
) -> EscalationReason | None:
    """`TECHNICAL_SPEC.md` §8's four triggers, now that Unit 3a makes the
    fourth reachable. `EscalationReason`'s own member order follows the
    spec's listing (`CONFLICTING_EVIDENCE`, `TOOL_UNAVAILABLE`,
    `INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING`,
    `RETRIEVAL_COVERAGE_INSUFFICIENT`); the checks below deliberately do
    not -- `TOOL_UNAVAILABLE` is checked first, ahead of the spec's own
    listing order, and `RETRIEVAL_COVERAGE_INSUFFICIENT` is checked
    immediately after it, for the same reason. Pure, and called from both
    the router (to pick the edge) and the node (to build the `interrupt()`
    payload) -- the same reuse `_tools_left` already gets elsewhere in this
    file, not two independent readings of the same rule.

    A receipt going `UNAVAILABLE` is checked first regardless of what the
    model concluded from the checks that did run -- an owner should see a
    diagnosis reached with missing data before anything else.
    `RETRIEVAL_COVERAGE_INSUFFICIENT` is the same kind of missing-data
    signal, placed right after it: `retrieved_passage_count` counts
    passages this run actually **retrieved** (`len(state["runbook_
    passages"])` at both call sites below), never how many the model
    **cited** in `runbook_citations`. That is a deliberate, load-bearing
    choice, not an oversight: if this trigger is ever changed to read
    citations instead, a hallucinated `passage_id` becomes a way to
    manufacture (or dodge) an owner-facing escalation -- talking past the
    owner gate `TECHNICAL_SPEC.md` §8 exists to guarantee. It is exactly
    why `_build_report`'s own unresolved-citation check (see its docstring)
    is allowed to be non-fatal: nothing about a forged citation can move
    this trigger, because this trigger never looks at citations at all. The
    remaining two triggers are mutually exclusive by construction today (one
    requires `INSUFFICIENT_EVIDENCE`, the other's citation field has no rule
    forbidding it on either disposition, but only `DIAGNOSED` runs in this
    milestone's fixtures ever populate `contrary_evidence_ids`) -- the order
    below is still asserted by a dedicated test, since "mutually exclusive
    today" is a fact about current fixtures, not a promise about the fields.
    """
    if any(receipt.outcome is ToolOutcome.UNAVAILABLE for receipt in receipts):
        return EscalationReason.TOOL_UNAVAILABLE
    retrieval_attempted = any(
        receipt.tool is ToolName.SEARCH_RUNBOOKS
        and receipt.policy_result is PolicyResult.ALLOWED
        and receipt.state is ReceiptState.SETTLED
        for receipt in receipts
    )
    if retrieval_attempted and retrieved_passage_count == 0:
        return EscalationReason.RETRIEVAL_COVERAGE_INSUFFICIENT
    if (
        assessment.disposition is ModelDisposition.INSUFFICIENT_EVIDENCE
        and _tools_left(receipts, budgets) > 0
    ):
        return EscalationReason.INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING
    if assessment.contrary_evidence_ids:
        return EscalationReason.CONFLICTING_EVIDENCE
    return None


def _parse_resume_decision(
    value: object,
) -> tuple[Literal["accept", "reject"], str | None] | None:
    """Unit 2c's resume-value contract: `Command(resume=...)` must carry a
    mapping with a `decision` key and a `rejection_note` key, the same
    shape `causalops.approvals.OwnerDecision.resume_value()` produces.
    Returns `None` for anything else -- wrong type, missing/unknown
    `decision`, or a `rejection_note` that does not match `decision`
    (present on an accept, missing, or whitespace-only on a reject) -- so
    the caller's retry loop treats a malformed resume exactly like an
    unrecognized one.

    This is the same pairing check `OwnerDecision` already enforces at the
    CLI boundary, repeated here because the CLI is not the only caller of
    `Command(resume=...)` -- tests call it directly, and nothing stops a
    future caller from doing the same -- so this node cannot assume its
    input was ever validated upstream. A reject's note is stripped before
    the emptiness check and the stripped text is what is returned, so a
    caller that bypasses `OwnerDecision` entirely still lands a normalized
    note in `GraphState` and the finalized report, not a whitespace-only
    one `EscalationRecord`'s own check would then have to catch instead.
    """
    if not isinstance(value, Mapping):
        return None
    decision = value.get("decision")
    note = value.get("rejection_note")
    if decision == "accept":
        return ("accept", None) if note is None else None
    if decision == "reject":
        if not isinstance(note, str):
            return None
        stripped = note.strip()
        return ("reject", stripped) if stripped else None
    return None


def _make_escalation_interrupt(budgets: Budgets, event_clock: Clock) -> GraphNode:
    def escalation_interrupt(state: GraphState) -> dict[str, Any]:
        receipts = _rebuild_receipts(state)
        assessment = FinalAssessment.model_validate(state["assessment"])
        store = _rebuild_store(state["incident_id"], state["evidence"])
        # The router only routes here when this is not `None`
        # (`_make_route_after_final_assessment` below calls the same pure
        # function on the same state) -- this repeats that computation
        # rather than trusting the edge, the same "recompute, don't trust
        # the caller" posture `final_assessment`'s own `assert
        # counters.stop_reason is not None` already takes for an equally
        # router-guaranteed invariant.
        reason = _escalation_reason(
            receipts, assessment, budgets, len(state["runbook_passages"])
        )
        assert reason is not None

        # Everything above this line is a pure read of state -- no
        # `recorder.event`, no reservation, no write. `TECHNICAL_SPEC.md`
        # §5 requires an interrupt node to be side-effect-free before
        # calling `interrupt()`, and a probe against the installed
        # LangGraph confirmed why: only this node re-runs on resume, from
        # its own top, so anything written here would run twice. The
        # payload itself mirrors §8's interrupt-payload fields this
        # milestone can supply; `thread_id`/`run_id`/`checkpoint_id` are
        # not included here because the node has no way to know its own
        # checkpoint id before it exists -- `run_graph_investigation`
        # attaches all three once `.invoke()` returns.
        payload: dict[str, JsonValue] = {
            "reason": reason.value,
            "evidence_ids": [record.evidence_id for record in store.ordered()],
            "remaining_check_count": _tools_left(receipts, budgets),
        }
        raw_decision = interrupt(payload)
        resolved = _parse_resume_decision(raw_decision)
        # A bad resume value must not raise here. LangGraph persists a
        # resume value against the interrupt id and replays it on every
        # later resume of the same interrupt -- reproduced against a real
        # `SqliteSaver`: raising on a typo left the thread permanently
        # stuck replaying that same bad value on every subsequent resume,
        # valid ones included, because 2b never finalizes on pause and so
        # never gets a fresh interrupt id to retry against. Re-interrupting
        # instead asks again, under the same id, so a later valid decision
        # still settles the run. `retry` is not read by anything in this
        # milestone; it exists so a caller re-reading the pending
        # interrupt's payload after a bad attempt can tell a first ask from
        # a retry apart, the same way `payload` already tells them why the
        # run paused at all. Unit 2c: a bare string (still sent by every
        # test that resumes a plain accept/reject through
        # `resume_graph_run`'s pre-2c call sites) is exactly as invalid as
        # a typo now -- `_parse_resume_decision` requires the mapping
        # shape `causalops.approvals.OwnerDecision.resume_value()`
        # produces, so `Command(resume="accept")` re-interrupts too.
        while resolved is None:
            raw_decision = interrupt({**payload, "retry": True})
            resolved = _parse_resume_decision(raw_decision)
        decision, rejection_note = resolved

        # Nothing above this line ever executes a second time without also
        # being discarded -- an interrupted attempt raises instead of
        # returning, so no partial write from it ever reaches state. Only
        # code from here down runs exactly once, on the attempt that
        # settles this loop and falls through instead of interrupting
        # again. This node still crosses no I/O boundary here -- the
        # settling code below only rebuilds a local recorder from state and
        # returns a plain dict, the same shape `normalize_evidence` and
        # `final_report` already return without a `try`/`except` of their
        # own -- so there is nothing this node holds that a crash here
        # would lose beyond what a crash in either of those already loses.
        #
        # `rejection_note`'s prose stays out of `events.jsonl` on purpose --
        # the event's own `decision` field is the plain accept/reject
        # string, matching every other event this node has always
        # recorded. The note has two durable homes already (the
        # `owner_decisions` row and `EscalationRecord.rejection_note`); a
        # third with no reader is not needed.
        recorder = _rebuild_recorder(state, event_clock)
        recorder.event(
            GraphPhase.ESCALATION_INTERRUPT.value,
            "escalation_decided",
            reason=reason.value,
            decision=decision,
        )
        return {
            "phase": GraphPhase.ESCALATION_INTERRUPT.value,
            "escalation_reason": reason.value,
            "escalation_decision": decision,
            "rejection_note": rejection_note,
            "events": _dump_events(recorder),
        }

    return escalation_interrupt


def _make_final_report(
    budgets: Budgets,
    clock: Clock,
    # Deliberately not `clock`: see `build_graph`'s docstring for why event
    # timestamps and domain-timing reads must not share ticks.
    event_clock: Clock,
) -> GraphNode:
    def final_report(state: GraphState) -> dict[str, Any]:
        report = _build_report(state, budgets, clock)
        recorder = _rebuild_recorder(state, event_clock)
        # `_build_report` already folded the same computation into
        # `report.limitations` as owner-readable text; this recomputes it
        # from the same pure helper to put the fact into the audit trail
        # too, as a count rather than free text. Never fires on any
        # scenario that predates `search_runbooks` (its `runbook_citations`
        # is always empty), so no existing frozen event list moves.
        assessment = (
            FinalAssessment.model_validate(state["assessment"])
            if state["assessment"] is not None
            else None
        )
        unresolved = _unresolved_runbook_citations(assessment, _rebuild_passages(state))
        if unresolved:
            recorder.event(
                GraphPhase.FINAL_REPORT.value,
                "runbook_citation_unresolved",
                count=len(unresolved),
            )
        return {
            "phase": GraphPhase.FINAL_REPORT.value,
            "report": report.model_dump(mode="json"),
            "events": _dump_events(recorder),
        }

    return final_report


def route_after_investigate(state: GraphState) -> str:
    if state["pending_proposal"] is not None:
        return "dispatch_tool"
    if state["failure_reason"] is not None:
        return "final_report"
    return "final_assessment"


def _make_route_after_normalize(
    budgets: Budgets, clock: Clock
) -> Callable[[GraphState], str]:
    def route_after_normalize(state: GraphState) -> str:
        if state["failure_reason"] is not None:
            return "final_report"
        receipts = _rebuild_receipts(state)
        # Lab-defect-fix Unit 3, W1/Q2. This router used to cap at
        # `state["model_turn"] < 2`, reproducing `workflow.py`'s retired loop,
        # which called `plan_second_check()` at most once regardless of
        # whether the second proposal was allowed or denied. That bound made
        # a denial cost a turn permanently: `eda0135b…`'s real incident asked
        # for `list_recent_changes` first, was denied on a strict window
        # comparison, and never got a second chance at any check at all. With
        # Unit 3's window clamp making that class of denial rare rather than
        # frequent, a denial should no longer be able to consume a run's last
        # opportunity, so the `< 2` term is dropped.
        #
        # `_tools_left(...) > 0` (at most two *executed* checks -- a denial
        # never spends a slot) and `_model_calls_left(...) >= 2` (at most
        # four model calls total, with one always reserved for
        # `final_assessment`: one more for the next `INVESTIGATE` turn, one
        # for the assessment that must follow it) are both unchanged and
        # still do all the real bounding -- `>= 2`, not `> 0`, is load-
        # bearing: a repaired turn consumes two of the four calls
        # (`_StageCounters.record_call` increments on every attempt,
        # `_ask_with_repair` calls `ask_once` a second time on a repair), so
        # weakening it to `> 0` would let a run spend its last call on a
        # proposal and reach `final_assessment` with nothing left,
        # `MODEL_CALL_BUDGET_EXHAUSTED` instead of a real disposition.
        #
        # `state["model_turn"] < budgets.model_calls` is a new hard backstop,
        # not a rewritten rule: with the `< 2` term gone, nothing else in
        # this condition bounds `model_turn` by construction (every
        # `INVESTIGATE` turn past 0 maps to the same `HYPOTHESIS_UPDATE`
        # stage, so the model contract itself no longer limits it to two
        # turns) -- this backstop guarantees the loop still terminates even
        # if a future budget change ever let `_model_calls_left(...) >= 2`
        # stay true for more turns than `model_calls` alone would allow.
        if (
            _tools_left(receipts, budgets) > 0
            and _model_calls_left(state["model_calls_used"], budgets) >= 2
            and not _expired(state["started_at"], budgets, clock)
            and state["model_turn"] < budgets.model_calls
        ):
            return "investigate"
        return "final_assessment"

    return route_after_normalize


def _make_route_after_final_assessment(
    budgets: Budgets, *, suppress_escalation: bool = False
) -> Callable[[GraphState], str]:
    """`suppress_escalation` (Unit 3c) is the scored-run mode's only real
    mechanism: `TECHNICAL_SPEC.md` §10 requires the paired live comparison to
    "not invoke the escalation path." Set `True`, this router never reaches
    `_escalation_reason` at all -- not "compute the reason but ignore it,"
    which would still leave a way for a future edit to accidentally wire the
    result back in, but skip the call entirely, so a scored run's route to
    `"final_report"` cannot depend on what the trigger check would have said.
    `causalops.evaluate_cli` is the only caller that ever passes `True`;
    every other caller (through `build_graph`'s own default) gets today's
    unmodified escalation behaviour.
    """

    def route_after_final_assessment(state: GraphState) -> str:
        if state["failure_reason"] is not None:
            # A crashed, invalid, or forged-citation `final_assessment` has
            # no assessment to evaluate triggers against -- `state["assessment"]`
            # is `None` on every path that sets `failure_reason` here (see
            # `final_assessment`'s own `failed_state`). A failed-safe run is
            # never escalated, the same bypass rule `route_after_investigate`
            # and `route_after_normalize` already apply.
            return "final_report"
        if suppress_escalation:
            return "final_report"
        receipts = _rebuild_receipts(state)
        assessment = FinalAssessment.model_validate(state["assessment"])
        if (
            _escalation_reason(
                receipts, assessment, budgets, len(state["runbook_passages"])
            )
            is not None
        ):
            return "escalation_interrupt"
        return "final_report"

    return route_after_final_assessment


def build_graph(
    scope: IncidentScope,
    packet: InitialAlertPacket,
    budgets: Budgets,
    clock: Clock,
    model: ToolCallingModel,
    dispatch_registry: Mapping[ToolName, ToolWrapper],
    checkpointer: BaseCheckpointSaver[str],
    *,
    event_clock: Clock,
    suppress_escalation: bool = False,
    no_tool_baseline: bool = False,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """`clock` times domain data -- budget expiry, tool duration, timestamps
    a report or a citation might carry. `event_clock` times `RunEvent.at`
    only. They are the same function in production (`cli.py` passes
    `utc_now` for both), but keeping them as distinct parameters means
    recording an event never perturbs a reading `clock` would otherwise have
    produced -- book-keeping does not compete with domain data for ticks off
    the same clock. `test_graph_frozen_reports.py` is the reason this
    matters operationally, not just in principle: its `StepClock` advances
    on every read regardless of purpose, so entangling the two would shift
    evidence timestamps by however many events happened to be recorded
    first, for a reason that has nothing to do with the evidence itself.

    `suppress_escalation` and `no_tool_baseline` (Unit 3c) are independent
    switches for `causalops.evaluate_cli`'s paired live comparison, both
    defaulted `False` so every existing caller -- `run_graph_investigation`'s
    own default, `cli.py`, and every test that predates this unit -- keeps
    today's graph unchanged.

    `no_tool_baseline=True` builds a strictly smaller graph -- `investigate`,
    `dispatch_tool`, and `normalize_evidence` are never added as nodes at
    all, not merely left unreached, and `START` edges directly to
    `final_assessment` -- because `TECHNICAL_SPEC.md` §10's no-tool baseline
    has to mean the model never sees a domain-tool schema, not "tools bound
    but budget exhausted." `_make_final_assessment` already tolerates empty
    receipts/evidence/passages at `model_turn=0` on every existing call
    path (a `final_assessment` reached after zero investigate turns is not a
    new state this graph has never produced), so no node factory changes are
    needed for this topology, only which edges are wired.

    `prompts.py`'s `SYSTEM_TEXT` is unchanged for this smaller graph, on
    purpose -- it still tells the model it "may ask for registered
    read-only checks by name and typed arguments" even though
    `no_tool_baseline=True` never binds a single one. This was reviewed and
    deliberately kept identical, not overlooked: `TECHNICAL_SPEC.md`'s
    paired-comparison rule requires identical model, initial packet,
    budgets, taxonomy, and safe prompt constraints wherever applicable, and
    the prompt itself is one of those constraints -- diverging it between
    the two conditions, even just to drop a sentence that no longer
    applies, would itself become the confound this comparison exists to
    avoid. See `TECHNICAL_OVERVIEW.md`'s "Unit 3c" section for the full
    reasoning.

    `suppress_escalation=True` is unrelated to topology -- see
    `_make_route_after_final_assessment`'s own docstring for the mechanism.
    Both flags are set together for the baseline run (a model working from
    zero evidence commonly lands on `INSUFFICIENT_EVIDENCE` with checks
    still available, which would otherwise trigger
    `INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING`) and for the tool-enabled
    scored run alike -- both must stay unpaused for a scored evaluation to
    run unattended.
    """
    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    graph.add_node(
        "final_assessment",
        _make_final_assessment(
            scope, packet, budgets, clock=clock, event_clock=event_clock, model=model
        ),
    )
    graph.add_node(
        "escalation_interrupt", _make_escalation_interrupt(budgets, event_clock)
    )
    graph.add_node(
        "final_report", _make_final_report(budgets, clock, event_clock=event_clock)
    )

    if no_tool_baseline:
        graph.add_edge(START, "final_assessment")
    else:
        graph.add_node(
            "investigate",
            _make_investigate(
                scope,
                packet,
                budgets,
                clock=clock,
                event_clock=event_clock,
                model=model,
            ),
        )
        graph.add_node(
            "dispatch_tool",
            _make_dispatch_tool(
                scope,
                budgets,
                clock=clock,
                event_clock=event_clock,
                registry=dispatch_registry,
            ),
        )
        graph.add_node("normalize_evidence", _make_normalize_evidence(event_clock))

        graph.add_edge(START, "investigate")
        graph.add_conditional_edges(
            "investigate",
            route_after_investigate,
            {
                "dispatch_tool": "dispatch_tool",
                "final_assessment": "final_assessment",
                "final_report": "final_report",
            },
        )
        graph.add_edge("dispatch_tool", "normalize_evidence")
        graph.add_conditional_edges(
            "normalize_evidence",
            _make_route_after_normalize(budgets, clock),
            {
                "investigate": "investigate",
                "final_assessment": "final_assessment",
                "final_report": "final_report",
            },
        )

    graph.add_conditional_edges(
        "final_assessment",
        _make_route_after_final_assessment(
            budgets, suppress_escalation=suppress_escalation
        ),
        {
            "escalation_interrupt": "escalation_interrupt",
            "final_report": "final_report",
        },
    )
    graph.add_edge("escalation_interrupt", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile(checkpointer=checkpointer)


def _settle_invocation(
    compiled: CompiledStateGraph[GraphState, None, GraphState, GraphState],
    config: RunnableConfig,
    invoke: Callable[[], dict[str, Any]],
    budgets: Budgets,
    clock: Clock,
    event_clock: Clock,
    recorder: RunRecorder,
    fallback_state: GraphState,
) -> InvestigationResult | EscalatedInvestigation:
    """The tail every caller of `.invoke()` against this graph needs,
    regardless of what started the turn: crash containment, syncing the
    caller's `recorder`, and the terminal-vs-paused split. Unit 2c factored
    this out of `run_graph_investigation` so `resume_graph_investigation`
    (below) shares it rather than duplicating roughly eighty lines of
    crash-recovery and result-assembly logic with a chance to drift between
    the two copies.

    `invoke` is a zero-argument closure so this function does not care
    whether the caller is calling `compiled.invoke(initial_state, config)`
    (a fresh start) or `compiled.invoke(Command(resume=...), config)` (a
    resume) -- everything below this point treats both identically, which
    is exactly the claim being made by sharing this code at all.
    `fallback_state` stands in for what `run_graph_investigation` used to
    call `initial_state` in its own crash-containment branches: the state
    to fall back to if `compiled.get_state(config).values` is somehow
    empty. A resume always has a real checkpoint to read by the time it
    reaches here -- `resume_graph_investigation`'s own pending-interrupt
    assertion has already confirmed that -- so it passes the state it read
    while confirming that, never a manufactured empty one.
    """
    try:
        raw_state = invoke()
    except GraphBubbleUp:
        # A control-flow signal (interrupt, drain, parent command), not a
        # failure -- Milestone 2 adds `interrupt()`, and this must keep
        # propagating rather than being turned into a safe report.
        #
        # It must still not lose every event recorded before the signal
        # fired. The caller's `recorder` is empty at this exact point on a
        # fresh start: the seed event went to a separate seed recorder, not
        # this one, and every node event since then lives only in state --
        # synced onto `recorder` at this function's normal return below,
        # which a `raise` never reaches. Reading the last checkpoint here
        # and syncing from it is a data-recovery step, the same one the
        # `except Exception` branch below already performs to build a
        # report; it is not a write, so it does not conflict with an
        # interrupt node needing to be side-effect-free before
        # `interrupt()`.
        #
        # This read is I/O against the same database `checkpointer` uses,
        # so it can itself fail -- a locked, full, or corrupt database
        # raises `sqlite3.OperationalError`, not `GraphBubbleUp`. The
        # control-flow signal must survive a checkpoint-read failure and
        # keep propagating either way: losing a best-effort event-recovery
        # step is acceptable, but replacing the signal with an unhandled
        # database error that `main` cannot format into `FAIL <CODE>
        # <message>` is not. This is the one place in this file that reads
        # a checkpoint outside a context already prepared to build a report
        # from what it finds, so it is the one place that read needs its
        # own guard.
        try:
            checkpoint = compiled.get_state(config).values or fallback_state
            state = cast(GraphState, checkpoint)
            recorder.recorded = [
                RunEvent.model_validate(dump) for dump in state["events"]
            ]
        except Exception:
            pass
        raise
    except Exception as error:
        # Unmodeled failure: a node bug, or GraphRecursionError (subclasses
        # RecursionError -> RuntimeError -> Exception, so it lands here too).
        # This is not the tool-crash or model-call-crash path -- `dispatch_tool`,
        # `investigate`, and `final_assessment` above already turn their own
        # crashes into a normal state update -- so there is no node-local
        # object to rescue here, only the last checkpoint LangGraph itself
        # wrote. That checkpoint carries its own `events` list, so the
        # `internal_error` event is folded into a recorder rebuilt from it,
        # the same way every node above records into state rather than onto
        # this function's own `recorder` parameter directly.
        #
        # This fold is not itself a durable write: `state` here is a local
        # dict, never passed back through `checkpointer.put(...)`, so the
        # `internal_error` event reaches this run's report and the caller's
        # `recorder` mirror below, but not the checkpoint store itself. A
        # second process reopening this thread's checkpoint after this crash
        # would not see it -- there is nothing to resume into anyway, since
        # this path is a terminal `FAILED_SAFE`, not a pause.
        checkpoint = compiled.get_state(config).values or fallback_state
        state = cast(GraphState, checkpoint)
        crash_recorder = _rebuild_recorder(state, event_clock)
        crash_recorder.event(
            GraphPhase.FINAL_REPORT.value, "internal_error", error=type(error).__name__
        )
        state = cast(GraphState, {**state, "events": _dump_events(crash_recorder)})
        report = _build_report(
            state, budgets, clock, force_failure_reason=ReasonCode.INTERNAL_ERROR
        )
        raw_state = {**state, "report": report.model_dump(mode="json")}

    final_state = cast(GraphState, raw_state)
    # The caller's `recorder` never runs live inside a node -- every event
    # above was recorded into a state-rebuilt local recorder instead. This
    # assignment REPLACES `recorder.recorded` wholesale with the complete
    # event list from state; it does not append to whatever the caller's
    # object already held. That is safe today because every caller
    # constructs a fresh, empty `RunRecorder` immediately before calling
    # this function, so replace and extend are indistinguishable in
    # practice -- but a future caller that reuses a `RunRecorder` across
    # calls would lose whatever it held before this one. Every existing
    # caller that reads `recorder.events` after this call still sees
    # exactly what it saw before this unit.
    recorder.recorded = [
        RunEvent.model_validate(dump) for dump in final_state["events"]
    ]

    # Unit 2b. `.invoke()` returns normally on a real pause -- it does not
    # raise, so this is not an `except GraphBubbleUp:` case (a probe against
    # the installed LangGraph confirmed this before any of this file was
    # written) -- with `"__interrupt__"` present alongside whatever state the
    # graph had committed before `escalation_interrupt` ran. Checked on
    # `raw_state`, not `final_state`: `"__interrupt__"` is not a `GraphState`
    # key, so reading it off the `GraphState`-typed name would not type-check.
    # This must run before the `InvestigationReport.model_validate` call
    # below -- on a pause, `final_state["report"]` is still `None`, and that
    # call would raise a `ValidationError` instead of returning a paused
    # result.
    interrupts = raw_state.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        checkpoint_id = str(
            compiled.get_state(config).config["configurable"]["checkpoint_id"]
        )
        store = _rebuild_store(final_state["incident_id"], final_state["evidence"])
        receipts = tuple(_rebuild_receipts(final_state))
        return EscalatedInvestigation(
            thread_id=final_state["investigation_id"],
            run_id=final_state["run_id"],
            checkpoint_id=checkpoint_id,
            reason=EscalationReason(payload["reason"]),
            evidence=store.ordered(),
            receipts=receipts,
            remaining_check_count=payload["remaining_check_count"],
            proposal_fingerprint=None,
        )

    report = InvestigationReport.model_validate(final_state["report"])
    store = _rebuild_store(final_state["incident_id"], final_state["evidence"])
    receipts = tuple(_rebuild_receipts(final_state))
    return InvestigationResult(
        report=report, evidence=store.ordered(), receipts=receipts
    )


def run_graph_investigation(
    scope: IncidentScope,
    packet: InitialAlertPacket,
    initial_evidence: Sequence[Evidence],
    model: ToolCallingModel,
    dispatch_registry: Mapping[ToolName, ToolWrapper],
    recorder: RunRecorder,
    budgets: Budgets = DEFAULT_BUDGETS,
    clock: Clock = utc_now,
    *,
    investigation_id: str | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    model_name: str = REPLAY_MODEL_NAME,
    suppress_escalation: bool = False,
    no_tool_baseline: bool = False,
) -> InvestigationResult | EscalatedInvestigation:
    """Run one investigation to completion, or to its first pause.

    `suppress_escalation`/`no_tool_baseline` (Unit 3c) pass straight through
    to `build_graph` -- see its own docstring for what each one does. Both
    default `False`, so this function's behaviour is unchanged for every
    caller written before this unit. Only `causalops.evaluate_cli` ever
    passes either as `True`; `resume_graph_investigation` below never takes
    these two, because a scored run that never pauses has nothing to
    resume.

    `investigation_id` doubles as LangGraph's own `thread_id`
    (`TECHNICAL_SPEC.md:140-142`). It is `None` for every caller today --
    minting a fresh one is what every existing caller already gets -- and
    becomes a real input once Milestone 2's resume path needs to reopen a
    specific thread rather than start a new one.

    `model_name` (Unit 3b-2) is defaulted, not required: it exists so
    `GraphState["model_name"]` carries the label a resumed thread's artifact
    should use, and every one of the ~28 test call sites across this
    project predates that need and passes a `ReplayToolCallingModel` -- the
    default keeps every one of them behaviourally unchanged. Only `cli.py`'s
    real `investigate` dispatch passes a real value.

    This function takes no `resume` argument. It always starts a fresh run
    from `initial_state` below; resuming a paused thread is
    `resume_graph_investigation`, below, Unit 2c's real resumable entry
    point -- the two share `_settle_invocation`'s crash containment and
    result assembly rather than each carrying its own copy.

    `checkpointer` defaults to a fresh, process-local `InMemorySaver()` so
    every existing caller (tests included) keeps today's behaviour exactly:
    nothing is written to disk unless a caller (`cli.py`, for the real CLI
    path) explicitly supplies a durable one.
    """
    if investigation_id is None:
        investigation_id = new_opaque_id()
    # `recorder.clock` -- not the `clock` parameter above -- times every
    # `RunEvent`. Before this unit, the caller's `RunRecorder` was the only
    # thing that ever called `.event(...)`, so its own clock never shared
    # ticks with `clock`, which times domain data (budget expiry, tool
    # duration, evidence timestamps). Rebuilding a recorder from state inside
    # each node must preserve that same isolation -- see `build_graph`'s
    # docstring for why entangling the two is an observable behaviour change,
    # not just a style choice.
    event_clock = recorder.clock
    # `run_id` is internal bookkeeping -- unlike `investigation_id`, evidence
    # IDs, and receipt IDs, it is never cited by a model, displayed in a
    # report, or looked up by an owner. It is minted with `uuid4().hex`
    # directly rather than through `new_opaque_id()`, the shared minting
    # function those other, user-visible IDs go through: an identifier
    # nothing outside this function reads has no business drawing from the
    # same counter as one a citation or an audit trail depends on.
    #
    # (Confirmed, not just argued: `evidence.py`/`graph.py`/`tool_wrappers.py`
    # all bind `new_opaque_id()` to one name each, and
    # `test_graph_frozen_reports.py`'s `_install_counting_ids` patches all
    # three to a single shared counter so its literals are deterministic --
    # routing `run_id` through that counter shifts every ID minted
    # afterward and fails 5 tests, verified by mutation.)
    run_id = uuid4().hex
    started_at = clock()
    seed_recorder = RunRecorder(event_clock)
    seed_recorder.event(
        GraphPhase.CREATED.value,
        "investigation_started",
        incident=scope.incident_id,
        # Unit 3c. `run_id` is internal bookkeeping and deliberately absent
        # from `InvestigationReport` itself (see this function's own
        # comment on `run_id`, above) -- but `causalops.evaluate_cli` needs
        # it to look up this run's own cost from `cost_ledger.
        # run_cost_totals`, which is keyed by `run_id`, not
        # `investigation_id`. Recording it on this one event (`events.jsonl`
        # is not schema-frozen the way `InvestigationReport` is; no test
        # pins this event's exact field set) is a smaller, more honest fix
        # than adding a field to the report schema that Unit 3c does not
        # otherwise need.
        run_id=run_id,
    )
    initial_state: GraphState = {
        "investigation_id": investigation_id,
        "run_id": run_id,
        "incident_id": scope.incident_id,
        "model_name": model_name,
        "phase": GraphPhase.CREATED.value,
        "model_turn": 0,
        "model_calls_used": 0,
        "repairs_used": 0,
        "invalid_responses": 0,
        "started_at": started_at.isoformat(),
        "context_digest": "",
        "seen_fingerprints": [],
        "receipts": [],
        "evidence": [record.model_dump(mode="json") for record in initial_evidence],
        "runbook_passages": [],
        "retrieval_mode": RetrievalMode.DISABLED.value,
        "events": _dump_events(seed_recorder),
        "pending_proposal": None,
        "assessment": None,
        "usage": None,
        "failure_reason": None,
        "escalation_reason": None,
        "escalation_decision": None,
        "rejection_note": None,
        "report": None,
    }

    compiled = build_graph(
        scope,
        packet,
        budgets,
        clock,
        model,
        dispatch_registry,
        checkpointer if checkpointer is not None else InMemorySaver(),
        event_clock=event_clock,
        suppress_escalation=suppress_escalation,
        no_tool_baseline=no_tool_baseline,
    )
    config: RunnableConfig = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"thread_id": investigation_id},
    }

    def invoke() -> dict[str, Any]:
        return cast(dict[str, Any], compiled.invoke(initial_state, config))

    return _settle_invocation(
        compiled, config, invoke, budgets, clock, event_clock, recorder, initial_state
    )


def resume_graph_investigation(
    thread_id: str,
    checkpointer: BaseCheckpointSaver[str],
    scope: IncidentScope,
    packet: InitialAlertPacket,
    model: ToolCallingModel,
    dispatch_registry: Mapping[ToolName, ToolWrapper],
    recorder: RunRecorder,
    decision: Literal["accept", "reject"],
    rejection_note: str | None,
    budgets: Budgets = DEFAULT_BUDGETS,
    clock: Clock = utc_now,
) -> InvestigationResult:
    """Unit 2c's real resumable entry point -- the one `run_graph_investigation`
    used to say a resume path still owed. Resumes `thread_id` with one
    already-validated owner decision and settles it to a terminal
    `InvestigationResult`.

    Always returns `InvestigationResult`, never `EscalatedInvestigation`:
    `decision`/`rejection_note` have already passed
    `causalops.approvals.OwnerDecision`'s validation before this function is
    ever called, so `escalation_interrupt`'s own resume-parsing loop
    (`_parse_resume_decision`) exits on the first pass and the run falls
    straight through to `final_report` -- there is no path back to a second
    pause in Unit 2c's graph (the spec's "approve one additional check"
    route to `DISPATCH_TOOL` does not exist yet; see this module's
    docstring and the `TECHNICAL_SPEC.md` Unit 2c amendment).

    `cli.py` owns every guard this function assumes has already passed:
    that `thread_id` names a real, pending interrupt, that `decision`/
    `rejection_note` are a validated pair, and that the owner's decision is
    already durably recorded in `owner_decisions` before this function is
    ever called (`TECHNICAL_SPEC.md:170-172`'s record-before-resume rule).
    This function's only job is the graph turn itself -- the assertion
    below is a "recompute, don't trust the caller" check of the one
    precondition (a real checkpoint exists) that would otherwise fail
    silently rather than loudly.
    """
    event_clock = recorder.clock
    compiled = build_graph(
        scope,
        packet,
        budgets,
        clock,
        model,
        dispatch_registry,
        checkpointer,
        event_clock=event_clock,
    )
    config: RunnableConfig = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
    }
    pre_resume_state = compiled.get_state(config).values
    assert pre_resume_state, (
        f"resume_graph_investigation called for thread {thread_id!r} with "
        "no checkpoint -- the caller's pending-interrupt guard should have "
        "refused before this function was ever called"
    )
    resume_value: dict[str, JsonValue] = {
        "decision": decision,
        "rejection_note": rejection_note,
    }

    def invoke() -> dict[str, Any]:
        return cast(
            dict[str, Any], compiled.invoke(Command(resume=resume_value), config)
        )

    result = _settle_invocation(
        compiled,
        config,
        invoke,
        budgets,
        clock,
        event_clock,
        recorder,
        cast(GraphState, pre_resume_state),
    )
    assert isinstance(result, InvestigationResult)
    return result
