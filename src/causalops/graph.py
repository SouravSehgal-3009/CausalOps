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

`ReplayToolCallingModel` is the only model type this file binds -- not the
plain `ReasoningModel` protocol `workflow.py` used. That is a known,
deliberate gap: a `propose()`-shaped protocol for the tool-calling adapter
would be speculative with only one implementation to validate its shape
against. It closes when the live Claude adapter unit adds a second one.
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, TypedDict, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import JsonValue

from causalops.domain import (
    DEFAULT_BUDGETS,
    Budgets,
    Clock,
    Disposition,
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
    RootCauseCode,
    ToolProposal,
    ToolReceipt,
    Versions,
    utc_now,
)
from causalops.evidence import EvidenceStore, digest_text, new_opaque_id
from causalops.models import ModelRequest, ReplayToolCallingModel, Stage, parse_response
from causalops.policy import POLICY_VERSION
from causalops.prompts import (
    PROMPT_VERSION,
    STAGE_INSTRUCTIONS,
    SYSTEM_TEXT,
    render_context,
)
from causalops.run_records import RunEvent, RunRecorder
from causalops.tool_calls import (
    NativeToolCall,
    parse_tool_call,
    select_single_tool_call,
)
from causalops.tool_wrappers import ReservationLedger, ToolWrapper
from causalops.tools import TOOL_REGISTRY_VERSION, ToolName

# The library default is 10007 (`_internal/_config.py:32`), meant for graphs
# whose shape isn't known ahead of time. This graph's longest real path is
# eight supersteps (investigate, dispatch, normalize, investigate, dispatch,
# normalize, final_assessment, final_report); 25 leaves headroom for a
# guard bug without letting a real one spin for thousands of steps first.
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
    return [ToolReceipt.model_validate(dump) for dump in state["receipts"]]


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
    started_at = datetime.fromisoformat(state["started_at"])
    finished_at = clock()
    usage = (
        ModelUsage.model_validate(state["usage"])
        if state["usage"] is not None
        else None
    )
    limitations: tuple[str, ...] = (
        () if usage is not None else ("this model reports no token usage",)
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
        limitations=limitations,
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
    """
    evidence, markers = store.context_evidence()
    context = render_context(
        packet,
        scope,
        evidence,
        markers,
        _model_calls_left(model_calls_used, budgets),
        _tools_left(receipts, budgets),
    )
    request = ModelRequest(
        stage=stage,
        system_text=SYSTEM_TEXT,
        context_text=f"{context}\n\n## Task\n{STAGE_INSTRUCTIONS[stage]}",
        repair_errors=repair_errors,
    )
    digest = digest_text(
        request.system_text + request.context_text + (request.repair_errors or "")
    )
    return request, digest


def _ask_with_repair[T](
    counters: _StageCounters,
    budgets: Budgets,
    clock: Clock,
    started_at: str,
    ask_once: Callable[[str | None], tuple[T | None, str]],
    on_stop: Callable[[ReasonCode], None],
    on_invalid: Callable[[], None],
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
    on_invalid()
    if not counters.may_repair(budgets):
        counters.stop_reason = ReasonCode.REPAIR_EXHAUSTED
        on_stop(counters.stop_reason)
        return None
    counters.repairs_used += 1
    parsed, _ = ask_once(errors)
    if parsed is None:
        counters.invalid_responses += 1
        on_invalid()
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
    model: ReplayToolCallingModel,
) -> GraphNode:
    def investigate(state: GraphState) -> dict[str, Any]:
        turn_index = state["model_turn"]
        stage = Stage.INITIAL_PLAN if turn_index == 0 else Stage.HYPOTHESIS_UPDATE
        schema = InitialPlan if turn_index == 0 else HypothesisUpdate
        recorder = _rebuild_recorder(state, event_clock)
        recorder.event(GraphPhase.INVESTIGATE.value, "stage_started", stage=stage.value)

        receipts = _rebuild_receipts(state)
        store = _rebuild_store(state["incident_id"], state["evidence"])
        counters = _StageCounters(
            state["model_calls_used"],
            state["repairs_used"],
            state["invalid_responses"],
            state["usage"],
            state["context_digest"],
        )
        last_tool_call: NativeToolCall | None = None

        def ask_once(repair_errors: str | None) -> tuple[Any, str]:
            nonlocal last_tool_call
            request, digest = _render_stage_request(
                packet,
                scope,
                store,
                receipts,
                budgets,
                counters.model_calls_used,
                stage,
                repair_errors,
            )
            counters.record_call(digest)
            turn = model.propose(request, schema)
            counters.record_usage(turn.usage)
            last_tool_call = turn.tool_call
            return turn.parsed, turn.errors

        def log_invalid() -> None:
            recorder.event(
                GraphPhase.INVESTIGATE.value, "invalid_response", stage=stage.value
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

            proposal: ToolProposal | None = None
            if last_tool_call is not None:
                # The round trip through the same encode/decode functions a
                # live Claude adapter's message would have to pass through --
                # not a shortcut through the parsed stage's own `.proposal`
                # field directly. Both functions are proven lossless for any
                # `ToolProposal` the domain layer can construct
                # (`test_tool_calls.py`), so a `None` here is this file's
                # bug, not the model's.
                selected = select_single_tool_call([last_tool_call])
                if selected is None:
                    raise AssertionError(
                        "a single-element tool call list failed self-selection"
                    )
                proposal = parse_tool_call(selected)
                if proposal is None:
                    raise AssertionError(
                        "a just-encoded tool call failed to parse back -- "
                        "to_tool_call/parse_tool_call round trip is broken"
                    )

            recorder.event(
                GraphPhase.INVESTIGATE.value,
                "stage_finished",
                proposed=proposal is not None,
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
            if result.receipt.policy_result is PolicyResult.DENIED:
                recorder.event(
                    GraphPhase.DISPATCH_TOOL.value,
                    "proposal_denied",
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
                    outcome=(
                        result.receipt.outcome.value if result.receipt.outcome else ""
                    ),
                    duration_ms=result.receipt.duration_ms,
                )

            evidence = state["evidence"]
            if result.evidence is not None:
                evidence = [*evidence, result.evidence.model_dump(mode="json")]
            return {
                "phase": GraphPhase.DISPATCH_TOOL.value,
                "receipts": [r.model_dump(mode="json") for r in ledger.receipts()],
                "seen_fingerprints": sorted(seen),
                "pending_proposal": None,
                "evidence": evidence,
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
    model: ReplayToolCallingModel,
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
            )
            counters.record_call(digest)
            response = model.respond(request)
            counters.record_usage(response.usage)
            return parse_response(FinalAssessment, response.content)

        def log_invalid() -> None:
            recorder.event(
                GraphPhase.FINAL_ASSESSMENT.value,
                "invalid_response",
                stage=Stage.FINAL_ASSESSMENT.value,
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


def _make_final_report(budgets: Budgets, clock: Clock) -> GraphNode:
    def final_report(state: GraphState) -> dict[str, Any]:
        report = _build_report(state, budgets, clock)
        return {
            "phase": GraphPhase.FINAL_REPORT.value,
            "report": report.model_dump(mode="json"),
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
        # `workflow.py`'s retired loop called `plan_second_check()` at most
        # once, from `run()` -- never a third time, regardless of whether the
        # second proposal was allowed or denied. A denial does not spend a
        # slot (`ReservationLedger.slots_left()`), so `tools_left()` alone
        # cannot bound the turn count the way it did in the loop, where there
        # was structurally no third ask. `investigate` maps every turn past 0
        # to `HYPOTHESIS_UPDATE` (`graph.py`'s stage mapping), and there is no
        # third planning stage in the model contract at all -- a phantom
        # third turn would ask a stage the contract cannot express, not
        # merely one the loop skipped. `model_turn` is 1 once turn 0's
        # dispatch has run and 2 once turn 1's has; capping at `< 2`
        # reproduces the loop's own former bound instead of the budget's
        # incidental one.
        if (
            state["model_turn"] < 2
            and _tools_left(receipts, budgets) > 0
            and _model_calls_left(state["model_calls_used"], budgets) >= 2
            and not _expired(state["started_at"], budgets, clock)
        ):
            return "investigate"
        return "final_assessment"

    return route_after_normalize


def build_graph(
    scope: IncidentScope,
    packet: InitialAlertPacket,
    budgets: Budgets,
    clock: Clock,
    model: ReplayToolCallingModel,
    dispatch_registry: Mapping[ToolName, ToolWrapper],
    checkpointer: BaseCheckpointSaver[str],
    *,
    event_clock: Clock,
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
    """
    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    graph.add_node(
        "investigate",
        _make_investigate(
            scope, packet, budgets, clock=clock, event_clock=event_clock, model=model
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
    graph.add_node(
        "final_assessment",
        _make_final_assessment(
            scope, packet, budgets, clock=clock, event_clock=event_clock, model=model
        ),
    )
    graph.add_node("final_report", _make_final_report(budgets, clock))

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
    graph.add_edge("final_assessment", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile(checkpointer=checkpointer)


def run_graph_investigation(
    scope: IncidentScope,
    packet: InitialAlertPacket,
    initial_evidence: Sequence[Evidence],
    model: ReplayToolCallingModel,
    dispatch_registry: Mapping[ToolName, ToolWrapper],
    recorder: RunRecorder,
    budgets: Budgets = DEFAULT_BUDGETS,
    clock: Clock = utc_now,
    *,
    investigation_id: str | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> InvestigationResult:
    """Run one investigation to completion.

    `investigation_id` doubles as LangGraph's own `thread_id`
    (`TECHNICAL_SPEC.md:140-142`). It is `None` for every caller today --
    minting a fresh one is what every existing caller already gets -- and
    becomes a real input once Milestone 2's resume path needs to reopen a
    specific thread rather than start a new one.

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
        GraphPhase.CREATED.value, "investigation_started", incident=scope.incident_id
    )
    initial_state: GraphState = {
        "investigation_id": investigation_id,
        "run_id": run_id,
        "incident_id": scope.incident_id,
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
        "events": _dump_events(seed_recorder),
        "pending_proposal": None,
        "assessment": None,
        "usage": None,
        "failure_reason": None,
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
    )
    config: RunnableConfig = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"thread_id": investigation_id},
    }

    try:
        raw_state = cast(dict[str, Any], compiled.invoke(initial_state, config))
    except GraphBubbleUp:
        # A control-flow signal (interrupt, drain, parent command), not a
        # failure -- Milestone 2 adds `interrupt()`, and this must keep
        # propagating rather than being turned into a safe report.
        #
        # It must still not lose every event recorded before the signal
        # fired. The caller's `recorder` is empty at this exact point: the
        # seed event went to `seed_recorder`, not `recorder`, and every node
        # event since then lives only in state -- synced onto `recorder` at
        # this function's normal return below, which a `raise` never
        # reaches. Reading the last checkpoint here and syncing from it is a
        # data-recovery step, the same one the `except Exception` branch
        # below already performs to build a report; it is not a write, so it
        # does not conflict with an interrupt node needing to be
        # side-effect-free before `interrupt()`.
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
            checkpoint = compiled.get_state(config).values or initial_state
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
        checkpoint = compiled.get_state(config).values or initial_state
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
    report = InvestigationReport.model_validate(final_state["report"])
    store = _rebuild_store(final_state["incident_id"], final_state["evidence"])
    receipts = tuple(_rebuild_receipts(final_state))
    return InvestigationResult(
        report=report, evidence=store.ordered(), receipts=receipts
    )
