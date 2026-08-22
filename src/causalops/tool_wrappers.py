"""The only path from a parsed tool proposal to a backend.

`TECHNICAL_SPEC.md` §5 makes a direct backend binding a P0 trust-boundary
violation: a dispatch node may reach a backend only through a policy wrapper,
never a bare backend function. This module is that wrapper for all five
registered tools. Unit 1a proved the shape against `query_logs` and a real
backend first; `_make_wrapper` below is that same dispatch body generalised
over which tool, which argument type, and which backend seam -- proven under
`mypy --strict` including the negative case (a backend typed for the wrong
tool is a type error, not a runtime surprise) before it replaced the four
near-identical copies four factories would have been. Unit 3a added a fifth
tool, `search_runbooks`, and one branch inside `_make_wrapper`'s own dispatch
body (on the *type* of result its `run_check` seam returns) rather than a
second factory -- the plan's own owner decision, "one wrapper factory, one
dispatch path."

Nothing here imports `causalops.telemetry` or `causalops.prometheus` -- see
`tests/security/test_tool_boundary.py` for the AST test that checks that, plus
the wrapper-identity and spy-backend tests that check what an import scan
cannot: that every registered dispatch callable was actually produced by the
factory below, and that a denied proposal never reaches a backend.

Order is the whole point: authorize, reserve, dispatch, settle exactly once.
The now-retired `workflow.py` called `record_tool_executed()` only after
`run_check` returned, so a backend that raised there left no receipt at all --
the crash was invisible in `receipts.jsonl`. Reserving before dispatch makes
that crash visible in-process: the call to the backend below is deliberately
outside any `try`/`except`, so a raising backend still leaves a `RESERVED` receipt in
`ledger.receipts()`. Durability across a process restart -- so the crash is
visible after the fact too, not just to a caller still holding the ledger --
is Milestone 2's job, once graph state is checkpointed to SQLite.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from causalops.domain import (
    Budgets,
    CheckOutcome,
    Clock,
    Evidence,
    IncidentScope,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RetrievalMode,
    RunbookCheckOutcome,
    RunbookPassage,
    ToolOutcome,
    ToolProposal,
    ToolReceipt,
)
from causalops.evidence import build_evidence, new_opaque_id
from causalops.policy import authorize
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    QueryLogsArguments,
    QueryMetricArguments,
    SearchRunbooksArguments,
    ToolName,
)


class ReceiptAlreadySettled(Exception):
    """A receipt was settled twice, or settled without ever being reserved."""

    def __init__(self, receipt_id: str) -> None:
        super().__init__(
            f"receipt {receipt_id} is not a reserved receipt waiting to settle"
        )
        self.receipt_id = receipt_id


class ReservationLedger:
    """Owns the real remaining-slot count for one investigation's checks, and
    is the single authoritative list of every receipt -- denied, reserved, or
    settled -- this dispatch loop has produced.

    `authorize()`'s `tools_remaining` argument is an advisory snapshot the
    caller supplies. This ledger is the thing that actually spends a slot, in
    the same call that records the `RESERVED` receipt, so nothing can
    authorize two checks against one remaining slot. A denied proposal is
    recorded too (via `record()`) but never spends a slot -- the same rule
    the now-retired loop already enforced (`test_graph.py`'s
    `test_a_denied_proposal_costs_a_model_call_but_no_check_slot`, ported
    from the loop's own test of the same name), now enforced here as well.

    `slots_left()` counts only receipts whose `policy_result` is `ALLOWED`
    (`RESERVED` or `SETTLED`, either way), so it cannot drift from the
    receipt list and a denial cannot spend budget it never used. This whole
    class is a stopgap regardless: once Milestone 2 checkpoints graph state
    to SQLite, the receipts themselves are what has to survive a restart, and
    this bookkeeping becomes free functions over a `receipts:
    tuple[ToolReceipt, ...]` tuple living in graph state -- no separate
    ledger object required.
    """

    def __init__(self, executed_tools_budget: int) -> None:
        self._budget = executed_tools_budget
        self._receipts: dict[str, ToolReceipt] = {}
        self._evidence: dict[str, Evidence] = {}

    @classmethod
    def from_receipts(
        cls, receipts: Sequence[ToolReceipt], executed_tools_budget: int
    ) -> Self:
        """Rebuild a ledger from a prior dispatch's full receipt list.

        Graph state holds receipts as plain JSON, not a live ledger -- see
        the class docstring above on why nothing survives off-state. A graph
        dispatch node reconstructs a ledger this way on every call, so
        `slots_left()` always reflects exactly what state already recorded
        and a rebuilt ledger can never drift from the one that wrote those
        receipts in the first place.
        """
        ledger = cls(executed_tools_budget)
        for receipt in receipts:
            if receipt.receipt_id in ledger._receipts:
                raise ValueError(
                    f"duplicate receipt_id {receipt.receipt_id} in from_receipts"
                )
            ledger._receipts[receipt.receipt_id] = receipt
        return ledger

    def slots_left(self) -> int:
        spent = sum(
            1
            for receipt in self._receipts.values()
            if receipt.policy_result is PolicyResult.ALLOWED
        )
        return self._budget - spent

    def reserve(
        self,
        *,
        incident_id: str,
        tool: ToolName,
        fingerprint: str,
        requested_at: datetime,
    ) -> ToolReceipt | None:
        """Atomically spend one slot and record a `RESERVED` receipt, or
        refuse without spending anything if none remain."""
        if self.slots_left() <= 0:
            return None
        receipt = ToolReceipt(
            receipt_id=new_opaque_id(),
            incident_id=incident_id,
            tool=tool,
            fingerprint=fingerprint,
            policy_result=PolicyResult.ALLOWED,
            state=ReceiptState.RESERVED,
            requested_at=requested_at,
            duration_ms=0,
        )
        self._receipts[receipt.receipt_id] = receipt
        return receipt

    def settle(
        self,
        *,
        receipt_id: str,
        outcome: ToolOutcome,
        reason_code: ReasonCode | None,
        duration_ms: int,
        result_digest: str | None,
        evidence_id: str | None,
        evidence: Evidence | None = None,
    ) -> ToolReceipt:
        """Replace a reserved receipt with its settled result. Refuses a
        second settle, and refuses settling a receipt that was never
        reserved -- constructing a new `ToolReceipt`, never
        `model_copy(update=...)`, which would skip the lifecycle validator.

        `evidence`, when given, is stored here too, keyed by `receipt_id` --
        symmetric with how the receipt itself is already durable in this
        ledger the instant this call returns. Optional and defaulted to
        `None` so every existing caller that only ever had a digest and an id
        to give (never the `Evidence` object itself) is unaffected. This is
        the fix for the settle-then-crash window `_make_wrapper.dispatch`
        used to leave open: before this, only the receipt's `evidence_id`/
        `result_digest` survived a crash between this call returning and the
        `DispatchResult` it builds being handed back to the caller: the
        `Evidence` record those fields point at died with the wrapper's frame.
        See `evidence()` below and `graph.py`'s `dispatch_tool`, which reads it
        in its own `except` handler."""
        current = self._receipts.get(receipt_id)
        if current is None or current.state is not ReceiptState.RESERVED:
            raise ReceiptAlreadySettled(receipt_id)
        settled = ToolReceipt(
            receipt_id=current.receipt_id,
            incident_id=current.incident_id,
            tool=current.tool,
            fingerprint=current.fingerprint,
            policy_result=current.policy_result,
            state=ReceiptState.SETTLED,
            outcome=outcome,
            reason_code=reason_code,
            requested_at=current.requested_at,
            duration_ms=duration_ms,
            result_digest=result_digest,
            evidence_id=evidence_id,
        )
        self._receipts[receipt_id] = settled
        if evidence is not None:
            self._evidence[receipt_id] = evidence
        return settled

    def record(self, receipt: ToolReceipt) -> None:
        """Record an already-settled receipt that skipped reservation
        entirely -- a denial, which never spends a check slot (see
        `slots_left`). Raises on a receipt still `RESERVED` (denials never
        enter that lifecycle), a `receipt_id` already present, or anything
        other than a denial: an `ALLOWED` receipt must go through
        `reserve()`/`settle()` so it spends the slot `slots_left()` accounts
        for, not through here."""
        if receipt.state is not ReceiptState.SETTLED:
            raise ValueError(f"receipt {receipt.receipt_id} is not settled")
        if receipt.receipt_id in self._receipts:
            raise ValueError(f"receipt {receipt.receipt_id} was already recorded")
        if receipt.policy_result is not PolicyResult.DENIED:
            raise ValueError(
                f"receipt {receipt.receipt_id} is not a denial -- an ALLOWED "
                "receipt must go through reserve()/settle(), not record()"
            )
        self._receipts[receipt.receipt_id] = receipt

    def receipts(self) -> tuple[ToolReceipt, ...]:
        """Every receipt this ledger has produced, in call order: denied,
        reserved, and settled alike."""
        return tuple(self._receipts.values())

    def evidence(self) -> tuple[Evidence, ...]:
        """Every `Evidence` record a `settle()` call durably recorded, in
        call order. A ledger built by `from_receipts` never populates this --
        prior evidence already lives in graph state's own `evidence` list, so
        only the current dispatch's own settle (if it ran before a crash)
        ever appears here. That is what makes it safe for a caller to append
        this tuple onto state's evidence list without checking for
        duplicates."""
        return tuple(self._evidence.values())


class DispatchResult(BaseModel):
    """What one dispatch attempt produced: the receipt, settled or denied,
    the evidence it produced, if any, and the free-text `message` a denial
    carries (`PolicyDecision.message`). The message lives here, not on
    `ToolReceipt`, so nothing that serialises a receipt into graph state
    changes shape -- it survives exactly long enough for the caller (today,
    `graph.py`'s `dispatch_tool`) to put it in a `proposal_denied` event.
    Defaulted so the seven other call sites that only ever look at `receipt`
    or `evidence` are unaffected.

    `passages`/`retrieval_mode` are Unit 3a's own additions, for exactly one
    tool: `search_runbooks`. `_make_wrapper` below branches on the *type* of
    result its `run_check` seam returns (`RunbookCheckOutcome` vs.
    `CheckOutcome`) to decide which of `evidence` or `passages` to populate
    -- never both -- so a `DispatchResult` for any of the other four tools
    always carries `passages=()`. `retrieval_mode` is set from
    `RunbookCheckOutcome.retrieval_mode` whenever that branch runs, even
    when `passages` ends up empty -- see `RetrievalMode`'s own docstring for
    why `graph.py` must not infer it from an empty passage list instead."""

    model_config = ConfigDict(frozen=True)

    receipt: ToolReceipt
    evidence: Evidence | None = None
    passages: tuple[RunbookPassage, ...] = ()
    retrieval_mode: RetrievalMode | None = None
    message: str = ""


DispatchFn = Callable[
    [ToolProposal, IncidentScope, set[str], Budgets, ReservationLedger, Clock],
    DispatchResult,
]


# Only the factories below hold this. A `ToolWrapper` built without it raises
# at construction time -- see `ToolWrapper.__post_init__`.
_WRAPPER_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class ToolWrapper:
    """A dispatch-registry entry. Construction is enforced, not just
    conventional: `__post_init__` rejects any instance not built with
    `_WRAPPER_FACTORY_TOKEN`, a module-private sentinel only the factories
    below hold, so a hand-built `ToolWrapper` around an arbitrary closure
    raises `TypeError` instead of silently joining the registry. The
    wrapper-identity test in `test_tool_boundary.py` checks exactly that --
    not only `isinstance(x, ToolWrapper)`, which a hand-built instance would
    also satisfy, but that direct construction is refused.
    """

    tool: ToolName
    dispatch: DispatchFn
    _factory_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _WRAPPER_FACTORY_TOKEN:
            raise TypeError(
                "ToolWrapper must be built by a wrapper factory "
                "(e.g. query_logs_wrapper), not constructed directly"
            )


def _denied_receipt(
    proposal: ToolProposal,
    scope: IncidentScope,
    fingerprint: str,
    reason_code: ReasonCode | None,
    message: str,
    clock: Clock,
) -> DispatchResult:
    receipt = ToolReceipt(
        receipt_id=new_opaque_id(),
        incident_id=scope.incident_id,
        tool=proposal.tool,
        fingerprint=fingerprint,
        policy_result=PolicyResult.DENIED,
        outcome=ToolOutcome.NOT_EXECUTED,
        reason_code=reason_code,
        requested_at=clock(),
        duration_ms=0,
    )
    return DispatchResult(receipt=receipt, message=message)


def _make_wrapper[ArgsT: BaseModel](
    tool: ToolName,
    arguments_type: type[ArgsT],
    run_check: Callable[[ArgsT, IncidentScope], CheckOutcome | RunbookCheckOutcome],
) -> ToolWrapper:
    """Builds the one path from a parsed proposal for `tool` to a backend.

    `run_check` is the backend seam, mirroring `telemetry.py`'s own private
    `_registered_check_runner`'s closure pattern (Unit 3b-4 addendum, C6:
    made private there, since nothing in `cli.py` calls it) -- this module
    needs no import of `telemetry.py` or `prometheus.py` themselves, which
    is exactly what the AST import test checks.

    The seam takes the `IncidentScope` for all five tools, even though only
    `query_metric`'s backend reads it (the PromQL `incident` label --
    `prometheus.py:177-179` -- is cross-incident isolation, and unlike
    `paths`/`base_url`/`timeout` it is per-dispatch, not something a caller
    can close over). The other four ignore it -- `search_runbooks` included,
    since guidance is not incident-scoped (`policy.py`'s own branch for it
    never reads `scope` either). One seam shape beats one identical one per
    tool plus a fifth that differs only in an unused parameter.

    `arguments_type` narrows `proposal.arguments` -- a `ToolArguments` union
    member -- down to `ArgsT` for the `isinstance` check below, and that
    narrowing is what lets `run_check(proposal.arguments, scope)` type-check
    under `mypy --strict`: `ArgsT` is inferred once, consistently, from both
    `arguments_type` and `run_check`'s own parameter type, so a factory call
    that pairs the wrong backend with the wrong argument type is a `mypy`
    error, not a runtime one. That binding is only between `arguments_type`
    and `run_check` -- `tool` is a plain `ToolName` with no type relationship
    to either, so mypy cannot catch a factory call where `tool` itself
    disagrees with `arguments_type` (`_make_wrapper(ToolName.GET_TOPOLOGY,
    QueryLogsArguments, run_logs)` type-checks fine). The error message below
    names what this wrapper instance actually expects and actually got,
    never `tool`, so it stays informative even under that kind of
    construction-time mismatch.

    Unit 3a widens `run_check`'s return type to `CheckOutcome |
    RunbookCheckOutcome` -- this factory is shared by all five registered
    tools, and `dispatch` below branches on which one of those two shapes
    came back to decide whether the result is incident evidence or runbook
    guidance. That widening is deliberately *not* repeated on the five named
    alias functions below or on `dispatch_registry`'s own keyword
    parameters: each keeps the narrow return type its own tool actually
    produces (`CheckOutcome` for the first four, `RunbookCheckOutcome` for
    `search_runbooks_wrapper` alone). Python's `Callable` return type is
    covariant, so a narrowly-typed backend still satisfies this factory's
    wider parameter with no cast -- but the reverse is not offered: nothing
    here lets a caller wire a `RunbookCheckOutcome`-returning backend into
    `query_logs_wrapper`'s `run_logs` parameter, because that parameter is
    typed for `CheckOutcome` only. Those five narrow annotations, not this
    factory's internal union, are what stops a retrieval backend from being
    wired to `query_logs` and type-checking clean -- the same class of
    mismatch this docstring already concedes `mypy` cannot catch through
    `tool` alone, closed here instead through each alias's own signature.
    """

    def dispatch(
        proposal: ToolProposal,
        scope: IncidentScope,
        seen_fingerprints: set[str],
        budgets: Budgets,
        ledger: ReservationLedger,
        clock: Clock,
    ) -> DispatchResult:
        if not isinstance(proposal.arguments, arguments_type):
            raise ValueError(
                f"{tool.value} wrapper expects {arguments_type.__name__}, "
                f"got {type(proposal.arguments).__name__}"
            )
        decision = authorize(
            proposal, scope, seen_fingerprints, budgets, ledger.slots_left()
        )
        # A fingerprint is marked seen whether the decision allows or denies
        # it, matching the retired loop's own order: a denial is not a reason
        # to let the same proposal be retried.
        seen_fingerprints.add(decision.fingerprint)
        if decision.result is PolicyResult.DENIED:
            result = _denied_receipt(
                proposal,
                scope,
                decision.fingerprint,
                decision.reason_code,
                decision.message,
                clock,
            )
            ledger.record(result.receipt)
            return result

        reserved = ledger.reserve(
            incident_id=scope.incident_id,
            tool=proposal.tool,
            fingerprint=decision.fingerprint,
            requested_at=clock(),
        )
        if reserved is None:
            # authorize() was fed ledger.slots_left() directly above, so the
            # two can never disagree -- reaching here would mean that
            # invariant broke, which is worth failing loudly for rather than
            # manufacturing a denial that would read as a real policy
            # decision in evaluation data.
            raise AssertionError(
                "ledger reservation refused immediately after authorize() "
                "used the same slots_left() value -- should be unreachable"
            )

        outcome = run_check(  # not caught -- see module docstring
            proposal.arguments, scope
        )

        if isinstance(outcome, RunbookCheckOutcome):
            # The retrieval branch: a passage is guidance, never `Evidence`
            # -- `TECHNICAL_SPEC.md` §6/§7 draw that line, `EvidenceKind`'s
            # own docstring explains why nothing here can blur it.
            # `result_digest`/`evidence_id` stay `None` regardless of
            # outcome, matching `Evidence`-free settlement; `retrieval_mode`
            # is carried onto `DispatchResult` from the outcome even when
            # `passages` is empty (a failed or empty search still ran in a
            # real mode) -- see `RetrievalMode`'s own docstring.
            settled = ledger.settle(
                receipt_id=reserved.receipt_id,
                outcome=outcome.outcome,
                reason_code=outcome.reason_code,
                duration_ms=outcome.duration_ms,
                result_digest=None,
                evidence_id=None,
            )
            return DispatchResult(
                receipt=settled,
                passages=outcome.passages,
                retrieval_mode=outcome.retrieval_mode,
            )

        # The unchanged four-tool path: every successful check still mints
        # an incident-scoped `Evidence` record unconditionally on `EXECUTED`.
        evidence = None
        if outcome.outcome is ToolOutcome.EXECUTED:
            evidence = build_evidence(
                incident_id=scope.incident_id,
                kind=outcome.kind,
                source=outcome.source,
                observed_at=clock(),
                summary=outcome.summary,
                payload=outcome.payload,
                receipt_id=reserved.receipt_id,
            )
        settled = ledger.settle(
            receipt_id=reserved.receipt_id,
            outcome=outcome.outcome,
            reason_code=outcome.reason_code,
            duration_ms=outcome.duration_ms,
            result_digest=evidence.content_hash if evidence else None,
            evidence_id=evidence.evidence_id if evidence else None,
            evidence=evidence,
        )
        return DispatchResult(receipt=settled, evidence=evidence)

    return ToolWrapper(
        tool=tool, dispatch=dispatch, _factory_token=_WRAPPER_FACTORY_TOKEN
    )


def query_metric_wrapper(
    run_metric: Callable[[QueryMetricArguments, IncidentScope], CheckOutcome],
) -> ToolWrapper:
    """Thin named alias over `_make_wrapper` -- see its docstring. The one
    wrapper whose backend seam actually reads the `IncidentScope` it is
    given."""
    return _make_wrapper(ToolName.QUERY_METRIC, QueryMetricArguments, run_metric)


def query_logs_wrapper(
    run_logs: Callable[[QueryLogsArguments, IncidentScope], CheckOutcome],
) -> ToolWrapper:
    """Thin named alias over `_make_wrapper` -- see its docstring."""
    return _make_wrapper(ToolName.QUERY_LOGS, QueryLogsArguments, run_logs)


def list_recent_changes_wrapper(
    run_changes: Callable[[ListRecentChangesArguments, IncidentScope], CheckOutcome],
) -> ToolWrapper:
    """Thin named alias over `_make_wrapper` -- see its docstring."""
    return _make_wrapper(
        ToolName.LIST_RECENT_CHANGES, ListRecentChangesArguments, run_changes
    )


def get_topology_wrapper(
    run_topology: Callable[[GetTopologyArguments, IncidentScope], CheckOutcome],
) -> ToolWrapper:
    """Thin named alias over `_make_wrapper` -- see its docstring. `policy.authorize`
    gives `get_topology` its own branch (a cross-incident id is the only way to
    deny it), but that is a policy-module concern, invisible here."""
    return _make_wrapper(ToolName.GET_TOPOLOGY, GetTopologyArguments, run_topology)


def search_runbooks_wrapper(
    run_search: Callable[[SearchRunbooksArguments, IncidentScope], RunbookCheckOutcome],
) -> ToolWrapper:
    """Thin named alias over `_make_wrapper` -- see its docstring. The one
    wrapper whose backend seam returns `RunbookCheckOutcome`, not
    `CheckOutcome` -- narrowly typed here on purpose, see `_make_wrapper`'s
    own docstring for why that narrowing, not this factory's internal
    union, is what keeps a retrieval backend from being wired to the wrong
    tool. `policy.authorize` gives `search_runbooks` its own branch too (a
    passage limit above `budgets.runbook_passages` is the only way to deny
    it), invisible here the same way `get_topology`'s is."""
    return _make_wrapper(ToolName.SEARCH_RUNBOOKS, SearchRunbooksArguments, run_search)


def dispatch_registry(
    *,
    run_metric: Callable[[QueryMetricArguments, IncidentScope], CheckOutcome],
    run_logs: Callable[[QueryLogsArguments, IncidentScope], CheckOutcome],
    run_changes: Callable[[ListRecentChangesArguments, IncidentScope], CheckOutcome],
    run_topology: Callable[[GetTopologyArguments, IncidentScope], CheckOutcome],
    run_search: Callable[[SearchRunbooksArguments, IncidentScope], RunbookCheckOutcome],
) -> dict[ToolName, ToolWrapper]:
    """Every value here is wrapper-produced, for all five registered tools.

    Keyword-only, and all five required: a call built for the old
    signature fails immediately with a `TypeError`, not a silently
    reordered or partially-defaulted backend, and there is no way to build
    a partial registry through this function at all. A caller that
    genuinely needs a partial registry (see
    `test_an_unwrapped_tool_proposal_is_refused_before_a_backend_is_reached`)
    builds the `dict[ToolName, ToolWrapper]` directly -- it is exactly the
    type this function returns.
    """
    return {
        ToolName.QUERY_METRIC: query_metric_wrapper(run_metric),
        ToolName.QUERY_LOGS: query_logs_wrapper(run_logs),
        ToolName.LIST_RECENT_CHANGES: list_recent_changes_wrapper(run_changes),
        ToolName.GET_TOPOLOGY: get_topology_wrapper(run_topology),
        ToolName.SEARCH_RUNBOOKS: search_runbooks_wrapper(run_search),
    }
