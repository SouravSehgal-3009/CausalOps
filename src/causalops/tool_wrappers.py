"""The only path from a parsed tool proposal to a backend.

`TECHNICAL_SPEC.md` §5 makes a direct backend binding a P0 trust-boundary
violation: a dispatch node may reach a backend only through a policy wrapper,
never a bare backend function. This module is that wrapper for `query_logs`,
the first of the four tools (the other three follow in Unit 1c once this
shape is proven against a real backend).

Nothing here imports `causalops.telemetry` or `causalops.prometheus` -- see
`tests/security/test_tool_boundary.py` for the AST test that checks that, plus
the wrapper-identity and spy-backend tests that check what an import scan
cannot: that every registered dispatch callable was actually produced by the
factory below, and that a denied proposal never reaches a backend.

Order is the whole point: authorize, reserve, dispatch, settle exactly once.
`workflow.py:302` calls `record_tool_executed()` only after `run_check`
returns, so a backend that raises today leaves no receipt at all -- the crash
is invisible in `receipts.jsonl`. Reserving before dispatch makes that crash
visible in-process: the call to the backend below is deliberately outside any
`try`/`except`, so a raising backend still leaves a `RESERVED` receipt in
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
    ToolOutcome,
    ToolProposal,
    ToolReceipt,
)
from causalops.evidence import build_evidence, new_opaque_id
from causalops.policy import authorize
from causalops.tools import QueryLogsArguments, ToolName


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
    the legacy loop already enforces (`test_workflow.py`'s
    `test_a_denied_proposal_costs_a_model_call_but_no_check_slot`), now
    enforced here as well.

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
    ) -> ToolReceipt:
        """Replace a reserved receipt with its settled result. Refuses a
        second settle, and refuses settling a receipt that was never
        reserved -- constructing a new `ToolReceipt`, never
        `model_copy(update=...)`, which would skip the lifecycle validator."""
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


class DispatchResult(BaseModel):
    """What one dispatch attempt produced: the receipt, settled or denied, and
    the evidence it produced, if any."""

    model_config = ConfigDict(frozen=True)

    receipt: ToolReceipt
    evidence: Evidence | None = None


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
    return DispatchResult(receipt=receipt)


def query_logs_wrapper(
    run_logs: Callable[[QueryLogsArguments], CheckOutcome],
) -> ToolWrapper:
    """Builds the one path from a parsed `query_logs` proposal to a backend.

    `run_logs` is the backend seam, mirroring `registered_check_runner`'s
    existing closure pattern in `telemetry.py` -- this module needs no import
    of `telemetry.py` itself, which is exactly what the AST import test
    checks.
    """

    def dispatch(
        proposal: ToolProposal,
        scope: IncidentScope,
        seen_fingerprints: set[str],
        budgets: Budgets,
        ledger: ReservationLedger,
        clock: Clock,
    ) -> DispatchResult:
        if not isinstance(proposal.arguments, QueryLogsArguments):
            raise ValueError(
                f"query_logs wrapper received {proposal.tool.value} arguments"
            )
        decision = authorize(
            proposal, scope, seen_fingerprints, budgets, ledger.slots_left()
        )
        # A fingerprint is marked seen whether the decision allows or denies
        # it, matching workflow.py's existing order: a denial is not a reason
        # to let the same proposal be retried.
        seen_fingerprints.add(decision.fingerprint)
        if decision.result is PolicyResult.DENIED:
            result = _denied_receipt(
                proposal, scope, decision.fingerprint, decision.reason_code, clock
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

        outcome = run_logs(proposal.arguments)  # not caught -- see module docstring
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
        )
        return DispatchResult(receipt=settled, evidence=evidence)

    return ToolWrapper(
        tool=ToolName.QUERY_LOGS,
        dispatch=dispatch,
        _factory_token=_WRAPPER_FACTORY_TOKEN,
    )


def dispatch_registry(
    run_logs: Callable[[QueryLogsArguments], CheckOutcome],
) -> dict[ToolName, ToolWrapper]:
    """Every value here is wrapper-produced. Unit 1c adds the other three
    tools; nothing may ever add a bare backend callable to this table."""
    return {ToolName.QUERY_LOGS: query_logs_wrapper(run_logs)}
