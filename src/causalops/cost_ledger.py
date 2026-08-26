"""Unit 3b-2: the application-wide cost ledger -- `TECHNICAL_SPEC.md` §10's
`LIVE_EVALUATION_MAX_USD` gate, and the amended §5 model-request idempotency
record, unified into one table.

One row *is* both things at once, not two tables kept in sync: a cost
reservation only ever exists in the context of one specific model request,
so the row's primary key -- `(run_id, graph_phase, model_turn,
context_digest)`, §5's amended key exactly -- is simultaneously "the request
this reservation belongs to" and "the request this PENDING record is for."
Splitting them would let a reservation and its request record disagree about
whether a retry is the same request or a new one.

Lives in `checkpoints.db` beside `owner_decisions`
(`approvals.py`) and LangGraph's own tables -- `TECHNICAL_SPEC.md` §5's
amendment ("SQLite stores checkpoints and approval/audit records... and the
application-wide cost ledger") admits exactly this, for the same reason
`approvals.py`'s docstring already gives for `owner_decisions`: one physical
file, not a second database `CLAUDE.md` forbids.

`CheckpointStoreError`/`CheckpointStoreReasonCode` (`approvals.py`) are
reused here for one failure only: the store itself could not be read or
written (disk error, locked file). A reservation correctly refused because
it would exceed the ceiling is not that -- it is an ordinary, expected
policy refusal, the same category `tool_wrappers.py`'s
`ReservationLedger.reserve()` already has for a spent tool-execution slot
(it returns `None` rather than raising a store error). `CostCeilingExceeded`
below is this module's equivalent, raised as a distinguishable exception
rather than returned as `None`/a sentinel because the caller
(`live_model.py`, inside `ToolCallingModel.propose`/`.respond`) has to
signal "refuse, do not repair, do not retry" up through `graph.py`'s
`ask_once` -- a `None` return there is already the graph's own vocabulary
for "invalid output, retry with a repair," which is the one thing a cost
refusal must never be mistaken for.

**What `LIVE_EVALUATION_MAX_USD` actually bounds.** This module refuses to
*authorize a new reservation* once known spend would exceed the ceiling
(`record_reservation_before_request` below) -- it is not a guarantee that the
real dollar total never exceeds the configured figure. The real cost of a
request is only known after it settles (`settle_reservation`), so a single
request whose actual bill comes in above its own conservative reservation can
push cumulative spend transiently past the ceiling, by at most that one
request's worst-case estimation error, before the next reservation check
observes it. `_reserved_and_settled_total`'s own docstring below explains why
a settled row is counted at its true cost from that point forward -- the
overrun cannot silently repeat -- but nothing can retroactively un-authorize
the request that already caused it, since refusing a request needs to happen
before its real cost exists to check against.

**`RESERVATION_CEILING_BUFFER_USD` narrows that gap; it does not close it.**
`record_reservation_before_request` stops authorizing new reservations a
fixed dollar amount short of the configured ceiling, not at the ceiling
itself, so the one request that overruns its own reservation has less room
to push real spend past the configured figure before the next check catches
up. This is defense-in-depth on top of the honest limit already described
above -- it is still not a mathematical guarantee, since a single request's
overrun is still only knowable after that request has already settled.
"""

import logging
import math
import sqlite3
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    model_validator,
)

from causalops.approvals import CheckpointStoreError, CheckpointStoreReasonCode
from causalops.tools import UtcDatetime

# `settle_reservation`'s overrun diagnostic below used to go through
# `warnings.warn`, whose default action is configuration-dependent: under
# `PYTHONWARNINGS=error` (or any `warnings.simplefilter("error")` in the
# calling process), `warnings.warn` raises instead of returning, which would
# discard the `CostLedgerRow` this function is about to return even though
# `conn.commit()` above already succeeded and the row is genuinely `SETTLED`.
# A module logger's `.warning()` call never raises under any standard
# configuration, so this function's return value can never depend on the
# caller's or environment's warning-filter settings. This is the first use
# of `logging` in this codebase; a caller who wants these diagnostics
# surfaced configures a handler on this logger (or the root logger) the
# normal way -- with no handler configured, Python's logging module already
# defaults to at most one "no handlers found" notice on stderr the first
# time a logger with no handlers is used, never a raised exception.
_LOGGER = logging.getLogger(__name__)

# A fixed dollar margin `record_reservation_before_request` reserves below
# `ceiling_usd` before authorizing a new reservation -- see this module's
# own "RESERVATION_CEILING_BUFFER_USD narrows that gap" paragraph above for
# what it does and does not guarantee. Lives here, not beside
# `DEFAULT_LIVE_EVALUATION_MAX_USD` in `live_setup.py`, because the actual
# per-request ceiling check happens only inside this module; `live_setup.py`
# and `.env.example` still document its existence for an owner reading the
# ceiling's own configuration story end to end. `live_setup.
# live_evaluation_ceiling_usd` also imports this constant directly, to
# reject a configured `LIVE_EVALUATION_MAX_USD` at or below it before a run
# ever starts -- a ceiling that small would leave `remaining` permanently
# negative here, refusing every single reservation with no indication the
# real problem is the configured ceiling itself.
#
# Sized from this project's own real live-call evidence, not picked to make
# the arithmetic merely balance:
#   - The only documented reservation-vs-actual gap on one real settled
#     request -- $0.024198 actual against a $0.022168 reservation, the
#     Unit 3b-3 smoke call's INITIAL_PLAN turn recomputed at full output
#     saturation, under the input ratio since tightened by that same
#     finding (`TECHNICAL_OVERVIEW.md`, "The smoke call's findings") -- was
#     about $0.002.
#   - The largest full live run recorded to date, across every model call
#     it made, totalled $0.059998 (`LIVE_MODEL_RELIABILITY_FINDINGS.md`,
#     call 1: `configuration_change`, 4 settled reservations).
#   - The largest theoretical SINGLE-request reservation under the current
#     pricing snapshot and token caps -- a proposal turn carrying the full
#     9,600-token prose budget plus the current 12,829-token tool-schema
#     payload, at the full 1,600-token output allowance -- prices to about
#     $0.0608 by `pricing.PricingSnapshot.reservation_usd`'s own formula.
# $0.10 comfortably clears all three: roughly 50x the one measured overrun,
# and still leaves headroom over even the largest theoretical single-request
# reservation this application can currently construct.
RESERVATION_CEILING_BUFFER_USD = 0.10


class CostCeilingExceeded(Exception):
    """A reservation was refused *before sending* because it would push
    total reserved-plus-settled spend past the application-wide ceiling.

    Carries the numbers a report/log needs to explain the refusal without a
    second query: what this one request would have cost, and how much
    headroom was actually left. Not a `CheckpointStoreError` -- see this
    module's docstring for why.
    """

    def __init__(self, reservation_usd: float, remaining_usd: float) -> None:
        super().__init__(
            f"reservation of ${reservation_usd:.4f} would exceed the "
            f"${remaining_usd:.4f} remaining under the application-wide ceiling"
        )
        self.reservation_usd = reservation_usd
        self.remaining_usd = remaining_usd


class AmbiguousReservationNotResent(Exception):
    """A reservation already existed for this exact request key before this
    call -- `record_reservation_before_request` returned it with
    `is_new=False` -- so the caller (`live_model.py`'s `_send`) refuses to
    invoke the provider under it, rather than assume a second send is safe.

    Unit 3b-4 addendum, Group B. `record_reservation_before_request` used to
    return an existing row indistinguishably from a freshly-inserted one,
    and `_send` invoked the provider unconditionally either way. A crash
    between reserving and settling, followed by a LangGraph resume that
    re-renders the identical stage (same `context_digest`), correctly read
    the SAME ledger row back (no double-counted dollar in this table) but
    still sent a SECOND real paid request under it -- the exact "reissue an
    ambiguous model request" `TECHNICAL_SPEC.md` §5 forbids, in the one
    scenario the idempotency key exists to prevent.

    Both states the pre-existing row could be in are refused the same way,
    not two different judgment calls:

    - `RESERVED`: an earlier attempt at this exact request may already be
      in flight, or may have crashed after an earlier send started but
      before this process observed the result -- this module cannot tell
      those apart, so it refuses rather than guess a second send is safe.
    - `SETTLED`: an earlier attempt already completed. There is no stored
      response to return instead of resending -- `CostLedgerRow` records
      only cost and token counts, never the model's actual output -- and
      resending would definitely pay a second time for a request that
      already succeeded once. Reconstructing or caching the historical
      response is future work if ever needed; refusing is the only choice
      available today that cannot silently cost money either way.

    Defined here, not in `live_model.py`: `graph.py` catches this alongside
    `CostCeilingExceeded` without importing `live_model.py` at all (it
    depends only on the `ToolCallingModel` protocol, never the concrete
    live adapter -- `causalops.live_model` does not appear anywhere in
    `graph.py`, confirmed by grep) -- the same reason `CostCeilingExceeded`
    itself lives here rather than in the module that raises `InputTooLarge`.
    """

    def __init__(self, row: "CostLedgerRow") -> None:
        super().__init__(
            f"a {row.state} reservation already exists for run "
            f"{row.run_id!r}, phase {row.graph_phase!r}, turn "
            f"{row.model_turn} -- refusing to send a second request under "
            "the same key. Start a fresh investigation instead of resuming "
            "this thread: resuming re-renders the identical request and "
            "hits this same refusal again, every time."
        )
        self.row = row


class CostLedgerRow(BaseModel):
    """One model request's reservation, and its settlement once the request
    has actually returned. `state == "RESERVED"` until `settle_reservation`
    updates it -- the same `RESERVED`/`SETTLED` vocabulary
    `tool_wrappers.py`'s `ReceiptState` already uses for the analogous tool
    receipt lifecycle, reused here rather than inventing a second one."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    graph_phase: str
    model_turn: int
    context_digest: str
    state: Literal["RESERVED", "SETTLED"]
    reserved_usd: float = Field(ge=0, allow_inf_nan=False)
    actual_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    reserved_at: UtcDatetime
    settled_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def lifecycle_is_complete(self) -> "CostLedgerRow":
        metadata = (
            self.actual_usd,
            self.input_tokens,
            self.output_tokens,
            self.settled_at,
        )
        if self.state == "RESERVED" and any(value is not None for value in metadata):
            raise ValueError("RESERVED rows cannot contain settlement metadata")
        if self.state == "SETTLED" and any(value is None for value in metadata):
            raise ValueError("SETTLED rows require complete settlement metadata")
        return self


# The column list every `SELECT` against `cost_ledger` reads, in the exact
# order `_parse_cost_ledger_row` below expects -- one shared string instead
# of the same 11 names retyped at each call site, which had drifted into
# three verbatim copies before this fix.
_COST_LEDGER_COLUMNS = (
    "run_id, graph_phase, model_turn, context_digest, state, reserved_usd, "
    "actual_usd, input_tokens, output_tokens, reserved_at, settled_at"
)


def _parse_cost_ledger_row(raw: tuple[Any, ...]) -> CostLedgerRow:
    """Turns one `_COST_LEDGER_COLUMNS`-shaped fetched row into a validated
    `CostLedgerRow`. `_read_row`, `run_cost_totals`, and
    `_reserved_and_settled_total` each used to repeat this same 11-field
    mapping as their own verbatim dict literal; this is the one copy.

    Raises `pydantic.ValidationError` on a malformed row rather than
    catching it -- every caller already wraps its own read in a `try` that
    converts that into a `CheckpointStoreError` with a message naming its
    own operation (settlement, ceiling total, per-run total), which this
    shared helper cannot know."""
    return CostLedgerRow(
        run_id=raw[0],
        graph_phase=raw[1],
        model_turn=raw[2],
        context_digest=raw[3],
        state=raw[4],
        reserved_usd=raw[5],
        actual_usd=raw[6],
        input_tokens=raw[7],
        output_tokens=raw[8],
        reserved_at=raw[9],
        settled_at=raw[10],
    )


def ensure_cost_ledger_table(conn: sqlite3.Connection) -> None:
    """Idempotent, matching `approvals.py`'s `ensure_decisions_table` --
    every call site can call it before every read or write without
    coordinating who ran it first."""
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_ledger (
                run_id TEXT NOT NULL,
                graph_phase TEXT NOT NULL,
                model_turn INTEGER NOT NULL,
                context_digest TEXT NOT NULL,
                state TEXT NOT NULL,
                reserved_usd REAL NOT NULL,
                actual_usd REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                reserved_at TEXT NOT NULL,
                settled_at TEXT,
                PRIMARY KEY (run_id, graph_phase, model_turn, context_digest)
            )
            """
        )
        conn.commit()
    except sqlite3.Error as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger table unavailable: {error}",
        ) from error


def _read_row(
    conn: sqlite3.Connection,
    run_id: str,
    graph_phase: str,
    model_turn: int,
    context_digest: str,
) -> CostLedgerRow | None:
    row = conn.execute(
        f"SELECT {_COST_LEDGER_COLUMNS} FROM cost_ledger "
        "WHERE run_id = ? AND graph_phase = ? AND model_turn = ? "
        "AND context_digest = ?",
        (run_id, graph_phase, model_turn, context_digest),
    ).fetchone()
    if row is None:
        return None
    try:
        return _parse_cost_ledger_row(row)
    except ValidationError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger contains an invalid row: {error}",
        ) from error


def _valid_money(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return amount


def _valid_tokens(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def run_cost_totals(conn: sqlite3.Connection, run_id: str) -> tuple[float, float, bool]:
    """`(reserved_usd, actual_usd, fully_settled)` for one `run_id` only --
    the per-run figures `EvaluationRecord` needs (Unit 3c), a different
    question from `_reserved_and_settled_total`'s application-wide ceiling
    sum right below. Reads the same table through the same connection every
    other reservation/settlement call already uses; this adds no new
    tracking mechanism, only a differently-scoped read of the one that
    exists.

    `actual_usd` sums only `SETTLED` rows -- a still-`RESERVED` row for this
    run has no real cost yet to report, and `COALESCE(SUM(...), 0.0)` on an
    empty match returns `0.0` rather than `NULL`, so a run with no rows at
    all (nothing ever sent) reports `(0.0, 0.0, True)` rather than raising.

    `fully_settled` is `True` only when every row for this `run_id` has
    reached `SETTLED` -- `actual_usd == 0.0` alone is NOT a reliable signal
    for "nothing settled": a run whose first three of four model calls
    settle normally while the fourth stays `RESERVED` (a timeout, a crash
    mid-call) has a non-zero, but PARTIAL, `actual_usd`, and reporting that
    partial sum as if it were the run's complete cost would silently
    understate it. This checks the actual row states directly rather than
    inferring completeness from whether the sum happens to be zero.
    """
    ensure_cost_ledger_table(conn)
    try:
        rows = conn.execute(
            f"SELECT {_COST_LEDGER_COLUMNS} FROM cost_ledger WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        parsed = [_parse_cost_ledger_row(row) for row in rows]
    except ValidationError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger contains an invalid row: {error}",
        ) from error
    except sqlite3.Error as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger unreadable for run totals: {error}",
        ) from error
    return (
        sum(row.reserved_usd for row in parsed),
        sum(row.actual_usd or 0.0 for row in parsed if row.state == "SETTLED"),
        all(row.state == "SETTLED" for row in parsed),
    )


def _reserved_and_settled_total(conn: sqlite3.Connection) -> float:
    """Every dollar this application has ever spent or committed to spend,
    across every run -- `TECHNICAL_SPEC.md` §10's ceiling is
    application-wide, not per-investigation, so this sums the whole table,
    not one `run_id`.

    A still-`RESERVED` row (never settled -- a crash, a timeout, missing
    provider usage) counts at its `reserved_usd` amount, unchanged from this
    function's original behaviour: §10's own "ambiguous requests retain [the
    reservation]" rule -- or a crash loop could spend past the cap one
    silently-forgotten reservation at a time.

    A `SETTLED` row counts at `max(reserved_usd, actual_usd)`, not
    `reserved_usd` alone. The reservation is meant to be a conservative
    upper bound on the real bill, but that is an assumption about the
    pricing snapshot and token estimate it was computed from, not a
    guarantee -- `settle_reservation` below warns, rather than silently
    accepting it, whenever a real bill comes in above its own reservation.
    If that ever happens and this function kept summing `reserved_usd`
    regardless, the ceiling's running total would understate real spend by
    exactly the overrun, forever: every later reservation would be approved
    against a total that no longer reflects what was actually billed. Taking
    the greater of the two closes that gap without changing anything about
    the ordinary case, where `actual_usd <= reserved_usd` and `max` is a
    no-op.

    Computed in Python over one fetch of every row, not a single SQL
    aggregate `MAX()` expression: SQL's aggregate `MAX()` collapses to one
    value across every row it matches, which answers a different question
    from the one needed here -- each row's own `(reserved_usd, actual_usd)`
    pair has to be compared independently, and only then summed.
    """
    total = 0.0
    try:
        rows = conn.execute(
            f"SELECT {_COST_LEDGER_COLUMNS} FROM cost_ledger"
        ).fetchall()
        for raw in rows:
            row = _parse_cost_ledger_row(raw)
            if row.state == "SETTLED":
                assert row.actual_usd is not None
                total += max(row.reserved_usd, row.actual_usd)
            else:
                total += row.reserved_usd
    except ValidationError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger contains an invalid row: {error}",
        ) from error
    except sqlite3.Error as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger unreadable for ceiling total: {error}",
        ) from error
    return total


def record_reservation_before_request(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    graph_phase: str,
    model_turn: int,
    context_digest: str,
    reserved_usd: float,
    requested_at: datetime,
    ceiling_usd: float,
) -> tuple[CostLedgerRow, bool]:
    """Reserve one request's worst-case cost against the application-wide
    ceiling, or refuse before anything is sent. Returns `(row, is_new)`:
    `is_new` is `True` only when this call is the one that inserted `row`.

    Authorizes a new reservation only while `accounted_spend + reserved_usd
    <= ceiling_usd - RESERVATION_CEILING_BUFFER_USD`, not up to `ceiling_usd`
    itself -- see this module's own docstring and that constant's for what
    the reserved margin narrows and why it cannot close the gap outright.

    Idempotent on the amended §5 key at the LEDGER level: an identical
    retry (the same stage re-rendered byte-for-byte after a crash between
    reserving and settling) reads the existing row back rather than
    reserving a second dollar amount -- the same "record before the risky
    operation, retry reads back" rule `approvals.py`'s
    `record_decision_before_resume` establishes for owner decisions,
    applied here to money instead of a decision.

    Unit 3b-4 addendum, Group B. Idempotent bookkeeping is NOT the same
    claim as "safe to send again": this function used to return the
    existing row indistinguishably from a freshly-inserted one, and
    `live_model.py`'s `_send` sent the request unconditionally either way
    -- a crash-then-resume with the same `context_digest` reserved against
    the same row correctly (no double-counted dollar in this table) but
    still invoked the provider a second time for real money, the exact
    "reissue an ambiguous model request" `TECHNICAL_SPEC.md` §5 forbids.
    This function still owns only the ledger's bookkeeping -- whether to
    actually send is `_send`'s decision, informed by the `is_new` flag this
    now returns, the same division of responsibility `CostCeilingExceeded`
    already has (raised here, decided how to route by `graph.py`).

    The read-check-insert sequence runs inside one `BEGIN IMMEDIATE`
    transaction so a second connection cannot insert a competing
    reservation between this function's sum and its insert -- SQLite's
    file-level write lock, not an application-level one. Proven here against
    the realistic case this codebase actually produces (a crash, then a
    sequential retry from the same or a fresh connection); not verified
    under genuine multi-threaded concurrency, which nothing in this
    single-process CLI creates today.
    """
    ensure_cost_ledger_table(conn)
    try:
        reserved_usd = _valid_money(reserved_usd, "reserved_usd")
        ceiling_usd = _valid_money(ceiling_usd, "ceiling_usd")
    except ValueError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"invalid cost ledger input: {error}",
        ) from error
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.Error as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger unavailable for reservation: {error}",
        ) from error
    try:
        existing = _read_row(conn, run_id, graph_phase, model_turn, context_digest)
        if existing is not None:
            conn.commit()
            return existing, False
        spent = _reserved_and_settled_total(conn)
        remaining = ceiling_usd - RESERVATION_CEILING_BUFFER_USD - spent
        if reserved_usd > remaining:
            conn.rollback()
            raise CostCeilingExceeded(reserved_usd, max(remaining, 0.0))
        conn.execute(
            "INSERT INTO cost_ledger "
            "(run_id, graph_phase, model_turn, context_digest, state, "
            "reserved_usd, reserved_at) VALUES (?, ?, ?, ?, 'RESERVED', ?, ?)",
            (
                run_id,
                graph_phase,
                model_turn,
                context_digest,
                reserved_usd,
                requested_at.isoformat(),
            ),
        )
        conn.commit()
    except sqlite3.Error as error:
        conn.rollback()
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger unwritable: {error}",
        ) from error
    except BaseException:
        conn.rollback()
        raise
    row = _read_row(conn, run_id, graph_phase, model_turn, context_digest)
    assert row is not None, (
        "just-committed reservation is not readable back -- cost_ledger's "
        "own write path is broken, not a caller error"
    )
    return row, True


def settle_reservation(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    graph_phase: str,
    model_turn: int,
    context_digest: str,
    actual_usd: float,
    input_tokens: int,
    output_tokens: int,
    settled_at: datetime,
) -> CostLedgerRow:
    """Records the real cost once a request has actually returned.

    Only ever updates a row already `RESERVED` for this exact key -- a
    settlement with no matching reservation is a caller bug (every send
    goes through `record_reservation_before_request` first) and is refused
    loudly rather than silently inserting a row that never went through the
    ceiling check.

    If `actual_usd` comes in ABOVE the row's own `reserved_usd` -- the
    reservation's pessimistic estimate turned out to be wrong -- this still
    commits the true `actual_usd` (the row has to record what was really
    billed, not a number massaged to look conservative) and logs a warning
    naming both figures, so the overrun is visible rather than silently
    absorbed. It is a logged diagnostic, not a raised error that could abort
    the settlement: the money is already spent by the time this function
    runs, there is nothing left to refuse, and the caller still needs the
    real `CostLedgerRow` back. Logging rather than `warnings.warn` is
    deliberate here, not a style choice -- see this module's own `_LOGGER`
    comment for why `warnings.warn` could turn an already-committed, valid
    settlement into an exception depending on the caller's warning-filter
    configuration. `_reserved_and_settled_total` above already accounts
    correctly for a row in this state once it lands (counting it at its
    true, higher cost) -- this log line is the diagnostic signal for
    whoever configured the pricing snapshot or token estimate to
    recalibrate it, not a correctness gap in the ceiling itself.
    """
    ensure_cost_ledger_table(conn)
    try:
        actual_usd = _valid_money(actual_usd, "actual_usd")
        input_tokens = _valid_tokens(input_tokens, "input_tokens")
        output_tokens = _valid_tokens(output_tokens, "output_tokens")
    except ValueError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.RESERVATION_NOT_SETTLEABLE,
            f"invalid settlement metadata: {error}",
        ) from error
    try:
        cursor = conn.execute(
            "UPDATE cost_ledger SET state = 'SETTLED', actual_usd = ?, "
            "input_tokens = ?, output_tokens = ?, settled_at = ? "
            "WHERE run_id = ? AND graph_phase = ? AND model_turn = ? "
            "AND context_digest = ? AND state = 'RESERVED'",
            (
                actual_usd,
                input_tokens,
                output_tokens,
                settled_at.isoformat(),
                run_id,
                graph_phase,
                model_turn,
                context_digest,
            ),
        )
        if cursor.rowcount == 0:
            conn.rollback()
            raise CheckpointStoreError(
                CheckpointStoreReasonCode.RESERVATION_NOT_SETTLEABLE,
                "no RESERVED cost_ledger row for request key "
                f"{(run_id, graph_phase, model_turn, context_digest)} to settle "
                "-- the row is absent or not RESERVED",
            )
        conn.commit()
    except sqlite3.Error as error:
        conn.rollback()
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"cost_ledger unwritable while settling: {error}",
        ) from error
    row = _read_row(conn, run_id, graph_phase, model_turn, context_digest)
    assert row is not None, "just-settled row vanished before it could be read back"
    if row.actual_usd is not None and row.actual_usd > row.reserved_usd:
        _LOGGER.warning(
            "cost_ledger settlement for run %r, phase %r, turn %d billed "
            "$%.4f, above its $%.4f reservation -- the pessimistic estimate "
            "under-reserved this request; recalibrate the pricing snapshot "
            "or token estimate this reservation was computed from",
            run_id,
            graph_phase,
            model_turn,
            row.actual_usd,
            row.reserved_usd,
        )
    return row
