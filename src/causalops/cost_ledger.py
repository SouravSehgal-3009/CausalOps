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
"""

import sqlite3
import warnings
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from causalops.approvals import CheckpointStoreError, CheckpointStoreReasonCode
from causalops.tools import UtcDatetime


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
    state: str
    reserved_usd: float = Field(ge=0)
    actual_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reserved_at: UtcDatetime
    settled_at: UtcDatetime | None = None


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
        "SELECT run_id, graph_phase, model_turn, context_digest, state, "
        "reserved_usd, actual_usd, input_tokens, output_tokens, "
        "reserved_at, settled_at FROM cost_ledger "
        "WHERE run_id = ? AND graph_phase = ? AND model_turn = ? "
        "AND context_digest = ?",
        (run_id, graph_phase, model_turn, context_digest),
    ).fetchone()
    if row is None:
        return None
    return CostLedgerRow(
        run_id=row[0],
        graph_phase=row[1],
        model_turn=row[2],
        context_digest=row[3],
        state=row[4],
        reserved_usd=row[5],
        actual_usd=row[6],
        input_tokens=row[7],
        output_tokens=row[8],
        reserved_at=row[9],
        settled_at=row[10],
    )


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
    reserved = conn.execute(
        "SELECT COALESCE(SUM(reserved_usd), 0.0) FROM cost_ledger WHERE run_id = ?",
        (run_id,),
    )
    (reserved_usd,) = reserved.fetchone()
    actual = conn.execute(
        "SELECT COALESCE(SUM(actual_usd), 0.0) FROM cost_ledger "
        "WHERE run_id = ? AND state = 'SETTLED'",
        (run_id,),
    )
    (actual_usd,) = actual.fetchone()
    unsettled = conn.execute(
        "SELECT COUNT(*) FROM cost_ledger WHERE run_id = ? AND state != 'SETTLED'",
        (run_id,),
    )
    (unsettled_count,) = unsettled.fetchone()
    return float(reserved_usd), float(actual_usd), unsettled_count == 0


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
    rows = conn.execute("SELECT state, reserved_usd, actual_usd FROM cost_ledger")
    total = 0.0
    for state, reserved_usd, actual_usd in rows.fetchall():
        if state == "SETTLED":
            total += max(reserved_usd, actual_usd)
        else:
            total += reserved_usd
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
        remaining = ceiling_usd - spent
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
    billed, not a number massaged to look conservative) and raises a
    `RuntimeWarning` naming both figures, so the overrun is visible rather
    than silently absorbed. It is a warning, not a raised error that could
    abort the settlement: the money is already spent by the time this
    function runs, there is nothing left to refuse, and the caller still
    needs the real `CostLedgerRow` back. `_reserved_and_settled_total`
    above already accounts correctly for a row in this state once it lands
    (counting it at its true, higher cost) -- this warning is the
    diagnostic signal for whoever configured the pricing snapshot or token
    estimate to recalibrate it, not a correctness gap in the ceiling itself.
    """
    ensure_cost_ledger_table(conn)
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
        warnings.warn(
            f"cost_ledger settlement for run {run_id!r}, phase {graph_phase!r}, "
            f"turn {model_turn} billed ${row.actual_usd:.4f}, above its "
            f"${row.reserved_usd:.4f} reservation -- the pessimistic estimate "
            "under-reserved this request; recalibrate the pricing snapshot or "
            "token estimate this reservation was computed from",
            RuntimeWarning,
            stacklevel=2,
        )
    return row
