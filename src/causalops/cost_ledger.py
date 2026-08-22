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


def _reserved_and_settled_total(conn: sqlite3.Connection) -> float:
    """Every dollar this application has ever reserved, across every run --
    `TECHNICAL_SPEC.md` §10's ceiling is application-wide, not
    per-investigation, so this sums the whole table, not one `run_id`.
    Reading `reserved_usd` for *every* row (settled or still outstanding)
    rather than `actual_usd` for settled ones is deliberate: a `RESERVED`
    row that never settles (crash, timeout, missing usage) must keep
    counting against the ceiling at its reserved amount -- §10's own
    "ambiguous requests retain [the reservation]" rule -- or a crash loop
    could spend past the cap one silently-forgotten reservation at a time.
    """
    total = conn.execute("SELECT COALESCE(SUM(reserved_usd), 0.0) FROM cost_ledger")
    (value,) = total.fetchone()
    return float(value)


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
) -> CostLedgerRow:
    """Reserve one request's worst-case cost against the application-wide
    ceiling, or refuse before anything is sent.

    Idempotent on the amended §5 key: an identical retry (the same stage
    re-rendered byte-for-byte after a crash between reserving and settling)
    reads the existing row back rather than reserving twice -- the same
    "record before the risky operation, retry reads back" rule
    `approvals.py`'s `record_decision_before_resume` establishes for owner
    decisions, applied here to money instead of a decision.

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
            return existing
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
    return row


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
                CheckpointStoreReasonCode.STORE_UNAVAILABLE,
                f"no RESERVED cost_ledger row for {(run_id, graph_phase, model_turn)} "
                "to settle -- settle called without a prior reservation",
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
    return row
