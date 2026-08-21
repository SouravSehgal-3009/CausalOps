"""Unit 2c: the owner's accept/reject decision on a paused investigation.

Two things live in one module because they share one job: the resume value
`graph.py`'s `escalation_interrupt` node sees and the row `owner_decisions`
keeps must never disagree about what the owner actually decided.
`OwnerDecision` normalizes and validates a decision exactly once, at
construction, at the CLI boundary -- before either durable write -- so both
copies hold identical bytes.

This is the first code in the project that takes an authorization
instruction from outside the process (`TECHNICAL_SPEC.md` §12's dual-review
trigger for this unit): `causalops approve`/`reject` run in a second process
from the one that paused, with nothing but a thread id and, for a reject,
free text a person typed.

`CheckpointStoreError`/`CheckpointStoreReasonCode` (Unit 2d) are named for
`results/checkpoints.db`, not for "approval," even though they are defined
here: `owner_decisions` and LangGraph's own `checkpoints`/`writes` tables
live in that one physical file, and `cli.py`'s `_sqlite_checkpointer` --
used unconditionally by a plain `causalops investigate`, with no approval in
sight -- raises the same type on a failure to open it. The name was
`ApprovalError` through Unit 2c, when this module was the only thing that
ever opened that file; keeping that name once a second, approval-free
caller needed it too would have left it wrong at half its call sites.
"""

import sqlite3
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from causalops.tools import UtcDatetime


class CheckpointStoreReasonCode(StrEnum):
    """Stable codes for a refusal reading or writing `results/checkpoints.db`.

    A new vocabulary, not `causalops.domain.ReasonCode` (receipt/report
    outcomes) or `causalops.scenario_control.LabReasonCode` (lab/scenario
    commands) -- these failures are about an authorization instruction from
    outside the process, or about the local store itself, a genuinely
    different category from either. `THREAD_NOT_FOUND`,
    `NO_PENDING_INTERRUPT`, `CONFLICTING_DECISION`, and
    `INVALID_REJECTION_NOTE` are reachable only through `causalops
    approve`/`reject`; `STORE_UNAVAILABLE` is also raised by
    `cli._sqlite_checkpointer`, reachable from a plain `causalops
    investigate` too, since both open the same file.
    """

    THREAD_NOT_FOUND = "THREAD_NOT_FOUND"
    NO_PENDING_INTERRUPT = "NO_PENDING_INTERRUPT"
    CONFLICTING_DECISION = "CONFLICTING_DECISION"
    INVALID_REJECTION_NOTE = "INVALID_REJECTION_NOTE"
    STORE_UNAVAILABLE = "STORE_UNAVAILABLE"


class CheckpointStoreError(Exception):
    """`results/checkpoints.db` could not be opened or used, with one stable
    reason code -- the same shape `causalops.run_records.RunRecordError` and
    `causalops.scenario_control.LabError` already use, so `cli.py`'s `main`
    can format all three into one `FAIL <CODE> <message>` contract."""

    def __init__(self, reason_code: CheckpointStoreReasonCode, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class OwnerDecision(BaseModel):
    """One accept or reject, validated at the CLI boundary before anything
    durable is written.

    `rejection_note` is required exactly on `"reject"` and forbidden on
    `"accept"` -- the same pairing `causalops.domain.EscalationRecord`
    enforces on the finalized report, checked here first, at construction,
    so a malformed note fails loudly with nothing durable written yet and
    no thread left stuck behind a record it can never satisfy.
    `graph.py`'s `escalation_interrupt` node enforces the identical pairing
    a third time on the resume value itself, since the CLI is not the only
    caller of `Command(resume=...)` -- tests call it directly, and nothing
    stops a future caller from doing the same.

    Whitespace is stripped before the length check, so `reject "   "` is
    refused exactly like omitting the reason is -- both would otherwise
    leave a database row and a report field with no real content. Overflow
    is refused, never truncated: a silent truncation loses whatever the
    owner wrote past the bound, with no later assertion able to see it.
    Normalizing once here, ahead of both durable writes, is what guarantees
    the `owner_decisions` row and `EscalationRecord.rejection_note` hold
    identical bytes.
    """

    model_config = ConfigDict(frozen=True)

    decision: Literal["accept", "reject"]
    rejection_note: str | None = Field(default=None, max_length=300)

    @model_validator(mode="before")
    @classmethod
    def _normalize_note(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        note = data.get("rejection_note")
        if not isinstance(note, str):
            return data
        return {**data, "rejection_note": note.strip() or None}

    @model_validator(mode="after")
    def _check_pairing(self) -> Self:
        if self.decision == "reject" and not self.rejection_note:
            raise ValueError("reject requires a non-empty rejection note")
        if self.decision == "accept" and self.rejection_note is not None:
            raise ValueError("accept must not carry a rejection note")
        return self

    def resume_value(self) -> dict[str, JsonValue]:
        """The exact shape `graph.py`'s `escalation_interrupt` node
        validates against `Command(resume=...)` (`_parse_resume_decision`)."""
        return {"decision": self.decision, "rejection_note": self.rejection_note}


class DecisionRow(BaseModel):
    """One append-only row read back from `owner_decisions`.

    `check_rejection_note_pairing` mirrors `causalops.domain.EscalationRecord`'s
    validator of the same name: a row is only ever written by
    `record_decision_before_resume` from an already-validated `OwnerDecision`,
    but a row is data at rest, not a value this module fully controls end to
    end -- `sqlite3` has no schema-level way to enforce "non-null iff
    decision='reject'" the way the composite primary key enforces
    uniqueness, and a hand-edited or corrupted database row is exactly the
    input this validator exists to refuse rather than pass through.
    """

    model_config = ConfigDict(frozen=True)

    thread_id: str
    checkpoint_id: str
    decision: Literal["accept", "reject"]
    rejection_note: str | None
    decided_at: UtcDatetime

    @model_validator(mode="after")
    def check_rejection_note_pairing(self) -> Self:
        if self.decision == "reject" and not (
            self.rejection_note and self.rejection_note.strip()
        ):
            raise ValueError("a rejection row must carry a non-empty rejection note")
        if self.decision == "accept" and self.rejection_note is not None:
            raise ValueError("an acceptance row must not carry a rejection note")
        return self

    def matches(self, owner_decision: OwnerDecision) -> bool:
        """Whether a fresh request is the *same* decision as this row, not
        merely the same accept/reject choice. The note is the field most
        likely to legitimately change when a person retries -- re-explaining
        themselves -- so a decision-only comparison would silently discard
        the corrected reasoning instead of flagging the retry as
        conflicting."""
        return (
            self.decision == owner_decision.decision
            and self.rejection_note == owner_decision.rejection_note
        )


def ensure_decisions_table(conn: sqlite3.Connection) -> None:
    """`owner_decisions` coexists with `SqliteSaver`'s own `checkpoints`/
    `writes` tables in the same `results/checkpoints.db` file -- no second
    database (`CLAUDE.md`'s constraints forbid one, and the spec already
    assigns approval records to SQLite). Idempotent, so every call site can
    call it before every read or write without coordinating who ran it
    first."""
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS owner_decisions (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                rejection_note TEXT,
                decided_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_id)
            )
            """
        )
        conn.commit()
    except sqlite3.Error as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"owner_decisions table unavailable: {error}",
        ) from error


def read_decision_for_thread(
    conn: sqlite3.Connection, thread_id: str
) -> DecisionRow | None:
    """The retry-detection read: by `thread_id` alone, not the composite
    `(thread_id, checkpoint_id)` primary key the write below uses.

    A settled thread's *current* checkpoint id is not the one its decision
    was written against -- `escalation_interrupt` and `final_report` each
    commit a checkpoint after the resume, so the composite key identifies
    the write, never a later lookup against an already-advanced checkpoint.
    Unit 2c's reachable state has at most one decision per thread (there is
    no "approve one more check" path yet -- see `graph.py`'s module
    docstring and the `TECHNICAL_SPEC.md` Unit 2c amendment), so the most
    recent row for a thread is unambiguous.
    """
    try:
        row = conn.execute(
            "SELECT thread_id, checkpoint_id, decision, rejection_note, decided_at "
            "FROM owner_decisions WHERE thread_id = ? "
            "ORDER BY decided_at DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    except sqlite3.Error as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"owner_decisions unreadable: {error}",
        ) from error
    if row is None:
        return None
    # A row this module itself wrote can never fail `DecisionRow`'s own
    # validation -- `record_decision_before_resume` only ever inserts an
    # already-validated `OwnerDecision`. A row that *does* fail it (a bad
    # `decision` literal, an unparseable `decided_at`, a mis-paired note --
    # hand-edited or corrupted) is this store's own problem, not the
    # caller's decision to sort out, so it is reported the same way a
    # `sqlite3.Error` above already is: a refusal, not a traceback.
    try:
        return DecisionRow(
            thread_id=row[0],
            checkpoint_id=row[1],
            decision=row[2],
            rejection_note=row[3],
            decided_at=row[4],
        )
    except ValidationError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"owner_decisions holds an unreadable row for {thread_id}: {error}",
        ) from error


def record_decision_before_resume(
    conn: sqlite3.Connection,
    thread_id: str,
    checkpoint_id: str,
    owner_decision: OwnerDecision,
    decided_at: datetime,
) -> None:
    """Writes one append-only row before the caller ever calls
    `Command(resume=...)` (`TECHNICAL_SPEC.md:170-172`'s record-before-resume
    rule) -- no update or delete path exists anywhere in this module.

    Called only after the caller has already confirmed, via
    `read_decision_for_thread`, that no row exists yet for this thread at
    all; an `IntegrityError` here is therefore a genuine race between two
    concurrent callers racing to record the first decision for the same
    paused checkpoint, not the expected retry path -- a retry is handled
    entirely by the caller's own `read_decision_for_thread`/`.matches()`
    check before this function is ever called.
    """
    try:
        conn.execute(
            "INSERT INTO owner_decisions "
            "(thread_id, checkpoint_id, decision, rejection_note, decided_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                checkpoint_id,
                owner_decision.decision,
                owner_decision.rejection_note,
                decided_at.isoformat(),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.CONFLICTING_DECISION,
            f"{thread_id} already has a recorded decision for checkpoint "
            f"{checkpoint_id}",
        ) from error
    except sqlite3.Error as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.STORE_UNAVAILABLE,
            f"owner_decisions unwritable: {error}",
        ) from error
