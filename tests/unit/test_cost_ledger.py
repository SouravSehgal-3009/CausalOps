"""Unit 3b-2: `cost_ledger`'s reserve-before-send gate and its settlement.

Every test here uses `sqlite3.connect(":memory:")` -- no `checkpoints.db`
file, no network, no live model. This is the durable half of the cost gate;
`test_live_model.py` covers the adapter that calls it.
"""

import sqlite3
from datetime import UTC, datetime

import pytest

from causalops.approvals import CheckpointStoreError
from causalops.cost_ledger import (
    AmbiguousReservationNotResent,
    CostCeilingExceeded,
    CostLedgerRow,
    ensure_cost_ledger_table,
    record_reservation_before_request,
    settle_reservation,
)

NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    ensure_cost_ledger_table(connection)
    return connection


def _reserve(
    conn: sqlite3.Connection,
    *,
    run_id: str = "run-1",
    graph_phase: str = "INVESTIGATE",
    model_turn: int = 0,
    context_digest: str = "digest-1",
    reserved_usd: float = 0.01,
    ceiling_usd: float = 2.00,
) -> tuple[CostLedgerRow, bool]:
    return record_reservation_before_request(
        conn,
        run_id=run_id,
        graph_phase=graph_phase,
        model_turn=model_turn,
        context_digest=context_digest,
        reserved_usd=reserved_usd,
        requested_at=NOW,
        ceiling_usd=ceiling_usd,
    )


def test_ensure_cost_ledger_table_is_idempotent(conn: sqlite3.Connection) -> None:
    ensure_cost_ledger_table(conn)
    ensure_cost_ledger_table(conn)  # must not raise a second time


def test_a_reservation_is_recorded_reserved(conn: sqlite3.Connection) -> None:
    row, is_new = _reserve(conn, reserved_usd=0.05)

    assert row.state == "RESERVED"
    assert row.reserved_usd == pytest.approx(0.05)
    assert row.actual_usd is None
    assert row.settled_at is None
    assert is_new is True


def test_an_identical_retry_reads_back_the_same_row_not_a_second_one(
    conn: sqlite3.Connection,
) -> None:
    """The mutation-critical assertion: delete the existing-row lookup in
    `record_reservation_before_request` and this test must fail, not just
    pass by coincidence. It asserts both that the *count* of rows for one
    key stays one, and that a second reservation attempt does not add a
    second dollar amount to the running total -- either alone could pass
    against a subtly wrong fix; both together cannot.

    Unit 3b-4 addendum, Group B: the ledger-level bookkeeping this test
    covers (one row, one dollar amount) is correct and unaffected by that
    fix -- what changed is that `is_new` now tells the caller which of the
    two identical-looking calls actually inserted the row, which is what
    `test_live_model.py`'s new transport-invocation test uses to refuse the
    second one before it reaches the provider."""
    first_row, first_is_new = _reserve(conn, reserved_usd=0.05)
    second_row, second_is_new = _reserve(conn, reserved_usd=0.05)

    assert first_row == second_row
    assert first_is_new is True
    assert second_is_new is False
    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 1
    total = conn.execute("SELECT SUM(reserved_usd) FROM cost_ledger").fetchone()[0]
    assert total == pytest.approx(0.05)


def test_ambiguous_reservation_message_names_the_state_key_and_next_action() -> None:
    """Direct unit coverage of the exception `_send` (`live_model.py`)
    raises when `record_reservation_before_request` reports `is_new=False`
    -- constructed here without a live model or a real send, since the
    class itself lives in this module.

    Post-freeze review, P3-4: the message used to name only the state and
    key, leaving an owner reading `FAIL AMBIGUOUS_MODEL_REQUEST ...` with
    no indication of what to do next -- a real, narrow dead end for a
    `SETTLED` row specifically (a crash after `settle_reservation` commits
    but before the LangGraph checkpoint saves leaves that thread refusing
    forever). It now says so explicitly: start a fresh investigation
    rather than resuming."""
    row = CostLedgerRow(
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
        state="RESERVED",
        reserved_usd=0.05,
        reserved_at=NOW,
    )

    error = AmbiguousReservationNotResent(row)

    assert error.row is row
    assert "RESERVED" in str(error)
    assert "run-1" in str(error)
    assert "fresh investigation" in str(error)
    assert "resuming" in str(error)


def test_a_different_key_reserves_a_second_row(conn: sqlite3.Connection) -> None:
    _reserve(conn, model_turn=0, reserved_usd=0.05)
    _reserve(conn, model_turn=1, reserved_usd=0.05)

    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 2


def test_a_reservation_over_the_ceiling_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(conn, reserved_usd=3.00, ceiling_usd=2.00)

    assert excinfo.value.reservation_usd == pytest.approx(3.00)
    assert excinfo.value.remaining_usd == pytest.approx(2.00)


def test_a_refused_reservation_writes_nothing(conn: sqlite3.Connection) -> None:
    """Mutation-critical alongside the retry test above: if the ceiling
    check ran *after* the insert instead of before, this would still raise
    -- but it would also leave a row behind, silently spending the very
    reservation it claimed to refuse."""
    with pytest.raises(CostCeilingExceeded):
        _reserve(conn, reserved_usd=3.00, ceiling_usd=2.00)

    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 0


def test_the_ceiling_accounts_for_earlier_reservations_in_the_same_run(
    conn: sqlite3.Connection,
) -> None:
    """Each individual reservation is well under the ceiling, but their sum
    is not -- the gate must refuse the second one, not just check each
    request in isolation."""
    _reserve(conn, model_turn=0, reserved_usd=1.50, ceiling_usd=2.00)

    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(conn, model_turn=1, reserved_usd=1.50, ceiling_usd=2.00)

    assert excinfo.value.remaining_usd == pytest.approx(0.50)


def test_an_unsettled_reservation_still_counts_against_the_ceiling(
    conn: sqlite3.Connection,
) -> None:
    """`TECHNICAL_SPEC.md` section 5: 'a timeout, crash, or missing provider
    usage never reissues that key ... the reservation left visible for
    accounting.' A `RESERVED` row that never settles must still spend its
    share of the ceiling on the next request -- otherwise a crash loop could
    spend past the cap one silently-forgotten reservation at a time."""
    _reserve(conn, model_turn=0, reserved_usd=2.00, ceiling_usd=2.00)
    # Never settled -- simulates a crash between reserving and settling.

    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(conn, model_turn=1, reserved_usd=0.01, ceiling_usd=2.00)

    assert excinfo.value.remaining_usd == pytest.approx(0.0)


def test_settle_reservation_records_the_real_cost(conn: sqlite3.Connection) -> None:
    _reserve(conn, reserved_usd=0.05)

    row = settle_reservation(
        conn,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
        actual_usd=0.012,
        input_tokens=1000,
        output_tokens=200,
        settled_at=NOW,
    )

    assert row.state == "SETTLED"
    assert row.actual_usd == pytest.approx(0.012)
    assert row.input_tokens == 1000
    assert row.output_tokens == 200
    assert row.settled_at is not None


def test_settling_a_never_reserved_key_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(CheckpointStoreError):
        settle_reservation(
            conn,
            run_id="ghost",
            graph_phase="INVESTIGATE",
            model_turn=0,
            context_digest="digest-1",
            actual_usd=0.01,
            input_tokens=1,
            output_tokens=1,
            settled_at=NOW,
        )


def test_settling_an_already_settled_row_is_refused(conn: sqlite3.Connection) -> None:
    """Settlement is exactly-once per reservation -- a second settle call
    for the same key (a caller bug, since every real send goes through
    `record_reservation_before_request` first) must not silently overwrite
    the first settlement's figures."""
    _reserve(conn, reserved_usd=0.05)
    settle_reservation(
        conn,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
        actual_usd=0.01,
        input_tokens=100,
        output_tokens=50,
        settled_at=NOW,
    )

    with pytest.raises(CheckpointStoreError):
        settle_reservation(
            conn,
            run_id="run-1",
            graph_phase="INVESTIGATE",
            model_turn=0,
            context_digest="digest-1",
            actual_usd=999.0,
            input_tokens=1,
            output_tokens=1,
            settled_at=NOW,
        )
