"""`cost_ledger`'s reserve-before-send gate and its settlement.

Every test here uses `sqlite3.connect(":memory:")` -- no `checkpoints.db`
file, no network, no live model. This is the durable half of the cost gate;
`test_live_model.py` covers the adapter that calls it.
"""

import logging
import sqlite3
import warnings
from datetime import UTC, datetime

import pytest

from causalops.approvals import CheckpointStoreError, CheckpointStoreReasonCode
from causalops.cost_ledger import (
    RESERVATION_CEILING_BUFFER_USD,
    AmbiguousReservationNotResent,
    CostCeilingExceeded,
    CostLedgerRow,
    ensure_cost_ledger_table,
    record_reservation_before_request,
    run_cost_totals,
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

    The ledger-level bookkeeping this test
    covers (one row, one dollar amount) is correct and unaffected by a
    later fix -- what changed is that `is_new` now tells the caller which of the
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

    The message used to name only the state and
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
    # `ceiling_usd` carries `RESERVATION_CEILING_BUFFER_USD` on top of the
    # 2.00 this test is actually about, so the buffered remaining this
    # assertion checks reproduces the pre-buffer arithmetic exactly --
    # `test_a_reservation_within_the_raw_ceiling_but_inside_the_buffer_is_
    # refused` below is the test that exercises the buffer itself.
    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(
            conn, reserved_usd=3.00, ceiling_usd=2.00 + RESERVATION_CEILING_BUFFER_USD
        )

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


@pytest.mark.parametrize("reserved_usd", [float("nan"), float("inf"), -0.01, True])
def test_invalid_reservation_money_is_refused_before_mutation(
    conn: sqlite3.Connection, reserved_usd: object
) -> None:
    with pytest.raises(CheckpointStoreError):
        _reserve(conn, reserved_usd=reserved_usd)  # type: ignore[arg-type]

    assert conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0] == 0


def test_the_ceiling_accounts_for_earlier_reservations_in_the_same_run(
    conn: sqlite3.Connection,
) -> None:
    """Each individual reservation is well under the ceiling, but their sum
    is not -- the gate must refuse the second one, not just check each
    request in isolation."""
    # `ceiling_usd` carries the buffer on top of the 2.00 this test is
    # actually about -- see the comment on the boundary test above.
    ceiling_usd = 2.00 + RESERVATION_CEILING_BUFFER_USD
    _reserve(conn, model_turn=0, reserved_usd=1.50, ceiling_usd=ceiling_usd)

    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(conn, model_turn=1, reserved_usd=1.50, ceiling_usd=ceiling_usd)

    assert excinfo.value.remaining_usd == pytest.approx(0.50)


def test_an_unsettled_reservation_still_counts_against_the_ceiling(
    conn: sqlite3.Connection,
) -> None:
    """`TECHNICAL_SPEC.md` section 5: 'a timeout, crash, or missing provider
    usage never reissues that key ... the reservation left visible for
    accounting.' A `RESERVED` row that never settles must still spend its
    share of the ceiling on the next request -- otherwise a crash loop could
    spend past the cap one silently-forgotten reservation at a time."""
    # `ceiling_usd` carries the buffer on top of the 2.00 this test is
    # actually about -- see the comment on the boundary test above.
    ceiling_usd = 2.00 + RESERVATION_CEILING_BUFFER_USD
    _reserve(conn, model_turn=0, reserved_usd=2.00, ceiling_usd=ceiling_usd)
    # Never settled -- simulates a crash between reserving and settling.

    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(conn, model_turn=1, reserved_usd=0.01, ceiling_usd=ceiling_usd)

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
    with pytest.raises(CheckpointStoreError) as excinfo:
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
    assert (
        excinfo.value.reason_code
        is CheckpointStoreReasonCode.RESERVATION_NOT_SETTLEABLE
    )
    assert "('ghost', 'INVESTIGATE', 0, 'digest-1')" in str(excinfo.value)
    assert "absent or not RESERVED" in str(excinfo.value)


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

    with pytest.raises(CheckpointStoreError) as excinfo:
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
    assert (
        excinfo.value.reason_code
        is CheckpointStoreReasonCode.RESERVATION_NOT_SETTLEABLE
    )
    assert "('run-1', 'INVESTIGATE', 0, 'digest-1')" in str(excinfo.value)
    assert "absent or not RESERVED" in str(excinfo.value)
    row = conn.execute(
        "SELECT state, actual_usd, input_tokens, output_tokens FROM cost_ledger"
    ).fetchone()
    assert row == ("SETTLED", 0.01, 100, 50)


def test_run_cost_totals_sums_only_one_run_id(conn: sqlite3.Connection) -> None:
    """A different question from `_reserved_and_settled_total`'s
    application-wide ceiling sum: two turns of `run-1`, settled, plus one
    turn of an unrelated `run-2` that must not leak into `run-1`'s
    totals."""
    _reserve(conn, run_id="run-1", model_turn=0, context_digest="d0", reserved_usd=0.02)
    settle_reservation(
        conn,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="d0",
        actual_usd=0.015,
        input_tokens=900,
        output_tokens=100,
        settled_at=NOW,
    )
    _reserve(
        conn,
        run_id="run-1",
        graph_phase="FINAL_ASSESSMENT",
        model_turn=1,
        context_digest="d1",
        reserved_usd=0.03,
    )
    settle_reservation(
        conn,
        run_id="run-1",
        graph_phase="FINAL_ASSESSMENT",
        model_turn=1,
        context_digest="d1",
        actual_usd=0.021,
        input_tokens=1200,
        output_tokens=150,
        settled_at=NOW,
    )
    _reserve(conn, run_id="run-2", model_turn=0, context_digest="d0", reserved_usd=0.05)
    settle_reservation(
        conn,
        run_id="run-2",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="d0",
        actual_usd=0.04,
        input_tokens=2000,
        output_tokens=200,
        settled_at=NOW,
    )

    reserved_usd, actual_usd, fully_settled = run_cost_totals(conn, "run-1")

    assert reserved_usd == pytest.approx(0.05)
    assert actual_usd == pytest.approx(0.036)
    assert fully_settled is True


def test_run_cost_totals_counts_a_never_settled_reservation_only_as_reserved(
    conn: sqlite3.Connection,
) -> None:
    """A `RESERVED` row that never settles still counts toward
    `reserved_usd` (matching `_reserved_and_settled_total`'s own "a crash
    loop must not silently forget a reservation" rule) but contributes
    nothing to `actual_usd` -- there is no real cost yet to report. Also
    the "nothing ever settled" case for `fully_settled`: a run with one
    row, and that row still `RESERVED`, is not fully settled."""
    _reserve(conn, run_id="run-1", reserved_usd=0.07)

    reserved_usd, actual_usd, fully_settled = run_cost_totals(conn, "run-1")

    assert reserved_usd == pytest.approx(0.07)
    assert actual_usd == 0.0
    assert fully_settled is False


def test_run_cost_totals_for_an_unknown_run_id_is_zero(
    conn: sqlite3.Connection,
) -> None:
    """No rows at all for this `run_id` -- `fully_settled` is vacuously
    `True` (there is no outstanding reservation to report), matching
    `(0.0, 0.0)`'s own "nothing ever sent" reading."""
    reserved_usd, actual_usd, fully_settled = run_cost_totals(conn, "no-such-run")

    assert reserved_usd == 0.0
    assert actual_usd == 0.0
    assert fully_settled is True


def test_run_cost_totals_wraps_a_malformed_table_as_a_store_error(
    conn: sqlite3.Connection,
) -> None:
    """`run_cost_totals`'s own three `SELECT`s used to run unwrapped, unlike
    every other function in this module (`record_reservation_before_request`,
    `settle_reservation`), which all catch `sqlite3.Error` and re-raise the
    module's own `CheckpointStoreError(STORE_UNAVAILABLE, ...)`. A malformed
    or corrupted ledger schema -- simulated here by replacing `cost_ledger`
    with a table missing the columns these queries select -- used to raise a
    raw `sqlite3.OperationalError` instead of that established, actionable
    error contract."""
    conn.execute("DROP TABLE cost_ledger")
    conn.execute("CREATE TABLE cost_ledger (run_id TEXT NOT NULL)")
    conn.commit()

    with pytest.raises(CheckpointStoreError) as excinfo:
        run_cost_totals(conn, "run-1")

    assert excinfo.value.reason_code == CheckpointStoreReasonCode.STORE_UNAVAILABLE
    assert "run totals" in str(excinfo.value)


def test_settle_reservation_logs_when_the_real_bill_exceeds_the_reservation(
    conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """The invariant-breach half of the P1 fix: a settlement is never
    refused for coming in above its own reservation (the money is already
    spent, and the row must record what was really billed), but it must not
    be silent about it either -- a logged warning naming both figures is
    the signal that the pessimistic estimate this reservation was computed
    from needs recalibrating. Item 2 (P2) replaced `warnings.warn` with
    `logging` here -- see `test_settle_reservation_survives_warnings_as_
    errors_on_an_overrun` below for why."""
    _reserve(conn, reserved_usd=0.01)

    with caplog.at_level(logging.WARNING, logger="causalops.cost_ledger"):
        row = settle_reservation(
            conn,
            run_id="run-1",
            graph_phase="INVESTIGATE",
            model_turn=0,
            context_digest="digest-1",
            actual_usd=0.03,
            input_tokens=1000,
            output_tokens=500,
            settled_at=NOW,
        )

    assert row.actual_usd == pytest.approx(0.03)
    assert row.reserved_usd == pytest.approx(0.01)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "0.0300" in message
    assert "0.0100" in message


def test_settle_reservation_does_not_log_when_the_bill_stays_under_the_reservation(
    conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """The ordinary case -- `actual_usd <= reserved_usd` -- must stay
    silent, or every normal settlement would spuriously warn."""
    _reserve(conn, reserved_usd=0.05)

    with caplog.at_level(logging.WARNING, logger="causalops.cost_ledger"):
        settle_reservation(
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

    assert caplog.records == []


def test_settle_reservation_survives_warnings_as_errors_on_an_overrun(
    conn: sqlite3.Connection,
) -> None:
    """Direct proof of the P2 fix -- not just a happy-path run under
    default settings. `PYTHONWARNINGS=error` (or any caller/environment
    that sets `warnings.simplefilter("error")`) turns
    `warnings.warn(..., RuntimeWarning)` into a raised exception: the exact
    configuration `settle_reservation` used to run this same overrun branch
    under, back when it called `warnings.warn` instead of today's module
    logger. This test proves both halves directly, under the identical
    filter, in the identical process:

    1. `warnings.warn(..., RuntimeWarning)` genuinely does raise under this
       filter -- confirming the premise that the old code's bug was real,
       not hypothetical.
    2. `settle_reservation` itself -- already committed to the database,
       running the same `actual_usd > reserved_usd` branch that used to
       call `warnings.warn` -- does NOT raise under the identical filter,
       and returns the genuine, already-settled row. It no longer goes
       anywhere near `warnings.warn`, so its return value cannot depend on
       the caller's warning-filter configuration."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        with pytest.raises(RuntimeWarning):
            warnings.warn(
                "reproducing the old failure mode", RuntimeWarning, stacklevel=2
            )

        _reserve(conn, reserved_usd=0.01)
        row = settle_reservation(
            conn,
            run_id="run-1",
            graph_phase="INVESTIGATE",
            model_turn=0,
            context_digest="digest-1",
            actual_usd=0.03,
            input_tokens=1000,
            output_tokens=500,
            settled_at=NOW,
        )

    assert row.actual_usd == pytest.approx(0.03)
    assert row.reserved_usd == pytest.approx(0.01)
    assert row.state == "SETTLED"


def test_an_overrun_settlement_is_counted_at_its_true_cost_against_the_ceiling(
    conn: sqlite3.Connection,
) -> None:
    """Reproduces the exact reported scenario: a $0.01 reservation
    that settles at $0.03 under a $0.02 application-wide ceiling. Before
    this fix, `_reserved_and_settled_total` summed `reserved_usd`
    unconditionally, so the ceiling's running total still believed only
    $0.01 had ever been spent -- a further $0.01 request would have been
    wrongly approved, letting real spend reach $0.04 against a stated
    "hard" $0.02 limit. After the fix, the settled row counts at its true
    $0.03, so even a tiny further reservation against the same ceiling is
    correctly refused.

    `ceiling_usd` carries `RESERVATION_CEILING_BUFFER_USD` on top of the
    0.02 this scenario is actually about, reproducing the exact
    original report's own arithmetic unchanged -- `RESERVATION_CEILING_BUFFER_USD`
    exceeds 0.02 outright, so without this the very first reservation below
    would be refused for the wrong reason before this test ever reaches the
    overrun it exists to cover."""
    ceiling_usd = 0.02 + RESERVATION_CEILING_BUFFER_USD
    _reserve(conn, run_id="run-1", reserved_usd=0.01, ceiling_usd=ceiling_usd)
    settle_reservation(
        conn,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
        actual_usd=0.03,
        input_tokens=1000,
        output_tokens=500,
        settled_at=NOW,
    )

    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(
            conn,
            run_id="run-2",
            model_turn=0,
            context_digest="digest-2",
            reserved_usd=0.001,
            ceiling_usd=ceiling_usd,
        )

    assert excinfo.value.remaining_usd == pytest.approx(0.0)


def test_a_reservation_within_the_raw_ceiling_but_inside_the_buffer_is_refused(
    conn: sqlite3.Connection,
) -> None:
    """The margin itself. Ceiling 1.00, 0.85 already spent (itself under the
    buffered remaining of 0.90, so it is authorized normally), then 0.10
    more requested: 0.85 + 0.10 = 0.95 fits under the raw `ceiling_usd` of
    1.00 -- before this fix, `record_reservation_before_request` would have
    authorized it. It does not fit under `1.00 - RESERVATION_CEILING_
    BUFFER_USD` (0.90 remaining after the first reservation leaves 0.05), so
    it must now be refused instead."""
    ceiling_usd = 1.00
    _reserve(conn, model_turn=0, reserved_usd=0.85, ceiling_usd=ceiling_usd)

    with pytest.raises(CostCeilingExceeded) as excinfo:
        _reserve(conn, model_turn=1, reserved_usd=0.10, ceiling_usd=ceiling_usd)

    assert excinfo.value.remaining_usd == pytest.approx(0.05)


def test_run_cost_totals_reports_partial_settlement_as_not_fully_settled(
    conn: sqlite3.Connection,
) -> None:
    """The P2 this fix exists for: a run with 3 of 4 model calls settled and
    the 4th still `RESERVED` (a timeout or crash mid-call, short of a clean
    settle) has a non-zero, but PARTIAL, `actual_usd` -- `actual_usd == 0.0`
    is not a reliable signal that something is missing here, since the sum
    of the 3 settled rows is a real, non-zero number that nonetheless
    understates the run's true final cost. `fully_settled` must say `False`
    based on row state, not on whether the sum happens to be zero."""
    for turn in range(3):
        _reserve(
            conn,
            run_id="run-1",
            model_turn=turn,
            context_digest=f"d{turn}",
            reserved_usd=0.02,
        )
        settle_reservation(
            conn,
            run_id="run-1",
            graph_phase="INVESTIGATE",
            model_turn=turn,
            context_digest=f"d{turn}",
            actual_usd=0.015,
            input_tokens=500,
            output_tokens=50,
            settled_at=NOW,
        )
    # The 4th reservation is made but never settled -- simulates a timeout
    # or crash after the 4th request was reserved but before it returned.
    _reserve(conn, run_id="run-1", model_turn=3, context_digest="d3", reserved_usd=0.02)

    reserved_usd, actual_usd, fully_settled = run_cost_totals(conn, "run-1")

    assert reserved_usd == pytest.approx(0.08)
    assert actual_usd == pytest.approx(0.045)
    assert actual_usd > 0.0  # the partial sum is real and non-zero
    assert fully_settled is False
