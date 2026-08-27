"""Tests for `causalops.live_setup`, extracted from `test_cli.py` alongside
the `_build_model_and_registry`/`_live_evaluation_ceiling_usd`
extraction out of `cli.py` -- both `causalops.cli` and
`causalops.evaluate_cli` build a live model/tool registry and resolve the
cost ceiling through this shared module now, so its tests live here rather
than under a CLI-specific test module.
"""

import sqlite3

import pytest

from causalops.approvals import CheckpointStoreError, CheckpointStoreReasonCode
from causalops.cost_ledger import (
    RESERVATION_CEILING_BUFFER_USD,
    ensure_cost_ledger_table,
)
from causalops.live_model import (
    LiveClaudeModel,
    MinimumReservationProbeTransport,
    build_minimum_final_assessment_request,
)
from causalops.live_setup import (
    DEFAULT_LIVE_EVALUATION_MAX_USD,
    LIVE_EVALUATION_MAX_USD_VARIABLE,
    MINIMUM_POSSIBLE_RESERVATION_USD,
    MINIMUM_USABLE_CEILING_USD,
    live_evaluation_ceiling_usd,
)


def test_live_evaluation_ceiling_reads_a_valid_value() -> None:
    assert live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "0.50"}) == (
        0.50
    )


def test_live_evaluation_ceiling_defaults_when_absent() -> None:
    assert live_evaluation_ceiling_usd({}) == DEFAULT_LIVE_EVALUATION_MAX_USD


def test_live_evaluation_ceiling_defaults_when_blank() -> None:
    """Regression coverage for the one case that must keep defaulting:
    blank (whitespace-only) is the same as unset, since nothing was
    actually typed for this fallback to override."""
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: ""})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "   "})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )


def test_live_evaluation_ceiling_rejects_a_malformed_value() -> None:
    """A malformed value used to fall back to `DEFAULT_LIVE_EVALUATION_MAX_USD`
    silently -- but that default ($5.00) is far LARGER than a malformed
    input suggests the owner intended, so silently defaulting it is the
    more permissive, more dangerous surprise this module's reasoning about
    the too-small case already refuses to allow elsewhere. `"0.05USD"`
    is a deliberate probe: it looks like a plausible small figure, and
    used to silently authorize the full $5.00 default instead."""
    with pytest.raises(CheckpointStoreError) as excinfo:
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "0.05USD"})
    assert excinfo.value.reason_code == CheckpointStoreReasonCode.CEILING_MALFORMED
    assert "0.05USD" in str(excinfo.value)

    with pytest.raises(CheckpointStoreError) as excinfo:
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "not-a-number"})
    assert excinfo.value.reason_code == CheckpointStoreReasonCode.CEILING_MALFORMED


def test_live_evaluation_ceiling_rejects_a_non_positive_value() -> None:
    """`0` and a negative value used to fall back to the $5.00 default too
    -- the same silent-more-permissive-surprise problem as the malformed
    case above. Both are well-formed, finite numbers, so they are refused
    as `CEILING_BELOW_RESERVATION_BUFFER` (trivially at or below the
    positive `MINIMUM_USABLE_CEILING_USD` floor), not `CEILING_MALFORMED`,
    which is reserved for values that are not even a well-formed finite
    number."""
    for raw in ("-1", "0"):
        with pytest.raises(CheckpointStoreError) as excinfo:
            live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: raw})
        assert (
            excinfo.value.reason_code
            == CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER
        )


def test_live_evaluation_ceiling_rejects_a_non_finite_value() -> None:
    """`float("inf")` passes a naive `> 0` check (`inf > 0` is `True`), so
    an earlier version of this validation let `LIVE_EVALUATION_MAX_USD=inf`
    disable the cost ceiling entirely. All three non-finite shapes now
    raise `CEILING_MALFORMED` rather than silently falling back to the
    default -- a change from the prior fallback behaviour this test file
    used to pin, since falling back to $5.00 for an `inf` or `nan` input is
    itself a silent, more-permissive-than-typed surprise."""
    for raw in ("inf", "-inf", "nan"):
        with pytest.raises(CheckpointStoreError) as excinfo:
            live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: raw})
        assert excinfo.value.reason_code == CheckpointStoreReasonCode.CEILING_MALFORMED


def test_live_evaluation_ceiling_rejects_a_value_at_or_below_the_true_floor() -> None:
    """The true minimum usable ceiling is not just `RESERVATION_CEILING_
    BUFFER_USD` ($0.10) -- `cost_ledger.record_reservation_before_request`
    always subtracts that buffer from `ceiling_usd` before checking
    remaining budget, but the cheapest possible real reservation is not $0
    either, it is `MINIMUM_POSSIBLE_RESERVATION_USD` -- derived by actually
    running the smallest real FINAL_ASSESSMENT request through `_send`'s
    own reservation code (`live_model.minimum_possible_reservation_usd`),
    not a hand-reconstructed approximation of one. A ceiling like $0.11 --
    comfortably above the $0.10 buffer alone -- still leaves only $0.01 of
    headroom once the buffer is subtracted, far less than even the
    cheapest possible request, so it must be rejected too. Tests the
    buffer-only boundary ($0.10, $0.11) plus the corrected boundary
    (`MINIMUM_USABLE_CEILING_USD` itself, which still leaves zero
    headroom). Also pins two literal figures earlier, narrower versions of
    this floor used to wrongly accept: $0.116 (a zero-input floor's
    threshold) and $0.117 (a tool-schema-only floor's threshold, still
    missing `SYSTEM_TEXT`'s own token cost) -- both must now be rejected
    under the real, fully-derived floor."""
    for raw in (
        f"{RESERVATION_CEILING_BUFFER_USD:.2f}",
        "0.11",
        "0.116",
        "0.117",
        # `.6f`, not `.4f`: `MINIMUM_USABLE_CEILING_USD` needs six decimal
        # places to hit the boundary exactly -- `.4f` could round it UP to
        # a value strictly above the true floor that the gate would
        # wrongly accept, defeating this exact-boundary check.
        f"{MINIMUM_USABLE_CEILING_USD:.6f}",
    ):
        with pytest.raises(CheckpointStoreError) as excinfo:
            live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: raw})
        assert (
            excinfo.value.reason_code
            == CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER
        )
        assert f"{MINIMUM_USABLE_CEILING_USD:.6f}" in str(excinfo.value)


def test_live_evaluation_ceiling_still_accepts_a_value_above_the_true_floor() -> None:
    """Regression coverage for the existing behaviour: a ceiling well above
    `MINIMUM_USABLE_CEILING_USD` still resolves exactly as before this fix
    -- both right at the boundary that first becomes valid (a cent above the
    floor) and the project's own real configured default."""
    just_above_floor = MINIMUM_USABLE_CEILING_USD + 0.01
    assert live_evaluation_ceiling_usd(
        {LIVE_EVALUATION_MAX_USD_VARIABLE: f"{just_above_floor:.6f}"}
    ) == pytest.approx(just_above_floor)
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "5.00"}) == 5.00
    )


def test_live_evaluation_ceiling_honours_a_well_formed_typo() -> None:
    """The boundary this fix does *not* move: `500` typed for `5.00` (the
    default, raised from 2.00) is a well-formed positive, finite
    number and is honoured as written, same as any other config value --
    guarding against a fat-fingered magnitude is the owner's job, not this
    parser's. This test exists so the boundary is documented, not implied."""
    assert live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "500"}) == (
        500.0
    )


def test_accepted_ceiling_actually_authorizes_a_minimal_reservation() -> None:
    """Config-time acceptance alone does not prove the ceiling is genuinely
    usable -- a ceiling could pass `live_evaluation_ceiling_usd` and still
    refuse every real request.

    An earlier version of this test only proved internal arithmetic
    consistency: it hand-fed `reserved_usd=MINIMUM_POSSIBLE_RESERVATION_USD`
    directly into `cost_ledger.record_reservation_before_request`, which
    cannot catch `MINIMUM_POSSIBLE_RESERVATION_USD` itself being wrong --
    exactly the failure mode that let this floor stay too low across three
    earlier rounds. This version closes that gap for real: it builds the
    same minimal FINAL_ASSESSMENT request `live_model.
    minimum_possible_reservation_usd` used to derive the floor
    (`live_model.build_minimum_final_assessment_request`), and sends it
    through a REAL `LiveClaudeModel.respond()` call -- `_send`'s actual
    reservation code, not a copy of it -- against a fake, no-network
    transport, under a ceiling accepted right at the boundary. If
    `MINIMUM_POSSIBLE_RESERVATION_USD` were ever wrong again, this is the
    test that would fail: `respond()` would raise `CostCeilingExceeded`
    instead of returning."""
    ceiling = MINIMUM_USABLE_CEILING_USD + 0.001
    accepted = live_evaluation_ceiling_usd(
        {LIVE_EVALUATION_MAX_USD_VARIABLE: f"{ceiling:.6f}"}
    )
    assert accepted == pytest.approx(ceiling)

    conn = sqlite3.connect(":memory:")
    ensure_cost_ledger_table(conn)
    request = build_minimum_final_assessment_request()
    model = LiveClaudeModel(
        conn,
        ceiling_usd=accepted,
        client=MinimumReservationProbeTransport(),  # type: ignore[arg-type]
    )

    model.respond(request)  # must not raise CostCeilingExceeded

    row = conn.execute(
        "SELECT reserved_usd, state FROM cost_ledger WHERE run_id = ?",
        (request.run_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(MINIMUM_POSSIBLE_RESERVATION_USD)
    assert row[1] == "SETTLED"
