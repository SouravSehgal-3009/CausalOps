"""Tests for `causalops.live_setup`, extracted from `test_cli.py` alongside
Unit 3c's `_build_model_and_registry`/`_live_evaluation_ceiling_usd`
extraction out of `cli.py` -- both `causalops.cli` and
`causalops.evaluate_cli` build a live model/tool registry and resolve the
cost ceiling through this shared module now, so its tests live here rather
than under a CLI-specific test module.
"""

import pytest

from causalops.approvals import CheckpointStoreError, CheckpointStoreReasonCode
from causalops.cost_ledger import RESERVATION_CEILING_BUFFER_USD
from causalops.live_setup import (
    DEFAULT_LIVE_EVALUATION_MAX_USD,
    LIVE_EVALUATION_MAX_USD_VARIABLE,
    live_evaluation_ceiling_usd,
)


def test_live_evaluation_ceiling_reads_a_valid_value() -> None:
    assert live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "0.50"}) == (
        0.50
    )


def test_live_evaluation_ceiling_defaults_when_absent() -> None:
    assert live_evaluation_ceiling_usd({}) == DEFAULT_LIVE_EVALUATION_MAX_USD


def test_live_evaluation_ceiling_falls_back_on_a_malformed_value() -> None:
    """Deliberately silent, unlike most malformed-input handling in this
    project: the fallback (`DEFAULT_LIVE_EVALUATION_MAX_USD`) is the
    *smallest* plausible ceiling, so defaulting here can only make the gate
    stricter than a typo intended, never more permissive -- see
    `live_setup.py`'s own comment on `DEFAULT_LIVE_EVALUATION_MAX_USD`."""
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "not-a-number"})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )


def test_live_evaluation_ceiling_falls_back_on_a_non_positive_value() -> None:
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "-1"})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "0"})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )


def test_live_evaluation_ceiling_falls_back_on_a_non_finite_value() -> None:
    """P3-4's regression test. `float("inf")` passes `> 0` (`inf > 0` is
    `True`), so before this fix `LIVE_EVALUATION_MAX_USD=inf` disabled the
    cost ceiling entirely rather than falling back to the smallest plausible
    default -- mutation-verified when this test was first written. `-inf` is
    covered by the same `math.isfinite` guard, not by `> 0` (`-inf > 0` is
    already `False`), so it is asserted here too rather than left to the
    other branch's luck. `nan` already fails `> 0` (every comparison against
    `nan` is `False`), so it was already covered before that fix -- pinned
    here explicitly so a future refactor that touches the comparison cannot
    silently drop that coverage without a test noticing."""
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "inf"})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "-inf"})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "nan"})
        == DEFAULT_LIVE_EVALUATION_MAX_USD
    )


def test_live_evaluation_ceiling_rejects_a_value_at_or_below_the_buffer() -> None:
    """The P2 this fix exists for: `cost_ledger.record_reservation_before_
    request` always subtracts `RESERVATION_CEILING_BUFFER_USD` ($0.10) from
    `ceiling_usd` before checking remaining budget, so a configured ceiling
    at or below that buffer leaves `remaining` permanently negative --
    `CostCeilingExceeded` on literally every reservation, with nothing in
    that error naming the real problem as the configured ceiling itself.
    Unlike the three malformed *shapes* the tests above cover, this is a
    well-formed positive finite number -- so it is rejected here, at
    config-resolution time, with a `CheckpointStoreError` that names both
    the configured value and the buffer, rather than silently falling back
    to a much larger default (which would defeat a deliberately small
    configured cap) or accepted and left to fail confusingly on every
    subsequent request.

    Tests both the exact boundary (`== RESERVATION_CEILING_BUFFER_USD`,
    which still leaves zero headroom -- `ceiling - buffer - spent == 0` can
    never authorize a positive reservation) and a value below it."""
    for raw in (f"{RESERVATION_CEILING_BUFFER_USD:.2f}", "0.05", "0.01"):
        with pytest.raises(CheckpointStoreError) as excinfo:
            live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: raw})
        assert (
            excinfo.value.reason_code
            == CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER
        )
        assert raw in str(excinfo.value) or f"{float(raw):.4f}" in str(excinfo.value)
        assert f"{RESERVATION_CEILING_BUFFER_USD:.2f}" in str(excinfo.value)


def test_live_evaluation_ceiling_still_accepts_a_value_above_the_buffer() -> None:
    """Regression coverage for the existing behaviour: a ceiling well above
    `RESERVATION_CEILING_BUFFER_USD` still resolves exactly as before this
    fix -- both right at the boundary that first becomes valid (a cent
    above the buffer) and the project's own real configured default."""
    just_above_buffer = RESERVATION_CEILING_BUFFER_USD + 0.01
    assert live_evaluation_ceiling_usd(
        {LIVE_EVALUATION_MAX_USD_VARIABLE: f"{just_above_buffer:.2f}"}
    ) == pytest.approx(just_above_buffer)
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "5.00"}) == 5.00
    )


def test_live_evaluation_ceiling_honours_a_well_formed_typo() -> None:
    """The boundary P3-4 does *not* move: `500` typed for `5.00` (Unit
    3b-3's default, raised from 2.00) is a well-formed positive, finite
    number and is honoured as written, same as any other config value --
    guarding against a fat-fingered magnitude is the owner's job, not this
    parser's. This test exists so the boundary is documented, not implied."""
    assert live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "500"}) == (
        500.0
    )
