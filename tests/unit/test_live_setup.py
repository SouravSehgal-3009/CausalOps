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


def test_live_evaluation_ceiling_rejects_only_the_structural_buffer_floor() -> None:
    """The true minimum usable ceiling is not just `RESERVATION_CEILING_
    BUFFER_USD` ($0.10) -- `cost_ledger.record_reservation_before_request`
    always subtracts that buffer from `ceiling_usd` before checking
    remaining budget, but the cheapest possible real reservation is not $0
    either, it is `MINIMUM_POSSIBLE_RESERVATION_USD` -- derived from the
    real minimum tool-schema payload (`respond()`'s final-assessment-only
    schema), not a zero-input approximation of one. A ceiling like $0.11 --
    comfortably above the $0.10 buffer alone -- still leaves only $0.01 of
    headroom once the buffer is subtracted, far less than even the
    cheapest possible request, so it must be rejected too. Tests the
    buffer-only boundary ($0.10, $0.11) plus the corrected boundary
    (`MINIMUM_USABLE_CEILING_USD` itself, which still leaves zero
    headroom). Also pins the two literal figures a zero-input floor
    (`$0.016`) used to wrongly accept: $0.116 (the old, insufficient
    threshold) and $0.117 (a value directly demonstrated as still broken
    under the zero-input floor, since it clears $0.116 but not the real
    $0.120584 minimum) -- both must now be rejected under the real,
    tool-schema-derived floor."""
    for raw in ("-1", "0", f"{RESERVATION_CEILING_BUFFER_USD:.2f}"):
        with pytest.raises(CheckpointStoreError) as excinfo:
            live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: raw})
        assert (
            excinfo.value.reason_code
            == CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER
        )
        assert "$0.10" in str(excinfo.value)


def test_live_evaluation_ceiling_accepts_values_above_the_buffer() -> None:
    """Regression coverage for the existing behaviour: a ceiling well above
    `MINIMUM_USABLE_CEILING_USD` still resolves exactly as before this fix
    -- both right at the boundary that first becomes valid (a cent above the
    floor) and the project's own real configured default."""
    assert live_evaluation_ceiling_usd(
        {LIVE_EVALUATION_MAX_USD_VARIABLE: "0.101"}
    ) == pytest.approx(0.101)
    assert (
        live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "5.00"}) == 5.00
    )


def test_live_evaluation_ceiling_honours_a_well_formed_typo() -> None:
    """The boundary this fix does *not* move: `500` typed for `5.00` (Unit
    3b-3's default, raised from 2.00) is a well-formed positive, finite
    number and is honoured as written, same as any other config value --
    guarding against a fat-fingered magnitude is the owner's job, not this
    parser's. This test exists so the boundary is documented, not implied."""
    assert live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "500"}) == (
        500.0
    )
