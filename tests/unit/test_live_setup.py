"""Tests for `causalops.live_setup`, extracted from `test_cli.py` alongside
Unit 3c's `_build_model_and_registry`/`_live_evaluation_ceiling_usd`
extraction out of `cli.py` -- both `causalops.cli` and
`causalops.evaluate_cli` build a live model/tool registry and resolve the
cost ceiling through this shared module now, so its tests live here rather
than under a CLI-specific test module.
"""

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


def test_live_evaluation_ceiling_honours_a_well_formed_typo() -> None:
    """The boundary P3-4 does *not* move: `500` typed for `5.00` (Unit
    3b-3's default, raised from 2.00) is a well-formed positive, finite
    number and is honoured as written, same as any other config value --
    guarding against a fat-fingered magnitude is the owner's job, not this
    parser's. This test exists so the boundary is documented, not implied."""
    assert live_evaluation_ceiling_usd({LIVE_EVALUATION_MAX_USD_VARIABLE: "500"}) == (
        500.0
    )
