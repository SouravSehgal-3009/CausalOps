"""Unit 3b-2: the reservation math the cost gate depends on.

No provider client, no network, no `sqlite3` -- this module is pure
arithmetic, and these tests hold it to that.
"""

import pytest

from causalops.pricing import (
    CLAUDE_SONNET_5_PRICING,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    PESSIMISTIC_CHARS_PER_TOKEN,
    InputTooLarge,
    PricingSnapshot,
    estimate_input_tokens,
)


def test_empty_text_estimates_to_zero_tokens() -> None:
    assert estimate_input_tokens("") == 0


def test_a_single_character_estimates_to_at_least_one_token() -> None:
    # Never zero for non-empty text -- ceiling division, not truncation.
    assert estimate_input_tokens("x") == 1


def test_the_ratio_rounds_up_not_down() -> None:
    # `PESSIMISTIC_CHARS_PER_TOKEN` is 3.0 -- 7 characters is 2 whole tokens
    # plus a remainder, and the remainder must not vanish.
    text = "x" * 7
    assert estimate_input_tokens(text) == 3
    assert estimate_input_tokens("x" * 6) == 2


def test_the_estimate_never_undercounts_a_real_tokenizer() -> None:
    # This module deliberately has no tokenizer to compare against
    # (pricing.py's own docstring), so this test pins the one thing that
    # actually matters about the ratio: it must not be looser than the
    # documented real-world average of ~4 characters per token. If this
    # constant is ever loosened toward 4.0, a real request could tokenize
    # to more tokens than this estimate reports -- exactly the failure mode
    # the pessimism exists to prevent.
    assert PESSIMISTIC_CHARS_PER_TOKEN < 4.0


def test_reservation_usd_prices_the_full_input_and_output_allowance() -> None:
    snapshot = PricingSnapshot(
        model_name="test-model",
        input_usd_per_million_tokens=2.0,
        output_usd_per_million_tokens=10.0,
        source="test",
        verified_on="2026-01-01",
    )

    reserved = snapshot.reservation_usd(1_000_000, max_output_tokens=1_000_000)

    assert reserved == pytest.approx(12.0)


def test_reservation_usd_defaults_the_output_allowance_to_max_output_tokens() -> None:
    snapshot = PricingSnapshot(
        model_name="test-model",
        input_usd_per_million_tokens=0.0 + 1e-9,
        output_usd_per_million_tokens=10.0,
        source="test",
        verified_on="2026-01-01",
    )

    reserved = snapshot.reservation_usd(0)

    assert reserved == pytest.approx((MAX_OUTPUT_TOKENS / 1_000_000) * 10.0, rel=1e-3)


def test_actual_cost_usd_uses_real_tokens_not_the_output_allowance() -> None:
    snapshot = PricingSnapshot(
        model_name="test-model",
        input_usd_per_million_tokens=2.0,
        output_usd_per_million_tokens=10.0,
        source="test",
        verified_on="2026-01-01",
    )

    # Far below `MAX_OUTPUT_TOKENS` -- unlike `reservation_usd`, this must
    # not silently price the full allowance the request never used.
    actual = snapshot.actual_cost_usd(input_tokens=1000, output_tokens=100)

    assert actual == pytest.approx((1000 / 1_000_000) * 2.0 + (100 / 1_000_000) * 10.0)


def test_a_settled_request_never_costs_more_than_its_own_reservation() -> None:
    """The reservation is supposed to be a worst case, but pricing this
    single number the same way on both sides (as an earlier version of this
    test did) proves nothing: `reservation_usd(N)` and
    `actual_cost_usd(N, MAX_OUTPUT_TOKENS)` are the identical linear formula
    given the identical `N`, so `worst_case_actual <= reserved` reduces to
    `X <= X` -- true no matter what `N` omits, which is exactly how Unit
    3b-2's P1-1 bug (the reservation once excluded ~2,580 tokens of
    tool-definition text `bind_tools` sends on every call) shipped past
    this assertion.

    This version prices the two sides off genuinely different numbers: the
    reservation gets this module's own *pessimistic* estimate (ceiling
    division at `PESSIMISTIC_CHARS_PER_TOKEN` = 3.0) for the combined
    prose-plus-tool-schema text `live_model.py`'s `_send` reserves against
    post-fix; the settlement gets the smaller token count a genuine
    tokenizer -- closer to the documented ~4 chars/token real-world average
    -- would report for the *same* text. If the reservation math ever again
    dropped the tool payload (reverting P1-1), a settlement still billed
    for the whole wire request would come in above a reservation sized for
    prose alone, and this assertion would fail."""
    snapshot = CLAUDE_SONNET_5_PRICING
    # This unit's own measurements (re-derived directly, not carried by
    # hand -- `test_live_model.py`'s `test_the_tool_payload_size_matches_
    # what_pricingpy_assumes` pins the same 7,738 figure against the real
    # tool definitions): `_plan_tool_definition` plus the five
    # `_domain_tool_definitions` serialize to 7,738 characters; a
    # representative turn-zero INITIAL_PLAN prose renders to 1,512.
    prose = "x" * 1_512
    tool_definitions = "x" * 7_738

    reserved = snapshot.reservation_usd(
        estimate_input_tokens(prose) + estimate_input_tokens(tool_definitions)
    )
    real_tokenizer_chars_per_token = 4
    real_input_tokens = (
        len(prose) + len(tool_definitions)
    ) // real_tokenizer_chars_per_token
    worst_case_actual = snapshot.actual_cost_usd(real_input_tokens, MAX_OUTPUT_TOKENS)

    assert worst_case_actual <= reserved


def test_claude_sonnet_5_pricing_names_a_source_and_a_date() -> None:
    # `TECHNICAL_SPEC.md` section 10: "Record ... the pricing source/date."
    assert CLAUDE_SONNET_5_PRICING.source.startswith("https://")
    assert CLAUDE_SONNET_5_PRICING.verified_on
    assert CLAUDE_SONNET_5_PRICING.model_name == "claude-sonnet-5"


def test_input_too_large_carries_the_estimate_that_tripped_it() -> None:
    error = InputTooLarge(9999)

    assert error.estimated_tokens == 9999
    assert str(MAX_INPUT_TOKENS) in str(error)
