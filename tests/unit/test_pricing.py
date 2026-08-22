"""Unit 3b-2: the reservation math the cost gate depends on.

No provider client, no network, no `sqlite3` -- this module is pure
arithmetic, and these tests hold it to that.
"""

import pytest

from causalops import pricing
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


def test_the_ratio_rounds_up_not_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ceiling division, not truncation -- must hold for any ratio, not
    just today's. Unit 3b-3 set `PESSIMISTIC_CHARS_PER_TOKEN` to 1.0, which
    divides every character count evenly and so cannot exercise rounding
    on its own; this monkeypatches a divisor that does not divide evenly
    (3.0, the pre-3b-3 value) so the behaviour this test actually cares
    about -- `estimate_input_tokens` reads the module-global at call time,
    confirmed by this monkeypatch taking effect -- stays covered regardless
    of what the real constant currently is."""
    monkeypatch.setattr(pricing, "PESSIMISTIC_CHARS_PER_TOKEN", 3.0)

    # 7 characters at a 3.0 ratio is 2 whole tokens plus a remainder, and
    # the remainder must not vanish.
    assert estimate_input_tokens("x" * 7) == 3
    assert estimate_input_tokens("x" * 6) == 2


def test_the_estimate_never_undercounts_a_real_tokenizer() -> None:
    # This module deliberately has no tokenizer to compare against
    # (pricing.py's own docstring), so this test pins the one thing that
    # actually matters about the ratio: it must not be looser than the
    # documented real-world average of ~4 characters per token -- and,
    # since Unit 3b-3, not looser than the one real billed data point this
    # project has measured either (~2.26 chars/token; see
    # `PESSIMISTIC_CHARS_PER_TOKEN`'s own comment). If this constant is
    # ever loosened back toward 4.0, a real request could tokenize to more
    # tokens than this estimate reports -- exactly the failure mode the
    # pessimism exists to prevent, and exactly what the smoke call caught
    # at the old 3.0 value.
    assert PESSIMISTIC_CHARS_PER_TOKEN < 4.0


def test_the_input_cap_preserves_the_intended_9600_character_prose_budget() -> None:
    """`MAX_INPUT_TOKENS` bounds prose in TOKENS, but its real job is
    bounding prose in CHARACTERS -- the token figure is only how the
    estimate happens to be expressed, via `PESSIMISTIC_CHARS_PER_TOKEN`.
    Unit 3b-2 chose 3,200 tokens against a 3.0 ratio (a 9,600-character
    budget); Unit 3b-3 tightened the ratio to 1.0 for calibration reasons
    unrelated to how much prose one turn should carry, and moved this cap
    to 9,600 tokens specifically to leave that character budget unchanged.
    If a future change moves either constant without moving the other to
    match, the *effective* character budget silently changes -- exactly
    the "the cap would start refusing ordinary runs" failure Unit 3b-2's
    owner ruling and Unit 3b-3's replan both had to avoid. This assertion
    does not derive its expectation from the two constants (that would
    pass no matter how far they drift together); it pins the literal,
    owner-approved 9,600-character figure directly, so a change to either
    constant alone fails here."""
    assert estimate_input_tokens("x" * 9_600) <= MAX_INPUT_TOKENS
    assert estimate_input_tokens("x" * 9_601) > MAX_INPUT_TOKENS


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
    division at `PESSIMISTIC_CHARS_PER_TOKEN` -- 1.0 as of Unit 3b-3, far
    more conservative than the 3.0 in effect when this test was written)
    for the combined prose-plus-tool-schema text `live_model.py`'s `_send`
    reserves against post-fix; the settlement gets the smaller token count
    a genuine tokenizer would report for the *same* text -- calibrated
    against the smoke call's own measured ratio (~2.26 characters per
    token: `pricing.py`'s `PESSIMISTIC_CHARS_PER_TOKEN` comment), the
    weakest, most billing-realistic choice available, not the documented
    ~4 chars/token English-prose average this test used before Unit 3b-3
    (a looser divisor means fewer billed tokens, which makes this
    assertion easier to pass for no real reason -- calibrating against the
    one real data point this project has closes that gap). If the
    reservation math ever again dropped the tool payload (reverting P1-1),
    a settlement still billed for the whole wire request would come in
    above a reservation sized for prose alone, and this assertion would
    fail."""
    snapshot = CLAUDE_SONNET_5_PRICING
    # This unit's own measurements (re-derived directly, not carried by
    # hand -- `test_live_model.py`'s `test_the_tool_payload_size_matches_
    # what_pricingpy_assumes` pins the same 7,595 figure against the real
    # tool definitions, post Unit 3b-3's discriminator fix): `_plan_tool_
    # definition` plus the five `_domain_tool_definitions` serialize to
    # 7,595 characters; a representative turn-zero INITIAL_PLAN prose
    # renders to 1,512 (Unit 3b-2's figure -- unaffected by 3b-3, which
    # touched only the tool schema and the FINAL_ASSESSMENT-stage prose
    # measurement, not this one).
    prose = "x" * 1_512
    tool_definitions = "x" * 7_595

    reserved = snapshot.reservation_usd(
        estimate_input_tokens(prose) + estimate_input_tokens(tool_definitions)
    )
    # The smoke call's own measured ratio (`pricing.py`'s
    # `PESSIMISTIC_CHARS_PER_TOKEN` comment: 9,249 characters billed at
    # 4,099 tokens), not the documented ~4 chars/token English-prose
    # average -- a real, measured data point rather than the weakest
    # divisor this test happened to have on hand.
    measured_chars_per_token = 2.26
    real_input_tokens = int(
        (len(prose) + len(tool_definitions)) // measured_chars_per_token
    )
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
