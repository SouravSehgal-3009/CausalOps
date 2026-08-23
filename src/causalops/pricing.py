"""Frozen pricing and conservative reservation math for provider requests."""

from pydantic import BaseModel, ConfigDict, Field

# Keep the rendered-prose budget at 9,600 characters under the conservative
# one-character-per-token estimate.
MAX_INPUT_TOKENS = 9_600
MAX_OUTPUT_TOKENS = 1_600

# Unit 3b-2, P2-4. `TECHNICAL_OVERVIEW.md`'s "Default limits" table's own
# "Model call | 90 seconds" row, unenforced until now. Lives beside the
# other two limits above rather than in `live_model.py` for the same reason
# they do: one file an owner can read to see every number this gate bounds,
# not three.
MAX_REQUEST_SECONDS = 90.0

# A one-character ratio is a deliberately conservative empirical estimate,
# not a universal tokenizer guarantee. Recalibrate only if saved provider usage
# exceeds it, using that measured artifact rather than a preference.
PESSIMISTIC_CHARS_PER_TOKEN = 1.0


def estimate_input_tokens(text: str) -> int:
    """Estimate supplied serialized text, not provider-side prompt overhead.

    The 1.0 ratio is conservative empirical policy, not a tokenizer guarantee;
    recalibrate only when saved provider usage demonstrates underestimation.
    """
    if not text:
        return 0
    return -(-len(text) // int(PESSIMISTIC_CHARS_PER_TOKEN))  # ceiling division


class InputTooLarge(Exception):
    """Refuse an oversized prose request before sending or reserving it.

    The input cap excludes tool schemas so normal turns retain their context
    budget; reservations include every billed schema token separately.
    """

    def __init__(self, estimated_tokens: int) -> None:
        super().__init__(
            f"estimated {estimated_tokens} input tokens exceeds the "
            f"{MAX_INPUT_TOKENS}-token cap"
        )
        self.estimated_tokens = estimated_tokens


class PricingSnapshot(BaseModel):
    """One model's per-token rate, with the source and date `TECHNICAL_SPEC.md`
    §10 requires every evaluation record to cite. Frozen so a reservation and
    the settlement it is checked against always read the same numbers within
    one process."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    input_usd_per_million_tokens: float = Field(gt=0)
    output_usd_per_million_tokens: float = Field(gt=0)
    source: str
    verified_on: str

    def reservation_usd(
        self, input_tokens: int, max_output_tokens: int = MAX_OUTPUT_TOKENS
    ) -> float:
        """Worst-case cost of one request: every counted input token plus
        every token `max_tokens` would allow the response to spend, both
        upper bounds. `input_tokens` is deliberately a parameter, not this
        snapshot's own estimate call -- the input-cap check
        (`live_model.py`) and this reservation both need the same estimate,
        and computing it twice would let the two silently disagree about
        what a request costs."""
        input_cost = (input_tokens / 1_000_000) * self.input_usd_per_million_tokens
        output_cost = (
            max_output_tokens / 1_000_000
        ) * self.output_usd_per_million_tokens
        return input_cost + output_cost

    def actual_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """The real cost of one settled request, from the provider's own
        reported token counts -- never an estimate. Settlement (`cost_ledger.
        settle_reservation`) records this alongside the reservation it
        replaces, so a report can show both the worst-case and the actual
        figure side by side."""
        input_cost = (input_tokens / 1_000_000) * self.input_usd_per_million_tokens
        output_cost = (output_tokens / 1_000_000) * self.output_usd_per_million_tokens
        return input_cost + output_cost


# `claude-sonnet-5`'s standard (post-introductory) per-token rate, confirmed
# via two independent web searches against platform.claude.com's own pricing
# page on the date below -- not a figure carried over from training data.
# `TECHNICAL_SPEC.md` §10 requires this source/date to travel with every
# evaluation record; re-verify against
# https://platform.claude.com/docs/en/about-claude/pricing before trusting
# this snapshot for real spend if much time has passed since `verified_on`.
CLAUDE_SONNET_5_PRICING = PricingSnapshot(
    model_name="claude-sonnet-5",
    input_usd_per_million_tokens=2.00,
    output_usd_per_million_tokens=10.00,
    source="https://platform.claude.com/docs/en/about-claude/pricing",
    verified_on="2026-08-22",
)
