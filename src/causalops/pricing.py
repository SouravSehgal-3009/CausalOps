"""Unit 3b-2: a frozen pricing snapshot and the conservative-by-construction
reservation math the cost gate needs before it ever sends a request.

Deliberately no tokenizer. `TECHNICAL_SPEC.md` §10 calls for "a conservative
reservation using the pricing snapshot and the request's bounded input/output
allowance," and `TECHNICAL_OVERVIEW.md`'s "Default limits" table already
specifies both bounds this module turns into dollars: a 9,600-token input cap
and a 1,600-token `max_tokens` output cap. Counting tokens exactly would need
Anthropic's own tokenizer as a dependency for one estimate this module
deliberately keeps pessimistic instead -- see `estimate_input_tokens` below
for why overestimating is the safe direction and underestimating is not.
"""

from pydantic import BaseModel, ConfigDict, Field

# Unit 3b-3. `MAX_INPUT_TOKENS`'s real job is bounding a request's rendered
# PROSE in CHARACTERS -- "how much context can one turn carry" is a
# character-shaped question, and expressing the bound in tokens is only a
# unit conversion through `PESSIMISTIC_CHARS_PER_TOKEN` below. That
# conversion means the two constants are coupled: moving the ratio without
# moving this cap silently changes how many characters of prose the gate
# actually admits. Unit 3b-2 chose 3,200 tokens against a 3.0 ratio -- a
# 9,600-character budget. Unit 3b-3's smoke call measured the ratio as too
# loose (see `PESSIMISTIC_CHARS_PER_TOKEN`'s own comment) and tightened it
# to 1.0; this constant moved to 9,600 tokens in the same unit specifically
# to preserve that same 9,600-character budget, not to grant a larger one.
# `test_pricing.py`'s `test_the_input_cap_preserves_the_intended_9600_
# character_prose_budget` pins the 9,600-character figure directly (not
# derived from these two constants) so a future change to either one alone
# fails a test instead of silently refusing ordinary runs.
MAX_INPUT_TOKENS = 9_600
MAX_OUTPUT_TOKENS = 1_600

# Unit 3b-2, P2-4. `TECHNICAL_OVERVIEW.md`'s "Default limits" table's own
# "Model call | 90 seconds" row, unenforced until now. Lives beside the
# other two limits above rather than in `live_model.py` for the same reason
# they do: one file an owner can read to see every number this gate bounds,
# not three.
MAX_REQUEST_SECONDS = 90.0

# Unit 3b-3: this is now a CHECKED empirical bound, not just a documented
# judgment call -- the owner's first live run gave this module's own
# docstring the calibration point it asked for. That run's INITIAL_PLAN
# turn composed 9,249 characters (1,511 prose + 7,738 tool-definition
# payload, the pre-3b-3 figure) and the provider billed 4,099 input tokens
# -- a real ratio of about 2.26 characters per token. The OLD constant
# (3.0) would have estimated 3,084 tokens for that same request (`_send`
# estimates prose and tools as two separate ceiling divisions, 504 + 2,580,
# not one combined division over 9,249 characters -- that would give
# 3,083, one less, because ceiling division does not distribute over a
# sum): BELOW the 4,099 actually billed, a 33% undercount. That is the
# exact failure this
# module's whole design exists to prevent (an estimate must never sit below
# what a real tokenizer reports), and the settled `cost_ledger` rows only
# stayed `actual_usd <= reserved_usd` because output happened to land under
# its allowance both times -- at output saturation the same request would
# have measurably violated that invariant ($0.024198 actual against
# $0.022168 reserved).
#
# One measurement cannot separate Anthropic's own tool-use system prompt
# (additive, fixed -- invisible to `json.dumps(tools)`) from this project's
# prose simply being denser than assumed (proportional) as the cause of the
# gap. That is why the fix is the RATIO, not an additive constant: a lower
# ratio over-corrects a large prose payload if the true cause is additive,
# which is the safe direction to be wrong in; a fixed additive constant
# under-corrects exactly as prose grows, which is not. 1.0 gives an
# estimate at least 2x the one measured real data point -- the "100% buffer"
# the owner approved from this same measurement, re-derived here rather
# than asserted. A future live run's settled `input_tokens` is the next
# calibration point; if the estimate is ever found sitting below a real
# billed count again, that is the evidence this module's own docstring asks
# for to move this constant again -- not a preference for a tighter number.
PESSIMISTIC_CHARS_PER_TOKEN = 1.0


def estimate_input_tokens(text: str) -> int:
    """An empirical upper bound on `text`'s real token count -- not a
    guarantee, given `PESSIMISTIC_CHARS_PER_TOKEN`'s own reasoning above
    (now a checked empirical bound against one real measurement, per that
    constant's own comment -- still not a guarantee against every possible
    request). Two things a character count of `text` alone cannot see: the
    provider's own tool-use system prompt, injected server-side once tools
    are bound (`json.dumps(tools)` in `live_model.py`'s `_send` only
    serializes what this module sends, not what Anthropic prepends to it),
    and ordinary message-envelope/protocol overhead. The owner's first live
    call measured their *combined* effect against one request shape (see
    `PESSIMISTIC_CHARS_PER_TOKEN`'s comment for the numbers) without being
    able to separate the two -- see the smoke-call runbook
    (`TECHNICAL_OVERVIEW.md`) for how a future run extends this calibration.

    Where this bound actually matters for `reservation_usd`'s own
    worst-case claim: only once a response's real output token count
    approaches `max_output_tokens` (`T_out` saturating `MAX_OUTPUT_TOKENS`)
    does the reservation stop having slack to absorb an input
    underestimate -- below that, the unused output allowance is cushion
    the two unmodelled contributions above would have to exceed before
    `actual_usd` could exceed `reserved_usd`. Rounded up, not truncated --
    a 1-character text must estimate to at least 1 token, never 0."""
    if not text:
        return 0
    return -(-len(text) // int(PESSIMISTIC_CHARS_PER_TOKEN))  # ceiling division


class InputTooLarge(Exception):
    """Refused *before sending*: the rendered request's pessimistic token
    estimate exceeds `MAX_INPUT_TOKENS`.

    `TECHNICAL_OVERVIEW.md`'s "Default limits" table has carried this cap as
    "specified, not enforced" since Phase 2; this is the enforcement, and it
    refuses rather than truncates -- silently cutting context is a value
    lost where no assertion downstream could ever see it happened. Raised,
    not returned as an error string, for the same reason `CostCeilingExceeded`
    is raised in `cost_ledger.py`: `graph.py`'s `ask_once` must be able to
    tell "refuse, do not repair" apart from "invalid output, repair once" --
    repairing an oversized request by appending a correction message can
    only make it larger, never smaller, so treating this as ordinary invalid
    output would waste the one repair slot on a request no repair could fix.

    Unit 3b-2, P1-1: **this cap counts prose only** -- `live_model.py`'s
    `_send` estimates it from `system_text + content` (the rendered human
    message, repair correction included when present), deliberately never
    the six tool definitions `bind_tools` also sends on every call
    (`_plan_tool_definition`/`_domain_tool_definitions`, a fixed
    ~7,595-token payload -- 7,595 characters, the two now equal because
    Unit 3b-3 set `PESSIMISTIC_CHARS_PER_TOKEN` to 1.0 -- measured directly
    via `estimate_input_tokens(json.dumps(tools))` and pinned by a
    dedicated `test_live_model.py` test, not a number carried by hand
    between this docstring and its two other citations -- `live_model.py`'s
    own comment on `_send`, and the "Default limits" row in
    `TECHNICAL_OVERVIEW.md`). Folding the tool schema into *this* cap would
    leave only ~2,005 tokens of prose headroom (`MAX_INPUT_TOKENS -
    7,595`); a FINAL_ASSESSMENT turn against the project's own
    `tests/unit/fake_incident.py` scenario -- the smallest incident checked
    into this repo -- already renders to 1,280 tokens of prose with zero
    tool-check evidence added (Unit 3b-2's "512 tokens" claim for this same
    illustration was never pinned by a test and could not be reproduced
    from the real `render_context`/`Budgets` call site during Unit 3b-3's
    review; 1,280 is measured directly and pinned by
    `test_live_model.py`'s `test_the_smallest_final_assessment_prose_
    matches_what_inputtoolarge_assumes`), so folding tools into the cap
    would still leave less than double the smallest scenario's own prose as
    headroom before any evidence is added, not just an unusually large run.
    The dollar *reservation* this gate books (`_send`'s `reserved_usd`) is
    not scoped this way: it counts the tool payload too, because a
    reservation that ignores real, billed tokens is not conservative. The
    two numbers this module produces from the same request are
    intentionally different figures for different questions -- "is this
    request shaped like every other turn" versus "what could this request
    really cost."
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
