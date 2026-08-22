"""Unit 3b-2: a frozen pricing snapshot and the conservative-by-construction
reservation math the cost gate needs before it ever sends a request.

Deliberately no tokenizer. `TECHNICAL_SPEC.md` §10 calls for "a conservative
reservation using the pricing snapshot and the request's bounded input/output
allowance," and `TECHNICAL_OVERVIEW.md`'s "Default limits" table already
specifies both bounds this module turns into dollars: a 3,200-token input cap
and a 1,600-token `max_tokens` output cap. Counting tokens exactly would need
Anthropic's own tokenizer as a dependency for one estimate this module
deliberately keeps pessimistic instead -- see `estimate_input_tokens` below
for why overestimating is the safe direction and underestimating is not.
"""

from pydantic import BaseModel, ConfigDict, Field

# `TECHNICAL_OVERVIEW.md`'s "Default limits" table -- not invented here.
MAX_INPUT_TOKENS = 3_200
MAX_OUTPUT_TOKENS = 1_600

# Unit 3b-2, P2-4. `TECHNICAL_OVERVIEW.md`'s "Default limits" table's own
# "Model call | 90 seconds" row, unenforced until now. Lives beside the
# other two limits above rather than in `live_model.py` for the same reason
# they do: one file an owner can read to see every number this gate bounds,
# not three.
MAX_REQUEST_SECONDS = 90.0

# English prose tokenizes at roughly 4 characters per token across GPT- and
# Claude-family tokenizers (short punctuation-heavy tokens push the true
# average down, long common words push it up). This module assumes 3.0, not
# the closer-to-real 4.0: a *lower* chars-per-token ratio produces a *higher*
# estimated token count for the same text, and the estimate's only job is to
# never sit below what a real tokenizer would report -- a request this
# module clears as "3,199 estimated tokens" must never actually be a 3,300
# token request the gate let through. There is no tokenizer in this
# dependency tree to calibrate against directly (a deliberate omission, not
# an oversight -- see this module's docstring), so 3.0 is a documented
# judgment call, not a measured constant; a reviewer who wants a tighter or
# more defensible margin should say what evidence would set it, not just
# what number they would prefer.
PESSIMISTIC_CHARS_PER_TOKEN = 3.0


def estimate_input_tokens(text: str) -> int:
    """An empirical upper bound on `text`'s real token count -- not a
    guarantee, given `PESSIMISTIC_CHARS_PER_TOKEN`'s own reasoning above
    (a documented judgment call, not a measured constant). Two things a
    character count of `text` alone cannot see: the provider's own
    tool-use system prompt, injected server-side once tools are bound
    (`json.dumps(tools)` in `live_model.py`'s `_send` only serializes what
    this module sends, not what Anthropic prepends to it), and ordinary
    message-envelope/protocol overhead. Neither has been measured against
    a real billed request yet -- see the smoke-call runbook
    (`TECHNICAL_OVERVIEW.md`) for how the owner's first live call turns
    this from an assumption into a checked one.

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
    ~2,580-token payload -- 7,738 characters, measured directly via
    `estimate_input_tokens(json.dumps(tools))` and pinned by a dedicated
    `test_live_model.py` test, not a number carried by hand between this
    docstring and its two other citations -- `live_model.py`'s own
    comment on `_send`, and the "Default limits" row in
    `TECHNICAL_OVERVIEW.md`). Folding the tool schema into *this* cap
    would leave only ~620 tokens of prose headroom
    (`MAX_INPUT_TOKENS - 2,580`); a FINAL_ASSESSMENT turn against the
    project's own `tests/unit/fake_incident.py` scenario -- the smallest
    incident checked into this repo -- already renders to 512 tokens of
    prose with zero tool-check evidence added, so folding tools into the
    cap would refuse an ordinary run on the turn that ends it, not just
    an unusually large one. The dollar *reservation* this gate books
    (`_send`'s `reserved_usd`) is not scoped this way: it
    counts the tool payload too, because a reservation that ignores real,
    billed tokens is not conservative. The two numbers this module
    produces from the same request are intentionally different figures
    for different questions -- "is this request shaped like every other
    turn" versus "what could this request really cost."
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
