"""Parses a native provider tool call into a strict registered `ToolProposal`.

`TECHNICAL_SPEC.md` §5 requires the live Claude adapter and the replay adapter
to speak the same native-tool-call protocol: the model proposes a check by
calling a registered tool, not by writing a `proposal` field into a structured
JSON response. `NativeToolCall` models that shape without depending on
`langchain-core`'s own type, so the replay adapter (Unit 1b) can emit an
"equivalent message" as the spec requires without importing anything from the
new dependency this unit deliberately leaves unused.

Nothing here calls a backend or a policy wrapper. This module only answers:
is there exactly one call, and does it parse into a well-formed proposal?
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from causalops.domain import ToolProposal
from causalops.tools import ToolArguments


class NativeToolCall(BaseModel):
    """One provider tool call, matching LangChain's `ToolCall` shape
    (`name`, `args`, `id`, `type`) field-for-field without importing it."""

    model_config = ConfigDict(frozen=True)

    name: str
    args: dict[str, JsonValue]
    id: str
    type: Literal["tool_call"] = "tool_call"


def select_single_tool_call(calls: Sequence[NativeToolCall]) -> NativeToolCall | None:
    """A model turn proposes exactly one check when invoked: zero or
    two-or-more calls in one turn is refused here, before any policy wrapper
    sees either of them."""
    if len(calls) != 1:
        return None
    return calls[0]


# `call.args` for a registered tool is flat: the tool's own typed fields plus
# `evidence_gap`/`expected_observation` as siblings, with `tool` included so
# the discriminated union below can resolve it independently of `call.name`.
# Pydantic's discriminator lookup needs that field present in the raw dict. The
# rationale keys are removed before validating the typed arguments, so the
# registered argument models can reject every other unknown key.
arguments_adapter: TypeAdapter[ToolArguments] = TypeAdapter(ToolArguments)


def summarize_errors(error: ValidationError) -> str:
    """A short, non-secret account of why a value did not fit its schema.

    Moved here from `models.py` -- `parse_tool_call` below needs the same
    formatter `parse_response` uses, and `models.py` already imports from
    this module (`NativeToolCall`, `to_tool_call`), so this is the lower
    module in that one existing import edge. `models.py` now imports this
    name instead of defining it, rather than each module keeping its own
    copy of the same pydantic-error-summary logic.
    """
    parts = [
        f"{'.'.join(str(step) for step in item['loc'])}: {item['msg']}"
        for item in error.errors()[:5]
    ]
    return "; ".join(parts)


def parse_tool_call(call: NativeToolCall) -> tuple[ToolProposal | None, str]:
    """Validate `call.args` into a whole `ToolProposal`, refusing anything
    ambiguous or malformed rather than guessing at what the model meant, and
    saying why -- a live model's repair budget is one shot, so "topic:
    Input should be 'gateway_errors', ..." is fixable where "your arguments
    were invalid" would waste it.

    The confused-deputy rule: `call.name` (the provider-controlled tool
    selection) and `arguments.tool` (the discriminator inside the
    provider-controlled args) are validated independently and must agree.
    `name="query_logs"` with `query_metric`-shaped arguments is well-formed to
    each side and ambiguous to a dispatcher, so it is refused here rather than
    left for the wrapper to guess about.
    """
    argument_values = {
        key: value
        for key, value in call.args.items()
        if key not in {"evidence_gap", "expected_observation"}
    }
    try:
        arguments = arguments_adapter.validate_python(argument_values)
    except ValidationError as error:
        return None, summarize_errors(error)
    if call.name != arguments.tool.value:
        return None, (
            f"the tool call's name {call.name!r} does not match its "
            f"arguments' declared tool {arguments.tool.value!r}"
        )
    gap = call.args.get("evidence_gap")
    observation = call.args.get("expected_observation")
    if not isinstance(gap, str) or not isinstance(observation, str):
        return (
            None,
            "evidence_gap and expected_observation must both be present strings",
        )
    try:
        return (
            ToolProposal(
                arguments=arguments, evidence_gap=gap, expected_observation=observation
            ),
            "",
        )
    except ValidationError as error:
        return None, summarize_errors(error)


def to_tool_call(proposal: ToolProposal, call_id: str) -> NativeToolCall:
    """The encoder direction: what a replay adapter emits so a scripted
    proposal and a live provider's tool call parse through the same code path
    in `parse_tool_call` above."""
    args = proposal.arguments.model_dump(mode="json")
    args["evidence_gap"] = proposal.evidence_gap
    args["expected_observation"] = proposal.expected_observation
    return NativeToolCall(name=proposal.tool.value, args=args, id=call_id)
