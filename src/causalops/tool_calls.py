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
from causalops.tools import ToolArguments, ToolName


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
# hypotheses and rationale siblings. Native provider calls already identify the
# selected registered tool by `call.name`, so the provider-facing schema omits
# the duplicate `tool` discriminator. It is restored only for the internal
# discriminated union after the name has been checked against `ToolName`.
# The non-argument siblings are removed before strict validation, so every
# other unknown key remains a refusal.
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

    The native name must be one of the registered tools. If a caller includes
    the legacy internal `tool` field, it must be a string matching that name;
    otherwise the name is canonicalized into the internal model. This preserves
    policy fingerprints and wrapper contracts without asking the provider to
    emit a redundant discriminator.
    """
    try:
        declared_name = ToolName(call.name)
    except ValueError:
        return None, f"unknown registered tool name {call.name!r}"
    if "tool" in call.args:
        declared_tool = call.args["tool"]
        if not isinstance(declared_tool, str):
            return None, "tool must be a string when present"
        if declared_tool != declared_name.value:
            return None, (
                f"the tool call's name {call.name!r} does not match its "
                f"arguments' declared tool {declared_tool!r}"
            )
    argument_values = {
        key: value
        for key, value in call.args.items()
        if key not in {"evidence_gap", "expected_observation", "hypotheses"}
    }
    argument_values["tool"] = declared_name.value
    try:
        arguments = arguments_adapter.validate_python(argument_values)
    except ValidationError as error:
        return None, summarize_errors(error)
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
