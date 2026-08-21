from fake_incident import logs_proposal, metric_proposal

from causalops.tool_calls import (
    NativeToolCall,
    parse_tool_call,
    select_single_tool_call,
    to_tool_call,
)


def test_a_proposal_round_trips_through_a_native_tool_call() -> None:
    proposal = logs_proposal()

    call = to_tool_call(proposal, "call-1")
    parsed, reason = parse_tool_call(call)

    assert parsed == proposal
    assert reason == ""


def test_the_encoded_call_carries_the_tool_name_and_id() -> None:
    call = to_tool_call(metric_proposal(), "call-7")

    assert call.name == "query_metric"
    assert call.id == "call-7"
    assert call.type == "tool_call"


def test_no_tool_calls_in_a_turn_is_refused() -> None:
    assert select_single_tool_call([]) is None


def test_two_tool_calls_in_one_turn_are_refused_before_any_wrapper() -> None:
    first = to_tool_call(logs_proposal(), "call-1")
    second = to_tool_call(metric_proposal(), "call-2")

    assert select_single_tool_call([first, second]) is None


def test_exactly_one_tool_call_is_selected() -> None:
    only = to_tool_call(logs_proposal(), "call-1")

    assert select_single_tool_call([only]) == only


def test_a_name_that_does_not_match_the_arguments_tool_is_refused() -> None:
    """Confused deputy: `name` and `arguments.tool` are well-formed to each
    side alone and validated independently, so a mismatch is refused rather
    than resolved by trusting one side over the other."""
    metric_call = to_tool_call(metric_proposal(), "call-1")
    confused = NativeToolCall(name="query_logs", args=metric_call.args, id="call-1")

    proposal, reason = parse_tool_call(confused)

    assert proposal is None
    assert "query_logs" in reason
    assert "query_metric" in reason


def test_args_that_do_not_validate_as_any_registered_tool_are_refused() -> None:
    bogus = NativeToolCall(
        name="query_logs", args={"tool": "run_shell", "command": "whoami"}, id="c"
    )

    proposal, reason = parse_tool_call(bogus)

    assert proposal is None
    assert "run_shell" in reason


def test_args_missing_the_rationale_fields_are_refused() -> None:
    call = to_tool_call(logs_proposal(), "call-1")
    args_without_rationale = dict(call.args)
    del args_without_rationale["evidence_gap"]
    stripped = NativeToolCall(name=call.name, args=args_without_rationale, id=call.id)

    proposal, reason = parse_tool_call(stripped)

    assert proposal is None
    assert reason != ""


def test_a_rationale_field_over_the_length_bound_is_refused() -> None:
    """The final `except ValidationError` in `parse_tool_call` -- `arguments`
    and the two rationale fields all pass their own individual checks above
    it, but `ToolProposal`'s own `Field(max_length=300)` on `evidence_gap`
    still has to reject the whole proposal, not merely truncate it."""
    call = to_tool_call(logs_proposal(), "call-1")
    args = dict(call.args)
    args["evidence_gap"] = "x" * 301
    overlong = NativeToolCall(name=call.name, args=args, id=call.id)

    proposal, reason = parse_tool_call(overlong)

    assert proposal is None
    assert "evidence_gap" in reason


def test_extra_keys_in_args_do_not_stop_the_arguments_from_parsing() -> None:
    """The tool schema is flat, so `evidence_gap`/`expected_observation` sit
    beside the typed fields; the discriminated union has to ignore them
    without choking, since it never declares `extra="forbid"`."""
    call = to_tool_call(metric_proposal(), "call-1")

    parsed, reason = parse_tool_call(call)

    assert parsed is not None
    assert reason == ""
    assert parsed.arguments.tool.value == "query_metric"
