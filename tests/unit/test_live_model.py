"""Unit 3b-2: `LiveClaudeModel` against a fake `ChatAnthropic`-shaped client.

No test in this file constructs a real `langchain_anthropic.ChatAnthropic`
-- every `LiveClaudeModel` here is built with `client=FakeChatAnthropic(...)`
(`live_model.py`'s own test seam), so nothing beneath any test ever attempts
a network call, and `tests/conftest.py`'s loopback-only guard is never even
exercised by this file. `sqlite3.connect(":memory:")` stands in for
`checkpoints.db`.

The owner's standing instruction for this unit: "make the refusal path as
well tested as the success path." The refusal-path tests below
(`test_the_cost_ceiling_refuses_before_sending`,
`test_an_oversized_request_refuses_before_reserving_or_sending`,
`test_a_failed_send_leaves_the_reservation_reserved`,
`test_missing_usage_metadata_leaves_the_reservation_reserved`) each assert
not just that the right exception is raised, but that nothing was sent
and/or nothing was wrongly settled -- the two ways a gate can look like it
refused while actually letting the spend through.
"""

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall, invalid_tool_call

from causalops.cost_ledger import CostCeilingExceeded, ensure_cost_ledger_table
from causalops.domain import FinalAssessment, HypothesisUpdate, InitialPlan
from causalops.live_model import (
    RECORD_FINAL_ASSESSMENT_TOOL_NAME,
    RECORD_PLAN_TOOL_NAME,
    LiveClaudeModel,
    MissingCredential,
    MissingProviderUsage,
    _build_chat_anthropic,
)
from causalops.models import ModelRequest, Stage
from causalops.pricing import (
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_REQUEST_SECONDS,
    InputTooLarge,
    PricingSnapshot,
    estimate_input_tokens,
)
from causalops.tool_calls import select_single_tool_call
from causalops.tools import ToolName

NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=UTC)
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
CHEAP_PRICING = PricingSnapshot(
    model_name="test-model",
    input_usd_per_million_tokens=2.0,
    output_usd_per_million_tokens=10.0,
    source="test",
    verified_on="2026-01-01",
)


class _FakeBoundModel:
    def __init__(self, queue: list[AIMessage | Exception], sent: list[Any]) -> None:
        self._queue = queue
        self._sent = sent

    def invoke(self, messages: Any) -> AIMessage:
        self._sent.append(messages)
        next_item = self._queue.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class FakeChatAnthropic:
    """A minimal stand-in for `langchain_anthropic.ChatAnthropic`, exposing
    only the `.bind_tools(tools).invoke(messages)` surface `live_model.py`
    actually calls."""

    def __init__(self, responses: list[AIMessage | Exception]) -> None:
        self._queue = responses
        self.sent: list[Any] = []
        self.bound_tools: list[list[dict[str, Any]]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> _FakeBoundModel:
        self.bound_tools.append(tools)
        return _FakeBoundModel(self._queue, self.sent)


def make_model(
    conn: sqlite3.Connection,
    responses: list[AIMessage | Exception],
    *,
    ceiling_usd: float = 2.00,
    pricing: PricingSnapshot = CHEAP_PRICING,
    credential_present: bool = True,
) -> tuple[LiveClaudeModel, FakeChatAnthropic]:
    fake = FakeChatAnthropic(responses)
    model = LiveClaudeModel(
        conn,
        ceiling_usd=ceiling_usd,
        pricing=pricing,
        clock=lambda: NOW,
        client=fake,  # type: ignore[arg-type]
        credential_present=credential_present,
    )
    return model, fake


def make_request(
    *,
    stage: Stage = Stage.INITIAL_PLAN,
    model_turn: int = 0,
    digest: str = "digest-1",
    repair_errors: str | None = None,
) -> ModelRequest:
    return ModelRequest(
        stage=stage,
        system_text="system",
        context_text="context",
        repair_errors=repair_errors,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=model_turn,
        context_digest=digest,
    )


def hypotheses_args() -> list[dict[str, Any]]:
    return [
        {
            "root_cause": "CONFIG_CHANGE",
            "rank": 1,
            "missing_evidence": "whether the deploy correlates",
        },
        {
            "root_cause": "RESOURCE_POOL_SATURATION",
            "rank": 2,
            "missing_evidence": "pool utilization at the time",
        },
    ]


def plan_call(*, stop_reason: str | None, call_id: str = "plan-1") -> ToolCall:
    return ToolCall(
        name=RECORD_PLAN_TOOL_NAME,
        args={"hypotheses": hypotheses_args(), "stop_reason": stop_reason},
        id=call_id,
        type="tool_call",
    )


def metric_call(call_id: str = "domain-1", service: str = "gateway") -> ToolCall:
    return ToolCall(
        name=ToolName.QUERY_METRIC.value,
        args={
            "tool": ToolName.QUERY_METRIC.value,
            "template": "gateway_error_rate",
            "service": service,
            "window_start": "2026-08-16T10:00:00+00:00",
            "window_end": "2026-08-16T10:10:00+00:00",
            "evidence_gap": "whether gateway errors are elevated",
            "expected_observation": "an error rate spike",
        },
        id=call_id,
        type="tool_call",
    )


def message(
    tool_calls: list[ToolCall],
    *,
    usage: dict[str, int] | None = USAGE,
    invalid: list[Any] | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=tool_calls,
        invalid_tool_calls=invalid or [],
        usage_metadata=usage,  # type: ignore[arg-type]
    )


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    ensure_cost_ledger_table(connection)
    return connection


# --- propose(): tool binding shape --------------------------------------


def test_propose_binds_the_plan_tool_and_all_five_domain_tools(
    conn: sqlite3.Connection,
) -> None:
    model, fake = make_model(conn, [message([plan_call(stop_reason="no more checks")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    names = {tool["name"] for tool in tools}
    assert names == {
        RECORD_PLAN_TOOL_NAME,
        ToolName.QUERY_METRIC.value,
        ToolName.QUERY_LOGS.value,
        ToolName.LIST_RECENT_CHANGES.value,
        ToolName.GET_TOPOLOGY.value,
        ToolName.SEARCH_RUNBOOKS.value,
    }


def test_propose_reserves_at_least_the_full_wire_payload(
    conn: sqlite3.Connection,
) -> None:
    """P1-1's regression test. Before this unit's fix, `_send` reserved
    against prose alone, never the ~2,580-token tool schema `bind_tools`
    also sends on every call -- this would have failed against the frozen
    code (mutation-verified: reverting the reservation math to prose-only
    drops `reserved_usd` well below this floor). See
    `test_the_tool_payload_size_matches_what_pricingpy_assumes` below for
    the pinned, directly-measured figure this comment's "~2,580" restates
    in prose."""
    model, fake = make_model(conn, [message([plan_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    minimum_expected_reserved_usd = CHEAP_PRICING.reservation_usd(
        estimate_input_tokens(json.dumps(tools))
    )
    row = conn.execute(
        "SELECT reserved_usd FROM cost_ledger WHERE run_id = 'run-1'"
    ).fetchone()
    assert row is not None
    assert row[0] >= minimum_expected_reserved_usd


def test_the_plan_tool_definition_ships_no_leaked_engineering_docstring(
    conn: sqlite3.Connection,
) -> None:
    """P2-5's regression test for `PlanRecord`. `model_json_schema()`
    promotes a class docstring to `input_schema["description"]`; `PlanRecord`
    now has none, on purpose, so nothing here should either."""
    model, fake = make_model(conn, [message([plan_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    (plan_tool,) = [tool for tool in tools if tool["name"] == RECORD_PLAN_TOOL_NAME]
    assert "description" not in plan_tool["input_schema"]


def test_the_final_assessment_tool_definition_drops_schema_version_and_description(
    conn: sqlite3.Connection,
) -> None:
    """P2-5's regression test for `FinalAssessment`. Verified present in
    `FinalAssessment.model_json_schema()`'s own `properties` before this
    fix and correctly absent from every domain tool and `PlanRecord`
    already -- this closes the one place it still leaked."""
    call = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause": "UNDETERMINED",
            "uncertainty": "not enough evidence",
            "next_step": "check the gateway logs",
        },
        id="fa-1",
        type="tool_call",
    )
    model, fake = make_model(conn, [message([call])])

    model.respond(make_request(stage=Stage.FINAL_ASSESSMENT))

    (tools,) = fake.bound_tools
    (definition,) = tools
    assert "schema_version" not in definition["input_schema"]["properties"]
    assert "description" not in definition["input_schema"]


def test_the_tool_payload_size_matches_what_pricingpy_assumes(
    conn: sqlite3.Connection,
) -> None:
    """Post-freeze review's stale-figure finding: `pricing.py`'s
    `InputTooLarge` docstring and `live_model.py`'s own comment on `_send`
    both cite this payload's size in prose, and the cited figure has
    already gone stale once (8,329 chars, before P2-5 shrank the payload by
    stripping `PlanRecord`'s docstring and `FinalAssessment`'s
    `schema_version`/`description`) without a test noticing. This pins the
    real, measured figure so a third silent drift fails a test instead of
    only a docstring."""
    model, fake = make_model(conn, [message([plan_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    payload = json.dumps(tools)
    assert len(payload) == 7_738
    assert estimate_input_tokens(payload) == 2_580


# --- propose(): the valid shapes -----------------------------------------


def test_propose_returns_a_stop_reason_turn_with_no_tool_call(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(
        conn, [message([plan_call(stop_reason="ready for a final assessment")])]
    )

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is not None
    assert turn.parsed.proposal is None
    assert turn.parsed.stop_reason == "ready for a final assessment"
    assert turn.tool_call == ()
    assert turn.usage is not None
    assert turn.usage.input_tokens == 100


def test_propose_returns_a_native_tool_call_when_a_check_is_proposed(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([plan_call(stop_reason=None), metric_call()])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is not None
    assert turn.parsed.stop_reason is None
    assert turn.parsed.proposal is not None
    assert turn.parsed.proposal.arguments.tool is ToolName.QUERY_METRIC
    assert len(turn.tool_call) == 1
    assert turn.tool_call[0].name == ToolName.QUERY_METRIC.value


# --- propose(): the invalid shapes, each a distinct, informative reason --


def test_propose_is_invalid_when_record_plan_is_never_called(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([metric_call()])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert RECORD_PLAN_TOOL_NAME in turn.errors
    assert turn.tool_call == ()


def test_propose_is_invalid_when_record_plan_args_fail_validation(
    conn: sqlite3.Connection,
) -> None:
    bad_plan = ToolCall(
        name=RECORD_PLAN_TOOL_NAME,
        # Only one hypothesis -- `PlanRecord.hypotheses` requires 2-3.
        args={"hypotheses": hypotheses_args()[:1], "stop_reason": "done"},
        id="plan-1",
        type="tool_call",
    )
    model, _ = make_model(conn, [message([bad_plan])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.errors


def test_propose_is_invalid_when_record_plan_is_called_twice_in_one_turn(
    conn: sqlite3.Connection,
) -> None:
    """P3-2's regression test. `_split_tool_calls` used to keep only the
    most recently seen `record_plan` call (`plan_call = call`,
    unconditionally, in a loop), silently discarding an earlier one --
    mutation-verified: reverting to that assignment leaves this test
    passing on the *second* call's `stop_reason` (the one the loop saw
    last) instead of refusing, so the error-message assertion below is
    what actually catches the regression, not merely `turn.parsed is
    None`."""
    first = plan_call(stop_reason="first attempt", call_id="plan-1")
    second = plan_call(stop_reason="second attempt", call_id="plan-2")
    model, _ = make_model(conn, [message([first, second])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert RECORD_PLAN_TOOL_NAME in turn.errors
    assert "2 times" in turn.errors


def test_propose_is_invalid_when_neither_check_nor_stop_reason_is_given(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([plan_call(stop_reason=None)])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert "stop_reason" in turn.errors
    assert turn.tool_call == ()


def test_propose_is_invalid_when_a_check_and_a_stop_reason_both_appear(
    conn: sqlite3.Connection,
) -> None:
    """The reconciliation this adapter exists for: Claude's two channels
    (`record_plan`'s `stop_reason` and a genuine domain tool call)
    contradicting each other in the same turn."""
    model, _ = make_model(
        conn, [message([plan_call(stop_reason="actually stopping"), metric_call()])]
    )

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert "not both" in turn.errors
    # The domain call still travels through `tool_call` even though this
    # turn is invalid -- nothing downstream needs it here, but withholding
    # it would not be more correct, just less informative if a caller ever
    # wanted to log what was actually proposed alongside the contradiction.
    assert len(turn.tool_call) == 1


def test_propose_is_invalid_when_the_domain_call_args_are_malformed(
    conn: sqlite3.Connection,
) -> None:
    bad_metric = ToolCall(
        name=ToolName.QUERY_METRIC.value,
        args={"tool": ToolName.QUERY_METRIC.value},  # missing every other field
        id="domain-1",
        type="tool_call",
    )
    model, _ = make_model(conn, [message([plan_call(stop_reason=None), bad_metric])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.errors


def test_propose_with_two_domain_calls_leaves_select_single_tool_call_to_refuse(
    conn: sqlite3.Connection,
) -> None:
    """Two domain tool calls in one turn is a live provider's own version
    of the multi-call case replay cannot produce. This adapter still
    returns a valid `parsed` (decoded from the first candidate, genuinely,
    not a placeholder) and *all* candidates in `tool_call`, so `graph.py`'s
    own `select_single_tool_call` -- unchanged by this unit -- is the thing
    that actually refuses it, with its own specific "N checks in one turn"
    message, exactly as it does against replay."""
    model, _ = make_model(
        conn,
        [
            message(
                [
                    plan_call(stop_reason=None),
                    metric_call(call_id="domain-1", service="gateway"),
                    metric_call(call_id="domain-2", service="billing"),
                ]
            )
        ],
    )

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is not None
    assert len(turn.tool_call) == 2
    assert select_single_tool_call(turn.tool_call) is None


def test_propose_treats_a_provider_invalid_tool_call_as_invalid_output(
    conn: sqlite3.Connection,
) -> None:
    invalid = invalid_tool_call(
        name=RECORD_PLAN_TOOL_NAME,
        args="{not valid json",
        id="bad-1",
        error="could not parse arguments",
    )
    model, _ = make_model(conn, [message([], invalid=[invalid])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.tool_call == ()


def test_propose_against_hypothesis_update_uses_the_same_reconciliation(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([plan_call(stop_reason=None), metric_call()])])

    turn = model.propose(make_request(stage=Stage.HYPOTHESIS_UPDATE), HypothesisUpdate)

    assert turn.parsed is not None
    assert isinstance(turn.parsed, HypothesisUpdate)


# --- respond(): FINAL_ASSESSMENT -----------------------------------------


def test_respond_binds_only_the_final_assessment_tool(conn: sqlite3.Connection) -> None:
    call = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause": "UNDETERMINED",
            "uncertainty": "not enough evidence",
            "next_step": "check the gateway logs",
        },
        id="fa-1",
        type="tool_call",
    )
    model, fake = make_model(conn, [message([call])])

    response = model.respond(make_request(stage=Stage.FINAL_ASSESSMENT))

    (tools,) = fake.bound_tools
    assert [tool["name"] for tool in tools] == [RECORD_FINAL_ASSESSMENT_TOOL_NAME]
    assert response.content["disposition"] == "INSUFFICIENT_EVIDENCE"
    FinalAssessment.model_validate(response.content)  # round-trips cleanly


def test_respond_returns_empty_content_when_the_tool_is_never_called(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([])])

    response = model.respond(make_request(stage=Stage.FINAL_ASSESSMENT))

    assert response.content == {}


def test_respond_returns_empty_content_on_a_provider_invalid_tool_call(
    conn: sqlite3.Connection,
) -> None:
    invalid = invalid_tool_call(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args="{not valid",
        id="bad-1",
        error="unparseable",
    )
    model, _ = make_model(conn, [message([], invalid=[invalid])])

    response = model.respond(make_request(stage=Stage.FINAL_ASSESSMENT))

    assert response.content == {}


# --- the refusal path: as tested as the success path ---------------------


def test_the_cost_ceiling_refuses_before_sending(conn: sqlite3.Connection) -> None:
    model, fake = make_model(
        conn, [message([plan_call(stop_reason="done")])], ceiling_usd=0.0
    )

    with pytest.raises(CostCeilingExceeded):
        model.propose(make_request(), InitialPlan)

    assert fake.sent == []
    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 0


def test_a_missing_credential_refuses_before_reserving_or_sending(
    conn: sqlite3.Connection,
) -> None:
    """P3-3's regression test. `_send`'s order used to be estimate -> reserve
    -> invoke, with no credential check at all -- a broken-key run wrote a
    permanent `RESERVED` row for every attempt before the real `TypeError`
    surfaced deep inside `.invoke()`. Mutation-verified: removing the
    `credential_present` check in `_send` leaves `fake.sent` empty here too
    (the fake client never raises on a missing key the way a real one
    would), but this test's `cost_ledger` assertion still catches it --
    without the check, `record_reservation_before_request` runs and writes
    a row before `MissingCredential` would have been raised."""
    model, fake = make_model(
        conn, [message([plan_call(stop_reason="done")])], credential_present=False
    )

    with pytest.raises(MissingCredential):
        model.propose(make_request(), InitialPlan)

    assert fake.sent == []
    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 0


def test_a_present_credential_does_not_refuse(conn: sqlite3.Connection) -> None:
    """The default (`credential_present=True`, matching every other test in
    this file) must not be affected by this check -- a real key present
    proceeds exactly as before."""
    model, _ = make_model(conn, [message([plan_call(stop_reason="done")])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is not None


def test_an_oversized_request_refuses_before_reserving_or_sending(
    conn: sqlite3.Connection,
) -> None:
    model, fake = make_model(conn, [message([plan_call(stop_reason="done")])])
    oversized = ModelRequest(
        stage=Stage.INITIAL_PLAN,
        system_text="x" * (MAX_INPUT_TOKENS * 3 + 100),
        context_text="",
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
    )

    with pytest.raises(InputTooLarge):
        model.propose(oversized, InitialPlan)

    assert fake.sent == []
    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 0


def test_a_failed_send_leaves_the_reservation_reserved(
    conn: sqlite3.Connection,
) -> None:
    model, fake = make_model(conn, [TimeoutError("provider timed out")])

    with pytest.raises(TimeoutError):
        model.propose(make_request(), InitialPlan)

    row = conn.execute(
        "SELECT state FROM cost_ledger WHERE run_id = 'run-1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "RESERVED"


def test_missing_usage_metadata_leaves_the_reservation_reserved(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([plan_call(stop_reason="done")], usage=None)])

    with pytest.raises(MissingProviderUsage):
        model.propose(make_request(), InitialPlan)

    row = conn.execute(
        "SELECT state FROM cost_ledger WHERE run_id = 'run-1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "RESERVED"


def test_a_successful_call_reserves_then_settles(conn: sqlite3.Connection) -> None:
    model, _ = make_model(conn, [message([plan_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    row = conn.execute(
        "SELECT state, actual_usd, input_tokens, output_tokens "
        "FROM cost_ledger WHERE run_id = 'run-1'"
    ).fetchone()
    assert row is not None
    state, actual_usd, input_tokens, output_tokens = row
    assert state == "SETTLED"
    assert actual_usd is not None and actual_usd > 0
    assert input_tokens == 100
    assert output_tokens == 20


def test_two_distinct_turns_settle_two_distinct_rows(conn: sqlite3.Connection) -> None:
    """A repair changes `context_digest` (`repair_errors` is part of the
    hash), so the original ask and its repair are two distinct idempotency
    keys, not a retry of the same one -- each reserves and settles its own
    row. (True same-key idempotency -- the ledger never double-*reserving*
    one key -- is `test_cost_ledger.py`'s job; nothing in this codebase
    calls `_send` twice for one already-settled key, since there is no
    automatic retry anywhere in `src/`.)"""
    model, _ = make_model(
        conn,
        [
            message([plan_call(stop_reason="first")]),
            message([plan_call(stop_reason="second")]),
        ],
    )

    model.propose(make_request(digest="digest-1"), InitialPlan)
    model.propose(make_request(digest="digest-2"), InitialPlan)

    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 2


# --- the estimate matches what is actually sent --------------------------


def test_a_repair_turns_estimate_counts_the_correction_header(
    conn: sqlite3.Connection,
) -> None:
    """Post-freeze review, caveat 1 part 1. Before this fix, `_send`
    estimated `system_text + context_text + repair_errors` directly --
    23 characters short of what the human message (`content`) actually
    contains once `"\\n\\n## Correction needed\\n"` is spliced in ahead of
    `repair_errors` on a repair turn. The estimate is now computed from
    the same `content` string `_send` sends, not a reconstructed proxy
    that can drift from it. Mutation-verified: reverting to the direct
    three-way sum leaves this assertion computing a `reserved_usd` 23
    characters (`ceil(23/3) = 8` tokens' worth of reservation, at
    `CHEAP_PRICING`'s rates) below what this test expects."""
    model, fake = make_model(conn, [message([plan_call(stop_reason="done")])])
    repaired = make_request(repair_errors="fix the tool call", digest="digest-repair")

    model.propose(repaired, InitialPlan)

    (tools,) = fake.bound_tools
    header = "\n\n## Correction needed\nfix the tool call"
    expected_prose_tokens = estimate_input_tokens(
        repaired.system_text + repaired.context_text + header
    )
    expected_tool_tokens = estimate_input_tokens(json.dumps(tools))
    expected_reserved_usd = CHEAP_PRICING.reservation_usd(
        expected_prose_tokens + expected_tool_tokens
    )
    row = conn.execute(
        "SELECT reserved_usd FROM cost_ledger WHERE context_digest = ?",
        (repaired.context_digest,),
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(expected_reserved_usd)


# --- _build_chat_anthropic: the factory, not the private attribute -------


def test_build_chat_anthropic_pins_the_four_bounded_construction_choices() -> None:
    """Post-freeze review, caveat 2. Both reviewers rejected asserting on
    `LiveClaudeModel`'s private `_client` attribute; this asserts on
    `_build_chat_anthropic`'s own return value instead --
    `default_request_timeout`/`max_tokens`/`max_retries`/`model` are all
    public pydantic fields on `ChatAnthropic` (confirmed against the
    installed `langchain-anthropic` package's own `model_fields`, not a
    private `causalops` attribute). `max_retries` is pinned alongside the
    timeout deliberately: a silent SDK-level retry would send a second
    request under a reservation sized for one, the same failure family
    P1-1 closed for the reservation math itself. Constructing a
    `ChatAnthropic` performs no I/O -- nothing here reaches
    `tests/conftest.py`'s network guard, and no `ANTHROPIC_API_KEY` is
    needed."""
    client = _build_chat_anthropic(CHEAP_PRICING)

    assert client.default_request_timeout == MAX_REQUEST_SECONDS
    assert client.max_tokens == MAX_OUTPUT_TOKENS
    assert client.max_retries == 0
    assert client.model == CHEAP_PRICING.model_name
