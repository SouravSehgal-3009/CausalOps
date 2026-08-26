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
from fake_incident import alert_packet, incident_scope
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall, invalid_tool_call
from pydantic import ValidationError

from causalops.cost_ledger import (
    AmbiguousReservationNotResent,
    CostCeilingExceeded,
    ensure_cost_ledger_table,
    record_reservation_before_request,
    settle_reservation,
)
from causalops.domain import (
    Budgets,
    FinalAssessment,
    HypothesisUpdate,
    InitialPlan,
    PolicyResult,
    ReasonCode,
    RetrievalMode,
    RunbookPassage,
    ToolProposal,
)
from causalops.live_model import (
    RECORD_FINAL_ASSESSMENT_TOOL_NAME,
    RECORD_STOP_TOOL_NAME,
    LiveClaudeModel,
    MissingCredential,
    MissingProviderUsage,
    _build_chat_anthropic,
    _domain_tool_definitions,
    _final_assessment_tool_definition,
    _stop_tool_definition,
)
from causalops.models import ModelRequest, Stage
from causalops.policy import authorize
from causalops.pricing import (
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_REQUEST_SECONDS,
    InputTooLarge,
    PricingSnapshot,
    estimate_input_tokens,
)
from causalops.prompts import STAGE_INSTRUCTIONS, SYSTEM_TEXT, render_context
from causalops.tools import SearchRunbooksArguments, ToolName

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
        self.bind_kwargs: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> _FakeBoundModel:
        self.bound_tools.append(tools)
        self.bind_kwargs.append(kwargs)
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


def stop_call(*, stop_reason: str | None, call_id: str = "stop-1") -> ToolCall:
    return ToolCall(
        name=RECORD_STOP_TOOL_NAME,
        args={"hypotheses": hypotheses_args(), "stop_reason": stop_reason},
        id=call_id,
        type="tool_call",
    )


def metric_call(call_id: str = "domain-1", service: str = "gateway") -> ToolCall:
    return ToolCall(
        name=ToolName.QUERY_METRIC.value,
        args={
            "template": "gateway_error_rate",
            "service": service,
            "window_start": "2026-08-16T10:00:00+00:00",
            "window_end": "2026-08-16T10:10:00+00:00",
            "hypotheses": hypotheses_args(),
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
    content: Any = "",
) -> AIMessage:
    return AIMessage(
        content=content,
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


def test_propose_binds_the_stop_tool_and_all_five_domain_tools(
    conn: sqlite3.Connection,
) -> None:
    model, fake = make_model(conn, [message([stop_call(stop_reason="no more checks")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    names = {tool["name"] for tool in tools}
    assert names == {
        RECORD_STOP_TOOL_NAME,
        ToolName.QUERY_METRIC.value,
        ToolName.QUERY_LOGS.value,
        ToolName.LIST_RECENT_CHANGES.value,
        ToolName.GET_TOPOLOGY.value,
        ToolName.SEARCH_RUNBOOKS.value,
    }
    assert fake.bind_kwargs == [{"parallel_tool_calls": False}]


def test_propose_reserves_at_least_the_full_wire_payload(
    conn: sqlite3.Connection,
) -> None:
    """P1-1's regression test. Before this unit's fix, `_send` reserved
    against prose alone, never the current ~12,829-token tool schema `bind_tools`
    also sends on every call -- this would have failed against the frozen
    code (mutation-verified: reverting the reservation math to prose-only
    drops `reserved_usd` well below this floor). See
    `test_the_tool_payload_size_matches_what_pricingpy_assumes` below for
    the pinned, directly-measured figure this comment's "~12,829" restates
    in prose."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])

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


def test_the_stop_tool_definition_ships_no_leaked_engineering_docstring(
    conn: sqlite3.Connection,
) -> None:
    """The stop-record schema keeps maintainer prose out of the request."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    (plan_tool,) = [tool for tool in tools if tool["name"] == RECORD_STOP_TOOL_NAME]
    assert "description" not in plan_tool["input_schema"]


def test_the_stop_tool_definition_requires_a_non_null_stop_reason(
    conn: sqlite3.Connection,
) -> None:
    """A stop is an explicit, meaningful alternative to a check call."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    (plan_tool,) = [tool for tool in tools if tool["name"] == RECORD_STOP_TOOL_NAME]
    stop_reason_property = plan_tool["input_schema"]["properties"]["stop_reason"]
    assert "default" not in stop_reason_property
    assert stop_reason_property["type"] == "string"
    assert stop_reason_property["minLength"] == 1
    assert stop_reason_property["maxLength"] == 300
    assert "stop_reason" in plan_tool["input_schema"]["required"]


def test_the_final_assessment_tool_definition_drops_schema_version_and_description(
    conn: sqlite3.Connection,
) -> None:
    """P2-5's regression test for `FinalAssessment`. Verified present in
    `FinalAssessment.model_json_schema()`'s own `properties` before this
    fix and correctly absent from every domain tool and `StopRecord`
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
    """Pin the emitted proposal schema so reservations and current docs
    cannot silently drift from its measured serialized size."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    payload = json.dumps(tools)
    # Lab-defect-fix Unit 3, W1: `QueryMetricArguments`/`QueryLogsArguments`/
    # `ListRecentChangesArguments` each gained two optional window fields
    # with their own descriptions, so the emitted schema grew from 12,011 to
    # 12,824 bytes -- not a drift, the mechanical cost of Q1's window
    # contract, five-fold-duplicated the same way `_domain_tool_definitions`'s
    # own docstring already explains for `HypothesesRecord`'s schema.
    # Fix F1: `MetricTemplate.RESOURCE_POOL_IN_USE` (21 chars) renamed to
    # `RESOURCE_POOL_UTILIZATION` (26 chars) inside `QueryMetricArguments`'s
    # own schema -- a real +5-byte drift this pin exists to catch, not one
    # to explain away, so it moves the literal rather than the comment.
    assert len(payload) == 12_829
    # Unit 3b-3: `PESSIMISTIC_CHARS_PER_TOKEN` is 1.0, so ceiling division
    # makes the token estimate equal the character count exactly -- this
    # is the real behaviour, not a coincidence to simplify away.
    assert estimate_input_tokens(payload) == 12_829


def test_the_respond_tool_payload_size_matches_what_pricingpy_assumes(
    conn: sqlite3.Connection,
) -> None:
    """Post-freeze review, N1. The test above pins ONLY `propose()`'s
    payload (`_stop_tool_definition()` plus the five `_domain_tool_
    definitions()`) -- `_final_assessment_tool_definition()` is never in
    that binding (confirmed by reading `propose()`/`respond()` directly:
    they bind two disjoint tool lists). `_send`'s earlier comment called
    one figure per-stage while only ONE of the two stages was
    ever actually pinned; `respond()`'s payload, priced by `reservation_
    usd` on every FINAL_ASSESSMENT turn exactly the way `propose()`'s is
    on every INVESTIGATE turn, had no test noticing if it drifted."""
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
    payload = json.dumps(tools)
    assert len(payload) == 2_292
    assert estimate_input_tokens(payload) == 2_292


def test_domain_tool_schemas_use_native_names_and_require_hypotheses(
    conn: sqlite3.Connection,
) -> None:
    """Native call names select internal tool variants; check schemas require
    hypotheses while omitting the redundant provider-facing discriminator."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])

    model.propose(make_request(), InitialPlan)

    (tools,) = fake.bound_tools
    domain_tools = [tool for tool in tools if tool["name"] != RECORD_STOP_TOOL_NAME]
    assert len(domain_tools) == 5
    for tool in domain_tools:
        properties = tool["input_schema"]["properties"]
        assert "tool" not in properties
        assert "tool" not in tool["input_schema"]["required"]
        assert "hypotheses" in tool["input_schema"]["required"]


def test_the_smallest_final_assessment_prose_matches_what_inputtoolarge_assumes() -> (
    None
):
    """The "Default limits" table cites how large a FINAL_ASSESSMENT turn's
    prose gets with no evidence added,
    to argue folding the tool schema into `MAX_INPUT_TOKENS` would refuse
    ordinary runs. Unit 3b-2's version of that figure ("512 tokens") was
    never pinned by a test, and Unit 3b-3's review could not reproduce it
    from the real `render_context`/`Budgets` call site `graph.py`'s
    `_render_stage_request` actually uses -- the closest reconstruction,
    at the OLD 3.0 ratio, was 427 tokens, not a match. Rather than carry
    the unreproducible number forward, this computes the real figure
    directly from the same inputs `_render_stage_request` builds (`_model_
    calls_left` is `budgets.model_calls - model_calls_used`; `_tools_left`
    on zero receipts is `budgets.executed_tools`, since nothing has been
    reserved from that ledger yet) against this repo's own smallest
    checked-in scenario (`tests/unit/fake_incident.py`), one INITIAL_PLAN
    turn already spent and stopped, no evidence added -- and pins it, so
    the two docstrings that cite it can point at a test instead of a
    number carried by hand a third time."""
    scope = incident_scope()
    packet = alert_packet()
    budgets = Budgets()
    model_calls_used = 1  # the INITIAL_PLAN turn that stopped immediately

    context = render_context(
        packet,
        scope,
        evidence=[],
        markers=[],
        model_calls_left=budgets.model_calls - model_calls_used,
        checks_left=budgets.executed_tools,
        passages=(),
    )
    context_text = f"{context}\n\n## Task\n{STAGE_INSTRUCTIONS[Stage.FINAL_ASSESSMENT]}"
    total = SYSTEM_TEXT + context_text

    # Lab-defect-fix Unit 3, W1: `SYSTEM_TEXT` gained one sentence stating
    # the window default/clamp contract, so this prose figure moved from
    # 1,392 to 1,561. Lab-defect-fix Unit 4, W18: `SYSTEM_TEXT` gained one
    # more sentence stating the CONFIG_CHANGE label convention, moving it
    # again, from 1,561 to 1,747. A Codex review round on Unit 4 rewrote
    # that same sentence to state the label-priority rule explicitly
    # (a more specific label wins over CONFIG_CHANGE when its own evidence
    # is present), moving it a third time, from 1,747 to 1,895 -- the same
    # mechanism `test_graph_frozen_reports.py`'s own module docstring
    # already documents for `final_context_digest`.
    assert len(total) == 1_895
    # Unit 3b-3: ratio 1.0 makes the token estimate equal the character
    # count -- the real behaviour, asserted directly rather than derived.
    assert estimate_input_tokens(total) == 1_895


def test_a_post_retrieval_proposal_sends_when_only_its_schema_exceeds_the_cap(
    conn: sqlite3.Connection,
) -> None:
    """The prose cap admits a real post-retrieval proposal; reservations
    still price the proposal schema separately."""
    scope = incident_scope()
    packet = alert_packet()
    budgets = Budgets()
    model_calls_used = 1
    passages = tuple(
        RunbookPassage(
            passage_id=f"runbook-{i}",
            content="x" * 800,
            source_version="v1",
            content_hash="hash",
            score=1.0,
            retrieval_mode=RetrievalMode.FTS5_LEXICAL,
        )
        for i in range(budgets.runbook_passages)
    )

    context = render_context(
        packet,
        scope,
        evidence=[],
        markers=[],
        model_calls_left=budgets.model_calls - model_calls_used,
        checks_left=budgets.executed_tools - 1,
        passages=passages,
    )
    context_text = (
        f"{context}\n\n## Task\n{STAGE_INSTRUCTIONS[Stage.HYPOTHESIS_UPDATE]}"
    )
    total = SYSTEM_TEXT + context_text
    proposal_schema_tokens = estimate_input_tokens(
        json.dumps([_stop_tool_definition(), *_domain_tool_definitions()])
    )
    prose_tokens = estimate_input_tokens(total)
    request = ModelRequest(
        stage=Stage.HYPOTHESIS_UPDATE,
        system_text=SYSTEM_TEXT,
        context_text=context_text,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=1,
        context_digest="post-retrieval",
    )
    model, fake = make_model(conn, [message([metric_call()])])

    assert prose_tokens <= MAX_INPUT_TOKENS
    assert prose_tokens + proposal_schema_tokens > MAX_INPUT_TOKENS
    assert model.propose(request, HypothesisUpdate).parsed is not None
    assert fake.sent


def test_a_request_at_the_intended_9600_character_budget_is_not_refused(
    conn: sqlite3.Connection,
) -> None:
    """The end-to-end sibling of `test_pricing.py`'s cap-preserves-the-
    budget sentinel: proves the admitted side of the boundary through a
    real `propose()` call, not just the arithmetic. A system+context
    request at exactly the owner-approved 9,600-character prose budget
    must still be accepted -- if a future change to `PESSIMISTIC_CHARS_
    PER_TOKEN` or `MAX_INPUT_TOKENS` alone silently shrinks the effective
    character budget, an ordinary-sized request starts refusing here."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])
    at_budget = ModelRequest(
        stage=Stage.INITIAL_PLAN,
        system_text="x" * 9_600,
        context_text="",
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
    )

    turn = model.propose(at_budget, InitialPlan)

    assert turn.parsed is not None
    assert fake.sent != []


# --- propose(): the valid shapes -----------------------------------------


def test_propose_returns_a_stop_reason_turn_with_no_tool_call(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(
        conn, [message([stop_call(stop_reason="ready for a final assessment")])]
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
    model, _ = make_model(conn, [message([metric_call()])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is not None
    assert turn.parsed.stop_reason is None
    assert turn.parsed.proposal is not None
    assert turn.parsed.proposal.arguments.tool is ToolName.QUERY_METRIC
    assert len(turn.tool_call) == 1
    assert turn.tool_call[0].name == ToolName.QUERY_METRIC.value


# --- propose(): the invalid shapes, each a distinct, informative reason --


def test_propose_is_invalid_when_no_tool_is_called(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert "exactly one" in turn.errors
    assert turn.tool_call == ()


def test_propose_is_invalid_when_record_stop_args_fail_validation(
    conn: sqlite3.Connection,
) -> None:
    bad_plan = ToolCall(
        name=RECORD_STOP_TOOL_NAME,
        # Only one hypothesis -- `StopRecord.hypotheses` requires 2-3.
        args={"hypotheses": hypotheses_args()[:1], "stop_reason": "done"},
        id="plan-1",
        type="tool_call",
    )
    model, _ = make_model(conn, [message([bad_plan])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.errors


def test_propose_is_invalid_when_two_tools_are_called(
    conn: sqlite3.Connection,
) -> None:
    """The live adapter rejects all multi-call shapes before graph dispatch."""
    first = stop_call(stop_reason="first attempt", call_id="stop-1")
    second = stop_call(stop_reason="second attempt", call_id="stop-2")
    model, _ = make_model(conn, [message([first, second])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert "2 tools" in turn.errors


def test_propose_is_invalid_when_stop_reason_is_null(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([stop_call(stop_reason=None)])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert "string" in turn.errors
    assert turn.tool_call == ()


def test_propose_is_invalid_when_stop_reason_is_empty(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([stop_call(stop_reason="")])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert "at least 1 character" in turn.errors
    assert turn.tool_call == ()


def test_propose_is_invalid_when_stop_and_check_are_both_called(
    conn: sqlite3.Connection,
) -> None:
    """A stop call and a check call violate the exact-one-call protocol."""
    model, _ = make_model(
        conn, [message([stop_call(stop_reason="actually stopping"), metric_call()])]
    )

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert "2 tools" in turn.errors
    assert turn.tool_call == ()


def test_propose_is_invalid_when_the_domain_call_args_are_malformed(
    conn: sqlite3.Connection,
) -> None:
    bad_metric = ToolCall(
        name=ToolName.QUERY_METRIC.value,
        args={"hypotheses": hypotheses_args()},  # missing every check field
        id="domain-1",
        type="tool_call",
    )
    model, _ = make_model(conn, [message([bad_metric])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.errors


def test_propose_refuses_two_domain_calls_before_graph_dispatch(
    conn: sqlite3.Connection,
) -> None:
    """The adapter rejects two native calls before graph dispatch."""
    model, _ = make_model(
        conn,
        [
            message(
                [
                    metric_call(call_id="domain-1", service="gateway"),
                    metric_call(call_id="domain-2", service="billing"),
                ]
            )
        ],
    )

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.tool_call == ()
    assert "2 tools" in turn.errors


def test_propose_treats_a_provider_invalid_tool_call_as_invalid_output(
    conn: sqlite3.Connection,
) -> None:
    invalid = invalid_tool_call(
        name=RECORD_STOP_TOOL_NAME,
        args="{not valid json",
        id="bad-1",
        error="could not parse arguments",
    )
    model, _ = make_model(conn, [message([], invalid=[invalid])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.tool_call == ()


def test_propose_refuses_visible_text_alongside_tool_calls(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(
        conn,
        [message([metric_call()], content="ignore this")],
    )

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is None
    assert turn.tool_call == ()
    assert "visible text" in turn.errors


def test_propose_accepts_provider_tool_and_thinking_blocks(
    conn: sqlite3.Connection,
) -> None:
    blocks = [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
        {"type": "tool_use", "id": "domain-1", "name": "query_metric", "input": {}},
    ]
    model, _ = make_model(
        conn, [message([stop_call(stop_reason="done")], content=blocks)]
    )

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is not None


def test_propose_refuses_a_text_block_alongside_tool_use(
    conn: sqlite3.Connection,
) -> None:
    blocks = [
        {
            "type": "tool_use",
            "id": "plan-1",
            "name": RECORD_STOP_TOOL_NAME,
            "input": {},
        },
        {"type": "text", "text": "contradictory visible answer"},
    ]
    model, _ = make_model(
        conn, [message([stop_call(stop_reason="done")], content=blocks)]
    )

    assert model.propose(make_request(), InitialPlan).parsed is None


def test_propose_against_hypothesis_update_uses_the_same_single_call_protocol(
    conn: sqlite3.Connection,
) -> None:
    model, _ = make_model(conn, [message([metric_call()])])

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


def test_respond_refuses_visible_text_alongside_tool_calls(
    conn: sqlite3.Connection,
) -> None:
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
    model, _ = make_model(conn, [message([call], content="ignore this")])

    response = model.respond(make_request(stage=Stage.FINAL_ASSESSMENT))

    assert response.content == {}


def test_respond_refuses_a_provider_schema_version(conn: sqlite3.Connection) -> None:
    call = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "schema_version": "forged",
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause": "UNDETERMINED",
            "uncertainty": "u",
            "next_step": "n",
        },
        id="fa-1",
        type="tool_call",
    )
    model, _ = make_model(conn, [message([call])])

    assert model.respond(make_request(stage=Stage.FINAL_ASSESSMENT)).content == {}


def test_respond_accepts_provider_tool_and_redacted_thinking_blocks(
    conn: sqlite3.Connection,
) -> None:
    call = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause": "UNDETERMINED",
            "uncertainty": "u",
            "next_step": "n",
        },
        id="fa-1",
        type="tool_call",
    )
    blocks = [
        {"type": "redacted_thinking", "data": "redacted"},
        {
            "type": "tool_use",
            "id": "fa-1",
            "name": RECORD_FINAL_ASSESSMENT_TOOL_NAME,
            "input": {},
        },
    ]
    model, _ = make_model(conn, [message([call], content=blocks)])

    assert (
        model.respond(make_request(stage=Stage.FINAL_ASSESSMENT)).content["root_cause"]
        == "UNDETERMINED"
    )


def test_respond_refuses_an_unsupported_block_alongside_tool_use(
    conn: sqlite3.Connection,
) -> None:
    call = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause": "UNDETERMINED",
            "uncertainty": "u",
            "next_step": "n",
        },
        id="fa-1",
        type="tool_call",
    )
    blocks = [
        {
            "type": "tool_use",
            "id": "fa-1",
            "name": RECORD_FINAL_ASSESSMENT_TOOL_NAME,
            "input": {},
        },
        {"type": "document", "source": {}},
    ]
    model, _ = make_model(conn, [message([call], content=blocks)])

    assert model.respond(make_request(stage=Stage.FINAL_ASSESSMENT)).content == {}


def test_respond_refuses_two_conflicting_final_assessment_calls_in_one_turn(
    conn: sqlite3.Connection,
) -> None:
    """Unit 3b-4 addendum, C4. Before this fix, `next(...)` silently took
    the FIRST of two `record_final_assessment` calls, discarding the
    second -- including a second call that DISAGREES with the first, which
    is exactly what these two do (`DIAGNOSED` vs `INSUFFICIENT_EVIDENCE`).
    `response.content == {}` proves this refuses the whole turn rather
    than silently picking either disposition; a downstream diagnosis that
    happened to match the first call's answer, by coincidence, would not
    have caught this."""
    first = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "disposition": "DIAGNOSED",
            "root_cause": "CONFIG_CHANGE",
            "supporting_evidence_ids": ["e1"],
            "uncertainty": "u",
            "next_step": "n",
        },
        id="fa-1",
        type="tool_call",
    )
    second = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause": "UNDETERMINED",
            "uncertainty": "u2",
            "next_step": "n2",
        },
        id="fa-2",
        type="tool_call",
    )
    model, _ = make_model(conn, [message([first, second])])

    response = model.respond(make_request(stage=Stage.FINAL_ASSESSMENT))

    assert response.content == {}


def test_respond_refuses_a_matching_call_alongside_an_unbound_extra_call(
    conn: sqlite3.Connection,
) -> None:
    """Post-freeze review, Finding 3. C4's own fix above checked only
    `len(matching_calls) != 1` -- a turn with exactly one `record_final_
    assessment` call AND some other tool name `respond()` never bound
    (`_final_assessment_tool_definition()` is the only tool offered) would
    have passed that check, silently dropping the extra call the same way
    C4 was built to stop happening for a second MATCHING call. Not proven
    reachable against a real provider offline, per the installed
    `langchain-anthropic` source correctness read -- but the fix costs one
    more length check, so it is applied regardless."""
    matching = ToolCall(
        name=RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        args={
            "disposition": "INSUFFICIENT_EVIDENCE",
            "root_cause": "UNDETERMINED",
            "uncertainty": "u",
            "next_step": "n",
        },
        id="fa-1",
        type="tool_call",
    )
    unbound_extra = ToolCall(
        name="some_other_tool",
        args={"anything": "at all"},
        id="extra-1",
        type="tool_call",
    )
    model, _ = make_model(conn, [message([matching, unbound_extra])])

    response = model.respond(make_request(stage=Stage.FINAL_ASSESSMENT))

    assert response.content == {}


# --- the refusal path: as tested as the success path ---------------------


def test_the_cost_ceiling_refuses_before_sending(conn: sqlite3.Connection) -> None:
    model, fake = make_model(
        conn, [message([stop_call(stop_reason="done")])], ceiling_usd=0.0
    )

    with pytest.raises(CostCeilingExceeded):
        model.propose(make_request(), InitialPlan)

    assert fake.sent == []
    rows = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()[0]
    assert rows == 0


def test_a_pending_reservation_refuses_to_resend_without_touching_the_transport(
    conn: sqlite3.Connection,
) -> None:
    """Unit 3b-4 addendum, Group B, codex P1 -- the double-spend bug.
    Simulates the exact scenario `TECHNICAL_SPEC.md` §5's idempotency key
    exists for: a crash between an earlier reserve and its settle, then a
    LangGraph resume that re-renders the identical stage. Pre-inserts a
    `RESERVED` row for exactly the key `propose()` below will compute
    (`make_request()`'s `run_id`/`graph_phase`/`model_turn`/`digest`
    defaults), simulating that earlier, unsettled attempt directly against
    `cost_ledger` rather than via a real `_send` call.

    Before this fix, `record_reservation_before_request` returned that
    existing row and `_send` proceeded to call the transport anyway --
    `test_an_identical_retry_reads_back_the_same_row_not_a_second_one`
    (`test_cost_ledger.py`) only ever checked the *ledger* stayed
    one-row-one-dollar; it never checked whether `self._client.bind_tools
    (...).invoke(...)` was reached a second time. This is that missing
    check, at the one layer that can actually see it: `fake.sent` records
    every `invoke()` call `_send` makes, so `fake.sent == []` here is
    direct proof the provider was never touched, not an inference from
    ledger state."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])
    record_reservation_before_request(
        conn,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
        reserved_usd=0.01,
        requested_at=NOW,
        ceiling_usd=2.00,
    )

    with pytest.raises(AmbiguousReservationNotResent):
        model.propose(make_request(), InitialPlan)

    assert fake.sent == []
    row = conn.execute(
        "SELECT state FROM cost_ledger WHERE run_id = 'run-1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "RESERVED"


def test_a_settled_reservation_also_refuses_to_resend(conn: sqlite3.Connection) -> None:
    """Post-freeze review, P2-1. The sibling test above only covers a
    pre-existing `RESERVED` row; this covers `SETTLED` -- a real, tempting
    future "optimization" is `if not is_new and reservation.state ==
    "RESERVED":`, refusing only the unsettled case and letting a `SETTLED`
    one through on the theory that "it already succeeded, so a retry is
    just wasteful, not dangerous." That reasoning is backwards: a
    `SETTLED` row means the earlier request was ALREADY billed, so
    resending it is strictly worse than the `RESERVED` case -- the money
    is definitely gone, not merely possibly in flight. Settling the
    pre-existing row first (mirroring a real earlier `_send` call that
    completed normally) proves the refusal does not depend on which state
    the stale reservation is in."""
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])
    record_reservation_before_request(
        conn,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
        reserved_usd=0.01,
        requested_at=NOW,
        ceiling_usd=2.00,
    )
    settle_reservation(
        conn,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest-1",
        actual_usd=0.008,
        input_tokens=100,
        output_tokens=20,
        settled_at=NOW,
    )

    with pytest.raises(AmbiguousReservationNotResent):
        model.propose(make_request(), InitialPlan)

    assert fake.sent == []
    row = conn.execute(
        "SELECT state FROM cost_ledger WHERE run_id = 'run-1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "SETTLED"


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
        conn, [message([stop_call(stop_reason="done")])], credential_present=False
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
    model, _ = make_model(conn, [message([stop_call(stop_reason="done")])])

    turn = model.propose(make_request(), InitialPlan)

    assert turn.parsed is not None


def test_an_oversized_request_refuses_before_reserving_or_sending(
    conn: sqlite3.Connection,
) -> None:
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])
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
    model, _ = make_model(conn, [message([stop_call(stop_reason="done")], usage=None)])

    with pytest.raises(MissingProviderUsage):
        model.propose(make_request(), InitialPlan)

    row = conn.execute(
        "SELECT state FROM cost_ledger WHERE run_id = 'run-1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "RESERVED"


def test_a_successful_call_reserves_then_settles(conn: sqlite3.Connection) -> None:
    model, _ = make_model(conn, [message([stop_call(stop_reason="done")])])

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
            message([stop_call(stop_reason="first")]),
            message([stop_call(stop_reason="second")]),
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
    model, fake = make_model(conn, [message([stop_call(stop_reason="done")])])
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


# --- Unit 3b-4, item 5: the schema-vs-application cross-check ------------
#
# The root-cause investigation behind this unit found five payloads the
# emitted wire schema ACCEPTS that the application REFUSES -- a class of
# gap Unit 3b-3's own review could not have caught, because it tested the
# schema against itself ("is `default` consistent with `required`?"), never
# against the application code that actually refuses a run. `schema_accepts`
# below is a small, dependency-free validator over exactly the JSON Schema
# keywords these seven current tool schemas use -- adapted from
# `3b4-KEEP-schema_gap_check.py` (the investigation's own working
# prototype, verified there to reproduce all five contracts) into a real,
# asserting test. `KNOWN_PROSE_ONLY_CONTRACTS` is the registry every
# payload the two checks below exercise must be named in, with a pointer to
# the prose that actually carries the rule -- `test_an_undocumented_prose_
# only_contract_fails_the_check` demonstrates what happens to one that
# is not.
#
# A NAMED LIMIT, post-freeze review, N4: this mechanism checks ONE tool's
# schema against ONE payload for that same tool -- it structurally cannot
# express, and so cannot detect, any rule that spans more than one schema
# or more than one call in a turn. None of these are expressible through
# `schema_accepts`, and none can ever appear in `KNOWN_PROSE_ONLY_
# CONTRACTS` via this mechanism, no matter how thoroughly the registry is
# audited:
# - A provider response calling more than one tool -- a cross-call count,
#   not a single payload. `LiveClaudeModel.propose()` rejects it before the
#   graph sees a candidate.
# - A forged evidence-id citation (`ReasonCode.FORGED_EVIDENCE_REFERENCE`)
#   -- refused by cross-referencing a live evidence STORE built from the
#   run so far, external to any tool's own schema entirely.
# This is a boundary on what the mechanism CAN see, not an incomplete
# audit of what it has looked at -- a future engineer should not read the
# five (now four, item 1 closed one) known contracts as the exhaustive
# list of every prose-only gap this codebase could ever have, only the
# ones expressible as one schema's own single-payload shape.


def resolve_schema_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        schema = root["$defs"][schema["$ref"].split("/")[-1]]
    return schema


def schema_accepts(
    schema: dict[str, Any], value: Any, root: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """True if `value` satisfies `schema`, covering only the keywords
    `query_metric`/`query_logs`/`list_recent_changes`/`get_topology`/
    `search_runbooks`/`record_stop`/`record_final_assessment`'s emitted
    schemas actually use -- not a general-purpose JSON Schema validator."""
    root = root or schema
    schema = resolve_schema_ref(schema, root)
    if "anyOf" in schema:
        if any(schema_accepts(s, value, root)[0] for s in schema["anyOf"]):
            return True, ""
        return False, "no anyOf branch matched"
    kind = schema.get("type")
    if kind == "null":
        return (value is None), "not null"
    if value is None and kind is not None:
        return False, f"null against type {kind}"
    if kind == "object":
        if not isinstance(value, dict):
            return False, "not an object"
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(schema.get("properties", {}))
            if unknown:
                return False, f"unknown properties {sorted(unknown)}"
        for name in schema.get("required", []):
            if name not in value:
                return False, f"missing required '{name}'"
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                ok, why = schema_accepts(sub, value[name], root)
                if not ok:
                    return False, f"{name}: {why}"
        return True, ""
    if kind == "array":
        if not isinstance(value, list):
            return False, "not an array"
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False, "too few items"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False, "too many items"
        for item in value:
            ok, why = schema_accepts(schema.get("items", {}), item, root)
            if not ok:
                return False, f"item: {why}"
        return True, ""
    if kind == "string":
        if not isinstance(value, str):
            return False, "not a string"
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False, "too short"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False, "too long"
        if "enum" in schema and value not in schema["enum"]:
            return False, "not in enum"
        if "const" in schema and value != schema["const"]:
            return False, "not the const"
        return True, ""
    if kind == "integer":
        if not isinstance(value, int):
            return False, "not an integer"
        if "minimum" in schema and value < schema["minimum"]:
            return False, "below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return False, "above maximum"
        return True, ""
    if "enum" in schema:
        return (value in schema["enum"]), "not in enum"
    return True, ""


# Post-freeze review, P2-4. `schema_accepts` above is a hand-maintained
# mirror of the JSON Schema keywords the seven current emitted tool schemas
# use, not a general-purpose validator -- and a hand-maintained mirror can
# silently narrow: `minItems`/`maxItems`/`minimum`/`maximum` were dropped
# somewhere between `3b4-KEEP-schema_gap_check.py` (the investigation's own
# scratch script, which already implements all four) and this file, in the
# WEAKENING direction. Correctness measured the concrete consequence:
# `record_stop` with a single hypothesis, `Hypothesis.rank=0`, and
# `search_runbooks.limit=999` are all real pydantic rejections that
# `schema_accepts` reported as "accepted." Nothing among the known
# contracts below happened to depend on any of the four, so nothing was
# misclassified as a RESULT of the gap -- but the mechanism's own coverage
# claim ("covering only the keywords these schemas actually use") was
# false, silently, with no test noticing.
#
# The two sets below, plus the walker and the test after them, are the
# stronger fix the owner chose over just restoring the four keywords:
# a coverage assertion that catches the NEXT dropped keyword automatically,
# not just this one restored by hand.
_SCHEMA_ACCEPTS_HANDLED_KEYWORDS = frozenset(
    {
        "anyOf",
        "type",
        "required",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "enum",
        "const",
        "minimum",
        "maximum",
        "additionalProperties",
    }
)
# Keywords the real emitted schemas carry that are structural (navigated
# to reach a nested schema, never themselves checked against a value) or
# pure annotation (documentation text no provider or this application
# enforces as a constraint) -- deliberately NOT "things schema_accepts
# happens not to implement yet." Adding a keyword here is a claim that it
# can never be the kind of gap this check exists to find.
_SCHEMA_ACCEPTS_ANNOTATION_KEYWORDS = frozenset(
    {
        "$ref",
        "$defs",
        "properties",
        "items",
        "title",
        "description",
        "format",
        # `default` never restricts which values a payload may send --
        # it is a hint for what an OMITTED field would resolve to, not a
        # constraint on what is present. Confirmed real, not guessed: this
        # test failed on exactly this keyword the first time it ran,
        # naming `"default"` (e.g. `supporting_evidence_ids`'s `[]`).
        "default",
    }
)
# Round 8 review, P3. The KEYWORD `"type"` being recognized (above) is a
# different claim from every VALUE it can take being one `schema_accepts`
# actually branches on. Today's real schemas only ever use `"null"`,
# `"object"`, `"array"`, `"string"`, `"integer"` -- all five handled below
# -- so there is no live gap. But a future field emitting `"type":
# "number"` (a float field, say) would fall through every `kind ==` branch
# in `schema_accepts` to its final `return True, ""` -- silently accepting
# any value -- while the KEYWORD-level coverage test above it would still
# pass, since `"type"` itself is allowlisted regardless of which value it
# holds. `_collect_schema_keywords` below collects both.
_SCHEMA_ACCEPTS_HANDLED_TYPES = frozenset(
    {"null", "object", "array", "string", "integer"}
)


def _collect_schema_keywords(node: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Walks one schema node the same way `schema_accepts` itself
    navigates it -- `properties`/`$defs` VALUES are nested schemas to
    recurse into, but their own keys are arbitrary field/type names picked
    by a domain model, never JSON Schema keywords, so they are deliberately
    NOT collected (unlike a naive "every dict key anywhere" walk, which
    would pollute the result with names like `"hypotheses"` or
    `"stop_reason"` and make this assertion fail for the wrong reason).

    Returns `(keywords, type_values)` from the same traversal: the
    KEYWORDS a schema uses, and separately, every VALUE its `"type"`
    keyword actually takes (`"string"`, `"integer"`, ...) -- see
    `_SCHEMA_ACCEPTS_HANDLED_TYPES`'s comment for why the two need
    checking separately."""
    keywords = set(node.keys())
    types: set[str] = set()
    kind = node.get("type")
    if isinstance(kind, str):
        types.add(kind)
    elif isinstance(kind, list):
        types.update(value for value in kind if isinstance(value, str))
    if "anyOf" in node:
        for sub in node["anyOf"]:
            sub_keywords, sub_types = _collect_schema_keywords(sub)
            keywords |= sub_keywords
            types |= sub_types
    if "properties" in node:
        for sub in node["properties"].values():
            sub_keywords, sub_types = _collect_schema_keywords(sub)
            keywords |= sub_keywords
            types |= sub_types
    if "items" in node:
        sub_keywords, sub_types = _collect_schema_keywords(node["items"])
        keywords |= sub_keywords
        types |= sub_types
    if "$defs" in node:
        for sub in node["$defs"].values():
            sub_keywords, sub_types = _collect_schema_keywords(sub)
            keywords |= sub_keywords
            types |= sub_types
    return keywords, types


def test_schema_accepts_implements_every_keyword_the_real_schemas_use() -> None:
    """The coverage assertion itself: walks all seven current emitted tool
    schemas (five domain tools, `record_stop`, `record_final_assessment`),
    collects every keyword actually present, and fails -- naming the
    specific keyword -- if one is neither handled by `schema_accepts` nor
    explicitly allowlisted as non-constraining. Mutation-verified: removing
    `"minimum"`/`"maximum"` from `_SCHEMA_ACCEPTS_HANDLED_KEYWORDS` alone
    (the real `schema_accepts` implementation untouched) fails this test,
    proving it is sensitive to exactly the class of drift that dropped all
    four keywords between the scratch script and this file.

    Round 8 review, P3. Also asserts on the second half of
    `_collect_schema_keywords`'s return: every VALUE the real schemas'
    `"type"` keyword takes must be one `schema_accepts` has a branch for,
    not just a keyword the mirror above recognizes by name."""
    tools = [
        *_domain_tool_definitions(),
        _stop_tool_definition(),
        _final_assessment_tool_definition(),
    ]
    real_keywords: set[str] = set()
    real_types: set[str] = set()
    for tool in tools:
        keywords, types = _collect_schema_keywords(tool["input_schema"])
        real_keywords |= keywords
        real_types |= types

    known = _SCHEMA_ACCEPTS_HANDLED_KEYWORDS | _SCHEMA_ACCEPTS_ANNOTATION_KEYWORDS
    unhandled = real_keywords - known
    assert not unhandled, (
        f"real emitted schemas use {sorted(unhandled)}, which schema_accepts "
        "neither handles nor allowlists as non-constraining -- implement it "
        "or allowlist it with a stated reason"
    )

    unhandled_types = real_types - _SCHEMA_ACCEPTS_HANDLED_TYPES
    assert not unhandled_types, (
        f"real emitted schemas use type value(s) {sorted(unhandled_types)}, "
        "which schema_accepts has no `kind ==` branch for -- it would "
        'silently fall through to `return True, ""` and accept any value'
    )


def test_the_type_coverage_check_catches_an_unhandled_type_value() -> None:
    """Direct proof for the type-VALUE half of the coverage assertion
    above, independent of whatever the real schemas happen to use today: a
    synthetic schema carrying `"type": "number"`, a value
    `_SCHEMA_ACCEPTS_HANDLED_TYPES` does not include, is picked up by
    `_collect_schema_keywords`'s second return value -- proving the
    mechanism would name a future unhandled type rather than let it hide
    behind `"type"` the KEYWORD already being recognized."""
    _keywords, types = _collect_schema_keywords({"type": "number"})
    assert types == {"number"}
    assert types - _SCHEMA_ACCEPTS_HANDLED_TYPES == {"number"}


# label -> where the rule this payload violates actually lives, since the
# schema itself cannot express it. Every payload exercised by the two tests
# below must have an entry here.
KNOWN_PROSE_ONLY_CONTRACTS: dict[str, str] = {
    "record_final_assessment: DIAGNOSED, no supporting_evidence_ids": (
        "domain.py's FinalAssessment.check_terminal_invariants -- "
        "'a diagnosis must cite supporting evidence' -- and "
        "FinalAssessment.supporting_evidence_ids's own description."
    ),
    "record_final_assessment: DIAGNOSED + root_cause UNDETERMINED": (
        "domain.py's FinalAssessment.check_terminal_invariants -- "
        "'a diagnosis needs a root cause other than UNDETERMINED' -- "
        "and FinalAssessment.disposition's own description."
    ),
    "record_final_assessment: INSUFFICIENT_EVIDENCE + a real root cause": (
        "domain.py's FinalAssessment.check_terminal_invariants -- "
        "'an abstention requires UNDETERMINED' -- and FinalAssessment."
        "root_cause's own description."
    ),
}


# Post-freeze review, N2 -- proves the open #27 finding for real.
# `KNOWN_PROSE_ONLY_CONTRACTS` above only ever pointed at where the
# documentation prose LIVES, in free text a human reads; nothing ever
# verified the prose actually EXISTS. Correctness proved the gap with a
# mutation: deleting `FinalAssessment.disposition`'s `Field(description=
# ...)` in `domain.py` left all 529 tests green, because the only test
# that could have noticed (`test_the_tool_payload_size_matches_what_
# pricingpy_assumes`) measures total payload characters, not any one
# field's content -- and that test measures `propose()`'s payload only
# (N1, above), which does not even contain `record_final_assessment`'s
# schema.
#
# Each entry below names the SAME contract from `KNOWN_PROSE_ONLY_
# CONTRACTS`, paired with the exact tool name, the exact property path in
# its emitted schema, and a literal substring of that property's CURRENT
# `description` -- checked directly against the real emitted schema by
# the test below, never inferred from the free-text pointer above (which
# stays free text for a human to read; this is its machine-checkable
# other half).
_WIRE_VISIBLE_PROSE_PROOF: dict[str, tuple[str, tuple[str, ...], str]] = {
    "record_final_assessment: DIAGNOSED, no supporting_evidence_ids": (
        RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        ("supporting_evidence_ids",),
        "A DIAGNOSED assessment must cite at least one entry here.",
    ),
    "record_final_assessment: DIAGNOSED + root_cause UNDETERMINED": (
        RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        ("disposition",),
        "DIAGNOSED requires a root_cause other than UNDETERMINED",
    ),
    "record_final_assessment: INSUFFICIENT_EVIDENCE + a real root cause": (
        RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        ("root_cause",),
        "UNDETERMINED when disposition is INSUFFICIENT_EVIDENCE",
    ),
}


# Round 4 review, F2. The subset check below used to prove only that every
# PROOF entry is registered -- it never proved the reverse, that every
# REGISTERED contract has a proof. A fifth prose-only contract added to
# `KNOWN_PROSE_ONLY_CONTRACTS` without a matching `_WIRE_VISIBLE_PROSE_
# PROOF` entry would pass silently, with no wire-proof requirement on it
# at all -- exactly the shape of gap N2 closed for the first four.
#
# A registered contract's prose does not always live in a single JSON
# schema field's `description`, though -- some rules are raised as a
# Python error string (e.g. a `ValueError` message in `graph.py` or
# `live_model.py`) rather than shipped to the provider as schema text, and
# `_WIRE_VISIBLE_PROSE_PROOF`'s tuple shape (tool name, field path,
# description substring) has no way to express that. Naming such a
# contract here, with a stated reason, is how it opts out of the wire-
# proof requirement without silently defeating this test for a future
# contract that legitimately needs to.
#
# Round 6 review. This registry was a bare `frozenset[str]`, its two
# siblings above (`KNOWN_PROSE_ONLY_CONTRACTS`, `_WIRE_VISIBLE_PROSE_
# PROOF`) both force a reason/pointer alongside every entry; a frozenset
# has no room for one, so a future exemption added here would carry no
# stated reason at all -- exactly the discipline this comment already
# describes but the type could not enforce. `dict[str, str]` (label ->
# reason) matches its siblings' shape; empty today, so this is a type
# change with no data migration.
_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF: dict[str, str] = {}


def test_wire_visible_prose_proof_only_names_registered_contracts() -> None:
    """Guards `_WIRE_VISIBLE_PROSE_PROOF` itself against drifting out of
    sync with the registry it is meant to verify, in both directions --
    every label it names must actually be in `KNOWN_PROSE_ONLY_CONTRACTS`
    (a stray/renamed proof entry), and every label in
    `KNOWN_PROSE_ONLY_CONTRACTS` must appear either as a proof or as a
    documented, reasoned exemption in
    `_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF` (an unproven contract)."""
    proven = set(_WIRE_VISIBLE_PROSE_PROOF)
    exempted = set(_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF)
    registered = set(KNOWN_PROSE_ONLY_CONTRACTS)
    assert not (proven & exempted), (
        f"{proven & exempted} is both proven and exempted -- pick one"
    )
    assert proven | exempted == registered, (
        f"missing wire proof or a documented exemption for "
        f"{registered - proven - exempted}; stray proof/exemption entries "
        f"not in KNOWN_PROSE_ONLY_CONTRACTS: {(proven | exempted) - registered}"
    )


def test_every_wire_proof_exemption_carries_a_real_reason() -> None:
    """Round 7 review. `_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF` became a
    `dict[str, str]` in round 6 so an exemption could not be added without
    a stated reason, matching its two sibling registries -- but the test
    above only ever reads `set(_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF)`,
    the dict's KEYS, and never looks at the values it exists to force. A
    future entry like `{"some_new_contract": ""}` would pass every
    existing test in this file. The registry is empty today, so this
    needs no data migration -- it only closes the enforcement gap for
    whenever an entry is first added."""
    for label, reason in _PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF.items():
        assert reason.strip(), (
            f"{label!r} is exempted from the wire-proof requirement with "
            "no stated reason -- every exemption in this registry must "
            "say why, matching KNOWN_PROSE_ONLY_CONTRACTS and "
            "_WIRE_VISIBLE_PROSE_PROOF"
        )


def test_the_registrys_pointed_at_descriptions_are_actually_present() -> None:
    """The N2 fix itself. For each known prose-only contract that claims a
    specific field's own description carries the rule, this looks up the
    REAL emitted schema for that tool, navigates to the named field, and
    asserts the expected substring is actually present in its
    `description` -- not merely that a `description` key exists (empty or
    unrelated text would pass a weaker check, and would not have caught
    correctness's demonstration). Mutation-verified separately: deleting
    `FinalAssessment.disposition`'s `description=...` in `domain.py` (the
    exact gap correctness demonstrated survives today) makes this test
    fail, naming the missing substring."""
    tools = {
        tool["name"]: tool["input_schema"]
        for tool in (
            *_domain_tool_definitions(),
            _stop_tool_definition(),
            _final_assessment_tool_definition(),
        )
    }
    for label, (
        tool_name,
        field_path,
        expected_substring,
    ) in _WIRE_VISIBLE_PROSE_PROOF.items():
        node = tools[tool_name]
        for field_name in field_path:
            node = node["properties"][field_name]
        description = node.get("description", "")
        assert expected_substring in description, (
            f"{label!r}: expected {'.'.join(field_path)!r} in {tool_name!r}'s "
            f"schema to carry {expected_substring!r} in its description, "
            f"found {description!r}"
        )


def assert_documented_prose_only_contract(
    label: str, schema: dict[str, Any], payload: dict[str, Any], *, app_refuses: bool
) -> None:
    """The cross-check itself. A payload the schema accepts and the
    application refuses is exactly the shape of gap this unit's
    investigation was launched to find; requiring `label` to already be in
    `KNOWN_PROSE_ONLY_CONTRACTS` is what makes a new, undocumented one fail
    this assertion instead of shipping quietly."""
    accepted, why = schema_accepts(schema, payload)
    assert accepted, f"{label}: expected the schema to accept this payload ({why})"
    if app_refuses:
        assert label in KNOWN_PROSE_ONLY_CONTRACTS, (
            f"{label!r} is accepted by the emitted schema and refused by "
            "the application -- a prose-only contract -- but is not named "
            "in KNOWN_PROSE_ONLY_CONTRACTS with a pointer to the prose "
            "that carries it"
        )


_HYPOTHESES_PAYLOAD = [
    {"root_cause": "CONFIG_CHANGE", "rank": 1, "missing_evidence": "a"},
    {"root_cause": "RESOURCE_POOL_SATURATION", "rank": 2, "missing_evidence": "b"},
]


def test_the_stop_reason_omission_is_refused_at_the_schema_level() -> None:
    """The single-call stop schema requires an explicit stop reason."""
    plan_schema = _stop_tool_definition()["input_schema"]
    payload = {"hypotheses": _HYPOTHESES_PAYLOAD}

    accepted, why = schema_accepts(plan_schema, payload)

    assert accepted is False
    assert why == "missing required 'stop_reason'"


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        (
            "record_final_assessment: DIAGNOSED, no supporting_evidence_ids",
            {
                "disposition": "DIAGNOSED",
                "root_cause": "CONFIG_CHANGE",
                "uncertainty": "u",
                "next_step": "n",
            },
        ),
        (
            "record_final_assessment: DIAGNOSED + root_cause UNDETERMINED",
            {
                "disposition": "DIAGNOSED",
                "root_cause": "UNDETERMINED",
                "supporting_evidence_ids": ["e1"],
                "uncertainty": "u",
                "next_step": "n",
            },
        ),
        (
            "record_final_assessment: INSUFFICIENT_EVIDENCE + a real root cause",
            {
                "disposition": "INSUFFICIENT_EVIDENCE",
                "root_cause": "CONFIG_CHANGE",
                "uncertainty": "u",
                "next_step": "n",
            },
        ),
    ],
)
def test_each_known_final_assessment_contract_is_schema_accepted_and_app_refused(
    label: str, payload: dict[str, Any]
) -> None:
    fa_schema = _final_assessment_tool_definition()["input_schema"]

    try:
        FinalAssessment.model_validate(payload)
        app_refuses = False
    except ValidationError:
        app_refuses = True

    assert_documented_prose_only_contract(
        label, fa_schema, payload, app_refuses=app_refuses
    )
    assert app_refuses, f"{label}: FinalAssessment no longer refuses this payload"


def test_an_undocumented_prose_only_contract_fails_the_check() -> None:
    """Demonstrates item 5's mechanism has teeth, using a real, not
    contrived, gap this unit did not add to `KNOWN_PROSE_ONLY_CONTRACTS`.

    `contrary_evidence_ids` was this test's original example -- it fed the
    identical forged-citation check `supporting_evidence_ids` does, but was
    left undocumented by item 4's first pass. The owner ruled that gap
    fixed (see `FinalAssessment.contrary_evidence_ids`'s own description in
    `domain.py`), which correctly makes it disappear from this test too --
    keeping it here after fixing it would demonstrate nothing.

    `search_runbooks.limit` is its replacement, a *different* gap in the
    same "schema allows more than the application actually admits" family,
    explicitly recorded as open and NOT in this unit's approved scope
    (`3b4-approved-scope.md`'s P3-2): the wire schema admits `limit` up to
    20 (`SearchRunbooksArguments.limit`, `le=20`), but `policy.authorize`
    denies anything above `Budgets.runbook_passages` (5) with
    `ReasonCode.RESULT_LIMIT_EXCEEDED` -- a real policy denial, invoked
    here directly rather than simulated, that this unit does not fix
    because P3-2 was explicitly ruled out of scope."""
    domain_tools = _domain_tool_definitions()
    (search_runbooks_tool,) = [
        tool for tool in domain_tools if tool["name"] == ToolName.SEARCH_RUNBOOKS.value
    ]
    schema = search_runbooks_tool["input_schema"]
    payload = {
        "topic": "gateway_errors",
        "limit": 6,
        "hypotheses": _HYPOTHESES_PAYLOAD,
        "evidence_gap": "whether known gateway-error causes match this incident",
        "expected_observation": "guidance naming a matching known cause",
    }
    label = "search_runbooks: limit above Budgets.runbook_passages"
    assert label not in KNOWN_PROSE_ONLY_CONTRACTS

    arguments = SearchRunbooksArguments.model_validate(
        {
            **{
                key: value
                for key, value in payload.items()
                if key not in {"evidence_gap", "expected_observation", "hypotheses"}
            },
            "tool": ToolName.SEARCH_RUNBOOKS.value,
        }
    )
    proposal = ToolProposal(
        arguments=arguments,
        evidence_gap=payload["evidence_gap"],
        expected_observation=payload["expected_observation"],
    )
    decision = authorize(
        proposal,
        incident_scope(),
        seen_fingerprints=set(),
        budgets=Budgets(),
        tools_remaining=4,
    )
    assert decision.result is PolicyResult.DENIED
    assert decision.reason_code is ReasonCode.RESULT_LIMIT_EXCEEDED

    with pytest.raises(AssertionError, match="not named in KNOWN_PROSE_ONLY_CONTRACTS"):
        assert_documented_prose_only_contract(label, schema, payload, app_refuses=True)
