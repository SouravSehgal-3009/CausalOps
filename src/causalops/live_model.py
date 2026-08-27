"""`LiveClaudeModel`, the live Claude adapter.

Implements the exact `propose`/`respond` shape `models.py`'s `ToolCallingModel`
protocol names -- the same shape `ReplayToolCallingModel` implements --
so `graph.py` cannot tell which one it is bound to. Built once per
investigation by `live_setup.build_model_and_registry`, the same way
`ReplayToolCallingModel` is today.

Tool schemas for the five registered checks are derived from `tools.py`'s
`ToolArguments` union (`tools.py:135`'s own comment: "a second lookup table
would be a competing source of truth") -- never hand-written a second time
here. Every domain tool carries the required ranked hypotheses as well as its
check rationale. The provider's native tool name is the wire discriminator;
`tool_calls.py` restores the internal `ToolArguments.tool` field after
checking that name is registered.

Single native-call proposal protocol
------------------------------------
`InitialPlan`/`HypothesisUpdate` need 2-3 ranked hypotheses and exactly one
of a check proposal or a stop reason. The live adapter receives both fields
through one native tool call, never a second structured-content channel.

This adapter uses six tools but requires *exactly one* native call per turn.
The five real checks include hypotheses in their own arguments. The sixth,
adapter-internal `record_stop`, carries those hypotheses plus a required,
non-empty reason for ending investigation. It is deliberately not part of
`ToolName`/`ToolArguments`, and is never passed to a policy wrapper or tool
backend. `parallel_tool_calls=False` asks the provider for the same one-call
shape; cardinality, unknown names, malformed calls, and visible content are
still rejected locally and use the graph's normal repair path.
"""

import json
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolCall
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from causalops.cost_ledger import (
    AmbiguousReservationNotResent,
    ensure_cost_ledger_table,
    record_reservation_before_request,
    settle_reservation,
)
from causalops.domain import (
    SCHEMA_VERSION,
    Budgets,
    FinalAssessment,
    GatewaySymptom,
    Hypothesis,
    IncidentScope,
    InitialAlertPacket,
    ModelUsage,
    ToolProposal,
    utc_now,
)
from causalops.evidence import digest_text
from causalops.models import (
    ModelRequest,
    ModelResponse,
    ProposedTurn,
    Stage,
    parse_response,
)
from causalops.pricing import (
    CLAUDE_SONNET_5_PRICING,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_REQUEST_SECONDS,
    InputTooLarge,
    PricingSnapshot,
    estimate_input_tokens,
)
from causalops.prompts import STAGE_INSTRUCTIONS, SYSTEM_TEXT, render_context
from causalops.tool_calls import NativeToolCall, parse_tool_call, summarize_errors
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    QueryLogsArguments,
    QueryMetricArguments,
    SearchRunbooksArguments,
    ToolName,
)

# `TECHNICAL_OVERVIEW.md`'s "Default limits" table: "Claude ... claude-sonnet-5
# ... specified for the live adapter." Not a free choice this module makes.
MODEL_NAME = "claude-sonnet-5"

# Neither collides with any `ToolName` value (`tools.py`) -- Claude echoes
# back exactly the tool name it was given, so this module's own
# classification of which channel a call belongs to is a simple name
# comparison, not a heuristic.
RECORD_STOP_TOOL_NAME = "record_stop"
RECORD_FINAL_ASSESSMENT_TOOL_NAME = "record_final_assessment"


# Reuse `domain.Hypothesis` rather than redeclaring its fields. These models
# are only tool-input records: domain tools use `HypothesesRecord`; the
# adapter-internal stop tool adds its required reason in `StopRecord`.
# Neither exposes application `schema_version` to the provider. They have no
# class docstrings because Pydantic would include that maintainer prose in the
# billed provider schema; the tool definitions supply concise model guidance.
class HypothesesRecord(BaseModel):
    # `extra="forbid"` rationale: see `tools.py`'s `QueryMetricArguments`.
    model_config = ConfigDict(frozen=True, extra="forbid")

    hypotheses: tuple[Hypothesis, ...] = Field(min_length=2, max_length=3)


class StopRecord(HypothesesRecord):
    # A stop turn is the alternative to a domain-tool proposal, so this field
    # is required and non-empty rather than a conditional field shared with
    # check calls.
    stop_reason: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Why you are stopping instead of proposing a check, in 300 "
            "characters or fewer. This must be a non-empty reason."
        ),
    )


# (typed-args class, registered tool name, one-line description for Claude)
# -- the description is the only thing this module writes by hand for each
# tool; the argument shape itself always comes from `model_json_schema()`.
_DOMAIN_TOOL_SPECS: tuple[tuple[type[BaseModel], ToolName, str], ...] = (
    (
        QueryMetricArguments,
        ToolName.QUERY_METRIC,
        "Query one Prometheus metric template for one service over a time window.",
    ),
    (
        QueryLogsArguments,
        ToolName.QUERY_LOGS,
        "Query recent application logs for one service, filtered to one category.",
    ),
    (
        ListRecentChangesArguments,
        ToolName.LIST_RECENT_CHANGES,
        "List recent deploys/config changes for one service over a time window.",
    ),
    (
        GetTopologyArguments,
        ToolName.GET_TOPOLOGY,
        "Get the incident's service dependency topology.",
    ),
    (
        SearchRunbooksArguments,
        ToolName.SEARCH_RUNBOOKS,
        "Search the offline runbook corpus for guidance on one closed topic.",
    ),
)

# Both fields already carried `maxLength: 300`
# (provider-unenforced, per the root-cause investigation) without their
# description ever stating the bound in words -- the same gap other
# `domain.py` fields already close, missed here because these two are
# synthetic properties this module injects rather than a `domain.py`
# field. More exposed than any of those: both are
# REQUIRED on every domain-tool call, not once per run.
_RATIONALE_PROPERTIES: dict[str, JsonValue] = {
    "evidence_gap": {
        "type": "string",
        "maxLength": 300,
        "description": (
            "What this check is meant to settle -- the open question, in "
            "300 characters or fewer."
        ),
    },
    "expected_observation": {
        "type": "string",
        "maxLength": 300,
        "description": (
            "What a result confirming your leading hypothesis would show, "
            "in 300 characters or fewer."
        ),
    },
}


def _strip_maintainer_prose(
    schema: dict[str, Any], *, keep_defs: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Remove billed class-level prose while preserving chosen field guidance."""
    schema = dict(schema)
    schema.pop("description", None)
    defs = schema.get("$defs")
    if defs:
        stripped_defs: dict[str, Any] = {}
        for name, definition in defs.items():
            definition = dict(definition)
            if name not in keep_defs:
                definition.pop("description", None)
            stripped_defs[name] = definition
        schema["$defs"] = stripped_defs
    return schema


def _domain_tool_definitions() -> list[dict[str, Any]]:
    """Anthropic-format tool definitions for the five registered checks,
    derived from `ToolArguments` so code and wire schemas cannot drift.

    The provider tool name selects the internal argument variant, so the
    duplicate internal `tool` discriminator is deliberately not exposed.

    This is the mechanical reason the emitted schema payload is 13,404
    tokens (`_send`'s own comment pins the exact figure): Anthropic tool
    schemas are self-contained, with no cross-tool `$ref`, so each of the
    five loop iterations below embeds its own full copy of
    `HypothesesRecord`'s schema (`hypotheses_schema` and its `$defs`) rather
    than sharing one. The five-fold duplication of that one shared schema
    is essentially the whole size increase from the two-call protocol this
    project used before the single-call rewrite -- not five
    independently large tools.
    """
    definitions: list[dict[str, Any]] = []
    for arguments_cls, tool_name, description in _DOMAIN_TOOL_SPECS:
        schema = _strip_maintainer_prose(arguments_cls.model_json_schema())
        properties = dict(schema.get("properties", {}))
        properties.pop("tool")
        hypotheses_schema = HypothesesRecord.model_json_schema()
        properties["hypotheses"] = hypotheses_schema["properties"]["hypotheses"]
        properties.update(_RATIONALE_PROPERTIES)
        schema["properties"] = properties
        schema["$defs"] = {
            **schema.get("$defs", {}),
            **hypotheses_schema.get("$defs", {}),
        }
        required = list(schema.get("required", []))
        if "tool" in required:
            required.remove("tool")
        for name in ("hypotheses", *_RATIONALE_PROPERTIES):
            if name not in required:
                required.append(name)
        schema["required"] = required
        definitions.append(
            {
                "name": tool_name.value,
                "description": description,
                "input_schema": schema,
            }
        )
    return definitions


def _stop_tool_definition() -> dict[str, Any]:
    # Keeps `Hypothesis.__doc__` ("Rank is not a
    # probability") -- genuine guidance about how to fill the field in --
    # while stripping any other class-level docstring `StopRecord`'s own
    # `$defs` might carry (`RootCauseCode` has none today; this keeps the
    # function correct if that ever changes).
    schema = _strip_maintainer_prose(
        StopRecord.model_json_schema(), keep_defs=frozenset({"Hypothesis"})
    )
    return {
        "name": RECORD_STOP_TOOL_NAME,
        "description": (
            "Use this instead of a check tool when no safe check would help. "
            "Record 2-3 ranked hypotheses and a non-empty stop reason. "
            "Call exactly one tool total on each turn."
        ),
        "input_schema": schema,
    }


def _final_assessment_schema() -> dict[str, Any]:
    """`domain.py`'s `FinalAssessment` -- unlike this module's own `StopRecord`
    -- is a shared domain model, not written just for this tool definition, so
    its `model_json_schema()` carries two things this tool's `input_schema`
    must not ship to Claude as-is:

    - `schema_version`: application bookkeeping (`domain.SCHEMA_VERSION`)
      the model has no business setting. Every domain tool
      (`_domain_tool_definitions`) and `StopRecord` already omit it; this
      is `FinalAssessment`'s own equivalent.
    - `description` (this schema's own, and every nested `$defs` entry's,
      per `_strip_maintainer_prose`): pydantic promotes `FinalAssessment
      .__doc__` ("Its schema cannot express FAILED_SAFE") to the schema's
      own top-level key, and `ModelDisposition.__doc__` ("FAILED_SAFE is
      absent on purpose") the same way into `$defs`. Both name an
      application-side type boundary for a maintainer reading `domain.py`,
      not guidance about how to fill the tool in -- the hand-written
      `"description"` string in `_final_assessment_tool_definition` below,
      and the field-level `description`s already added directly to
      `FinalAssessment`'s own fields, already say what Claude needs to
      know.
    """
    schema = _strip_maintainer_prose(FinalAssessment.model_json_schema())
    properties = {
        name: value
        for name, value in schema.get("properties", {}).items()
        if name != "schema_version"
    }
    schema["properties"] = properties
    schema["required"] = [
        name for name in schema.get("required", []) if name != "schema_version"
    ]
    return schema


def _final_assessment_tool_definition() -> dict[str, Any]:
    return {
        "name": RECORD_FINAL_ASSESSMENT_TOOL_NAME,
        # States the same three terminal-disposition
        # invariants `domain.py`'s `check_terminal_invariants` enforces,
        # restated per-field on `disposition`/`root_cause`/
        # `supporting_evidence_ids` themselves (see those fields'
        # `description`s in `domain.py`) -- named here too so the rule is
        # visible from the tool's own one-line summary, not only once a
        # model is already reading an individual field.
        "description": (
            "Call this once to record your final diagnosis or abstention. "
            "A DIAGNOSED assessment needs a root_cause other than "
            "UNDETERMINED and at least one supporting_evidence_ids entry; "
            "an abstention (INSUFFICIENT_EVIDENCE) needs root_cause "
            "UNDETERMINED."
        ),
        "input_schema": _final_assessment_schema(),
    }


class MissingCredential(Exception):
    """No `ANTHROPIC_API_KEY` was present when this run started.

    Raised in `_send`, before *any* reservation, on every
    turn a credential is missing -- not just the first. `ChatAnthropic()`
    does not raise at construction even with no key (confirmed against the
    installed SDK); the actual failure was a `TypeError` deep inside
    `.invoke()`, which `_send`'s estimate -> reserve -> invoke ordering
    reached *after* writing a reservation. A broken-key run could write up
    to `Budgets.model_calls` permanent `RESERVED` cost_ledger rows that
    never settle, each one still counting against the application-wide
    ceiling forever -- this check refuses before that write instead.

    This module never reads `ANTHROPIC_API_KEY` itself
    (`tests/security/test_credential_isolation.py` proves it never imports
    `os` or names the variable in code) -- `cli.py`'s `_build_model_and_
    registry` checks presence the same way `doctor.check_api_key` does and
    passes the *result*, a plain `bool`, into `LiveClaudeModel`'s
    constructor. `credential_present` defaults `True` so every existing
    test that injects a fake client (this module's own test seam) is
    unaffected unless it opts in.
    """


class MissingProviderUsage(Exception):
    """The provider returned a response with no usage metadata at all.

    `TECHNICAL_SPEC.md` §5's durable-operation rules name this explicitly:
    "a response that omits provider usage" is one of the conditions the
    amended `PENDING` record exists to survive, alongside a timeout or a
    mid-request crash -- "the reservation left visible for accounting," not
    silently accepted as free. Raised rather than settling with a
    substituted value, so the reservation stays `RESERVED` (never `SETTLED`
    on a guess) and this surfaces as `FAILED_SAFE`/`INTERNAL_ERROR` through
    `graph.py`'s existing blanket handler -- the same path a crash or
    timeout already takes, not a new failure category.
    """


def _to_native_tool_call(call: ToolCall) -> NativeToolCall:
    """`langchain_core.messages.tool.ToolCall` (a `TypedDict`: `name`,
    `args`, `id`, `type`) matches `NativeToolCall` field-for-field -- the
    same claim `tool_calls.py`'s own docstring makes about the provider
    shape this module was modelled on. `id` is nullable in LangChain's type
    but never actually `None` for a real Anthropic `tool_use` block; the
    fallback exists so a malformed or synthetic message fails a downstream
    validation loudly rather than raising `TypeError` constructing this
    object."""
    call_id = call.get("id") or "missing-call-id"
    return NativeToolCall(name=call["name"], args=call["args"], id=call_id)


def _has_visible_content(content: object) -> bool:
    """Native tool-call turns have no second, unstructured answer channel.

    A block type outside `{"tool_use", "thinking", "redacted_thinking"}`
    (ordinary visible text, or a block type this module has never seen) is
    refused rather than ignored: narrative text sitting alongside a tool
    call is ambiguous about whether the model actually committed to that
    call or was still reasoning out loud in a channel the application does
    not read, and this project needs one unambiguous decision per turn, not
    a mix it would have to guess how to resolve. An earlier version of this
    function rejected every list-typed response outright, which would also
    have refused a genuine turn carrying only `tool_use`/`thinking` blocks
    -- the ordinary shape once extended thinking is on
    (`_build_chat_anthropic` sets `thinking={"type": "adaptive"}`
    unconditionally) -- burning the run's one repair slot on a wholly valid
    turn; fixed by allow-listing the three
    real provider block types explicitly instead of rejecting every list.
    """
    if content in ("", []):
        return False
    if not isinstance(content, list):
        return True
    return any(
        not isinstance(block, dict)
        or block.get("type") not in {"tool_use", "thinking", "redacted_thinking"}
        for block in content
    )


def _build_chat_anthropic(pricing: PricingSnapshot) -> ChatAnthropic:
    """The real transport `LiveClaudeModel` sends through, module-level so
    `tests/unit/test_live_model.py` can assert on its return value directly
    -- `default_request_timeout`/`max_tokens`/`max_retries`/`model` are all
    public pydantic fields (confirmed against the installed
    `langchain-anthropic` package's own `model_fields`, not assumed), so a
    test reaching them is reading this SDK's public contract, not a private
    attribute of this project's own. Constructing a `ChatAnthropic` performs
    no I/O -- the client only connects on its first real request -- so this
    is safe to call under `tests/conftest.py`'s network guard.

    `ChatAnthropic()` resolves `ANTHROPIC_API_KEY` from the environment
    itself when no `anthropic_api_key=` is passed -- this module never
    reads, stores, formats, or logs the key's value, the
    "environment-only credential" rule `TECHNICAL_OVERVIEW.md`'s threat
    table names (`doctor.py`'s `check_api_key` already gates its presence
    before a live run starts; `cli.py`'s `credential_present` check gates
    it again, cheaply, before this function is ever called for a real run
    -- see `MissingCredential`'s docstring). `max_retries=0`:
    `TECHNICAL_OVERVIEW.md`'s "Default limits" table specifies no
    automatic retries anywhere in this project, and the cost gate in
    `_send` already made one deliberate send decision for this
    reservation -- a silent SDK-level retry would send a second request
    under a reservation sized for one, the same failure mode the reservation
    math itself exists to close.
    `thinking`/`reasoning_effort`: Claude Sonnet 5 runs adaptive thinking
    regardless (omitting `thinking` already runs adaptive per the
    installed SDK's own model-support table), set explicitly here so the
    request is self-documenting rather than relying on a default an owner
    reading this code cannot see. No `temperature`/`top_p`/`top_k`:
    `TECHNICAL_OVERVIEW.md` specifies none, and Sonnet 5 rejects all three
    outright once `thinking` is on.

    Keyword-only aliases (`model_name`/`max_tokens_to_sample`/`effort`),
    not the plain field names (`model`/`max_tokens`/`reasoning_effort`) a
    reader would expect from `ChatAnthropic.model_fields`: pydantic's
    `dataclass_transform`-synthesized `__init__` mypy checks against uses
    each field's `alias` when one is declared, and all three have one here
    (confirmed against the installed `langchain-anthropic` package, not
    assumed) -- `model_config`'s `populate_by_name=True` means the plain
    names work at runtime too, but only the aliases satisfy `mypy src lab`.

    `timeout=MAX_REQUEST_SECONDS` closes
    `TECHNICAL_OVERVIEW.md`'s "Default limits" table's own "Model call |
    90 seconds | specified, not enforced" row -- a request that hangs past
    this many seconds raises rather than tying up the process indefinitely,
    the same "bounded, not unbounded" reasoning every other limit in that
    table already gets. `stop=None` is the field's own real default
    (`stop_sequences`, `None`) -- passed explicitly only because mypy's
    pydantic-field-driven constructor synthesis reports it as required
    despite that default (confirmed at a Python prompt against the
    installed package: `is_required() == False`). Behaviourally inert;
    here to satisfy `mypy src lab`, not to change the default.
    """
    return ChatAnthropic(
        model_name=pricing.model_name,
        max_tokens_to_sample=MAX_OUTPUT_TOKENS,
        max_retries=0,
        thinking={"type": "adaptive"},
        effort="medium",
        timeout=MAX_REQUEST_SECONDS,
        stop=None,
    )


class LiveClaudeModel:
    """The live Claude adapter. Constructed once per investigation by
    `live_setup.build_model_and_registry`, exactly like
    `ReplayToolCallingModel` is today -- one instance answers every stage of
    one run.
    """

    def __init__(
        self,
        checkpoints_conn: sqlite3.Connection,
        *,
        ceiling_usd: float,
        pricing: PricingSnapshot = CLAUDE_SONNET_5_PRICING,
        clock: Callable[[], datetime] = utc_now,
        client: ChatAnthropic | None = None,
        credential_present: bool = True,
    ) -> None:
        self._conn = checkpoints_conn
        self._pricing = pricing
        self._ceiling_usd = ceiling_usd
        self._clock = clock
        # A plain `bool`, checked by `cli.py` before this
        # object is even constructed -- see `MissingCredential`'s docstring
        # for why the check lives there and not here. Defaults `True` so a
        # test-seam construction (`client=` below) that never passes this
        # keyword keeps behaving as if the credential were always present.
        self._credential_present = credential_present
        # Test seam: `tests/unit/test_live_model.py` passes a fake here so
        # this module's logic (schema derivation, single-call validation,
        # and the cost gate) is exercised without constructing
        # a real `ChatAnthropic` -- no test ever reaches `tests/conftest.
        # py`'s network guard through this path, because nothing beneath
        # it ever tries to connect. `_build_chat_anthropic` above is what a
        # test asserts against instead, directly, when it wants to pin
        # what a *real* client would have been constructed with.
        self._client = client if client is not None else _build_chat_anthropic(pricing)

    def _send(self, request: ModelRequest, tools: list[dict[str, Any]]) -> AIMessage:
        """Reserve, send, settle -- the one call-site every `propose`/
        `respond` path below goes through, so the gate cannot be bypassed by
        a future call site that forgets it."""
        if not self._credential_present:
            # Checked first, before the input-cap check and
            # before any reservation -- the cheapest possible refusal, on
            # every turn a credential is missing, not just the first. See
            # `MissingCredential`'s docstring for the money bug this closes.
            raise MissingCredential(
                f"{request.stage.value} turn refused: no credential was "
                "present when this run started"
            )
        # `content` is composed
        # here, before either the cap check or the reservation below, so
        # both price the *actual* rendered human message -- correction
        # header included when `repair_errors` is set -- rather than a
        # proxy that reconstructs `content` a second time, later, and can
        # silently drift from what estimation saw. An earlier version's
        # estimate summed `system_text + context_text + repair_errors`
        # directly, undercounting the 23 literal characters of
        # `"\n\n## Correction needed\n"` that `content` gets on a repair
        # turn -- billed by the provider, never estimated.
        content = request.context_text
        if request.repair_errors:
            content = f"{content}\n\n## Correction needed\n{request.repair_errors}"
        estimated_input_tokens = estimate_input_tokens(request.system_text + content)
        if estimated_input_tokens > MAX_INPUT_TOKENS:
            raise InputTooLarge(estimated_input_tokens)
        # The reservation must price what actually goes out
        # on the wire: prose *plus* the tool schema `bind_tools` sends on
        # every call -- this `tools` list differs by caller, so the fixed
        # payload size differs by STAGE, not one shared figure: `propose()`
        # binds `_stop_tool_definition()` plus the five `_domain_tool_
        # definitions()`, 13,404 tokens in the current emitted schema
        # (a `MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_
        # CAPACITY` rename -- from `RESOURCE_POOL_UTILIZATION`, a name that
        # falsely implied boundedness -- added 10 bytes to `query_metric`'s
        # embedded schema over the prior 12,829; a later change then added a
        # `Field(description=...)` to `QueryMetricArguments.service` and
        # `QueryLogsArguments.service`, growing the figure again, 12,839 ->
        # 13,404, the third growth of this figure to date);
        # `respond()` binds only `_final_assessment_tool_definition()`,
        # 2,292 tokens in the current emitted schema.
        # `test_live_model.py` pins both figures separately, so this
        # comment cannot drift from either real payload the way an earlier
        # version of the `propose()` figure already did, three times, and
        # the `respond()` figure did once by simply never being measured.
        # `MAX_INPUT_TOKENS`'s cap above deliberately stays prose-only --
        # see `InputTooLarge`'s docstring for why folding tools into the
        # cap is the wrong fix -- but a dollar reservation that omits real,
        # billed tokens is not conservative: counting the tool schema here
        # is what keeps `actual_usd <= reserved_usd` the ORDINARY case
        # (`TECHNICAL_OVERVIEW.md`'s own corrected wording -- not a
        # code-enforced invariant; `cost_ledger.settle_reservation` still
        # commits and logs a real overrun rather than refusing it, since the
        # money is already spent by the time settlement runs).
        tool_definition_tokens = estimate_input_tokens(json.dumps(tools))
        reserved_usd = self._pricing.reservation_usd(
            estimated_input_tokens + tool_definition_tokens
        )
        requested_at = self._clock()
        # Raises `CostCeilingExceeded` and writes nothing further if this
        # reservation would exceed the remaining application-wide balance --
        # refuse rather than send, `TECHNICAL_SPEC.md` §10's own words.
        reservation, is_new = record_reservation_before_request(
            self._conn,
            run_id=request.run_id,
            graph_phase=request.graph_phase,
            model_turn=request.model_turn,
            context_digest=request.context_digest,
            reserved_usd=reserved_usd,
            requested_at=requested_at,
            ceiling_usd=self._ceiling_usd,
        )
        # `is_new is False` means a reservation
        # for this exact key already existed BEFORE this call -- a crash
        # between an earlier reserve and its settle, then a resume that
        # re-renders this identical stage, would land here. Checked before
        # the provider is ever invoked, not after: see
        # `AmbiguousReservationNotResent`'s docstring for why both possible
        # states of the pre-existing row (`RESERVED` or `SETTLED`) are
        # refused the same way rather than one being treated as safe to
        # resend.
        if not is_new:
            raise AmbiguousReservationNotResent(reservation)
        messages = [
            SystemMessage(content=request.system_text),
            HumanMessage(content=content),
        ]
        # If this raises (timeout, connection error, provider 5xx --
        # anything), the reservation above stays `RESERVED` and this
        # function never reaches `settle_reservation` below: the caller's
        # exception propagates to `graph.py`'s existing blanket
        # `except Exception`, which reports `FAILED_SAFE`/`INTERNAL_ERROR`
        # with the reservation intact for accounting, exactly
        # `TECHNICAL_SPEC.md` §5's "a timeout, crash, or missing provider
        # usage never reissues that key" rule.
        message = self._client.bind_tools(tools, parallel_tool_calls=False).invoke(
            messages
        )
        usage = message.usage_metadata
        if usage is None:
            raise MissingProviderUsage(
                f"{request.stage.value} turn returned no usage metadata"
            )
        settled_at = self._clock()
        settle_reservation(
            self._conn,
            run_id=request.run_id,
            graph_phase=request.graph_phase,
            model_turn=request.model_turn,
            context_digest=request.context_digest,
            actual_usd=self._pricing.actual_cost_usd(
                usage["input_tokens"], usage["output_tokens"]
            ),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            settled_at=settled_at,
        )
        return message

    def _usage(self, message: AIMessage) -> ModelUsage:
        # `_send` already raised `MissingProviderUsage` if this were absent,
        # so by the time any caller in this module reaches here it is real.
        usage = message.usage_metadata
        assert usage is not None, "settled message with no usage_metadata"
        return ModelUsage(
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]
        )

    def respond(self, request: ModelRequest) -> ModelResponse:
        """FINAL_ASSESSMENT has no proposal to encode -- one forced-shape
        tool (`record_final_assessment`), no domain tools offered at all."""
        message = self._send(request, [_final_assessment_tool_definition()])
        usage = self._usage(message)
        if _has_visible_content(message.content):
            return ModelResponse(content={}, usage=usage)
        matching_calls = [
            call
            for call in message.tool_calls
            if call["name"] == RECORD_FINAL_ASSESSMENT_TOOL_NAME
        ]
        # No error channel on `ModelResponse` -- an empty `content` dict
        # fails `FinalAssessment`'s own required-field validation for a
        # genuine, informative reason (`disposition`/`root_cause` missing)
        # rather than this module fabricating one. This module used to
        # take the FIRST matching call via `next(...)`,
        # silently discarding a second, possibly conflicting, call NAMED
        # `record_final_assessment` -- the same rule the proposal path's
        # exact-one-call validation in `propose()` already enforces. Zero
        # or two-or-more MATCHING calls are refused the same way, through
        # the same repair path, rather than the codebase silently picking
        # a winner.
        #
        # Checking only the matching-name count is still not enough: a
        # turn that sends exactly
        # one `record_final_assessment` call ALONGSIDE some other,
        # unbound tool name would pass through under a matching-name-only
        # check, silently dropping the extra call the same way this whole
        # check exists to prevent. `message.tool_calls`'s installed
        # client (`langchain-anthropic==1.6.1`, confirmed by reading
        # `output_parsers.py:80-92`) copies whatever tool name the
        # provider sends with no validation against the bound list, so
        # this is not proven reachable offline -- but nothing rules it
        # out either, and the fix costs one more length check.
        if (
            len(message.tool_calls) != 1
            or len(matching_calls) != 1
            or message.invalid_tool_calls
            or "schema_version" in matching_calls[0]["args"]
        ):
            return ModelResponse(content={}, usage=usage)
        return ModelResponse(
            content={**matching_calls[0]["args"], "schema_version": SCHEMA_VERSION},
            usage=usage,
        )

    def propose[StageModel: BaseModel](
        self, request: ModelRequest, schema: type[StageModel]
    ) -> ProposedTurn[StageModel]:
        """`schema` is `InitialPlan` or `HypothesisUpdate`, the same
        contract `ReplayToolCallingModel.propose` documents. The one native
        call carries either the check proposal or the stop record."""
        tools = [_stop_tool_definition(), *_domain_tool_definitions()]
        message = self._send(request, tools)
        usage = self._usage(message)
        if _has_visible_content(message.content):
            return ProposedTurn(
                parsed=None,
                errors="tool-call response must not include visible text",
                tool_call=(),
                usage=usage,
            )
        if message.invalid_tool_calls:
            reasons = "; ".join(
                f"{call.get('name') or '<unnamed>'}: "
                f"{call.get('error') or 'unparseable'}"
                for call in message.invalid_tool_calls
            )
            return ProposedTurn(
                parsed=None,
                errors=f"malformed tool call(s): {reasons}",
                tool_call=(),
                usage=usage,
            )
        if len(message.tool_calls) != 1:
            return ProposedTurn(
                parsed=None,
                errors=(
                    f"the model called {len(message.tool_calls)} tools in one turn; "
                    "exactly one is required"
                ),
                tool_call=(),
                usage=usage,
            )
        call = message.tool_calls[0]
        if call["name"] == RECORD_STOP_TOOL_NAME:
            try:
                stop = StopRecord.model_validate(call["args"])
            except ValidationError as error:
                return ProposedTurn(
                    parsed=None,
                    errors=summarize_errors(error),
                    tool_call=(),
                    usage=usage,
                )
            parsed, errors = _finish_plan(
                schema, stop.hypotheses, None, stop.stop_reason
            )
            return ProposedTurn(parsed=parsed, errors=errors, tool_call=(), usage=usage)
        if call["name"] not in {tool.value for tool in ToolName}:
            return ProposedTurn(
                parsed=None,
                errors=f"unknown proposal tool {call['name']!r}",
                tool_call=(),
                usage=usage,
            )
        native_call = _to_native_tool_call(call)
        proposal, reason = parse_tool_call(native_call)
        if proposal is None:
            return ProposedTurn(parsed=None, errors=reason, tool_call=(), usage=usage)
        try:
            hypotheses = HypothesesRecord.model_validate(
                {"hypotheses": call["args"].get("hypotheses")}
            ).hypotheses
        except ValidationError as error:
            return ProposedTurn(
                parsed=None, errors=summarize_errors(error), tool_call=(), usage=usage
            )
        parsed, errors = _finish_plan(schema, hypotheses, proposal, None)
        return ProposedTurn(
            parsed=parsed, errors=errors, tool_call=(native_call,), usage=usage
        )


class MinimumReservationProbeTransport:
    """A `ChatAnthropic`-shaped stand-in used to size `live_setup.py`'s
    cost-ceiling floor at import time (`minimum_possible_reservation_usd`
    below), and reused directly by `tests/unit/test_live_setup.py` to prove
    that floor genuinely authorizes a real reservation, not just that the
    arithmetic is self-consistent -- the same test-seam contract
    `LiveClaudeModel.__init__`'s own `client` parameter already documents
    for `tests/unit/test_live_model.py`'s fakes (`FakeChatAnthropic`). Never
    sends anything over the network. The response content is irrelevant:
    `_send` writes the reservation both call sites read back BEFORE it ever
    calls `.invoke`, so a bare empty response is enough to let `respond()`
    run to completion."""

    def bind_tools(
        self, tools: list[dict[str, Any]], **kwargs: Any
    ) -> "MinimumReservationProbeTransport":
        return self

    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[],
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


# The `run_id` `build_minimum_final_assessment_request` keys its synthetic
# minimal request on -- `tests/unit/test_live_setup.py`'s authorization test
# reads the ledger row back by this same `run_id`, so both use one shared
# constant rather than each hand-typing the same literal string.
MINIMUM_RESERVATION_PROBE_RUN_ID = "minimum-reservation-probe"


def build_minimum_final_assessment_request() -> ModelRequest:
    """The smallest real FINAL_ASSESSMENT request `graph.py`'s
    `_render_stage_request` could ever build: the one INITIAL_PLAN turn
    every run spends before FINAL_ASSESSMENT can be reached, no evidence
    gathered, and a synthetic single-service incident standing in for
    whatever real incident triggered the call. This module cannot import
    `tests/unit/fake_incident.py`'s own checked-in scenario (application
    code does not depend on the test suite), so it builds an equivalent
    minimal incident locally instead -- only the *shape* has to match that
    fixture (one service, no evidence, one model call already spent), not
    its literal ID strings. `tests/unit/test_live_model.py`'s
    `test_the_smallest_final_assessment_prose_matches_what_inputtoolarge_
    assumes` pins the real fixture's own figure separately, for the same
    prose this function's own synthetic incident renders.

    Used by `minimum_possible_reservation_usd` below, and directly by
    `tests/unit/test_live_setup.py`'s authorization test -- both need the
    exact same minimal request, not two hand-typed near-copies of it.
    """
    scope = IncidentScope(
        incident_id=MINIMUM_RESERVATION_PROBE_RUN_ID,
        services=("service",),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        endpoint="/",
    )
    packet = InitialAlertPacket(
        incident_id=scope.incident_id,
        window_start=scope.started_at,
        window_end=scope.ended_at,
        endpoint=scope.endpoint,
        symptom=GatewaySymptom.ELEVATED_LATENCY,
        services=scope.services,
        alerted_at=scope.started_at,
        alert_source_version="probe",
        symptom_evidence_id="probe-symptom",
        topology_evidence_id="probe-topology",
    )
    budgets = Budgets()
    model_calls_used = 1  # the INITIAL_PLAN turn every run spends first
    context = render_context(
        packet,
        scope,
        evidence=(),
        markers=(),
        model_calls_left=budgets.model_calls - model_calls_used,
        checks_left=budgets.executed_tools,
        passages=(),
    )
    context_text = f"{context}\n\n## Task\n{STAGE_INSTRUCTIONS[Stage.FINAL_ASSESSMENT]}"
    return ModelRequest(
        stage=Stage.FINAL_ASSESSMENT,
        system_text=SYSTEM_TEXT,
        context_text=context_text,
        run_id=MINIMUM_RESERVATION_PROBE_RUN_ID,
        graph_phase="FINAL_ASSESSMENT",
        model_turn=model_calls_used,
        context_digest=digest_text(SYSTEM_TEXT + context_text),
    )


def minimum_possible_reservation_usd(
    pricing: PricingSnapshot = CLAUDE_SONNET_5_PRICING,
) -> float:
    """The smallest `reserved_usd` `_send` could ever compute for a real
    FINAL_ASSESSMENT call.

    `live_setup.py`'s `MINIMUM_USABLE_CEILING_USD` cost-ceiling floor needs
    this figure, and three earlier versions of that floor each
    hand-reconstructed `_send`'s reservation formula in `live_setup.py`
    itself and missed one real component of it -- first the tool schema's
    own token cost, then `SYSTEM_TEXT`'s. Rather than hand-copy the formula
    a fourth time, this function calls `LiveClaudeModel.respond()` --
    `_send`'s real, unmodified reservation code -- against a throwaway
    in-memory ledger and a fake transport that never touches the network,
    so this number cannot silently drift from `_send`'s real behaviour
    again even if that formula changes again later.
    """
    conn = sqlite3.connect(":memory:")
    try:
        ensure_cost_ledger_table(conn)
        request = build_minimum_final_assessment_request()
        # A large sentinel ceiling: this probe exists to measure a
        # reservation, not to test the ceiling gate, so it must never be
        # the thing that refuses it.
        probe = LiveClaudeModel(
            conn,
            ceiling_usd=1_000_000.0,
            pricing=pricing,
            client=MinimumReservationProbeTransport(),  # type: ignore[arg-type]
        )
        probe.respond(request)
        row = conn.execute(
            "SELECT reserved_usd FROM cost_ledger WHERE run_id = ?",
            (request.run_id,),
        ).fetchone()
        assert row is not None, (
            "the probe's own reservation was not written back -- "
            "record_reservation_before_request is broken, not this probe"
        )
        return float(row[0])
    finally:
        conn.close()


def _finish_plan[StageModel: BaseModel](
    schema: type[StageModel],
    hypotheses: Sequence[Hypothesis],
    proposal: ToolProposal | None,
    stop_reason: str | None,
) -> tuple[StageModel | None, str]:
    """Assembles the full `InitialPlan`/`HypothesisUpdate` payload and
    validates it through the same `parse_response` every other stage
    result goes through -- `check_proposal_or_stop`'s "exactly one of
    proposal/stop_reason" invariant is enforced there, not duplicated here."""
    payload: dict[str, JsonValue] = {
        "hypotheses": [hypothesis.model_dump(mode="json") for hypothesis in hypotheses],
        "proposal": proposal.model_dump(mode="json") if proposal is not None else None,
        "stop_reason": stop_reason,
    }
    return parse_response(schema, payload)
