"""Typed contracts shared by the workflow, its artifacts, and the reasoning model.

Every contract lives here because they reference each other constantly: a report
holds an assessment, an assessment cites evidence, evidence points at a receipt.
Splitting them would spread one vocabulary across several files.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from causalops.tools import ToolArguments, ToolName, UtcDatetime

SCHEMA_VERSION = "1"

# The `model_name` label a report/artifact carries for a replay-backed
# investigation. Lives here, not in `cli.py` (where it originated) or
# `graph.py` (which also needs it), because it has two readers:
# `graph.py`'s `run_graph_investigation` seeds it into
# `GraphState["model_name"]` as the default for every caller that does not
# pass a live model's own name, and `cli.py` reads it back on a resumed
# thread instead of hardcoding it a second time -- one shared constant, not
# two independent string literals that could drift.
REPLAY_MODEL_NAME = "replay"


class RootCauseCode(StrEnum):
    CONFIG_CHANGE = "CONFIG_CHANGE"
    DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION = "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION"
    RESOURCE_POOL_SATURATION = "RESOURCE_POOL_SATURATION"
    UNDETERMINED = "UNDETERMINED"


class Disposition(StrEnum):
    DIAGNOSED = "DIAGNOSED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    FAILED_SAFE = "FAILED_SAFE"


class ModelDisposition(StrEnum):
    """The dispositions a model may select. FAILED_SAFE is absent on purpose."""

    DIAGNOSED = "DIAGNOSED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceKind(StrEnum):
    SYMPTOM = "SYMPTOM"
    TOPOLOGY = "TOPOLOGY"
    METRIC = "METRIC"
    LOG = "LOG"
    CHANGE = "CHANGE"

    # Deliberately no RUNBOOK member. An `Evidence` record is an
    # incident-scoped observation gathered by a diagnostic check; a
    # `RunbookPassage` is retrieved guidance and can never prove an incident
    # cause or satisfy an incident-evidence predicate, so it must never be
    # constructible as an `Evidence.kind`. `RunbookCheckOutcome` below draws
    # that same line in types: it has no `kind` field at all, so nothing can
    # construct a runbook result that would reach `evidence.py`'s
    # `CONTEXT_QUOTAS[record.kind]` lookup. Adding a member here would give
    # runbook text a `CheckOutcome.kind` to travel under, undoing that.


class RetrievalMode(StrEnum):
    """Which runbook-retrieval backend served this run, or `DISABLED` if
    none did. `DISABLED` is the default and means retrieval was never
    dispatched this run -- no `search_runbooks` proposal was ever allowed
    and settled. It does *not* mean "found nothing": a proposal that ran in
    `FTS5_LEXICAL` mode and retrieved zero passages is `FTS5_LEXICAL`, not
    `DISABLED` -- that case is `RETRIEVAL_COVERAGE_INSUFFICIENT` instead, a
    different fact about the same run. "Disabled" could otherwise read
    ambiguously between "never dispatched" and "dispatched but found
    nothing"; this codebase resolves it to "never dispatched." See
    `graph.py`'s `dispatch_tool` for where this is set, from the backend's
    own configuration rather than from what it found."""

    DISABLED = "disabled"
    FTS5_LEXICAL = "fts5_lexical"
    PINECONE_SEMANTIC = "pinecone_semantic"


class GatewaySymptom(StrEnum):
    ELEVATED_ERRORS = "ELEVATED_ERRORS"
    ELEVATED_LATENCY = "ELEVATED_LATENCY"
    ELEVATED_ERRORS_AND_LATENCY = "ELEVATED_ERRORS_AND_LATENCY"


class MetricSampleStatus(StrEnum):
    """What a `query_metric` check's raw Prometheus
    response actually contained -- describing only what the query
    returned, never claiming anything about what exists in Prometheus's
    own storage. Before this, `sample_count: 0, max_value: 0.0` rendered
    identically whether the series had genuinely never been scraped, a
    series existed but no grid point resolved to a sample, or every
    returned reading was unreadable -- three different facts collapsed
    into one indistinguishable zero.

    Deliberately neutral names, not `SERIES_ABSENT`: an empty `result`
    list proves only that this one query, over this one window, matched
    nothing -- not that the series does not exist (a `rate(...)` query
    needs two samples inside its lookback, so a real, existing series can
    still return empty here). Naming that state as if it proved absence
    would replace one uninterpretable zero with a confidently wrong one,
    which is worse.

    `MULTIPLE_SERIES` is a label on top of the same sample data every
    other status describes, never a gate that withholds it: a
    multi-series response still has real samples in its first series
    (`parse_samples` always reads `series[0]`, regardless of how many
    series came back), and the payload/summary must still carry them --
    see `run_metric_check`'s own comment on this. A reader who mistakes
    `MULTIPLE_SERIES` for "no usable data" would make the interpretability
    problem this status exists to fix worse, not better.
    """

    MULTIPLE_SERIES = "MULTIPLE_SERIES"
    NO_RETURNED_SERIES = "NO_RETURNED_SERIES"
    NO_USABLE_SAMPLES = "NO_USABLE_SAMPLES"
    ALL_READINGS_DISCARDED = "ALL_READINGS_DISCARDED"
    SAMPLED = "SAMPLED"


class PolicyResult(StrEnum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"


class ToolOutcome(StrEnum):
    EXECUTED = "EXECUTED"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"
    NOT_EXECUTED = "NOT_EXECUTED"


class ReasonCode(StrEnum):
    """Stable codes that appear in receipts, reports, and records."""

    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CROSS_INCIDENT_REQUEST = "CROSS_INCIDENT_REQUEST"
    UNKNOWN_SERVICE = "UNKNOWN_SERVICE"
    OUTSIDE_INCIDENT_WINDOW = "OUTSIDE_INCIDENT_WINDOW"
    # `policy.authorize` denies with this code
    # when it is handed a proposal whose window has not been resolved
    # (`window_start`/`window_end` is `None`) -- the ordinary wrapper path
    # (`tool_wrappers.ToolWrapper.dispatch`) always resolves the window
    # first via `resolve_effective_window`, so this only fires for a direct
    # or future caller of `authorize` that skips that step. A denial, not a
    # crash: the same "make errors actionable" reasoning every other
    # malformed-input branch in `authorize` already follows.
    UNRESOLVED_WINDOW = "UNRESOLVED_WINDOW"
    RESULT_LIMIT_EXCEEDED = "RESULT_LIMIT_EXCEEDED"
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    TOOL_ERROR = "TOOL_ERROR"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    REPAIR_EXHAUSTED = "REPAIR_EXHAUSTED"
    MODEL_CALL_BUDGET_EXHAUSTED = "MODEL_CALL_BUDGET_EXHAUSTED"
    WALL_CLOCK_EXPIRED = "WALL_CLOCK_EXPIRED"
    FORGED_EVIDENCE_REFERENCE = "FORGED_EVIDENCE_REFERENCE"
    RESULT_ALREADY_FINALIZED = "RESULT_ALREADY_FINALIZED"
    # A live model call the cost gate refused *before sending*,
    # because the reservation it would need would exceed
    # `LIVE_EVALUATION_MAX_USD`'s remaining balance. Deliberately its own
    # code, not `BUDGET_EXHAUSTED`: that code already means "the *count*
    # budget this investigation itself tracks is spent" (model calls, tool
    # slots, repairs -- all in `Budgets`, all per-investigation). The cost
    # ceiling is an *application-wide* dollar balance no single investigation
    # owns, so conflating the two would make a report say "this investigation
    # ran out of turns" when what actually happened is "a wholly separate
    # concern -- money across every run this app has ever made -- ran out."
    # An owner reading `reason_code` needs to tell those apart to know
    # whether raising `Budgets.model_calls` would even help (it would not).
    COST_CEILING_EXCEEDED = "COST_CEILING_EXCEEDED"
    # The rendered request's pessimistic token estimate exceeded
    # the 9,600-token input cap (`pricing.MAX_INPUT_TOKENS`). Its own code, not
    # `COST_CEILING_EXCEEDED`: one is about a dollar balance across every
    # run this application has made, the other is about one request's shape
    # regardless of what anything has ever cost -- an owner reading
    # `reason_code` needs to tell "we are out of money" apart from "this one
    # request was too big" (the second is fixable by trimming context; the
    # first is not, no matter how small the next request is).
    INPUT_TOKEN_CAP_EXCEEDED = "INPUT_TOKEN_CAP_EXCEEDED"
    # `cost_ledger.record_reservation_before_request` found a reservation
    # already on file for this exact request
    # key (`run_id`, `graph_phase`, `model_turn`, `context_digest`) -- this
    # attempt did not create a new one. `live_model.py`'s `_send` refuses to
    # invoke the provider under a pre-existing reservation rather than
    # assume a retry is safe: a `RESERVED` existing row means an earlier
    # attempt at this exact request may already be in flight or may have
    # crashed after paying but before this process learned about it: this
    # module cannot tell those apart, so it refuses to send a second real
    # request rather than guess; a `SETTLED` one means an earlier attempt
    # already completed and this module has no way to recover the original
    # response to return instead (`CostLedgerRow` stores only cost and token
    # counts, not response content) -- either way, resending is the one
    # option guaranteed to risk paying twice for a single logical request.
    # Its own code, not `COST_CEILING_EXCEEDED`: that code means "no money is
    # left"; this one means "money may already be spent on this exact
    # request" -- an owner reading `reason_code` needs to know a retry with
    # a smaller budget would not help here, unlike a ceiling refusal.
    AMBIGUOUS_MODEL_REQUEST = "AMBIGUOUS_MODEL_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GraphPhase(StrEnum):
    """The LangGraph phases of the investigation workflow: existing CLI /
    report -> investigator -> policy-wrapped read-only tools -> investigator
    -> normalized evidence -> final assessment -> escalation interrupt ->
    report.

    This describes the graph the workflow's own design defines, not just
    what `graph.py` builds today. Names every phase a state machine can
    reach, including ones no code visits yet -- the same precedent
    `InvestigationState`, the retired loop orchestrator's own state enum,
    set before it was removed once `workflow.py` no longer used it.
    """

    CREATED = "CREATED"
    INVESTIGATE = "INVESTIGATE"
    DISPATCH_TOOL = "DISPATCH_TOOL"
    NORMALIZE_EVIDENCE = "NORMALIZE_EVIDENCE"
    FINAL_ASSESSMENT = "FINAL_ASSESSMENT"
    ESCALATION_INTERRUPT = "ESCALATION_INTERRUPT"
    FINAL_REPORT = "FINAL_REPORT"


class EscalationReason(StrEnum):
    """The four deterministic conditions under which an investigation pauses
    for owner escalation, all four now reachable.

    `graph.py`'s `_escalation_reason` checks a receipt outcome, the model's
    own disposition, and its own contrary-citation list for the first
    three -- all already in state by the time it runs.
    `RETRIEVAL_COVERAGE_INSUFFICIENT` needed `search_runbooks`, which did not
    exist when the other three became reachable; it was named here anyway,
    the same precedent `GraphPhase` itself sets above -- naming a state
    before anything reaches it costs nothing and meant this enum did not
    need a second, breaking edit once retrieval landed. `_escalation_reason`
    fires it when an `ALLOWED`, `SETTLED` `search_runbooks` receipt exists
    but this run retrieved zero passages.
    """

    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    # Deliberately the same literal as `ReasonCode.TOOL_UNAVAILABLE`'s
    # value -- both name the same underlying fact from two different
    # vocabularies. Both are `StrEnum`, so
    # `EscalationReason.TOOL_UNAVAILABLE == ReasonCode.TOOL_UNAVAILABLE` is
    # `True` (plain string equality) even though they are different classes
    # and `is` says they are not the same member -- a trap for an `==`
    # comparison written against the wrong vocabulary. `ReasonCode` names a
    # receipt outcome; this enum names why a run paused because of one.
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING = (
        "INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING"
    )
    RETRIEVAL_COVERAGE_INSUFFICIENT = "RETRIEVAL_COVERAGE_INSUFFICIENT"


class Budgets(BaseModel):
    """The limits this step enforces. Provider limits arrive with the Claude model."""

    model_config = ConfigDict(frozen=True)

    wall_clock_seconds: int = 360
    tool_timeout_seconds: int = 10
    model_calls: int = 4
    executed_tools: int = 2
    # `policy.py`'s new `SearchRunbooksArguments` branch checks
    # `arguments.limit` against this, the same role `log_rows` plays for
    # `QueryLogsArguments.row_limit`. A `search_runbooks` call still spends
    # one of the two `executed_tools` slots above -- runbook retrieval
    # counts against that same shared per-investigation budget, not a
    # separate one -- this only bounds how many passages one call may ask
    # for.
    runbook_passages: int = 5
    # `may_repair` (`graph.py`) checks `repairs_used < repairs` with no other
    # gate, so this field alone now bounds how many structured-output
    # repairs one run may attempt -- it needs its own validation rather than
    # trusting every caller to pass a sane value. `le=2`, not unbounded:
    # `repairs_used` is cumulative across the whole run in `GraphState`, not
    # reset per stage, so `repairs=2` covers the two kinds of stage a repair
    # could ever be spent on -- an `INVESTIGATE` turn's invalid response, and
    # the separate `FINAL_ASSESSMENT` turn that must eventually follow it --
    # each getting its own repair rather than the first one to fail claiming
    # the whole budget. (A run with several `INVESTIGATE` turns could in
    # principle want more than one INVESTIGATE-side repair; `le=2` is this
    # design's intended per-run allocation, not a proof that a third repair
    # could never help.) Nothing in production constructs `repairs=2` today
    # (the default stays 1); the bound exists so a future caller cannot pass
    # a value this design has no stated meaning for.
    repairs: int = Field(default=1, ge=0, le=2)
    log_rows: int = 40


DEFAULT_BUDGETS = Budgets()


class Versions(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    prompt_version: str
    policy_version: str
    tool_registry_version: str


class ModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int


class IncidentScope(BaseModel):
    """Opaque identity plus the only services and window a check may touch."""

    model_config = ConfigDict(frozen=True)

    incident_id: str
    environment: Literal["local-lab"] = "local-lab"
    services: tuple[str, ...] = Field(min_length=1)
    started_at: UtcDatetime
    ended_at: UtcDatetime
    endpoint: str

    @model_validator(mode="after")
    def check_window(self) -> Self:
        if self.ended_at <= self.started_at:
            raise ValueError("incident window must end after it starts")
        return self


class Evidence(BaseModel):
    """One stored observation that an answer can cite."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    evidence_id: str
    incident_id: str
    kind: EvidenceKind
    source: str
    observed_at: UtcDatetime
    summary: str = Field(max_length=400)
    payload: dict[str, JsonValue]
    receipt_id: str | None = None
    content_hash: str


class InitialAlertPacket(BaseModel):
    """The answer-neutral starting point, identical for every system under test."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    incident_id: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    endpoint: str
    symptom: GatewaySymptom
    services: tuple[str, ...] = Field(min_length=1)
    alerted_at: UtcDatetime
    alert_source_version: str
    symptom_evidence_id: str
    topology_evidence_id: str


class StoredIncident(BaseModel):
    """What the scenario controller leaves behind for `investigate` to pick up.

    Investigator-visible only: an opaque scope, the answer-neutral packet, and the
    packet's own evidence. Expected outcomes live elsewhere.

    Three fields carry an `incident_id`, and this
    validator is what confirms all three actually agree -- an auditable
    list, the same discipline `live_model.py`'s `_RATIONALE_PROPERTIES`
    and `test_live_model.py`'s `KNOWN_PROSE_ONLY_CONTRACTS` already use
    elsewhere in this codebase for "here is exactly what is covered, and
    where":

    - `scope.incident_id` -- the reference every other field is checked
      against, by `check_identity_agrees` below.
    - `packet.incident_id` -- checked against `scope.incident_id` by
      `check_identity_agrees` below.
    - `evidence[i].incident_id`, for every item -- checked against
      `scope.incident_id` by `check_identity_agrees` below (this is the
      same rule `evidence.EvidenceStore.add`'s own `ValueError` enforces
      at insertion time for a run already in progress; this validator is
      what enforces it at LOAD time, before a stored artifact from disk
      ever reaches that far).

    `cli.py`'s `run_investigate_command`/`run_decision_command` check a
    FOURTH identity fact this validator cannot: that `scope.incident_id`
    matches the `runs/<incident_id>/` DIRECTORY NAME the artifact was
    loaded from. That is not a fact about the artifact's own internal
    consistency (what this validator checks) but about the artifact
    against its filesystem location, which only a caller holding both can
    verify.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    scope: IncidentScope
    packet: InitialAlertPacket
    evidence: tuple[Evidence, ...]

    @model_validator(mode="after")
    def check_identity_agrees(self) -> Self:
        """Closes a real safe-failure breakage a mismatched artifact could
        cause: `graph.py`'s `_rebuild_store` raises `ValueError` on an
        `evidence[i].incident_id`
        mismatch, and that function is called from BOTH the normal
        `_build_report` path and the outer crash-containment path that
        exists to catch exactly this kind of failure -- a mismatched
        artifact that got past loading would raise the identical error a
        second time, from inside the handler meant to catch the first one,
        and escape `main()`'s `(LabError, RunRecordError,
        CheckpointStoreError)` catch entirely. Checking identity HERE, at
        load time (`_load_stored_artifact` in `cli.py` already converts a
        `ValidationError` into a clean `LabError(CORRUPT_ARTIFACT)`, no new
        error handling needed), closes the gap before either graph path
        ever sees the artifact -- `EvidenceStore.add`'s own `ValueError`
        stays exactly as it is, an internal invariant guard for a run
        already in progress, not the thing fixed here."""
        if self.packet.incident_id != self.scope.incident_id:
            raise ValueError(
                f"packet.incident_id {self.packet.incident_id!r} does not "
                f"match scope.incident_id {self.scope.incident_id!r}"
            )
        for item in self.evidence:
            if item.incident_id != self.scope.incident_id:
                raise ValueError(
                    f"evidence {item.evidence_id} has incident_id "
                    f"{item.incident_id!r}, not scope.incident_id "
                    f"{self.scope.incident_id!r}"
                )
        return self


class Hypothesis(BaseModel):
    """A possible cause and what would settle it. Rank is not a probability."""

    # `extra="forbid"` rationale: see `tools.py`'s `QueryMetricArguments`.
    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause: RootCauseCode
    rank: int = Field(ge=1)
    supporting_evidence_ids: tuple[str, ...] = ()
    contrary_evidence_ids: tuple[str, ...] = ()
    # `maxLength: 300` is a schema keyword Anthropic's
    # structured outputs do not enforce server-side -- a real live call once
    # exceeded this same 300-char shape on `FinalAssessment.uncertainty`'s
    # first, uncorrected attempt (that field's bound has since been raised
    # to 600 -- see its comment below) -- so prose stating the bound in
    # words is the only mechanism that can actually keep a model under it,
    # and the limit is named here rather than left to the schema alone.
    # This field's own bound stays 300: real evidence-heavy runs have shown
    # no failures here, only on `uncertainty`.
    missing_evidence: str = Field(
        max_length=300,
        description=(
            "The open question that would settle this hypothesis, in 300 "
            "characters or fewer."
        ),
    )


class ToolProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    arguments: ToolArguments
    # These two, and `FinalAssessment.next_step` below, were checked
    # against real saved investigation data alongside `uncertainty` and
    # `stop_reason` before deciding not to raise them too, not left
    # unexamined: `evidence_gap` and `expected_observation` measured
    # max 215/232 chars (n=531 each, median 52/40), `next_step` measured
    # max 216 chars (n=585, median 56) -- all three at least 68 characters
    # under the 300 bound with zero observed failures, unlike `uncertainty`.
    evidence_gap: str = Field(max_length=300)
    expected_observation: str = Field(max_length=300)

    @property
    def tool(self) -> ToolName:
        return self.arguments.tool


class ReceiptState(StrEnum):
    """Has this check run yet? Separate from `ToolOutcome`, which answers what
    happened once it did. A wrapper must reserve budget before it calls a
    backend; `RESERVED` is that reservation made visible, so a crash between
    reserving and settling still leaves a receipt instead of vanishing
    silently (see `tool_wrappers.py`)."""

    RESERVED = "RESERVED"
    SETTLED = "SETTLED"


class ToolReceipt(BaseModel):
    """What a proposal did, whether or not it was allowed to run.

    `state` defaults to `SETTLED` because every receipt built before this field
    existed is one-shot: policy decides, the backend runs (or doesn't), and the
    receipt is written once with a known outcome. Only the reservation path in
    `tool_wrappers.py` ever constructs a `RESERVED` receipt.

    `arguments` defaults to `None` for the same reason `state` does above:
    every receipt built before this field existed predates it entirely. `None`
    here means "this receipt predates the field," never "this check ran with
    no arguments" -- every tool this application registers requires at least
    one argument, so a settled receipt for a real check always has a real
    `ToolArguments` value once this field is populated. `ledger.reserve`,
    `ledger.settle`, and `_denied_receipt` in `tool_wrappers.py` always set it
    on a freshly built receipt; only a receipt round-tripped from a
    `receipts.jsonl` or checkpoint written before this field existed can carry
    `None`. This carries the *effective* arguments a backend actually ran (or
    would have run, for a denial) -- today identical
    to what the model requested, since nothing yet normalizes a proposal's
    window before dispatch; a later change that adds clamping only has to set
    this field to the clamped value at the same construction sites, no shape
    change here. No `SCHEMA_VERSION` bump: an added optional field breaks no
    reader, the same reasoning `state`'s own addition and `GraphState.model_
    name`'s addition both already used.

    On a denial, this is the wrapper's
    *resolved* attempt to authorize -- the value it tried to check the
    model's request against -- not necessarily the value the model wrote
    verbatim; the as-proposed (raw) window is separately recorded on
    `investigate`'s own `proposal_recorded` event for the same
    `proposal_turn`, joinable via `dispatch_tool`'s
    `proposal_denied` event's own `proposal_turn`/`receipt_id`.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    receipt_id: str
    incident_id: str
    tool: ToolName
    fingerprint: str
    policy_result: PolicyResult
    state: ReceiptState = ReceiptState.SETTLED
    outcome: ToolOutcome | None = None
    reason_code: ReasonCode | None = None
    requested_at: UtcDatetime
    duration_ms: int = Field(ge=0)
    result_digest: str | None = None
    evidence_id: str | None = None
    arguments: ToolArguments | None = None

    @model_validator(mode="after")
    def check_lifecycle_coherence(self) -> Self:
        carries_a_result = (
            self.outcome is not None
            or self.result_digest is not None
            or self.evidence_id is not None
        )
        if self.state is ReceiptState.RESERVED and carries_a_result:
            raise ValueError(
                "a reserved receipt has not run yet and cannot carry a result"
            )
        if self.state is ReceiptState.SETTLED and self.outcome is None:
            raise ValueError("a settled receipt must carry an outcome")
        return self


class InitialPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    hypotheses: tuple[Hypothesis, ...] = Field(min_length=2, max_length=3)
    proposal: ToolProposal | None = None
    stop_reason: str | None = Field(default=None, min_length=1, max_length=600)

    @model_validator(mode="after")
    def check_proposal_or_stop(self) -> Self:
        # A stage either proposes one check or explains why it stops, never both.
        if (self.proposal is None) == (self.stop_reason is None):
            raise ValueError("give either a tool proposal or a stop reason")
        return self


class HypothesisUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    hypotheses: tuple[Hypothesis, ...] = Field(min_length=2, max_length=3)
    proposal: ToolProposal | None = None
    stop_reason: str | None = Field(default=None, min_length=1, max_length=600)

    @model_validator(mode="after")
    def check_proposal_or_stop(self) -> Self:
        if (self.proposal is None) == (self.stop_reason is None):
            raise ValueError("give either a tool proposal or a stop reason")
        return self


class FinalAssessment(BaseModel):
    """The model's diagnosis or abstention. Its schema cannot express FAILED_SAFE."""

    # `extra="forbid"` rationale: see `tools.py`'s `QueryMetricArguments`.
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    # `check_terminal_invariants` below enforces a rule
    # invisible to both the emitted schema and a one-line tool description --
    # a `model_validator(mode="after")` cannot appear in `model_json_schema()`
    # 's output, so nothing in the wire schema told Claude that a DIAGNOSED
    # assessment needs both a real root cause and cited evidence, or that an
    # abstention needs UNDETERMINED. These three field descriptions state the
    # same terminal-disposition rule the validator enforces, so a model
    # reading the tool schema sees the invariant before it is refused for
    # missing it, not only after.
    #
    # This and `root_cause`'s description below restate the
    # SAME rule `_final_assessment_tool_definition`'s hand-written tool-level
    # description already states in `live_model.py`. Deliberate,
    # not accidental duplication to "simplify away": the tool-level
    # description is redundant WITH these two exactly because a model reads
    # both, and it is unconfirmed whether Anthropic's parser honours a
    # `description` sibling to a property's own `$ref` the way it honours
    # the tool's own top-level description -- collapsing to one copy risks
    # losing the guidance entirely if the sibling form is ever silently
    # ignored. Load-bearing redundancy, not duplication to prune.
    #
    # The final sentence below ("An abstention must still
    # cite...") is ALSO deliberately duplicated in `prompts.py`'s
    # `STAGE_INSTRUCTIONS[Stage.FINAL_ASSESSMENT]`, a different duplication
    # than the one above -- that copy reaches the rendered prompt text, this
    # one reaches only this tool-call JSON schema, so a model needs both to
    # see the rule regardless of which channel it actually reads. See
    # `test_live_model.py`'s `test_the_respond_tool_payload_size_matches_
    # what_pricingpy_assumes` docstring for how this sentence's token cost
    # is accounted for on each side.
    disposition: ModelDisposition = Field(
        description=(
            "DIAGNOSED requires a root_cause other than UNDETERMINED and at "
            "least one supporting_evidence_ids entry. Use "
            "INSUFFICIENT_EVIDENCE with root_cause UNDETERMINED to abstain "
            "instead of guessing. An abstention must still cite the evidence "
            "that made the causes indistinguishable in supporting_evidence_ids, "
            "not leave it empty."
        )
    )
    root_cause: RootCauseCode = Field(
        description=(
            "UNDETERMINED when disposition is INSUFFICIENT_EVIDENCE; a "
            "specific cause other than UNDETERMINED when disposition is "
            "DIAGNOSED."
        )
    )
    # `graph.py`'s `store.unknown_ids(cited)` check runs after the model's
    # response is parsed, so a forged or mistyped id in EITHER field below is
    # terminal with no repair (`ReasonCode.FORGED_EVIDENCE_REFERENCE`) --
    # unlike a malformed shape, there is no second chance to fix a wrong id, so
    # the instruction has to land on the first attempt. Both fields must state
    # this: `graph.py:1069`'s `cited = parsed.supporting_evidence_ids +
    # parsed.contrary_evidence_ids` feeds both fields into the identical check,
    # so documenting one and leaving its sibling silent would be the same
    # mistake in two places.
    supporting_evidence_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "A DIAGNOSED assessment must cite at least one entry here. Copy "
            "evidence ids exactly as they appear in the Evidence section -- "
            "an id that was not retrieved this run fails the investigation."
        ),
    )
    contrary_evidence_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "Copy evidence ids exactly as they appear in the Evidence "
            "section -- an id that was not retrieved this run fails the "
            "investigation."
        ),
    )
    # The final assessment stores incident-evidence citations separately
    # from runbook-guidance citations. Deliberately
    # not merged into the two fields above -- `check_terminal_invariants`
    # below is unchanged, so "a diagnosis must cite supporting evidence"
    # still means incident evidence only, and `evaluation.py`'s
    # `citations_are_valid`/`citations_sufficient` (which read only the two
    # fields above) score exactly what they scored before this field
    # existed.
    runbook_citations: tuple[str, ...] = ()
    # See `Hypothesis.missing_evidence`'s comment above -- a live run once
    # exceeded this field's then-300 bound on an unrepaired first attempt,
    # and the provider does not enforce `maxLength` server-side, so the word
    # limit has to be stated in prose. Later real evidence-heavy runs (a
    # measured accuracy-vs-evidence-budget curve, `executed_tools`=2/3/4)
    # showed this field's median length grows with the amount of evidence
    # gathered (210->230->240 chars across the curve) and hit 300 outright at
    # the top of that range, exhausting the run's repair budget with no
    # attempt left -- raised to 600 to give real headroom above the observed
    # growth, not just above one failure.
    uncertainty: str = Field(
        max_length=600,
        description=(
            "What remains unresolved about this diagnosis, in 600 characters or fewer."
        ),
    )
    next_step: str = Field(
        max_length=300,
        description=(
            "The single next action you would recommend, in 300 characters or fewer."
        ),
    )

    @model_validator(mode="after")
    def check_terminal_invariants(self) -> Self:
        if self.disposition is ModelDisposition.DIAGNOSED:
            if self.root_cause is RootCauseCode.UNDETERMINED:
                raise ValueError(
                    "a diagnosis needs a root cause other than UNDETERMINED"
                )
            if not self.supporting_evidence_ids:
                raise ValueError("a diagnosis must cite supporting evidence")
            return self
        if self.root_cause is not RootCauseCode.UNDETERMINED:
            raise ValueError("an abstention requires UNDETERMINED")
        return self


class CheckOutcome(BaseModel):
    """What running one approved check produced."""

    model_config = ConfigDict(frozen=True)

    outcome: ToolOutcome
    kind: EvidenceKind
    source: str
    summary: str
    payload: dict[str, JsonValue] = {}
    reason_code: ReasonCode | None = None
    duration_ms: int = 0


class RunbookPassage(BaseModel):
    """One retrieved passage of guidance, carrying the fields a citation
    needs to be checked and reproduced: `passage_id, content, source_version,
    content_hash, score, retrieval_mode`.

    Deliberately not `Evidence`: no `incident_id` (guidance is not
    incident-scoped) and no `EvidenceKind` (see `EvidenceKind`'s own
    docstring for why). `content_hash` reuses `evidence.digest_text` over
    `content`, the same hashing `evidence.content_hash` already uses for
    tool-result payloads.
    """

    model_config = ConfigDict(frozen=True)

    passage_id: str
    content: str = Field(max_length=800)
    source_version: str
    content_hash: str
    score: float
    retrieval_mode: RetrievalMode


class RunbookCheckOutcome(BaseModel):
    """What running an approved `search_runbooks` check produced --
    `CheckOutcome`'s sibling for the one tool whose result is guidance, not
    evidence.

    Structurally, not just conventionally, distinct from `CheckOutcome`:
    there is no `kind` field, so nothing here can reach
    `evidence.CONTEXT_QUOTAS[record.kind]`'s lookup, and `tool_wrappers.py`'s
    `_make_wrapper` uses `isinstance(outcome, RunbookCheckOutcome)` to route
    a result to `DispatchResult.passages` instead of minting `Evidence` --
    a type distinction a shared `CheckOutcome.kind` value could not have
    given it. `retrieval_mode` is set here by the backend regardless of how
    many passages it found (including on a failed or empty search) -- see
    `RetrievalMode`'s own docstring for why the report must never infer
    `disabled` from an empty `passages` tuple.
    """

    model_config = ConfigDict(frozen=True)

    outcome: ToolOutcome
    passages: tuple[RunbookPassage, ...] = ()
    retrieval_mode: RetrievalMode
    reason_code: ReasonCode | None = None
    duration_ms: int = 0


# `RunCheck` is the callable shape a check backend implements. Production wires
# typed per-tool backends through `tool_wrappers.dispatch_registry`; tests pass
# simple doubles that match this alias directly.
RunCheck = Callable[[ToolProposal, IncidentScope], CheckOutcome]
Clock = Callable[[], datetime]


class EscalationRecord(BaseModel):
    """Whether this investigation paused for the owner, and what they decided.

    A separate type from `EscalationReason` alone, because a report needs to
    carry both the trigger and the outcome together -- `reason` without a
    `decision` would leave a reader unable to tell a still-open escalation
    from a resolved one, and `graph.py`'s `_build_report` only ever
    constructs this once both are known. `decision` matches the literal
    resume values `escalation_interrupt` accepts, not a past-tense
    rewording, so there is one vocabulary for the same concept across the
    node, the graph state, and the report.

    `rejection_note` carries the owner's reason for
    `causalops reject <thread-id> <reason>`. Named `rejection_note`, not
    `owner_reason` (which would collide with `reason` above -- the enum
    trigger, a different concept) or `decision_note` (which would wrongly
    imply an accept can carry one). `check_rejection_note_pairing` mirrors
    `InvestigationReport.check_terminal_invariants`'s own idiom and is the
    same pairing rule `causalops.approvals.OwnerDecision` enforces at the
    CLI boundary and `graph.py`'s `_parse_resume_decision` enforces on the
    resume value -- checked three times, at the three points a caller
    could reach this model from, not three independent proof techniques
    the way the trust boundary's AST scan / wrapper-identity / spy-backend
    controls are. `OwnerDecision` is the only one of the three a
    `causalops reject` argument is guaranteed to pass through; the other
    two exist because `Command(resume=...)` and this model's own
    constructor are both reachable directly (tests already do it), so each
    repeats the identical whitespace-aware check -- `not
    rejection_note.strip()` treats a whitespace-only note as missing,
    matching `OwnerDecision`'s own normalization -- rather than trusting
    that every value reaching it already passed through the CLI.
    """

    model_config = ConfigDict(frozen=True)

    reason: EscalationReason
    decision: Literal["accept", "reject"]
    rejection_note: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def check_rejection_note_pairing(self) -> Self:
        if self.decision == "reject" and not (
            self.rejection_note and self.rejection_note.strip()
        ):
            raise ValueError("a rejection must carry a non-empty rejection note")
        if self.decision == "accept" and self.rejection_note is not None:
            raise ValueError("an acceptance must not carry a rejection note")
        return self


class InvestigationReport(BaseModel):
    """The finalized result of one investigation."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    investigation_id: str
    incident_id: str
    disposition: Disposition
    root_cause: RootCauseCode
    assessment: FinalAssessment | None = None
    reason_code: ReasonCode | None = None
    budgets: Budgets
    versions: Versions
    started_at: UtcDatetime
    finished_at: UtcDatetime
    latency_ms: int = Field(ge=0)
    model_calls_used: int = Field(ge=0)
    repairs_used: int = Field(ge=0)
    tools_executed: int = Field(ge=0)
    invalid_responses: int = Field(ge=0)
    usage: ModelUsage | None = None
    final_context_digest: str
    evidence_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    # `retrieval_mode` is `DISABLED` on every run that never dispatched an
    # allowed `search_runbooks` check, and is set from the backend's own
    # configuration otherwise, never inferred from whether any passage came
    # back (`RetrievalMode`'s docstring). `runbook_passage_ids` mirrors
    # `evidence_ids`/`receipt_ids`'s own auditability: an owner can re-resolve
    # every id the assessment's `runbook_citations` names against this list the
    # same way `evidence_ids` lets them re-resolve `supporting_evidence_ids`.
    retrieval_mode: RetrievalMode = RetrievalMode.DISABLED
    runbook_passage_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    # `None` for every run that never escalated -- most runs. Deliberately not
    # folded into `limitations`: that field is free text an evaluator would
    # have to string-match to answer "did this escalate, and what did the owner
    # decide," where this field answers it directly. Additive and optional, so
    # `check_terminal_invariants` below needs no change -- disposition,
    # root_cause, assessment, and reason_code are unaffected by whether an
    # owner accepted or rejected the assessment they describe.
    escalation: EscalationRecord | None = None

    @model_validator(mode="after")
    def check_terminal_invariants(self) -> Self:
        if self.disposition is Disposition.DIAGNOSED:
            if self.root_cause is RootCauseCode.UNDETERMINED:
                raise ValueError(
                    "a diagnosis needs a root cause other than UNDETERMINED"
                )
            if self.assessment is None:
                raise ValueError("a diagnosis requires a model assessment")
            return self
        if self.root_cause is not RootCauseCode.UNDETERMINED:
            raise ValueError("a non-diagnosis requires UNDETERMINED")
        if self.disposition is Disposition.INSUFFICIENT_EVIDENCE:
            if self.assessment is None:
                raise ValueError("an abstention requires a model assessment")
            return self
        if self.disposition is Disposition.FAILED_SAFE:
            if self.assessment is not None:
                raise ValueError(
                    "FAILED_SAFE comes from application code, with no assessment"
                )
            if self.reason_code is None:
                raise ValueError("FAILED_SAFE requires a reason code")
        return self


def utc_now() -> datetime:
    return datetime.now(UTC)


class InvestigationResult(BaseModel):
    """The report plus the artifacts that belong beside it when it is finalized.

    Domain vocabulary, not orchestrator-specific: `graph.py`'s LangGraph
    orchestrator produces one of these from a finished run today, and so did
    the retired `workflow.py` loop it replaced.
    """

    model_config = ConfigDict(frozen=True)

    report: InvestigationReport
    evidence: tuple[Evidence, ...]
    receipts: tuple[ToolReceipt, ...]


class EscalatedInvestigation(BaseModel):
    """A paused investigation awaiting the owner's accept/reject decision.

    A sibling to `InvestigationResult`, not a variant of it: a paused run has
    no `InvestigationReport` yet, since its disposition is not resolved, so
    it cannot satisfy that model's strict terminal-disposition validator
    (`check_terminal_invariants` above) or its required `report` field.
    `evidence`/`receipts` mirror `InvestigationResult`'s own shape exactly,
    so an owner inspecting a paused run sees the same policy-authorized tool
    evidence a finished one would show -- `graph.py`'s
    `run_graph_investigation` already has both in hand at pause time, from
    the same state a finished run's report is built from, so carrying them
    costs nothing extra. `checkpoint_id`/`reason`/`remaining_check_count`/
    `proposal_fingerprint` are the LangGraph interrupt's own payload;
    `proposal_fingerprint` is always `None` today -- nothing in the
    codebase can produce a policy-approved next-check proposal at escalation
    time yet (see `graph.py`'s module docstring).
    """

    model_config = ConfigDict(frozen=True)

    thread_id: str
    run_id: str
    checkpoint_id: str
    reason: EscalationReason
    evidence: tuple[Evidence, ...]
    receipts: tuple[ToolReceipt, ...]
    remaining_check_count: int = Field(ge=0)
    proposal_fingerprint: str | None = None
