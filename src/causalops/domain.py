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


class GatewaySymptom(StrEnum):
    ELEVATED_ERRORS = "ELEVATED_ERRORS"
    ELEVATED_LATENCY = "ELEVATED_LATENCY"
    ELEVATED_ERRORS_AND_LATENCY = "ELEVATED_ERRORS_AND_LATENCY"


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
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GraphPhase(StrEnum):
    """The LangGraph phases from `TECHNICAL_SPEC.md` §5's diagram.

    This describes the graph the spec defines, not just what Unit 1a builds --
    `graph.py` (Unit 1b) is the first consumer. Names every phase a state
    machine can reach, including ones no code visits yet -- the precedent set
    by `InvestigationState`, the retired loop orchestrator's own state enum
    (removed in Unit 1d-2, once `workflow.py` no longer used it).
    """

    CREATED = "CREATED"
    INVESTIGATE = "INVESTIGATE"
    DISPATCH_TOOL = "DISPATCH_TOOL"
    NORMALIZE_EVIDENCE = "NORMALIZE_EVIDENCE"
    FINAL_ASSESSMENT = "FINAL_ASSESSMENT"
    ESCALATION_INTERRUPT = "ESCALATION_INTERRUPT"
    FINAL_REPORT = "FINAL_REPORT"


class EscalationReason(StrEnum):
    """`TECHNICAL_SPEC.md` §8's four deterministic escalation triggers.

    Only the first three are reachable from Unit 2b: `graph.py`'s
    `_escalation_reason` checks a receipt outcome, the model's own
    disposition, and its own contrary-citation list -- all already in state.
    `RETRIEVAL_COVERAGE_INSUFFICIENT` needs `search_runbooks`, which does not
    exist until Milestone 3's retrieval lands (`TECHNICAL_SPEC.md`'s Unit 0
    amendment). It is named here anyway, the same precedent `GraphPhase`
    itself sets above: naming a state before anything reaches it costs
    nothing and means this enum doesn't need a second, breaking edit when
    Milestone 3 arrives.
    """

    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    # `TECHNICAL_SPEC.md` §8 mandates this exact literal, which is also
    # `ReasonCode.TOOL_UNAVAILABLE`'s value. Both are `StrEnum`, so
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
    repairs: int = 1
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
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    scope: IncidentScope
    packet: InitialAlertPacket
    evidence: tuple[Evidence, ...]


class Hypothesis(BaseModel):
    """A possible cause and what would settle it. Rank is not a probability."""

    model_config = ConfigDict(frozen=True)

    root_cause: RootCauseCode
    rank: int = Field(ge=1)
    supporting_evidence_ids: tuple[str, ...] = ()
    contrary_evidence_ids: tuple[str, ...] = ()
    missing_evidence: str = Field(max_length=300)


class ToolProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    arguments: ToolArguments
    evidence_gap: str = Field(max_length=300)
    expected_observation: str = Field(max_length=300)

    @property
    def tool(self) -> ToolName:
        return self.arguments.tool


class ReceiptState(StrEnum):
    """Has this check run yet? Separate from `ToolOutcome`, which answers what
    happened once it did. `TECHNICAL_SPEC.md` §5 requires a wrapper to reserve
    budget before it calls a backend; `RESERVED` is that reservation made
    visible, so a crash between reserving and settling still leaves a receipt
    instead of vanishing silently (see `tool_wrappers.py`)."""

    RESERVED = "RESERVED"
    SETTLED = "SETTLED"


class ToolReceipt(BaseModel):
    """What a proposal did, whether or not it was allowed to run.

    `state` defaults to `SETTLED` because every receipt built before this unit
    is one-shot: policy decides, the backend runs (or doesn't), and the receipt
    is written once with a known outcome. Only the new reservation path in
    `tool_wrappers.py` ever constructs a `RESERVED` receipt.
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
    stop_reason: str | None = Field(default=None, max_length=300)

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
    stop_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def check_proposal_or_stop(self) -> Self:
        if (self.proposal is None) == (self.stop_reason is None):
            raise ValueError("give either a tool proposal or a stop reason")
        return self


class FinalAssessment(BaseModel):
    """The model's diagnosis or abstention. Its schema cannot express FAILED_SAFE."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    disposition: ModelDisposition
    root_cause: RootCauseCode
    supporting_evidence_ids: tuple[str, ...] = ()
    contrary_evidence_ids: tuple[str, ...] = ()
    uncertainty: str = Field(max_length=300)
    next_step: str = Field(max_length=300)

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


# Step 3 supplies the registry-backed runner. Until the lab exists there is nothing
# for the loop to call, so the caller passes one in and tests pass doubles.
RunCheck = Callable[[ToolProposal, IncidentScope], CheckOutcome]
Clock = Callable[[], datetime]


class EscalationRecord(BaseModel):
    """Whether this investigation paused for the owner, and what they decided.

    A separate type from `EscalationReason` alone, because a report needs to
    carry both the trigger and the outcome together -- `reason` without a
    `decision` would leave a reader unable to tell a still-open escalation
    from a resolved one, and `graph.py`'s `_build_report` only ever
    constructs this once both are known. `decision` matches the literal
    resume values `escalation_interrupt` accepts (`"accept"`/`"reject"`),
    not a past-tense rewording, so there is one vocabulary for the same
    concept across the node, the graph state, and the report.
    """

    model_config = ConfigDict(frozen=True)

    reason: EscalationReason
    decision: Literal["accept", "reject"]


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
    limitations: tuple[str, ...] = ()
    # Unit 2b. `None` for every run that never escalated -- which is every
    # run before this unit and most runs after it. Deliberately not folded
    # into `limitations`: that field is free text an evaluator would have to
    # string-match to answer "did this escalate, and what did the owner
    # decide," where this field answers it directly. Additive and optional,
    # so `check_terminal_invariants` below needs no change -- disposition,
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
    `workflow.py`'s loop before it was retired in Unit 1d-2.
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
    `proposal_fingerprint` are `TECHNICAL_SPEC.md` §8's interrupt payload;
    `proposal_fingerprint` is always `None` in Unit 2b -- nothing in the
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
