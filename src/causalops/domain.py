"""Typed contracts shared by the workflow, its artifacts, and the reasoning model.

Every contract lives here because they reference each other constantly: a report
holds an assessment, an assessment cites evidence, evidence points at a receipt.
Splitting them would spread one vocabulary across several files.
"""

from collections.abc import Callable
from datetime import datetime
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


class InvestigationState(StrEnum):
    CREATED = "CREATED"
    PLAN_FIRST_CHECK = "PLAN_FIRST_CHECK"
    VALIDATE_FIRST_CHECK = "VALIDATE_FIRST_CHECK"
    EXECUTE_FIRST_CHECK = "EXECUTE_FIRST_CHECK"
    UPDATE_AND_PLAN_SECOND = "UPDATE_AND_PLAN_SECOND"
    VALIDATE_SECOND_CHECK = "VALIDATE_SECOND_CHECK"
    EXECUTE_SECOND_CHECK = "EXECUTE_SECOND_CHECK"
    FINAL_ASSESSMENT = "FINAL_ASSESSMENT"


class Budgets(BaseModel):
    """The limits this step enforces. Provider limits arrive with the Claude model."""

    model_config = ConfigDict(frozen=True)

    wall_clock_seconds: int = 360
    tool_timeout_seconds: int = 10
    model_calls: int = 4
    executed_tools: int = 2
    repairs: int = 1
    log_rows: int = 40


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


class ToolReceipt(BaseModel):
    """What a proposal did, whether or not it was allowed to run."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    receipt_id: str
    incident_id: str
    tool: ToolName
    fingerprint: str
    policy_result: PolicyResult
    outcome: ToolOutcome
    reason_code: ReasonCode | None = None
    requested_at: UtcDatetime
    duration_ms: int = Field(ge=0)
    result_digest: str | None = None
    evidence_id: str | None = None


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
