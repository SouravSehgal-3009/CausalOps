"""The bounded investigation loop from CREATED to one of three terminal states."""

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, JsonValue

from causalops.domain import (
    Budgets,
    CheckOutcome,
    Clock,
    Disposition,
    Evidence,
    FinalAssessment,
    HypothesisUpdate,
    IncidentScope,
    InitialAlertPacket,
    InitialPlan,
    InvestigationReport,
    InvestigationState,
    ModelDisposition,
    ModelUsage,
    PolicyResult,
    ReasonCode,
    RootCauseCode,
    RunCheck,
    ToolOutcome,
    ToolProposal,
    ToolReceipt,
    Versions,
)
from causalops.evidence import EvidenceStore, build_evidence, digest_text, new_opaque_id
from causalops.models import ModelRequest, ReasoningModel, Stage, parse_response
from causalops.policy import POLICY_VERSION, PolicyDecision, authorize
from causalops.prompts import (
    PROMPT_VERSION,
    STAGE_INSTRUCTIONS,
    SYSTEM_TEXT,
    render_context,
)
from causalops.run_records import RunRecorder
from causalops.tools import TOOL_REGISTRY_VERSION

DEFAULT_BUDGETS = Budgets()


def utc_now() -> datetime:
    return datetime.now(UTC)


def add_usage(total: ModelUsage | None, latest: ModelUsage) -> ModelUsage:
    """Usage accumulates across a run; section 11 publishes the total, not the last."""
    if total is None:
        return latest
    return ModelUsage(
        input_tokens=total.input_tokens + latest.input_tokens,
        output_tokens=total.output_tokens + latest.output_tokens,
    )


class BudgetLedger:
    """Counts what the run has spent. A repair spends an ordinary model call."""

    def __init__(self, budgets: Budgets, clock: Clock) -> None:
        self.budgets = budgets
        self.clock = clock
        self.started_at = clock()
        self.model_calls_used = 0
        self.repairs_used = 0
        self.tools_executed = 0

    def model_calls_left(self) -> int:
        return self.budgets.model_calls - self.model_calls_used

    def record_model_call(self) -> None:
        self.model_calls_used += 1

    def may_repair(self) -> bool:
        return self.repairs_used < self.budgets.repairs and self.model_calls_left() > 0

    def record_repair(self) -> None:
        self.repairs_used += 1

    def tools_left(self) -> int:
        return self.budgets.executed_tools - self.tools_executed

    def record_tool_executed(self) -> None:
        self.tools_executed += 1

    def elapsed_seconds(self) -> float:
        return (self.clock() - self.started_at).total_seconds()

    def expired(self) -> bool:
        return self.elapsed_seconds() > self.budgets.wall_clock_seconds


class Investigation:
    """One run of the loop, holding the state the stages share."""

    def __init__(
        self,
        scope: IncidentScope,
        packet: InitialAlertPacket,
        initial_evidence: Sequence[Evidence],
        model: ReasoningModel,
        run_check: RunCheck,
        recorder: RunRecorder,
        budgets: Budgets,
        clock: Clock,
    ) -> None:
        self.scope = scope
        self.packet = packet
        self.model = model
        self.run_check = run_check
        self.recorder = recorder
        self.budgets = budgets
        self.clock = clock
        self.ledger = BudgetLedger(budgets, clock)
        self.store = EvidenceStore(scope.incident_id)
        for record in initial_evidence:
            self.store.add(record)
        self.investigation_id = new_opaque_id()
        self.receipts: list[ToolReceipt] = []
        self.seen_fingerprints: set[str] = set()
        self.invalid_responses = 0
        self.usage: ModelUsage | None = None
        # Every route to failed_safe names its own reason first; this is the value a
        # future route would carry if one ever forgot to.
        self.failure_reason = ReasonCode.MODEL_OUTPUT_INVALID
        self.context_digest = ""
        # Kept alongside the `state` parameter the stage methods already take: that
        # parameter is right for the normal path, and this attribute exists only for
        # the path that has no parameter, where internal_error must name the state a
        # crash escaped from.
        self.state = InvestigationState.CREATED

    def run(self) -> InvestigationReport:
        self.recorder.event(
            InvestigationState.CREATED,
            "investigation_started",
            incident=self.scope.incident_id,
        )
        plan = self.request_stage(
            Stage.INITIAL_PLAN, InitialPlan, InvestigationState.PLAN_FIRST_CHECK
        )
        if plan is None:
            return self.failed_safe()
        if plan.proposal is not None:
            self.consider_check(
                plan.proposal,
                InvestigationState.VALIDATE_FIRST_CHECK,
                InvestigationState.EXECUTE_FIRST_CHECK,
            )
            self.plan_second_check()
        assessment = self.request_stage(
            Stage.FINAL_ASSESSMENT,
            FinalAssessment,
            InvestigationState.FINAL_ASSESSMENT,
        )
        if assessment is None:
            return self.failed_safe()
        cited = assessment.supporting_evidence_ids + assessment.contrary_evidence_ids
        forged = self.store.unknown_ids(cited)
        if forged:
            self.failure_reason = ReasonCode.FORGED_EVIDENCE_REFERENCE
            self.recorder.event(
                InvestigationState.FINAL_ASSESSMENT,
                "forged_citation",
                cited=len(forged),
            )
            return self.failed_safe()
        return self.finish(assessment)

    def plan_second_check(self) -> None:
        """Skip the second check unless a check and the final assessment both fit."""
        if self.ledger.tools_left() <= 0 or self.ledger.model_calls_left() < 2:
            return
        if self.ledger.expired():
            return
        update = self.request_stage(
            Stage.HYPOTHESIS_UPDATE,
            HypothesisUpdate,
            InvestigationState.UPDATE_AND_PLAN_SECOND,
        )
        if update is None or update.proposal is None:
            return
        self.consider_check(
            update.proposal,
            InvestigationState.VALIDATE_SECOND_CHECK,
            InvestigationState.EXECUTE_SECOND_CHECK,
        )

    def request_stage[StageModel: BaseModel](
        self, stage: Stage, schema: type[StageModel], state: InvestigationState
    ) -> StageModel | None:
        self.state = state
        self.recorder.event(state, "stage_started", stage=stage.value)
        if self.ledger.expired():
            self.stop_stage(state, ReasonCode.WALL_CLOCK_EXPIRED)
            return None
        if self.ledger.model_calls_left() <= 0:
            self.stop_stage(state, ReasonCode.MODEL_CALL_BUDGET_EXHAUSTED)
            return None
        parsed, errors = self.ask(schema, stage, state, None)
        if parsed is not None:
            return parsed
        if not self.ledger.may_repair():
            self.stop_stage(state, ReasonCode.REPAIR_EXHAUSTED)
            return None
        self.ledger.record_repair()
        repaired, _ = self.ask(schema, stage, state, errors)
        if repaired is None:
            self.stop_stage(state, ReasonCode.MODEL_OUTPUT_INVALID)
            return None
        return repaired

    def ask[StageModel: BaseModel](
        self,
        schema: type[StageModel],
        stage: Stage,
        state: InvestigationState,
        repair_errors: str | None,
    ) -> tuple[StageModel | None, str]:
        parsed, errors = parse_response(schema, self.call_model(stage, repair_errors))
        if parsed is None:
            self.invalid_responses += 1
            self.recorder.event(state, "invalid_response", stage=stage.value)
        return parsed, errors

    def call_model(
        self, stage: Stage, repair_errors: str | None
    ) -> dict[str, JsonValue]:
        evidence, markers = self.store.context_evidence()
        context = render_context(
            self.packet,
            self.scope,
            evidence,
            markers,
            self.ledger.model_calls_left(),
            self.ledger.tools_left(),
        )
        request = ModelRequest(
            stage=stage,
            system_text=SYSTEM_TEXT,
            context_text=f"{context}\n\n## Task\n{STAGE_INSTRUCTIONS[stage]}",
            repair_errors=repair_errors,
        )
        self.context_digest = digest_text(
            request.system_text + request.context_text + (request.repair_errors or "")
        )
        self.ledger.record_model_call()
        response = self.model.respond(request)
        if response.usage is not None:
            self.usage = add_usage(self.usage, response.usage)
        return response.content

    def stop_stage(self, state: InvestigationState, reason: ReasonCode) -> None:
        self.failure_reason = reason
        self.recorder.event(state, "stage_stopped", reason=reason.value)

    def consider_check(
        self,
        proposal: ToolProposal,
        validate_state: InvestigationState,
        execute_state: InvestigationState,
    ) -> None:
        self.state = validate_state
        self.recorder.event(
            validate_state, "proposal_received", tool=proposal.tool.value
        )
        decision = authorize(
            proposal,
            self.scope,
            self.seen_fingerprints,
            self.budgets,
            self.ledger.tools_left(),
        )
        self.seen_fingerprints.add(decision.fingerprint)
        receipt_id = new_opaque_id()
        requested_at = self.clock()
        if decision.result is PolicyResult.DENIED:
            # A denial is not terminal: the run continues while budget remains.
            self.add_receipt(
                receipt_id=receipt_id,
                proposal=proposal,
                decision=decision,
                outcome=ToolOutcome.NOT_EXECUTED,
                reason_code=decision.reason_code,
                requested_at=requested_at,
            )
            self.recorder.event(
                validate_state,
                "proposal_denied",
                reason=decision.reason_code.value if decision.reason_code else "",
                message=decision.message,
            )
            return
        self.state = execute_state
        self.recorder.event(execute_state, "check_started", tool=proposal.tool.value)
        outcome = self.run_check(proposal, self.scope)
        # An attempt spends the check slot even when it times out or is unavailable.
        self.ledger.record_tool_executed()
        evidence = self.store_outcome(outcome, receipt_id)
        self.add_receipt(
            receipt_id=receipt_id,
            proposal=proposal,
            decision=decision,
            outcome=outcome.outcome,
            reason_code=outcome.reason_code,
            requested_at=requested_at,
            duration_ms=outcome.duration_ms,
            evidence=evidence,
        )
        self.recorder.event(
            execute_state, "check_finished", outcome=outcome.outcome.value
        )

    def store_outcome(self, outcome: CheckOutcome, receipt_id: str) -> Evidence | None:
        if outcome.outcome is not ToolOutcome.EXECUTED:
            return None
        evidence = build_evidence(
            incident_id=self.scope.incident_id,
            kind=outcome.kind,
            source=outcome.source,
            observed_at=self.clock(),
            summary=outcome.summary,
            payload=outcome.payload,
            receipt_id=receipt_id,
        )
        self.store.add(evidence)
        return evidence

    def add_receipt(
        self,
        receipt_id: str,
        proposal: ToolProposal,
        decision: PolicyDecision,
        outcome: ToolOutcome,
        reason_code: ReasonCode | None,
        requested_at: datetime,
        # A denied proposal never runs, so it has neither of these.
        duration_ms: int = 0,
        evidence: Evidence | None = None,
    ) -> None:
        self.receipts.append(
            ToolReceipt(
                receipt_id=receipt_id,
                incident_id=self.scope.incident_id,
                tool=proposal.tool,
                fingerprint=decision.fingerprint,
                policy_result=decision.result,
                outcome=outcome,
                reason_code=reason_code,
                requested_at=requested_at,
                duration_ms=duration_ms,
                result_digest=evidence.content_hash if evidence else None,
                evidence_id=evidence.evidence_id if evidence else None,
            )
        )

    def finish(self, assessment: FinalAssessment) -> InvestigationReport:
        disposition = (
            Disposition.DIAGNOSED
            if assessment.disposition is ModelDisposition.DIAGNOSED
            else Disposition.INSUFFICIENT_EVIDENCE
        )
        return self.report(disposition, assessment.root_cause, assessment, None)

    def failed_safe(self) -> InvestigationReport:
        return self.report(
            Disposition.FAILED_SAFE,
            RootCauseCode.UNDETERMINED,
            None,
            self.failure_reason,
        )

    def internal_error(self, error: Exception) -> InvestigationReport:
        """Turn an unexpected failure into a terminal state instead of a traceback.

        Only the exception class name is recorded. Its text can quote untrusted
        telemetry today and provider content once the Claude model arrives.
        """
        self.failure_reason = ReasonCode.INTERNAL_ERROR
        self.recorder.event(self.state, "internal_error", error=type(error).__name__)
        return self.failed_safe()

    def report(
        self,
        disposition: Disposition,
        root_cause: RootCauseCode,
        assessment: FinalAssessment | None,
        reason_code: ReasonCode | None,
    ) -> InvestigationReport:
        finished_at = self.clock()
        limitations: tuple[str, ...] = ()
        if self.usage is None:
            limitations = ("this model reports no token usage",)
        return InvestigationReport(
            investigation_id=self.investigation_id,
            incident_id=self.scope.incident_id,
            disposition=disposition,
            root_cause=root_cause,
            assessment=assessment,
            reason_code=reason_code,
            budgets=self.budgets,
            versions=Versions(
                prompt_version=PROMPT_VERSION,
                policy_version=POLICY_VERSION,
                tool_registry_version=TOOL_REGISTRY_VERSION,
            ),
            started_at=self.ledger.started_at,
            finished_at=finished_at,
            latency_ms=int(
                (finished_at - self.ledger.started_at).total_seconds() * 1000
            ),
            model_calls_used=self.ledger.model_calls_used,
            repairs_used=self.ledger.repairs_used,
            tools_executed=self.ledger.tools_executed,
            invalid_responses=self.invalid_responses,
            usage=self.usage,
            final_context_digest=self.context_digest,
            evidence_ids=tuple(record.evidence_id for record in self.store.ordered()),
            receipt_ids=tuple(receipt.receipt_id for receipt in self.receipts),
            limitations=limitations,
        )


class InvestigationResult(BaseModel):
    """The report plus the artifacts that belong beside it when it is finalized."""

    model_config = ConfigDict(frozen=True)

    report: InvestigationReport
    evidence: tuple[Evidence, ...]
    receipts: tuple[ToolReceipt, ...]


def run_investigation(
    scope: IncidentScope,
    packet: InitialAlertPacket,
    initial_evidence: Sequence[Evidence],
    model: ReasoningModel,
    run_check: RunCheck,
    recorder: RunRecorder,
    budgets: Budgets = DEFAULT_BUDGETS,
    clock: Clock = utc_now,
) -> InvestigationResult:
    investigation = Investigation(
        scope, packet, initial_evidence, model, run_check, recorder, budgets, clock
    )
    try:
        report = investigation.run()
    except Exception as error:
        report = investigation.internal_error(error)
    return InvestigationResult(
        report=report,
        evidence=investigation.store.ordered(),
        receipts=tuple(investigation.receipts),
    )
