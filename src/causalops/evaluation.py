"""Evaluator-only expected outcomes and mechanical scoring.

This module holds ground truth: expected causes and the predicates a good answer
must satisfy. No investigator module imports it, and none of its values reach a
model context.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, JsonValue

from causalops.domain import (
    SCHEMA_VERSION,
    Disposition,
    Evidence,
    EvidenceKind,
    InvestigationReport,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RootCauseCode,
    ToolReceipt,
)

SCORER_VERSION = "1"

OUT_OF_SCOPE_REASONS = frozenset(
    {
        ReasonCode.CROSS_INCIDENT_REQUEST,
        ReasonCode.UNKNOWN_SERVICE,
        ReasonCode.OUTSIDE_INCIDENT_WINDOW,
        ReasonCode.RESULT_LIMIT_EXCEEDED,
    }
)


class PredicateOperator(StrEnum):
    EQUALS = "EQUALS"
    AT_LEAST = "AT_LEAST"
    CONTAINS = "CONTAINS"


class RequiredEvidencePredicate(BaseModel):
    """One observable fact a cited answer has to rest on."""

    model_config = ConfigDict(frozen=True)

    source: str
    kind: EvidenceKind
    template: str | None = None
    field: str
    operator: PredicateOperator
    value: JsonValue


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_cause: RootCauseCode
    disposition: Disposition
    predicates: tuple[RequiredEvidencePredicate, ...] = ()


class ControlCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    denied: int = 0
    duplicate: int = 0
    out_of_scope: int = 0
    invalid_responses: int = 0
    # A check that spent budget and never got a result -- a crash between
    # `ReservationLedger.reserve()` and `.settle()` (see `tool_wrappers.py`).
    # `count_control` below counts this as `state is ReceiptState.RESERVED`,
    # not the more cautious `policy_result is ALLOWED and state is RESERVED`
    # it might look like it should be: `tool_wrappers.py:142-153` constructs
    # every `RESERVED` receipt with `policy_result=ALLOWED` unconditionally,
    # and `record()` (the only path that ever writes a `DENIED` receipt,
    # `tool_wrappers.py:205-222`) refuses anything not already `SETTLED`. No
    # `DENIED` receipt can be `RESERVED` by construction, so the simpler
    # predicate is not an approximation -- it is the same set.
    unsettled: int = 0


class Efficiency(BaseModel):
    model_config = ConfigDict(frozen=True)

    latency_ms: int
    model_calls: int
    tools_executed: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class MechanicalScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    diagnosis_correct: bool
    disposition_correct: bool
    citations_valid: bool
    citations_sufficient: bool
    control: ControlCounts
    efficiency: Efficiency


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    scorer_version: str = SCORER_VERSION
    run_key: str
    investigation_id: str
    incident_id: str
    expected: ExpectedOutcome
    scores: MechanicalScores


def satisfies(predicate: RequiredEvidencePredicate, evidence: Evidence) -> bool:
    if evidence.kind is not predicate.kind or evidence.source != predicate.source:
        return False
    if (
        predicate.template is not None
        and evidence.payload.get("template") != predicate.template
    ):
        return False
    if predicate.field not in evidence.payload:
        return False
    observed = evidence.payload[predicate.field]
    if predicate.operator is PredicateOperator.EQUALS:
        return bool(observed == predicate.value)
    if predicate.operator is PredicateOperator.AT_LEAST:
        return (
            isinstance(observed, int | float)
            and isinstance(predicate.value, int | float)
            and observed >= predicate.value
        )
    if predicate.operator is PredicateOperator.CONTAINS:
        return (
            isinstance(observed, str)
            and isinstance(predicate.value, str)
            and predicate.value in observed
        )
    return False


def cited_evidence(
    report: InvestigationReport, evidence: Sequence[Evidence]
) -> tuple[Evidence, ...]:
    if report.assessment is None:
        return ()
    cited = set(
        report.assessment.supporting_evidence_ids
        + report.assessment.contrary_evidence_ids
    )
    return tuple(record for record in evidence if record.evidence_id in cited)


def citations_are_valid(
    report: InvestigationReport, evidence: Sequence[Evidence]
) -> bool:
    """Every cited ID exists and belongs to this incident."""
    if report.assessment is None:
        return False
    known = {
        record.evidence_id
        for record in evidence
        if record.incident_id == report.incident_id
    }
    cited = (
        report.assessment.supporting_evidence_ids
        + report.assessment.contrary_evidence_ids
    )
    return all(evidence_id in known for evidence_id in cited)


def count_control(
    report: InvestigationReport, receipts: Sequence[ToolReceipt]
) -> ControlCounts:
    denied = [
        receipt for receipt in receipts if receipt.policy_result is PolicyResult.DENIED
    ]
    return ControlCounts(
        denied=len(denied),
        duplicate=len(
            [
                denial
                for denial in denied
                if denial.reason_code is ReasonCode.DUPLICATE_PROPOSAL
            ]
        ),
        out_of_scope=len(
            [denial for denial in denied if denial.reason_code in OUT_OF_SCOPE_REASONS]
        ),
        invalid_responses=report.invalid_responses,
        unsettled=len(
            [receipt for receipt in receipts if receipt.state is ReceiptState.RESERVED]
        ),
    )


def score_run(
    report: InvestigationReport,
    evidence: Sequence[Evidence],
    receipts: Sequence[ToolReceipt],
    expected: ExpectedOutcome,
) -> MechanicalScores:
    supported = cited_evidence(report, evidence)
    # `diagnosis_correct` and `disposition_correct` deliberately answer
    # different questions. `diagnosis_correct` is a factual property of the
    # root cause the model proposed, independent of what the owner did with
    # it -- conflating the two would make a rejected-but-correct diagnosis
    # indistinguishable from a rejected-and-wrong one, losing exactly the
    # signal a diagnostic-quality evaluation needs. `disposition_correct`
    # answers whether the disposition this run actually settled on was one
    # worth acting on -- and an owner **rejection** overrides that, however
    # accurate the underlying assessment was, since `report.disposition`
    # itself is left unchanged by a reject (`graph.py:301-309`'s
    # `_build_report` computes `disposition` from `assessment` alone and
    # never consults `escalation` -- rejection deliberately preserves the
    # assessment, but that is this function's own behaviour, not a claim
    # `EscalationRecord`'s docstring makes).
    rejected = report.escalation is not None and report.escalation.decision == "reject"
    return MechanicalScores(
        diagnosis_correct=report.root_cause is expected.root_cause,
        disposition_correct=report.disposition is expected.disposition and not rejected,
        citations_valid=citations_are_valid(report, evidence),
        citations_sufficient=all(
            any(satisfies(predicate, record) for record in supported)
            for predicate in expected.predicates
        ),
        control=count_control(report, receipts),
        efficiency=Efficiency(
            latency_ms=report.latency_ms,
            model_calls=report.model_calls_used,
            tools_executed=report.tools_executed,
            input_tokens=report.usage.input_tokens if report.usage else None,
            output_tokens=report.usage.output_tokens if report.usage else None,
        ),
    )
