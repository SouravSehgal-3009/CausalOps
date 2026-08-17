from datetime import timedelta

from fake_incident import (
    INCIDENT_ID,
    WINDOW_END,
    WINDOW_START,
    packet_evidence,
)
from pydantic import JsonValue

from causalops.domain import (
    Budgets,
    Disposition,
    Evidence,
    EvidenceKind,
    FinalAssessment,
    InvestigationReport,
    ModelDisposition,
    ModelUsage,
    PolicyResult,
    ReasonCode,
    RootCauseCode,
    ToolOutcome,
    ToolReceipt,
    Versions,
)
from causalops.evaluation import (
    SCORER_VERSION,
    EvaluationRecord,
    ExpectedOutcome,
    PredicateOperator,
    RequiredEvidencePredicate,
    satisfies,
    score_run,
)
from causalops.evidence import build_evidence
from causalops.tools import ToolName

TIMEOUT_PAYLOAD: dict[str, JsonValue] = {
    "template": "downstream_timeout_rate",
    "timeouts_per_minute": 12,
    "note": "inventory timed out repeatedly",
}


def timeout_evidence() -> Evidence:
    return build_evidence(
        incident_id=INCIDENT_ID,
        kind=EvidenceKind.METRIC,
        source="query_metric",
        observed_at=WINDOW_START + timedelta(minutes=5),
        summary="inventory timeouts rose",
        payload=TIMEOUT_PAYLOAD,
    )


def timeout_predicate(
    field: str = "timeouts_per_minute",
    operator: PredicateOperator = PredicateOperator.AT_LEAST,
    value: JsonValue = 10,
) -> RequiredEvidencePredicate:
    return RequiredEvidencePredicate(
        source="query_metric",
        kind=EvidenceKind.METRIC,
        template="downstream_timeout_rate",
        field=field,
        operator=operator,
        value=value,
    )


def receipt(
    policy_result: PolicyResult = PolicyResult.ALLOWED,
    outcome: ToolOutcome = ToolOutcome.EXECUTED,
    reason_code: ReasonCode | None = None,
) -> ToolReceipt:
    return ToolReceipt(
        receipt_id="receipt-1",
        incident_id=INCIDENT_ID,
        tool=ToolName.QUERY_METRIC,
        fingerprint="fingerprint",
        policy_result=policy_result,
        outcome=outcome,
        reason_code=reason_code,
        requested_at=WINDOW_START,
        duration_ms=5,
    )


def diagnosed_report(cited: tuple[str, ...]) -> InvestigationReport:
    return InvestigationReport(
        investigation_id="inv-1",
        incident_id=INCIDENT_ID,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
        assessment=FinalAssessment(
            disposition=ModelDisposition.DIAGNOSED,
            root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
            supporting_evidence_ids=cited,
            uncertainty="the pool was never ruled out completely",
            next_step="raise the inventory timeout and watch the gateway",
        ),
        budgets=Budgets(),
        versions=Versions(
            prompt_version="1", policy_version="1", tool_registry_version="1"
        ),
        started_at=WINDOW_START,
        finished_at=WINDOW_END,
        latency_ms=1800,
        model_calls_used=3,
        repairs_used=0,
        tools_executed=2,
        invalid_responses=1,
        usage=ModelUsage(input_tokens=900, output_tokens=300),
        final_context_digest="digest",
    )


def expected_diagnosis() -> ExpectedOutcome:
    return ExpectedOutcome(
        root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
        disposition=Disposition.DIAGNOSED,
        predicates=(timeout_predicate(),),
    )


def test_a_predicate_matches_the_observation_it_describes() -> None:
    evidence = timeout_evidence()

    assert satisfies(timeout_predicate(), evidence)
    assert satisfies(timeout_predicate(value=12), evidence)
    assert not satisfies(timeout_predicate(value=20), evidence)


def test_a_predicate_checks_source_kind_and_template() -> None:
    evidence = timeout_evidence()

    assert not satisfies(
        timeout_predicate().model_copy(update={"source": "query_logs"}), evidence
    )
    assert not satisfies(
        timeout_predicate().model_copy(update={"kind": EvidenceKind.LOG}), evidence
    )
    assert not satisfies(
        timeout_predicate().model_copy(update={"template": "gateway_error_rate"}),
        evidence,
    )
    assert not satisfies(timeout_predicate(field="missing_field"), evidence)


def test_equals_and_contains_operators() -> None:
    evidence = timeout_evidence()

    assert satisfies(
        timeout_predicate(
            field="template",
            operator=PredicateOperator.EQUALS,
            value="downstream_timeout_rate",
        ),
        evidence,
    )
    assert satisfies(
        timeout_predicate(
            field="note", operator=PredicateOperator.CONTAINS, value="timed out"
        ),
        evidence,
    )
    assert not satisfies(
        timeout_predicate(
            field="note", operator=PredicateOperator.CONTAINS, value="pool exhausted"
        ),
        evidence,
    )


def test_a_correct_cited_diagnosis_scores_on_every_count() -> None:
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))

    scores = score_run(report, [evidence], [receipt()], expected_diagnosis())

    assert scores.diagnosis_correct
    assert scores.disposition_correct
    assert scores.citations_valid
    assert scores.citations_sufficient


def test_citing_evidence_that_misses_the_required_fact_is_insufficient() -> None:
    weak = packet_evidence()[0]
    report = diagnosed_report((weak.evidence_id,))

    scores = score_run(report, [weak], [receipt()], expected_diagnosis())

    assert scores.citations_valid
    assert not scores.citations_sufficient


def test_a_citation_from_another_incident_is_invalid() -> None:
    stranger = timeout_evidence().model_copy(update={"incident_id": "other-incident"})
    report = diagnosed_report((stranger.evidence_id,))

    scores = score_run(report, [stranger], [receipt()], expected_diagnosis())

    assert not scores.citations_valid


def test_a_wrong_cause_or_disposition_scores_false() -> None:
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    expected = expected_diagnosis().model_copy(
        update={
            "root_cause": RootCauseCode.CONFIG_CHANGE,
            "disposition": Disposition.INSUFFICIENT_EVIDENCE,
        }
    )

    scores = score_run(report, [evidence], [receipt()], expected)

    assert not scores.diagnosis_correct
    assert not scores.disposition_correct


def test_control_behaviour_counts_denials_by_kind() -> None:
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    receipts = [
        receipt(),
        receipt(
            policy_result=PolicyResult.DENIED,
            outcome=ToolOutcome.NOT_EXECUTED,
            reason_code=ReasonCode.DUPLICATE_PROPOSAL,
        ),
        receipt(
            policy_result=PolicyResult.DENIED,
            outcome=ToolOutcome.NOT_EXECUTED,
            reason_code=ReasonCode.UNKNOWN_SERVICE,
        ),
    ]

    control = score_run(report, [evidence], receipts, expected_diagnosis()).control

    assert control.denied == 2
    assert control.duplicate == 1
    assert control.out_of_scope == 1
    assert control.invalid_responses == 1


def test_efficiency_reports_what_the_run_actually_spent() -> None:
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))

    efficiency = score_run(
        report, [evidence], [receipt()], expected_diagnosis()
    ).efficiency

    assert efficiency.latency_ms == 1800
    assert efficiency.model_calls == 3
    assert efficiency.tools_executed == 2
    assert efficiency.input_tokens == 900
    assert efficiency.output_tokens == 300


def test_an_evaluation_record_keeps_the_expected_outcome_beside_the_scores() -> None:
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    scores = score_run(report, [evidence], [receipt()], expected_diagnosis())

    record = EvaluationRecord(
        run_key="evaluation-1/causalops/1",
        investigation_id=report.investigation_id,
        incident_id=report.incident_id,
        expected=expected_diagnosis(),
        scores=scores,
    )
    restored = EvaluationRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.scorer_version == SCORER_VERSION
    assert restored.expected.predicates[0].field == "timeouts_per_minute"


def test_a_run_with_no_assessment_cites_nothing() -> None:
    evidence = timeout_evidence()
    failed = diagnosed_report((evidence.evidence_id,)).model_copy(
        update={
            "disposition": Disposition.FAILED_SAFE,
            "root_cause": RootCauseCode.UNDETERMINED,
            "assessment": None,
            "reason_code": ReasonCode.MODEL_OUTPUT_INVALID,
        }
    )

    scores = score_run(failed, [evidence], [receipt()], expected_diagnosis())

    assert not scores.citations_valid
    assert not scores.citations_sufficient
