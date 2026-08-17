import pytest
from fake_incident import (
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    alert_packet,
    hypotheses,
    incident_scope,
    metric_proposal,
)
from pydantic import ValidationError

from causalops.doctor import CheckResult, DoctorReport
from causalops.domain import (
    Budgets,
    Disposition,
    Evidence,
    FinalAssessment,
    IncidentScope,
    InitialPlan,
    InvestigationReport,
    ModelDisposition,
    ReasonCode,
    RootCauseCode,
    ToolReceipt,
    Versions,
)


def assessment(
    disposition: ModelDisposition = ModelDisposition.DIAGNOSED,
    root_cause: RootCauseCode = RootCauseCode.CONFIG_CHANGE,
    supporting: tuple[str, ...] = (SYMPTOM_EVIDENCE_ID,),
) -> FinalAssessment:
    return FinalAssessment(
        disposition=disposition,
        root_cause=root_cause,
        supporting_evidence_ids=supporting,
        uncertainty="the change and the pool both fit the timing",
        next_step="confirm the change with the owner",
    )


def report(
    disposition: Disposition,
    root_cause: RootCauseCode,
    final: FinalAssessment | None,
    reason_code: ReasonCode | None = None,
) -> InvestigationReport:
    return InvestigationReport(
        investigation_id="inv-1",
        incident_id="inc-1",
        disposition=disposition,
        root_cause=root_cause,
        assessment=final,
        reason_code=reason_code,
        budgets=Budgets(),
        versions=Versions(
            prompt_version="1", policy_version="1", tool_registry_version="1"
        ),
        started_at=WINDOW_START,
        finished_at=WINDOW_END,
        latency_ms=10,
        model_calls_used=3,
        repairs_used=0,
        tools_executed=1,
        invalid_responses=0,
        final_context_digest="digest",
    )


def test_a_diagnosis_needs_a_named_cause_and_a_citation() -> None:
    assert assessment().disposition is ModelDisposition.DIAGNOSED

    with pytest.raises(ValidationError):
        assessment(root_cause=RootCauseCode.UNDETERMINED)
    with pytest.raises(ValidationError):
        assessment(supporting=())


def test_an_abstention_requires_undetermined() -> None:
    abstention = assessment(
        disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
        root_cause=RootCauseCode.UNDETERMINED,
    )

    assert abstention.root_cause is RootCauseCode.UNDETERMINED

    with pytest.raises(ValidationError):
        assessment(
            disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
            root_cause=RootCauseCode.CONFIG_CHANGE,
        )


def test_the_model_schema_cannot_express_failed_safe() -> None:
    schema = FinalAssessment.model_json_schema()

    choices = schema["$defs"]["ModelDisposition"]["enum"]
    assert choices == ["DIAGNOSED", "INSUFFICIENT_EVIDENCE"]
    assert "FAILED_SAFE" not in choices


def test_a_report_pairs_each_disposition_with_its_only_valid_cause() -> None:
    diagnosed = report(Disposition.DIAGNOSED, RootCauseCode.CONFIG_CHANGE, assessment())
    assert diagnosed.disposition is Disposition.DIAGNOSED

    with pytest.raises(ValidationError):
        report(Disposition.DIAGNOSED, RootCauseCode.UNDETERMINED, assessment())
    with pytest.raises(ValidationError):
        report(Disposition.DIAGNOSED, RootCauseCode.CONFIG_CHANGE, None)
    with pytest.raises(ValidationError):
        report(Disposition.INSUFFICIENT_EVIDENCE, RootCauseCode.CONFIG_CHANGE, None)
    with pytest.raises(ValidationError):
        report(Disposition.INSUFFICIENT_EVIDENCE, RootCauseCode.UNDETERMINED, None)


def test_failed_safe_carries_no_model_assessment() -> None:
    safe = report(
        Disposition.FAILED_SAFE,
        RootCauseCode.UNDETERMINED,
        None,
        ReasonCode.MODEL_OUTPUT_INVALID,
    )

    assert safe.reason_code is ReasonCode.MODEL_OUTPUT_INVALID

    with pytest.raises(ValidationError):
        report(
            Disposition.FAILED_SAFE,
            RootCauseCode.UNDETERMINED,
            assessment(),
            ReasonCode.MODEL_OUTPUT_INVALID,
        )
    with pytest.raises(ValidationError):
        report(Disposition.FAILED_SAFE, RootCauseCode.UNDETERMINED, None, None)


def test_a_stage_proposes_a_check_or_stops_but_not_both() -> None:
    assert InitialPlan(hypotheses=hypotheses(), proposal=metric_proposal()).proposal

    with pytest.raises(ValidationError):
        InitialPlan(hypotheses=hypotheses())
    with pytest.raises(ValidationError):
        InitialPlan(
            hypotheses=hypotheses(), proposal=metric_proposal(), stop_reason="both"
        )


def test_a_plan_keeps_two_or_three_hypotheses() -> None:
    with pytest.raises(ValidationError):
        InitialPlan(hypotheses=hypotheses()[:1], stop_reason="only one cause")


def test_an_incident_window_must_end_after_it_starts() -> None:
    with pytest.raises(ValidationError):
        IncidentScope(
            incident_id="inc-1",
            services=("gateway",),
            started_at=WINDOW_END,
            ended_at=WINDOW_START,
            endpoint="/api/orders",
        )


def test_contracts_are_frozen() -> None:
    scope = incident_scope()

    with pytest.raises(ValidationError):
        scope.incident_id = "another"


def test_persisted_and_model_facing_contracts_carry_a_schema_version() -> None:
    """Section 6 as amended: version what is stored or exchanged, nothing else."""
    for versioned in (Evidence, ToolReceipt, InvestigationReport, FinalAssessment):
        assert "schema_version" in versioned.model_fields

    for transient in (CheckResult, DoctorReport):
        assert "schema_version" not in transient.model_fields


def test_the_alert_packet_names_its_own_initial_evidence() -> None:
    packet = alert_packet()

    assert packet.symptom_evidence_id == SYMPTOM_EVIDENCE_ID
    assert packet.topology_evidence_id != packet.symptom_evidence_id
