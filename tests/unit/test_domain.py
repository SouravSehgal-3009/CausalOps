import pytest
from fake_incident import (
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    alert_packet,
    hypotheses,
    incident_scope,
    metric_proposal,
    packet_evidence,
)
from pydantic import ValidationError

from causalops.doctor import CheckResult, DoctorReport
from causalops.domain import (
    Budgets,
    Disposition,
    EscalatedInvestigation,
    EscalationReason,
    EscalationRecord,
    Evidence,
    FinalAssessment,
    GraphPhase,
    HypothesisUpdate,
    IncidentScope,
    InitialPlan,
    InvestigationReport,
    ModelDisposition,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RootCauseCode,
    StoredIncident,
    ToolOutcome,
    ToolReceipt,
    Versions,
)
from causalops.tools import ToolName


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
    escalation: EscalationRecord | None = None,
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
        escalation=escalation,
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


@pytest.mark.parametrize("stage", [InitialPlan, HypothesisUpdate])
def test_a_stage_stop_reason_cannot_be_empty(
    stage: type[InitialPlan] | type[HypothesisUpdate],
) -> None:
    with pytest.raises(ValidationError):
        stage(hypotheses=hypotheses(), stop_reason="")


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


def test_stored_incident_refuses_a_packet_incident_id_mismatch() -> None:
    """Post-freeze review, P1. `packet.incident_id` is rendered straight
    into the model's own prompt (`prompts.py`'s `render_context`:
    `f"incident: {packet.incident_id}"`) -- a mismatch against `scope.
    incident_id` would show the model a different incident than the one
    the run's directory, thread, and evidence are actually keyed on,
    with nothing else in the pipeline positioned to catch it before the
    prompt is built. `StoredIncident.check_identity_agrees` now refuses
    this at load time instead. (The `_rebuild_store` double-fault
    correctness traced -- a mismatched `evidence[i].incident_id` raising
    `ValueError` from both the normal report path and the crash-
    containment path meant to catch it, escaping `main()`'s catch tuple
    entirely -- is the sibling evidence-mismatch test's own reason, not
    this one's; a bad `packet.incident_id` never reaches `_rebuild_store`
    at all.)"""
    scope = incident_scope()
    mismatched_packet = alert_packet().model_copy(
        update={"incident_id": "not-the-scope-incident"}
    )

    with pytest.raises(ValidationError, match="packet.incident_id"):
        StoredIncident(
            scope=scope, packet=mismatched_packet, evidence=packet_evidence()
        )


def test_stored_incident_refuses_an_evidence_incident_id_mismatch() -> None:
    """Sibling of the packet-mismatch test above, for the third
    identity-bearing field `StoredIncident.check_identity_agrees` checks --
    a single mismatched evidence record among several is still refused,
    not just a wholesale-wrong tuple. This is the field the packet-
    mismatch test's own docstring points here for: a mismatched
    `evidence[i].incident_id` is what used to reach `graph.py`'s
    `_rebuild_store`, which raises `ValueError` on it from BOTH the
    normal `_build_report` path and the outer crash-containment path
    meant to catch exactly that failure -- escaping `main()`'s
    `(LabError, RunRecordError, CheckpointStoreError)` catch tuple
    entirely (see `TECHNICAL_OVERVIEW.md`'s "Second dual review on
    a44bf57" section for the full trace). `check_identity_agrees` refuses
    this at load time instead, before either graph path ever sees the
    artifact."""
    scope = incident_scope()
    packet = alert_packet()
    one_evidence, other_evidence = packet_evidence()
    mismatched_evidence = (
        one_evidence.model_copy(update={"incident_id": "not-the-scope-incident"}),
        other_evidence,
    )

    with pytest.raises(ValidationError, match="evidence"):
        StoredIncident(scope=scope, packet=packet, evidence=mismatched_evidence)


def test_stored_incident_accepts_a_fully_self_consistent_artifact() -> None:
    """The positive case, deliberately pinned alongside the two refusals
    above -- every fixture this suite already relies on (`_write_incident`
    in `test_approvals.py`/`test_cli.py`, `packet_evidence()` here) must
    keep passing `check_identity_agrees` unchanged, since none of them
    were built with this validator in mind."""
    scope = incident_scope()
    packet = alert_packet()
    evidence = packet_evidence()

    incident = StoredIncident(scope=scope, packet=packet, evidence=evidence)

    assert incident.scope.incident_id == packet.incident_id
    assert all(item.incident_id == scope.incident_id for item in evidence)


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


def test_the_graph_phase_enum_matches_the_seven_phases_in_the_spec() -> None:
    """`TECHNICAL_SPEC.md` §5's diagram, not just what Unit 1a implements --
    `graph.py` (Unit 1b) is the first consumer of this enum."""
    assert [phase.value for phase in GraphPhase] == [
        "CREATED",
        "INVESTIGATE",
        "DISPATCH_TOOL",
        "NORMALIZE_EVIDENCE",
        "FINAL_ASSESSMENT",
        "ESCALATION_INTERRUPT",
        "FINAL_REPORT",
    ]


def test_the_escalation_reason_enum_matches_the_four_triggers_in_the_spec() -> None:
    """`TECHNICAL_SPEC.md` §8's four deterministic escalation triggers, named
    in the order the spec lists them. `graph.py`'s `_escalation_reason` does
    *not* check them in this same order -- it checks `TOOL_UNAVAILABLE`
    first, deliberately ahead of the spec's own listing order, and says why
    in its own docstring. `RETRIEVAL_COVERAGE_INSUFFICIENT` is unreachable
    until Milestone 3's retrieval lands; it is named here anyway, the same
    precedent `GraphPhase` above already sets."""
    assert [reason.value for reason in EscalationReason] == [
        "CONFLICTING_EVIDENCE",
        "TOOL_UNAVAILABLE",
        "INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING",
        "RETRIEVAL_COVERAGE_INSUFFICIENT",
    ]


def test_a_report_carries_no_escalation_record_unless_the_run_escalated() -> None:
    """The default an ordinary, never-escalated run gets -- every report
    before Unit 2b, and most reports after it."""
    diagnosed = report(Disposition.DIAGNOSED, RootCauseCode.CONFIG_CHANGE, assessment())

    assert diagnosed.escalation is None


def test_an_escalation_record_survives_on_a_report_without_disturbing_it() -> None:
    """Adding `escalation` is additive: it does not interact with
    `check_terminal_invariants`, so a diagnosed report carrying one still
    validates, and the disposition/root_cause it names are unaffected by
    what the owner decided about it."""
    decided = report(
        Disposition.DIAGNOSED,
        RootCauseCode.CONFIG_CHANGE,
        assessment(),
        escalation=EscalationRecord(
            reason=EscalationReason.CONFLICTING_EVIDENCE,
            decision="reject",
            rejection_note="the two remaining causes were never separated",
        ),
    )

    assert decided.disposition is Disposition.DIAGNOSED
    assert decided.escalation is not None
    assert decided.escalation.reason is EscalationReason.CONFLICTING_EVIDENCE
    assert decided.escalation.decision == "reject"
    assert (
        decided.escalation.rejection_note
        == "the two remaining causes were never separated"
    )


def test_an_escalation_record_only_accepts_the_two_named_decisions() -> None:
    with pytest.raises(ValidationError):
        EscalationRecord(
            reason=EscalationReason.TOOL_UNAVAILABLE,
            decision="approve",  # type: ignore[arg-type]
        )


def test_a_rejection_without_a_note_is_refused() -> None:
    """`check_rejection_note_pairing`'s own reject-side check, exercised
    directly on `EscalationRecord` -- the same pairing rule
    `causalops.approvals.OwnerDecision` and `graph.py`'s
    `_parse_resume_decision` also enforce, at the two other points a
    caller could reach this model from."""
    with pytest.raises(ValidationError):
        EscalationRecord(reason=EscalationReason.TOOL_UNAVAILABLE, decision="reject")


def test_a_whitespace_only_rejection_note_is_refused() -> None:
    """A caller that reaches `EscalationRecord` directly, bypassing both
    `OwnerDecision`'s and `_parse_resume_decision`'s own whitespace
    stripping, must still be refused here -- a whitespace-only note is not
    content, and this model's own validator must not be the weak link that
    lets `report.py` render a blank-looking "- Owner's note:" line."""
    with pytest.raises(ValidationError):
        EscalationRecord(
            reason=EscalationReason.TOOL_UNAVAILABLE,
            decision="reject",
            rejection_note="   ",
        )


def test_an_acceptance_with_a_rejection_note_is_refused() -> None:
    """`check_rejection_note_pairing`'s accept-side check: a note has no
    meaning on a decision that was never rejected."""
    with pytest.raises(ValidationError):
        EscalationRecord(
            reason=EscalationReason.TOOL_UNAVAILABLE,
            decision="accept",
            rejection_note="should not be allowed here",
        )


def test_an_escalated_investigation_has_no_report_field() -> None:
    """The type-level proof this is a sibling to `InvestigationResult`, not
    a variant of it: a paused run's disposition is not resolved, so there is
    no `InvestigationReport` to carry, and this model does not pretend
    otherwise with an `Optional[report]`."""
    assert "report" not in EscalatedInvestigation.model_fields


def receipt(**overrides: object) -> ToolReceipt:
    fields: dict[str, object] = {
        "receipt_id": "receipt-1",
        "incident_id": "inc-1",
        "tool": ToolName.QUERY_LOGS,
        "fingerprint": "f" * 8,
        "policy_result": PolicyResult.ALLOWED,
        "state": ReceiptState.RESERVED,
        "requested_at": WINDOW_START,
        "duration_ms": 0,
    }
    fields.update(overrides)
    return ToolReceipt(**fields)  # type: ignore[arg-type]


def test_a_receipt_defaults_to_settled() -> None:
    receipt = ToolReceipt(
        receipt_id="receipt-1",
        incident_id="inc-1",
        tool=ToolName.QUERY_LOGS,
        fingerprint="f" * 8,
        policy_result=PolicyResult.ALLOWED,
        outcome=ToolOutcome.EXECUTED,
        requested_at=WINDOW_START,
        duration_ms=5,
    )

    assert receipt.state is ReceiptState.SETTLED


def test_a_reserved_receipt_with_no_result_yet_is_valid() -> None:
    built = receipt()

    assert built.state is ReceiptState.RESERVED
    assert built.outcome is None


def test_a_reserved_receipt_cannot_carry_an_outcome() -> None:
    with pytest.raises(ValidationError):
        receipt(outcome=ToolOutcome.EXECUTED)


def test_a_reserved_receipt_cannot_carry_a_result_digest() -> None:
    with pytest.raises(ValidationError):
        receipt(result_digest="digest")


def test_a_reserved_receipt_cannot_carry_an_evidence_id() -> None:
    with pytest.raises(ValidationError):
        receipt(evidence_id="evidence-1")


def test_a_settled_receipt_must_carry_an_outcome() -> None:
    with pytest.raises(ValidationError):
        receipt(state=ReceiptState.SETTLED)
