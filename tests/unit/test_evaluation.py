import json
from datetime import timedelta
from pathlib import Path

import pytest
from fake_incident import (
    INCIDENT_ID,
    WINDOW_END,
    WINDOW_START,
    packet_evidence,
)
from pydantic import JsonValue, ValidationError

from causalops.domain import (
    Budgets,
    Disposition,
    EscalationReason,
    EscalationRecord,
    Evidence,
    EvidenceKind,
    FinalAssessment,
    InvestigationReport,
    ModelDisposition,
    ModelUsage,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RetrievalMode,
    RootCauseCode,
    ToolOutcome,
    ToolReceipt,
    Versions,
)
from causalops.evaluation import (
    SCORER_VERSION,
    ControlCounts,
    Efficiency,
    EvaluationRecord,
    ExpectedOutcome,
    MechanicalScores,
    PredicateOperator,
    RequiredEvidencePredicate,
    satisfies,
    score_run,
    summarize_evaluation,
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


def reserved_receipt() -> ToolReceipt:
    """A check that spent budget and never got a result -- the same
    lifecycle `tool_wrappers.py`'s `ReservationLedger.reserve()` leaves
    behind when a crash lands between reserving and settling."""
    return ToolReceipt(
        receipt_id="receipt-reserved",
        incident_id=INCIDENT_ID,
        tool=ToolName.QUERY_METRIC,
        fingerprint="fingerprint-reserved",
        policy_result=PolicyResult.ALLOWED,
        state=ReceiptState.RESERVED,
        requested_at=WINDOW_START,
        duration_ms=0,
    )


def diagnosed_report(
    cited: tuple[str, ...], escalation: EscalationRecord | None = None
) -> InvestigationReport:
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
        escalation=escalation,
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
    assert scores.correct_and_grounded


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
    # The bug this assertion exists for: `cited_evidence` used to match
    # purely on `evidence_id`, with no filter on `record.incident_id` --
    # unlike `citations_are_valid`'s own `known` set, which already
    # filtered this way. The stranger's payload does satisfy
    # `expected_diagnosis()`'s predicate content-wise, so before the fix
    # this scored `citations_sufficient=True` off a citation the line above
    # already proves is invalid -- a cross-incident record must not be able
    # to satisfy sufficiency just because `citations_valid` alone caught it.
    assert not scores.citations_sufficient


def test_a_predicate_matched_only_by_contrary_evidence_is_insufficient() -> None:
    """The bug this test exists for: `cited_evidence` used to combine
    `supporting_evidence_ids` and `contrary_evidence_ids` into one set, so a
    diagnosis whose only predicate-matching evidence was filed as CONTRARY
    (evidence the model itself said argued against its own diagnosis, not
    for it) still scored `citations_sufficient=True`. That is backwards --
    sufficiency is supposed to verify the diagnosis rests on real supporting
    evidence, not evidence the model flagged as working against it.
    """
    matching = timeout_evidence()
    unrelated_but_real = packet_evidence()[0]
    report = diagnosed_report((unrelated_but_real.evidence_id,))
    assert report.assessment is not None
    report = report.model_copy(
        update={
            "assessment": report.assessment.model_copy(
                update={"contrary_evidence_ids": (matching.evidence_id,)}
            )
        }
    )

    scores = score_run(
        report, [matching, unrelated_but_real], [receipt()], expected_diagnosis()
    )

    # The predicate-matching record is a real, same-incident citation, so
    # validity is unaffected -- only sufficiency, which asks whether the
    # diagnosis rests on genuine SUPPORTING evidence, must go False.
    assert scores.citations_valid
    assert not scores.citations_sufficient


def test_no_predicate_family_is_not_applicable_on_a_correct_diagnosis() -> None:
    """The fix this test exists for: `all(())` is `True` in Python, so a
    family declaring no required-evidence predicate at all used to score
    `citations_sufficient=True` unconditionally -- `None` is the honest
    value instead: there was no predicate to satisfy, so this is not a
    signal about the citations at all. Every family in this corpus declares
    at least one predicate today; this test uses a synthetic
    empty-predicates `ExpectedOutcome` to keep covering the no-predicate
    case regardless."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    expected = expected_diagnosis().model_copy(update={"predicates": ()})

    scores = score_run(report, [evidence], [receipt()], expected)

    assert scores.diagnosis_correct
    assert scores.citations_sufficient is None
    assert scores.correct_and_grounded is None


def test_no_predicate_family_is_not_applicable_on_a_wrong_diagnosis() -> None:
    """The exact defect shape from the real saved run this fix responds to:
    a wrong diagnosis against a no-predicate family must NOT score
    `citations_sufficient=True` just because there was nothing to fail --
    that is the vacuous-truth bug. It must also not score `False`, which
    would penalize the family for a requirement it was never given."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    expected = expected_diagnosis().model_copy(
        update={"root_cause": RootCauseCode.CONFIG_CHANGE, "predicates": ()}
    )

    scores = score_run(report, [evidence], [receipt()], expected)

    assert not scores.diagnosis_correct
    assert scores.citations_sufficient is None
    assert scores.correct_and_grounded is None


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


def test_correct_and_grounded_requires_both_a_right_answer_and_grounded_citations() -> (
    None
):
    """The full truth table for `correct_and_grounded`. The non-obvious
    case (`diagnosis_correct=
    True`, `citations_sufficient=False` -> `correct_and_grounded=False`) is
    the entire reason this field exists apart from either input alone: a
    right answer resting on citations that don't actually satisfy the
    required predicate must not count as grounded."""
    evidence = timeout_evidence()

    # diagnosis_correct=True, citations_sufficient=True -> True.
    right_and_grounded = score_run(
        diagnosed_report((evidence.evidence_id,)),
        [evidence],
        [receipt()],
        expected_diagnosis(),
    )
    assert right_and_grounded.diagnosis_correct
    assert right_and_grounded.citations_sufficient
    assert right_and_grounded.correct_and_grounded is True

    # diagnosis_correct=True, citations_sufficient=False -> False.
    weak = packet_evidence()[0]
    right_but_ungrounded = score_run(
        diagnosed_report((weak.evidence_id,)),
        [weak],
        [receipt()],
        expected_diagnosis(),
    )
    assert right_but_ungrounded.diagnosis_correct
    assert not right_but_ungrounded.citations_sufficient
    assert right_but_ungrounded.correct_and_grounded is False

    # diagnosis_correct=False, citations_sufficient=True -> False.
    expected_other_cause = expected_diagnosis().model_copy(
        update={"root_cause": RootCauseCode.CONFIG_CHANGE}
    )
    wrong_but_grounded = score_run(
        diagnosed_report((evidence.evidence_id,)),
        [evidence],
        [receipt()],
        expected_other_cause,
    )
    assert not wrong_but_grounded.diagnosis_correct
    assert wrong_but_grounded.citations_sufficient
    assert wrong_but_grounded.correct_and_grounded is False

    # diagnosis_correct=False, citations_sufficient=False -> False.
    wrong_and_ungrounded = score_run(
        diagnosed_report((weak.evidence_id,)),
        [weak],
        [receipt()],
        expected_other_cause,
    )
    assert not wrong_and_ungrounded.diagnosis_correct
    assert not wrong_and_ungrounded.citations_sufficient
    assert wrong_and_ungrounded.correct_and_grounded is False


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


def test_control_counts_a_reserved_receipt_as_unsettled() -> None:
    """A check that spent budget and crashed before settling must be visible
    to the scorer -- before this field existed, `ControlCounts` had no field
    that could hold it at all."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    receipts = [receipt(), reserved_receipt()]

    control = score_run(report, [evidence], receipts, expected_diagnosis()).control

    assert control.unsettled == 1
    # Not a denial and not the settled receipt's own budget line -- the
    # reserved receipt is invisible to every other counter.
    assert control.denied == 0
    assert control.duplicate == 0
    assert control.out_of_scope == 0


def test_a_run_with_no_reserved_receipts_counts_zero_unsettled() -> None:
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))

    control = score_run(report, [evidence], [receipt()], expected_diagnosis()).control

    assert control.unsettled == 0


def test_an_owner_rejected_diagnosis_does_not_score_as_correct_disposition() -> None:
    """`report.disposition` stays `DIAGNOSED` on a reject -- rejection
    deliberately preserves the assessment, since `graph.py:301-309`'s
    `_build_report` computes `disposition` from `assessment` alone and never
    consults `escalation` -- so `disposition_correct` must read
    `report.escalation` directly, not just `report.disposition`, or an
    owner-rejected diagnosis would score as correct."""
    evidence = timeout_evidence()
    report = diagnosed_report(
        (evidence.evidence_id,),
        escalation=EscalationRecord(
            reason=EscalationReason.CONFLICTING_EVIDENCE,
            decision="reject",
            rejection_note="the citation looks wrong",
        ),
    )

    scores = score_run(report, [evidence], [receipt()], expected_diagnosis())

    assert not scores.disposition_correct
    # `diagnosis_correct` answers a different question -- whether the root
    # cause the model proposed matches ground truth -- and stays true
    # regardless of what the owner decided to do with it.
    assert scores.diagnosis_correct


def test_an_owner_accepted_diagnosis_still_scores_as_correct() -> None:
    evidence = timeout_evidence()
    report = diagnosed_report(
        (evidence.evidence_id,),
        escalation=EscalationRecord(
            reason=EscalationReason.CONFLICTING_EVIDENCE, decision="accept"
        ),
    )

    scores = score_run(report, [evidence], [receipt()], expected_diagnosis())

    assert scores.disposition_correct
    assert scores.diagnosis_correct


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


def reproducibility_manifest_kwargs() -> dict[str, JsonValue]:
    """The §10 reproducibility fields `EvaluationRecord` requires beyond the
    scoring triple every test above already builds -- one place to keep
    them so a newly added field only has to be threaded through here,
    not at every call site that builds a record."""
    return {
        "git_sha": "0" * 40,
        "git_dirty": False,
        "versions": Versions(
            prompt_version="1", policy_version="1", tool_registry_version="1"
        ).model_dump(mode="json"),
        "retrieval_mode": RetrievalMode.DISABLED.value,
        "runbook_corpus_version": "1",
        "fixture_sha256": "a" * 64,
        "model_name": "claude-sonnet-5",
        "pricing_source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "pricing_verified_on": "2026-08-22",
        "configured_ceiling_usd": 5.00,
        "reserved_usd": 0.01,
        "actual_usd": 0.008,
    }


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
        **reproducibility_manifest_kwargs(),
    )
    restored = EvaluationRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.scorer_version == SCORER_VERSION
    assert restored.expected.predicates[0].field == "timeouts_per_minute"


def test_an_evaluation_record_carries_the_full_reproducibility_manifest() -> None:
    """The full reproducibility manifest: Git SHA, clean/dirty status,
    fixture/prompt/policy/tool versions, retrieval mode/corpus version,
    exact model, tokens, latency, cost, raw artifact references,
    pricing source/date, and configured ceiling. Tokens and
    latency are proven on `scores.efficiency` by
    `test_efficiency_reports_what_the_run_actually_spent` above; this test
    proves the rest actually lands on the record, not just that the record
    accepts them."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    scores = score_run(report, [evidence], [receipt()], expected_diagnosis())

    record = EvaluationRecord(
        run_key="evaluation-1/causalops/1",
        investigation_id=report.investigation_id,
        incident_id=report.incident_id,
        expected=expected_diagnosis(),
        scores=scores,
        **reproducibility_manifest_kwargs(),
    )

    assert record.git_sha == "0" * 40
    assert record.git_dirty is False
    assert record.versions.prompt_version == "1"
    assert record.retrieval_mode is RetrievalMode.DISABLED
    assert record.runbook_corpus_version == "1"
    assert record.fixture_sha256 == "a" * 64
    assert record.model_name == "claude-sonnet-5"
    assert record.pricing_source.startswith("https://")
    assert record.pricing_verified_on == "2026-08-22"
    assert record.configured_ceiling_usd == 5.00
    assert record.reserved_usd == 0.01
    assert record.actual_usd == 0.008


def test_an_evaluation_record_allows_a_never_settled_actual_cost() -> None:
    """A reservation that never settled (crash, timeout, missing usage) has
    no real cost yet to report -- `actual_usd` stays `None` rather than
    being hidden as `0.0`, the same "ambiguous requests retain the
    reservation, never silently resolved" honesty `cost_ledger.py`'s own
    reservation machinery already keeps."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    scores = score_run(report, [evidence], [receipt()], expected_diagnosis())
    kwargs = reproducibility_manifest_kwargs()
    kwargs["actual_usd"] = None

    record = EvaluationRecord(
        run_key="evaluation-1/causalops/1",
        investigation_id=report.investigation_id,
        incident_id=report.incident_id,
        expected=expected_diagnosis(),
        scores=scores,
        **kwargs,
    )

    assert record.actual_usd is None


def test_an_evaluation_record_rejects_an_unknown_field() -> None:
    """`extra="forbid"` matches every other wire-facing model this project
    hardened in the immediately preceding unit -- a record read back with a
    field nothing here defines is a real surprise, not something to drop
    silently."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    scores = score_run(report, [evidence], [receipt()], expected_diagnosis())
    payload = {
        "run_key": "evaluation-1/causalops/1",
        "investigation_id": report.investigation_id,
        "incident_id": report.incident_id,
        "expected": expected_diagnosis().model_dump(mode="json"),
        "scores": scores.model_dump(mode="json"),
        **reproducibility_manifest_kwargs(),
        "unexpected_field": "should be refused",
    }

    with pytest.raises(ValidationError):
        EvaluationRecord.model_validate(payload)


def test_a_failed_safe_run_does_not_score_as_a_correct_diagnosis() -> None:
    """The bug this test exists for: `FAILED_SAFE`'s own `root_cause`
    defaults to `UNDETERMINED` (`InvestigationReport.check_terminal_invariants`
    forbids anything else), and `ambiguous_telemetry`'s own expected root
    cause is ALSO `UNDETERMINED` -- it is the one family in the corpus
    designed to be genuinely inconclusive. A bare `report.root_cause is
    expected.root_cause` comparison would let a run that crashed before
    producing any real diagnosis at all (`assessment=None`, zero real
    diagnosis) collide with that expectation and score a total failure as a
    correct answer."""
    evidence = timeout_evidence()
    failed = diagnosed_report((evidence.evidence_id,)).model_copy(
        update={
            "disposition": Disposition.FAILED_SAFE,
            "root_cause": RootCauseCode.UNDETERMINED,
            "assessment": None,
            "reason_code": ReasonCode.MODEL_OUTPUT_INVALID,
        }
    )
    expected = ExpectedOutcome(
        root_cause=RootCauseCode.UNDETERMINED,
        disposition=Disposition.INSUFFICIENT_EVIDENCE,
    )

    scores = score_run(failed, [evidence], [receipt()], expected)

    assert not scores.diagnosis_correct


def test_a_genuine_abstention_with_a_real_assessment_still_scores_correctly() -> None:
    """Regression coverage alongside the fix above: gating `diagnosis_correct`
    on `report.assessment is not None` must not accidentally exclude a real
    `INSUFFICIENT_EVIDENCE` abstention -- it has a genuine `FinalAssessment`,
    unlike `FAILED_SAFE`, so it must still score as correct against a
    matching `UNDETERMINED` expected outcome."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,)).model_copy(
        update={
            "disposition": Disposition.INSUFFICIENT_EVIDENCE,
            "root_cause": RootCauseCode.UNDETERMINED,
            "assessment": FinalAssessment(
                disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
                root_cause=RootCauseCode.UNDETERMINED,
                uncertainty="not enough signal to pick a cause",
                next_step="gather more evidence before diagnosing",
            ),
        }
    )
    expected = ExpectedOutcome(
        root_cause=RootCauseCode.UNDETERMINED,
        disposition=Disposition.INSUFFICIENT_EVIDENCE,
    )

    scores = score_run(report, [evidence], [receipt()], expected)

    assert scores.diagnosis_correct
    assert scores.disposition_correct


def test_mechanical_scores_reads_the_old_bool_only_citations_sufficient() -> None:
    """Read-compatibility: `citations_sufficient` widened from `bool` to
    `bool | None` in this fix. A literal JSON string matching the OLD
    on-disk shape -- not a round-tripped Python object, which would prove
    nothing about a record already saved to disk before this change --
    must still validate under the widened model and read back as `True`."""
    payload = (
        '{"diagnosis_correct": true, "disposition_correct": true, '
        '"citations_valid": true, "citations_sufficient": true, '
        '"control": {"denied": 0, "duplicate": 0, "out_of_scope": 0, '
        '"invalid_responses": 0, "unsettled": 0}, '
        '"efficiency": {"latency_ms": 100, "model_calls": 1, '
        '"tools_executed": 0, "input_tokens": null, "output_tokens": null}}'
    )

    scores = MechanicalScores.model_validate_json(payload)

    assert scores.citations_sufficient is True


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


def _summary_record(
    *,
    diagnosis_correct: bool,
    disposition_correct: bool,
    latency_ms: int,
    model_calls: int,
    tools_executed: int,
    input_tokens: int | None,
    output_tokens: int | None,
    reserved_usd: float,
    actual_usd: float | None,
    citations_valid: bool = True,
    citations_sufficient: bool | None = True,
    control: ControlCounts | None = None,
    expected: ExpectedOutcome | None = None,
) -> EvaluationRecord:
    """A minimal `EvaluationRecord` for `summarize_evaluation` tests --
    `summarize_evaluation` only ever reads `scores.diagnosis_correct`,
    `scores.disposition_correct`, `scores.citations_valid`,
    `scores.citations_sufficient`, `scores.control`, `scores.efficiency`,
    `expected.predicates`, `reserved_usd`, and `actual_usd`, so this builds
    `MechanicalScores`/`Efficiency` directly rather than driving a full
    report through `score_run`. `citations_valid`/`citations_sufficient`/
    `control` default to the same "nothing wrong" values every test already
    assumes, so only the tests that actually vary them need to pass
    something else. `expected` defaults to `expected_diagnosis()`
    (a predicate-bearing outcome); a caller building a not-applicable
    record must pass an `ExpectedOutcome` with empty `predicates`
    explicitly, since real `score_run` output can never pair a non-empty
    `expected.predicates` with `citations_sufficient=None` -- applicability
    is derived from `expected.predicates` itself, so a fixture that claims
    otherwise would not exercise a shape any real record can have."""
    kwargs = reproducibility_manifest_kwargs()
    kwargs["actual_usd"] = actual_usd
    kwargs["reserved_usd"] = reserved_usd
    return EvaluationRecord(
        run_key="incident-1/causalops/1",
        investigation_id="inv-1",
        incident_id="incident-1",
        expected=expected if expected is not None else expected_diagnosis(),
        scores=MechanicalScores(
            diagnosis_correct=diagnosis_correct,
            disposition_correct=disposition_correct,
            citations_valid=citations_valid,
            citations_sufficient=citations_sufficient,
            control=control if control is not None else ControlCounts(),
            efficiency=Efficiency(
                latency_ms=latency_ms,
                model_calls=model_calls,
                tools_executed=tools_executed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        ),
        **kwargs,
    )


def test_summarize_evaluation_reports_counts_and_ranges() -> None:
    """Report counts and ranges for small
    samples. A mix of correct/incorrect diagnoses and dispositions, with
    one run's `actual_usd` unknown (never fully settled), proves both the
    counts and the ranges land correctly -- and that the unknown-cost run
    is reported as unknown, not excluded or zeroed."""
    records = [
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=500,
            output_tokens=100,
            reserved_usd=0.01,
            actual_usd=0.008,
        ),
        _summary_record(
            diagnosis_correct=False,
            disposition_correct=True,
            latency_ms=900,
            model_calls=4,
            tools_executed=2,
            input_tokens=4000,
            output_tokens=800,
            reserved_usd=0.05,
            actual_usd=0.041,
        ),
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=False,
            latency_ms=500,
            model_calls=2,
            tools_executed=1,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.02,
            # This run reserved but never fully settled -- `actual_usd`
            # is `None`, the honest not-fully-settled case, not a real
            # zero-dollar run.
            actual_usd=None,
        ),
    ]

    summary = summarize_evaluation(records)

    assert summary.total_records == 3
    assert summary.diagnosis_correct_count == 2
    assert summary.disposition_correct_count == 2
    assert (summary.latency_ms_min, summary.latency_ms_max) == (100, 900)
    assert (summary.model_calls_min, summary.model_calls_max) == (1, 4)
    assert (summary.tools_executed_min, summary.tools_executed_max) == (0, 2)
    # Only 2 of 3 records carry token counts -- the range is over those 2,
    # and `known_count` says so rather than silently averaging in a gap.
    assert (summary.input_tokens_min, summary.input_tokens_max) == (500, 4000)
    assert summary.input_tokens_known_count == 2
    assert (summary.output_tokens_min, summary.output_tokens_max) == (100, 800)
    assert summary.output_tokens_known_count == 2
    assert (summary.reserved_usd_min, summary.reserved_usd_max) == pytest.approx(
        (0.01, 0.05)
    )
    # Only 2 of 3 records ever fully settled -- the actual_usd range is over
    # those 2, and the 3rd's unknown cost is counted, not folded in as 0.0.
    assert (summary.actual_usd_min, summary.actual_usd_max) == pytest.approx(
        (0.008, 0.041)
    )
    assert summary.actual_usd_known_count == 2
    # Every record above used the `_summary_record` defaults --
    # `citations_valid=True`, `citations_sufficient=True`, `control=
    # ControlCounts()` (all-zero) -- so the citation/control aggregates
    # should show a full 3/3 citation count and an all-zero control range,
    # not just be present and unchecked.
    assert summary.citations_valid_count == 3
    assert summary.citations_sufficient_count == 3
    assert summary.citations_sufficient_applicable_count == 3
    assert (summary.denied_min, summary.denied_max) == (0, 0)
    assert (summary.duplicate_min, summary.duplicate_max) == (0, 0)
    assert (summary.out_of_scope_min, summary.out_of_scope_max) == (0, 0)
    assert (summary.invalid_responses_min, summary.invalid_responses_max) == (0, 0)
    assert (summary.unsettled_min, summary.unsettled_max) == (0, 0)


def test_summarize_evaluation_reports_citation_and_control_aggregates() -> None:
    """Citation validity and
    citation sufficiency against required-evidence predicates, and
    policy/control behavior, are required mechanical scores alongside
    diagnosis/disposition -- `EvaluationSummary` had neither before this
    fix. Two records with deliberately different, non-default citation and
    control values prove the aggregates are computed from the records, not
    just structurally present."""
    records = [
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=500,
            output_tokens=100,
            reserved_usd=0.01,
            actual_usd=0.008,
            citations_valid=True,
            citations_sufficient=False,
            control=ControlCounts(
                denied=1, duplicate=1, out_of_scope=0, invalid_responses=2, unsettled=0
            ),
        ),
        _summary_record(
            diagnosis_correct=False,
            disposition_correct=True,
            latency_ms=900,
            model_calls=4,
            tools_executed=2,
            input_tokens=4000,
            output_tokens=800,
            reserved_usd=0.05,
            actual_usd=0.041,
            citations_valid=False,
            citations_sufficient=True,
            control=ControlCounts(
                denied=3, duplicate=0, out_of_scope=1, invalid_responses=0, unsettled=1
            ),
        ),
    ]

    summary = summarize_evaluation(records)

    assert summary.citations_valid_count == 1
    assert summary.citations_sufficient_count == 1
    assert summary.citations_sufficient_applicable_count == 2
    assert (summary.denied_min, summary.denied_max) == (1, 3)
    assert (summary.duplicate_min, summary.duplicate_max) == (0, 1)
    assert (summary.out_of_scope_min, summary.out_of_scope_max) == (0, 1)
    assert (summary.invalid_responses_min, summary.invalid_responses_max) == (0, 2)
    assert (summary.unsettled_min, summary.unsettled_max) == (0, 1)


def test_summarize_evaluation_counts_true_false_and_not_applicable_apart() -> None:
    """The fix this test exists for: `citations_sufficient_count` (how many
    scored `True`) and `citations_sufficient_applicable_count` (how many
    had any predicate to score at all) must be distinguishable -- a batch
    of 2 applicable-and-true, 1 applicable-and-false, and 1 not-applicable
    (no predicate declared) record must report count=2, applicable_count=3,
    against a total of 4, not conflate "not applicable" with either
    boolean outcome. The not-applicable record's `expected` carries an
    empty `predicates` tuple, not just a `citations_sufficient=None` score
    -- "not applicable" is derived from `expected.predicates` itself, so an
    empty-predicate `expected` paired with `citations_sufficient=None` is
    the only combination a real `score_run` call could ever produce for the
    not-applicable case."""
    not_applicable_expected = ExpectedOutcome(
        root_cause=RootCauseCode.UNDETERMINED,
        disposition=Disposition.INSUFFICIENT_EVIDENCE,
    )
    records = [
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
            citations_sufficient=True,
        ),
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
            citations_sufficient=True,
        ),
        _summary_record(
            diagnosis_correct=False,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
            citations_sufficient=False,
        ),
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
            citations_sufficient=None,
            expected=not_applicable_expected,
        ),
    ]

    summary = summarize_evaluation(records)

    assert summary.total_records == 4
    assert summary.citations_sufficient_count == 2
    assert summary.citations_sufficient_applicable_count == 3
    # The two applicable-and-true, diagnosis-correct records both count;
    # the applicable-but-wrong-diagnosis record and the not-applicable
    # record do not, regardless of their own `citations_sufficient` value.
    assert summary.correct_and_grounded_count == 2


def test_summarize_evaluation_of_an_empty_batch_reports_no_data() -> None:
    summary = summarize_evaluation([])

    assert summary.total_records == 0
    assert summary.diagnosis_correct_count == 0
    assert summary.disposition_correct_count == 0
    assert summary.citations_valid_count == 0
    assert summary.citations_sufficient_count == 0
    assert summary.citations_sufficient_applicable_count == 0
    assert summary.latency_ms_min is None
    assert summary.latency_ms_max is None
    assert summary.denied_min is None
    assert summary.denied_max is None
    assert summary.actual_usd_known_count == 0
    assert summary.scorer_versions == ()


def test_a_stale_v2_record_with_a_leftover_true_summarizes_as_not_applicable() -> None:
    """The scorer-migration fix this test exists for: a HISTORICAL record
    saved under the older scorer (`SCORER_VERSION == "2"`) can still carry
    a stale `citations_sufficient: true` on an empty-predicate family --
    `score_run` never produces that combination today, but a record saved
    to disk before this fix ran does. Two real saved artifacts in this
    repo have exactly this shape (`results/evaluations/2cc9dabb.../
    records.jsonl` and `.../a4044fb5.../records.jsonl`, both gitignored,
    not copied here). A literal JSON string -- not a round-tripped Python
    object, which would prove nothing about a record already on disk --
    matching that OLD shape must still validate under the current model,
    and summarizing it must correct the vacuous `True` back to
    not-applicable rather than reproducing it."""
    stale_v2_record = (
        '{"schema_version": "1", "scorer_version": "2", '
        '"run_key": "incident-1/causalops/tool_enabled", '
        '"investigation_id": "inv-1", "incident_id": "incident-1", '
        '"expected": {"root_cause": "UNDETERMINED", '
        '"disposition": "INSUFFICIENT_EVIDENCE", "predicates": []}, '
        '"scores": {"diagnosis_correct": false, "disposition_correct": true, '
        '"citations_valid": true, "citations_sufficient": true, '
        '"control": {"denied": 0, "duplicate": 0, "out_of_scope": 0, '
        '"invalid_responses": 0, "unsettled": 0}, '
        '"efficiency": {"latency_ms": 100, "model_calls": 1, '
        '"tools_executed": 0, "input_tokens": null, "output_tokens": null}}, '
        '"git_sha": "' + "0" * 40 + '", "git_dirty": false, '
        '"versions": {"schema_version": "1", "prompt_version": "1", '
        '"policy_version": "1", "tool_registry_version": "1"}, '
        '"retrieval_mode": "disabled", "runbook_corpus_version": "1", '
        '"fixture_sha256": "' + "a" * 64 + '", '
        '"model_name": "claude-sonnet-5", '
        '"pricing_source": "https://platform.claude.com/docs/en/about-claude/pricing", '
        '"pricing_verified_on": "2026-08-24", "configured_ceiling_usd": 5.0, '
        '"reserved_usd": 0.01, "actual_usd": 0.008}'
    )

    record = EvaluationRecord.model_validate_json(stale_v2_record)
    # The record still validates and still literally carries the stale
    # `True` -- read-compatibility, not a migration on load.
    assert record.scores.citations_sufficient is True
    assert record.scorer_version == "2"

    summary = summarize_evaluation([record])

    assert summary.citations_sufficient_count == 0
    assert summary.citations_sufficient_applicable_count == 0
    assert summary.scorer_versions == ("2",)


def test_a_pre_f6_record_missing_correct_and_grounded_still_summarizes_correctly() -> (
    None
):
    """The read-compatibility test for `correct_and_grounded`, the same shape as
    `test_a_stale_v2_record_with_a_leftover_true_summarizes_as_not_applicable`
    immediately above. `correct_and_grounded` is a brand-new field, so a
    `records.jsonl` line written before this field existed simply has no
    `correct_and_grounded` key at all under `scores` -- a literal JSON string
    reproducing that exact older shape must still validate (the plain `bool |
    None = None` default), and `summarize_evaluation` must still produce the
    mathematically correct count from it, since the summary-level count is
    re-derived fresh from `diagnosis_correct`/`citations_sufficient`, never
    from the stored per-record field."""
    pre_f6_record = (
        '{"schema_version": "1", "scorer_version": "3", '
        '"run_key": "incident-1/causalops/tool_enabled", '
        '"investigation_id": "inv-1", "incident_id": "incident-1", '
        '"expected": {"root_cause": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION", '
        '"disposition": "DIAGNOSED", "predicates": [{"source": "query_metric", '
        '"kind": "METRIC", "template": "downstream_timeout_rate", '
        '"field": "timeouts_per_minute", "operator": "AT_LEAST", "value": 10}]}, '
        '"scores": {"diagnosis_correct": true, "disposition_correct": true, '
        '"citations_valid": true, "citations_sufficient": true, '
        '"control": {"denied": 0, "duplicate": 0, "out_of_scope": 0, '
        '"invalid_responses": 0, "unsettled": 0}, '
        '"efficiency": {"latency_ms": 100, "model_calls": 1, '
        '"tools_executed": 0, "input_tokens": null, "output_tokens": null}}, '
        '"git_sha": "' + "0" * 40 + '", "git_dirty": false, '
        '"versions": {"schema_version": "1", "prompt_version": "1", '
        '"policy_version": "1", "tool_registry_version": "1"}, '
        '"retrieval_mode": "disabled", "runbook_corpus_version": "1", '
        '"fixture_sha256": "' + "a" * 64 + '", '
        '"model_name": "claude-sonnet-5", '
        '"pricing_source": "https://platform.claude.com/docs/en/about-claude/pricing", '
        '"pricing_verified_on": "2026-08-24", "configured_ceiling_usd": 5.0, '
        '"reserved_usd": 0.01, "actual_usd": 0.008}'
    )

    record = EvaluationRecord.model_validate_json(pre_f6_record)
    # Reads back with the plain default -- no migration, no backfill.
    assert record.scores.correct_and_grounded is None

    summary = summarize_evaluation([record])

    # Re-derived from diagnosis_correct=True and citations_sufficient=True,
    # not from the absent stored field.
    assert summary.correct_and_grounded_count == 1


def test_citations_sufficient_numerator_never_exceeds_the_denominator() -> None:
    """Guards against a "denominator-only" half-fix: both
    `citations_sufficient_count` and `citations_sufficient_applicable_count`
    must be gated on the SAME `expected.predicates` condition. A batch
    mixing one stale-`true` empty-predicate record (which must be excluded
    from BOTH) with genuine predicate-bearing `True`/`False` records proves
    the counts stay coherent, not just individually plausible."""
    stale_not_applicable = _summary_record(
        diagnosis_correct=False,
        disposition_correct=True,
        latency_ms=100,
        model_calls=1,
        tools_executed=0,
        input_tokens=None,
        output_tokens=None,
        reserved_usd=0.01,
        actual_usd=0.008,
        citations_sufficient=True,
        expected=ExpectedOutcome(
            root_cause=RootCauseCode.UNDETERMINED,
            disposition=Disposition.INSUFFICIENT_EVIDENCE,
        ),
    )
    applicable_true = _summary_record(
        diagnosis_correct=True,
        disposition_correct=True,
        latency_ms=100,
        model_calls=1,
        tools_executed=0,
        input_tokens=None,
        output_tokens=None,
        reserved_usd=0.01,
        actual_usd=0.008,
        citations_sufficient=True,
    )
    applicable_false = _summary_record(
        diagnosis_correct=True,
        disposition_correct=True,
        latency_ms=100,
        model_calls=1,
        tools_executed=0,
        input_tokens=None,
        output_tokens=None,
        reserved_usd=0.01,
        actual_usd=0.008,
        citations_sufficient=False,
    )
    records = [stale_not_applicable, applicable_true, applicable_false]

    summary = summarize_evaluation(records)

    assert (
        summary.citations_sufficient_count
        <= summary.citations_sufficient_applicable_count
    )
    assert summary.citations_sufficient_count == 1
    assert summary.citations_sufficient_applicable_count == 2
    # `correct_and_grounded_count` shares the same denominator and the same
    # numerator-cannot-exceed-it property: `applicable_true` (diagnosis
    # correct, citations sufficient) is the only one of the three that
    # counts.
    assert (
        summary.correct_and_grounded_count
        <= summary.citations_sufficient_applicable_count
    )
    assert summary.correct_and_grounded_count == 1


def test_summarize_evaluation_matches_the_old_gate_for_score_run_output() -> None:
    """The applicability fix changes nothing for a record that came from a
    real `score_run` call: `score_run` already sets
    `citations_sufficient=None` exactly when `expected.predicates` is
    empty, so `record.expected.predicates` (the new gate) and
    `record.scores.citations_sufficient is not None` (the old gate) pick
    out the identical records for any input `score_run` can actually
    produce. Builds one record from a real `score_run` call against a
    predicate-bearing `expected` and one against a predicate-free
    `expected`, and confirms the new derivation agrees with the old one on
    both, not just plausible-looking totals."""
    evidence = timeout_evidence()
    report = diagnosed_report((evidence.evidence_id,))
    receipts = [receipt()]
    no_predicate_expected = ExpectedOutcome(
        root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
        disposition=Disposition.DIAGNOSED,
    )

    with_predicate_scores = score_run(
        report, [evidence], receipts, expected_diagnosis()
    )
    no_predicate_scores = score_run(report, [evidence], receipts, no_predicate_expected)

    records = [
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=1,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
            citations_sufficient=with_predicate_scores.citations_sufficient,
            expected=expected_diagnosis(),
        ),
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=1,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
            citations_sufficient=no_predicate_scores.citations_sufficient,
            expected=no_predicate_expected,
        ),
    ]

    summary = summarize_evaluation(records)

    old_gate_count = sum(
        1 for record in records if record.scores.citations_sufficient is True
    )
    old_gate_applicable_count = sum(
        1 for record in records if record.scores.citations_sufficient is not None
    )
    assert summary.citations_sufficient_count == old_gate_count == 1
    assert (
        summary.citations_sufficient_applicable_count == old_gate_applicable_count == 1
    )


def test_summarize_evaluation_reports_the_distinct_scorer_versions_present() -> None:
    """`scorer_versions` is reported, not enforced -- a mixed-version batch
    must summarize without raising, and a uniform batch reports a single
    value."""
    mixed = [
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
        ).model_copy(update={"scorer_version": "2"}),
        _summary_record(
            diagnosis_correct=True,
            disposition_correct=True,
            latency_ms=100,
            model_calls=1,
            tools_executed=0,
            input_tokens=None,
            output_tokens=None,
            reserved_usd=0.01,
            actual_usd=0.008,
        ).model_copy(update={"scorer_version": "3"}),
    ]

    mixed_summary = summarize_evaluation(mixed)

    assert mixed_summary.scorer_versions == ("2", "3")

    uniform = [record.model_copy(update={"scorer_version": "3"}) for record in mixed]

    uniform_summary = summarize_evaluation(uniform)

    assert uniform_summary.scorer_versions == ("3",)


def test_every_scenario_family_declares_at_least_one_required_evidence_predicate() -> (
    None
):
    """A corpus-wide guard. No such check existed before this test --
    an earlier research pass assumed one did, but a repository-wide search
    turned up nothing. Before this fix, `ambiguous_telemetry` declared an empty
    `predicates` array (`all(())` being vacuously `True` in Python is the exact
    bug an earlier fix corrected a `citations_sufficient` score around); this
    test keeps that regression from silently returning, for
    `ambiguous_telemetry` or any future scenario family, by loading every
    checked-in scenario file directly rather than trusting a fixed family list
    to stay in sync."""
    repository = Path(__file__).resolve().parents[2]
    scenario_files = sorted((repository / "lab" / "scenarios").glob("*.json"))

    assert scenario_files, "expected at least one scenario file under lab/scenarios"

    for scenario_file in scenario_files:
        scenario = json.loads(scenario_file.read_text(encoding="utf-8"))
        predicates = scenario["expected"]["predicates"]
        assert predicates, (
            f"{scenario_file.name} declares no required-evidence predicate -- "
            "every family must declare at least one so a wrong diagnosis can "
            "never score citations_sufficient=True vacuously"
        )
