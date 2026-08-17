from pathlib import Path

from fake_incident import (
    FIXTURE_DIR,
    SYMPTOM_EVIDENCE_ID,
    StepClock,
    UsageReportingModel,
    alert_packet,
    assessment_json,
    check_runner,
    incident_scope,
    logs_proposal,
    metric_proposal,
    packet_evidence,
    plan_json,
    replay_model,
    update_json,
)

from causalops.domain import (
    Budgets,
    CheckOutcome,
    Disposition,
    IncidentScope,
    ModelDisposition,
    ModelUsage,
    PolicyResult,
    ReasonCode,
    RootCauseCode,
    RunCheck,
    ToolOutcome,
    ToolProposal,
)
from causalops.models import ReasoningModel, ReplayReasoningModel
from causalops.run_records import RunRecorder
from causalops.workflow import InvestigationResult, run_investigation


def investigate(
    model: ReasoningModel,
    check: RunCheck | None = None,
    budgets: Budgets | None = None,
    clock: StepClock | None = None,
) -> tuple[InvestigationResult, RunRecorder]:
    ticking = clock or StepClock()
    recorder = RunRecorder(ticking)
    result = run_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        model,
        check or check_runner(),
        recorder,
        budgets or Budgets(),
        ticking,
    )
    return result, recorder


def fixture_model(name: str) -> ReplayReasoningModel:
    return ReplayReasoningModel(FIXTURE_DIR / name)


def test_a_scripted_diagnosis_runs_both_checks_and_diagnoses() -> None:
    result, _ = investigate(fixture_model("valid_diagnosis.json"))
    report = result.report

    assert report.disposition is Disposition.DIAGNOSED
    assert report.root_cause is RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION
    assert report.model_calls_used == 3
    assert report.tools_executed == 2
    assert report.reason_code is None
    assert len(result.receipts) == 2
    assert len(result.evidence) == 4


def test_a_scripted_abstention_stops_early_and_abstains() -> None:
    result, _ = investigate(fixture_model("correct_abstention.json"))
    report = result.report

    assert report.disposition is Disposition.INSUFFICIENT_EVIDENCE
    assert report.root_cause is RootCauseCode.UNDETERMINED
    assert report.tools_executed == 1
    assert report.model_calls_used == 3


def test_one_invalid_response_is_repaired_and_the_run_continues() -> None:
    result, _ = investigate(fixture_model("repair_then_valid.json"))
    report = result.report

    assert report.disposition is Disposition.DIAGNOSED
    assert report.repairs_used == 1
    assert report.invalid_responses == 1
    # The repair spends an ordinary model call: plan, repair, update, assessment.
    assert report.model_calls_used == 4


def test_a_repair_tells_the_model_what_was_wrong() -> None:
    model = fixture_model("repair_then_valid.json")

    investigate(model)

    first, repair = model.requests[0], model.requests[1]
    assert first.repair_errors is None
    assert repair.stage is first.stage
    assert repair.repair_errors and "hypotheses" in repair.repair_errors


def test_a_run_with_no_repair_budget_stops_at_the_first_invalid_response() -> None:
    result, _ = investigate(
        fixture_model("malformed_output.json"), budgets=Budgets(repairs=0)
    )

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.REPAIR_EXHAUSTED
    assert result.report.repairs_used == 0
    assert result.report.model_calls_used == 1


def test_an_unavailable_check_is_recorded_without_evidence() -> None:
    result, _ = investigate(
        fixture_model("valid_diagnosis.json"),
        check=check_runner(
            outcome=ToolOutcome.UNAVAILABLE, reason_code=ReasonCode.TOOL_UNAVAILABLE
        ),
    )

    assert result.receipts[0].outcome is ToolOutcome.UNAVAILABLE
    assert result.receipts[0].reason_code is ReasonCode.TOOL_UNAVAILABLE
    assert result.receipts[0].result_digest is None
    assert len(result.evidence) == 2


def test_a_response_that_stays_invalid_fails_safe() -> None:
    result, _ = investigate(fixture_model("malformed_output.json"))
    report = result.report

    assert report.disposition is Disposition.FAILED_SAFE
    assert report.root_cause is RootCauseCode.UNDETERMINED
    assert report.reason_code is ReasonCode.MODEL_OUTPUT_INVALID
    assert report.assessment is None
    assert report.invalid_responses == 2
    assert report.tools_executed == 0


def test_a_denied_proposal_costs_a_model_call_but_no_check_slot(tmp_path: Path) -> None:
    model = replay_model(
        tmp_path,
        {
            "initial_plan": [plan_json(metric_proposal(service="billing"))],
            "hypothesis_update": [update_json(stop_reason="no safe check remains")],
            "final_assessment": [
                assessment_json(
                    disposition=ModelDisposition.INSUFFICIENT_EVIDENCE,
                    root_cause=RootCauseCode.UNDETERMINED,
                )
            ],
        },
    )

    result, _ = investigate(model)

    receipt = result.receipts[0]
    assert receipt.policy_result is PolicyResult.DENIED
    assert receipt.reason_code is ReasonCode.UNKNOWN_SERVICE
    assert receipt.outcome is ToolOutcome.NOT_EXECUTED
    # A denial is not terminal: the run reached a valid abstention anyway.
    assert result.report.disposition is Disposition.INSUFFICIENT_EVIDENCE
    assert result.report.tools_executed == 0
    assert result.report.model_calls_used == 3
    assert len(result.evidence) == 2


def test_the_same_proposal_twice_is_denied_as_a_duplicate(tmp_path: Path) -> None:
    model = replay_model(
        tmp_path,
        {
            "initial_plan": [plan_json(metric_proposal())],
            "hypothesis_update": [update_json(metric_proposal())],
            "final_assessment": [assessment_json()],
        },
    )

    result, _ = investigate(model)

    assert result.receipts[0].policy_result is PolicyResult.ALLOWED
    assert result.receipts[1].reason_code is ReasonCode.DUPLICATE_PROPOSAL
    assert result.report.tools_executed == 1


def test_a_check_that_times_out_still_spends_its_slot() -> None:
    result, _ = investigate(
        fixture_model("valid_diagnosis.json"),
        check=check_runner(
            outcome=ToolOutcome.TIMEOUT, reason_code=ReasonCode.TOOL_TIMEOUT
        ),
    )

    assert result.report.tools_executed == 2
    assert [receipt.outcome for receipt in result.receipts] == [
        ToolOutcome.TIMEOUT,
        ToolOutcome.TIMEOUT,
    ]
    assert result.receipts[0].evidence_id is None
    assert len(result.evidence) == 2


def test_a_cited_evidence_id_that_does_not_exist_fails_safe(tmp_path: Path) -> None:
    model = replay_model(
        tmp_path,
        {
            "initial_plan": [plan_json(stop_reason="the alert is enough")],
            "final_assessment": [assessment_json(supporting=("made-up-evidence",))],
        },
    )

    result, _ = investigate(model)

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.FORGED_EVIDENCE_REFERENCE


def test_running_out_of_model_calls_fails_safe() -> None:
    result, _ = investigate(
        fixture_model("valid_diagnosis.json"), budgets=Budgets(model_calls=1)
    )

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.MODEL_CALL_BUDGET_EXHAUSTED
    assert result.report.model_calls_used == 1


def test_the_second_check_is_skipped_when_the_assessment_would_not_fit() -> None:
    result, _ = investigate(
        fixture_model("valid_diagnosis.json"), budgets=Budgets(model_calls=2)
    )

    assert result.report.disposition is Disposition.DIAGNOSED
    assert result.report.model_calls_used == 2
    assert result.report.tools_executed == 1


def test_a_run_past_its_wall_clock_fails_safe() -> None:
    result, _ = investigate(
        fixture_model("valid_diagnosis.json"), clock=StepClock(step_seconds=400)
    )

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.WALL_CLOCK_EXPIRED
    assert result.report.model_calls_used == 0


def test_the_second_check_is_never_proposed_once_the_slots_are_gone(
    tmp_path: Path,
) -> None:
    """Section 5 skips the remaining check rather than proposing one that must fail."""
    model = replay_model(
        tmp_path,
        {
            "initial_plan": [plan_json(metric_proposal())],
            "hypothesis_update": [update_json(logs_proposal())],
            "final_assessment": [assessment_json()],
        },
    )

    result, recorder = investigate(model, budgets=Budgets(executed_tools=1))

    assert result.report.tools_executed == 1
    assert len(result.receipts) == 1
    assert result.report.disposition is Disposition.DIAGNOSED
    assert "UPDATE_AND_PLAN_SECOND" not in {event.state for event in recorder.events}


def test_a_failing_check_records_the_error_without_evidence() -> None:
    result, _ = investigate(
        fixture_model("valid_diagnosis.json"),
        check=check_runner(
            outcome=ToolOutcome.ERROR, reason_code=ReasonCode.TOOL_ERROR
        ),
    )

    assert result.receipts[0].outcome is ToolOutcome.ERROR
    assert result.receipts[0].reason_code is ReasonCode.TOOL_ERROR
    assert len(result.evidence) == 2


def test_a_naive_window_from_the_model_still_produces_a_report(
    tmp_path: Path,
) -> None:
    """A timestamp with no timezone used to raise instead of ending the run."""
    naive = plan_json(metric_proposal())
    naive["proposal"]["arguments"]["window_start"] = "2026-08-16T10:00:00"

    result, _ = investigate(replay_model(tmp_path, {"initial_plan": [naive, naive]}))

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.MODEL_OUTPUT_INVALID
    assert result.report.invalid_responses == 2


def test_an_unexpected_failure_becomes_a_terminal_state(tmp_path: Path) -> None:
    def exploding_check(proposal: ToolProposal, scope: IncidentScope) -> CheckOutcome:
        raise RuntimeError("boom-with-sensitive-detail")

    result, recorder = investigate(
        fixture_model("valid_diagnosis.json"), check=exploding_check
    )

    assert result.report.disposition is Disposition.FAILED_SAFE
    assert result.report.reason_code is ReasonCode.INTERNAL_ERROR
    recorded = "".join(event.model_dump_json() for event in recorder.events)
    assert "RuntimeError" in recorded
    assert "boom-with-sensitive-detail" not in recorded
    assert "boom-with-sensitive-detail" not in result.report.model_dump_json()


def test_token_usage_adds_up_across_every_model_call() -> None:
    model = UsageReportingModel(
        fixture_model("valid_diagnosis.json"),
        ModelUsage(input_tokens=1000, output_tokens=200),
    )

    result, _ = investigate(model)

    assert result.report.model_calls_used == 3
    assert result.report.usage == ModelUsage(input_tokens=3000, output_tokens=600)
    assert result.report.limitations == ()


def test_the_report_names_the_evidence_and_receipts_it_rests_on() -> None:
    result, _ = investigate(fixture_model("valid_diagnosis.json"))

    assert SYMPTOM_EVIDENCE_ID in result.report.evidence_ids
    assert set(result.report.receipt_ids) == {
        receipt.receipt_id for receipt in result.receipts
    }
    assert result.report.usage is None
    assert result.report.limitations == ("this model reports no token usage",)


def test_the_same_run_produces_the_same_context_digest(tmp_path: Path) -> None:
    script = {
        "initial_plan": [plan_json(stop_reason="the alert is enough")],
        "final_assessment": [assessment_json(supporting=(SYMPTOM_EVIDENCE_ID,))],
    }

    first, _ = investigate(replay_model(tmp_path, script))
    second, _ = investigate(replay_model(tmp_path, script))

    assert first.report.final_context_digest == second.report.final_context_digest
    assert first.report.final_context_digest


def test_the_run_records_its_states_in_order() -> None:
    _, recorder = investigate(fixture_model("valid_diagnosis.json"))

    names = [event.name for event in recorder.events]
    assert names[0] == "investigation_started"
    assert "check_finished" in names
    assert [event.sequence for event in recorder.events] == list(
        range(1, len(recorder.events) + 1)
    )
    assert "EXECUTE_SECOND_CHECK" in {event.state for event in recorder.events}
