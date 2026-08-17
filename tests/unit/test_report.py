from fake_incident import INCIDENT_ID, WINDOW_END, WINDOW_START, packet_evidence

from causalops.domain import (
    Budgets,
    Disposition,
    FinalAssessment,
    InvestigationReport,
    ModelDisposition,
    PolicyResult,
    ReasonCode,
    RootCauseCode,
    ToolOutcome,
    ToolReceipt,
    Versions,
)
from causalops.report import REPLAY_CAVEAT, render_report

VERSIONS = Versions(prompt_version="1", policy_version="1", tool_registry_version="1")


def diagnosed_report(cited: tuple[str, ...]) -> InvestigationReport:
    return InvestigationReport(
        investigation_id="inv-1",
        incident_id=INCIDENT_ID,
        disposition=Disposition.DIAGNOSED,
        root_cause=RootCauseCode.CONFIG_CHANGE,
        assessment=FinalAssessment(
            disposition=ModelDisposition.DIAGNOSED,
            root_cause=RootCauseCode.CONFIG_CHANGE,
            supporting_evidence_ids=cited,
            uncertainty="the change list is short",
            next_step="ask the owner to confirm the setting",
        ),
        budgets=Budgets(),
        versions=VERSIONS,
        started_at=WINDOW_START,
        finished_at=WINDOW_END,
        latency_ms=1200,
        model_calls_used=3,
        repairs_used=0,
        tools_executed=2,
        invalid_responses=0,
        final_context_digest="a" * 64,
    )


def failed_safe_report() -> InvestigationReport:
    return InvestigationReport(
        investigation_id="inv-2",
        incident_id=INCIDENT_ID,
        disposition=Disposition.FAILED_SAFE,
        root_cause=RootCauseCode.UNDETERMINED,
        reason_code=ReasonCode.MODEL_OUTPUT_INVALID,
        budgets=Budgets(),
        versions=VERSIONS,
        started_at=WINDOW_START,
        finished_at=WINDOW_END,
        latency_ms=400,
        model_calls_used=2,
        repairs_used=1,
        tools_executed=0,
        invalid_responses=2,
        final_context_digest="b" * 64,
    )


def denied_receipt() -> ToolReceipt:
    return ToolReceipt(
        receipt_id="receipt-1",
        incident_id=INCIDENT_ID,
        tool="query_logs",  # type: ignore[arg-type]
        fingerprint="f" * 8,
        policy_result=PolicyResult.DENIED,
        outcome=ToolOutcome.NOT_EXECUTED,
        reason_code=ReasonCode.UNKNOWN_SERVICE,
        requested_at=WINDOW_START,
        duration_ms=0,
    )


def test_a_diagnosis_report_shows_what_it_rests_on() -> None:
    symptom, topology = packet_evidence()

    text = render_report(
        diagnosed_report((symptom.evidence_id,)), [symptom, topology], [], "replay"
    )

    assert "# Investigation inv-1" in text
    assert "**DIAGNOSED**" in text
    assert "**CONFIG_CHANGE**" in text
    assert symptom.evidence_id in text
    assert topology.evidence_id not in text
    assert "2 evidence records were collected" in text


def test_a_replay_run_says_it_is_not_evidence_of_accuracy() -> None:
    text = render_report(diagnosed_report(("evidence-1",)), [], [], "replay")

    assert REPLAY_CAVEAT in text


def test_a_safe_failure_explains_itself_and_cites_nothing() -> None:
    text = render_report(failed_safe_report(), [], [], "replay")

    assert "**FAILED_SAFE**" in text
    assert "`MODEL_OUTPUT_INVALID`" in text
    assert "No evidence was cited." in text
    assert "protected itself" in text


def test_every_check_appears_including_the_denied_ones() -> None:
    text = render_report(
        diagnosed_report(("evidence-1",)), [], [denied_receipt()], "replay"
    )

    assert "| `query_logs` | DENIED | NOT_EXECUTED | `UNKNOWN_SERVICE` | 0 ms |" in text


def test_the_budget_section_reports_what_was_spent() -> None:
    text = render_report(diagnosed_report(("evidence-1",)), [], [], "replay")

    assert "- Model calls: 3 of 4" in text
    assert "- Checks executed: 2 of 2" in text
    assert "- Token usage: not reported by this model" in text


def test_the_report_carries_no_evaluator_words() -> None:
    text = render_report(diagnosed_report(("evidence-1",)), [], [], "replay").lower()

    for evaluator_word in ("expected", "predicate", "seed", "scenario"):
        assert evaluator_word not in text
