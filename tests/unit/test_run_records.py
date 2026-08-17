import json
from pathlib import Path

import pytest
from fake_incident import (
    FIXTURE_DIR,
    StepClock,
    alert_packet,
    check_runner,
    incident_scope,
    packet_evidence,
)

from causalops.domain import Budgets, ReasonCode
from causalops.models import ReplayReasoningModel
from causalops.run_records import (
    RunRecorder,
    RunRecordError,
    finalize_investigation,
)
from causalops.workflow import InvestigationResult, run_investigation


def finished_run() -> tuple[InvestigationResult, RunRecorder]:
    clock = StepClock()
    recorder = RunRecorder(clock)
    result = run_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json"),
        check_runner(),
        recorder,
        Budgets(),
        clock,
    )
    return result, recorder


def finalize(results_root: Path) -> tuple[Path, InvestigationResult]:
    result, recorder = finished_run()
    written = finalize_investigation(
        results_root,
        result.report,
        recorder.events,
        result.evidence,
        result.receipts,
        "# Investigation\n",
    )
    return written, result


def test_events_are_numbered_and_time_stamped() -> None:
    recorder = RunRecorder(StepClock())
    recorder.event("CREATED", "investigation_started", incident="inc-1")
    recorder.event("FINAL_ASSESSMENT", "stage_started", stage="final_assessment")

    first, second = recorder.events
    assert (first.sequence, second.sequence) == (1, 2)
    assert first.fields == {"incident": "inc-1"}
    assert second.at > first.at


def test_finalizing_writes_the_report_and_its_artifacts(tmp_path: Path) -> None:
    written, result = finalize(tmp_path)

    assert written == tmp_path / "investigations" / result.report.investigation_id
    report = json.loads((written / "report.json").read_text(encoding="utf-8"))
    assert report["disposition"] == "DIAGNOSED"
    assert report["schema_version"] == "1"
    assert (
        len((written / "evidence.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    )
    assert (
        len((written / "receipts.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    )
    assert (written / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert (written / "report.md").read_text(encoding="utf-8") == "# Investigation\n"


def test_finalizing_leaves_no_staging_directory_behind(tmp_path: Path) -> None:
    finalize(tmp_path)

    leftovers = [path.name for path in (tmp_path / "investigations").iterdir()]
    assert not [name for name in leftovers if name.startswith(".staging")]


def test_a_finalized_investigation_is_never_overwritten(tmp_path: Path) -> None:
    written, result = finalize(tmp_path)
    original = (written / "report.json").read_text(encoding="utf-8")

    with pytest.raises(RunRecordError) as refused:
        finalize_investigation(tmp_path, result.report, (), (), (), "# Investigation\n")

    assert refused.value.reason_code is ReasonCode.RESULT_ALREADY_FINALIZED
    assert (written / "report.json").read_text(encoding="utf-8") == original


def test_artifacts_are_utf8_json_lines(tmp_path: Path) -> None:
    written, _ = finalize(tmp_path)

    lines = (written / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert {record["incident_id"] for record in records} == {
        incident_scope().incident_id
    }
    assert all(record["schema_version"] == "1" for record in records)


def test_results_live_under_the_investigation_id(tmp_path: Path) -> None:
    written, result = finalize(tmp_path)

    assert written.parent.name == "investigations"
    assert written.name == result.report.investigation_id
    assert written.is_dir()
