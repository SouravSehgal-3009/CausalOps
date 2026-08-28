import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_incident import (
    FIXTURE_DIR,
    RecordingLogsBackend,
    RecordingMetricBackend,
    StepClock,
    alert_packet,
    assessment_json,
    incident_scope,
    metric_proposal,
    packet_evidence,
    plan_json,
    registry_with,
    replay_model,
    update_json,
)

from causalops.domain import Budgets, InvestigationResult, ReasonCode
from causalops.graph import run_graph_investigation
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import (
    RunEvent,
    RunRecorder,
    RunRecordError,
    finalize_investigation,
    write_jsonl,
)


def finished_run() -> tuple[InvestigationResult, RunRecorder]:
    """Driven through the graph orchestrator: this file's subject
    is artifact writing, never which orchestrator produced the run, so it
    re-points at `run_graph_investigation` rather than staying tied to the
    retiring loop. `valid_diagnosis.json` has no `{{...}}` placeholders --
    it is orchestrator-independent, the same file `test_workflow.py` used --
    so no substitution dance is needed to reuse it here."""
    clock = StepClock()
    recorder = RunRecorder(clock)
    model = ReplayToolCallingModel(
        ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
    )
    registry = registry_with(
        run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
    )
    result = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        model,
        registry,
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


def test_a_replay_run_answers_which_window_and_what_was_expected_from_disk_alone(
    tmp_path: Path,
) -> None:
    """The live-reproduction proof this persistence exists for.

    An earlier investigation had to run a fresh,
    paid, live reproduction against the Docker lab to answer "which window
    produced this zero, and what did the model expect to see" for the
    billed evaluation run's own saved artifacts -- neither question was
    answerable from `receipts.jsonl`/`events.jsonl` alone. This proves that
    gap is closed, with a $0 replay run: after `finalize_investigation`
    writes the artifacts to disk, re-reading *only* the JSON files back --
    no live objects, no lab query -- recovers the exact window the metric
    check used, the hypothesis/evidence-gap/expected-observation reasoning
    that led to proposing it, and the join (`proposal_turn` and
    `receipt_id`) tying the proposal record to the receipt that settled it.
    """
    clock = StepClock()
    recorder = RunRecorder(clock)
    proposal = metric_proposal()
    script = {
        "initial_plan": [plan_json(proposal=proposal)],
        "hypothesis_update": [update_json(stop_reason="nothing else to check")],
        "final_assessment": [assessment_json()],
    }
    model = ReplayToolCallingModel(replay_model(tmp_path, script))
    registry = registry_with(run_metric=RecordingMetricBackend())

    result = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        model,
        registry,
        recorder,
        Budgets(),
        clock,
    )
    written = finalize_investigation(
        tmp_path / "results",
        result.report,
        recorder.events,
        result.evidence,
        result.receipts,
        "# Investigation\n",
    )

    receipts_on_disk = [
        json.loads(line)
        for line in (written / "receipts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    events_on_disk = [
        json.loads(line)
        for line in (written / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    # Which window did this query use? -- the receipt on disk carries the
    # exact effective arguments the backend ran, byte for byte the same
    # dump the proposal's own `ToolArguments` would produce.
    (only_receipt,) = receipts_on_disk
    expected_arguments = proposal.arguments.model_dump(mode="json")
    assert only_receipt["arguments"] == expected_arguments
    assert (
        only_receipt["arguments"]["window_start"] == expected_arguments["window_start"]
    )
    assert only_receipt["arguments"]["window_end"] == expected_arguments["window_end"]

    # What did the model expect to see, and why did it ask? -- the
    # `investigate`-side proposal record on disk.
    proposal_recorded = next(
        event for event in events_on_disk if event["name"] == "proposal_recorded"
    )
    assert proposal_recorded["fields"]["arguments"] == expected_arguments
    assert proposal_recorded["fields"]["evidence_gap"] == proposal.evidence_gap
    assert (
        proposal_recorded["fields"]["expected_observation"]
        == proposal.expected_observation
    )
    assert len(proposal_recorded["fields"]["hypotheses"]) >= 2

    # The join: `proposal_turn` ties the proposal record to the outcome
    # event, and `receipt_id` ties the outcome event to the settled receipt
    # that actually carries the arguments above -- all recoverable from
    # `events.jsonl`/`receipts.jsonl` alone, no in-memory state involved.
    check_finished = next(
        event for event in events_on_disk if event["name"] == "check_finished"
    )
    assert check_finished["fields"]["receipt_id"] == only_receipt["receipt_id"]
    assert (
        check_finished["fields"]["proposal_turn"]
        == proposal_recorded["fields"]["proposal_turn"]
    )


def _run_event(sequence: int) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        at=datetime(2026, 1, 1, tzinfo=UTC),
        state="CREATED",
        name="investigation_started",
    )


def test_write_jsonl_replaces_the_target_in_one_atomic_rename(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"

    write_jsonl(target, [_run_event(1), _run_event(2)])

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sequence"] == 1
    # No stray temporary file left beside the real target.
    assert list(tmp_path.iterdir()) == [target]


def test_write_jsonl_is_atomic_a_failed_write_leaves_the_original_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this test exists for: `write_jsonl` used to be
    `path.write_text(...)`, which truncates `path` before writing a byte of
    new content. `evaluate_cli.py`'s `run_evaluation` calls this function on
    the SAME real `records.jsonl` after every completed run in a batch --
    a crash mid-write there used to be able to destroy every already-scored,
    already-paid-for record along with it, not just lose the newest one.

    Simulates that crash by monkeypatching `Path.write_text` to write only
    half its content and then raise -- the same failure shape a killed
    process or a full disk produces mid-write -- and proves the ORIGINAL
    file is completely unaffected, not truncated or corrupted, because the
    write lands in a temporary sibling file and only a successful write is
    ever renamed onto the real target.

    Mutation-critical: reverting `write_jsonl` to the old in-place
    `path.write_text(...)` form makes this test fail (the original content
    is destroyed rather than preserved), confirmed by hand and then
    reverted back.
    """
    target = tmp_path / "records.jsonl"
    original = '{"schema_version": "1", "sequence": 0, "kept": true}\n'
    target.write_text(original, encoding="utf-8")

    real_write_text = Path.write_text

    def failing_write_text(
        self: Path, data: str, *args: object, **kwargs: object
    ) -> int:
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)  # type: ignore[arg-type]
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    with pytest.raises(OSError, match="simulated crash mid-write"):
        write_jsonl(target, [_run_event(1)])

    assert target.read_text(encoding="utf-8") == original
    leftovers = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith("records.jsonl.tmp-")
    ]
    assert leftovers == []
