"""Unit 3c: `causalops-evaluate`'s own pure helpers and orchestration.

`run_evaluation`'s real path always builds a live `LiveClaudeModel`
(`causalops.live_setup.build_model_and_registry` with `model_choice="claude"`
hardcoded -- evaluate has no replay mode) and drives real scenario-controller
traffic against a running Docker lab. Neither is available to this fast,
network-free suite (`tests/conftest.py`'s network guard covers the whole
session), so the orchestration test below monkeypatches
`causalops.evaluate_cli.build_model_and_registry`,
`causalops.evaluate_cli.start_scenario`, and
`causalops.evaluate_cli.reset_scenario` -- the same seam-testing approach
`test_live_model.py` already uses for `LiveClaudeModel` itself (a fake
`client=`), applied here to this script's own three external dependencies.
The docker-marked `tests/integration/` suite is where a real lab is
required; this file never needs one.
"""

import hashlib
import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_incident import (
    RecordingLogsBackend,
    RecordingMetricBackend,
    alert_packet,
    incident_scope,
    packet_evidence,
    registry_with,
)

from causalops.domain import (
    Budgets,
    Disposition,
    IncidentScope,
    RootCauseCode,
    StoredIncident,
)
from causalops.evaluate_cli import (
    EVALUATION_FAMILIES,
    _fixture_sha256,
    _git_provenance,
    _load_expected_outcome,
    _new_evaluation_target,
    _run_id_from_events,
    build_parser,
    main,
    render_evaluation_summary,
    run_evaluation,
)
from causalops.evaluation import (
    EvaluationRecord,
    EvaluationSummary,
    summarize_evaluation,
)
from causalops.evidence import new_opaque_id
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunEvent
from causalops.scenario_control import run_paths

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "causalops" / "replay_fixtures"
)


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _init_git_repo(root: Path) -> None:
    _run_git(root, "init", "--quiet")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    _run_git(root, "add", "tracked.txt")
    _run_git(root, "commit", "--quiet", "-m", "initial")


def test_build_parser_accepts_no_arguments() -> None:
    parsed = build_parser().parse_args([])

    assert parsed == build_parser().parse_args([])


def test_git_provenance_reads_a_clean_commit(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    sha, dirty = _git_provenance(tmp_path)

    assert len(sha) == 40
    assert dirty is False


def test_git_provenance_detects_a_dirty_tree(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("two\n", encoding="utf-8")

    _, dirty = _git_provenance(tmp_path)

    assert dirty is True


def test_fixture_sha256_matches_the_scenario_files_own_bytes(tmp_path: Path) -> None:
    scenarios = tmp_path / "lab" / "scenarios"
    scenarios.mkdir(parents=True)
    content = b'{"family": "ambiguous_telemetry"}'
    (scenarios / "ambiguous_telemetry.json").write_bytes(content)

    assert (
        _fixture_sha256(tmp_path, "ambiguous_telemetry")
        == hashlib.sha256(content).hexdigest()
    )


def test_load_expected_outcome_reads_the_evaluator_directory(tmp_path: Path) -> None:
    evaluator_dir = tmp_path / "evaluator"
    evaluator_dir.mkdir()
    (evaluator_dir / "expected.json").write_text(
        json.dumps(
            {
                "seed": "evaluation",
                "family": "configuration_change",
                "root_cause": "CONFIG_CHANGE",
                "disposition": "DIAGNOSED",
                "predicates": [
                    {
                        "source": "list_recent_changes",
                        "kind": "CHANGE",
                        "field": "summaries",
                        "operator": "CONTAINS",
                        "value": "require_order_token",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    expected = _load_expected_outcome(tmp_path)

    assert expected.root_cause is RootCauseCode.CONFIG_CHANGE
    assert expected.disposition is Disposition.DIAGNOSED
    assert len(expected.predicates) == 1
    assert expected.predicates[0].field == "summaries"


def test_run_id_from_events_finds_the_investigation_started_run_id() -> None:
    events = [
        RunEvent(
            sequence=0,
            at=datetime(2026, 8, 24, tzinfo=UTC),
            state="CREATED",
            name="investigation_started",
            fields={"incident": "inc-1", "run_id": "run-abc"},
        )
    ]

    assert _run_id_from_events(events) == "run-abc"


def test_run_id_from_events_raises_when_absent() -> None:
    with pytest.raises(RuntimeError):
        _run_id_from_events([])


def test_run_evaluation_drives_every_family_as_a_baseline_then_tool_enabled_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the orchestration `run_evaluation` is responsible for: one
    `start_scenario(..., "evaluation")` per family, two scored runs per
    incident in baseline-then-tool-enabled order, one `reset_scenario` per
    family even though nothing raises, and a resulting `EvaluationRecord`
    per run carrying the evaluator-only expected outcome -- without ever
    handing that expected outcome to the investigation call itself (this
    script builds it and the model+registry from two entirely separate
    functions; `_run_one` never receives `expected` as investigator input,
    only as a scoring argument after the run has already finished)."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )

    started: list[tuple[str, str]] = []
    reset: list[str] = []

    def fake_start_scenario(root: Path, family: str, seed: str) -> str:
        started.append((family, seed))
        incident_id = new_opaque_id()
        scope = IncidentScope.model_validate(
            {**incident_scope().model_dump(), "incident_id": incident_id}
        )
        packet = alert_packet().model_copy(update={"incident_id": incident_id})
        evidence = tuple(
            record.model_copy(update={"incident_id": incident_id})
            for record in packet_evidence()
        )
        # `alert_packet()`'s own evidence-id fields still point at
        # `packet_evidence()`'s fixed ids -- those ids are unaffected by
        # this incident_id substitution, so the packet stays internally
        # consistent without also rewriting them.
        incident = StoredIncident(scope=scope, packet=packet, evidence=evidence)
        paths = run_paths(root, incident_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
        (paths.root / "evaluator").mkdir(exist_ok=True)
        (paths.root / "evaluator" / "expected.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "family": family,
                    "root_cause": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
                    "disposition": "DIAGNOSED",
                    "predicates": [],
                }
            ),
            encoding="utf-8",
        )
        return incident_id

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        reset.append(incident_id)

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        assert model_choice == "claude"
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    target = _new_evaluation_target(tmp_path)
    records = run_evaluation(tmp_path, target)

    assert [family for family, seed in started] == list(EVALUATION_FAMILIES)
    assert all(seed == "evaluation" for _, seed in started)
    assert len(reset) == len(EVALUATION_FAMILIES)
    assert len(records) == 2 * len(EVALUATION_FAMILIES)
    modes = [record.run_key.rsplit("/", 1)[-1] for record in records]
    assert modes == ["no_tool_baseline", "tool_enabled"] * len(EVALUATION_FAMILIES)
    for record in records:
        assert record.model_name == "fake-claude-model"
        assert record.git_sha == "f" * 40
        assert record.git_dirty is False
        assert (
            record.expected.root_cause
            is RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION
        )


def test_records_already_scored_before_a_crash_survive_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The P1 this fix round exists for: `run_evaluation` must rewrite
    `records.jsonl` after every completed run, not batch everything to the
    end. Proves it by making the run that would build the SECOND family's
    model construction raise -- after the first family's baseline and
    tool-enabled runs have both already completed and been appended -- and
    then reading `records.jsonl` back from disk (not the in-memory
    exception, and not the in-memory `records` list this function never
    even gets to return) to confirm those two already-scored, already-paid-
    for records are durable despite the crash. Reading only the in-memory
    list would not prove anything: the bug this test guards against was
    exactly that records could be correct in memory yet never reach disk
    before a crash discarded them."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )

    def fake_start_scenario(root: Path, family: str, seed: str) -> str:
        incident_id = new_opaque_id()
        scope = IncidentScope.model_validate(
            {**incident_scope().model_dump(), "incident_id": incident_id}
        )
        packet = alert_packet().model_copy(update={"incident_id": incident_id})
        evidence = tuple(
            record.model_copy(update={"incident_id": incident_id})
            for record in packet_evidence()
        )
        incident = StoredIncident(scope=scope, packet=packet, evidence=evidence)
        paths = run_paths(root, incident_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
        (paths.root / "evaluator").mkdir(exist_ok=True)
        (paths.root / "evaluator" / "expected.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "family": family,
                    "root_cause": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
                    "disposition": "DIAGNOSED",
                    "predicates": [],
                }
            ),
            encoding="utf-8",
        )
        return incident_id

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        pass

    call_count = 0

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        nonlocal call_count
        call_count += 1
        # Calls 1 and 2 are the first family's baseline and tool-enabled
        # runs -- let both succeed and land on disk. Call 3 is the SECOND
        # family's baseline run -- raise there, simulating a real mid-batch
        # failure (a crashed request, a killed process) after real, billed
        # work has already produced scoreable results.
        if call_count == 3:
            raise RuntimeError("simulated mid-batch failure on the third run")
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    target = _new_evaluation_target(tmp_path)
    records_path = target / "records.jsonl"

    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        run_evaluation(tmp_path, target)

    assert call_count == 3

    on_disk = [
        EvaluationRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(on_disk) == 2
    modes = [record.run_key.rsplit("/", 1)[-1] for record in on_disk]
    assert modes == ["no_tool_baseline", "tool_enabled"]
    # Both surviving records share one incident_id -- the first family's --
    # confirming they are the pair from before the crash, not some other mix.
    incident_ids = {record.run_key.split("/", 1)[0] for record in on_disk}
    assert len(incident_ids) == 1
    for record in on_disk:
        assert record.model_name == "fake-claude-model"


def test_the_original_run_failure_survives_cleanup_also_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The P2 this fix exists for: `run_evaluation`'s `finally:
    reset_scenario(...)` used to be unconditional -- if the scored run
    inside `try` failed (a real, billed run failure) AND `reset_scenario` in
    `finally` ALSO raised, Python's ordinary exception chaining would
    replace the original exception with the cleanup failure, burying the
    actual reason a billed run failed behind an unrelated lab-reset
    problem. Simulates both failing on the very first family and confirms
    the ORIGINAL run failure is what actually propagates and gets reported
    -- the cleanup failure is only printed, never silently discarded."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )

    def fake_start_scenario(root: Path, family: str, seed: str) -> str:
        incident_id = new_opaque_id()
        scope = IncidentScope.model_validate(
            {**incident_scope().model_dump(), "incident_id": incident_id}
        )
        packet = alert_packet().model_copy(update={"incident_id": incident_id})
        evidence = tuple(
            record.model_copy(update={"incident_id": incident_id})
            for record in packet_evidence()
        )
        incident = StoredIncident(scope=scope, packet=packet, evidence=evidence)
        paths = run_paths(root, incident_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
        (paths.root / "evaluator").mkdir(exist_ok=True)
        (paths.root / "evaluator" / "expected.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "family": family,
                    "root_cause": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
                    "disposition": "DIAGNOSED",
                    "predicates": [],
                }
            ),
            encoding="utf-8",
        )
        return incident_id

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        raise RuntimeError("simulated reset_scenario cleanup failure")

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        raise RuntimeError("simulated billed-run failure")

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    target = _new_evaluation_target(tmp_path)

    with pytest.raises(RuntimeError, match="simulated billed-run failure"):
        run_evaluation(tmp_path, target)

    out = capsys.readouterr().out
    assert "RESET_SCENARIO_FAILED_DURING_CLEANUP" in out
    assert "simulated reset_scenario cleanup failure" in out


def test_render_evaluation_summary_shows_counts_and_ranges_not_a_percentile() -> None:
    """The P2 this fix exists for: `main()` used to print one line per
    record and never a batch-level aggregate at all. `TECHNICAL_SPEC.md`
    §10 forbids p95 or any other broad performance claim from a small
    synthetic sample -- this asserts the rendered text carries the counts
    and ranges an `EvaluationSummary` holds, including the honest "how many
    runs' cost is actually known" annotation from Item 2, and never mentions
    a percentile."""
    summary = EvaluationSummary(
        total_records=2,
        diagnosis_correct_count=1,
        disposition_correct_count=2,
        latency_ms_min=100,
        latency_ms_max=900,
        model_calls_min=1,
        model_calls_max=4,
        tools_executed_min=0,
        tools_executed_max=2,
        input_tokens_min=500,
        input_tokens_max=4000,
        input_tokens_known_count=2,
        output_tokens_min=100,
        output_tokens_max=800,
        output_tokens_known_count=2,
        reserved_usd_min=0.01,
        reserved_usd_max=0.05,
        actual_usd_min=0.008,
        actual_usd_max=0.008,
        actual_usd_known_count=1,
    )

    rendered = render_evaluation_summary(summary)

    assert "evaluation summary: 2 record(s)" in rendered
    assert "diagnosis_correct:   1/2" in rendered
    assert "disposition_correct: 2/2" in rendered
    assert "latency_ms:      100-900" in rendered
    assert "model_calls:     1-4" in rendered
    assert "tools_executed:  0-2" in rendered
    assert "input_tokens:    500-4000 (2/2 known)" in rendered
    assert "output_tokens:   100-800 (2/2 known)" in rendered
    assert "reserved_usd:    0.0100-0.0500" in rendered
    assert "actual_usd:      0.0080-0.0080 (1/2 known)" in rendered
    assert "p95" not in rendered.lower()
    assert "percentile" not in rendered.lower()


def test_render_evaluation_summary_of_an_empty_batch_says_no_data() -> None:
    rendered = render_evaluation_summary(summarize_evaluation([]))

    assert "evaluation summary: 0 record(s)" in rendered
    assert "no data" in rendered


def test_main_writes_a_summary_alongside_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end through `main()`: the aggregate summary this fix adds must
    actually reach both stdout and a saved `summary.json`, not just exist as
    a pure function nothing calls."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)

    def fake_start_scenario(root: Path, family: str, seed: str) -> str:
        incident_id = new_opaque_id()
        scope = IncidentScope.model_validate(
            {**incident_scope().model_dump(), "incident_id": incident_id}
        )
        packet = alert_packet().model_copy(update={"incident_id": incident_id})
        evidence = tuple(
            record.model_copy(update={"incident_id": incident_id})
            for record in packet_evidence()
        )
        incident = StoredIncident(scope=scope, packet=packet, evidence=evidence)
        paths = run_paths(root, incident_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
        (paths.root / "evaluator").mkdir(exist_ok=True)
        (paths.root / "evaluator" / "expected.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "family": family,
                    "root_cause": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
                    "disposition": "DIAGNOSED",
                    "predicates": [],
                }
            ),
            encoding="utf-8",
        )
        return incident_id

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        pass

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "evaluation summary: 8 record(s)" in out
    assert "summary:" in out
    summary_path = (
        next((tmp_path / "results" / "evaluations").iterdir()) / "summary.json"
    )
    assert str(summary_path) in out
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["total_records"] == 8


def test_main_fails_cleanly_when_the_evaluation_target_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The P2 this fix exists for: `_new_evaluation_target` used to run
    before `main`'s `try:` block, so a failure there (read-only filesystem,
    full disk, permission error) escaped as a raw, uncaught traceback
    instead of the clean `FAIL ...` every other refusal path in this script
    already gives. `records.jsonl` genuinely does not exist yet for this
    specific failure, so this only asserts the clean failure message, not a
    'records so far' line."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def failing_target(root: Path) -> Path:
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr("causalops.evaluate_cli._new_evaluation_target", failing_target)

    exit_code = main([])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAIL EVALUATION_TARGET_UNWRITABLE" in out
    assert "simulated read-only filesystem" in out
