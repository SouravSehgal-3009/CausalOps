import json
from pathlib import Path

import pytest
from fake_incident import (
    alert_packet,
    change_row,
    incident_scope,
    packet_evidence,
    write_changes,
)
from fake_machine import FAKE_API_KEY, FakeProbe

from causalops import cli
from causalops.cli import MODEL_CHECK_NOTE, build_parser, exit_code, render_report
from causalops.doctor import CheckResult, CheckStatus, DoctorReasonCode, DoctorReport
from causalops.domain import StoredIncident, ToolOutcome, ToolReceipt
from causalops.scenario_control import LabError, LabReasonCode
from causalops.telemetry import RunPaths

PASSING_CHECK = CheckResult(
    name="docker",
    status=CheckStatus.PASS,
    message="`docker version` succeeded.",
)
WARNING_CHECK = CheckResult(
    name="available_memory",
    status=CheckStatus.WARN,
    reason_code=DoctorReasonCode.LOW_AVAILABLE_MEMORY,
    message="Only 1.9 GiB RAM is available; the lab may be slow below 2.5 GiB.",
)
FAILING_CHECK = CheckResult(
    name="api_key",
    status=CheckStatus.FAIL,
    reason_code=DoctorReasonCode.MISSING_API_KEY,
    message="Set ANTHROPIC_API_KEY in the environment before a live run.",
)


def test_doctor_subcommand_parses() -> None:
    arguments = build_parser().parse_args(["doctor"])

    assert arguments.command == "doctor"


def test_missing_subcommand_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([])

    assert exit_info.value.code == 2


def test_unknown_subcommand_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["investigate"])

    assert exit_info.value.code == 2


def test_passing_and_warning_report_exits_zero() -> None:
    report = DoctorReport(checks=(PASSING_CHECK, WARNING_CHECK))

    assert exit_code(report) == 0


def test_failing_report_exits_one() -> None:
    report = DoctorReport(checks=(PASSING_CHECK, FAILING_CHECK))

    assert exit_code(report) == 1


def test_rendered_report_shows_reason_codes_and_the_summary() -> None:
    report = DoctorReport(checks=(PASSING_CHECK, WARNING_CHECK, FAILING_CHECK))

    text = render_report(report)

    assert "LOW_AVAILABLE_MEMORY" in text
    assert "MISSING_API_KEY" in text
    assert "doctor: FAILED (1 check)" in text
    assert MODEL_CHECK_NOTE in text


def test_rendered_clean_report_says_ok_and_notes_the_missing_model_check() -> None:
    report = DoctorReport(checks=(PASSING_CHECK, WARNING_CHECK))

    text = render_report(report)

    assert "doctor: OK" in text
    assert MODEL_CHECK_NOTE in text


def test_rendered_columns_fit_the_longest_name_and_code() -> None:
    report = DoctorReport(checks=(PASSING_CHECK, FAILING_CHECK))

    first, second = render_report(report).splitlines()[:2]

    assert first.index(PASSING_CHECK.message) == second.index(FAILING_CHECK.message)


def test_the_lab_and_scenario_subcommands_parse() -> None:
    parser = build_parser()

    assert parser.parse_args(["lab", "up"]).action == "up"
    assert parser.parse_args(["lab", "down"]).action == "down"
    started = parser.parse_args(
        ["scenario", "start", "a_family", "--seed", "development"]
    )
    assert (started.family, started.seed) == ("a_family", "development")
    assert parser.parse_args(["scenario", "reset", "abc"]).incident_id == "abc"


def test_investigate_accepts_only_an_id_and_a_model_it_has() -> None:
    parser = build_parser()

    parsed = parser.parse_args(["investigate", "abc", "--model", "replay"])
    assert (parsed.incident_id, parsed.model) == ("abc", "replay")

    with pytest.raises(SystemExit):
        parser.parse_args(["investigate", "abc", "--model", "claude"])
    with pytest.raises(SystemExit):
        parser.parse_args(["scenario", "start", "a_family"])


def test_a_lab_command_reports_a_refusal_with_its_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def no_docker(root: Path) -> None:
        raise LabError(LabReasonCode.DOCKER_UNAVAILABLE, "docker compose did not run")

    monkeypatch.setattr(cli, "lab_up", no_docker)

    assert cli.main(["lab", "up"]) == 1
    assert "FAIL DOCKER_UNAVAILABLE" in capsys.readouterr().out


def test_a_lab_command_that_works_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "lab_down", lambda root: None)

    assert cli.main(["lab", "down"]) == 0
    assert "lab: down" in capsys.readouterr().out


def test_investigating_an_unknown_incident_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli.main(["investigate", "deadbeef", "--model", "replay"]) == 1
    assert "FAIL INCIDENT_NOT_FOUND" in capsys.readouterr().out


def test_main_returns_zero_for_a_healthy_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "SystemProbe", FakeProbe)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)

    assert cli.main(["doctor"]) == 0

    output = capsys.readouterr().out
    assert "doctor: OK" in output
    assert FAKE_API_KEY not in output


def test_main_returns_one_and_prints_the_reason_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "SystemProbe", FakeProbe)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli.main(["doctor"]) == 1
    assert "MISSING_API_KEY" in capsys.readouterr().out


def test_main_reports_a_missing_project_root_without_creating_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "not-a-project"
    workspace.mkdir()
    assert not any(
        (directory / "pyproject.toml").is_file()
        for directory in (workspace, *workspace.parents)
    )
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(cli, "SystemProbe", FakeProbe)

    assert cli.main(["doctor"]) == 1
    assert "PROJECT_ROOT_NOT_FOUND" in capsys.readouterr().out
    assert list(workspace.iterdir()) == []


def test_investigate_runs_an_investigation_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one end-to-end CLI test that actually runs an investigation
    (`tests/integration/test_configuration_change.py`) is docker-marked and
    CI-deselected. Nothing else in the fast suite calls `run_investigation`
    through the CLI -- a `run_logs_check` signature change or a
    `REPLAY_FIXTURE` rename could ship green. This fabricates a
    `StoredIncident`, a log file, and a changes file directly, the way the
    docker-marked test relies on the scenario controller to do, and drives
    `cli.main` through the real graph-orchestrated `investigate` path.
    `lab_diagnosis.json` scripts two checks -- `query_logs` then
    `list_recent_changes` -- so without a `changes.json` for the second, that
    check would return `UNAVAILABLE`.

    `report.tools_executed` cannot prove the second check actually ran: it
    counts reserved slots, not successes, so it reads 2 whether
    `list_recent_changes` succeeded or came back `UNAVAILABLE`. Reading
    `receipts.jsonl` and asserting both outcomes are `EXECUTED` is the only
    check here that would fail if the `changes.json` write above were
    deleted -- following `test_configuration_change.py`'s own pattern for
    reading a finished run's artifacts back off disk."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    scope = incident_scope()
    packet = alert_packet()
    incident = StoredIncident(scope=scope, packet=packet, evidence=packet_evidence())
    paths = RunPaths(root=tmp_path / "runs" / scope.incident_id)
    paths.root.mkdir(parents=True)
    paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
    paths.logs.mkdir(parents=True)
    (paths.logs / "orders.jsonl").write_text(
        json.dumps(
            {
                "at": scope.started_at.isoformat(),
                "request_id": "r1",
                "service": "orders",
                "severity": "error",
                "event": "config_rejected_request",
                "fields": {"config_key": "require_order_token", "detail": "x"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_changes(paths, [change_row(offset=60)])

    exit_status = cli.main(["investigate", scope.incident_id, "--model", "replay"])

    assert exit_status == 0
    printed = capsys.readouterr().out
    assert "DIAGNOSED CONFIG_CHANGE" in printed
    assert "artifacts:" in printed

    investigation_id = printed.strip().splitlines()[-2]
    receipts_file = (
        tmp_path / "results" / "investigations" / investigation_id / "receipts.jsonl"
    )
    receipts = [
        ToolReceipt.model_validate_json(line)
        for line in receipts_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [receipt.outcome for receipt in receipts] == [
        ToolOutcome.EXECUTED,
        ToolOutcome.EXECUTED,
    ]


def test_investigate_reports_a_locked_checkpoint_store_instead_of_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unit 2d: before this unit, `cli._sqlite_checkpointer`'s bare
    `sqlite3.connect(...)` let a locked/unopenable database escape as a raw
    traceback -- measured against exactly this scenario (the database path
    itself occupied by a directory, so `sqlite3.connect` raises
    `OperationalError: unable to open database file` immediately at
    connect time, rather than the lazier corrupt-existing-file case a
    connect-time wrap cannot catch).

    A read-only `results/` directory (via `os.chmod`) proved the same
    connect-time wrap in an earlier version of this test, but only on
    POSIX -- Windows ignores `os.chmod` for directories entirely, which is
    why that version passed CI on Ubuntu and went uncaught on Windows.
    Occupying `checkpoints.db`'s own path with a directory reaches the
    identical `sqlite3.connect()` call inside the identical
    `try`/`except (OSError, sqlite3.Error)` on every platform, since it is
    SQLite's own file-open logic refusing a directory, not an OS
    permission bit."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    scope = incident_scope()
    incident = StoredIncident(
        scope=scope, packet=alert_packet(), evidence=packet_evidence()
    )
    paths = RunPaths(root=tmp_path / "runs" / scope.incident_id)
    paths.root.mkdir(parents=True)
    paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "checkpoints.db").mkdir()
    assert (results_dir / "checkpoints.db").is_dir(), (
        "setup did not leave a directory at checkpoints.db's path -- "
        "this test cannot prove anything here"
    )

    exit_status = cli.main(["investigate", scope.incident_id, "--model", "replay"])

    assert exit_status == 1
    assert "FAIL STORE_UNAVAILABLE" in capsys.readouterr().out


def test_investigate_pauses_and_reports_escalation_without_finalizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unit 2b: `lab_diagnosis.json` proposes `query_logs` then
    `list_recent_changes`; leaving no `changes.json` behind (unlike the
    end-to-end test above, which writes one) makes the second check come
    back `UNAVAILABLE` -- a real `TOOL_UNAVAILABLE` escalation trigger, not
    a scripted one, reached through the exact same real-backend path
    `cli.py` wires for a live incident. This is the CLI's own contract for
    a paused run: print the reason and thread id, exit with a code distinct
    from 0 and 1, and never call `finalize_investigation` -- the mutation
    target for that early return is deleting it. Verified (not guessed): the
    result is not `RESULT_ALREADY_FINALIZED` surfacing as a `RunRecordError`,
    since `finalize_investigation` never even runs -- `result.report` is
    read first, at the `print` two lines below the deleted branch, and
    `EscalatedInvestigation` has no `report` attribute, so mypy catches the
    missing narrowing statically and the deleted branch also fails this test
    at runtime with a raw `AttributeError` instead of a clean escalated
    exit."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    scope = incident_scope()
    packet = alert_packet()
    incident = StoredIncident(scope=scope, packet=packet, evidence=packet_evidence())
    paths = RunPaths(root=tmp_path / "runs" / scope.incident_id)
    paths.root.mkdir(parents=True)
    paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
    paths.logs.mkdir(parents=True)
    (paths.logs / "orders.jsonl").write_text(
        json.dumps(
            {
                "at": scope.started_at.isoformat(),
                "request_id": "r1",
                "service": "orders",
                "severity": "error",
                "event": "config_rejected_request",
                "fields": {"config_key": "require_order_token", "detail": "x"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Deliberately no `write_changes(paths, ...)` call -- the absent
    # `changes.json` is what makes `list_recent_changes` come back
    # `UNAVAILABLE` instead of `EXECUTED`.

    exit_status = cli.main(["investigate", scope.incident_id, "--model", "replay"])

    assert exit_status == cli.EXIT_ESCALATED
    assert exit_status not in (0, 1)
    printed = capsys.readouterr().out
    assert "ESCALATED TOOL_UNAVAILABLE" in printed
    assert "remaining checks:" in printed
    assert not (tmp_path / "results" / "investigations").exists()
