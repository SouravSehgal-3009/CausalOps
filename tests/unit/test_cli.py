from pathlib import Path

import pytest
from fake_machine import FAKE_API_KEY, FakeProbe

from causalops import cli
from causalops.cli import MODEL_CHECK_NOTE, build_parser, exit_code, render_report
from causalops.doctor import CheckResult, CheckStatus, DoctorReasonCode, DoctorReport

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
