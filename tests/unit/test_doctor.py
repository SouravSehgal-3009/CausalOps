from pathlib import Path

from fake_machine import FAKE_API_KEY, HEALTHY_ENVIRONMENT, FakeProbe

from causalops.cli import render_report
from causalops.doctor import (
    ADVISORY_AVAILABLE_MEMORY_BYTES,
    MINIMUM_FREE_DISK_BYTES,
    MINIMUM_TOTAL_MEMORY_BYTES,
    CheckResult,
    CheckStatus,
    DoctorReasonCode,
    DoctorReport,
    ProjectPaths,
    find_project_root,
    run_doctor,
)


def check_named(report: DoctorReport, name: str) -> CheckResult:
    return next(check for check in report.checks if check.name == name)


def test_healthy_machine_passes_every_check(tmp_path: Path) -> None:
    report = run_doctor(ProjectPaths(root=tmp_path), FakeProbe(), HEALTHY_ENVIRONMENT)

    assert [check.name for check in report.checks] == [
        "windows_version",
        "total_memory",
        "available_memory",
        "free_disk",
        "writable_directories",
        "docker",
        "api_key",
    ]
    assert report.failures == ()
    assert all(check.status is CheckStatus.PASS for check in report.checks)


def test_healthy_defaults_sit_above_every_threshold() -> None:
    """Keeps the boundary tests below meaningful: they must set their own values."""
    probe = FakeProbe()

    assert probe.total_memory > MINIMUM_TOTAL_MEMORY_BYTES
    assert probe.available_memory > ADVISORY_AVAILABLE_MEMORY_BYTES
    assert probe.free_disk > MINIMUM_FREE_DISK_BYTES


def test_non_windows_machine_fails(tmp_path: Path) -> None:
    report = run_doctor(
        ProjectPaths(root=tmp_path), FakeProbe(build=None), HEALTHY_ENVIRONMENT
    )

    check = check_named(report, "windows_version")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.UNSUPPORTED_OS


def test_windows_10_build_fails(tmp_path: Path) -> None:
    report = run_doctor(
        ProjectPaths(root=tmp_path), FakeProbe(build=19045), HEALTHY_ENVIRONMENT
    )

    assert check_named(report, "windows_version").reason_code is (
        DoctorReasonCode.UNSUPPORTED_OS
    )


def test_total_memory_below_threshold_fails(tmp_path: Path) -> None:
    probe = FakeProbe(total_memory=MINIMUM_TOTAL_MEMORY_BYTES - 1)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "total_memory")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.INSUFFICIENT_TOTAL_MEMORY


def test_total_memory_exactly_at_threshold_passes(tmp_path: Path) -> None:
    probe = FakeProbe(total_memory=MINIMUM_TOTAL_MEMORY_BYTES)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    assert check_named(report, "total_memory").status is CheckStatus.PASS


def test_total_memory_failure_message_carries_raw_bytes(tmp_path: Path) -> None:
    """Rounded GiB on both sides can print as equal while the comparison fails."""
    probe = FakeProbe(total_memory=MINIMUM_TOTAL_MEMORY_BYTES - 1)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    message = check_named(report, "total_memory").message
    assert f"{MINIMUM_TOTAL_MEMORY_BYTES - 1:,} bytes" in message
    assert f"{MINIMUM_TOTAL_MEMORY_BYTES:,} bytes" in message


def test_unreadable_total_memory_fails_with_a_stable_code(tmp_path: Path) -> None:
    probe = FakeProbe(total_memory=OSError("memory unavailable"))

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "total_memory")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.SYSTEM_READ_FAILED


def test_low_available_memory_warns_but_does_not_fail(tmp_path: Path) -> None:
    probe = FakeProbe(available_memory=ADVISORY_AVAILABLE_MEMORY_BYTES - 1)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "available_memory")
    assert check.status is CheckStatus.WARN
    assert check.reason_code is DoctorReasonCode.LOW_AVAILABLE_MEMORY
    assert report.failures == ()


def test_available_memory_exactly_at_threshold_passes(tmp_path: Path) -> None:
    probe = FakeProbe(available_memory=ADVISORY_AVAILABLE_MEMORY_BYTES)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    assert check_named(report, "available_memory").status is CheckStatus.PASS


def test_unreadable_available_memory_warns_and_never_fails(tmp_path: Path) -> None:
    """Section 3 keeps this check advisory, so even a failed reading must not fail."""
    probe = FakeProbe(available_memory=OSError("counter unavailable"))

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "available_memory")
    assert check.status is CheckStatus.WARN
    assert check.reason_code is DoctorReasonCode.SYSTEM_READ_FAILED
    assert report.failures == ()


def test_free_disk_below_threshold_fails(tmp_path: Path) -> None:
    probe = FakeProbe(free_disk=MINIMUM_FREE_DISK_BYTES - 1)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "free_disk")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.INSUFFICIENT_FREE_DISK


def test_free_disk_exactly_at_threshold_passes(tmp_path: Path) -> None:
    probe = FakeProbe(free_disk=MINIMUM_FREE_DISK_BYTES)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    assert check_named(report, "free_disk").status is CheckStatus.PASS


def test_free_disk_is_measured_on_the_project_root_drive(tmp_path: Path) -> None:
    probe = FakeProbe()

    run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    assert probe.disk_paths == [Path(tmp_path.anchor)]


def test_unreadable_free_disk_fails_with_a_stable_code(tmp_path: Path) -> None:
    probe = FakeProbe(free_disk=FileNotFoundError("volume unavailable"))

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "free_disk")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.SYSTEM_READ_FAILED


def test_missing_docker_fails(tmp_path: Path) -> None:
    report = run_doctor(
        ProjectPaths(root=tmp_path), FakeProbe(docker=False), HEALTHY_ENVIRONMENT
    )

    check = check_named(report, "docker")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.DOCKER_UNAVAILABLE


def test_missing_api_key_fails(tmp_path: Path) -> None:
    report = run_doctor(ProjectPaths(root=tmp_path), FakeProbe(), {})

    check = check_named(report, "api_key")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.MISSING_API_KEY


def test_blank_api_key_fails(tmp_path: Path) -> None:
    report = run_doctor(
        ProjectPaths(root=tmp_path), FakeProbe(), {"ANTHROPIC_API_KEY": "   "}
    )

    assert check_named(report, "api_key").reason_code is (
        DoctorReasonCode.MISSING_API_KEY
    )


def test_api_key_value_never_appears_in_the_report(tmp_path: Path) -> None:
    report = run_doctor(ProjectPaths(root=tmp_path), FakeProbe(), HEALTHY_ENVIRONMENT)

    assert FAKE_API_KEY not in report.model_dump_json()
    assert FAKE_API_KEY not in render_report(report)


def test_missing_run_directories_are_created(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)

    report = run_doctor(paths, FakeProbe(), HEALTHY_ENVIRONMENT)

    assert check_named(report, "writable_directories").status is CheckStatus.PASS
    assert paths.runs.is_dir()
    assert paths.results.is_dir()


def test_scratch_file_leaves_nothing_behind(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)

    run_doctor(paths, FakeProbe(), HEALTHY_ENVIRONMENT)

    assert list(paths.runs.iterdir()) == []
    assert list(paths.results.iterdir()) == []


def test_scratch_file_handles_a_non_ascii_utf8_path(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path / "prüfung ✓")

    report = run_doctor(paths, FakeProbe(), HEALTHY_ENVIRONMENT)

    assert check_named(report, "writable_directories").status is CheckStatus.PASS
    assert paths.runs.is_dir()


def test_unwritable_run_directory_fails(tmp_path: Path) -> None:
    blocking_file = tmp_path / "root-is-a-file"
    blocking_file.write_text("not a directory", encoding="utf-8")

    report = run_doctor(
        ProjectPaths(root=blocking_file), FakeProbe(), HEALTHY_ENVIRONMENT
    )

    check = check_named(report, "writable_directories")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.RUN_DIRECTORY_NOT_WRITABLE


def test_every_failure_is_reported_together(tmp_path: Path) -> None:
    probe = FakeProbe(build=19045, total_memory=1, free_disk=1, docker=False)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, {})

    assert {check.reason_code for check in report.failures} == {
        DoctorReasonCode.UNSUPPORTED_OS,
        DoctorReasonCode.INSUFFICIENT_TOTAL_MEMORY,
        DoctorReasonCode.INSUFFICIENT_FREE_DISK,
        DoctorReasonCode.DOCKER_UNAVAILABLE,
        DoctorReasonCode.MISSING_API_KEY,
    }


def test_find_project_root_walks_up_to_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    nested = tmp_path / "src" / "causalops"
    nested.mkdir(parents=True)

    assert find_project_root(nested) == tmp_path


def test_find_project_root_returns_none_when_there_is_no_pyproject(
    tmp_path: Path,
) -> None:
    assert not any(
        (directory / "pyproject.toml").is_file()
        for directory in (tmp_path, *tmp_path.parents)
    )

    assert find_project_root(tmp_path) is None


def test_project_paths_keep_windows_drive_and_separators(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)

    assert paths.runs.anchor == tmp_path.anchor
    assert paths.runs.name == "runs"
    assert paths.runs.parent == tmp_path
