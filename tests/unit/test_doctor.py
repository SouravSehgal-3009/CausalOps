import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
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
        "operating_system",
        "total_memory",
        "available_memory",
        "free_disk",
        "writable_directories",
        "checkpoint_database",
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


def test_windows_10_build_passes_now_that_the_os_gate_is_open(tmp_path: Path) -> None:
    report = run_doctor(
        ProjectPaths(root=tmp_path), FakeProbe(build=19045), HEALTHY_ENVIRONMENT
    )

    check = check_named(report, "operating_system")
    assert check.status is CheckStatus.PASS
    assert "19045" in check.message


def test_windows_without_a_build_number_still_passes(tmp_path: Path) -> None:
    report = run_doctor(
        ProjectPaths(root=tmp_path), FakeProbe(build=None), HEALTHY_ENVIRONMENT
    )

    check = check_named(report, "operating_system")
    assert check.status is CheckStatus.PASS
    assert check.message
    assert "build" not in check.message


def test_linux_on_x86_64_passes_and_says_which_platform_it_judged(
    tmp_path: Path,
) -> None:
    probe = FakeProbe(
        system="Linux", release="6.8.0-45-generic", machine="x86_64", build=None
    )

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "operating_system")
    assert check.status is CheckStatus.PASS
    assert check.message == "Detected Linux 6.8.0-45-generic (x86_64)."
    assert report.failures == ()


def test_linux_on_another_architecture_now_passes(tmp_path: Path) -> None:
    probe = FakeProbe(
        system="Linux", release="6.8.0-45-generic", machine="aarch64", build=None
    )

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "operating_system")
    assert check.status is CheckStatus.PASS
    assert "aarch64" in check.message


def test_darwin_now_passes_and_names_the_platform(tmp_path: Path) -> None:
    probe = FakeProbe(system="Darwin", release="23.6.0", machine="arm64", build=None)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "operating_system")
    assert check.status is CheckStatus.PASS
    assert "Darwin" in check.message


def test_a_blank_operating_system_reading_fails(tmp_path: Path) -> None:
    probe = FakeProbe(system="")

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    check = check_named(report, "operating_system")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.OS_UNREADABLE
    assert check.message
    assert "None" not in check.message
    assert "()" not in check.message


def test_a_blank_release_or_machine_still_passes_with_a_clean_message(
    tmp_path: Path,
) -> None:
    """`release()`/`machine()` can independently return "" even when `system`
    is readable -- the check still only fails on a blank `system`, and the
    message must not show a stray double space, empty parens, or comma.
    """
    both_blank = FakeProbe(system="Linux", release="", machine="", build=None)
    release_blank = FakeProbe(system="Linux", release="", machine="x86_64", build=None)
    machine_blank = FakeProbe(
        system="Linux", release="6.8.0-45-generic", machine="", build=None
    )

    for probe in (both_blank, release_blank, machine_blank):
        report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)
        check = check_named(report, "operating_system")
        assert check.status is CheckStatus.PASS
        assert "  " not in check.message
        assert "()" not in check.message
        assert not check.message.rstrip(".").endswith(",")

    assert (
        check_named(
            run_doctor(ProjectPaths(root=tmp_path), both_blank, HEALTHY_ENVIRONMENT),
            "operating_system",
        ).message
        == "Detected Linux."
    )
    assert (
        check_named(
            run_doctor(ProjectPaths(root=tmp_path), release_blank, HEALTHY_ENVIRONMENT),
            "operating_system",
        ).message
        == "Detected Linux (x86_64)."
    )
    assert (
        check_named(
            run_doctor(ProjectPaths(root=tmp_path), machine_blank, HEALTHY_ENVIRONMENT),
            "operating_system",
        ).message
        == "Detected Linux 6.8.0-45-generic."
    )


def test_the_same_checks_run_on_linux(tmp_path: Path) -> None:
    """Nothing is skipped for want of a platform equivalent."""
    probe = FakeProbe(system="Linux", release="6.8.0", machine="x86_64", build=None)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, HEALTHY_ENVIRONMENT)

    assert len(report.checks) == 8
    assert all(check.status is CheckStatus.PASS for check in report.checks)


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


def test_missing_api_key_warns_but_does_not_fail(tmp_path: Path) -> None:
    # A warning, not a hard failure: `replay` runs entirely without
    # this key, so its absence is surfaced but does not fail `doctor`
    # outright -- `test_doctor_report_with_no_failures_is_ok` (below) is the
    # confinement test proving a report with only this WARN still reports
    # `doctor: OK`.
    report = run_doctor(ProjectPaths(root=tmp_path), FakeProbe(), {})

    check = check_named(report, "api_key")
    assert check.status is CheckStatus.WARN
    assert check.reason_code is DoctorReasonCode.MISSING_API_KEY
    assert not report.failures


def test_blank_api_key_warns(tmp_path: Path) -> None:
    report = run_doctor(
        ProjectPaths(root=tmp_path), FakeProbe(), {"ANTHROPIC_API_KEY": "   "}
    )

    check = check_named(report, "api_key")
    assert check.status is CheckStatus.WARN
    assert check.reason_code is DoctorReasonCode.MISSING_API_KEY


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


def test_a_missing_checkpoint_database_passes_without_creating_one(
    tmp_path: Path,
) -> None:
    """A project that has never run `investigate` has no `checkpoints.db`
    yet -- that is healthy, not a failure, and the check must not create the
    file itself while confirming that (`test_scratch_file_leaves_nothing_
    behind` already pins the sibling guarantee for `writable_directories`;
    this is the same guarantee for the new check)."""
    paths = ProjectPaths(root=tmp_path)

    report = run_doctor(paths, FakeProbe(), HEALTHY_ENVIRONMENT)

    check = check_named(report, "checkpoint_database")
    assert check.status is CheckStatus.PASS
    assert not paths.checkpoints_db.exists()


def test_a_valid_checkpoint_database_passes(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    paths.results.mkdir(parents=True)
    with closing(sqlite3.connect(str(paths.checkpoints_db))) as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        conn.commit()

    report = run_doctor(paths, FakeProbe(), HEALTHY_ENVIRONMENT)

    assert check_named(report, "checkpoint_database").status is CheckStatus.PASS


def test_a_corrupt_checkpoint_database_fails_with_a_stable_code(
    tmp_path: Path,
) -> None:
    """`sqlite3.connect` itself is lazy and succeeds even against a file
    that is not a real database -- the failure only surfaces on first use,
    which is exactly why this check has to run one, not just open the
    file."""
    paths = ProjectPaths(root=tmp_path)
    paths.results.mkdir(parents=True)
    paths.checkpoints_db.write_text("not a real sqlite database", encoding="utf-8")

    report = run_doctor(paths, FakeProbe(), HEALTHY_ENVIRONMENT)

    check = check_named(report, "checkpoint_database")
    assert check.status is CheckStatus.FAIL
    assert check.reason_code is DoctorReasonCode.CHECKPOINT_DATABASE_UNREADABLE


def test_an_os_error_from_is_file_fails_cleanly_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Path.is_file()` itself can raise `OSError` (`PermissionError`) against
    a directory this process cannot even stat -- a root-owned `results/` from
    the Docker lab reaches this in practice, and is exactly why `is_file()`
    sits *inside* `check_checkpoint_database`'s own `try` rather than as a
    guard before it: `run_doctor_command` runs outside `main`'s own
    `try`/`except`, so a check that lets that escape crashes the exact
    command whose purpose is reporting a broken machine without crashing,
    and would discard the whole report -- including `writable_directories`'s
    own correct `RUN_DIRECTORY_NOT_WRITABLE`, found one check earlier. Both
    must survive together, even though the setup below gives each its own
    unrelated cause rather than one shared directory.

    `os.chmod` cannot reproduce that root-owned-directory condition
    portably: on Windows it only toggles the read-only bit and never blocks
    directory traversal, so a chmod-based version of this test silently
    passed hollow there (root cause of the Windows CI break this test
    replaces -- a precondition asserting the setup actually took effect,
    like the one below, is exactly what would have caught it). The
    `PermissionError` is induced directly instead:
    `Path.is_file` is patched to raise only for `checkpoints_db`'s own
    path, real for every other path, so the induced fault is the one line
    `check_checkpoint_database`'s `try` protects, on every platform --
    proving the `try` placement rather than reproducing the real trigger.

    The two failures below now come from two *independent* mechanisms --
    `writable_directories` fails because `paths.runs` occupies a plain file
    (portable: `Path.mkdir(exist_ok=True)` re-raises whenever the existing
    path is not a directory, which is Python-level logic in `pathlib`
    itself, not an OS permission check), `checkpoint_database` fails from
    the monkeypatch above. Neither mechanism can accidentally mask the
    other's failure, which is a stronger proof than the single shared
    `chmod` this replaces."""
    paths = ProjectPaths(root=tmp_path)
    paths.runs.write_text("not a directory", encoding="utf-8")
    assert paths.runs.is_file(), (
        "setup did not leave a plain file at runs/'s own path -- "
        "this test cannot prove anything here"
    )

    real_is_file = Path.is_file

    def is_file_that_raises_for_the_checkpoint_database(
        self: Path, *args: object, **kwargs: object
    ) -> bool:
        if self == paths.checkpoints_db:
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(
        Path, "is_file", is_file_that_raises_for_the_checkpoint_database
    )

    report = run_doctor(paths, FakeProbe(), HEALTHY_ENVIRONMENT)

    directories_check = check_named(report, "writable_directories")
    assert directories_check.status is CheckStatus.FAIL
    assert directories_check.reason_code is DoctorReasonCode.RUN_DIRECTORY_NOT_WRITABLE
    database_check = check_named(report, "checkpoint_database")
    assert database_check.status is CheckStatus.FAIL
    assert database_check.reason_code is (
        DoctorReasonCode.CHECKPOINT_DATABASE_UNREADABLE
    )


def test_every_failure_is_reported_together(tmp_path: Path) -> None:
    probe = FakeProbe(system="", total_memory=1, free_disk=1, docker=False)

    report = run_doctor(ProjectPaths(root=tmp_path), probe, {})

    # `MISSING_API_KEY` is deliberately absent here (a WARN, not a FAIL):
    # `report.failures` counts `FAIL` only, and the missing
    # key still shows up as a `WARN`, asserted separately below.
    assert {check.reason_code for check in report.failures} == {
        DoctorReasonCode.OS_UNREADABLE,
        DoctorReasonCode.INSUFFICIENT_TOTAL_MEMORY,
        DoctorReasonCode.INSUFFICIENT_FREE_DISK,
        DoctorReasonCode.DOCKER_UNAVAILABLE,
    }
    warnings = {
        check.reason_code for check in report.checks if check.status is CheckStatus.WARN
    }
    assert DoctorReasonCode.MISSING_API_KEY in warnings


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


def test_checkpoints_db_names_the_file_cli_py_actually_opens(tmp_path: Path) -> None:
    """Pinned against the literal path, independently of `checkpoints_db`
    itself -- every other test in this module builds its fixture file
    *through* the same accessor it then checks, so a typo in the property
    would rename both sides together and pass unnoticed. `cli.py`'s
    `_sqlite_checkpointer`/`run_decision_command` both call this accessor
    now instead of spelling `root / "results" / "checkpoints.db"`
    independently; this is the one place that literal is still spelled out,
    on purpose, so a drift between the two is visible here."""
    paths = ProjectPaths(root=tmp_path)

    assert paths.checkpoints_db == tmp_path / "results" / "checkpoints.db"
