"""Local environment checks behind `causalops doctor`."""

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from causalops.system_probe import SystemProbe

# TECHNICAL_OVERVIEW.md section 3 states the memory thresholds in GiB and the disk
# threshold in GB, so memory uses 1024**3 and disk uses 10**9 on purpose.
MINIMUM_TOTAL_MEMORY_BYTES = int(7.5 * 1024**3)
ADVISORY_AVAILABLE_MEMORY_BYTES = int(2.5 * 1024**3)
MINIMUM_FREE_DISK_BYTES = 12 * 10**9

# Windows 11 reports major version 10, so the build number is what separates it.
FIRST_WINDOWS_11_BUILD = 22000

# Supported Linux is a kernel and an architecture, not a distribution. Everything a
# distribution would stand in for -- Python, Docker, the filesystem, memory, disk --
# already has its own check, and each of those says what is actually wrong.
SUPPORTED_LINUX_MACHINES = frozenset({"x86_64", "amd64"})

API_KEY_VARIABLE = "ANTHROPIC_API_KEY"


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class DoctorReasonCode(StrEnum):
    UNSUPPORTED_OS = "UNSUPPORTED_OS"
    INSUFFICIENT_TOTAL_MEMORY = "INSUFFICIENT_TOTAL_MEMORY"
    LOW_AVAILABLE_MEMORY = "LOW_AVAILABLE_MEMORY"
    INSUFFICIENT_FREE_DISK = "INSUFFICIENT_FREE_DISK"
    RUN_DIRECTORY_NOT_WRITABLE = "RUN_DIRECTORY_NOT_WRITABLE"
    DOCKER_UNAVAILABLE = "DOCKER_UNAVAILABLE"
    MISSING_API_KEY = "MISSING_API_KEY"
    SYSTEM_READ_FAILED = "SYSTEM_READ_FAILED"
    PROJECT_ROOT_NOT_FOUND = "PROJECT_ROOT_NOT_FOUND"


class ProjectPaths(BaseModel):
    """Where transient lab state and finalized results live."""

    model_config = ConfigDict(frozen=True)

    root: Path

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def results(self) -> Path:
        return self.root / "results"


def find_project_root(start: Path) -> Path | None:
    """Nearest directory at or above `start` holding pyproject.toml, else None.

    Returning None instead of falling back to `start` keeps doctor from creating
    run directories in whatever directory the command happened to run in.
    """
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return None


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: CheckStatus
    message: str
    reason_code: DoctorReasonCode | None = None


class DoctorReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    checks: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)


def format_gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def format_gb(value: int) -> str:
    return f"{value / 10**9:.1f} GB"


def describe_os_error(error: OSError) -> str:
    return error.strerror or str(error)


def failed_reading(
    name: str,
    reading: str,
    error: OSError,
    status: CheckStatus = CheckStatus.FAIL,
) -> CheckResult:
    """A machine reading that raised, reported as a stable code instead of a crash."""
    return CheckResult(
        name=name,
        status=status,
        reason_code=DoctorReasonCode.SYSTEM_READ_FAILED,
        message=f"Could not read {reading}: {describe_os_error(error)}",
    )


def unsupported_os(message: str) -> CheckResult:
    return CheckResult(
        name="operating_system",
        status=CheckStatus.FAIL,
        reason_code=DoctorReasonCode.UNSUPPORTED_OS,
        message=message,
    )


def check_operating_system(probe: SystemProbe) -> CheckResult:
    """Pass on Windows 11 or Linux x86-64, and say which one this machine is."""
    found = probe.operating_system()
    if found.system == "Windows":
        if found.windows_build is None:
            return unsupported_os("This machine reports Windows with no build number.")
        if found.windows_build < FIRST_WINDOWS_11_BUILD:
            return unsupported_os(
                f"Windows build {found.windows_build} is older than Windows 11."
            )
        return CheckResult(
            name="operating_system",
            status=CheckStatus.PASS,
            message=f"Windows 11 (build {found.windows_build}).",
        )
    if found.system == "Linux":
        if found.machine.lower() not in SUPPORTED_LINUX_MACHINES:
            return unsupported_os(
                f"Linux on {found.machine} is not supported; CausalOps needs x86-64."
            )
        return CheckResult(
            name="operating_system",
            status=CheckStatus.PASS,
            message=f"Linux {found.release} ({found.machine}).",
        )
    return unsupported_os(
        f"{found.system or 'This machine'} is neither Windows 11 nor Linux x86-64."
    )


def check_total_memory(probe: SystemProbe) -> CheckResult:
    try:
        total = probe.total_memory_bytes()
    except OSError as error:
        return failed_reading("total_memory", "total memory", error)
    if total < MINIMUM_TOTAL_MEMORY_BYTES:
        return CheckResult(
            name="total_memory",
            status=CheckStatus.FAIL,
            reason_code=DoctorReasonCode.INSUFFICIENT_TOTAL_MEMORY,
            # Failure messages carry raw bytes because rounded values on both sides
            # can print as equal while the comparison still fails.
            message=(
                f"Detected {format_gib(total)} ({total:,} bytes) total RAM; "
                f"{format_gib(MINIMUM_TOTAL_MEMORY_BYTES)} "
                f"({MINIMUM_TOTAL_MEMORY_BYTES:,} bytes) is required."
            ),
        )
    return CheckResult(
        name="total_memory",
        status=CheckStatus.PASS,
        message=f"Detected {format_gib(total)} total RAM.",
    )


def check_available_memory(probe: SystemProbe) -> CheckResult:
    """Advisory only: available memory changes while the system runs, so this
    check warns and never fails, including when the reading itself fails."""
    try:
        available = probe.available_memory_bytes()
    except OSError as error:
        return failed_reading(
            "available_memory", "available memory", error, status=CheckStatus.WARN
        )
    if available < ADVISORY_AVAILABLE_MEMORY_BYTES:
        return CheckResult(
            name="available_memory",
            status=CheckStatus.WARN,
            reason_code=DoctorReasonCode.LOW_AVAILABLE_MEMORY,
            message=(
                f"Only {format_gib(available)} RAM is available; the lab may be "
                f"slow below {format_gib(ADVISORY_AVAILABLE_MEMORY_BYTES)}."
            ),
        )
    return CheckResult(
        name="available_memory",
        status=CheckStatus.PASS,
        message=f"{format_gib(available)} RAM is available.",
    )


def check_free_disk(probe: SystemProbe, root: Path) -> CheckResult:
    drive = Path(root.anchor)
    try:
        free = probe.free_disk_bytes(drive)
    except OSError as error:
        return failed_reading("free_disk", f"free disk on {drive}", error)
    if free < MINIMUM_FREE_DISK_BYTES:
        return CheckResult(
            name="free_disk",
            status=CheckStatus.FAIL,
            reason_code=DoctorReasonCode.INSUFFICIENT_FREE_DISK,
            message=(
                f"{drive} has {format_gb(free)} ({free:,} bytes) free; "
                f"{format_gb(MINIMUM_FREE_DISK_BYTES)} "
                f"({MINIMUM_FREE_DISK_BYTES:,} bytes) is required."
            ),
        )
    return CheckResult(
        name="free_disk",
        status=CheckStatus.PASS,
        message=f"{drive} has {format_gb(free)} free.",
    )


def try_write_in_directory(directory: Path) -> str | None:
    """Create the directory if needed, write and delete a UTF-8 file, report trouble.

    The scratch file carries a unique name so the check can never delete a finalized
    result that already lives under `results/`.
    """
    scratch_file = directory / f".causalops-write-scratch-{uuid4().hex}.txt"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        scratch_file.write_text("causalops write check ✓\n", encoding="utf-8")
        scratch_file.unlink()
    except OSError as error:
        return describe_os_error(error)
    return None


def check_writable_directories(paths: ProjectPaths) -> CheckResult:
    for directory in (paths.runs, paths.results):
        trouble = try_write_in_directory(directory)
        if trouble is not None:
            return CheckResult(
                name="writable_directories",
                status=CheckStatus.FAIL,
                reason_code=DoctorReasonCode.RUN_DIRECTORY_NOT_WRITABLE,
                message=f"Cannot write inside {directory}: {trouble}",
            )
    return CheckResult(
        name="writable_directories",
        status=CheckStatus.PASS,
        message=f"{paths.runs} and {paths.results} are writable.",
    )


def check_docker(probe: SystemProbe) -> CheckResult:
    if not probe.docker_responds():
        return CheckResult(
            name="docker",
            status=CheckStatus.FAIL,
            reason_code=DoctorReasonCode.DOCKER_UNAVAILABLE,
            message="`docker version` did not succeed. Start Docker Desktop.",
        )
    return CheckResult(
        name="docker",
        status=CheckStatus.PASS,
        message="`docker version` succeeded.",
    )


def check_api_key(environment: Mapping[str, str]) -> CheckResult:
    # Only presence is inspected. The value never reaches a report field or message.
    if environment.get(API_KEY_VARIABLE, "").strip():
        return CheckResult(
            name="api_key",
            status=CheckStatus.PASS,
            message=f"{API_KEY_VARIABLE} is set in the environment.",
        )
    return CheckResult(
        name="api_key",
        status=CheckStatus.FAIL,
        reason_code=DoctorReasonCode.MISSING_API_KEY,
        message=f"Set {API_KEY_VARIABLE} in the environment before a live run.",
    )


def project_root_not_found(start: Path) -> CheckResult:
    return CheckResult(
        name="project_root",
        status=CheckStatus.FAIL,
        reason_code=DoctorReasonCode.PROJECT_ROOT_NOT_FOUND,
        message=(
            f"No pyproject.toml at or above {start}. "
            "Run causalops from the CausalOps project directory."
        ),
    )


def run_doctor(
    paths: ProjectPaths,
    probe: SystemProbe,
    environment: Mapping[str, str],
) -> DoctorReport:
    """Run every local check and report all of them, not just the first failure."""
    return DoctorReport(
        checks=(
            check_operating_system(probe),
            check_total_memory(probe),
            check_available_memory(probe),
            check_free_disk(probe, paths.root),
            check_writable_directories(paths),
            check_docker(probe),
            check_api_key(environment),
        )
    )
