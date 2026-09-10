"""Symlink-safe, descriptor-anchored reads of a finalized investigation's
report artifact.

Extracted out of `SqliteReplayControlPlane` so both it and
`FirestoreReplayControlPlane` share exactly one implementation of this
security-sensitive path -- two independently-maintained copies of a
symlink-attack guard is a real way for one to quietly drift and reintroduce
the vulnerability the other still guards against.
"""

import errno
import os
import stat
from pathlib import Path


def validate_report_reference(investigation_id: str, report_artifact: str) -> Path:
    relative_path = Path(report_artifact)
    expected_path = Path(investigation_id) / "report.md"
    if relative_path != expected_path:
        raise ValueError("report_artifact must be the investigation's own report.md")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("report_artifact must stay beneath the artifact root")
    return relative_path


def artifact_path(
    artifacts_root: Path, investigation_id: str, report_artifact: str
) -> Path:
    validate_report_reference(investigation_id, report_artifact)
    if artifacts_root.is_symlink():
        raise ValueError("artifact root must not be a symbolic link")
    root = artifacts_root.resolve()
    investigation_directory = root / investigation_id
    candidate = investigation_directory / "report.md"
    try:
        root_mode = root.lstat().st_mode
        directory_mode = investigation_directory.lstat().st_mode
        report_mode = candidate.lstat().st_mode
    except OSError as error:
        raise ValueError(
            "report_artifact must name an existing regular file"
        ) from error
    if not stat.S_ISDIR(root_mode):
        raise ValueError("artifact root must be a directory")
    if stat.S_ISLNK(directory_mode) or stat.S_ISLNK(report_mode):
        raise ValueError("report_artifact and its directory must not be symbolic links")
    if not stat.S_ISDIR(directory_mode) or not stat.S_ISREG(report_mode):
        raise ValueError("report_artifact must name an existing regular file")
    return candidate


def read_report_snapshot(
    artifacts_root: Path, investigation_id: str, report_artifact: str
) -> str:
    """Read the report through descriptor-anchored, no-follow opens."""
    validate_report_reference(investigation_id, report_artifact)
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise ValueError("platform lacks safe no-follow artifact reads")
    # typeshed omits O_DIRECTORY/O_NOFOLLOW on win32 (they're POSIX-only), so
    # a static `os.O_DIRECTORY` attribute access fails mypy there even though
    # the hasattr guard above already keeps this branch unreachable on that
    # platform. getattr() sidesteps the stub instead of needing a
    # `type: ignore` that would be "unused" on the POSIX runners where the
    # attribute really does exist.
    o_directory: int = getattr(os, "O_DIRECTORY")  # noqa: B009
    o_nofollow: int = getattr(os, "O_NOFOLLOW")  # noqa: B009
    flags = os.O_RDONLY | o_directory | o_nofollow
    root_fd: int | None = None
    directory_fd: int | None = None
    report_fd: int | None = None
    try:
        root_fd = os.open(artifacts_root, flags)
        directory_fd = os.open(investigation_id, flags, dir_fd=root_fd)
        report_fd = os.open(
            "report.md",
            os.O_RDONLY | o_nofollow,
            dir_fd=directory_fd,
        )
        report_stat = os.fstat(report_fd)
        if not stat.S_ISREG(report_stat.st_mode) or report_stat.st_nlink != 1:
            raise ValueError("report_artifact must name an existing regular file")
        with os.fdopen(report_fd, "rb", closefd=True) as report_file:
            report_fd = None
            return report_file.read().decode("utf-8")
    except OSError as error:
        # Darwin reports O_DIRECTORY|O_NOFOLLOW on a directory symlink as
        # ENOTDIR; Linux reports ELOOP. Both mean the anchored walk refused
        # a link rather than following it.
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                "report_artifact and its directory must not be symbolic links"
            ) from error
        raise ValueError("report_artifact is not readable UTF-8") from error
    except UnicodeDecodeError as error:
        raise ValueError("report_artifact is not readable UTF-8") from error
    finally:
        if report_fd is not None:
            os.close(report_fd)
        if directory_fd is not None:
            os.close(directory_fd)
        if root_fd is not None:
            os.close(root_fd)
