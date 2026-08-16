"""Reads facts about the local machine for `causalops doctor`."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

DOCKER_TIMEOUT_SECONDS = 10


class SystemProbe:
    """The only code that touches the real machine, so tests inject a fake instead."""

    def windows_build(self) -> int | None:
        """Windows build number, or None when this is not Windows at all."""
        if platform.system() != "Windows":
            return None
        return sys.getwindowsversion().build

    def total_memory_bytes(self) -> int:
        return int(psutil.virtual_memory().total)

    def available_memory_bytes(self) -> int:
        return int(psutil.virtual_memory().available)

    def free_disk_bytes(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def docker_responds(self) -> bool:
        """True when `docker version` exits 0 within the timeout."""
        try:
            completed = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=DOCKER_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0
