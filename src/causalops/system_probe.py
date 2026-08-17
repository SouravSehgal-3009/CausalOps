"""Reads facts about the local machine for `causalops doctor`."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict

DOCKER_TIMEOUT_SECONDS = 10


class OperatingSystem(BaseModel):
    """What the machine says it is."""

    model_config = ConfigDict(frozen=True)

    system: str
    release: str
    machine: str
    windows_build: int | None = None


class SystemProbe:
    """The only code that touches the real machine, so tests inject a fake instead."""

    def operating_system(self) -> OperatingSystem:
        """The readings the OS check judges, on whichever platform this is.

        The build number is read behind `sys.platform`, which is the comparison
        mypy narrows on. `platform.system()` looks equivalent but narrows nothing,
        so type checking off Windows would fail on a call that cannot exist there.
        """
        build: int | None = None
        if sys.platform == "win32":
            build = sys.getwindowsversion().build
        return OperatingSystem(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            windows_build=build,
        )

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
