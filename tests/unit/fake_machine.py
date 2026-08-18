from pathlib import Path

from causalops.doctor import (
    ADVISORY_AVAILABLE_MEMORY_BYTES,
    MINIMUM_FREE_DISK_BYTES,
    MINIMUM_TOTAL_MEMORY_BYTES,
)
from causalops.system_probe import OperatingSystem

FAKE_API_KEY = "sk-ant-fake-value-for-tests-0123456789"
HEALTHY_ENVIRONMENT = {"ANTHROPIC_API_KEY": FAKE_API_KEY}


def take_reading(value: int | OSError) -> int:
    """Fake readings may hold an OSError, which makes that machine read fail."""
    if isinstance(value, OSError):
        raise value
    return value


class FakeProbe:
    """Stands in for SystemProbe so tests never read the real machine.

    Healthy defaults sit well above every threshold, so a test that cares about a
    boundary has to say so.
    """

    def __init__(
        self,
        build: int | None = 26200,
        system: str = "Windows",
        release: str = "11",
        machine: str = "AMD64",
        total_memory: int | OSError = MINIMUM_TOTAL_MEMORY_BYTES * 2,
        available_memory: int | OSError = ADVISORY_AVAILABLE_MEMORY_BYTES * 2,
        free_disk: int | OSError = MINIMUM_FREE_DISK_BYTES * 2,
        docker: bool = True,
    ) -> None:
        self.build = build
        self.system = system
        self.release = release
        self.machine = machine
        self.total_memory = total_memory
        self.available_memory = available_memory
        self.free_disk = free_disk
        self.docker = docker
        self.disk_paths: list[Path] = []

    def operating_system(self) -> OperatingSystem:
        return OperatingSystem(
            system=self.system,
            release=self.release,
            machine=self.machine,
            windows_build=self.build,
        )

    def total_memory_bytes(self) -> int:
        return take_reading(self.total_memory)

    def available_memory_bytes(self) -> int:
        return take_reading(self.available_memory)

    def free_disk_bytes(self, path: Path) -> int:
        self.disk_paths.append(path)
        return take_reading(self.free_disk)

    def docker_responds(self) -> bool:
        return self.docker
