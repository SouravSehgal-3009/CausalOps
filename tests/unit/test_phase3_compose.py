"""Hermetic checks for the VM-only Ollama Compose image guard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# run_compose.sh is a bash script (`#!/usr/bin/env bash`, bash arrays,
# `[[ ]]`, `BASH_SOURCE`) that docs/mcp-approval/RESTRICTED_HANDOFF.md
# documents as VM-only, never meant to run on a Windows host at all.
# Every test here execs it
# directly via subprocess.run([str(SCRIPT), ...]); Windows has no shebang
# dispatch at the CreateProcess level, so that always raises
# `OSError: [WinError 193] %1 is not a valid Win32 application` -- not a
# flake, a structural mismatch between what this file tests and what
# Windows can run.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "run_compose.sh is a VM-only bash script; Windows cannot exec it "
        "directly (no shebang dispatch), and it is not meant to run there"
    ),
)

SCRIPT = Path(__file__).parents[2] / "infra" / "ollama" / "run_compose.sh"
VALID_IMAGE = "registry.example.invalid/ollama@sha256:" + "a" * 64


def write_env(path: Path, image: str) -> None:
    path.write_text(f"CAUSALOPS_OLLAMA_IMAGE={image}\n", encoding="utf-8")


def run_guard(
    env_file: Path, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), "--env-file", str(env_file), *arguments],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def test_compose_guard_accepts_one_digest_qualified_image(tmp_path: Path) -> None:
    env_file = tmp_path / "compose.env"
    write_env(env_file, VALID_IMAGE)

    result = run_guard(env_file, "--validate")

    assert result.returncode == 0


@pytest.mark.parametrize(
    "image",
    ["ollama/ollama:latest", "registry.example.invalid/ollama@sha256:not-a-digest"],
)
def test_compose_guard_rejects_mutable_or_malformed_images(
    tmp_path: Path, image: str
) -> None:
    env_file = tmp_path / "compose.env"
    write_env(env_file, image)

    result = run_guard(env_file, "--validate")

    assert result.returncode == 2
    assert "immutable @sha256" in result.stderr


@pytest.mark.parametrize(
    "contents",
    [
        f"CAUSALOPS_OLLAMA_IMAGE={VALID_IMAGE}\n"
        "CAUSALOPS_OLLAMA_IMAGE = ollama/ollama:latest\n",
        f"CAUSALOPS_OLLAMA_IMAGE={VALID_IMAGE}\n"
        "CAUSALOPS_OLLAMA_IMAGE: ollama/ollama:latest\n",
        f"CAUSALOPS_OLLAMA_IMAGE={VALID_IMAGE}\nOTHER_VARIABLE=value\n",
        'CAUSALOPS_OLLAMA_IMAGE="${UNREVIEWED_IMAGE}"\n',
    ],
)
def test_compose_guard_rejects_alternate_or_additional_dotenv_assignments(
    tmp_path: Path, contents: str
) -> None:
    env_file = tmp_path / "compose.env"
    env_file.write_text(contents, encoding="utf-8")

    result = run_guard(env_file, "--validate")

    assert result.returncode == 2


def test_compose_guard_rejects_non_vm_execution_before_invoking_docker(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "compose.env"
    write_env(env_file, VALID_IMAGE)

    result = run_guard(env_file, "config")

    assert result.returncode == 2
    assert "CAUSALOPS_EXECUTION_ENV=vm" in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ("-f", "override.yml", "config"),
        ("-foverride.yml", "config"),
        ("--file=override.yml", "config"),
        ("--env-file", "override.env", "config"),
        ("--env-file=override.env", "config"),
        ("--project-directory", "unreviewed-project", "config"),
    ],
)
def test_compose_guard_rejects_forwarded_configuration_source_flags(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    env_file = tmp_path / "compose.env"
    write_env(env_file, VALID_IMAGE)
    environment = {**os.environ, "CAUSALOPS_EXECUTION_ENV": "vm"}

    result = run_guard(env_file, *arguments, env=environment)

    assert result.returncode == 2
    assert "configuration-source override" in result.stderr


def test_compose_guard_removes_an_inherited_image_override(tmp_path: Path) -> None:
    env_file = tmp_path / "compose.env"
    write_env(env_file, VALID_IMAGE)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${CAUSALOPS_OLLAMA_IMAGE+x} ]]; then exit 19; fi\n"
        "if [[ ${COMPOSE_FILE+x} || ${COMPOSE_ENV_FILES+x} ]]; then exit 20; fi\n"
        "printf '%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAUSALOPS_EXECUTION_ENV": "vm",
        "CAUSALOPS_OLLAMA_IMAGE": "ollama/ollama:latest",
        "COMPOSE_FILE": "unreviewed.yml",
        "COMPOSE_ENV_FILES": "unreviewed.env",
    }

    result = run_guard(env_file, "config", env=environment)

    assert result.returncode == 0
    assert "--env-file" in result.stdout
    assert "docker-compose.yml config" in result.stdout
