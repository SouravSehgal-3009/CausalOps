"""Provider profiles kept at process composition boundaries.

The investigation graph deliberately knows only ``ToolCallingModel``.  This
module is the small, immutable vocabulary that a CLI, worker, or deployment
composition root can use before constructing that protocol implementation.
Profiles are never written into graph state, evidence, or reports.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ProfileKind(StrEnum):
    """Closed identifiers accepted by provider composition roots."""

    REPLAY_HOSTED = "replay-hosted"
    CLAUDE_LEGACY_DISABLED = "claude-legacy-disabled"
    OLLAMA_QWEN35_EXPERIMENT = "ollama-qwen35-experiment"


@dataclass(frozen=True)
class ModelProfile:
    """An immutable provider capability declaration, not run-time state."""

    kind: ProfileKind
    model_name: str
    hosted_eligible: bool
    vm_only: bool
    experimental: bool


REPLAY_HOSTED = ModelProfile(
    kind=ProfileKind.REPLAY_HOSTED,
    model_name="replay",
    hosted_eligible=True,
    vm_only=False,
    experimental=False,
)
CLAUDE_LEGACY_DISABLED = ModelProfile(
    kind=ProfileKind.CLAUDE_LEGACY_DISABLED,
    model_name="claude-sonnet-5",
    hosted_eligible=False,
    vm_only=False,
    experimental=False,
)
OLLAMA_QWEN35_EXPERIMENT = ModelProfile(
    kind=ProfileKind.OLLAMA_QWEN35_EXPERIMENT,
    model_name="qwen3.5:4b",
    hosted_eligible=False,
    vm_only=True,
    experimental=True,
)

MODEL_PROFILES: Mapping[ProfileKind, ModelProfile] = MappingProxyType(
    {
        profile.kind: profile
        for profile in (
            REPLAY_HOSTED,
            CLAUDE_LEGACY_DISABLED,
            OLLAMA_QWEN35_EXPERIMENT,
        )
    }
)


def profile_for(kind: ProfileKind) -> ModelProfile:
    """Returns the canonical immutable profile for a closed identifier."""
    return MODEL_PROFILES[kind]
