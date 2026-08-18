"""Reasoning models, meaning the model adapters themselves.

The typed contracts live in `domain.py`; this module holds the thing that answers a
stage request. Replay reads checked-in fixtures, and Phase 3 adds the Claude model
behind the same protocol.
"""

import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from causalops.domain import ModelUsage


class Stage(StrEnum):
    INITIAL_PLAN = "initial_plan"
    HYPOTHESIS_UPDATE = "hypothesis_update"
    FINAL_ASSESSMENT = "final_assessment"


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Stage
    system_text: str
    context_text: str
    repair_errors: str | None = None


class ModelResponse(BaseModel):
    """Raw structured content. The caller validates it against the stage schema."""

    model_config = ConfigDict(frozen=True)

    content: dict[str, JsonValue]
    usage: ModelUsage | None = None


class ReasoningModel(Protocol):
    def respond(self, request: ModelRequest) -> ModelResponse: ...


def summarize_errors(error: ValidationError) -> str:
    """A short, non-secret account of why a response did not fit its schema."""
    parts = [
        f"{'.'.join(str(step) for step in item['loc'])}: {item['msg']}"
        for item in error.errors()[:5]
    ]
    return "; ".join(parts)


def parse_response[StageModel: BaseModel](
    schema: type[StageModel], content: dict[str, JsonValue]
) -> tuple[StageModel | None, str]:
    """The stage object, or nothing and a short account of what did not fit."""
    try:
        return schema.model_validate(content), ""
    except ValidationError as error:
        return None, summarize_errors(error)


class ReplayFixtureError(Exception):
    """A fixture does not script the response the workflow just asked for."""


# A checked-in fixture cannot know the opaque IDs and window of a live incident, so
# it writes these five names and the model fills them in. The list is closed: this
# is substitution, not a template language.
CALLER_PLACEHOLDERS = (
    "incident_id",
    "window_start",
    "window_end",
    "symptom_evidence_id",
)
LAST_CHECK_PLACEHOLDER = "evidence_from_last_check"


class ReplayReasoningModel:
    """Deterministic stage responses from one checked-in opaque fixture.

    Asking for the same stage twice returns the next scripted response, which is how
    a repair after an invalid response is scripted.
    """

    def __init__(
        self, fixture_path: Path, substitutions: Mapping[str, str] | None = None
    ) -> None:
        script = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.responses: dict[str, list[dict[str, JsonValue]]] = script["responses"]
        self.used: dict[str, int] = {}
        self.substitutions = dict(substitutions or {})
        unknown = set(self.substitutions) - set(CALLER_PLACEHOLDERS)
        if unknown:
            raise ReplayFixtureError(f"unknown placeholders: {sorted(unknown)}")
        # Kept so tests can inspect exactly what the workflow sent.
        self.requests: list[ModelRequest] = []

    def respond(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        stage = request.stage.value
        scripted = self.responses.get(stage, [])
        index = self.used.get(stage, 0)
        if index >= len(scripted):
            raise ReplayFixtureError(f"fixture has no response {index + 1} for {stage}")
        self.used[stage] = index + 1
        return ModelResponse(content=self.fill(scripted[index], request.context_text))

    def fill(
        self, scripted: dict[str, JsonValue], context_text: str
    ) -> dict[str, JsonValue]:
        text = json.dumps(scripted)
        for name, value in self.substitutions.items():
            text = text.replace(f"{{{{{name}}}}}", value)
        if f"{{{{{LAST_CHECK_PLACEHOLDER}}}}}" in text:
            text = text.replace(
                f"{{{{{LAST_CHECK_PLACEHOLDER}}}}}",
                self.evidence_from_last_check(context_text),
            )
        filled: dict[str, JsonValue] = json.loads(text)
        return filled

    def evidence_from_last_check(self, context_text: str) -> str:
        """The newest evidence ID in the context that the caller did not supply.

        This is the one thing a static script cannot do for itself: cite an ID it
        could only have learned by reading the context it was just handed. Evidence
        is listed oldest first, so the last unfamiliar ID is the newest check result.
        """
        supplied = set(self.substitutions.values())
        listed = [
            line.split()[1]
            for line in context_text.splitlines()
            if line.startswith("- ") and len(line.split()) > 1
        ]
        for evidence_id in reversed(listed):
            if evidence_id not in supplied:
                return evidence_id
        raise ReplayFixtureError(
            "the fixture cites a check result, but no check evidence is in the context"
        )
