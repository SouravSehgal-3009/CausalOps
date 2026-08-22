"""Reasoning models, meaning the model adapters themselves.

The typed contracts live in `domain.py`; this module holds the thing that answers a
stage request. Replay reads checked-in fixtures, and Phase 3 adds the Claude model
behind the same protocol.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from causalops.domain import ModelUsage, ToolProposal
from causalops.tool_calls import NativeToolCall, summarize_errors, to_tool_call


class Stage(StrEnum):
    INITIAL_PLAN = "initial_plan"
    HYPOTHESIS_UPDATE = "hypothesis_update"
    FINAL_ASSESSMENT = "final_assessment"


class ModelRequest(BaseModel):
    """One stage's full ask.

    `run_id`/`graph_phase`/`model_turn`/`context_digest` (Unit 3b-2) are
    `TECHNICAL_SPEC.md` §5's amended model-request idempotency key --
    `run_id + graph_phase + model_turn + context_digest` -- carried on the
    request itself rather than threaded through `ToolCallingModel.propose`/
    `.respond` as extra parameters, so a live adapter can persist its own
    `PENDING` reservation record from the one object it already has, exactly
    where the spec says that record belongs ("the live adapter persist this
    record before sending a request"). Both the replay adapter and every
    existing test construct a `ModelRequest` only through
    `graph.py`'s `_render_stage_request`, so this is one call site to widen,
    not many -- and `ReplayReasoningModel.respond` matches fixtures on
    `request.stage` alone, so these four fields are inert to replay.
    """

    model_config = ConfigDict(frozen=True)

    stage: Stage
    system_text: str
    context_text: str
    repair_errors: str | None = None
    run_id: str
    graph_phase: str
    model_turn: int
    context_digest: str


class ModelResponse(BaseModel):
    """Raw structured content. The caller validates it against the stage schema."""

    model_config = ConfigDict(frozen=True)

    content: dict[str, JsonValue]
    usage: ModelUsage | None = None


class ReasoningModel(Protocol):
    def respond(self, request: ModelRequest) -> ModelResponse: ...


class ToolCallingModel(Protocol):
    """The `propose`/`respond` shape the graph orchestrator's INVESTIGATE and
    FINAL_ASSESSMENT nodes need. `ReplayToolCallingModel` below is its first
    implementation; a live Claude adapter is the second, which is what
    finally makes this a protocol worth naming rather than a one-off type.
    """

    def propose[StageModel: BaseModel](
        self, request: ModelRequest, schema: type[StageModel]
    ) -> "ProposedTurn[StageModel]": ...
    def respond(self, request: ModelRequest) -> ModelResponse: ...


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


@dataclass(frozen=True)
class ProposedTurn[StageModel: BaseModel]:
    """One INVESTIGATE-stage model turn.

    `tool_call` is the encoded form of `parsed.proposal`, non-empty only
    when the model proposed a check -- a sequence, not a single optional
    value, so the graph can tell "the model proposed nothing" (empty) apart
    from "the model proposed more than one call in a turn" (`len >= 2`),
    which a live provider's native tool-call channel can produce but this
    replay adapter never does. Decoding it back into a `ToolProposal` is the
    graph's job, not this adapter's -- `select_single_tool_call` and
    `parse_tool_call` are the same functions a live provider's tool-call
    message would have to pass through, and the graph's INVESTIGATE node
    calling them itself (rather than this adapter pre-decoding) is what
    proves replay exercises that boundary rather than shortcutting around it.
    """

    parsed: StageModel | None
    errors: str
    tool_call: Sequence[NativeToolCall]
    usage: ModelUsage | None


class ReplayToolCallingModel:
    """Wraps `ReplayReasoningModel` so the graph orchestrator sees the same
    native-tool-call protocol a live Claude adapter would produce, instead of
    reading `InitialPlan.proposal`/`HypothesisUpdate.proposal` off a
    structured response directly.

    `TECHNICAL_SPEC.md` §5 requires the replay adapter to emit an
    `AIMessage.tool_calls`-equivalent message, not merely a structured field
    a live provider happens not to use. `propose()` is that requirement: it
    turns a parsed proposal into a `NativeToolCall` with `to_tool_call`, the
    same encoder a live adapter's decoder (`parse_tool_call`) has to accept.
    `respond()` is a plain pass-through for `FINAL_ASSESSMENT`, which has no
    proposal to encode, so callers use one model object for the whole run.
    """

    def __init__(self, inner: ReplayReasoningModel) -> None:
        self.inner = inner
        self._next_call_id = 0

    @property
    def requests(self) -> list[ModelRequest]:
        """Delegates to the wrapped model so a parity test can compare the
        stage sequence the graph asked for against the loop's, the same way
        it reads `ReplayReasoningModel.requests` directly for the loop."""
        return self.inner.requests

    def respond(self, request: ModelRequest) -> ModelResponse:
        return self.inner.respond(request)

    def propose[StageModel: BaseModel](
        self, request: ModelRequest, schema: type[StageModel]
    ) -> ProposedTurn[StageModel]:
        """`schema` must be `InitialPlan` or `HypothesisUpdate` -- the two
        stages that carry a `proposal: ToolProposal | None` field and
        enforce exactly one of `proposal`/`stop_reason` (`domain.py`).
        `FinalAssessment` has no proposal to offer, so it is asked through
        `respond()` above instead.

        `schema`'s bound is plain `BaseModel`, not a narrower protocol,
        because a `Protocol` naming a pydantic field cannot also share
        `parse_response`'s `BaseModel` bound -- pydantic's model metaclass
        and `Protocol`'s cannot combine on one class. The `.proposal` access
        below is a runtime `isinstance` check instead of a static one; both
        call sites in `graph.py` are internal and pass only the two schemas
        this docstring names.
        """
        response = self.inner.respond(request)
        parsed, errors = parse_response(schema, response.content)
        if parsed is None:
            return ProposedTurn(
                parsed=None, errors=errors, tool_call=(), usage=response.usage
            )
        proposal = getattr(parsed, "proposal", None)
        if not isinstance(proposal, ToolProposal):
            return ProposedTurn(
                parsed=parsed, errors="", tool_call=(), usage=response.usage
            )
        self._next_call_id += 1
        encoded = to_tool_call(proposal, f"call-{self._next_call_id}")
        return ProposedTurn(
            parsed=parsed, errors="", tool_call=(encoded,), usage=response.usage
        )
