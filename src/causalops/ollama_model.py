"""VM-only candidate adapter for the local Qwen Ollama experiment.

It has no import-time I/O and is constructible only by a composition root that
has already verified the experimental VM profile.  The default transport is
intentionally standard-library based so this candidate adds no hosted-provider
dependency.  Unit tests can inject ``transport`` and never contact Ollama.
"""

import json
import os
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import BaseModel

from causalops.domain import FinalAssessment, ToolProposal
from causalops.models import (
    ModelRequest,
    ModelResponse,
    ProposedTurn,
    ToolCallingModel,
    parse_response,
)
from causalops.tool_calls import to_tool_call

OLLAMA_QWEN35_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
VM_EXECUTION_ENV_VARIABLE = "CAUSALOPS_EXECUTION_ENV"
VM_EXECUTION_ENV = "vm"
CANDIDATE_EVALUATION_VARIABLE = "CAUSALOPS_CANDIDATE_EVALUATION"

OllamaTransport = Callable[[str, dict[str, object]], dict[str, object]]


class OllamaModelError(RuntimeError):
    """Ollama was unavailable or returned an invalid candidate response."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuses redirects so a loopback request can never leave loopback."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def _candidate_environment_authorized(environment: Mapping[str, str]) -> bool:
    return environment.get(
        VM_EXECUTION_ENV_VARIABLE, ""
    ).strip().lower() == VM_EXECUTION_ENV and environment.get(
        CANDIDATE_EVALUATION_VARIABLE, ""
    ).strip().lower() in {"1", "true", "yes"}


def _validate_loopback_url(base_url: str) -> str:
    """Refuses endpoints outside the private VM's local Ollama service."""
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise OllamaModelError("Ollama candidate endpoint must be HTTP loopback")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise OllamaModelError(
            "Ollama candidate endpoint must not contain a path or query"
        )
    return base_url.rstrip("/")


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # ``urlopen`` installs proxy handlers from HTTP_PROXY/http_proxy by
    # default. An explicit empty handler keeps even a misconfigured VM from
    # forwarding candidate prompts or responses through an external proxy.
    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    # 30s was tuned before "think": false was set above; a real production
    # prompt (full system_text + tool schemas, not a trivial one) measured
    # just over 30s even with thinking disabled, and a repair-turn prompt
    # (original context plus the rejection reason appended) measured over
    # 120s. 200s keeps real margin above both while still failing well
    # inside the 360s investigation wall-clock budget if genuinely stuck.
    with opener.open(request, timeout=200) as response:  # noqa: S310 - loopback only
        decoded: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise OllamaModelError("Ollama response must be a JSON object")
    return decoded


class OllamaQwenToolCallingModel:
    """Qwen adapter for candidate evaluation, never a hosted control-plane model."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        transport: OllamaTransport | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        candidate_environment = environment if environment is not None else os.environ
        if not _candidate_environment_authorized(candidate_environment):
            raise OllamaModelError(
                "Ollama Qwen is available only to VM candidate-evaluation processes"
            )
        self._base_url = _validate_loopback_url(base_url)
        self._transport = transport or _post_json
        self._next_call_id = 0

    def respond(self, request: ModelRequest) -> ModelResponse:
        content = self._request_content(request, FinalAssessment)
        return ModelResponse(content=content)

    def propose[StageModel: BaseModel](
        self, request: ModelRequest, schema: type[StageModel]
    ) -> ProposedTurn[StageModel]:
        content = self._request_content(request, schema)
        parsed, errors = parse_response(schema, content)
        if parsed is None:
            return ProposedTurn(parsed=None, errors=errors, tool_call=(), usage=None)
        proposal = getattr(parsed, "proposal", None)
        if not isinstance(proposal, ToolProposal):
            return ProposedTurn(parsed=parsed, errors="", tool_call=(), usage=None)
        self._next_call_id += 1
        return ProposedTurn(
            parsed=parsed,
            errors="",
            tool_call=(to_tool_call(proposal, f"ollama-call-{self._next_call_id}"),),
            usage=None,
        )

    def _request_content(
        self, request: ModelRequest, schema: type[BaseModel]
    ) -> dict[str, Any]:
        user_content = request.context_text
        if request.repair_errors:
            user_content = (
                f"{user_content}\n\nPrevious output was rejected: "
                f"{request.repair_errors}\nReturn a corrected JSON object only."
            )
        payload: dict[str, object] = {
            "model": OLLAMA_QWEN35_MODEL,
            "stream": False,
            # qwen3.5's default extended "thinking" burns most of its token
            # budget on hidden reasoning before the schema-constrained
            # answer, which is what made every candidate call blow past the
            # 360s investigation wall-clock budget (measured: 400-900+s per
            # call). Disabling it dropped a real call to ~60s with a valid
            # JSON response.
            "think": False,
            "format": schema.model_json_schema(),
            "messages": [
                {"role": "system", "content": request.system_text},
                {"role": "user", "content": user_content},
            ],
        }
        response = self._transport(f"{self._base_url}/api/chat", payload)
        message = response.get("message")
        if not isinstance(message, dict):
            raise OllamaModelError("Ollama response has no message object")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaModelError("Ollama response has no text content")
        try:
            decoded: object = json.loads(content)
        except json.JSONDecodeError as error:
            raise OllamaModelError("Ollama response content is not JSON") from error
        if not isinstance(decoded, dict):
            raise OllamaModelError("Ollama response content must be a JSON object")
        return decoded


def assert_tool_calling_model(model: ToolCallingModel) -> ToolCallingModel:
    """Type-level boundary helper for composition roots and static checks."""
    return model
