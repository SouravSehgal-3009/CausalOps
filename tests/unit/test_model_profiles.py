"""Provider-profile isolation and local-only adapter tests."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from urllib.request import HTTPRedirectHandler, ProxyHandler

import pytest
from fake_incident import alert_packet, incident_scope, packet_evidence

import causalops.live_setup as live_setup
from causalops.domain import Budgets, FinalAssessment, InitialPlan, StoredIncident
from causalops.live_setup import (
    CANDIDATE_EVALUATION_VARIABLE,
    ENABLE_CLAUDE_VARIABLE,
    VM_EXECUTION_ENV,
    VM_EXECUTION_ENV_VARIABLE,
    ProviderDisabledError,
    build_ollama_candidate_model,
    claude_enabled,
    profile_for_legacy_choice,
)
from causalops.model_profiles import (
    CLAUDE_LEGACY_DISABLED,
    MODEL_PROFILES,
    OLLAMA_QWEN35_EXPERIMENT,
    REPLAY_HOSTED,
    ProfileKind,
    profile_for,
)
from causalops.models import ModelRequest, Stage
from causalops.ollama_model import OllamaQwenToolCallingModel, _post_json
from causalops.telemetry import RunPaths


def request() -> ModelRequest:
    return ModelRequest(
        stage=Stage.FINAL_ASSESSMENT,
        system_text="system",
        context_text="context",
        run_id="run-1",
        graph_phase="FINAL_ASSESSMENT",
        model_turn=1,
        context_digest="digest",
    )


def enable_candidate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VM_EXECUTION_ENV_VARIABLE, VM_EXECUTION_ENV)
    monkeypatch.setenv(CANDIDATE_EVALUATION_VARIABLE, "true")


def test_profiles_are_immutable_and_only_replay_is_hosted_eligible() -> None:
    assert profile_for(ProfileKind.REPLAY_HOSTED) is REPLAY_HOSTED
    assert REPLAY_HOSTED.hosted_eligible is True
    assert CLAUDE_LEGACY_DISABLED.hosted_eligible is False
    assert OLLAMA_QWEN35_EXPERIMENT.hosted_eligible is False
    assert OLLAMA_QWEN35_EXPERIMENT.vm_only is True
    with pytest.raises(FrozenInstanceError):
        REPLAY_HOSTED.model_name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        MODEL_PROFILES[ProfileKind.REPLAY_HOSTED] = REPLAY_HOSTED  # type: ignore[index]


def test_legacy_model_words_resolve_only_at_the_composition_boundary() -> None:
    assert profile_for_legacy_choice("replay") is REPLAY_HOSTED
    assert profile_for_legacy_choice("claude") is CLAUDE_LEGACY_DISABLED


def test_claude_gate_is_fail_closed_when_explicitly_configured() -> None:
    assert claude_enabled({}) is True
    assert claude_enabled({ENABLE_CLAUDE_VARIABLE: "true"}) is True
    for disabled in ("false", "0", "no", "", "unexpected"):
        assert claude_enabled({ENABLE_CLAUDE_VARIABLE: disabled}) is False


def test_disabled_claude_stops_before_credential_or_client_setup(
    tmp_path: Path,
) -> None:
    class DisabledEnvironment:
        def get(self, key: str, default: str | None = None) -> str | None:
            if key == ENABLE_CLAUDE_VARIABLE:
                return "false"
            raise AssertionError(f"disabled Claude read {key!r}")

    scope = incident_scope()
    incident = StoredIncident(
        scope=scope, packet=alert_packet(), evidence=packet_evidence()
    )
    database = tmp_path / "checkpoints.db"

    with pytest.raises(ProviderDisabledError, match="disabled by"):
        live_setup.build_model_and_registry(
            incident,
            RunPaths(root=tmp_path / "runs" / scope.incident_id),
            Budgets(),
            "claude",
            database,
            DisabledEnvironment(),
        )

    assert database.exists() is False


def test_ollama_candidate_refuses_non_vm_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ProviderDisabledError, match="VM-only"):
        build_ollama_candidate_model({})
    with pytest.raises(ProviderDisabledError, match="CANDIDATE_EVALUATION"):
        build_ollama_candidate_model({VM_EXECUTION_ENV_VARIABLE: VM_EXECUTION_ENV})
    enable_candidate_environment(monkeypatch)
    assert (
        build_ollama_candidate_model(
            {
                VM_EXECUTION_ENV_VARIABLE: VM_EXECUTION_ENV,
                CANDIDATE_EVALUATION_VARIABLE: "true",
            }
        ).__class__
        is OllamaQwenToolCallingModel
    )


def test_ollama_candidate_factory_uses_its_injected_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(VM_EXECUTION_ENV_VARIABLE, raising=False)
    monkeypatch.delenv(CANDIDATE_EVALUATION_VARIABLE, raising=False)

    model = build_ollama_candidate_model(
        {
            VM_EXECUTION_ENV_VARIABLE: VM_EXECUTION_ENV,
            CANDIDATE_EVALUATION_VARIABLE: "true",
        }
    )

    assert isinstance(model, OllamaQwenToolCallingModel)


def test_ollama_adapter_uses_an_injected_vm_local_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_candidate_environment(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        return {"message": {"content": '{"disposition": "INSUFFICIENT_EVIDENCE"}'}}

    model = OllamaQwenToolCallingModel(transport=transport)
    response = model.respond(request().model_copy(update={"repair_errors": "bad enum"}))

    assert response.content == {"disposition": "INSUFFICIENT_EVIDENCE"}
    assert calls[0][0] == "http://127.0.0.1:11434/api/chat"
    assert calls[0][1]["model"] == "qwen3.5:4b"
    assert calls[0][1]["format"] == FinalAssessment.model_json_schema()
    messages = calls[0][1]["messages"]
    assert isinstance(messages, list)
    system_message = messages[0]
    assert isinstance(system_message, dict)
    assert system_message["role"] == "system"
    # The adapter appends its own discriminator-field reminder after the
    # caller's system_text; assert the original text is still there
    # unmodified rather than duplicating that reminder's exact wording here.
    assert system_message["content"].startswith("system\n\n")
    assert messages[1] == {
        "role": "user",
        "content": (
            "context\n\nPrevious output was rejected: bad enum\n"
            "Return a corrected JSON object only."
        ),
    }


def test_ollama_proposal_sends_its_stage_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_candidate_environment(monkeypatch)
    calls: list[dict[str, object]] = []

    def transport(_: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"message": {"content": '{"hypotheses": [], "stop_reason": "done"}'}}

    model = OllamaQwenToolCallingModel(transport=transport)
    model.propose(request().model_copy(update={"stage": "initial_plan"}), InitialPlan)

    assert calls[0]["format"] == InitialPlan.model_json_schema()


def test_ollama_adapter_refuses_an_unauthorized_or_remote_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(VM_EXECUTION_ENV_VARIABLE, raising=False)
    monkeypatch.delenv(CANDIDATE_EVALUATION_VARIABLE, raising=False)
    with pytest.raises(Exception, match="VM candidate-evaluation"):
        OllamaQwenToolCallingModel()
    enable_candidate_environment(monkeypatch)
    with pytest.raises(Exception, match="loopback"):
        OllamaQwenToolCallingModel(
            base_url="https://ollama.example",
        )


def test_ollama_default_transport_disables_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_handlers: list[object] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"message": {"content": "{}"}}'

    class Opener:
        def open(self, _: object, *, timeout: int) -> Response:
            assert timeout == 200
            return Response()

    def build_proxy_free_opener(*handlers: object) -> Opener:
        captured_handlers.extend(handlers)
        return Opener()

    monkeypatch.setattr("causalops.ollama_model.build_opener", build_proxy_free_opener)
    _post_json("http://127.0.0.1:11434/api/chat", {"model": "qwen3.5:4b"})

    assert len(captured_handlers) == 2
    assert isinstance(captured_handlers[0], ProxyHandler)
    assert captured_handlers[0].proxies == {}
    assert isinstance(captured_handlers[1], HTTPRedirectHandler)
    assert (
        captured_handlers[1].redirect_request(
            object(), object(), 302, "Found", object(), "https://external.example"
        )
        is None
    )
