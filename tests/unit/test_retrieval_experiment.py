"""Hermetic checks for Phase 4's semantic-retrieval feature gate."""

from pathlib import Path

import pytest
from fake_incident import alert_packet, incident_scope, packet_evidence

from causalops.domain import Budgets, StoredIncident
from causalops.live_setup import _build_tool_registry, build_model_and_registry
from causalops.retrieval_experiment import (
    RAG_EXPERIMENT_ENABLED_VARIABLE,
    RetrievalExperimentDisabledError,
    rag_experiment_enabled,
    require_fts5_only,
)
from causalops.telemetry import RunPaths


@pytest.mark.parametrize("value", ("", "false", "0", "no", "unexpected"))
def test_rag_experiment_is_disabled_unless_explicitly_requested(value: str) -> None:
    assert not rag_experiment_enabled({RAG_EXPERIMENT_ENABLED_VARIABLE: value})
    require_fts5_only({RAG_EXPERIMENT_ENABLED_VARIABLE: value})


@pytest.mark.parametrize("value", ("1", "true", "YES"))
def test_requested_pinecone_path_fails_before_any_adapter_exists(value: str) -> None:
    environment = {RAG_EXPERIMENT_ENABLED_VARIABLE: value}

    assert rag_experiment_enabled(environment)
    with pytest.raises(RetrievalExperimentDisabledError, match="not approved or wired"):
        require_fts5_only(environment)


def test_registry_construction_refuses_semantic_retrieval_before_backend_setup(
    tmp_path: Path,
) -> None:
    """The composition seam cannot bypass the standalone configuration gate."""
    with pytest.raises(RetrievalExperimentDisabledError):
        _build_tool_registry(
            RunPaths(root=tmp_path),
            Budgets(),
            {RAG_EXPERIMENT_ENABLED_VARIABLE: "true"},
        )


def test_replay_composition_uses_the_injected_retrieval_configuration(
    tmp_path: Path,
) -> None:
    scope = incident_scope()
    incident = StoredIncident(
        scope=scope, packet=alert_packet(), evidence=packet_evidence()
    )
    with pytest.raises(RetrievalExperimentDisabledError):
        build_model_and_registry(
            incident,
            RunPaths(root=tmp_path / "runs" / scope.incident_id),
            Budgets(),
            "replay",
            tmp_path / "checkpoints.db",
            {RAG_EXPERIMENT_ENABLED_VARIABLE: "true"},
        )
