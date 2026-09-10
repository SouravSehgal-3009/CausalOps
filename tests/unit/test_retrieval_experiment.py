"""Hermetic checks for the semantic-retrieval feature gate: the flag reader
in `retrieval_experiment.py`, and `_build_tool_registry`'s branch on it.
The Pinecone branch is exercised through `PineconeRunbookIndex`'s own
`client=` injection seam (`tests/unit/test_pinecone_runbooks.py`), not
through a live network call here -- these tests only prove the gate
selects the right backend and fails closed without a credential.
"""

from pathlib import Path

import pytest
from fake_incident import alert_packet, incident_scope, packet_evidence

from causalops.domain import Budgets, StoredIncident
from causalops.live_setup import _build_tool_registry, build_model_and_registry
from causalops.pinecone_runbooks import (
    PINECONE_API_KEY_VARIABLE,
    PineconeRunbookIndexError,
)
from causalops.retrieval_experiment import (
    RAG_EXPERIMENT_ENABLED_VARIABLE,
    rag_experiment_enabled,
)
from causalops.telemetry import RunPaths


@pytest.mark.parametrize("value", ("", "false", "0", "no", "unexpected"))
def test_rag_experiment_is_disabled_unless_explicitly_requested(value: str) -> None:
    assert not rag_experiment_enabled({RAG_EXPERIMENT_ENABLED_VARIABLE: value})


@pytest.mark.parametrize("value", ("1", "true", "YES"))
def test_rag_experiment_reads_every_affirmative_spelling(value: str) -> None:
    assert rag_experiment_enabled({RAG_EXPERIMENT_ENABLED_VARIABLE: value})


def test_disabled_registry_construction_never_needs_a_pinecone_key(
    tmp_path: Path,
) -> None:
    """The default (disabled) path builds the FTS5 registry exactly as
    before -- no `PINECONE_API_KEY` in the environment, no error."""
    registry = _build_tool_registry(
        RunPaths(root=tmp_path),
        Budgets(),
        {RAG_EXPERIMENT_ENABLED_VARIABLE: "false"},
    )

    assert registry


def test_enabled_registry_construction_fails_closed_without_a_pinecone_key(
    tmp_path: Path,
) -> None:
    """Enabling the experiment without `PINECONE_API_KEY` set must refuse
    before any Pinecone network call, not silently fall back to FTS5."""
    with pytest.raises(PineconeRunbookIndexError, match=PINECONE_API_KEY_VARIABLE):
        _build_tool_registry(
            RunPaths(root=tmp_path),
            Budgets(),
            {RAG_EXPERIMENT_ENABLED_VARIABLE: "true"},
        )


def test_replay_composition_propagates_the_same_pinecone_gate_failure(
    tmp_path: Path,
) -> None:
    scope = incident_scope()
    incident = StoredIncident(
        scope=scope, packet=alert_packet(), evidence=packet_evidence()
    )
    with pytest.raises(PineconeRunbookIndexError, match=PINECONE_API_KEY_VARIABLE):
        build_model_and_registry(
            incident,
            RunPaths(root=tmp_path / "runs" / scope.incident_id),
            Budgets(),
            "replay",
            tmp_path / "checkpoints.db",
            {RAG_EXPERIMENT_ENABLED_VARIABLE: "true"},
        )
