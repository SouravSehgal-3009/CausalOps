"""Phase 4 retrieval-experiment guardrails.

Pinecone is deliberately absent from the application dependency set.  FTS5 is
the only production retrieval backend until a separately reviewed experiment
adapter and its VM-only composition root exist.  This module makes an attempt
to opt into that future path fail before any client, credential, or network
activity can occur.
"""

from collections.abc import Mapping

RAG_EXPERIMENT_ENABLED_VARIABLE = "RAG_EXPERIMENT_ENABLED"
_AFFIRMATIVE_VALUES = frozenset({"1", "true", "yes"})


class RetrievalExperimentDisabledError(RuntimeError):
    """A process attempted to select the unapproved semantic-retrieval path."""


def rag_experiment_enabled(environment: Mapping[str, str]) -> bool:
    """Return whether the process explicitly requested the experiment.

    The default is disabled, and unknown values are disabled rather than being
    interpreted permissively.  The caller can then fail closed with a useful,
    non-secret-bearing explanation.
    """
    return environment.get(RAG_EXPERIMENT_ENABLED_VARIABLE, "").strip().lower() in (
        _AFFIRMATIVE_VALUES
    )


def require_fts5_only(environment: Mapping[str, str]) -> None:
    """Refuse an unimplemented Pinecone selection before constructing I/O."""
    if rag_experiment_enabled(environment):
        raise RetrievalExperimentDisabledError(
            "Pinecone semantic retrieval is not approved or wired; "
            "RAG_EXPERIMENT_ENABLED must remain false"
        )
