"""The semantic-retrieval feature gate.

`RAG_EXPERIMENT_ENABLED` selects which `search_runbooks` backend
`live_setup._build_tool_registry` builds: unset/false (the default) keeps
today's FTS5 path (`runbooks.py`) completely unchanged; `true` builds
`pinecone_runbooks.PineconeRunbookIndex` instead. This module holds only
the flag-reading logic -- no Pinecone import, client, or credential lives
here, so reading the flag itself never risks any I/O.
"""

from collections.abc import Mapping

RAG_EXPERIMENT_ENABLED_VARIABLE = "RAG_EXPERIMENT_ENABLED"
_AFFIRMATIVE_VALUES = frozenset({"1", "true", "yes"})


def rag_experiment_enabled(environment: Mapping[str, str]) -> bool:
    """Return whether the process explicitly requested the Pinecone path.

    The default is disabled, and unknown values are disabled rather than being
    interpreted permissively.
    """
    return environment.get(RAG_EXPERIMENT_ENABLED_VARIABLE, "").strip().lower() in (
        _AFFIRMATIVE_VALUES
    )
