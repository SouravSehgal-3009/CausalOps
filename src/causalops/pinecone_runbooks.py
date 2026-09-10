"""Pinecone-backed semantic retrieval for `search_runbooks` -- the opt-in
second backend gated by `RAG_EXPERIMENT_ENABLED`
(`retrieval_experiment.py`). `runbooks.py`'s FTS5 index stays the default
and is untouched by this module; nothing here is imported or constructed
unless the experiment is explicitly enabled.

Mirrors `runbooks.py`'s exact interface: `PineconeRunbookIndex.search(topic,
limit)` returns the same `RunbookPassage` tuple shape, and
`run_runbook_search_pinecone` returns the same `RunbookCheckOutcome` shape
`run_runbook_search` does -- only `retrieval_mode` (`PINECONE_SEMANTIC`
instead of `FTS5_LEXICAL`) and the passages' actual source differ.

`RunbookTopic` stays the closed enum `tools.py` already defines;
`_TOPIC_SEMANTIC_QUERIES` below is the only place a topic becomes a
natural-language query string sent to Pinecone's hosted embedding model --
written by this module, never by a model, the same guarantee
`runbooks.py`'s own `_TOPIC_QUERIES` makes for FTS5 `MATCH` syntax.

The index itself (`causalops-runbooks`, `multilingual-e5-large` hosted
inference, no second embeddings key) was provisioned once, out of band --
this module only searches and upserts against it, it does not create it.
"""

import os
import time
from collections.abc import Mapping
from typing import Protocol, cast

from pinecone import Pinecone
from pinecone.exceptions import PineconeException

from causalops.domain import (
    ReasonCode,
    RetrievalMode,
    RunbookCheckOutcome,
    RunbookPassage,
    ToolOutcome,
)
from causalops.evidence import digest_text
from causalops.tools import RunbookTopic, SearchRunbooksArguments

PINECONE_API_KEY_VARIABLE = "PINECONE_API_KEY"
PINECONE_INDEX_NAME = "causalops-runbooks"
PINECONE_NAMESPACE = "runbooks"

_TOPIC_SEMANTIC_QUERIES: dict[RunbookTopic, str] = {
    RunbookTopic.GATEWAY_ERRORS: (
        "gateway error rate spike, narrow versus broad, cause and triage"
    ),
    RunbookTopic.GATEWAY_LATENCY: (
        "elevated gateway latency without a matching error spike, slow requests"
    ),
    RunbookTopic.DOWNSTREAM_TIMEOUTS: (
        "downstream service timeouts and retry amplification"
    ),
    RunbookTopic.RESOURCE_POOL_PRESSURE: (
        "connection or worker pool exhaustion, queue wait time, pool pressure"
    ),
    RunbookTopic.RECENT_CONFIG_CHANGES: (
        "recent configuration or rollout change correlated with an incident"
    ),
}


class PineconeRunbookIndexError(RuntimeError):
    """Construction failed: missing credential or an unreachable index."""


class _SearchHit(Protocol):
    id: str
    score: float
    fields: Mapping[str, object]


class _SearchResult(Protocol):
    hits: list[_SearchHit]


class _SearchResponse(Protocol):
    result: _SearchResult


class PineconeSearchClient(Protocol):
    """The one Pinecone SDK surface this module actually calls -- narrow
    enough for a unit test to fake directly instead of the whole SDK."""

    def search(
        self, *, namespace: str, top_k: int, inputs: Mapping[str, str]
    ) -> _SearchResponse: ...


class PineconeRunbookIndex:
    """A thin wrapper around one Pinecone index's `search`, scoped to this
    project's runbook namespace.

    Construction resolves `PINECONE_API_KEY` from `environment` (defaulting
    to `os.environ` -- the same pattern every other credential in this
    codebase uses, no `.env` loader) and fails loudly before any network
    call if it is missing, matching `RunbookIndex`'s own "fail at
    construction, not per search" posture for its corpus file. Passing
    `client` directly (a `PineconeSearchClient`) skips credential
    resolution and the SDK entirely -- how the unit tests exercise this
    class against a fake.
    """

    def __init__(
        self,
        client: PineconeSearchClient | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        index_name: str = PINECONE_INDEX_NAME,
        namespace: str = PINECONE_NAMESPACE,
    ) -> None:
        self._namespace = namespace
        if client is not None:
            self._client: PineconeSearchClient = client
            return
        env = environment if environment is not None else os.environ
        api_key = env.get(PINECONE_API_KEY_VARIABLE, "").strip()
        if not api_key:
            raise PineconeRunbookIndexError(
                f"{PINECONE_API_KEY_VARIABLE} is not set; refusing before any "
                "Pinecone network call"
            )
        pc = Pinecone(api_key=api_key)
        try:
            host = pc.describe_index(index_name)["host"]
            # `pc.Index()` is typed `Index | GrpcIndex` -- this project never
            # requests the grpc variant, so the real return value always has
            # the `search()` method `PineconeSearchClient` requires.
            self._client = cast("PineconeSearchClient", pc.Index(host=host))
        except PineconeException as exc:
            raise PineconeRunbookIndexError(
                f"could not open Pinecone index {index_name!r}: {exc}"
            ) from exc

    def search(self, topic: RunbookTopic, limit: int) -> tuple[RunbookPassage, ...]:
        """Ranked passages for `topic`, most relevant first -- Pinecone's
        `search()` already returns hits best-match-first, so no local
        re-sort is needed the way `runbooks.py`'s bm25 negation is."""
        response = self._client.search(
            namespace=self._namespace,
            top_k=limit,
            inputs={"text": _TOPIC_SEMANTIC_QUERIES[topic]},
        )
        passages = []
        for hit in response.result.hits:
            content = str(hit.fields["content"])
            passages.append(
                RunbookPassage(
                    passage_id=hit.id,
                    content=content,
                    source_version=str(hit.fields.get("source_version", "")),
                    content_hash=digest_text(content),
                    score=hit.score,
                    retrieval_mode=RetrievalMode.PINECONE_SEMANTIC,
                )
            )
        return tuple(passages)


def run_runbook_search_pinecone(
    arguments: SearchRunbooksArguments, index: PineconeRunbookIndex
) -> RunbookCheckOutcome:
    """The Pinecone-backed mirror of `runbooks.run_runbook_search`: same
    outcome shape, `live_setup._build_tool_registry` dispatches to one or
    the other, never both.

    `index.search(...)` is the one call here that can fail at request
    time -- a live `PineconeException`, not the credential/construction
    failure `PineconeRunbookIndex.__init__` already rules out. Caught here
    and turned into `UNAVAILABLE`, the same precedent
    `runbooks.run_runbook_search` sets for a live `sqlite3.Error`; anything
    else (a real bug) still propagates uncaught, matching that module's own
    "not swallowed" guarantee.
    """
    started = time.monotonic()
    try:
        passages = index.search(arguments.topic, arguments.limit)
    except PineconeException:
        return RunbookCheckOutcome(
            outcome=ToolOutcome.UNAVAILABLE,
            retrieval_mode=RetrievalMode.PINECONE_SEMANTIC,
            reason_code=ReasonCode.TOOL_UNAVAILABLE,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    return RunbookCheckOutcome(
        outcome=ToolOutcome.EXECUTED,
        passages=passages,
        retrieval_mode=RetrievalMode.PINECONE_SEMANTIC,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
