"""`pinecone_runbooks.py`'s backend, exercised through the injected
`client=` seam (`PineconeSearchClient`) -- no real Pinecone credential or
network call anywhere in this file. `tests/unit/test_live_setup.py`'s own
family-fixture completeness test has no equivalent here because there is
only one Pinecone index, not one per scenario family; the real-consumer
proof against the live index is a separate, manually-run check, not part
of the hermetic suite.
"""

from collections.abc import Mapping

import pytest
from pinecone.exceptions import PineconeException

from causalops.domain import RetrievalMode, ToolOutcome
from causalops.pinecone_runbooks import (
    PINECONE_API_KEY_VARIABLE,
    PineconeRunbookIndex,
    PineconeRunbookIndexError,
    run_runbook_search_pinecone,
)
from causalops.tools import RunbookTopic, SearchRunbooksArguments


class _FakeHit:
    def __init__(self, id: str, score: float, fields: Mapping[str, object]) -> None:
        self.id = id
        self.score = score
        self.fields = fields


class _FakeResult:
    def __init__(self, hits: list[_FakeHit]) -> None:
        self.hits = hits


class _FakeResponse:
    def __init__(self, hits: list[_FakeHit]) -> None:
        self.result = _FakeResult(hits)


class FakeSearchClient:
    """Records every `search()` call it receives and returns a scripted
    response -- the same fake-backend pattern `RecordingLogsBackend` and
    friends already use elsewhere for a narrow Protocol seam."""

    def __init__(self, hits: list[_FakeHit]) -> None:
        self._hits = hits
        self.calls: list[dict[str, object]] = []

    def search(
        self, *, namespace: str, top_k: int, inputs: Mapping[str, str]
    ) -> _FakeResponse:
        self.calls.append(
            {"namespace": namespace, "top_k": top_k, "inputs": dict(inputs)}
        )
        return _FakeResponse(self._hits)


class RefusingSearchClient:
    """Raises the real `PineconeException` base class a live query error
    would -- proves `run_runbook_search_pinecone` catches the actual
    exception type, not an assumption."""

    def search(
        self, *, namespace: str, top_k: int, inputs: Mapping[str, str]
    ) -> _FakeResponse:
        raise PineconeException("simulated live query failure")


def test_construction_without_a_client_or_api_key_refuses_before_any_io() -> None:
    with pytest.raises(PineconeRunbookIndexError, match=PINECONE_API_KEY_VARIABLE):
        PineconeRunbookIndex(environment={})


def test_construction_with_a_blank_api_key_refuses_before_any_io() -> None:
    with pytest.raises(PineconeRunbookIndexError, match=PINECONE_API_KEY_VARIABLE):
        PineconeRunbookIndex(environment={PINECONE_API_KEY_VARIABLE: "   "})


def test_search_maps_hits_into_runbook_passages_ranked_as_returned() -> None:
    client = FakeSearchClient(
        [
            _FakeHit(
                "runbook-downstream-timeouts-01",
                0.91,
                {"content": "first passage text", "source_version": "1"},
            ),
            _FakeHit(
                "runbook-downstream-timeouts-02",
                0.42,
                {"content": "second passage text", "source_version": "1"},
            ),
        ]
    )
    index = PineconeRunbookIndex(client)

    results = index.search(RunbookTopic.DOWNSTREAM_TIMEOUTS, limit=2)

    assert [passage.passage_id for passage in results] == [
        "runbook-downstream-timeouts-01",
        "runbook-downstream-timeouts-02",
    ]
    assert results[0].content == "first passage text"
    assert results[0].score == 0.91
    assert results[0].retrieval_mode is RetrievalMode.PINECONE_SEMANTIC
    assert results[0].content_hash


def test_search_sends_the_topics_fixed_semantic_query_not_a_free_text_one() -> None:
    client = FakeSearchClient([])
    index = PineconeRunbookIndex(client)

    index.search(RunbookTopic.RESOURCE_POOL_PRESSURE, limit=5)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["namespace"] == "runbooks"
    assert call["top_k"] == 5
    assert "pool" in str(call["inputs"]["text"]).lower()


def test_run_runbook_search_pinecone_executes_and_stamps_the_retrieval_mode() -> None:
    client = FakeSearchClient(
        [
            _FakeHit(
                "runbook-gateway-latency-01",
                0.7,
                {"content": "x", "source_version": "1"},
            )
        ]
    )
    index = PineconeRunbookIndex(client)
    arguments = SearchRunbooksArguments(topic=RunbookTopic.GATEWAY_LATENCY, limit=3)

    outcome = run_runbook_search_pinecone(arguments, index)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.retrieval_mode is RetrievalMode.PINECONE_SEMANTIC
    assert outcome.passages


def test_run_runbook_search_pinecone_turns_a_live_query_error_into_unavailable() -> (
    None
):
    index = PineconeRunbookIndex(RefusingSearchClient())
    arguments = SearchRunbooksArguments(topic=RunbookTopic.GATEWAY_ERRORS, limit=3)

    outcome = run_runbook_search_pinecone(arguments, index)

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.passages == ()
    assert outcome.retrieval_mode is RetrievalMode.PINECONE_SEMANTIC


def test_run_runbook_search_pinecone_does_not_swallow_a_non_pinecone_error() -> None:
    index = PineconeRunbookIndex(FakeSearchClient([]))

    def broken_search(topic: RunbookTopic, limit: int) -> tuple[()]:
        raise ValueError("not a PineconeException")

    index.search = broken_search  # type: ignore[method-assign]
    arguments = SearchRunbooksArguments(topic=RunbookTopic.GATEWAY_ERRORS, limit=3)

    with pytest.raises(ValueError, match="not a PineconeException"):
        run_runbook_search_pinecone(arguments, index)


def test_every_runbook_topic_has_a_fixed_semantic_query() -> None:
    from causalops.pinecone_runbooks import _TOPIC_SEMANTIC_QUERIES

    assert set(_TOPIC_SEMANTIC_QUERIES) == set(RunbookTopic)
