"""`FirestoreCheckpointSaver`'s own contract, against a fake in-memory
Firestore client -- no real GCP access, no network, no emulator.

The fake models exactly the operations `firestore_checkpointer.py` calls
(`collection`/`document`/`get`/`set`/`delete`/`where`/`order_by`/`limit`/
`stream`) against a flat path->data dict, faithfully enough to prove the
saver's own document-layout and query logic -- not a general-purpose
Firestore emulator.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fake_incident import (
    FIXTURE_DIR,
    SYMPTOM_EVIDENCE_ID,
    WINDOW_END,
    WINDOW_START,
    RecordingLogsBackend,
    alert_packet,
    incident_scope,
    logs_only_registry,
    packet_evidence,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from causalops.domain import (
    Budgets,
    EscalatedInvestigation,
    InvestigationResult,
    utc_now,
)
from causalops.firestore_checkpointer import (
    MAX_CHECKPOINT_BYTES,
    CheckpointTooLargeError,
    FirestoreCheckpointSaver,
)
from causalops.graph import resume_graph_investigation, run_graph_investigation
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunRecorder


class FakeDocumentSnapshot:
    def __init__(self, data: dict[str, Any] | None) -> None:
        self._data = data
        self.exists = data is not None

    def to_dict(self) -> dict[str, Any] | None:
        return dict(self._data) if self._data is not None else None


class FakeDocumentReference:
    def __init__(self, client: FakeFirestoreClient, path: tuple[str, ...]) -> None:
        self._client = client
        self._path = path

    def get(self) -> FakeDocumentSnapshot:
        return FakeDocumentSnapshot(self._client.documents.get(self._path))

    def set(self, document_data: dict[str, Any]) -> None:
        self._client.documents[self._path] = dict(document_data)

    def delete(self) -> None:
        self._client.documents.pop(self._path, None)

    def collection(self, collection_id: str) -> FakeCollectionReference:
        return FakeCollectionReference(self._client, (*self._path, collection_id))


class FakeCollectionReference:
    def __init__(
        self,
        client: FakeFirestoreClient,
        path: tuple[str, ...],
        *,
        filters: tuple[tuple[str, str, Any], ...] = (),
        order: tuple[str, str] | None = None,
        limit_n: int | None = None,
    ) -> None:
        self._client = client
        self._path = path
        self._filters = filters
        self._order = order
        self._limit_n = limit_n

    def document(self, document_id: str) -> FakeDocumentReference:
        return FakeDocumentReference(self._client, (*self._path, document_id))

    def where(
        self, field_path: str, op_string: str, value: object
    ) -> FakeCollectionReference:
        return FakeCollectionReference(
            self._client,
            self._path,
            filters=(*self._filters, (field_path, op_string, value)),
            order=self._order,
            limit_n=self._limit_n,
        )

    def order_by(
        self, field_path: str, direction: str = "ASCENDING"
    ) -> FakeCollectionReference:
        return FakeCollectionReference(
            self._client,
            self._path,
            filters=self._filters,
            order=(field_path, direction),
            limit_n=self._limit_n,
        )

    def limit(self, count: int) -> FakeCollectionReference:
        return FakeCollectionReference(
            self._client,
            self._path,
            filters=self._filters,
            order=self._order,
            limit_n=count,
        )

    @staticmethod
    def _matches(data: dict[str, Any], field: str, op: str, value: object) -> bool:
        actual = data.get(field)
        if op == "==":
            return bool(actual == value)
        if op == "<":
            return actual is not None and actual < value
        if op == ">":
            return actual is not None and actual > value
        raise NotImplementedError(f"fake Firestore client has no support for {op!r}")

    def stream(self) -> Iterator[FakeDocumentSnapshot]:
        depth = len(self._path) + 1
        rows = [
            data
            for path, data in self._client.documents.items()
            if len(path) == depth and path[: len(self._path)] == self._path
        ]
        for field, op, value in self._filters:
            rows = [row for row in rows if self._matches(row, field, op, value)]
        if self._order is not None:
            field, direction = self._order
            rows.sort(key=lambda row: row[field], reverse=(direction == "DESCENDING"))
        if self._limit_n is not None:
            rows = rows[: self._limit_n]
        return iter(FakeDocumentSnapshot(row) for row in rows)


class FakeFirestoreClient:
    """A real `firestore.Client()` is never constructed in these tests --
    `FirestoreCheckpointSaver.__init__` accepts an injected client
    precisely so ADC resolution never has to happen here."""

    def __init__(self) -> None:
        self.documents: dict[tuple[str, ...], dict[str, Any]] = {}

    def collection(self, collection_id: str) -> FakeCollectionReference:
        return FakeCollectionReference(self, (collection_id,))


def _config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _checkpoint(checkpoint_id: str, value: int) -> Checkpoint:
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2026-09-10T00:00:00+00:00",
        "channel_values": {"count": value},
        "channel_versions": {},
        "versions_seen": {},
        "updated_channels": None,
    }


def _metadata(step: int) -> CheckpointMetadata:
    return {"source": "loop", "step": step, "parents": {}}


def test_put_then_get_tuple_round_trips_the_checkpoint() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = _config("thread-1")

    saved_config = saver.put(config, _checkpoint("cp-1", 1), _metadata(0), {})
    tuple_out = saver.get_tuple(saved_config)

    assert tuple_out is not None
    assert tuple_out.checkpoint["id"] == "cp-1"
    assert tuple_out.checkpoint["channel_values"]["count"] == 1
    assert tuple_out.metadata["step"] == 0
    assert tuple_out.parent_config is None


def test_get_tuple_without_a_checkpoint_id_returns_the_latest() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = _config("thread-1")

    first = saver.put(config, _checkpoint("cp-1", 1), _metadata(0), {})
    second = saver.put(first, _checkpoint("cp-2", 2), _metadata(1), {})

    latest = saver.get_tuple(_config("thread-1"))

    assert latest is not None
    assert latest.checkpoint["id"] == "cp-2"
    assert latest.config == second


def test_put_records_the_parent_checkpoint_id() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = _config("thread-1")
    first = saver.put(config, _checkpoint("cp-1", 1), _metadata(0), {})

    saver.put(first, _checkpoint("cp-2", 2), _metadata(1), {})

    second_tuple = saver.get_tuple(_config("thread-1", "cp-2"))
    assert second_tuple is not None
    assert second_tuple.parent_config is not None
    assert second_tuple.parent_config["configurable"]["checkpoint_id"] == "cp-1"


def test_get_tuple_returns_none_for_an_unknown_thread() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())

    assert saver.get_tuple(_config("never-created")) is None


def test_put_over_900kib_is_refused_before_any_write() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = _config("thread-1")
    oversized = _checkpoint("cp-1", 1)
    oversized["channel_values"] = {"blob": "x" * (MAX_CHECKPOINT_BYTES + 1)}

    with pytest.raises(CheckpointTooLargeError, match="900"):
        saver.put(config, oversized, _metadata(0), {})

    assert saver.get_tuple(config) is None


def test_put_writes_then_get_tuple_returns_pending_writes_in_task_and_idx_order() -> (
    None
):
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = saver.put(_config("thread-1"), _checkpoint("cp-1", 1), _metadata(0), {})

    saver.put_writes(
        config, [("channel_b", "second"), ("channel_a", "first")], "task-1"
    )

    result = saver.get_tuple(config)
    assert result is not None
    assert result.pending_writes == [
        ("task-1", "channel_b", "second"),
        ("task-1", "channel_a", "first"),
    ]


def test_put_writes_ignores_a_repeat_write_to_a_regular_channel() -> None:
    """Mirrors `SqliteSaver`'s own `INSERT OR IGNORE` for ordinary
    channels: the first write to a given (task_id, idx) wins, a repeat is
    silently dropped, not overwritten."""
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = saver.put(_config("thread-1"), _checkpoint("cp-1", 1), _metadata(0), {})

    saver.put_writes(config, [("channel_a", "first")], "task-1")
    saver.put_writes(config, [("channel_a", "second")], "task-1")

    result = saver.get_tuple(config)
    assert result is not None
    assert result.pending_writes == [("task-1", "channel_a", "first")]


def test_put_writes_replaces_a_repeat_write_to_a_special_channel() -> None:
    """Mirrors `SqliteSaver`'s own `INSERT OR REPLACE` for the special
    WRITES_IDX_MAP channels (error/scheduled/interrupt/resume) -- these
    always overwrite, unlike ordinary channels. Reads the channel name out
    of `WRITES_IDX_MAP` itself rather than importing `ERROR` from
    `langgraph._internal._constants`, a private module."""
    from langgraph.checkpoint.base import WRITES_IDX_MAP

    special_channel = next(iter(WRITES_IDX_MAP))
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = saver.put(_config("thread-1"), _checkpoint("cp-1", 1), _metadata(0), {})

    saver.put_writes(config, [(special_channel, "first error")], "task-1")
    saver.put_writes(config, [(special_channel, "second error")], "task-1")

    result = saver.get_tuple(config)
    assert result is not None
    assert result.pending_writes == [("task-1", special_channel, "second error")]


def test_list_returns_checkpoints_newest_first() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = _config("thread-1")
    first = saver.put(config, _checkpoint("cp-1", 1), _metadata(0), {})
    saver.put(first, _checkpoint("cp-2", 2), _metadata(1), {})

    ids = [t.checkpoint["id"] for t in saver.list(_config("thread-1"))]

    assert ids == ["cp-2", "cp-1"]


def test_list_respects_before_and_limit() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = _config("thread-1")
    first = saver.put(config, _checkpoint("cp-1", 1), _metadata(0), {})
    second = saver.put(first, _checkpoint("cp-2", 2), _metadata(1), {})
    saver.put(second, _checkpoint("cp-3", 3), _metadata(2), {})

    ids = [
        t.checkpoint["id"]
        for t in saver.list(_config("thread-1"), before=_config("thread-1", "cp-3"))
    ]
    assert ids == ["cp-2", "cp-1"]

    limited = [t.checkpoint["id"] for t in saver.list(_config("thread-1"), limit=1)]
    assert limited == ["cp-3"]


def test_two_threads_never_see_each_others_checkpoints() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    saver.put(_config("thread-a"), _checkpoint("cp-1", 1), _metadata(0), {})
    saver.put(_config("thread-b"), _checkpoint("cp-1", 2), _metadata(0), {})

    a_tuple = saver.get_tuple(_config("thread-a"))
    b_tuple = saver.get_tuple(_config("thread-b"))

    assert a_tuple is not None and a_tuple.checkpoint["channel_values"]["count"] == 1
    assert b_tuple is not None and b_tuple.checkpoint["channel_values"]["count"] == 2


def test_delete_thread_removes_checkpoints_writes_and_the_thread_itself() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    config = saver.put(_config("thread-1"), _checkpoint("cp-1", 1), _metadata(0), {})
    saver.put_writes(config, [("channel_a", "value")], "task-1")

    saver.delete_thread("thread-1")

    assert saver.get_tuple(_config("thread-1")) is None
    assert list(saver.list(_config("thread-1"))) == []


def test_delete_thread_leaves_other_threads_untouched() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    saver.put(_config("thread-a"), _checkpoint("cp-1", 1), _metadata(0), {})
    saver.put(_config("thread-b"), _checkpoint("cp-1", 2), _metadata(0), {})

    saver.delete_thread("thread-a")

    assert saver.get_tuple(_config("thread-a")) is None
    remaining = saver.get_tuple(_config("thread-b"))
    assert remaining is not None
    assert remaining.checkpoint["channel_values"]["count"] == 2


def test_get_next_version_is_monotonically_increasing_and_string_typed() -> None:
    saver = FirestoreCheckpointSaver(client=FakeFirestoreClient())

    first = saver.get_next_version(None, None)
    second = saver.get_next_version(first, None)

    assert isinstance(first, str)
    assert isinstance(second, str)
    assert second > first


def test_a_real_graph_investigation_runs_end_to_end_on_this_checkpointer() -> None:
    """The decisive proof, not just the hand-built-fixture tests above:
    drives a REAL `run_graph_investigation` through THIS checkpointer,
    exercising LangGraph's own internal checkpoint-writing protocol
    (multiple channels, real channel versions, real pending-writes shapes)
    directly -- the same "prove it once against a real backend" approach
    `test_graph.py`'s own
    `test_an_unrecognised_resume_decision_re_pauses_instead_of_bricking_the_run`
    docstring describes having done for a real `SqliteSaver`. A checkpointer
    that only ever satisfies hand-built `Checkpoint`/`CheckpointMetadata`
    fixtures could still be wrong about a real shape LangGraph actually
    produces; this closes that gap."""
    model = ReplayToolCallingModel(
        ReplayReasoningModel(
            FIXTURE_DIR / "graph_single_check.json",
            substitutions={
                "incident_id": incident_scope().incident_id,
                "window_start": WINDOW_START.isoformat(),
                "window_end": WINDOW_END.isoformat(),
                "symptom_evidence_id": SYMPTOM_EVIDENCE_ID,
            },
        )
    )
    checkpointer = FirestoreCheckpointSaver(client=FakeFirestoreClient())
    registry = logs_only_registry(RecordingLogsBackend())

    result = run_graph_investigation(
        incident_scope(),
        alert_packet(),
        packet_evidence(),
        model,
        registry,
        RunRecorder(utc_now),
        Budgets(),
        investigation_id="firestore-checkpointer-integration",
        checkpointer=checkpointer,
    )

    if isinstance(result, EscalatedInvestigation):
        result = resume_graph_investigation(
            result.thread_id,
            checkpointer,
            incident_scope(),
            alert_packet(),
            model,
            registry,
            RunRecorder(utc_now),
            "accept",
            None,
            Budgets(),
        )

    assert isinstance(result, InvestigationResult)
    assert result.report.investigation_id == "firestore-checkpointer-integration"
