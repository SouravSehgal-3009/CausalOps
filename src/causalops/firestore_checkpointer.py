"""Firestore-backed LangGraph checkpoint saver.

Spec §3.1: "A Firestore-backed LangGraph checkpoint saver stores
strict-serialized graph state with optimistic revisions. It refuses
checkpoints above 900 KiB." Mirrors `langgraph.checkpoint.sqlite.SqliteSaver`'s
own sync-only method set exactly (`get_tuple`, `list`, `put`, `put_writes`,
`delete_thread`, `get_next_version`) -- this project's own graph usage is
synchronous throughout (`cli.py`'s `_sqlite_checkpointer` never reaches for
`AsyncSqliteSaver`), so async overrides are unnecessary here either.
`get_next_version` reuses `SqliteSaver`'s own monotonic
zero-padded-int-plus-random-hex string scheme verbatim -- a known-correct,
already-tested version format, not a reason to invent a new one.

Document layout was chosen specifically to need ZERO Firestore composite
indexes (only automatic single-field indexes, which need no
`google_firestore_index` terraform resource at all):

    checkpoint_threads/{thread_id}::{checkpoint_ns}/entries/{checkpoint_id}
        thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
        type, checkpoint (bytes), metadata (bytes)
      entries/{checkpoint_id}/writes/{task_id}::{idx}
        task_id, idx, channel, type, value (bytes)

Fetching one exact checkpoint (or its writes) needs no query at all -- the
document path is already fully determined by
`(thread_id, checkpoint_ns, checkpoint_id)`. "Latest checkpoint for a
thread" and `list()` both query only the `entries` subcollection of one
`checkpoint_threads` document -- a single-field order_by/range on
`checkpoint_id` (a monotonically increasing, lexicographically sortable ID
LangGraph itself guarantees), which Firestore indexes automatically.

`put_writes`'s conditional insert-or-ignore/insert-or-replace (mirroring
`SqliteSaver`'s own `INSERT OR IGNORE`/`INSERT OR REPLACE` split) is
implemented as a plain check-then-set, not a Firestore transaction: today's
deployment runs exactly one `ReplayWorker` processing one investigation at
a time (`BackgroundControlPlaneWorkers`), so there is no concurrent writer
to race against yet. This needs transactional hardening before -- not
after -- a multi-worker Pub/Sub fan-out (a later phase) makes concurrent
writers to the same checkpoint a real possibility.
"""

from __future__ import annotations

import builtins
import json
import random
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, cast

from google.cloud import firestore
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

MAX_CHECKPOINT_BYTES = 900 * 1024

_THREADS_COLLECTION = "checkpoint_threads"
_ENTRIES_SUBCOLLECTION = "entries"
_WRITES_SUBCOLLECTION = "writes"


class CheckpointTooLargeError(ValueError):
    """A checkpoint's serialized size exceeds the spec's 900 KiB cap."""


class _DocumentSnapshot(Protocol):
    exists: bool

    def to_dict(self) -> dict[str, Any] | None: ...


class _DocumentReference(Protocol):
    def get(self) -> _DocumentSnapshot: ...
    def set(self, document_data: dict[str, Any]) -> object: ...
    def delete(self) -> object: ...
    def collection(self, collection_id: str) -> _CollectionReference: ...


class _Query(Protocol):
    def where(self, field_path: str, op_string: str, value: object) -> _Query: ...
    def order_by(self, field_path: str, direction: str = ...) -> _Query: ...
    def limit(self, count: int) -> _Query: ...
    def stream(self) -> Iterator[_DocumentSnapshot]: ...


class _CollectionReference(_Query, Protocol):
    def document(self, document_id: str) -> _DocumentReference: ...


class _Client(Protocol):
    """The narrow seam this saver actually calls -- a real
    `firestore.Client` satisfies this structurally; tests inject a fake
    client shaped the same way, the same seam-testing approach this
    project already uses for `GcsArtifactStore`/`IdentityVerifier`."""

    def collection(self, collection_id: str) -> _CollectionReference: ...


def _thread_document_id(thread_id: str, checkpoint_ns: str) -> str:
    return f"{thread_id}::{checkpoint_ns}"


def _write_document_id(task_id: str, idx: int) -> str:
    return f"{task_id}::{idx}"


def _thread_doc(
    client: _Client, thread_id: str, checkpoint_ns: str
) -> _DocumentReference:
    return client.collection(_THREADS_COLLECTION).document(
        _thread_document_id(thread_id, checkpoint_ns)
    )


def _entries(
    client: _Client, thread_id: str, checkpoint_ns: str
) -> _CollectionReference:
    return _thread_doc(client, thread_id, checkpoint_ns).collection(
        _ENTRIES_SUBCOLLECTION
    )


class FirestoreCheckpointSaver(BaseCheckpointSaver[str]):
    """A synchronous `BaseCheckpointSaver` backed by Firestore."""

    def __init__(
        self, client: _Client | None = None, *, serde: SerializerProtocol | None = None
    ) -> None:
        super().__init__(serde=serde)
        self._client: _Client = (
            client if client is not None else cast(_Client, firestore.Client())
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        entries = _entries(self._client, thread_id, checkpoint_ns)

        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is not None:
            snapshot = entries.document(checkpoint_id).get()
            if not snapshot.exists:
                return None
            data = snapshot.to_dict()
            assert data is not None
        else:
            found = list(
                entries.order_by("checkpoint_id", direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream()
            )
            if not found:
                return None
            data = found[0].to_dict()
            assert data is not None
            checkpoint_id = str(data["checkpoint_id"])

        resolved_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_checkpoint_id = data.get("parent_checkpoint_id")
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
            if parent_checkpoint_id
            else None
        )
        pending_writes = self._read_writes(entries.document(checkpoint_id), sort=True)
        return CheckpointTuple(
            resolved_config,
            self.serde.loads_typed((data["type"], data["checkpoint"])),
            cast(CheckpointMetadata, json.loads(bytes(data["metadata"]))),
            parent_config,
            pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            raise ValueError(
                "FirestoreCheckpointSaver.list requires a config naming thread_id"
            )
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        query: _Query = _entries(self._client, thread_id, checkpoint_ns).order_by(
            "checkpoint_id", direction=firestore.Query.DESCENDING
        )
        if before is not None:
            before_id = get_checkpoint_id(before)
            if before_id is not None:
                query = query.where("checkpoint_id", "<", before_id)
        if limit is not None:
            query = query.limit(limit)

        for snapshot in query.stream():
            data = snapshot.to_dict()
            assert data is not None
            if filter and not all(
                json.loads(bytes(data["metadata"])).get(key) == value
                for key, value in filter.items()
            ):
                continue
            checkpoint_id = str(data["checkpoint_id"])
            parent_checkpoint_id = data.get("parent_checkpoint_id")
            entry_ref = _entries(self._client, thread_id, checkpoint_ns).document(
                checkpoint_id
            )
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                self.serde.loads_typed((data["type"], data["checkpoint"])),
                cast(CheckpointMetadata, json.loads(bytes(data["metadata"]))),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                self._read_writes(entry_ref, sort=True),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        if len(serialized_checkpoint) > MAX_CHECKPOINT_BYTES:
            raise CheckpointTooLargeError(
                f"checkpoint for thread {thread_id!r} is "
                f"{len(serialized_checkpoint)} bytes, over the "
                f"{MAX_CHECKPOINT_BYTES}-byte (900 KiB) cap"
            )
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")
        # A real document at the thread level, not just an implicit parent
        # of the `entries` subcollection -- `delete_thread` finds threads
        # by querying THIS collection, which only works if something was
        # actually written here.
        _thread_doc(self._client, thread_id, checkpoint_ns).set(
            {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        )
        entries = _entries(self._client, thread_id, checkpoint_ns)
        entries.document(checkpoint["id"]).set(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
                "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
                "type": type_,
                "checkpoint": serialized_checkpoint,
                "metadata": serialized_metadata,
            }
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"]["checkpoint_ns"])
        checkpoint_id = str(config["configurable"]["checkpoint_id"])
        entry_ref = _entries(self._client, thread_id, checkpoint_ns).document(
            checkpoint_id
        )
        writes_collection = entry_ref.collection(_WRITES_SUBCOLLECTION)
        replace_all = all(channel in WRITES_IDX_MAP for channel, _ in writes)
        for idx, (channel, value) in enumerate(writes):
            resolved_idx = WRITES_IDX_MAP.get(channel, idx)
            write_ref = writes_collection.document(
                _write_document_id(task_id, resolved_idx)
            )
            if not replace_all and write_ref.get().exists:
                continue
            value_type, serialized_value = self.serde.dumps_typed(value)
            write_ref.set(
                {
                    "task_id": task_id,
                    "idx": resolved_idx,
                    "channel": channel,
                    "type": value_type,
                    "value": serialized_value,
                }
            )

    def delete_thread(self, thread_id: str) -> None:
        threads = self._client.collection(_THREADS_COLLECTION)
        for snapshot in threads.where("thread_id", "==", str(thread_id)).stream():
            data = snapshot.to_dict()
            assert data is not None
            checkpoint_ns = str(data.get("checkpoint_ns", ""))
            entries = _entries(self._client, str(thread_id), checkpoint_ns)
            for entry_snapshot in entries.stream():
                entry_data = entry_snapshot.to_dict()
                assert entry_data is not None
                entry_ref = entries.document(str(entry_data["checkpoint_id"]))
                for write_snapshot in entry_ref.collection(
                    _WRITES_SUBCOLLECTION
                ).stream():
                    write_data = write_snapshot.to_dict()
                    assert write_data is not None
                    entry_ref.collection(_WRITES_SUBCOLLECTION).document(
                        _write_document_id(
                            str(write_data["task_id"]), int(write_data["idx"])
                        )
                    ).delete()
                entry_ref.delete()
            _thread_doc(self._client, str(thread_id), checkpoint_ns).delete()

    def _read_writes(
        self, entry_ref: _DocumentReference, *, sort: bool
    ) -> builtins.list[PendingWrite]:
        # `builtins.list`, not `list`: this class defines its own `list`
        # method (the required `BaseCheckpointSaver` interface method),
        # which shadows the builtin type within every method signature in
        # this class body -- a real Python name-resolution gotcha, not a
        # typo.
        rows = []
        for snapshot in entry_ref.collection(_WRITES_SUBCOLLECTION).stream():
            data = snapshot.to_dict()
            assert data is not None
            rows.append(data)
        if sort:
            rows.sort(key=lambda row: (str(row["task_id"]), int(row["idx"])))
        return [
            (
                str(row["task_id"]),
                str(row["channel"]),
                self.serde.loads_typed((row["type"], row["value"])),
            )
            for row in rows
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Reuses `SqliteSaver.get_next_version`'s own monotonic scheme
        verbatim -- a known-correct, already-tested version format."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
