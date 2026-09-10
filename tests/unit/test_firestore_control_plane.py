"""`FirestoreReplayControlPlane`'s own contract, against a fake in-memory
Firestore client -- no real GCP access, no network, no emulator. Mirrors
the behavioral contracts `test_api_runtime.py` already proves for
`SqliteReplayControlPlane`, adapted to this backend.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from causalops.api import (
    ControlPlaneConflictError,
    ControlPlaneNotFoundError,
    CreateInvestigationRequest,
    DecisionRequest,
    InvestigationStatus,
    ScenarioFamily,
)
from causalops.api_runtime import FinalizedWorkerOutcome, ReplayGraphJobRunner
from causalops.firestore_control_plane import (
    ControlPlaneIntegrityError,
    FirestoreReplayControlPlane,
)
from causalops.live_setup import HostedReplayRuntimeWiring

# Mirrors `test_api_runtime.py`'s own identically-named guard: report
# finalization needs os.O_DIRECTORY/os.O_NOFOLLOW, absent on win32.
requires_posix_no_follow_reads = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "hosted report finalization/delivery/retrieval needs "
        "os.O_DIRECTORY/os.O_NOFOLLOW, which do not exist on win32; "
        "unsupported on Windows today, not a test-only gap"
    ),
)


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

    def collection(self, collection_id: str) -> FakeCollectionReference:
        return FakeCollectionReference(self._client, (*self._path, collection_id))


class FakeCollectionReference:
    def __init__(
        self,
        client: FakeFirestoreClient,
        path: tuple[str, ...],
        *,
        filters: tuple[tuple[str, str, Any], ...] = (),
    ) -> None:
        self._client = client
        self._path = path
        self._filters = filters

    def document(self, document_id: str) -> FakeDocumentReference:
        return FakeDocumentReference(self._client, (*self._path, document_id))

    def where(
        self, field_path: str, op_string: str, value: object
    ) -> FakeCollectionReference:
        return FakeCollectionReference(
            self._client,
            self._path,
            filters=(*self._filters, (field_path, op_string, value)),
        )

    @staticmethod
    def _matches(data: dict[str, Any], field: str, op: str, value: object) -> bool:
        actual = data.get(field)
        if op == "==":
            return bool(actual == value)
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
        return iter(FakeDocumentSnapshot(row) for row in rows)


class FakeWriteBatch:
    def __init__(self, client: FakeFirestoreClient) -> None:
        self._client = client
        self._writes: list[tuple[Any, dict[str, Any]]] = []

    def set(self, reference: Any, document_data: dict[str, Any]) -> None:
        self._writes.append((reference, dict(document_data)))

    def commit(self) -> None:
        for reference, data in self._writes:
            reference.set(data)


class FakeFirestoreClient:
    """A real `firestore.Client()` is never constructed in these tests --
    `FirestoreReplayControlPlane` accepts an injected client precisely so
    ADC resolution never has to happen here."""

    def __init__(self) -> None:
        self.documents: dict[tuple[str, ...], dict[str, Any]] = {}

    def collection(self, collection_id: str) -> FakeCollectionReference:
        return FakeCollectionReference(self, (collection_id,))

    def batch(self) -> FakeWriteBatch:
        return FakeWriteBatch(self)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _plane(
    tmp_path: Path, *, clock: FakeClock | None = None, **kwargs: Any
) -> tuple[FirestoreReplayControlPlane, FakeFirestoreClient]:
    client = FakeFirestoreClient()
    plane = FirestoreReplayControlPlane(
        tmp_path / "artifacts",
        client=client,
        clock=clock or FakeClock(),
        **kwargs,
    )
    return plane, client


def _request() -> CreateInvestigationRequest:
    return CreateInvestigationRequest(
        scenario_family=ScenarioFamily.CONFIGURATION_CHANGE
    )


def _write_report(artifacts_root: Path, investigation_id: str, content: str) -> None:
    directory = artifacts_root / investigation_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.md").write_text(content, encoding="utf-8")


def test_create_then_status_and_events(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)

    created = plane.create("owner@example.com", _request(), "key-1")

    assert created.status is InvestigationStatus.QUEUED
    assert plane.status("owner@example.com", created.investigation_id) == created
    events = plane.events("owner@example.com", created.investigation_id)
    assert [event.name for event in events] == ["investigation_queued"]
    with pytest.raises(ControlPlaneNotFoundError):
        plane.status("other@example.com", created.investigation_id)


def test_repeated_idempotency_key_replays_the_same_investigation(
    tmp_path: Path,
) -> None:
    plane, _ = _plane(tmp_path)

    first = plane.create("owner@example.com", _request(), "key-1")
    second = plane.create("owner@example.com", _request(), "key-1")
    other_owner = plane.create("other@example.com", _request(), "key-1")
    different_key = plane.create("owner@example.com", _request(), "key-2")

    assert second == first
    assert other_owner.investigation_id != first.investigation_id
    assert different_key.investigation_id != first.investigation_id
    assert len(plane.events("owner@example.com", first.investigation_id)) == 1


def test_claim_next_returns_none_when_nothing_is_queued(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)

    assert plane.claim_next() is None


def test_claim_next_claims_exactly_one_job_and_marks_it_running(
    tmp_path: Path,
) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")

    claim = plane.claim_next()

    assert claim is not None
    assert claim.investigation_id == created.investigation_id
    assert claim.owner_email == "owner@example.com"
    assert claim.owner_decision is None
    assert plane.claim_next() is None  # only one RUNNING at a time


def test_claim_next_prefers_a_job_that_already_reserved_an_incident(
    tmp_path: Path,
) -> None:
    plane, _ = _plane(tmp_path)
    first = plane.create("owner@example.com", _request(), "key-1")
    plane.create("owner@example.com", _request(), "key-2")
    claim = plane.claim_next()
    assert claim is not None and claim.investigation_id == first.investigation_id
    plane.reserve_incident(first.investigation_id, claim.claim_token)
    plane.requeue_running(first.investigation_id, claim.claim_token)

    second_claim = plane.claim_next()

    assert second_claim is not None
    assert second_claim.investigation_id == first.investigation_id
    assert second_claim.incident_id == f"scenario{first.investigation_id}"


def test_claim_next_defers_every_job_while_a_reserved_incident_is_in_backoff(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    plane, _ = _plane(tmp_path, clock=clock)
    first = plane.create("owner@example.com", _request(), "key-1")
    second = plane.create("owner@example.com", _request(), "key-2")
    claim = plane.claim_next()
    assert claim is not None and claim.investigation_id == first.investigation_id
    plane.reserve_incident(first.investigation_id, claim.claim_token)
    plane.retry_running(first.investigation_id, claim.claim_token, RuntimeError("x"))

    # first is now QUEUED again with incident_id set and available_at in the
    # future (retry backoff) -- no job, including `second`, may be claimed.
    assert plane.claim_next() is None

    clock.now += 3600  # past any retry backoff
    resumed = plane.claim_next()
    assert resumed is not None
    assert resumed.investigation_id == first.investigation_id
    assert second.investigation_id != first.investigation_id


def test_mark_paused_then_decide_then_claim_next_carries_the_decision(
    tmp_path: Path,
) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    plane.mark_paused(created.investigation_id, "checkpoint-1", claim.claim_token)

    paused_status = plane.status("owner@example.com", created.investigation_id)
    assert paused_status.status is InvestigationStatus.PAUSED_APPROVAL

    accepted = plane.decide(
        "owner@example.com",
        created.investigation_id,
        DecisionRequest(decision="accept"),
    )
    assert accepted.status is InvestigationStatus.QUEUED
    # Repeating the identical decision is idempotent, not an error.
    assert (
        plane.decide(
            "owner@example.com",
            created.investigation_id,
            DecisionRequest(decision="accept"),
        )
        == accepted
    )
    with pytest.raises(ControlPlaneConflictError):
        plane.decide(
            "owner@example.com",
            created.investigation_id,
            DecisionRequest(
                decision="reject", rejection_note="a different decision now"
            ),
        )

    resumed_claim = plane.claim_next()
    assert resumed_claim is not None
    assert resumed_claim.checkpoint_id == "checkpoint-1"
    assert resumed_claim.owner_decision == DecisionRequest(decision="accept")

    event_names = [
        event.name
        for event in plane.events("owner@example.com", created.investigation_id)
    ]
    assert event_names == [
        "investigation_queued",
        "investigation_started",
        "owner_decision_requested",
        "owner_decision_recorded",
        "investigation_started",
    ]


def test_decide_refuses_when_not_waiting_for_a_decision(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")

    with pytest.raises(ControlPlaneConflictError):
        plane.decide(
            "owner@example.com",
            created.investigation_id,
            DecisionRequest(decision="accept"),
        )


def test_requeue_running_releases_a_crashed_claim(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None

    plane.requeue_running(created.investigation_id, claim.claim_token)

    assert (
        plane.status("owner@example.com", created.investigation_id).status
        is InvestigationStatus.QUEUED
    )
    assert plane.claim_next() is not None


def test_retry_running_exhausts_after_max_attempts(tmp_path: Path) -> None:
    clock = FakeClock()
    plane, _ = _plane(tmp_path, clock=clock, max_replay_attempts=2)
    created = plane.create("owner@example.com", _request(), "key-1")

    claim = plane.claim_next()
    assert claim is not None
    assert (
        plane.retry_running(created.investigation_id, claim.claim_token, ValueError())
        is True
    )

    clock.now += 3600  # past the retry backoff `retry_running` just scheduled
    claim2 = plane.claim_next()
    assert claim2 is not None
    assert (
        plane.retry_running(created.investigation_id, claim2.claim_token, ValueError())
        is False
    )

    final = plane.status("owner@example.com", created.investigation_id)
    assert final.status is InvestigationStatus.FAILED_SAFE


def test_finalize_then_report_round_trips_the_written_file(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    _write_report(tmp_path / "artifacts", created.investigation_id, "# Cited report")

    plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )

    finalized = plane.status("owner@example.com", created.investigation_id)
    assert finalized.status is InvestigationStatus.COMPLETED
    assert plane.report("owner@example.com", created.investigation_id) == (
        "# Cited report"
    )
    with pytest.raises(ControlPlaneConflictError):
        plane.finalize(
            created.investigation_id,
            f"{created.investigation_id}/report.md",
            claim.claim_token,
        )


def test_report_refuses_before_finalization(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")

    with pytest.raises(ControlPlaneConflictError):
        plane.report("owner@example.com", created.investigation_id)


def test_claim_delivery_then_mark_delivered(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    _write_report(tmp_path / "artifacts", created.investigation_id, "# Report")
    plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )

    delivery = plane.claim_delivery()
    assert delivery is not None
    assert delivery.destination_email == "owner@example.com"
    assert delivery.report_content == "# Report"
    assert plane.claim_delivery() is None  # already claimed (SENDING)

    plane.mark_delivered(created.investigation_id, delivery.claim_token)

    with pytest.raises(ControlPlaneConflictError):
        plane.mark_delivered(created.investigation_id, delivery.claim_token)


def test_release_delivery_returns_it_to_the_outbox(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    _write_report(tmp_path / "artifacts", created.investigation_id, "# Report")
    plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )
    delivery = plane.claim_delivery()
    assert delivery is not None

    plane.release_delivery(created.investigation_id, delivery.claim_token)

    again = plane.claim_delivery()
    assert again is not None
    assert again.investigation_id == created.investigation_id


def test_retry_delivery_exhausts_after_max_attempts(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path, max_delivery_attempts=1)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    _write_report(tmp_path / "artifacts", created.investigation_id, "# Report")
    plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )
    delivery = plane.claim_delivery()
    assert delivery is not None

    assert (
        plane.retry_delivery(
            created.investigation_id, delivery.claim_token, ValueError()
        )
        is False
    )
    assert plane.claim_delivery() is None


def test_renew_running_extends_the_lease(tmp_path: Path) -> None:
    clock = FakeClock()
    plane, _ = _plane(tmp_path, clock=clock)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None

    clock.now += plane.claim_lease_seconds - 1
    plane.renew_running(created.investigation_id, claim.claim_token)  # must not raise

    clock.now += plane.claim_lease_seconds - 1
    plane.renew_running(created.investigation_id, claim.claim_token)  # still renewed


def test_renew_running_refuses_after_the_lease_expires(tmp_path: Path) -> None:
    clock = FakeClock()
    plane, _ = _plane(tmp_path, clock=clock)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None

    clock.now += plane.claim_lease_seconds + 1

    with pytest.raises(ControlPlaneConflictError):
        plane.renew_running(created.investigation_id, claim.claim_token)


def test_expired_lease_is_reclaimed_on_the_next_claim_next(tmp_path: Path) -> None:
    clock = FakeClock()
    plane, _ = _plane(tmp_path, clock=clock)
    created = plane.create("owner@example.com", _request(), "key-1")
    first_claim = plane.claim_next()
    assert first_claim is not None

    clock.now += plane.claim_lease_seconds + 1

    # The reclaim itself happens inside this call (RUNNING -> QUEUED with a
    # retry backoff `available_at`) -- immediately re-claimable only once
    # that backoff has also passed, not within the same tick.
    assert plane.claim_next() is None
    clock.now += 3600

    second_claim = plane.claim_next()
    assert second_claim is not None
    assert second_claim.investigation_id == created.investigation_id
    assert second_claim.claim_token != first_claim.claim_token
    event_names = [
        event.name
        for event in plane.events("owner@example.com", created.investigation_id)
    ]
    assert "investigation_lease_expired" in event_names


def test_reserve_incident_is_idempotent_across_retries(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None

    first = plane.reserve_incident(created.investigation_id, claim.claim_token)
    second = plane.reserve_incident(created.investigation_id, claim.claim_token)

    assert first == second == f"scenario{created.investigation_id}"
    assert plane.incident_id_for(created.investigation_id) == first
    assert plane.scenario_status(first) is InvestigationStatus.RUNNING


def test_scenario_status_reflects_failure(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path, max_replay_attempts=1)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    incident_id = plane.reserve_incident(created.investigation_id, claim.claim_token)

    plane.retry_running(created.investigation_id, claim.claim_token, ValueError())

    assert plane.scenario_status(incident_id) is InvestigationStatus.FAILED_SAFE


def test_construction_rejects_non_positive_tuning_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="claim_lease_seconds"):
        _plane(tmp_path, claim_lease_seconds=0)
    with pytest.raises(ValueError, match="retry_base_seconds"):
        _plane(tmp_path, retry_base_seconds=-1)
    with pytest.raises(ValueError, match="max_replay_attempts"):
        _plane(tmp_path, max_replay_attempts=0)
    with pytest.raises(ValueError, match="max_delivery_attempts"):
        _plane(tmp_path, max_delivery_attempts=0)


def test_finalize_with_a_forged_report_path_is_refused(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    _write_report(tmp_path / "artifacts", created.investigation_id, "# Report")

    with pytest.raises(ValueError, match="own report.md"):
        plane.finalize(
            created.investigation_id, "../escape/report.md", claim.claim_token
        )


def test_corrupt_report_content_is_detected_on_read(tmp_path: Path) -> None:
    """Simulates storage corruption directly (no code path in this class
    can otherwise produce it): a mismatched `report_sha256` must be
    refused, not silently served."""
    plane, client = _plane(tmp_path)
    created = plane.create("owner@example.com", _request(), "key-1")
    claim = plane.claim_next()
    assert claim is not None
    _write_report(tmp_path / "artifacts", created.investigation_id, "# Report")
    plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )
    job_path = ("replay_jobs", created.investigation_id)
    client.documents[job_path]["report_sha256"] = "0" * 64

    with pytest.raises(ControlPlaneIntegrityError, match="corrupt"):
        plane.report("owner@example.com", created.investigation_id)


@requires_posix_no_follow_reads
def test_a_real_worker_run_finalizes_through_this_control_plane(
    tmp_path: Path,
) -> None:
    """The decisive proof, not just the hand-built-fixture tests above:
    drives a REAL `ReplayGraphJobRunner` (the actual hosted worker's own
    code path, `HostedReplayRuntimeWiring` and all) against THIS control
    plane -- the same "prove it once against a real consumer" approach
    `test_firestore_checkpointer.py`'s own
    `test_a_real_graph_investigation_runs_end_to_end_on_this_checkpointer`
    took, and that `test_api_runtime.py`'s own
    `test_runner_adopts_an_artifact_published_before_control_plane_finalization`
    already takes for `SqliteReplayControlPlane`. A control plane that only
    ever satisfies hand-built `claim_next`/`finalize` call sequences above
    could still be wrong about the exact claim shape a real runner
    produces; this closes that gap."""
    # `ReplayGraphJobRunner._has_finalized_artifact` hardcodes
    # `root / "results" / "investigations"` (matching
    # `finalize_investigation`'s own layout) -- the control plane's own
    # artifacts root must be exactly that path, not `_plane`'s usual
    # `tmp_path / "artifacts"` convenience default, for the runner and the
    # control plane to agree on where a recovered report lives.
    artifacts_root = tmp_path / "results" / "investigations"
    plane = FirestoreReplayControlPlane(artifacts_root, client=FakeFirestoreClient())
    created = plane.create("owner@example.com", _request(), "key-1")
    _write_report(artifacts_root, created.investigation_id, "# Recovered report")
    claim = plane.claim_next()
    assert claim is not None

    outcome = ReplayGraphJobRunner(tmp_path, plane, HostedReplayRuntimeWiring()).run(
        claim
    )

    assert outcome == FinalizedWorkerOutcome(
        report_artifact=f"{created.investigation_id}/report.md"
    )
    plane.finalize(created.investigation_id, outcome.report_artifact, claim.claim_token)
    assert plane.report("owner@example.com", created.investigation_id) == (
        "# Recovered report"
    )
