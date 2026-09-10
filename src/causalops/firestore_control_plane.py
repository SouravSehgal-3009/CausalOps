"""Firestore-backed replay control plane -- investigations, events, the
owner-decision ledger, and the delivery outbox (spec §3.1).

Mirrors `api_runtime.SqliteReplayControlPlane`'s full behavioral contract
method for method, verified directly against its own docstrings and SQL.
Two real simplifications fall out of moving to Firestore, not just a port:

1. **No decision-ledger/mirror split.** `SqliteReplayControlPlane` had to
   write an owner decision to `checkpoints.db` (the graph's own
   authoritative ledger, shared with `causalops.approvals`/the CLI) FIRST,
   then mirror it into `control-plane.db`, then reconcile the crash window
   between those two separate SQLite files on every `claim_next()` --
   `_ensure_ledger_decision`/`_ledger_decision`/`_reconcile_durable_
   decisions`. Once the hosted control plane's checkpointer is ALSO
   Firestore-backed (`firestore_checkpointer.py`), the "same backend for
   decisions and checkpoints" invariant that split existed to preserve is
   satisfied by ONE `owner_decisions` collection here -- no ledger/mirror
   distinction, no reconciliation pass needed at all. (The CLI's own local
   `investigate`/`approve`/`reject` path is unaffected: it keeps using
   `causalops.approvals`'s SQLite ledger against its own local
   `checkpoints.db`, since its own checkpoints stay SQLite -- CLI and the
   hosted API never share an investigation, so they never need to agree
   with each other, only internally with themselves.)
2. **Firestore's `WriteBatch`** gives real atomic multi-document writes
   (e.g. a job's status change plus its own new event) without needing a
   SQL-style transaction -- used wherever `SqliteReplayControlPlane` relied
   on one `BEGIN IMMEDIATE` transaction to write more than one row.

Concurrency model: every read-check-write sequence below is guarded by one
`threading.Lock`, mirroring what `BEGIN IMMEDIATE` gives
`SqliteReplayControlPlane` -- exclusive access for the duration of one
logical operation. This is correct and sufficient for today's deployment
(exactly one `BackgroundControlPlaneWorkers` process; see
`firestore_checkpointer.py`'s own docstring for the identical reasoning
about `put_writes`), and needs real Firestore transactions before -- not
after -- a multi-process Pub/Sub worker fan-out (a later phase) makes
concurrent writers from DIFFERENT processes a real possibility (a
`threading.Lock` only ever serializes within one process).

Local disk still holds the one thing that stays local regardless of
control-plane backend: `artifacts_root`, the working directory
`finalize_investigation` writes finished reports into before this class
snapshots their content into Firestore -- the same role it already plays
for `SqliteReplayControlPlane` and for `GcsArtifactStore`'s own upload
source.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from google.cloud import firestore

from causalops.api import (
    ControlPlaneConflictError,
    ControlPlaneNotFoundError,
    CreateInvestigationRequest,
    DecisionRequest,
    DeliveryClaim,
    InvestigationStatus,
    InvestigationView,
    ReplayControlPlane,
    ReplaySeed,
    TimelineEvent,
    WorkerClaim,
)
from causalops.report_snapshot import read_report_snapshot, validate_report_reference

DEFAULT_CLAIM_LEASE_SECONDS = 300.0
DEFAULT_RETRY_BASE_SECONDS = 5.0
DEFAULT_MAX_REPLAY_ATTEMPTS = 3
DEFAULT_MAX_DELIVERY_ATTEMPTS = 3

_JOBS_COLLECTION = "replay_jobs"
_EVENTS_SUBCOLLECTION = "events"
_DELIVERIES_COLLECTION = "report_deliveries"
_DECISIONS_COLLECTION = "owner_decisions"


class ControlPlaneIntegrityError(RuntimeError):
    """A stored document violates an invariant this class relies on."""


class _DocumentSnapshot(Protocol):
    exists: bool

    def to_dict(self) -> dict[str, Any] | None: ...


class _DocumentReference(Protocol):
    def get(self) -> _DocumentSnapshot: ...
    def set(self, document_data: dict[str, Any]) -> object: ...
    def collection(self, collection_id: str) -> _CollectionReference: ...


class _Query(Protocol):
    def where(self, field_path: str, op_string: str, value: object) -> _Query: ...
    def stream(self) -> Iterator[_DocumentSnapshot]: ...


class _CollectionReference(_Query, Protocol):
    def document(self, document_id: str) -> _DocumentReference: ...


class _WriteBatch(Protocol):
    def set(
        self, reference: _DocumentReference, document_data: dict[str, Any]
    ) -> None: ...
    def commit(self) -> object: ...


class _Client(Protocol):
    """The narrow seam this class actually calls -- a real
    `firestore.Client` satisfies this structurally; tests inject a fake
    client shaped the same way, the same seam-testing approach this
    project already uses for `GcsArtifactStore`/`FirestoreCheckpointSaver`.
    """

    def collection(self, collection_id: str) -> _CollectionReference: ...
    def batch(self) -> _WriteBatch: ...


def _validate_positive(**values: float | int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


class FirestoreReplayControlPlane(ReplayControlPlane):
    """Firestore-backed, owner-scoped job metadata, checkpoints-adjacent
    decisions, and delivery outbox."""

    def __init__(
        self,
        artifacts_root: Path,
        *,
        client: _Client | None = None,
        clock: Any = time.time,
        claim_lease_seconds: float = DEFAULT_CLAIM_LEASE_SECONDS,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        max_replay_attempts: int = DEFAULT_MAX_REPLAY_ATTEMPTS,
        max_delivery_attempts: int = DEFAULT_MAX_DELIVERY_ATTEMPTS,
    ) -> None:
        _validate_positive(
            claim_lease_seconds=claim_lease_seconds,
            retry_base_seconds=retry_base_seconds,
            max_replay_attempts=max_replay_attempts,
            max_delivery_attempts=max_delivery_attempts,
        )
        self._client: _Client = (
            client if client is not None else cast(_Client, firestore.Client())
        )
        self._artifacts_root = artifacts_root
        self._clock = clock
        self._claim_lease_seconds = claim_lease_seconds
        self._retry_base_seconds = retry_base_seconds
        self._max_replay_attempts = max_replay_attempts
        self._max_delivery_attempts = max_delivery_attempts
        self._lock = threading.Lock()

    # -- collection/document accessors --------------------------------

    def _jobs(self) -> _CollectionReference:
        return self._client.collection(_JOBS_COLLECTION)

    def _job_ref(self, investigation_id: str) -> _DocumentReference:
        return self._jobs().document(investigation_id)

    def _deliveries(self) -> _CollectionReference:
        return self._client.collection(_DELIVERIES_COLLECTION)

    def _delivery_ref(self, investigation_id: str) -> _DocumentReference:
        return self._deliveries().document(investigation_id)

    def _decisions(self) -> _CollectionReference:
        return self._client.collection(_DECISIONS_COLLECTION)

    def _decision_ref(self, investigation_id: str) -> _DocumentReference:
        return self._decisions().document(investigation_id)

    def _events(self, investigation_id: str) -> _CollectionReference:
        return self._job_ref(investigation_id).collection(_EVENTS_SUBCOLLECTION)

    # -- shared helpers --------------------------------------------------

    def _next_event_ordinal(self, investigation_id: str) -> int:
        return sum(1 for _ in self._events(investigation_id).stream())

    def _event(self, investigation_id: str, name: str, fields: dict[str, Any]) -> None:
        ordinal = self._next_event_ordinal(investigation_id)
        self._events(investigation_id).document(f"{ordinal:010d}").set(
            {"ordinal": ordinal, "name": name, "fields": fields}
        )

    def _stage_event(
        self,
        batch: _WriteBatch,
        investigation_id: str,
        name: str,
        fields: dict[str, Any],
    ) -> None:
        ordinal = self._next_event_ordinal(investigation_id)
        batch.set(
            self._events(investigation_id).document(f"{ordinal:010d}"),
            {"ordinal": ordinal, "name": name, "fields": fields},
        )

    @staticmethod
    def _view(data: dict[str, Any]) -> InvestigationView:
        if data.get("failure_reason") is not None:
            return InvestigationView(
                investigation_id=data["id"], status=InvestigationStatus.FAILED_SAFE
            )
        try:
            job_status = InvestigationStatus(data["status"])
        except ValueError as error:
            raise ControlPlaneIntegrityError(
                "invalid stored investigation status"
            ) from error
        return InvestigationView(investigation_id=data["id"], status=job_status)

    def _owned(self, investigation_id: str, owner_email: str) -> dict[str, Any]:
        snapshot = self._job_ref(investigation_id).get()
        if not snapshot.exists:
            raise ControlPlaneNotFoundError("investigation not found")
        data = snapshot.to_dict()
        assert data is not None
        if data.get("owner") != owner_email:
            raise ControlPlaneNotFoundError("investigation not found")
        return data

    def _retry_delay(self, attempt: int) -> float:
        return float(min(self._retry_base_seconds * (2 ** (attempt - 1)), 3600.0))

    def _reclaim_expired_investigations(self, now: float) -> None:
        for snapshot in (
            self._jobs()
            .where("status", "==", InvestigationStatus.RUNNING.value)
            .stream()
        ):
            data = snapshot.to_dict()
            assert data is not None
            expires_at = data.get("claim_expires_at")
            if expires_at is None or expires_at > now:
                continue
            investigation_id = data["id"]
            attempt = int(data["attempt_count"]) + 1
            if attempt >= self._max_replay_attempts:
                data.update(
                    status=InvestigationStatus.QUEUED.value,
                    claim_token=None,
                    claim_expires_at=None,
                    attempt_count=attempt,
                    failure_reason="retry_limit:lease_expired",
                )
                self._job_ref(investigation_id).set(data)
                self._event(
                    investigation_id, "investigation_failed", {"attempt_count": attempt}
                )
                continue
            retry_at = now + self._retry_delay(attempt)
            data.update(
                status=InvestigationStatus.QUEUED.value,
                claim_token=None,
                claim_expires_at=None,
                attempt_count=attempt,
                available_at=retry_at,
            )
            self._job_ref(investigation_id).set(data)
            self._event(
                investigation_id,
                "investigation_lease_expired",
                {"attempt_count": attempt, "retry_at": retry_at},
            )

    def _reclaim_expired_deliveries(self, now: float) -> None:
        for snapshot in self._deliveries().where("status", "==", "SENDING").stream():
            data = snapshot.to_dict()
            assert data is not None
            expires_at = data.get("claim_expires_at")
            if expires_at is None or expires_at > now:
                continue
            investigation_id = data["id"]
            attempt = int(data["attempt_count"]) + 1
            if attempt >= self._max_delivery_attempts:
                data.update(
                    status="PENDING",
                    claim_token=None,
                    claim_expires_at=None,
                    attempt_count=attempt,
                    failure_reason="retry_limit:lease_expired",
                )
                self._delivery_ref(investigation_id).set(data)
                self._event(
                    investigation_id,
                    "report_delivery_failed",
                    {"attempt_count": attempt},
                )
                continue
            retry_at = now + self._retry_delay(attempt)
            data.update(
                status="PENDING",
                claim_token=None,
                claim_expires_at=None,
                attempt_count=attempt,
                available_at=retry_at,
            )
            self._delivery_ref(investigation_id).set(data)
            self._event(
                investigation_id,
                "report_delivery_lease_expired",
                {"attempt_count": attempt, "retry_at": retry_at},
            )

    # -- ReplayControlPlane Protocol -------------------------------------

    def create(
        self,
        owner_email: str,
        request: CreateInvestigationRequest,
        idempotency_key: str,
    ) -> InvestigationView:
        with self._lock:
            for snapshot in self._jobs().where("owner", "==", owner_email).stream():
                data = snapshot.to_dict()
                assert data is not None
                if data.get("idempotency_key") == idempotency_key:
                    return self._view(data)
            investigation_id = uuid4().hex
            # Firestore documents carry no equivalent of SQLite's own
            # auto-increment `rowid`, which `claim_next()`'s SQL ORDER BY
            # relies on as its final FIFO tiebreaker -- an explicit
            # creation-order sequence, assigned once under this same lock,
            # replaces it directly rather than falling back to `id` (a
            # random UUID with no ordering meaning) or wall-clock time
            # (two creates can land on the same tick).
            sequence = sum(1 for _ in self._jobs().stream())
            data = {
                "id": investigation_id,
                "sequence": sequence,
                "owner": owner_email,
                "status": InvestigationStatus.QUEUED.value,
                "scenario_family": request.scenario_family.value,
                "seed": ReplaySeed.DEVELOPMENT.value,
                "idempotency_key": idempotency_key,
                "checkpoint_id": None,
                "incident_id": None,
                "report_artifact": None,
                "report_content": None,
                "report_sha256": None,
                "attempt_count": 0,
                "available_at": 0.0,
                "failure_reason": None,
                "claim_token": None,
                "claim_expires_at": None,
            }
            batch = self._client.batch()
            batch.set(self._job_ref(investigation_id), data)
            self._stage_event(
                batch, investigation_id, "investigation_queued", {"replay_only": True}
            )
            batch.commit()
        return InvestigationView(
            investigation_id=investigation_id, status=InvestigationStatus.QUEUED
        )

    def status(self, owner_email: str, investigation_id: str) -> InvestigationView:
        return self._view(self._owned(investigation_id, owner_email))

    def events(self, owner_email: str, investigation_id: str) -> list[TimelineEvent]:
        self._owned(investigation_id, owner_email)
        rows = []
        for snapshot in self._events(investigation_id).stream():
            data = snapshot.to_dict()
            assert data is not None
            rows.append(data)
        rows.sort(key=lambda row: int(row["ordinal"]))
        return [
            TimelineEvent(
                ordinal=row["ordinal"], name=row["name"], fields=row["fields"]
            )
            for row in rows
        ]

    def decide(
        self, owner_email: str, investigation_id: str, decision: DecisionRequest
    ) -> InvestigationView:
        with self._lock:
            job = self._owned(investigation_id, owner_email)
            decision_snapshot = self._decision_ref(investigation_id).get()
            if decision_snapshot.exists:
                prior_data = decision_snapshot.to_dict()
                assert prior_data is not None
                prior_decision = DecisionRequest(
                    decision=prior_data["decision"],
                    rejection_note=prior_data["rejection_note"],
                )
                checkpoint_id = job.get("checkpoint_id")
                if not checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "decision record has no paused checkpoint"
                    )
                if prior_data["checkpoint_id"] != checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "decision record belongs to a different checkpoint"
                    )
                if prior_decision != decision:
                    raise ControlPlaneConflictError(
                        "a different owner decision is recorded"
                    )
                return self._view(job)
            if job["status"] != InvestigationStatus.PAUSED_APPROVAL.value:
                raise ControlPlaneConflictError(
                    "investigation is not waiting for a decision"
                )
            checkpoint_id = job.get("checkpoint_id")
            if not checkpoint_id:
                raise ControlPlaneIntegrityError(
                    "paused investigation has no checkpoint"
                )
            job["status"] = InvestigationStatus.QUEUED.value
            batch = self._client.batch()
            batch.set(
                self._decision_ref(investigation_id),
                {
                    "checkpoint_id": checkpoint_id,
                    "decision": decision.decision,
                    "rejection_note": decision.rejection_note,
                },
            )
            batch.set(self._job_ref(investigation_id), job)
            self._stage_event(
                batch,
                investigation_id,
                "owner_decision_recorded",
                {"checkpoint_id": checkpoint_id, "decision": decision.decision},
            )
            batch.commit()
        return InvestigationView(
            investigation_id=investigation_id, status=InvestigationStatus.QUEUED
        )

    def report(self, owner_email: str, investigation_id: str) -> str:
        job = self._owned(investigation_id, owner_email)
        if job["status"] != InvestigationStatus.COMPLETED.value:
            raise ControlPlaneConflictError("report is not finalized")
        stored_artifact = job.get("report_artifact")
        if not isinstance(stored_artifact, str) or not stored_artifact:
            raise ControlPlaneIntegrityError(
                "finalized investigation has no report artifact"
            )
        stored_content = job.get("report_content")
        stored_sha256 = job.get("report_sha256")
        if not isinstance(stored_content, str) or not isinstance(stored_sha256, str):
            raise ControlPlaneIntegrityError(
                "finalized investigation has no report snapshot"
            )
        try:
            validate_report_reference(investigation_id, stored_artifact)
        except ValueError as error:
            raise ControlPlaneIntegrityError(
                "finalized investigation report is unavailable"
            ) from error
        actual_sha256 = hashlib.sha256(stored_content.encode("utf-8")).hexdigest()
        if actual_sha256 != stored_sha256:
            raise ControlPlaneIntegrityError(
                "finalized investigation report is corrupt"
            )
        return stored_content

    # -- worker-facing (scenario/lease/checkpoint) -----------------------

    @property
    def claim_lease_seconds(self) -> float:
        return self._claim_lease_seconds

    def renew_running(self, investigation_id: str, claim_token: str) -> None:
        with self._lock:
            snapshot = self._job_ref(investigation_id).get()
            now = self._clock()
            data = snapshot.to_dict() if snapshot.exists else None
            if (
                data is None
                or data["status"] != InvestigationStatus.RUNNING.value
                or data.get("claim_token") != claim_token
                or data.get("claim_expires_at") is None
                or data["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )
            data["claim_expires_at"] = now + self._claim_lease_seconds
            self._job_ref(investigation_id).set(data)

    def reserve_incident(self, investigation_id: str, claim_token: str) -> str:
        """Durably allocate a scenario identity before provisioning the lab.

        The job id is opaque and alphanumeric, so it is a stable incident
        id without introducing a second allocation that could be lost
        between a Firestore write and `start_scenario`.
        """
        incident_id = f"scenario{investigation_id}"
        with self._lock:
            snapshot = self._job_ref(investigation_id).get()
            if not snapshot.exists:
                raise ControlPlaneNotFoundError("investigation not found")
            data = snapshot.to_dict()
            assert data is not None
            now = self._clock()
            if (
                data["status"] != InvestigationStatus.RUNNING.value
                or data.get("claim_token") != claim_token
                or data.get("claim_expires_at") is None
                or data["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )
            stored_incident_id = data.get("incident_id")
            if stored_incident_id is None:
                data["incident_id"] = incident_id
                self._job_ref(investigation_id).set(data)
                self._event(
                    investigation_id, "scenario_reserved", {"incident_id": incident_id}
                )
                return incident_id
            if (
                not isinstance(stored_incident_id, str)
                or not stored_incident_id.isalnum()
            ):
                raise ControlPlaneIntegrityError("stored incident id is invalid")
            return stored_incident_id

    def scenario_status(self, incident_id: str) -> InvestigationStatus | None:
        """Return the owner-hidden job state associated with an active
        marker."""
        matches = list(self._jobs().where("incident_id", "==", incident_id).stream())
        if not matches:
            return None
        data = matches[0].to_dict()
        assert data is not None
        if data.get("failure_reason") is not None:
            return InvestigationStatus.FAILED_SAFE
        try:
            return InvestigationStatus(data["status"])
        except ValueError as error:
            raise ControlPlaneIntegrityError(
                "invalid stored investigation status"
            ) from error

    def incident_id_for(self, investigation_id: str) -> str | None:
        snapshot = self._job_ref(investigation_id).get()
        if not snapshot.exists:
            raise ControlPlaneNotFoundError("investigation not found")
        data = snapshot.to_dict()
        assert data is not None
        incident_id = data.get("incident_id")
        if incident_id is None:
            return None
        if not isinstance(incident_id, str) or not incident_id.isalnum():
            raise ControlPlaneIntegrityError("stored incident id is invalid")
        return incident_id

    # -- worker-facing (claim/finalize/retry) ----------------------------

    def claim_next(self) -> WorkerClaim | None:
        """Atomically claim one queued replay job for an injected worker."""
        with self._lock:
            now = self._clock()
            self._reclaim_expired_investigations(now)
            # The synthetic lab exposes one mutable active scenario. This
            # guard applies within this process (see the module docstring's
            # concurrency-model note), not merely within one thread, so two
            # workers cannot contaminate each other's telemetry while
            # processing different jobs.
            running = any(
                True
                for _ in self._jobs()
                .where("status", "==", InvestigationStatus.RUNNING.value)
                .stream()
            )
            if running:
                return None
            queued = []
            for snapshot in (
                self._jobs()
                .where("status", "==", InvestigationStatus.QUEUED.value)
                .stream()
            ):
                data = snapshot.to_dict()
                assert data is not None
                if data.get("failure_reason") is not None:
                    continue
                queued.append(data)
            # A retryable job that already reserved an incident retains the
            # sole lab scenario during backoff. Starting a different job in
            # that interval would make its telemetry observe the wrong
            # fault, so any such job blocks every OTHER job from starting
            # too -- exactly the sqlite version's own `deferred_scenario`
            # guard, checked across ALL queued jobs, not just the ones
            # otherwise eligible to claim right now.
            if any(
                data.get("incident_id") is not None and data["available_at"] > now
                for data in queued
            ):
                return None
            candidates = [data for data in queued if data["available_at"] <= now]
            if not candidates:
                return None
            candidates.sort(
                key=lambda data: (
                    0 if data.get("incident_id") is not None else 1,
                    data["available_at"],
                    data["sequence"],
                )
            )
            row = candidates[0]
            investigation_id = row["id"]
            claim_token = uuid4().hex
            row["status"] = InvestigationStatus.RUNNING.value
            row["claim_token"] = claim_token
            row["claim_expires_at"] = now + self._claim_lease_seconds
            self._job_ref(investigation_id).set(row)
            self._event(investigation_id, "investigation_started", {})
            decision_snapshot = self._decision_ref(investigation_id).get()
            owner_decision: DecisionRequest | None = None
            if decision_snapshot.exists:
                decision_data = decision_snapshot.to_dict()
                assert decision_data is not None
                try:
                    owner_decision = DecisionRequest(
                        decision=decision_data["decision"],
                        rejection_note=decision_data["rejection_note"],
                    )
                except ValueError as error:
                    raise ControlPlaneIntegrityError(
                        "stored owner decision is invalid"
                    ) from error
                checkpoint_id = row.get("checkpoint_id")
                if not isinstance(checkpoint_id, str) or not checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "resumed investigation has no checkpoint"
                    )
                if decision_data["checkpoint_id"] != checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "stored owner decision belongs to a different checkpoint"
                    )
            return WorkerClaim(
                investigation_id=row["id"],
                owner_email=row["owner"],
                scenario_family=row["scenario_family"],
                seed=row["seed"],
                checkpoint_id=row.get("checkpoint_id"),
                incident_id=row.get("incident_id"),
                owner_decision=owner_decision,
                claim_token=claim_token,
            )

    def mark_paused(
        self, investigation_id: str, checkpoint_id: str, claim_token: str
    ) -> None:
        """Persist the graph checkpoint before making a job resumable in
        the UI."""
        if not checkpoint_id.strip():
            raise ValueError("checkpoint_id must not be blank")
        with self._lock:
            snapshot = self._job_ref(investigation_id).get()
            if not snapshot.exists:
                raise ControlPlaneNotFoundError("investigation not found")
            data = snapshot.to_dict()
            assert data is not None
            now = self._clock()
            if (
                data["status"] != InvestigationStatus.RUNNING.value
                or data.get("claim_token") != claim_token
                or data.get("claim_expires_at") is None
                or data["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )
            data.update(
                status=InvestigationStatus.PAUSED_APPROVAL.value,
                checkpoint_id=checkpoint_id,
                claim_token=None,
                claim_expires_at=None,
            )
            batch = self._client.batch()
            batch.set(self._job_ref(investigation_id), data)
            self._stage_event(
                batch,
                investigation_id,
                "owner_decision_requested",
                {"checkpoint_id": checkpoint_id},
            )
            batch.commit()

    def requeue_running(self, investigation_id: str, claim_token: str) -> None:
        """Release a crashed worker claim without dropping its durable
        state."""
        with self._lock:
            snapshot = self._job_ref(investigation_id).get()
            data = snapshot.to_dict() if snapshot.exists else None
            if (
                data is None
                or data["status"] != InvestigationStatus.RUNNING.value
                or data.get("claim_token") != claim_token
            ):
                raise ControlPlaneConflictError(
                    "only a running investigation may be requeued"
                )
            data.update(
                status=InvestigationStatus.QUEUED.value,
                claim_token=None,
                claim_expires_at=None,
            )
            self._job_ref(investigation_id).set(data)
            self._event(investigation_id, "investigation_requeued", {})

    def retry_running(
        self, investigation_id: str, claim_token: str, error: Exception
    ) -> bool:
        """Schedule a bounded retry and return whether the job remains
        retryable."""
        with self._lock:
            snapshot = self._job_ref(investigation_id).get()
            if not snapshot.exists:
                raise ControlPlaneNotFoundError("investigation not found")
            data = snapshot.to_dict()
            assert data is not None
            if (
                data["status"] != InvestigationStatus.RUNNING.value
                or data.get("claim_token") != claim_token
            ):
                raise ControlPlaneConflictError(
                    "only a running investigation may be retried"
                )
            attempt = int(data["attempt_count"]) + 1
            if attempt >= self._max_replay_attempts:
                data.update(
                    status=InvestigationStatus.QUEUED.value,
                    claim_token=None,
                    claim_expires_at=None,
                    attempt_count=attempt,
                    failure_reason=f"retry_limit:{type(error).__name__}",
                )
                self._job_ref(investigation_id).set(data)
                self._event(
                    investigation_id,
                    "investigation_failed",
                    {"attempt_count": attempt},
                )
                return False
            retry_at = self._clock() + self._retry_delay(attempt)
            data.update(
                status=InvestigationStatus.QUEUED.value,
                claim_token=None,
                claim_expires_at=None,
                attempt_count=attempt,
                available_at=retry_at,
            )
            self._job_ref(investigation_id).set(data)
            self._event(
                investigation_id,
                "investigation_retry_scheduled",
                {"attempt_count": attempt, "retry_at": retry_at},
            )
            return True

    def finalize(
        self, investigation_id: str, report_artifact: str, claim_token: str
    ) -> None:
        """Record an already write-once artifact and enqueue owner-only
        delivery."""
        if not report_artifact.strip():
            raise ValueError("report_artifact must not be blank")
        with self._lock:
            snapshot = self._job_ref(investigation_id).get()
            if not snapshot.exists:
                raise ControlPlaneNotFoundError("investigation not found")
            data = snapshot.to_dict()
            assert data is not None
            now = self._clock()
            if data["status"] == InvestigationStatus.COMPLETED.value:
                raise ControlPlaneConflictError("investigation is already finalized")
            if (
                data["status"] != InvestigationStatus.RUNNING.value
                or data.get("claim_token") != claim_token
                or data.get("claim_expires_at") is None
                or data["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )
            report_content = read_report_snapshot(
                self._artifacts_root, investigation_id, report_artifact
            )
            report_sha256 = hashlib.sha256(report_content.encode("utf-8")).hexdigest()
            data.update(
                status=InvestigationStatus.COMPLETED.value,
                report_artifact=report_artifact,
                report_content=report_content,
                report_sha256=report_sha256,
                claim_token=None,
                claim_expires_at=None,
            )
            # Same FIFO tiebreak concern as `create()`'s own `sequence`
            # field (see its comment): a delivery's own `id` is the random
            # investigation UUID, not a creation-order signal.
            delivery_sequence = sum(1 for _ in self._deliveries().stream())
            batch = self._client.batch()
            batch.set(self._job_ref(investigation_id), data)
            batch.set(
                self._delivery_ref(investigation_id),
                {
                    "id": investigation_id,
                    "sequence": delivery_sequence,
                    "destination_email": data["owner"],
                    "status": "PENDING",
                    "claim_token": None,
                    "claim_expires_at": None,
                    "attempt_count": 0,
                    "available_at": 0.0,
                    "failure_reason": None,
                },
            )
            self._stage_event(batch, investigation_id, "report_finalized", {})
            batch.commit()

    # -- delivery-worker-facing -------------------------------------------

    def claim_delivery(self) -> DeliveryClaim | None:
        """Claim delivery after finalization; retry policy belongs to the
        sender."""
        with self._lock:
            now = self._clock()
            self._reclaim_expired_deliveries(now)
            candidates = []
            for snapshot in (
                self._deliveries().where("status", "==", "PENDING").stream()
            ):
                data = snapshot.to_dict()
                assert data is not None
                if data.get("failure_reason") is not None:
                    continue
                if data["available_at"] > now:
                    continue
                candidates.append(data)
            if not candidates:
                return None
            candidates.sort(key=lambda data: (data["available_at"], data["sequence"]))
            delivery = candidates[0]
            investigation_id = delivery["id"]
            job_snapshot = self._job_ref(investigation_id).get()
            if not job_snapshot.exists:
                raise ControlPlaneIntegrityError(
                    "delivery references a missing investigation"
                )
            job = job_snapshot.to_dict()
            assert job is not None
            claim_token = uuid4().hex
            delivery["status"] = "SENDING"
            delivery["claim_token"] = claim_token
            delivery["claim_expires_at"] = now + self._claim_lease_seconds
            self._delivery_ref(investigation_id).set(delivery)
            return DeliveryClaim(
                investigation_id=investigation_id,
                status=InvestigationStatus(job["status"]),
                report_artifact=job["report_artifact"],
                report_content=job["report_content"],
                report_sha256=job["report_sha256"],
                destination_email=delivery["destination_email"],
                delivery_id=investigation_id,
                claim_token=claim_token,
            )

    def mark_delivered(self, investigation_id: str, claim_token: str) -> None:
        with self._lock:
            snapshot = self._delivery_ref(investigation_id).get()
            now = self._clock()
            data = snapshot.to_dict() if snapshot.exists else None
            if (
                data is None
                or data["status"] != "SENDING"
                or data.get("claim_token") != claim_token
                or data.get("claim_expires_at") is None
                or data["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError("delivery is not currently claimed")
            data.update(status="SENT", claim_token=None, claim_expires_at=None)
            self._delivery_ref(investigation_id).set(data)
            self._event(investigation_id, "report_delivered", {})

    def renew_delivery(self, investigation_id: str, claim_token: str) -> None:
        """Extend an active delivery lease while its sender is still
        running."""
        with self._lock:
            snapshot = self._delivery_ref(investigation_id).get()
            now = self._clock()
            data = snapshot.to_dict() if snapshot.exists else None
            if (
                data is None
                or data["status"] != "SENDING"
                or data.get("claim_token") != claim_token
                or data.get("claim_expires_at") is None
                or data["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError("delivery is not currently claimed")
            data["claim_expires_at"] = now + self._claim_lease_seconds
            self._delivery_ref(investigation_id).set(data)

    def release_delivery(self, investigation_id: str, claim_token: str) -> None:
        """Return a failed delivery claim to the outbox for an explicit
        retry."""
        with self._lock:
            snapshot = self._delivery_ref(investigation_id).get()
            data = snapshot.to_dict() if snapshot.exists else None
            if (
                data is None
                or data["status"] != "SENDING"
                or data.get("claim_token") != claim_token
            ):
                raise ControlPlaneConflictError("delivery is not currently claimed")
            data.update(status="PENDING", claim_token=None, claim_expires_at=None)
            self._delivery_ref(investigation_id).set(data)
            self._event(investigation_id, "report_delivery_requeued", {})

    def retry_delivery(
        self, investigation_id: str, claim_token: str, error: Exception
    ) -> bool:
        """Schedule a bounded delivery retry without blocking other
        reports."""
        with self._lock:
            snapshot = self._delivery_ref(investigation_id).get()
            if not snapshot.exists:
                raise ControlPlaneNotFoundError("report delivery not found")
            data = snapshot.to_dict()
            assert data is not None
            if data["status"] != "SENDING" or data.get("claim_token") != claim_token:
                raise ControlPlaneConflictError("delivery is not currently claimed")
            attempt = int(data["attempt_count"]) + 1
            if attempt >= self._max_delivery_attempts:
                data.update(
                    status="PENDING",
                    claim_token=None,
                    claim_expires_at=None,
                    attempt_count=attempt,
                    failure_reason=f"retry_limit:{type(error).__name__}",
                )
                self._delivery_ref(investigation_id).set(data)
                self._event(
                    investigation_id,
                    "report_delivery_failed",
                    {"attempt_count": attempt},
                )
                return False
            retry_at = self._clock() + self._retry_delay(attempt)
            data.update(
                status="PENDING",
                claim_token=None,
                claim_expires_at=None,
                attempt_count=attempt,
                available_at=retry_at,
            )
            self._delivery_ref(investigation_id).set(data)
            self._event(
                investigation_id,
                "report_delivery_retry_scheduled",
                {"attempt_count": attempt, "retry_at": retry_at},
            )
            return True
