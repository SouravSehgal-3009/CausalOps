"""Durable replay-only control-plane composition and worker hand-off.

This module persists control-plane facts, not graph state. A replay worker
claims a queued job, runs the already-injected replay graph, and uses the
checkpoint and finalization methods below to make API resume and delivery safe
across process restarts. It has no provider-selection input.
"""

import hashlib
import json
import os
import sqlite3
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from threading import Event, Thread
from typing import Protocol, cast
from uuid import uuid4

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict

from causalops.api import (
    ControlPlaneConflictError,
    ControlPlaneNotFoundError,
    ControlPlaneWorkerService,
    CreateInvestigationRequest,
    DecisionRequest,
    DeliveryClaim,
    InvestigationStatus,
    InvestigationView,
    ReplayControlPlane,
    ReplaySeed,
    TimelineEvent,
    WorkerClaim,
    create_app,
)
from causalops.approvals import (
    CheckpointStoreError,
    DecisionRow,
    ensure_decisions_table,
    read_decision_for_thread,
    record_decision_before_resume,
)
from causalops.cli import _load_verified_incident, _sqlite_checkpointer
from causalops.doctor import ProjectPaths, find_project_root
from causalops.domain import Budgets, EscalatedInvestigation, utc_now
from causalops.firestore_checkpointer import FirestoreCheckpointSaver
from causalops.firestore_control_plane import FirestoreReplayControlPlane
from causalops.gcs_artifacts import ARTIFACT_NAMES, ArtifactStore, GcsArtifactStore
from causalops.google_identity import GoogleIdentityVerifier
from causalops.graph import (
    recover_graph_investigation,
    resume_graph_investigation,
    run_graph_investigation,
)
from causalops.live_setup import (
    VM_EXECUTION_ENV,
    VM_EXECUTION_ENV_VARIABLE,
    HostedReplayRuntimeWiring,
    ReplayRuntimeWiring,
)
from causalops.mcp_client_registry import (
    McpBackedLiveRuntimeWiring,
    McpBackedReplayRuntimeWiring,
)
from causalops.report import render_report as render_markdown_report
from causalops.report_snapshot import (
    artifact_path,
    read_report_snapshot,
    validate_report_reference,
)
from causalops.run_records import RunRecorder, finalize_investigation
from causalops.scenario_control import (
    LabError,
    active_incident_file,
    release_scenario,
    reset_scenario,
    start_scenario,
    validated_run_paths,
)


class ControlPlaneIntegrityError(RuntimeError):
    """A durable control-plane record cannot be safely interpreted."""


class PausedWorkerOutcome(BaseModel):
    """A worker stopped at a durable graph checkpoint for owner review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: str


class FinalizedWorkerOutcome(BaseModel):
    """A worker finalized its report with the repository's write-once store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_artifact: str


WorkerOutcome = PausedWorkerOutcome | FinalizedWorkerOutcome


class ReplayJobRunner(Protocol):
    """Injected replay-graph adapter; provider selection is not available here."""

    def run(self, claim: WorkerClaim) -> WorkerOutcome: ...


# `_sqlite_checkpointer` and a Firestore equivalent both open a
# `BaseCheckpointSaver[str]` for the caller's own `with` block -- a
# zero-argument factory returning that context manager lets
# `ReplayGraphJobRunner` pick either backend without knowing which one it
# got, the same swap `WorkerControlPlane` makes for the control plane.
CheckpointerFactory = Callable[[], AbstractContextManager[BaseCheckpointSaver[str]]]


class ReportDeliverySender(Protocol):
    """Injected sender for a finalized report delivery claim."""

    def send(self, claim: DeliveryClaim) -> None: ...


class WorkerControlPlane(Protocol):
    """The worker-facing surface `ReplayWorker`/`ReportDeliveryWorker`/
    `ReplayGraphJobRunner` actually call -- narrower than the full
    `SqliteReplayControlPlane`/`FirestoreReplayControlPlane` classes, and
    disjoint from `causalops.api.ReplayControlPlane` (that one is the
    owner-facing HTTP surface only: create/status/events/decide/report).
    Both concrete control planes satisfy this structurally without
    declaring it, the same seam-typing approach this project already uses
    for `ReplayJobRunner`/`ReportDeliverySender` above -- it lets `app()`
    choose either backend without either worker class importing the other
    backend's module.
    """

    @property
    def claim_lease_seconds(self) -> float: ...

    def claim_next(self) -> WorkerClaim | None: ...

    def reserve_incident(self, investigation_id: str, claim_token: str) -> str: ...

    def incident_id_for(self, investigation_id: str) -> str | None: ...

    def scenario_status(self, incident_id: str) -> InvestigationStatus | None: ...

    def renew_running(self, investigation_id: str, claim_token: str) -> None: ...

    def mark_paused(
        self, investigation_id: str, checkpoint_id: str, claim_token: str
    ) -> None: ...

    def retry_running(
        self, investigation_id: str, claim_token: str, error: Exception
    ) -> bool: ...

    def finalize(
        self, investigation_id: str, report_artifact: str, claim_token: str
    ) -> None: ...

    def claim_delivery(self) -> DeliveryClaim | None: ...

    def renew_delivery(self, investigation_id: str, claim_token: str) -> None: ...

    def mark_delivered(self, investigation_id: str, claim_token: str) -> None: ...

    def retry_delivery(
        self, investigation_id: str, claim_token: str, error: Exception
    ) -> bool: ...


DEFAULT_CLAIM_LEASE_SECONDS = 300.0
MINIMUM_HEARTBEAT_SECONDS = 0.05
DEFAULT_RETRY_BASE_SECONDS = 5.0
DEFAULT_MAX_REPLAY_ATTEMPTS = 3
DEFAULT_MAX_DELIVERY_ATTEMPTS = 3


class SqliteReplayControlPlane(ReplayControlPlane):
    """SQLite-backed, owner-scoped job metadata, checkpoints, and outbox."""

    def __init__(
        self,
        database: Path,
        artifacts_root: Path | None = None,
        checkpoint_database: Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        claim_lease_seconds: float = DEFAULT_CLAIM_LEASE_SECONDS,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        max_replay_attempts: int = DEFAULT_MAX_REPLAY_ATTEMPTS,
        max_delivery_attempts: int = DEFAULT_MAX_DELIVERY_ATTEMPTS,
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be positive")
        if retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be positive")
        if max_replay_attempts <= 0 or max_delivery_attempts <= 0:
            raise ValueError("maximum retry attempts must be positive")
        self._database = database
        self._artifacts_root = artifacts_root or database.parent / "investigations"
        self._checkpoint_database = (
            checkpoint_database or database.parent / "checkpoints.db"
        )
        self._clock = clock
        self._claim_lease_seconds = claim_lease_seconds
        self._retry_base_seconds = retry_base_seconds
        self._max_replay_attempts = max_replay_attempts
        self._max_delivery_attempts = max_delivery_attempts
        database.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS replay_jobs (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN
                        ('QUEUED', 'RUNNING', 'PAUSED_APPROVAL', 'COMPLETED')),
                    scenario_family TEXT NOT NULL,
                    seed TEXT NOT NULL,
                    checkpoint_id TEXT,
                    incident_id TEXT,
                    report_artifact TEXT,
                    report_content TEXT,
                    report_sha256 TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    claim_token TEXT,
                    claim_expires_at REAL
                );
                CREATE TABLE IF NOT EXISTS replay_events (
                    job_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    fields TEXT NOT NULL,
                    PRIMARY KEY (job_id, ordinal),
                    FOREIGN KEY (job_id) REFERENCES replay_jobs(id)
                );
                CREATE TABLE IF NOT EXISTS replay_decisions (
                    job_id TEXT PRIMARY KEY,
                    checkpoint_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    rejection_note TEXT,
                    FOREIGN KEY (job_id) REFERENCES replay_jobs(id)
                );
                CREATE TABLE IF NOT EXISTS report_deliveries (
                    job_id TEXT PRIMARY KEY,
                    destination_email TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN
                        ('PENDING', 'SENDING', 'SENT')),
                    claim_token TEXT,
                    claim_expires_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    FOREIGN KEY (job_id) REFERENCES replay_jobs(id)
                );
                """
            )
            self._add_column_if_missing(connection, "replay_jobs", "claim_token TEXT")
            self._add_column_if_missing(
                connection, "replay_jobs", "claim_expires_at REAL"
            )
            self._add_column_if_missing(connection, "replay_jobs", "incident_id TEXT")
            self._add_column_if_missing(
                connection, "replay_jobs", "report_content TEXT"
            )
            self._add_column_if_missing(connection, "replay_jobs", "report_sha256 TEXT")
            self._add_column_if_missing(
                connection, "replay_jobs", "attempt_count INTEGER NOT NULL DEFAULT 0"
            )
            self._add_column_if_missing(
                connection, "replay_jobs", "available_at REAL NOT NULL DEFAULT 0"
            )
            self._add_column_if_missing(
                connection, "replay_jobs", "failure_reason TEXT"
            )
            self._add_column_if_missing(
                connection, "replay_jobs", "idempotency_key TEXT"
            )
            # Partial unique index: a repeated (owner, idempotency_key) pair
            # is the create() route's own replay case, never a real
            # collision to reject at the database layer -- see create()'s
            # own docstring.
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS replay_jobs_owner_idempotency_key "
                "ON replay_jobs(owner, idempotency_key) "
                "WHERE idempotency_key IS NOT NULL"
            )
            self._add_column_if_missing(
                connection, "report_deliveries", "claim_token TEXT"
            )
            self._add_column_if_missing(
                connection, "report_deliveries", "claim_expires_at REAL"
            )
            self._add_column_if_missing(
                connection,
                "report_deliveries",
                "attempt_count INTEGER NOT NULL DEFAULT 0",
            )
            self._add_column_if_missing(
                connection,
                "report_deliveries",
                "available_at REAL NOT NULL DEFAULT 0",
            )
            self._add_column_if_missing(
                connection, "report_deliveries", "failure_reason TEXT"
            )

    @staticmethod
    def _add_column_if_missing(
        connection: sqlite3.Connection, table: str, definition: str
    ) -> None:
        column = definition.split(" ", maxsplit=1)[0]
        existing = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _decision_connection(self) -> Iterator[sqlite3.Connection]:
        """Open the graph's authoritative owner-decision ledger.

        The control-plane database only mirrors this ledger for efficient
        worker claims. CLI and API decisions must both use the checkpoint
        database, otherwise they can authorize opposite resumes of one graph
        thread.
        """
        try:
            self._checkpoint_database.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._checkpoint_database)
            ensure_decisions_table(connection)
        except (OSError, sqlite3.Error, CheckpointStoreError) as error:
            raise ControlPlaneIntegrityError(
                "owner decision ledger is unavailable"
            ) from error
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ledger_decision(self, investigation_id: str) -> DecisionRow | None:
        try:
            with self._decision_connection() as connection:
                return read_decision_for_thread(connection, investigation_id)
        except CheckpointStoreError as error:
            raise ControlPlaneIntegrityError(
                "owner decision ledger is unreadable"
            ) from error

    def _ensure_ledger_decision(
        self,
        investigation_id: str,
        checkpoint_id: str,
        decision: DecisionRequest,
    ) -> None:
        """Persist one first-decision-wins instruction before queueing work."""
        try:
            with self._decision_connection() as connection:
                existing = read_decision_for_thread(connection, investigation_id)
                if existing is not None:
                    if existing.checkpoint_id != checkpoint_id:
                        raise ControlPlaneIntegrityError(
                            "owner decision belongs to a different checkpoint"
                        )
                    if not existing.matches(decision):
                        raise ControlPlaneConflictError(
                            "a different owner decision is recorded"
                        )
                    return
                try:
                    record_decision_before_resume(
                        connection,
                        investigation_id,
                        checkpoint_id,
                        decision,
                        utc_now(),
                    )
                except CheckpointStoreError as error:
                    # The decision helper deliberately treats a unique-key
                    # race as a conflict. Re-read it so simultaneous API and
                    # CLI requests retain first-decision-wins semantics.
                    existing = read_decision_for_thread(connection, investigation_id)
                    if (
                        existing is not None
                        and existing.checkpoint_id == checkpoint_id
                        and existing.matches(decision)
                    ):
                        return
                    raise ControlPlaneConflictError(
                        "a different owner decision is recorded"
                    ) from error
        except CheckpointStoreError as error:
            raise ControlPlaneIntegrityError(
                "owner decision ledger is unavailable"
            ) from error

    @staticmethod
    def _decision_from_row(row: sqlite3.Row) -> DecisionRequest:
        try:
            return DecisionRequest(
                decision=row["decision"], rejection_note=row["rejection_note"]
            )
        except ValueError as error:
            raise ControlPlaneIntegrityError(
                "stored owner decision is invalid"
            ) from error

    def _reconcile_durable_decisions(self, connection: sqlite3.Connection) -> None:
        """Finish an API transition interrupted after its ledger write.

        A decision is deliberately written to ``owner_decisions`` first. If
        the process dies before its control-plane transaction commits, this
        reconciliation makes the already-authorized paused job queueable on
        the next worker poll without asking the owner to submit it again.
        """
        paused_rows = connection.execute(
            "SELECT id, checkpoint_id FROM replay_jobs WHERE status = ?",
            (InvestigationStatus.PAUSED_APPROVAL.value,),
        ).fetchall()
        for row in paused_rows:
            checkpoint_id = row["checkpoint_id"]
            if not isinstance(checkpoint_id, str) or not checkpoint_id:
                raise ControlPlaneIntegrityError(
                    "paused investigation has no checkpoint"
                )
            ledger = self._ledger_decision(row["id"])
            if ledger is None:
                continue
            if ledger.checkpoint_id != checkpoint_id:
                raise ControlPlaneIntegrityError(
                    "owner decision belongs to a different checkpoint"
                )
            decision = DecisionRequest(
                decision=ledger.decision, rejection_note=ledger.rejection_note
            )
            mirrored = connection.execute(
                "SELECT checkpoint_id, decision, rejection_note FROM replay_decisions "
                "WHERE job_id = ?",
                (row["id"],),
            ).fetchone()
            if mirrored is None:
                connection.execute(
                    "INSERT INTO replay_decisions "
                    "(job_id, checkpoint_id, decision, rejection_note) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        row["id"],
                        checkpoint_id,
                        decision.decision,
                        decision.rejection_note,
                    ),
                )
            else:
                if mirrored["checkpoint_id"] != checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "control-plane decision belongs to a different checkpoint"
                    )
                mirrored_decision = self._decision_from_row(mirrored)
                if mirrored_decision != decision:
                    raise ControlPlaneIntegrityError(
                        "control-plane decision disagrees with decision ledger"
                    )
            connection.execute(
                "UPDATE replay_jobs SET status = ? WHERE id = ?",
                (InvestigationStatus.QUEUED.value, row["id"]),
            )
            self._event(
                connection,
                row["id"],
                "owner_decision_recorded",
                {"checkpoint_id": checkpoint_id, "decision": decision.decision},
            )

    @staticmethod
    def _view(row: sqlite3.Row) -> InvestigationView:
        if row["failure_reason"] is not None:
            return InvestigationView(
                investigation_id=row["id"], status=InvestigationStatus.FAILED_SAFE
            )
        try:
            job_status = InvestigationStatus(row["status"])
        except ValueError as error:
            raise ControlPlaneIntegrityError(
                "invalid stored investigation status"
            ) from error
        return InvestigationView(investigation_id=row["id"], status=job_status)

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        investigation_id: str,
        name: str,
        fields: dict[str, object],
    ) -> None:
        ordinal = connection.execute(
            "SELECT COALESCE(MAX(ordinal) + 1, 0) FROM replay_events WHERE job_id = ?",
            (investigation_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO replay_events (job_id, ordinal, name, fields) "
            "VALUES (?, ?, ?, ?)",
            (investigation_id, ordinal, name, json.dumps(fields, sort_keys=True)),
        )

    def _reclaim_expired_investigations(
        self, connection: sqlite3.Connection, now: float
    ) -> None:
        rows = connection.execute(
            "SELECT id, attempt_count FROM replay_jobs WHERE status = ? "
            "AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?",
            (InvestigationStatus.RUNNING.value, now),
        ).fetchall()
        for row in rows:
            attempt = int(row["attempt_count"]) + 1
            if attempt >= self._max_replay_attempts:
                connection.execute(
                    "UPDATE replay_jobs SET status = ?, claim_token = NULL, "
                    "claim_expires_at = NULL, attempt_count = ?, failure_reason = ? "
                    "WHERE id = ?",
                    (
                        InvestigationStatus.QUEUED.value,
                        attempt,
                        "retry_limit:lease_expired",
                        row["id"],
                    ),
                )
                self._event(
                    connection,
                    row["id"],
                    "investigation_failed",
                    {"attempt_count": attempt},
                )
                continue
            retry_at = now + self._retry_delay(attempt)
            connection.execute(
                "UPDATE replay_jobs SET status = ?, claim_token = NULL, "
                "claim_expires_at = NULL, attempt_count = ?, available_at = ? "
                "WHERE id = ?",
                (
                    InvestigationStatus.QUEUED.value,
                    attempt,
                    retry_at,
                    row["id"],
                ),
            )
            self._event(
                connection,
                row["id"],
                "investigation_lease_expired",
                {"attempt_count": attempt, "retry_at": retry_at},
            )

    def _reclaim_expired_deliveries(
        self, connection: sqlite3.Connection, now: float
    ) -> None:
        rows = connection.execute(
            "SELECT job_id, attempt_count FROM report_deliveries "
            "WHERE status = 'SENDING' "
            "AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?",
            (now,),
        ).fetchall()
        for row in rows:
            attempt = int(row["attempt_count"]) + 1
            if attempt >= self._max_delivery_attempts:
                connection.execute(
                    "UPDATE report_deliveries SET status = 'PENDING', "
                    "claim_token = NULL, "
                    "claim_expires_at = NULL, attempt_count = ?, failure_reason = ? "
                    "WHERE job_id = ?",
                    (attempt, "retry_limit:lease_expired", row["job_id"]),
                )
                self._event(
                    connection,
                    row["job_id"],
                    "report_delivery_failed",
                    {"attempt_count": attempt},
                )
                continue
            retry_at = now + self._retry_delay(attempt)
            connection.execute(
                "UPDATE report_deliveries SET status = 'PENDING', claim_token = NULL, "
                "claim_expires_at = NULL, attempt_count = ?, available_at = ? "
                "WHERE job_id = ?",
                (attempt, retry_at, row["job_id"]),
            )
            self._event(
                connection,
                row["job_id"],
                "report_delivery_lease_expired",
                {"attempt_count": attempt, "retry_at": retry_at},
            )

    @staticmethod
    def _owned(
        connection: sqlite3.Connection, owner_email: str, investigation_id: str
    ) -> sqlite3.Row:
        row = cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT * FROM replay_jobs WHERE id = ? AND owner = ?",
                (investigation_id, owner_email),
            ).fetchone(),
        )
        if row is None:
            raise ControlPlaneNotFoundError("investigation not found")
        return row

    @staticmethod
    def _validate_report_reference(investigation_id: str, report_artifact: str) -> Path:
        return validate_report_reference(investigation_id, report_artifact)

    def _artifact_path(self, investigation_id: str, report_artifact: str) -> Path:
        return artifact_path(self._artifacts_root, investigation_id, report_artifact)

    def _read_report_snapshot(self, investigation_id: str, report_artifact: str) -> str:
        return read_report_snapshot(
            self._artifacts_root, investigation_id, report_artifact
        )

    @property
    def claim_lease_seconds(self) -> float:
        return self._claim_lease_seconds

    def renew_running(self, investigation_id: str, claim_token: str) -> None:
        """Extend the active worker's lease without changing job state."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            updated = connection.execute(
                "UPDATE replay_jobs SET claim_expires_at = ? WHERE id = ? "
                "AND status = ? AND claim_token = ? AND claim_expires_at > ?",
                (
                    now + self._claim_lease_seconds,
                    investigation_id,
                    InvestigationStatus.RUNNING.value,
                    claim_token,
                    now,
                ),
            ).rowcount
            if updated != 1:
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )

    def reserve_incident(self, investigation_id: str, claim_token: str) -> str:
        """Durably allocate a scenario identity before provisioning the lab.

        The job id is opaque and alphanumeric, so it is a stable incident id
        without introducing a second allocation that could be lost between a
        database commit and ``start_scenario``.
        """
        incident_id = f"scenario{investigation_id}"
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            row = connection.execute(
                "SELECT status, claim_token, claim_expires_at, incident_id "
                "FROM replay_jobs WHERE id = ?",
                (investigation_id,),
            ).fetchone()
            if row is None:
                raise ControlPlaneNotFoundError("investigation not found")
            if (
                row["status"] != InvestigationStatus.RUNNING.value
                or row["claim_token"] != claim_token
                or row["claim_expires_at"] is None
                or row["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )
            stored_incident_id = row["incident_id"]
            if stored_incident_id is None:
                connection.execute(
                    "UPDATE replay_jobs SET incident_id = ? WHERE id = ?",
                    (incident_id, investigation_id),
                )
                self._event(
                    connection,
                    investigation_id,
                    "scenario_reserved",
                    {"incident_id": incident_id},
                )
                return incident_id
            if (
                not isinstance(stored_incident_id, str)
                or not stored_incident_id.isalnum()
            ):
                raise ControlPlaneIntegrityError("stored incident id is invalid")
            return stored_incident_id

    def scenario_status(self, incident_id: str) -> InvestigationStatus | None:
        """Return the owner-hidden job state associated with an active marker."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT status, failure_reason FROM replay_jobs WHERE incident_id = ?",
                (incident_id,),
            ).fetchone()
        if row is None:
            return None
        if row["failure_reason"] is not None:
            return InvestigationStatus.FAILED_SAFE
        try:
            return InvestigationStatus(row["status"])
        except ValueError as error:
            raise ControlPlaneIntegrityError(
                "invalid stored investigation status"
            ) from error

    def incident_id_for(self, investigation_id: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT incident_id FROM replay_jobs WHERE id = ?", (investigation_id,)
            ).fetchone()
        if row is None:
            raise ControlPlaneNotFoundError("investigation not found")
        incident_id = row["incident_id"]
        if incident_id is None:
            return None
        if not isinstance(incident_id, str) or not incident_id.isalnum():
            raise ControlPlaneIntegrityError("stored incident id is invalid")
        return incident_id

    def create(
        self,
        owner_email: str,
        request: CreateInvestigationRequest,
        idempotency_key: str,
    ) -> InvestigationView:
        """A repeated `(owner_email, idempotency_key)` pair replays the
        prior `202` -- the existing investigation's CURRENT view, not a
        second investigation -- rather than raising on the unique index
        above. The seed is always `ReplaySeed.DEVELOPMENT`: the server
        chooses it, `CreateInvestigationRequest` has no seed field a caller
        could set."""
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM replay_jobs WHERE owner = ? AND idempotency_key = ?",
                (owner_email, idempotency_key),
            ).fetchone()
            if existing is not None:
                return self._view(existing)
            investigation_id = uuid4().hex
            connection.execute(
                "INSERT INTO replay_jobs "
                "(id, owner, status, scenario_family, seed, idempotency_key) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    investigation_id,
                    owner_email,
                    InvestigationStatus.QUEUED.value,
                    request.scenario_family.value,
                    ReplaySeed.DEVELOPMENT.value,
                    idempotency_key,
                ),
            )
            self._event(
                connection,
                investigation_id,
                "investigation_queued",
                {"replay_only": True},
            )
        return InvestigationView(
            investigation_id=investigation_id, status=InvestigationStatus.QUEUED
        )

    def status(self, owner_email: str, investigation_id: str) -> InvestigationView:
        with self._connection() as connection:
            return self._view(self._owned(connection, owner_email, investigation_id))

    def events(self, owner_email: str, investigation_id: str) -> list[TimelineEvent]:
        with self._connection() as connection:
            self._owned(connection, owner_email, investigation_id)
            rows = connection.execute(
                "SELECT ordinal, name, fields FROM replay_events "
                "WHERE job_id = ? ORDER BY ordinal",
                (investigation_id,),
            ).fetchall()
        events: list[TimelineEvent] = []
        for row in rows:
            try:
                fields = json.loads(row["fields"])
            except json.JSONDecodeError as error:
                raise ControlPlaneIntegrityError(
                    "invalid stored event fields"
                ) from error
            if not isinstance(fields, dict):
                raise ControlPlaneIntegrityError(
                    "stored event fields are not an object"
                )
            events.append(
                TimelineEvent(ordinal=row["ordinal"], name=row["name"], fields=fields)
            )
        return events

    def decide(
        self, owner_email: str, investigation_id: str, decision: DecisionRequest
    ) -> InvestigationView:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._owned(connection, owner_email, investigation_id)
            prior = connection.execute(
                "SELECT checkpoint_id, decision, rejection_note FROM replay_decisions "
                "WHERE job_id = ?",
                (investigation_id,),
            ).fetchone()
            if prior is not None:
                mirrored_decision = self._decision_from_row(prior)
                checkpoint_id = job["checkpoint_id"]
                if not isinstance(checkpoint_id, str) or not checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "decision record has no paused checkpoint"
                    )
                if prior["checkpoint_id"] != checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "decision record belongs to a different checkpoint"
                    )
                if mirrored_decision != decision:
                    raise ControlPlaneConflictError(
                        "a different owner decision is recorded"
                    )
                # Migrate an older control-plane-only decision into the
                # authoritative checkpoint ledger before treating a retry as
                # idempotent.
                self._ensure_ledger_decision(
                    investigation_id, checkpoint_id, mirrored_decision
                )
                return self._view(job)
            if job["status"] != InvestigationStatus.PAUSED_APPROVAL.value:
                raise ControlPlaneConflictError(
                    "investigation is not waiting for a decision"
                )
            checkpoint_id = job["checkpoint_id"]
            if not checkpoint_id:
                raise ControlPlaneIntegrityError(
                    "paused investigation has no checkpoint"
                )
            # This write intentionally precedes the control-plane queue
            # transition. ``_reconcile_durable_decisions`` recovers the
            # small cross-database crash window that follows it.
            self._ensure_ledger_decision(investigation_id, checkpoint_id, decision)
            connection.execute(
                "INSERT INTO replay_decisions "
                "(job_id, checkpoint_id, decision, rejection_note) VALUES (?, ?, ?, ?)",
                (
                    investigation_id,
                    checkpoint_id,
                    decision.decision,
                    decision.rejection_note,
                ),
            )
            connection.execute(
                "UPDATE replay_jobs SET status = ? WHERE id = ?",
                (InvestigationStatus.QUEUED.value, investigation_id),
            )
            self._event(
                connection,
                investigation_id,
                "owner_decision_recorded",
                {"checkpoint_id": checkpoint_id, "decision": decision.decision},
            )
            return InvestigationView(
                investigation_id=investigation_id, status=InvestigationStatus.QUEUED
            )

    def report(self, owner_email: str, investigation_id: str) -> str:
        with self._connection() as connection:
            job = self._owned(connection, owner_email, investigation_id)
        if job["status"] != InvestigationStatus.COMPLETED.value:
            raise ControlPlaneConflictError("report is not finalized")
        stored_artifact = job["report_artifact"]
        if not isinstance(stored_artifact, str) or not stored_artifact:
            raise ControlPlaneIntegrityError(
                "finalized investigation has no report artifact"
            )
        stored_content = job["report_content"]
        stored_sha256 = job["report_sha256"]
        if not isinstance(stored_content, str) or not isinstance(stored_sha256, str):
            raise ControlPlaneIntegrityError(
                "finalized investigation has no report snapshot"
            )
        try:
            self._validate_report_reference(investigation_id, stored_artifact)
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

    def claim_next(self) -> WorkerClaim | None:
        """Atomically claim one queued replay job for an injected worker."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            self._reclaim_expired_investigations(connection, now)
            self._reconcile_durable_decisions(connection)
            # The synthetic lab exposes one mutable active scenario. This
            # database guard applies across Uvicorn processes, not merely
            # threads in one process, so two workers cannot contaminate each
            # other's telemetry while processing different jobs.
            running = connection.execute(
                "SELECT 1 FROM replay_jobs WHERE status = ? LIMIT 1",
                (InvestigationStatus.RUNNING.value,),
            ).fetchone()
            if running is not None:
                return None
            # A retryable job that already reserved an incident retains the
            # sole lab scenario during backoff. Starting a different job in
            # that interval would make its telemetry observe the wrong fault.
            deferred_scenario = connection.execute(
                "SELECT 1 FROM replay_jobs WHERE status = ? "
                "AND failure_reason IS NULL AND incident_id IS NOT NULL "
                "AND available_at > ? LIMIT 1",
                (InvestigationStatus.QUEUED.value, now),
            ).fetchone()
            if deferred_scenario is not None:
                return None
            row = connection.execute(
                "SELECT * FROM replay_jobs WHERE status = ? "
                "AND failure_reason IS NULL AND available_at <= ? "
                "ORDER BY CASE WHEN incident_id IS NULL THEN 1 ELSE 0 END, "
                "available_at, rowid LIMIT 1",
                (InvestigationStatus.QUEUED.value, now),
            ).fetchone()
            if row is None:
                return None
            claim_token = uuid4().hex
            connection.execute(
                "UPDATE replay_jobs SET status = ?, claim_token = ?, "
                "claim_expires_at = ? WHERE id = ?",
                (
                    InvestigationStatus.RUNNING.value,
                    claim_token,
                    now + self._claim_lease_seconds,
                    row["id"],
                ),
            )
            self._event(connection, row["id"], "investigation_started", {})
            decision_row = connection.execute(
                "SELECT checkpoint_id, decision, rejection_note FROM replay_decisions "
                "WHERE job_id = ?",
                (row["id"],),
            ).fetchone()
            try:
                owner_decision = (
                    None
                    if decision_row is None
                    else self._decision_from_row(decision_row)
                )
            except ValueError as error:
                raise ControlPlaneIntegrityError(
                    "stored owner decision is invalid"
                ) from error
            if owner_decision is not None:
                checkpoint_id = row["checkpoint_id"]
                if not isinstance(checkpoint_id, str) or not checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "resumed investigation has no checkpoint"
                    )
                if decision_row["checkpoint_id"] != checkpoint_id:
                    raise ControlPlaneIntegrityError(
                        "stored owner decision belongs to a different checkpoint"
                    )
                self._ensure_ledger_decision(row["id"], checkpoint_id, owner_decision)
            return WorkerClaim(
                investigation_id=row["id"],
                owner_email=row["owner"],
                scenario_family=row["scenario_family"],
                seed=row["seed"],
                checkpoint_id=row["checkpoint_id"],
                incident_id=row["incident_id"],
                owner_decision=owner_decision,
                claim_token=claim_token,
            )

    def mark_paused(
        self, investigation_id: str, checkpoint_id: str, claim_token: str
    ) -> None:
        """Persist the graph checkpoint before making a job resumable in the UI."""
        if not checkpoint_id.strip():
            raise ValueError("checkpoint_id must not be blank")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            row = connection.execute(
                "SELECT status, claim_token, claim_expires_at FROM replay_jobs "
                "WHERE id = ?",
                (investigation_id,),
            ).fetchone()
            if row is None:
                raise ControlPlaneNotFoundError("investigation not found")
            if (
                row["status"] != InvestigationStatus.RUNNING.value
                or row["claim_token"] != claim_token
                or row["claim_expires_at"] is None
                or row["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )
            connection.execute(
                "UPDATE replay_jobs SET status = ?, checkpoint_id = ?, "
                "claim_token = NULL, claim_expires_at = NULL WHERE id = ?",
                (
                    InvestigationStatus.PAUSED_APPROVAL.value,
                    checkpoint_id,
                    investigation_id,
                ),
            )
            self._event(
                connection,
                investigation_id,
                "owner_decision_requested",
                {"checkpoint_id": checkpoint_id},
            )

    def requeue_running(self, investigation_id: str, claim_token: str) -> None:
        """Release a crashed worker claim without dropping its durable state."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE replay_jobs SET status = ?, claim_token = NULL, "
                "claim_expires_at = NULL WHERE id = ? AND status = ? "
                "AND claim_token = ?",
                (
                    InvestigationStatus.QUEUED.value,
                    investigation_id,
                    InvestigationStatus.RUNNING.value,
                    claim_token,
                ),
            ).rowcount
            if updated != 1:
                raise ControlPlaneConflictError(
                    "only a running investigation may be requeued"
                )
            self._event(connection, investigation_id, "investigation_requeued", {})

    def retry_running(
        self, investigation_id: str, claim_token: str, error: Exception
    ) -> bool:
        """Schedule a bounded retry and return whether the job remains retryable."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, claim_token, attempt_count FROM replay_jobs "
                "WHERE id = ?",
                (investigation_id,),
            ).fetchone()
            if row is None:
                raise ControlPlaneNotFoundError("investigation not found")
            if (
                row["status"] != InvestigationStatus.RUNNING.value
                or row["claim_token"] != claim_token
            ):
                raise ControlPlaneConflictError(
                    "only a running investigation may be retried"
                )
            attempt = int(row["attempt_count"]) + 1
            if attempt >= self._max_replay_attempts:
                connection.execute(
                    "UPDATE replay_jobs SET status = ?, claim_token = NULL, "
                    "claim_expires_at = NULL, attempt_count = ?, failure_reason = ? "
                    "WHERE id = ?",
                    (
                        InvestigationStatus.QUEUED.value,
                        attempt,
                        f"retry_limit:{type(error).__name__}",
                        investigation_id,
                    ),
                )
                self._event(
                    connection,
                    investigation_id,
                    "investigation_failed",
                    {"attempt_count": attempt},
                )
                return False
            retry_at = self._clock() + self._retry_delay(attempt)
            connection.execute(
                "UPDATE replay_jobs SET status = ?, claim_token = NULL, "
                "claim_expires_at = NULL, attempt_count = ?, available_at = ? "
                "WHERE id = ?",
                (
                    InvestigationStatus.QUEUED.value,
                    attempt,
                    retry_at,
                    investigation_id,
                ),
            )
            self._event(
                connection,
                investigation_id,
                "investigation_retry_scheduled",
                {"attempt_count": attempt, "retry_at": retry_at},
            )
            return True

    def _retry_delay(self, attempt: int) -> float:
        return float(min(self._retry_base_seconds * (2 ** (attempt - 1)), 3600.0))

    def finalize(
        self, investigation_id: str, report_artifact: str, claim_token: str
    ) -> None:
        """Record an already write-once artifact and enqueue owner-only delivery."""
        if not report_artifact.strip():
            raise ValueError("report_artifact must not be blank")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            row = connection.execute(
                "SELECT owner, status, claim_token, claim_expires_at FROM replay_jobs "
                "WHERE id = ?",
                (investigation_id,),
            ).fetchone()
            if row is None:
                raise ControlPlaneNotFoundError("investigation not found")
            if row["status"] == InvestigationStatus.COMPLETED.value:
                raise ControlPlaneConflictError("investigation is already finalized")
            if (
                row["status"] != InvestigationStatus.RUNNING.value
                or row["claim_token"] != claim_token
                or row["claim_expires_at"] is None
                or row["claim_expires_at"] <= now
            ):
                raise ControlPlaneConflictError(
                    "investigation claim is no longer current"
                )
            report_content = self._read_report_snapshot(
                investigation_id, report_artifact
            )
            report_sha256 = hashlib.sha256(report_content.encode("utf-8")).hexdigest()
            connection.execute(
                "UPDATE replay_jobs SET status = ?, report_artifact = ?, "
                "report_content = ?, report_sha256 = ?, claim_token = NULL, "
                "claim_expires_at = NULL WHERE id = ?",
                (
                    InvestigationStatus.COMPLETED.value,
                    report_artifact,
                    report_content,
                    report_sha256,
                    investigation_id,
                ),
            )
            connection.execute(
                "INSERT INTO report_deliveries (job_id, destination_email, status) "
                "VALUES (?, ?, 'PENDING')",
                (investigation_id, row["owner"]),
            )
            self._event(connection, investigation_id, "report_finalized", {})

    def claim_delivery(self) -> DeliveryClaim | None:
        """Claim delivery after finalization; retry policy belongs to the sender."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            self._reclaim_expired_deliveries(connection, now)
            row = connection.execute(
                "SELECT j.id, j.status, j.report_artifact, j.report_content, "
                "j.report_sha256, d.destination_email "
                "FROM report_deliveries d JOIN replay_jobs j ON j.id = d.job_id "
                "WHERE d.status = 'PENDING' AND d.failure_reason IS NULL "
                "AND d.available_at <= ? ORDER BY d.available_at, j.rowid LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            claim_token = uuid4().hex
            connection.execute(
                "UPDATE report_deliveries SET status = 'SENDING', claim_token = ?, "
                "claim_expires_at = ? WHERE job_id = ?",
                (claim_token, now + self._claim_lease_seconds, row["id"]),
            )
            return DeliveryClaim(
                investigation_id=row["id"],
                status=InvestigationStatus(row["status"]),
                report_artifact=row["report_artifact"],
                report_content=row["report_content"],
                report_sha256=row["report_sha256"],
                destination_email=row["destination_email"],
                delivery_id=row["id"],
                claim_token=claim_token,
            )

    def mark_delivered(self, investigation_id: str, claim_token: str) -> None:
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE report_deliveries SET status = 'SENT', claim_token = NULL, "
                "claim_expires_at = NULL WHERE job_id = ? AND status = 'SENDING' "
                "AND claim_token = ? AND claim_expires_at > ?",
                (investigation_id, claim_token, self._clock()),
            ).rowcount
            if updated != 1:
                raise ControlPlaneConflictError("delivery is not currently claimed")
            self._event(connection, investigation_id, "report_delivered", {})

    def renew_delivery(self, investigation_id: str, claim_token: str) -> None:
        """Extend an active delivery lease while its sender is still running."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            updated = connection.execute(
                "UPDATE report_deliveries SET claim_expires_at = ? "
                "WHERE job_id = ? AND status = 'SENDING' "
                "AND claim_token = ? AND claim_expires_at > ?",
                (
                    now + self._claim_lease_seconds,
                    investigation_id,
                    claim_token,
                    now,
                ),
            ).rowcount
            if updated != 1:
                raise ControlPlaneConflictError("delivery is not currently claimed")

    def release_delivery(self, investigation_id: str, claim_token: str) -> None:
        """Return a failed delivery claim to the outbox for an explicit retry."""
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE report_deliveries SET status = 'PENDING', claim_token = NULL, "
                "claim_expires_at = NULL WHERE job_id = ? AND status = 'SENDING' "
                "AND claim_token = ?",
                (investigation_id, claim_token),
            ).rowcount
            if updated != 1:
                raise ControlPlaneConflictError("delivery is not currently claimed")
            self._event(connection, investigation_id, "report_delivery_requeued", {})

    def retry_delivery(
        self, investigation_id: str, claim_token: str, error: Exception
    ) -> bool:
        """Schedule a bounded delivery retry without blocking other reports."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, claim_token, attempt_count FROM report_deliveries "
                "WHERE job_id = ?",
                (investigation_id,),
            ).fetchone()
            if row is None:
                raise ControlPlaneNotFoundError("report delivery not found")
            if row["status"] != "SENDING" or row["claim_token"] != claim_token:
                raise ControlPlaneConflictError("delivery is not currently claimed")
            attempt = int(row["attempt_count"]) + 1
            if attempt >= self._max_delivery_attempts:
                connection.execute(
                    "UPDATE report_deliveries SET status = 'PENDING', "
                    "claim_token = NULL, "
                    "claim_expires_at = NULL, attempt_count = ?, failure_reason = ? "
                    "WHERE job_id = ?",
                    (
                        attempt,
                        f"retry_limit:{type(error).__name__}",
                        investigation_id,
                    ),
                )
                self._event(
                    connection,
                    investigation_id,
                    "report_delivery_failed",
                    {"attempt_count": attempt},
                )
                return False
            retry_at = self._clock() + self._retry_delay(attempt)
            connection.execute(
                "UPDATE report_deliveries SET status = 'PENDING', claim_token = NULL, "
                "claim_expires_at = NULL, attempt_count = ?, available_at = ? "
                "WHERE job_id = ?",
                (attempt, retry_at, investigation_id),
            )
            self._event(
                connection,
                investigation_id,
                "report_delivery_retry_scheduled",
                {"attempt_count": attempt, "retry_at": retry_at},
            )
            return True


class ReplayWorker:
    """Runs one claimed replay job without knowing any provider implementation."""

    def __init__(
        self,
        control_plane: WorkerControlPlane,
        runner: ReplayJobRunner,
        *,
        reconcile: Callable[[], None] | None = None,
        on_paused: Callable[[WorkerClaim], None] | None = None,
        on_finalized: Callable[[WorkerClaim], None] | None = None,
        on_failed: Callable[[WorkerClaim], None] | None = None,
    ) -> None:
        self._control_plane = control_plane
        self._runner = runner
        self._reconcile = reconcile
        self._on_paused = on_paused
        self._on_finalized = on_finalized
        self._on_failed = on_failed

    def run_once(self) -> bool:
        if self._reconcile is not None:
            self._reconcile()
        claim = self._control_plane.claim_next()
        if claim is None:
            return False
        stop_heartbeat = Event()
        heartbeat_failures: list[Exception] = []

        def renew_lease() -> None:
            interval = max(
                self._control_plane.claim_lease_seconds / 3,
                MINIMUM_HEARTBEAT_SECONDS,
            )
            while not stop_heartbeat.wait(interval):
                try:
                    self._control_plane.renew_running(
                        claim.investigation_id, claim.claim_token
                    )
                except Exception as error:
                    heartbeat_failures.append(error)
                    stop_heartbeat.set()
                    return

        heartbeat = Thread(target=renew_lease, daemon=True)
        heartbeat.start()
        try:
            outcome = self._runner.run(claim)
            if heartbeat_failures:
                raise heartbeat_failures[0]
            if isinstance(outcome, PausedWorkerOutcome):
                self._control_plane.mark_paused(
                    claim.investigation_id, outcome.checkpoint_id, claim.claim_token
                )
                if self._on_paused is not None:
                    self._on_paused(claim)
            else:
                self._control_plane.finalize(
                    claim.investigation_id, outcome.report_artifact, claim.claim_token
                )
                if self._on_finalized is not None:
                    self._on_finalized(claim)
        except Exception as error:
            try:
                retryable = self._control_plane.retry_running(
                    claim.investigation_id, claim.claim_token, error
                )
                if not retryable and self._on_failed is not None:
                    self._on_failed(claim)
            except ControlPlaneConflictError:
                # A concurrent lease reclaim won the race. Its queued claim is
                # already recoverable, so do not mask the worker's real error.
                pass
            raise
        finally:
            stop_heartbeat.set()
            heartbeat.join()
        return True


class ReportDeliveryWorker:
    """Delivers one owner-only report and releases failures for a safe retry."""

    def __init__(
        self, control_plane: WorkerControlPlane, sender: ReportDeliverySender
    ) -> None:
        self._control_plane = control_plane
        self._sender = sender

    def run_once(self) -> bool:
        claim = self._control_plane.claim_delivery()
        if claim is None:
            return False
        stop_heartbeat = Event()
        heartbeat_failures: list[Exception] = []

        def renew_lease() -> None:
            interval = max(
                self._control_plane.claim_lease_seconds / 3,
                MINIMUM_HEARTBEAT_SECONDS,
            )
            while not stop_heartbeat.wait(interval):
                try:
                    self._control_plane.renew_delivery(
                        claim.investigation_id, claim.claim_token
                    )
                except Exception as error:
                    heartbeat_failures.append(error)
                    stop_heartbeat.set()
                    return

        heartbeat = Thread(target=renew_lease, daemon=True)
        heartbeat.start()
        try:
            self._sender.send(claim)
            if heartbeat_failures:
                raise heartbeat_failures[0]
            self._control_plane.mark_delivered(
                claim.investigation_id, claim.claim_token
            )
        except Exception as error:
            try:
                self._control_plane.retry_delivery(
                    claim.investigation_id, claim.claim_token, error
                )
            except ControlPlaneConflictError:
                # A reclaimed delivery belongs to another sender now. Its
                # lease recovery will decide the next durable transition.
                pass
            raise
        finally:
            stop_heartbeat.set()
            heartbeat.join()
        return True


class ReplayGraphJobRunner:
    """Concrete VM runner, wired to whichever `ReplayRuntimeWiring` `app()`
    constructed it with.

    The HTTP API itself never accepts a model selector --
    `CreateInvestigationRequest` has no such field, on any deployment. Which
    wiring this runner holds (scripted replay, or, since
    `CAUSALOPS_HOSTED_LIVE_MODEL`, a genuinely live `LiveClaudeModel`) is an
    operator's deployment-time choice via environment variable, resolved
    once in `app()` and fixed for this process's whole lifetime -- never a
    per-request or per-investigation choice. It is invoked only by the
    background worker, never during import or HTTP request handling.
    """

    def __init__(
        self,
        root: Path,
        control_plane: WorkerControlPlane,
        replay_wiring: ReplayRuntimeWiring,
        artifact_store: ArtifactStore | None = None,
        checkpointer_factory: CheckpointerFactory | None = None,
    ) -> None:
        self._root = root
        self._control_plane = control_plane
        self._replay_wiring = replay_wiring
        self._artifact_store = artifact_store
        self._checkpointer_factory: CheckpointerFactory = checkpointer_factory or (
            lambda: _sqlite_checkpointer(ProjectPaths(root=root).checkpoints_db)
        )

    def _upload_finalized_artifacts(self, investigation_id: str) -> None:
        """Best-available durable copy: no-op when no bucket is configured
        (`artifact_store is None`, the default -- unchanged behavior for
        every existing deployment/test). When configured, reads the 5
        artifacts back from the local directory `finalize_investigation`
        already wrote (or a prior crashed attempt already wrote -- see
        `_has_finalized_artifact`), so this is safe to call from either of
        `run()`'s two `FinalizedWorkerOutcome` paths, and safe to retry:
        `GcsArtifactStore` itself treats a repeat upload of an
        already-durable artifact as success, not an error."""
        if self._artifact_store is None:
            return
        directory = self._root / "results" / "investigations" / investigation_id
        artifacts = {
            name: (directory / name).read_text(encoding="utf-8")
            for name in ARTIFACT_NAMES
        }
        self._artifact_store.write_investigation_artifacts(investigation_id, artifacts)

    def _has_finalized_artifact(self, investigation_id: str) -> bool:
        """Recognize only the exact regular report a crashed worker may adopt."""
        directory = self._root / "results" / "investigations" / investigation_id
        report = directory / "report.md"
        try:
            return stat.S_ISDIR(directory.lstat().st_mode) and stat.S_ISREG(
                report.lstat().st_mode
            )
        except OSError:
            return False

    def reconcile_scenarios(self) -> None:
        """Finish marker cleanup left after a process died between transitions."""
        marker = active_incident_file(self._root)
        try:
            incident_id = marker.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return
        if not incident_id.isalnum():
            raise ControlPlaneIntegrityError("active scenario marker is invalid")
        scenario_status = self._control_plane.scenario_status(incident_id)
        if scenario_status is None:
            # No job carries this incident_id at all -- not "still in
            # progress" (that's QUEUED/RUNNING, left alone below), but a
            # genuinely orphaned marker (e.g. its job document was deleted
            # after `reserve_incident` already wrote this file, or a crash
            # landed between those two writes). Found live: with no branch
            # for this case, the marker survived forever and
            # `SCENARIO_ALREADY_ACTIVE` failed every future scenario start
            # until manually released.
            release_scenario(self._root, incident_id)
        elif scenario_status is InvestigationStatus.PAUSED_APPROVAL:
            release_scenario(self._root, incident_id)
        elif scenario_status in {
            InvestigationStatus.FAILED_SAFE,
            InvestigationStatus.COMPLETED,
        }:
            try:
                reset_scenario(self._root, incident_id)
            except (LabError, OSError):
                # The result is already durable. If a manual cleanup removed
                # the run first, releasing a matching marker is sufficient.
                release_scenario(self._root, incident_id)

    def release_paused_scenario(self, claim: WorkerClaim) -> None:
        incident_id = self._control_plane.incident_id_for(claim.investigation_id)
        if incident_id is not None:
            release_scenario(self._root, incident_id)

    def release_finalized_scenario(self, claim: WorkerClaim) -> None:
        incident_id = self._control_plane.incident_id_for(claim.investigation_id)
        if incident_id is None:
            return
        try:
            reset_scenario(self._root, incident_id)
        except (LabError, OSError):
            release_scenario(self._root, incident_id)

    def release_failed_scenario(self, claim: WorkerClaim) -> None:
        self.release_finalized_scenario(claim)

    def run(self, claim: WorkerClaim) -> WorkerOutcome:
        if self._has_finalized_artifact(claim.investigation_id):
            # ``finalize_investigation`` atomically renamed the complete
            # directory before a prior process died. Let the control plane
            # snapshot and publish it instead of running the graph again.
            self._upload_finalized_artifacts(claim.investigation_id)
            return FinalizedWorkerOutcome(
                report_artifact=f"{claim.investigation_id}/report.md"
            )
        incident_id = claim.incident_id
        if incident_id is None:
            incident_id = self._control_plane.reserve_incident(
                claim.investigation_id, claim.claim_token
            )
        paths = validated_run_paths(self._root, incident_id)
        if not paths.incident_file.is_file():
            if paths.root.exists():
                # A hard crash can leave a marker and a partial run before
                # incident.json is durable. It is safe to remove only this
                # reserved id and create it again with the same identity.
                reset_scenario(self._root, incident_id)
            start_scenario(
                self._root,
                claim.scenario_family.value,
                claim.seed.value,
                incident_id=incident_id,
            )
        paths, incident = _load_verified_incident(self._root, incident_id)
        budgets = Budgets()
        recorder = RunRecorder(utc_now)
        model, registry, model_name, release = self._replay_wiring.build(
            incident, paths, budgets, family=claim.scenario_family
        )
        try:
            with self._checkpointer_factory() as checkpointer:
                if claim.checkpoint_id is None:
                    # A durable graph checkpoint can outlive the control-plane
                    # transition that was supposed to name it. Recover it before
                    # supplying a new initial state, otherwise a reclaimed pause
                    # (or a completed graph before artifact finalization) would
                    # be executed a second time.
                    result = recover_graph_investigation(
                        claim.investigation_id,
                        checkpointer,
                        incident.scope,
                        incident.packet,
                        model,
                        registry,
                        recorder,
                        budgets,
                        utc_now,
                    )
                    if result is None:
                        result = run_graph_investigation(
                            incident.scope,
                            incident.packet,
                            incident.evidence,
                            model,
                            registry,
                            recorder,
                            budgets,
                            utc_now,
                            investigation_id=claim.investigation_id,
                            checkpointer=checkpointer,
                            model_name=model_name,
                        )
                else:
                    if claim.owner_decision is None:
                        raise ControlPlaneIntegrityError(
                            "resumed investigation has no owner decision"
                        )
                    result = resume_graph_investigation(
                        claim.investigation_id,
                        checkpointer,
                        incident.scope,
                        incident.packet,
                        model,
                        registry,
                        recorder,
                        claim.owner_decision.decision,
                        claim.owner_decision.rejection_note,
                        budgets,
                        utc_now,
                    )
            if isinstance(result, EscalatedInvestigation):
                return PausedWorkerOutcome(checkpoint_id=result.checkpoint_id)
            finalize_investigation(
                self._root / "results",
                result.report,
                recorder.events,
                result.evidence,
                result.receipts,
                render_markdown_report(
                    result.report, result.evidence, result.receipts, model_name
                ),
            )
            self._upload_finalized_artifacts(claim.investigation_id)
            return FinalizedWorkerOutcome(
                report_artifact=f"{claim.investigation_id}/report.md"
            )
        finally:
            release()


class FilesystemReportDeliverySender:
    """Persist an owner-addressed, idempotent delivery hand-off for an MTA."""

    def __init__(self, delivery_root: Path) -> None:
        self._delivery_root = delivery_root

    def send(self, claim: DeliveryClaim) -> None:
        self._delivery_root.mkdir(parents=True, exist_ok=True)
        target = self._delivery_root / f"{claim.delivery_id}.json"
        payload = json.dumps(
            {
                "destination_email": claim.destination_email,
                "report_content": claim.report_content,
                "report_sha256": claim.report_sha256,
            },
            sort_keys=True,
        )
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as delivery_file:
                delivery_file.write(payload)
                delivery_file.flush()
                os.fsync(delivery_file.fileno())
            os.link(temporary, target)
        except FileExistsError as error:
            if target.read_text(encoding="utf-8") != payload:
                raise ControlPlaneIntegrityError(
                    "delivery idempotency record conflicts"
                ) from error
        finally:
            temporary.unlink(missing_ok=True)


class BackgroundControlPlaneWorkers:
    """Run replay and delivery workers continuously for the deployed app."""

    def __init__(
        self,
        replay_worker: ReplayWorker,
        delivery_worker: ReportDeliveryWorker,
        *,
        poll_seconds: float = 0.25,
    ) -> None:
        self._replay_worker = replay_worker
        self._delivery_worker = delivery_worker
        self._poll_seconds = poll_seconds
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._poll_seconds * 4))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                progressed = self._replay_worker.run_once()
                progressed = self._delivery_worker.run_once() or progressed
            except Exception:
                # Claims are explicitly requeued by their worker; keep the
                # service alive for the next independent durable work item.
                progressed = False
            if not progressed:
                self._stop.wait(self._poll_seconds)


class NoOpWorkerService:
    """`app()`'s Cloud Run deployment must NOT run a `BackgroundControl
    PlaneWorkers` loop of its own -- it has no lab/docker access at all
    (`infra/gcp/cloud_run.tf`'s own scoping notes), so a job its own worker won the
    claim race for would fail immediately with no lab to reach. Found live
    this session as a real design gap once the Cloud Run split shipped:
    `app()` built a real worker unconditionally regardless of deployment
    target. `CAUSALOPS_RUN_WORKER=false` swaps this in instead -- the VM
    keeps running the real one."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def app() -> FastAPI:
    """Uvicorn factory with a real Google verifier and explicit configuration."""
    import os

    owners = frozenset(
        owner.strip()
        for owner in os.environ.get("CAUSALOPS_ALLOWED_OWNERS", "").split(",")
        if owner.strip()
    )
    google_client_id = os.environ.get("CAUSALOPS_GOOGLE_CLIENT_ID", "").strip()
    if not owners:
        raise RuntimeError("CAUSALOPS_ALLOWED_OWNERS must name at least one owner")
    if not google_client_id:
        raise RuntimeError("CAUSALOPS_GOOGLE_CLIENT_ID must be configured")
    root = Path(os.environ.get("CAUSALOPS_PROJECT_ROOT", Path.cwd())).resolve()
    if find_project_root(root) != root:
        raise RuntimeError("CAUSALOPS_PROJECT_ROOT must name the project root")
    # Phase C (Firestore migration): "sqlite" (the default) keeps every
    # existing deployment's behavior unchanged. "firestore" opts into the
    # provisioned `google_firestore_database.default`
    # (`infra/gcp/firestore.tf`) instead -- an explicit opt-in, not yet the
    # default, until a live VM run against it is validated the same way
    # Phase A's MCP-default flip needed both an equivalence test and a
    # live comparison run before becoming the default.
    control_plane_backend = (
        os.environ.get("CAUSALOPS_CONTROL_PLANE_BACKEND", "sqlite").strip().lower()
    )
    if control_plane_backend not in {"sqlite", "firestore"}:
        raise RuntimeError(
            "CAUSALOPS_CONTROL_PLANE_BACKEND must be 'sqlite' or 'firestore'"
        )
    control_plane: WorkerControlPlane
    checkpointer_factory: CheckpointerFactory | None
    if control_plane_backend == "firestore":
        control_plane = FirestoreReplayControlPlane(root / "results" / "investigations")
        checkpointer_factory = lambda: nullcontext(FirestoreCheckpointSaver())  # noqa: E731
    else:
        database = Path(
            os.environ.get("CAUSALOPS_CONTROL_PLANE_DB", "results/control-plane.db")
        )
        if not database.is_absolute():
            database = root / database
        control_plane = SqliteReplayControlPlane(
            database,
            artifacts_root=root / "results" / "investigations",
            checkpoint_database=ProjectPaths(root=root).checkpoints_db,
        )
        checkpointer_factory = None
    # MCP is the default hosted-worker dispatch path on the VM, per Phase
    # 3's exit criterion ("replay runs through worker/MCP") -- equivalence
    # with direct dispatch is proven in `test_mcp_policy_equivalence.py` and,
    # live, in a real `causalops-evaluate` run over the MCP transport.
    # `CAUSALOPS_MCP_DISPATCH` is now only an explicit escape hatch BACK to
    # direct dispatch (false/0/no), not an opt-in -- outside the VM
    # (`CAUSALOPS_EXECUTION_ENV` unset, e.g. local/test/CI) direct dispatch
    # stays the only choice, since MCP requires spawning a real subprocess.
    # Belt-and-suspenders gate: even with MCP dispatch selected here,
    # `mcp_policy_adapter._APPROVED_MCP_DISPATCH` being `None` still makes
    # every `McpBackedReplayRuntimeWiring.build()` call raise immediately
    # (see `policy_approved_mcp_server`) -- `ENABLE_CLAUDE=false` /
    # replay-only-by-default is never actually bypassed without the real,
    # reviewed approval record existing in source.
    mcp_dispatch_disabled = os.environ.get(
        "CAUSALOPS_MCP_DISPATCH", ""
    ).strip().lower() in {"0", "false", "no"}
    mcp_dispatch_requested = (
        os.environ.get(VM_EXECUTION_ENV_VARIABLE, "").strip().lower()
        == VM_EXECUTION_ENV
        and not mcp_dispatch_disabled
    )
    # Demo/ops-only override: every deployment that never sets this keeps
    # `REPLAY_FIXTURE` (`lab_diagnosis.json`), whose scripted final
    # assessment is always DIAGNOSED with no contrary evidence -- under
    # healthy conditions it can never trigger `PAUSED_APPROVAL` (confirmed
    # live this session: none of `_escalation_reason`'s 4 triggers can fire
    # from that exact script). Pointing this at a fixture that scripts
    # INSUFFICIENT_EVIDENCE with a tool budget still remaining (e.g.
    # `hosted_escalation_demo.json`) makes the pause/decide path reachable
    # on demand -- for a demo, not a permanent behavior change.
    replay_fixture_override = os.environ.get("CAUSALOPS_REPLAY_FIXTURE", "").strip()
    replay_fixture_kwargs = (
        {"fixture": Path(replay_fixture_override)} if replay_fixture_override else {}
    )
    # Operator-only deployment toggle, never a client-facing choice --
    # `CreateInvestigationRequest` still accepts no model field at all.
    # Read once here, at worker startup; `ReplayGraphJobRunner` then holds
    # exactly one `replay_wiring` instance for its whole process lifetime,
    # so every investigation this deployment runs -- fresh or resumed after
    # a pause -- uses whichever wiring this resolved to, with no
    # per-investigation choice to persist or get wrong on resume. Safe only
    # because `CAUSALOPS_ALLOWED_OWNERS` is expected to be a tight,
    # genuinely trusted allowlist on any deployment that sets this -- see
    # `docs/CLOUD_RUN_DEMO.md` and `infra/DEPLOYMENT.md`.
    hosted_live_model_enabled = os.environ.get(
        "CAUSALOPS_HOSTED_LIVE_MODEL", ""
    ).strip().lower() in {"1", "true", "yes"}
    replay_wiring: ReplayRuntimeWiring
    if hosted_live_model_enabled:
        # Live mode always uses the real MCP transport -- matching
        # `evaluate_cli.py`'s own posture, `build_claude_model_and_mcp_
        # registry` has no direct-dispatch alternative to offer, so this
        # branch ignores `mcp_dispatch_requested` entirely.
        replay_wiring = McpBackedLiveRuntimeWiring(
            ProjectPaths(root=root).checkpoints_db
        )
    elif mcp_dispatch_requested:
        replay_wiring = McpBackedReplayRuntimeWiring(**replay_fixture_kwargs)
    else:
        replay_wiring = HostedReplayRuntimeWiring(**replay_fixture_kwargs)
    # Optional: no bucket configured means no GCS upload, unchanged behavior
    # for every deployment that has not set this yet. `GcsArtifactStore`'s
    # own constructor resolves ADC (or impersonated credentials, when
    # `CAUSALOPS_ARTIFACT_SERVICE_ACCOUNT` is also set -- see
    # `infra/gcp/storage.tf`'s `vm_impersonates_control_plane` grant) at
    # this point, not at import time.
    artifact_bucket_name = os.environ.get("CAUSALOPS_ARTIFACT_BUCKET", "").strip()
    artifact_service_account = os.environ.get(
        "CAUSALOPS_ARTIFACT_SERVICE_ACCOUNT", ""
    ).strip()
    artifact_store = (
        GcsArtifactStore(
            artifact_bucket_name,
            target_service_account=artifact_service_account or None,
        )
        if artifact_bucket_name
        else None
    )
    runner = ReplayGraphJobRunner(
        root, control_plane, replay_wiring, artifact_store, checkpointer_factory
    )
    workers = BackgroundControlPlaneWorkers(
        ReplayWorker(
            control_plane,
            runner,
            reconcile=runner.reconcile_scenarios,
            on_paused=runner.release_paused_scenario,
            on_finalized=runner.release_finalized_scenario,
            on_failed=runner.release_failed_scenario,
        ),
        ReportDeliveryWorker(
            control_plane,
            FilesystemReportDeliverySender(root / "results" / "deliveries"),
        ),
    )
    # Cloud Run has no lab/docker access at all -- its own deployment must
    # set CAUSALOPS_RUN_WORKER=false so only the VM's `app()` process ever
    # starts the real background worker loop against the shared Firestore
    # control plane. Every existing (VM-only) deployment keeps running it,
    # unchanged, since this defaults to enabled.
    run_worker = os.environ.get("CAUSALOPS_RUN_WORKER", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    worker_service: ControlPlaneWorkerService = (
        workers if run_worker else NoOpWorkerService()
    )
    return create_app(
        control_plane,
        allowed_owners=owners,
        identity_verifier=GoogleIdentityVerifier(google_client_id),
        google_client_id=google_client_id,
        worker_service=worker_service,
    )
