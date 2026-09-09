"""Durable replay-only control-plane composition and worker hand-off.

This module persists control-plane facts, not graph state. A replay worker
claims a queued job, runs the already-injected replay graph, and uses the
checkpoint and finalization methods below to make API resume and delivery safe
across process restarts. It has no provider-selection input.
"""

import errno
import hashlib
import json
import os
import sqlite3
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Protocol, cast
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from causalops.api import (
    ControlPlaneConflictError,
    ControlPlaneNotFoundError,
    CreateInvestigationRequest,
    DecisionRequest,
    InvestigationStatus,
    InvestigationView,
    ReplayControlPlane,
    TimelineEvent,
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
from causalops.mcp_client_registry import McpBackedReplayRuntimeWiring
from causalops.report import render_report as render_markdown_report
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


class WorkerClaim(CreateInvestigationRequest):
    """The fixed replay inputs a worker receives after atomically claiming a job."""

    model_config = ConfigDict(frozen=True)

    investigation_id: str
    owner_email: str
    checkpoint_id: str | None
    incident_id: str | None = None
    owner_decision: DecisionRequest | None = None
    claim_token: str


class DeliveryClaim(InvestigationView):
    """A report delivery work item addressed only to the investigation owner."""

    model_config = ConfigDict(frozen=True)

    report_artifact: str
    report_content: str
    report_sha256: str
    destination_email: str
    delivery_id: str
    claim_token: str


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


class ReportDeliverySender(Protocol):
    """Injected sender for a finalized report delivery claim."""

    def send(self, claim: DeliveryClaim) -> None: ...


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
                        ('QUEUED', 'RUNNING', 'PAUSED', 'FINALIZED')),
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
            (InvestigationStatus.PAUSED.value,),
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
                investigation_id=row["id"], status=InvestigationStatus.FAILED
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
        relative_path = Path(report_artifact)
        expected_path = Path(investigation_id) / "report.md"
        if relative_path != expected_path:
            raise ValueError(
                "report_artifact must be the investigation's own report.md"
            )
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("report_artifact must stay beneath the artifact root")
        return relative_path

    def _artifact_path(self, investigation_id: str, report_artifact: str) -> Path:
        self._validate_report_reference(investigation_id, report_artifact)
        if self._artifacts_root.is_symlink():
            raise ValueError("artifact root must not be a symbolic link")
        root = self._artifacts_root.resolve()
        investigation_directory = root / investigation_id
        candidate = investigation_directory / "report.md"
        try:
            root_mode = root.lstat().st_mode
            directory_mode = investigation_directory.lstat().st_mode
            report_mode = candidate.lstat().st_mode
        except OSError as error:
            raise ValueError(
                "report_artifact must name an existing regular file"
            ) from error
        if not stat.S_ISDIR(root_mode):
            raise ValueError("artifact root must be a directory")
        if stat.S_ISLNK(directory_mode) or stat.S_ISLNK(report_mode):
            raise ValueError(
                "report_artifact and its directory must not be symbolic links"
            )
        if not stat.S_ISDIR(directory_mode) or not stat.S_ISREG(report_mode):
            raise ValueError("report_artifact must name an existing regular file")
        return candidate

    def _read_report_snapshot(self, investigation_id: str, report_artifact: str) -> str:
        """Read the report through descriptor-anchored, no-follow opens."""
        self._validate_report_reference(investigation_id, report_artifact)
        required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
        if any(not hasattr(os, flag) for flag in required_flags):
            raise ValueError("platform lacks safe no-follow artifact reads")
        # typeshed omits O_DIRECTORY/O_NOFOLLOW on win32 (they're POSIX-only),
        # so a static `os.O_DIRECTORY` attribute access fails mypy there even
        # though the hasattr guard above already keeps this branch
        # unreachable on that platform. getattr() sidesteps the platform
        # stub instead of needing a `type: ignore` that would be "unused"
        # on the POSIX runners where the attribute really does exist.
        o_directory: int = getattr(os, "O_DIRECTORY")  # noqa: B009
        o_nofollow: int = getattr(os, "O_NOFOLLOW")  # noqa: B009
        flags = os.O_RDONLY | o_directory | o_nofollow
        root_fd: int | None = None
        directory_fd: int | None = None
        report_fd: int | None = None
        try:
            root_fd = os.open(self._artifacts_root, flags)
            directory_fd = os.open(investigation_id, flags, dir_fd=root_fd)
            report_fd = os.open(
                "report.md",
                os.O_RDONLY | o_nofollow,
                dir_fd=directory_fd,
            )
            report_stat = os.fstat(report_fd)
            if not stat.S_ISREG(report_stat.st_mode) or report_stat.st_nlink != 1:
                raise ValueError("report_artifact must name an existing regular file")
            with os.fdopen(report_fd, "rb", closefd=True) as report_file:
                report_fd = None
                return report_file.read().decode("utf-8")
        except OSError as error:
            # Darwin reports O_DIRECTORY|O_NOFOLLOW on a directory symlink
            # as ENOTDIR; Linux reports ELOOP. Both mean the anchored walk
            # refused a link rather than following it.
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    "report_artifact and its directory must not be symbolic links"
                ) from error
            raise ValueError("report_artifact is not readable UTF-8") from error
        except UnicodeDecodeError as error:
            raise ValueError("report_artifact is not readable UTF-8") from error
        finally:
            if report_fd is not None:
                os.close(report_fd)
            if directory_fd is not None:
                os.close(directory_fd)
            if root_fd is not None:
                os.close(root_fd)

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
            return InvestigationStatus.FAILED
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
        self, owner_email: str, request: CreateInvestigationRequest
    ) -> InvestigationView:
        investigation_id = uuid4().hex
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO replay_jobs (id, owner, status, scenario_family, seed) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    investigation_id,
                    owner_email,
                    InvestigationStatus.QUEUED.value,
                    request.scenario_family.value,
                    request.seed.value,
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
            if job["status"] != InvestigationStatus.PAUSED.value:
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
        if job["status"] != InvestigationStatus.FINALIZED.value:
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
                (InvestigationStatus.PAUSED.value, checkpoint_id, investigation_id),
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
            if row["status"] == InvestigationStatus.FINALIZED.value:
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
                    InvestigationStatus.FINALIZED.value,
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
        control_plane: SqliteReplayControlPlane,
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
        self, control_plane: SqliteReplayControlPlane, sender: ReportDeliverySender
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
    """Concrete VM runner wired permanently to the replay model profile.

    The API never accepts a model selector. This adapter is the only factory
    runner and passes the fixed ``replay`` profile into the existing graph
    composition. It is invoked only by the background worker, never during
    import or HTTP request handling.
    """

    def __init__(
        self,
        root: Path,
        control_plane: SqliteReplayControlPlane,
        replay_wiring: ReplayRuntimeWiring,
    ) -> None:
        self._root = root
        self._control_plane = control_plane
        self._replay_wiring = replay_wiring

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
        if scenario_status is InvestigationStatus.PAUSED:
            release_scenario(self._root, incident_id)
        elif scenario_status in {
            InvestigationStatus.FAILED,
            InvestigationStatus.FINALIZED,
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
        checkpoint_database = ProjectPaths(root=self._root).checkpoints_db
        model, registry, model_name, release = self._replay_wiring.build(
            incident, paths, budgets
        )
        try:
            with _sqlite_checkpointer(checkpoint_database) as checkpointer:
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
    # Belt-and-suspenders gate: even if CAUSALOPS_MCP_DISPATCH=true is set by
    # mistake, mcp_policy_adapter._APPROVED_MCP_DISPATCH being None still
    # makes every McpBackedReplayRuntimeWiring.build() call raise
    # immediately (see policy_approved_mcp_server) -- ENABLE_CLAUDE=false /
    # replay-only-by-default is never actually bypassed without the real,
    # reviewed approval record existing in source.
    mcp_dispatch_requested = os.environ.get(
        VM_EXECUTION_ENV_VARIABLE, ""
    ).strip().lower() == VM_EXECUTION_ENV and os.environ.get(
        "CAUSALOPS_MCP_DISPATCH", ""
    ).strip().lower() in {"1", "true", "yes"}
    replay_wiring: ReplayRuntimeWiring = (
        McpBackedReplayRuntimeWiring()
        if mcp_dispatch_requested
        else HostedReplayRuntimeWiring()
    )
    runner = ReplayGraphJobRunner(root, control_plane, replay_wiring)
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
    return create_app(
        control_plane,
        allowed_owners=owners,
        identity_verifier=GoogleIdentityVerifier(google_client_id),
        google_client_id=google_client_id,
        worker_service=workers,
    )
