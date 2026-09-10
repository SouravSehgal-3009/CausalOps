import os
import sqlite3
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from causalops import api_runtime
from causalops.api import (
    ControlPlaneConflictError,
    ControlPlaneNotFoundError,
    CreateInvestigationRequest,
    DecisionRequest,
    InvestigationStatus,
    ScenarioFamily,
)
from causalops.api_runtime import (
    ControlPlaneIntegrityError,
    DeliveryClaim,
    FinalizedWorkerOutcome,
    PausedWorkerOutcome,
    ReplayGraphJobRunner,
    ReplayWorker,
    ReportDeliveryWorker,
    SqliteReplayControlPlane,
    WorkerClaim,
)
from causalops.approvals import ensure_decisions_table, record_decision_before_resume
from causalops.doctor import ProjectPaths, find_project_root
from causalops.domain import utc_now
from causalops.gcs_artifacts import ARTIFACT_NAMES
from causalops.live_setup import HostedReplayRuntimeWiring

# `_read_report_snapshot`'s descriptor-anchored, no-follow report read
# requires O_DIRECTORY/O_NOFOLLOW, which do not exist in the real `os`
# module on win32 (not just absent from typeshed's stubs there) -- every
# report read genuinely raises on real Windows today. This is a real,
# pre-existing gap in the hosted API, not a test-harness artifact: Windows
# is unsupported for report finalization/delivery/retrieval until a
# Windows-safe equivalent open path is built.
requires_posix_no_follow_reads = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "hosted report finalization/delivery/retrieval needs "
        "os.O_DIRECTORY/os.O_NOFOLLOW, which do not exist on win32; "
        "unsupported on Windows today, not a test-only gap"
    ),
)


class FakeRunner:
    def __init__(
        self, outcomes: list[PausedWorkerOutcome | FinalizedWorkerOutcome]
    ) -> None:
        self._outcomes = outcomes
        self.claims: list[WorkerClaim] = []

    def run(self, claim: WorkerClaim) -> PausedWorkerOutcome | FinalizedWorkerOutcome:
        self.claims.append(claim)
        return self._outcomes.pop(0)


class FakeSender:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.destinations: list[str] = []
        self.contents: list[str] = []

    def send(self, claim: DeliveryClaim) -> None:
        if self.failure is not None:
            raise self.failure
        self.destinations.append(claim.destination_email)
        self.contents.append(claim.report_content)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FailingRunner:
    def run(self, claim: WorkerClaim) -> FinalizedWorkerOutcome:
        raise RuntimeError("runner transient failure")


class SlowPausedRunner:
    def run(self, claim: WorkerClaim) -> PausedWorkerOutcome:
        time.sleep(1.5)
        return PausedWorkerOutcome(checkpoint_id="checkpoint-after-slow-run")


class SlowSender:
    def send(self, claim: DeliveryClaim) -> None:
        time.sleep(1.5)


def request() -> CreateInvestigationRequest:
    return CreateInvestigationRequest(
        scenario_family=ScenarioFamily.CONFIGURATION_CHANGE,
    )


def write_report(artifacts_root: Path, investigation_id: str, content: str) -> None:
    report = artifacts_root / investigation_id / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(content, encoding="utf-8")


def test_durable_replay_job_has_owner_scope_and_ordered_events(tmp_path: Path) -> None:
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db")
    created = control_plane.create("owner@example.com", request(), str(uuid4()))

    assert created.status is InvestigationStatus.QUEUED
    assert [
        event.name
        for event in control_plane.events("owner@example.com", created.investigation_id)
    ] == ["investigation_queued"]
    with pytest.raises(ControlPlaneNotFoundError):
        control_plane.status("other@example.com", created.investigation_id)


def test_repeated_idempotency_key_replays_the_same_investigation(
    tmp_path: Path,
) -> None:
    """A repeated `(owner, idempotency_key)` pair must return the SAME
    investigation, at its current status -- not create a second one. The
    spec requires this exact replay behavior on `POST /investigations`."""
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db")
    key = str(uuid4())

    first = control_plane.create("owner@example.com", request(), key)
    second = control_plane.create("owner@example.com", request(), key)

    assert first.investigation_id == second.investigation_id
    assert second.status is InvestigationStatus.QUEUED
    assert [
        event.name
        for event in control_plane.events("owner@example.com", first.investigation_id)
    ] == ["investigation_queued"]

    # A different owner with the SAME key string is a genuinely different
    # investigation -- the unique constraint is scoped per-owner.
    other_owner = control_plane.create("other@example.com", request(), key)
    assert other_owner.investigation_id != first.investigation_id

    # A different key for the SAME owner is also genuinely a new
    # investigation, not a replay.
    different_key = control_plane.create("owner@example.com", request(), str(uuid4()))
    assert different_key.investigation_id != first.investigation_id


def test_safe_resume_is_checkpointed_write_once_and_idempotent(tmp_path: Path) -> None:
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db")
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    claim = control_plane.claim_next()
    assert claim is not None
    assert claim.investigation_id == created.investigation_id
    control_plane.mark_paused(
        created.investigation_id, "checkpoint-1", claim.claim_token
    )

    accepted = control_plane.decide(
        "owner@example.com",
        created.investigation_id,
        DecisionRequest(decision="accept"),
    )
    assert accepted.status is InvestigationStatus.QUEUED
    assert (
        control_plane.decide(
            "owner@example.com",
            created.investigation_id,
            DecisionRequest(decision="accept"),
        )
        == accepted
    )
    resumed_claim = control_plane.claim_next()
    assert resumed_claim is not None
    assert resumed_claim.checkpoint_id == "checkpoint-1"
    assert resumed_claim.owner_decision == DecisionRequest(decision="accept")
    with pytest.raises(ControlPlaneConflictError):
        control_plane.decide(
            "owner@example.com",
            created.investigation_id,
            DecisionRequest(decision="reject", rejection_note="Need stronger evidence"),
        )
    assert [
        event.name
        for event in control_plane.events("owner@example.com", created.investigation_id)
    ] == [
        "investigation_queued",
        "investigation_started",
        "owner_decision_requested",
        "owner_decision_recorded",
        "investigation_started",
    ]


def test_api_decision_is_durable_in_the_shared_checkpoint_ledger(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "results" / "checkpoints.db"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", checkpoint_database=checkpoints
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    claim = control_plane.claim_next()
    assert claim is not None
    control_plane.mark_paused(
        created.investigation_id, "checkpoint-1", claim.claim_token
    )

    control_plane.decide(
        "owner@example.com",
        created.investigation_id,
        DecisionRequest(decision="accept"),
    )

    with sqlite3.connect(checkpoints) as connection:
        row = connection.execute(
            "SELECT checkpoint_id, decision, rejection_note FROM owner_decisions "
            "WHERE thread_id = ?",
            (created.investigation_id,),
        ).fetchone()
    assert row == ("checkpoint-1", "accept", None)


def test_worker_reconciles_a_ledger_write_interrupted_before_queueing(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "results" / "checkpoints.db"
    checkpoints.parent.mkdir()
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", checkpoint_database=checkpoints
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    claim = control_plane.claim_next()
    assert claim is not None
    control_plane.mark_paused(
        created.investigation_id, "checkpoint-1", claim.claim_token
    )
    with sqlite3.connect(checkpoints) as connection:
        ensure_decisions_table(connection)
        record_decision_before_resume(
            connection,
            created.investigation_id,
            "checkpoint-1",
            DecisionRequest(decision="accept"),
            utc_now(),
        )

    recovered = control_plane.claim_next()
    assert recovered is not None
    assert recovered.investigation_id == created.investigation_id
    assert recovered.owner_decision == DecisionRequest(decision="accept")


@requires_posix_no_follow_reads
def test_finalization_queues_one_owner_only_delivery(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Cited replay report")
    claim = control_plane.claim_next()
    assert claim is not None

    control_plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )
    write_report(artifacts_root, created.investigation_id, "# Mutated report")
    assert (
        control_plane.report("owner@example.com", created.investigation_id)
        == "# Cited replay report"
    )
    delivery = control_plane.claim_delivery()
    assert delivery is not None
    assert delivery.destination_email == "owner@example.com"
    assert delivery.report_content == "# Cited replay report"
    assert control_plane.claim_delivery() is None
    control_plane.mark_delivered(created.investigation_id, delivery.claim_token)
    with pytest.raises(ControlPlaneConflictError):
        control_plane.finalize(
            created.investigation_id, "replacement.md", "obsolete-token"
        )


@requires_posix_no_follow_reads
def test_finalization_and_report_refuse_another_investigations_artifact(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "investigations"
    database = tmp_path / "control-plane.db"
    control_plane = SqliteReplayControlPlane(database, artifacts_root=artifacts_root)
    first = control_plane.create("first@example.com", request(), str(uuid4()))
    second = control_plane.create("second@example.com", request(), str(uuid4()))
    write_report(artifacts_root, first.investigation_id, "# First report")
    write_report(artifacts_root, second.investigation_id, "# Second report")
    first_claim = control_plane.claim_next()
    assert first_claim is not None
    assert first_claim.investigation_id == first.investigation_id

    with pytest.raises(ValueError, match="investigation's own report"):
        control_plane.finalize(
            first.investigation_id,
            f"{second.investigation_id}/report.md",
            first_claim.claim_token,
        )
    control_plane.finalize(
        first.investigation_id,
        f"{first.investigation_id}/report.md",
        first_claim.claim_token,
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE replay_jobs SET report_artifact = ? WHERE id = ?",
            (f"{second.investigation_id}/report.md", first.investigation_id),
        )
    with pytest.raises(ControlPlaneIntegrityError):
        control_plane.report("first@example.com", first.investigation_id)


@requires_posix_no_follow_reads
def test_finalization_refuses_file_and_directory_symlinked_artifacts(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    file_link_job = control_plane.create("first@example.com", request(), str(uuid4()))
    other_job = control_plane.create("other@example.com", request(), str(uuid4()))
    write_report(artifacts_root, other_job.investigation_id, "# Other report")

    file_link_directory = artifacts_root / file_link_job.investigation_id
    file_link_directory.mkdir()
    (file_link_directory / "report.md").symlink_to(
        artifacts_root / other_job.investigation_id / "report.md"
    )

    file_link_claim = control_plane.claim_next()
    assert file_link_claim is not None
    with pytest.raises(ValueError, match="must not be symbolic links"):
        control_plane.finalize(
            file_link_job.investigation_id,
            f"{file_link_job.investigation_id}/report.md",
            file_link_claim.claim_token,
        )


@requires_posix_no_follow_reads
def test_finalization_refuses_a_directory_symlinked_artifact(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    directory_link_job = control_plane.create(
        "second@example.com", request(), str(uuid4())
    )
    other_job = control_plane.create("other@example.com", request(), str(uuid4()))
    write_report(artifacts_root, other_job.investigation_id, "# Other report")
    (artifacts_root / directory_link_job.investigation_id).symlink_to(
        artifacts_root / other_job.investigation_id, target_is_directory=True
    )
    directory_link_claim = control_plane.claim_next()
    assert directory_link_claim is not None
    with pytest.raises(ValueError, match="must not be symbolic links"):
        control_plane.finalize(
            directory_link_job.investigation_id,
            f"{directory_link_job.investigation_id}/report.md",
            directory_link_claim.claim_token,
        )


@requires_posix_no_follow_reads
def test_finalization_refuses_a_cross_owner_hard_link(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    linked_job = control_plane.create("owner@example.com", request(), str(uuid4()))
    other_job = control_plane.create("other@example.com", request(), str(uuid4()))
    write_report(artifacts_root, other_job.investigation_id, "# Other report")
    linked_directory = artifacts_root / linked_job.investigation_id
    linked_directory.mkdir()
    os.link(
        artifacts_root / other_job.investigation_id / "report.md",
        linked_directory / "report.md",
    )

    claim = control_plane.claim_next()
    assert claim is not None
    with pytest.raises(ValueError, match="existing regular file"):
        control_plane.finalize(
            linked_job.investigation_id,
            f"{linked_job.investigation_id}/report.md",
            claim.claim_token,
        )


@requires_posix_no_follow_reads
def test_report_snapshot_ignores_a_symlink_swapped_after_finalization(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    other = control_plane.create("other@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Owner report")
    write_report(artifacts_root, other.investigation_id, "# Other report")
    claim = control_plane.claim_next()
    assert claim is not None
    control_plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )
    report = artifacts_root / created.investigation_id / "report.md"
    report.unlink()
    report.symlink_to(artifacts_root / other.investigation_id / "report.md")

    assert (
        control_plane.report("owner@example.com", created.investigation_id)
        == "# Owner report"
    )


@requires_posix_no_follow_reads
def test_worker_uses_durable_checkpoint_and_finalizes_once(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Final report")
    runner = FakeRunner(
        [
            PausedWorkerOutcome(checkpoint_id="checkpoint-1"),
            FinalizedWorkerOutcome(
                report_artifact=f"{created.investigation_id}/report.md"
            ),
        ]
    )
    worker = ReplayWorker(control_plane, runner)

    assert worker.run_once()
    assert (
        control_plane.status("owner@example.com", created.investigation_id).status
        is InvestigationStatus.PAUSED_APPROVAL
    )
    control_plane.decide(
        "owner@example.com",
        created.investigation_id,
        DecisionRequest(decision="accept"),
    )
    assert worker.run_once()
    assert (
        control_plane.report("owner@example.com", created.investigation_id)
        == "# Final report"
    )
    assert runner.claims[1].owner_decision == DecisionRequest(decision="accept")
    assert not worker.run_once()


@requires_posix_no_follow_reads
def test_delivery_worker_releases_a_failure_and_retries(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "investigations"
    clock = FakeClock()
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root, clock=clock
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Final report")
    claim = control_plane.claim_next()
    assert claim is not None
    control_plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        claim.claim_token,
    )

    with pytest.raises(RuntimeError, match="mail transient failure"):
        ReportDeliveryWorker(
            control_plane, FakeSender(RuntimeError("mail transient failure"))
        ).run_once()
    sender = FakeSender()
    assert not ReportDeliveryWorker(control_plane, sender).run_once()
    clock.now += 5
    assert ReportDeliveryWorker(control_plane, sender).run_once()
    assert sender.destinations == ["owner@example.com"]
    assert not ReportDeliveryWorker(control_plane, sender).run_once()


def test_expired_worker_claim_is_reclaimed_and_stale_worker_is_refused(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db",
        clock=clock,
        claim_lease_seconds=10,
        retry_base_seconds=1,
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    first_claim = control_plane.claim_next()
    assert first_claim is not None

    clock.now += 11
    assert control_plane.claim_next() is None
    clock.now += 1
    recovered_claim = control_plane.claim_next()
    assert recovered_claim is not None
    assert recovered_claim.investigation_id == first_claim.investigation_id
    assert recovered_claim.claim_token != first_claim.claim_token
    with pytest.raises(ControlPlaneConflictError):
        control_plane.mark_paused(
            created.investigation_id, "stale-checkpoint", first_claim.claim_token
        )
    control_plane.mark_paused(
        created.investigation_id, "checkpoint-1", recovered_claim.claim_token
    )


def test_expired_claim_uses_the_same_terminal_retry_limit(tmp_path: Path) -> None:
    clock = FakeClock()
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db",
        clock=clock,
        claim_lease_seconds=1,
        max_replay_attempts=1,
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    assert control_plane.claim_next() is not None

    clock.now += 2
    assert control_plane.claim_next() is None
    assert (
        control_plane.status("owner@example.com", created.investigation_id).status
        is InvestigationStatus.FAILED_SAFE
    )


def test_scenario_identity_is_reserved_before_provisioning_and_survives_reclaim(
    tmp_path: Path,
) -> None:
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db")
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    first_claim = control_plane.claim_next()
    assert first_claim is not None

    incident_id = control_plane.reserve_incident(
        created.investigation_id, first_claim.claim_token
    )
    control_plane.requeue_running(created.investigation_id, first_claim.claim_token)
    recovered_claim = control_plane.claim_next()

    assert recovered_claim is not None
    assert recovered_claim.incident_id == incident_id


def test_only_one_job_can_claim_the_single_mutable_scenario(tmp_path: Path) -> None:
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db")
    control_plane.create("first@example.com", request(), str(uuid4()))
    control_plane.create("second@example.com", request(), str(uuid4()))

    assert control_plane.claim_next() is not None
    assert control_plane.claim_next() is None


@requires_posix_no_follow_reads
def test_runner_adopts_an_artifact_published_before_control_plane_finalization(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "results" / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Recovered report")
    claim = control_plane.claim_next()
    assert claim is not None

    outcome = ReplayGraphJobRunner(
        tmp_path, control_plane, HostedReplayRuntimeWiring()
    ).run(claim)

    assert outcome == FinalizedWorkerOutcome(
        report_artifact=f"{created.investigation_id}/report.md"
    )
    control_plane.finalize(
        created.investigation_id, outcome.report_artifact, claim.claim_token
    )
    assert control_plane.report("owner@example.com", created.investigation_id) == (
        "# Recovered report"
    )


class RecordingArtifactStore:
    """A fake `GcsArtifactStore` -- the same seam-testing approach this
    file already uses for `IdentityVerifier`/scenario control functions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def write_investigation_artifacts(
        self, investigation_id: str, artifacts: Mapping[str, str]
    ) -> None:
        self.calls.append((investigation_id, dict(artifacts)))


@requires_posix_no_follow_reads
def test_runner_uploads_adopted_artifacts_when_a_store_is_configured(
    tmp_path: Path,
) -> None:
    """The fast path (`_has_finalized_artifact`, a prior process already
    wrote locally and died) must ALSO upload to GCS when a store is
    configured -- not just the fresh-finalize path -- since a real crash
    could happen after the local write but before the original upload."""
    artifacts_root = tmp_path / "results" / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    investigation_directory = artifacts_root / created.investigation_id
    investigation_directory.mkdir(parents=True)
    expected_content = {name: f"content-for-{name}" for name in ARTIFACT_NAMES}
    for name, content in expected_content.items():
        (investigation_directory / name).write_text(content, encoding="utf-8")
    claim = control_plane.claim_next()
    assert claim is not None
    store = RecordingArtifactStore()

    outcome = ReplayGraphJobRunner(
        tmp_path, control_plane, HostedReplayRuntimeWiring(), store
    ).run(claim)

    assert outcome == FinalizedWorkerOutcome(
        report_artifact=f"{created.investigation_id}/report.md"
    )
    assert store.calls == [(created.investigation_id, expected_content)]


@requires_posix_no_follow_reads
def test_runner_never_touches_the_artifact_store_when_none_is_configured(
    tmp_path: Path,
) -> None:
    """The default (`artifact_store=None`) must stay a true no-op --
    proven by NOT configuring a store and confirming the adopted-artifact
    fast path still succeeds exactly as it did before this feature
    existed, matching every other `ReplayGraphJobRunner(...)` call site in
    this file that never passes one."""
    artifacts_root = tmp_path / "results" / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Recovered report")
    claim = control_plane.claim_next()
    assert claim is not None

    outcome = ReplayGraphJobRunner(
        tmp_path, control_plane, HostedReplayRuntimeWiring()
    ).run(claim)

    assert outcome == FinalizedWorkerOutcome(
        report_artifact=f"{created.investigation_id}/report.md"
    )


@requires_posix_no_follow_reads
def test_runner_releases_paused_and_finalized_scenario_markers(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "results" / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    claim = control_plane.claim_next()
    assert claim is not None
    incident_id = control_plane.reserve_incident(
        created.investigation_id, claim.claim_token
    )
    marker = tmp_path / "runs" / "active-incident.txt"
    marker.parent.mkdir()
    marker.write_text(incident_id, encoding="utf-8")
    runner = ReplayGraphJobRunner(tmp_path, control_plane, HostedReplayRuntimeWiring())

    control_plane.mark_paused(
        created.investigation_id, "checkpoint-1", claim.claim_token
    )
    runner.release_paused_scenario(claim)
    assert not marker.exists()

    control_plane.decide(
        "owner@example.com",
        created.investigation_id,
        DecisionRequest(decision="accept"),
    )
    resumed_claim = control_plane.claim_next()
    assert resumed_claim is not None
    marker.write_text(incident_id, encoding="utf-8")
    run_directory = tmp_path / "runs" / incident_id
    run_directory.mkdir()
    write_report(artifacts_root, created.investigation_id, "# Final report")
    control_plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        resumed_claim.claim_token,
    )
    runner.release_finalized_scenario(resumed_claim)
    assert not marker.exists()
    assert not run_directory.exists()


@requires_posix_no_follow_reads
def test_runner_reconciles_marker_cleanup_after_a_transition_crash(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "results" / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", artifacts_root=artifacts_root
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    claim = control_plane.claim_next()
    assert claim is not None
    incident_id = control_plane.reserve_incident(
        created.investigation_id, claim.claim_token
    )
    marker = tmp_path / "runs" / "active-incident.txt"
    marker.parent.mkdir()
    marker.write_text(incident_id, encoding="utf-8")
    runner = ReplayGraphJobRunner(tmp_path, control_plane, HostedReplayRuntimeWiring())

    control_plane.mark_paused(
        created.investigation_id, "checkpoint-1", claim.claim_token
    )
    runner.reconcile_scenarios()
    assert not marker.exists()

    control_plane.decide(
        "owner@example.com",
        created.investigation_id,
        DecisionRequest(decision="accept"),
    )
    resumed_claim = control_plane.claim_next()
    assert resumed_claim is not None
    marker.write_text(incident_id, encoding="utf-8")
    run_directory = tmp_path / "runs" / incident_id
    run_directory.mkdir()
    write_report(artifacts_root, created.investigation_id, "# Final report")
    control_plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        resumed_claim.claim_token,
    )
    runner.reconcile_scenarios()
    assert not marker.exists()
    assert not run_directory.exists()


def test_runner_releases_a_marker_whose_job_no_longer_exists(tmp_path: Path) -> None:
    """Found live: `reconcile_scenarios` had no branch for `scenario_status`
    returning `None` -- an orphaned marker (its job document deleted, or a
    crash between `reserve_incident` and the job write) survived forever,
    and every future scenario start failed with `SCENARIO_ALREADY_ACTIVE`
    until manually released. Distinct from QUEUED/RUNNING, which correctly
    still means "leave the marker alone" and isn't covered here."""
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db")
    marker = tmp_path / "runs" / "active-incident.txt"
    marker.parent.mkdir()
    marker.write_text("orphanedincidentid", encoding="utf-8")
    runner = ReplayGraphJobRunner(tmp_path, control_plane, HostedReplayRuntimeWiring())

    runner.reconcile_scenarios()

    assert not marker.exists()


def test_worker_runner_failure_requeues_its_own_claim(tmp_path: Path) -> None:
    clock = FakeClock()
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db", clock=clock)
    created = control_plane.create("owner@example.com", request(), str(uuid4()))

    with pytest.raises(RuntimeError, match="runner transient failure"):
        ReplayWorker(control_plane, FailingRunner()).run_once()
    assert (
        control_plane.status("owner@example.com", created.investigation_id).status
        is InvestigationStatus.QUEUED
    )
    assert control_plane.claim_next() is None
    clock.now += 5
    assert control_plane.claim_next() is not None


def test_retry_limit_marks_a_job_failed_and_unblocks_later_work(tmp_path: Path) -> None:
    clock = FakeClock()
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db",
        clock=clock,
        retry_base_seconds=10,
        max_replay_attempts=2,
    )
    first = control_plane.create("first@example.com", request(), str(uuid4()))
    failed_ids: list[str] = []

    with pytest.raises(RuntimeError):
        ReplayWorker(
            control_plane,
            FailingRunner(),
            on_failed=lambda claim: failed_ids.append(claim.investigation_id),
        ).run_once()
    clock.now += 10
    with pytest.raises(RuntimeError):
        ReplayWorker(
            control_plane,
            FailingRunner(),
            on_failed=lambda claim: failed_ids.append(claim.investigation_id),
        ).run_once()

    assert (
        control_plane.status("first@example.com", first.investigation_id).status
        is InvestigationStatus.FAILED_SAFE
    )
    assert failed_ids == [first.investigation_id]
    second = control_plane.create("second@example.com", request(), str(uuid4()))
    next_claim = control_plane.claim_next()
    assert next_claim is not None
    assert next_claim.investigation_id == second.investigation_id


def test_deferred_scenario_retry_keeps_other_jobs_from_using_its_lab(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", clock=clock, retry_base_seconds=10
    )
    first = control_plane.create("first@example.com", request(), str(uuid4()))
    first_claim = control_plane.claim_next()
    assert first_claim is not None
    control_plane.reserve_incident(first.investigation_id, first_claim.claim_token)
    control_plane.retry_running(
        first.investigation_id, first_claim.claim_token, RuntimeError("temporary")
    )
    control_plane.create("second@example.com", request(), str(uuid4()))

    assert control_plane.claim_next() is None
    clock.now += 10
    retried_claim = control_plane.claim_next()
    assert retried_claim is not None
    assert retried_claim.investigation_id == first.investigation_id


@requires_posix_no_follow_reads
def test_delivery_retry_limit_does_not_block_later_deliveries(tmp_path: Path) -> None:
    clock = FakeClock()
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db",
        artifacts_root=artifacts_root,
        clock=clock,
        retry_base_seconds=10,
        max_delivery_attempts=2,
    )
    first = control_plane.create("first@example.com", request(), str(uuid4()))
    write_report(artifacts_root, first.investigation_id, "# First report")
    first_job_claim = control_plane.claim_next()
    assert first_job_claim is not None
    control_plane.finalize(
        first.investigation_id,
        f"{first.investigation_id}/report.md",
        first_job_claim.claim_token,
    )

    with pytest.raises(RuntimeError):
        ReportDeliveryWorker(
            control_plane, FakeSender(RuntimeError("temporary"))
        ).run_once()
    clock.now += 10
    with pytest.raises(RuntimeError):
        ReportDeliveryWorker(
            control_plane, FakeSender(RuntimeError("temporary"))
        ).run_once()

    second = control_plane.create("second@example.com", request(), str(uuid4()))
    write_report(artifacts_root, second.investigation_id, "# Second report")
    second_job_claim = control_plane.claim_next()
    assert second_job_claim is not None
    control_plane.finalize(
        second.investigation_id,
        f"{second.investigation_id}/report.md",
        second_job_claim.claim_token,
    )
    next_delivery = control_plane.claim_delivery()
    assert next_delivery is not None
    assert next_delivery.investigation_id == second.investigation_id


@requires_posix_no_follow_reads
def test_worker_requeues_when_finalization_fails_after_the_runner_returns(
    tmp_path: Path,
) -> None:
    control_plane = SqliteReplayControlPlane(tmp_path / "control-plane.db")
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    runner = FakeRunner(
        [
            FinalizedWorkerOutcome(
                report_artifact=f"{created.investigation_id}/report.md"
            )
        ]
    )

    with pytest.raises(ValueError, match="not readable"):
        ReplayWorker(control_plane, runner).run_once()
    assert (
        control_plane.status("owner@example.com", created.investigation_id).status
        is InvestigationStatus.QUEUED
    )


def test_worker_renews_a_running_lease_for_a_slow_runner(tmp_path: Path) -> None:
    # A 1.2s lease keeps the heartbeat interval (max(lease/3, floor)) well
    # above MINIMUM_HEARTBEAT_SECONDS (0.4s vs a 0.05s floor), so a single
    # delayed thread wakeup under a loaded CI runner's scheduler jitter
    # cannot burn through the lease window before the next renewal fires --
    # the previous 0.15s lease put the interval exactly at that floor with
    # no margin, which is what made this test flake on macOS/Windows
    # runners (never on Linux) without ever having a real logic bug.
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db", claim_lease_seconds=1.2
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))

    assert ReplayWorker(control_plane, SlowPausedRunner()).run_once()
    assert (
        control_plane.status("owner@example.com", created.investigation_id).status
        is InvestigationStatus.PAUSED_APPROVAL
    )


@requires_posix_no_follow_reads
def test_delivery_worker_renews_a_lease_for_a_slow_sender(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "investigations"
    # Same margin reasoning as test_worker_renews_a_running_lease_for_a_slow_runner.
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db",
        artifacts_root=artifacts_root,
        claim_lease_seconds=1.2,
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Final report")
    job_claim = control_plane.claim_next()
    assert job_claim is not None
    control_plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        job_claim.claim_token,
    )

    assert ReportDeliveryWorker(control_plane, SlowSender()).run_once()
    with sqlite3.connect(tmp_path / "control-plane.db") as connection:
        status = connection.execute(
            "SELECT status FROM report_deliveries WHERE job_id = ?",
            (created.investigation_id,),
        ).fetchone()
    assert status == ("SENT",)


@requires_posix_no_follow_reads
def test_expired_delivery_claim_is_reclaimed_with_a_stable_idempotency_key(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    artifacts_root = tmp_path / "investigations"
    control_plane = SqliteReplayControlPlane(
        tmp_path / "control-plane.db",
        artifacts_root=artifacts_root,
        clock=clock,
        claim_lease_seconds=10,
        retry_base_seconds=1,
    )
    created = control_plane.create("owner@example.com", request(), str(uuid4()))
    write_report(artifacts_root, created.investigation_id, "# Final report")
    job_claim = control_plane.claim_next()
    assert job_claim is not None
    control_plane.finalize(
        created.investigation_id,
        f"{created.investigation_id}/report.md",
        job_claim.claim_token,
    )
    first_delivery = control_plane.claim_delivery()
    assert first_delivery is not None

    clock.now += 11
    assert control_plane.claim_delivery() is None
    clock.now += 1
    retry_delivery = control_plane.claim_delivery()
    assert retry_delivery is not None
    assert retry_delivery.delivery_id == first_delivery.delivery_id
    assert retry_delivery.claim_token != first_delivery.claim_token
    with pytest.raises(ControlPlaneConflictError):
        control_plane.mark_delivered(
            created.investigation_id, first_delivery.claim_token
        )
    control_plane.mark_delivered(created.investigation_id, retry_delivery.claim_token)


def test_app_factory_installs_a_configured_google_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            assert client_id == "client-id"

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("CAUSALOPS_CONTROL_PLANE_DB", str(tmp_path / "control.db"))
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )

    http = TestClient(api_runtime.app())
    assert (
        http.get(
            "/api/v1/investigations/missing", headers={"Authorization": "Bearer token"}
        ).status_code
        == 404
    )


def test_app_factory_rejects_an_unrecognised_control_plane_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("CAUSALOPS_CONTROL_PLANE_BACKEND", "mongodb")

    with pytest.raises(RuntimeError, match="CAUSALOPS_CONTROL_PLANE_BACKEND"):
        api_runtime.app()


def test_app_factory_selects_firestore_control_plane_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`app()`'s default stays sqlite -- this proves the opt-in wiring
    reaches `FirestoreReplayControlPlane`/`FirestoreCheckpointSaver`
    without constructing a real `firestore.Client()` (no ADC, no network),
    the same fake-the-real-dependency approach
    `test_app_factory_installs_a_configured_google_verifier` already takes
    for `GoogleIdentityVerifier`."""
    built: dict[str, object] = {}

    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            pass

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    class FakeFirestoreReplayControlPlane:
        def __init__(self, artifacts_root: Path) -> None:
            built["artifacts_root"] = artifacts_root
            self.claim_lease_seconds = 300.0

    class FakeFirestoreCheckpointSaver:
        def __init__(self) -> None:
            built["checkpointer"] = self

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("CAUSALOPS_CONTROL_PLANE_BACKEND", "firestore")
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )
    monkeypatch.setattr(
        api_runtime, "FirestoreReplayControlPlane", FakeFirestoreReplayControlPlane
    )
    monkeypatch.setattr(
        api_runtime, "FirestoreCheckpointSaver", FakeFirestoreCheckpointSaver
    )

    api_runtime.app()

    root = find_project_root(Path.cwd())
    assert built["artifacts_root"] == root / "results" / "investigations"
    assert "checkpointer" not in built  # only built lazily, inside a run()


def test_app_factory_honours_a_replay_fixture_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (unset) must keep every existing deployment's
    `HostedReplayRuntimeWiring()` exactly as before -- this proves the
    demo/ops-only `CAUSALOPS_REPLAY_FIXTURE` override actually reaches the
    wiring class's constructor, without needing a real graph run."""

    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            pass

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    built: dict[str, object] = {}

    class FakeHostedReplayRuntimeWiring:
        def __init__(self, fixture: Path | None = None) -> None:
            built["fixture"] = fixture

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("CAUSALOPS_REPLAY_FIXTURE", "/tmp/some_demo_fixture.json")
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )
    monkeypatch.setattr(
        api_runtime, "HostedReplayRuntimeWiring", FakeHostedReplayRuntimeWiring
    )

    api_runtime.app()

    assert built["fixture"] == Path("/tmp/some_demo_fixture.json")


def test_app_factory_default_omits_the_replay_fixture_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            pass

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    built: dict[str, object] = {"called_with_kwargs": None}

    class FakeHostedReplayRuntimeWiring:
        def __init__(self, **kwargs: object) -> None:
            built["called_with_kwargs"] = kwargs

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.delenv("CAUSALOPS_REPLAY_FIXTURE", raising=False)
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )
    monkeypatch.setattr(
        api_runtime, "HostedReplayRuntimeWiring", FakeHostedReplayRuntimeWiring
    )

    api_runtime.app()

    assert built["called_with_kwargs"] == {}


def test_app_factory_selects_live_wiring_when_hosted_live_model_is_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`CAUSALOPS_HOSTED_LIVE_MODEL` is an operator-only deployment toggle,
    read once at worker startup -- never a client-facing choice
    (`CreateInvestigationRequest` gains no field for this). Proves the flag
    actually reaches wiring selection, and that the live wiring is
    constructed with the same `checkpoints_db` path the sqlite control-plane
    branch already uses for its own checkpointer, without needing a real
    graph run or a real `ANTHROPIC_API_KEY`."""

    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            pass

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    built: dict[str, object] = {}

    class FakeMcpBackedLiveRuntimeWiring:
        def __init__(self, db_path: Path) -> None:
            built["db_path"] = db_path

    class FakeMcpBackedReplayRuntimeWiring:
        def __init__(self, **kwargs: object) -> None:
            built["replay_wiring_constructed"] = True

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("CAUSALOPS_PROJECT_ROOT", str(find_project_root(Path.cwd())))
    monkeypatch.setenv("CAUSALOPS_HOSTED_LIVE_MODEL", "true")
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )
    monkeypatch.setattr(
        api_runtime, "McpBackedLiveRuntimeWiring", FakeMcpBackedLiveRuntimeWiring
    )
    monkeypatch.setattr(
        api_runtime, "McpBackedReplayRuntimeWiring", FakeMcpBackedReplayRuntimeWiring
    )

    api_runtime.app()

    project_root = find_project_root(Path.cwd())
    assert project_root is not None
    assert built["db_path"] == ProjectPaths(root=project_root).checkpoints_db
    assert "replay_wiring_constructed" not in built


def test_app_factory_default_never_selects_live_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent `CAUSALOPS_HOSTED_LIVE_MODEL`, every existing deployment keeps
    its current replay-only behavior unchanged -- the live wiring class is
    never even constructed."""

    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            pass

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    built: dict[str, object] = {}

    class FakeMcpBackedLiveRuntimeWiring:
        def __init__(self, db_path: Path) -> None:
            built["live_wiring_constructed"] = True

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.delenv("CAUSALOPS_HOSTED_LIVE_MODEL", raising=False)
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )
    monkeypatch.setattr(
        api_runtime, "McpBackedLiveRuntimeWiring", FakeMcpBackedLiveRuntimeWiring
    )

    api_runtime.app()

    assert "live_wiring_constructed" not in built


def test_app_factory_disables_the_background_worker_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud Run has no lab/docker access -- its own deployment must set
    CAUSALOPS_RUN_WORKER=false so `create_app` gets a `NoOpWorkerService`
    instead of a real `BackgroundControlPlaneWorkers`, found live as a real
    gap this session (a real worker started unconditionally would race the
    VM's own worker for jobs it could never actually run)."""

    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            pass

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    captured: dict[str, object] = {}

    def fake_create_app(control_plane: object, **kwargs: object) -> str:
        captured["worker_service"] = kwargs["worker_service"]
        return "app"

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("CAUSALOPS_RUN_WORKER", "false")
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )
    monkeypatch.setattr(api_runtime, "create_app", fake_create_app)

    api_runtime.app()

    assert isinstance(captured["worker_service"], api_runtime.NoOpWorkerService)


def test_app_factory_default_runs_the_background_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGoogleIdentityVerifier:
        def __init__(self, client_id: str) -> None:
            pass

        def verify(self, bearer_token: str) -> str:
            return "owner@example.com"

    captured: dict[str, object] = {}

    def fake_create_app(control_plane: object, **kwargs: object) -> str:
        captured["worker_service"] = kwargs["worker_service"]
        return "app"

    monkeypatch.setenv("CAUSALOPS_ALLOWED_OWNERS", "owner@example.com")
    monkeypatch.setenv("CAUSALOPS_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.delenv("CAUSALOPS_RUN_WORKER", raising=False)
    monkeypatch.setattr(
        api_runtime, "GoogleIdentityVerifier", FakeGoogleIdentityVerifier
    )
    monkeypatch.setattr(api_runtime, "create_app", fake_create_app)

    api_runtime.app()

    assert isinstance(
        captured["worker_service"], api_runtime.BackgroundControlPlaneWorkers
    )
