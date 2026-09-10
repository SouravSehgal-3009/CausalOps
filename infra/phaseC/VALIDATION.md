# Phase C validation record — Firestore migration

Dedicated VM (`causalops-test`, `asia-south1-c`, project
`project-7b68a103-d482-4418-a35`). Run 2026-09-10.

Phase C replaces the SQLite-backed control plane and LangGraph checkpointer
with Firestore-backed equivalents, matching spec §3.1's
`investigations/{investigation_id}` / `investigations/{investigation_id}/events/{sequence}`
shape. Both backends are additive: `SqliteReplayControlPlane` stays the
default; Firestore is an explicit opt-in pending this validation record
(see §5).

## 1. Scope

- `src/causalops/firestore_checkpointer.py`: `FirestoreCheckpointSaver`, a
  full `BaseCheckpointSaver[str]` implementation (`get_tuple`, `list`,
  `put`, `put_writes`, `delete_thread`, `get_next_version`), including the
  900 KiB checkpoint-size refusal spec §3.1 requires (no such refusal
  existed anywhere in the codebase before this).
- `src/causalops/firestore_control_plane.py`: `FirestoreReplayControlPlane`,
  the full worker- and owner-facing surface `SqliteReplayControlPlane`
  exposes, backed by 4 Firestore collections (`replay_jobs`,
  `replay_jobs/{id}/events`, `report_deliveries`, `owner_decisions`), with
  Firestore `WriteBatch` replacing every `BEGIN IMMEDIATE` multi-row
  transaction, and a single `threading.Lock` replacing SQLite's exclusive
  connection for every read-check-write sequence (documented scope
  boundary: correct for one `BackgroundControlPlaneWorkers` process;
  needs real Firestore transactions before a multi-process Pub/Sub fan-out,
  a later phase).
- `src/causalops/report_snapshot.py`: symlink-safe report-reading logic
  extracted out of `SqliteReplayControlPlane` so both backends share one
  implementation of this security-sensitive path.
- `WorkerClaim`/`DeliveryClaim` moved from `api_runtime.py` into
  `causalops/api.py` so both control-plane backends' worker-facing methods
  return the same nominal type — a new `WorkerControlPlane` structural
  Protocol in `api_runtime.py` lets `ReplayGraphJobRunner`/`ReplayWorker`/
  `ReportDeliveryWorker` accept either backend without importing the other
  backend's module.
- `app()` factory: `CAUSALOPS_CONTROL_PLANE_BACKEND` (`sqlite`, the
  default, or `firestore`) selects the control plane and a matching
  checkpointer factory. Unset means fully unchanged behavior for every
  existing deployment.
- Terraform (`infra/phase2/main.tf`): `google_project_service.firestore`,
  `google_firestore_database.default` (Native mode, `us-central1`,
  `deletion_policy = "ABANDON"` — a destroy never deletes the underlying
  data), and `google_project_iam_member.vm_firestore_user` granting the
  VM's default compute service account `roles/datastore.user` directly
  (unlike the GCS bucket, Firestore has no per-collection IAM condition to
  scope this further — both `FirestoreCheckpointSaver` and
  `FirestoreReplayControlPlane` resolve `firestore.Client()` against ADC
  directly, not through `GcsArtifactStore`'s impersonation pattern).

## 2. Pre-live verification

- `uv run pytest -q` — 961 passed (up from 958 before this phase's test
  additions: 24 in `test_firestore_control_plane.py`, including one real
  `ReplayGraphJobRunner` end-to-end run against a fake Firestore client —
  the same "prove it against a real consumer, not just hand-built
  fixtures" approach already taken for `FirestoreCheckpointSaver`'s own
  `test_a_real_graph_investigation_runs_end_to_end_on_this_checkpointer`;
  plus 2 new `app()`-factory tests in `test_api_runtime.py` covering the
  `CAUSALOPS_CONTROL_PLANE_BACKEND` toggle, both with the real GCP clients
  faked out).
- `uv run ruff format --check . && uv run ruff check . && uv run mypy src
  lab` — clean, 49 source files.
- One real bug caught and fixed before this record: `claim_next()`'s FIFO
  tiebreak initially used the random investigation-id UUID (Firestore has
  no equivalent of SQLite's own monotonic `rowid`, which the original
  `ORDER BY ... j.rowid` relies on). Fixed by adding an explicit
  `sequence` integer field, assigned under the same lock at `create()`/
  `finalize()` time, used as the final sort key for both `claim_next()`
  and `claim_delivery()`.

## 3. Terraform apply

- `terraform init` — reused existing state, provider `hashicorp/google`
  v6.50.0 (no new provider version).
- `terraform plan` — 3 to add, 0 change, 0 destroy.
- `terraform apply` — **Apply complete! Resources: 3 added, 0 changed, 0
  destroyed.** `firestore_database_name = "(default)"`.
- `gcloud firestore databases list` — confirms `type: FIRESTORE_NATIVE`,
  `locationId: us-central1`, `deleteProtectionState:
  DELETE_PROTECTION_DISABLED` (matches `deletion_policy = "ABANDON"` — an
  explicit, deliberate choice for a project still under active
  development, not yet the final production posture).

## 4. Live validation against the real database (not the fake test client)

Two standalone scripts, run directly against real ADC and the real
Firestore database above (no emulator, no fake client) — the VM's default
compute service account, carrying `roles/datastore.user` from §1 and the
`cloud-platform` OAuth scope already widened during Phase B.

**Control plane + real `ReplayGraphJobRunner`** (`HostedReplayRuntimeWiring`,
the "already finalized artifact" adoption fast path — the same one
`test_runner_adopts_an_artifact_published_before_control_plane_finalization`
exercises, chosen to isolate the Firestore code path from the unrelated lab
scenario/graph pipeline already proven elsewhere):

| Check | Result |
|---|---|
| `create()` | QUEUED, real investigation id assigned |
| `claim_next()` | claimed exactly that job, real claim token |
| `ReplayGraphJobRunner.run()` | `FinalizedWorkerOutcome`, adopted the pre-written report |
| `finalize()` then `report()` | report content round-tripped exactly (34 chars written, 34 read back) |
| `status()` | `COMPLETED` |
| `events()` | `investigation_queued, investigation_started, report_finalized` — correct order, no `sequence`-tiebreak regression |
| Cleanup | job, events subcollection, delivery, and decision documents all deleted; `replay_jobs`/`report_deliveries`/`owner_decisions` collections confirmed empty afterward |

**`FirestoreCheckpointSaver`** (real client, not `FakeFirestoreClient`):

| Check | Result |
|---|---|
| `put()` then `get_tuple()` | round-tripped `channel_values`, checkpoint id, metadata step exactly |
| Second `put()` then `list()` | both checkpoints returned |
| `delete_thread()` | `get_tuple()` on the same thread returned `None` afterward |
| Cleanup | `checkpoints` collection confirmed empty afterward |

Both scripts ran clean on the first attempt — no IAM, scope, or API-enablement
error, confirming the terraform grants in §1/§3 are sufficient on their own
(no manual `gcloud` permission workaround was needed, unlike Phase B's
initial `ACCESS_TOKEN_SCOPE_INSUFFICIENT`).

## 5. Outcome

Both Firestore backends are built, unit-tested (including real-consumer
proof against a fake client), and now live-verified against the real
provisioned database — but `CAUSALOPS_CONTROL_PLANE_BACKEND` stays
`sqlite` by default. Flipping the default is a deliberate follow-up
decision, not a mechanical next step: the running VM's `results/control-
plane.db`/`results/checkpoints.db` hold real in-flight investigation state
today, and nothing in this phase migrates that data into Firestore. That
cutover — and the sqlite-vs-firestore data-migration question it raises —
needs its own explicit go-ahead before the default changes.

Phase D (Pub/Sub worker + Cloud Run) has not started.
