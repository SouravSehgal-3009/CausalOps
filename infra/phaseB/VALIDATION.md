# Phase B validation record — wiring the provisioned GCS bucket

Dedicated VM (`causalops-test`). Run 2026-09-10.

Phase B wires the already-provisioned `infra/phase2/main.tf` bucket
(`google_storage_bucket.replay_artifacts`, previously dead code — no
application file referenced it) into `finalize_investigation`'s output, per
spec §3.1: "Cloud Storage stores immutable final artifacts under
`investigations/{investigation_id}/`... Writes use generation
preconditions."

## 1. Scope

- New module `src/causalops/gcs_artifacts.py`: `GcsArtifactStore` uploads
  the 5 finalized artifacts (`report.json`, `report.md`, `events.jsonl`,
  `evidence.jsonl`, `receipts.jsonl`) under `investigations/{id}/`, one
  object per file, with `if_generation_match=0` — a retried delivery lands
  on an already-created object (`PreconditionFailed`) and is treated as
  success, not an error, matching the bucket's create-only IAM grant.
- `api_runtime.ReplayGraphJobRunner` gets an optional `artifact_store`
  constructor param (default `None` — unchanged behavior for every
  existing deployment/test). When configured, both `FinalizedWorkerOutcome`
  return paths (fresh finalize, and the crash-recovery fast path that
  adopts an artifact a prior process already wrote) upload before
  returning — a GCS failure propagates through the existing bounded-retry
  machinery the same way a `control_plane.finalize()` failure already does.
- `app()` factory: `CAUSALOPS_ARTIFACT_BUCKET` (and optional
  `CAUSALOPS_ARTIFACT_SERVICE_ACCOUNT` for impersonation) construct the
  store; unset means no GCS upload at all, current behavior preserved.
- New dependency: `google-cloud-storage`. Only reachable from
  `api_runtime.py` — never `cli.py`/`evaluate_cli.py`'s import chain
  (`tests/security/test_no_tracing.py` already proves those import
  cleanly with zero network activity).

## 2. Pre-live verification

- `uv run pytest -q` — 920 passed (up from 914 after Phase A; 6 new: 4 in
  `test_gcs_artifacts.py` covering upload/idempotency/impersonation-request/
  genuine-failure-propagation against a fake client, 2 in
  `test_api_runtime.py` covering both `ReplayGraphJobRunner` return paths
  with a recording fake store, plus a no-store-configured regression test).
- `uv run ruff format --check . && uv run ruff check . && uv run mypy src
  lab` — clean. `google.cloud`/`google.cloud.storage`/
  `google.api_core.exceptions` needed a scoped `[[tool.mypy.overrides]]`
  entry (no py.typed marker mypy recognizes) — narrowly scoped to exactly
  those three modules, not a blanket `google.*` exemption.

## 3. Real infrastructure change

Access design: the VM's actual runtime identity is its default Compute
Engine service account (`612086286944-compute@developer.gserviceaccount.com`),
not the dedicated `causalops-replay-control` SA the bucket's IAM grants
were written for (confirmed live via the metadata server before writing
any code) — no `google_service_account_key` exists for that SA (deliberate,
per the bucket resource's own comment: "no key ever created"). Chosen fix:
impersonation, not granting bucket roles directly to the VM's shared
default SA — preserves the existing least-privilege, one-SA-per-purpose
design `infra/phase2/main.tf` already established, at the cost of more
code and one more IAM resource.

Two new terraform resources, both applied live (owner-approved,
`terraform plan`/`apply` under the same broader ADC account Phase 2's
original apply used):

1. `google_project_service.iam_credentials` — enables
   `iamcredentials.googleapis.com`. Found necessary live: the first
   impersonation attempt failed `SERVICE_DISABLED` before this existed.
2. `google_service_account_iam_member.vm_impersonates_control_plane` —
   grants the VM's default SA `roles/iam.serviceAccountTokenCreator` on
   `causalops-replay-control`, scoped to that one service account, no
   project-wide grant.

Both `terraform apply`s: **Apply complete! Resources: 1 added, 0 changed, 0
destroyed** (once each).

## 4. Live write attempt — blocked by a real, pre-existing VM constraint

With both IAM changes live, a direct `GcsArtifactStore(...).
write_investigation_artifacts(...)` call still failed:

```
google.auth.exceptions.RefreshError: ('Unable to acquire impersonated credentials',
  ...'message': 'Request had insufficient authentication scopes.'...
  'reason': 'ACCESS_TOKEN_SCOPE_INSUFFICIENT'...
  'service': 'iamcredentials.googleapis.com'...)
```

Root-caused, not guessed: an initial smoke test appeared to fail on IAM
propagation delay, but the real cause (confirmed by temporarily moving
aside the cached `application_default_credentials.json` used for
`terraform apply` and re-testing) is that `google.auth.default()` was
picking up that cached broader-account ADC file rather than the VM's own
metadata-server identity — a confound in the TEST, not the code. With that
file moved aside, the real error surfaced: this is the **VM instance's own
attached-SA OAuth scopes** (a GCE instance-level setting, distinct from IAM
role bindings) not including anything that reaches
`iamcredentials.googleapis.com` at all. This is the exact same narrow-scope
constraint `infra/phase2/VALIDATION.md` already documented blocking the
original `terraform apply` (`devstorage.read_only` / `logging.write` /
`monitoring.write` / `service.management.readonly` / `servicecontrol` /
`trace.append` — no `cloud-platform`), now also blocking the *running
application's* impersonation calls, not just terraform.

No IAM change fixes this — it requires widening the VM instance's own
attached-SA scopes (`gcloud compute instances set-service-account
--scopes=...`), which needs the instance to be stopped and restarted. Owner
decision: defer this — this VM is the current session's own live
environment, and a restart is disruptive enough to schedule deliberately
rather than do mid-session.

## 5. VM scope widened, live write confirmed end-to-end

Owner stopped `causalops-test`, ran `gcloud compute instances
set-service-account causalops-test --zone=asia-south1-c
--scopes=cloud-platform`, started it again. Boot disk is persistent
(`/dev/root`, `infra/phase2/VALIDATION.md` §1) — this session's own saved
state survived the restart; only the live process needed resuming.

Post-restart, metadata server confirms the widened scope:
`https://www.googleapis.com/auth/cloud-platform`.

Re-ran the exact §4 smoke test (cached broader ADC file moved aside again,
so the test exercises the VM's own metadata-server identity, not the
operator's cached account):

| Check | Result |
|---|---|
| Upload via impersonated credentials | **succeeded** (`upload ok`, no exception) — the `ACCESS_TOKEN_SCOPE_INSUFFICIENT` failure is gone |
| Object genuinely landed in the bucket | Read back via a second impersonated client (the same identity path the app uses, not the broader account): `exists, size: 13, generation: 1788986044877498`, content matched exactly what was written |
| Idempotent retry | Uploading the identical `(investigation_id, content)` pair a second time raised nothing — `PreconditionFailed` correctly swallowed, confirming the retry-safety design (§1) live, not just against the fake client in `test_gcs_artifacts.py` |
| Cleanup | Smoke-test object deleted via the broader (bucket-owner) account — confirmed the app's own impersonated identity has no delete permission (`objectCreator`/`objectViewer` only, per `infra/phase2/main.tf`'s original design), matching the write-once/immutable-artifact intent |

## 6. Outcome

Application code, both terraform resources, and the VM's own instance
scopes are all in place and confirmed working end-to-end against the real
bucket: upload, read-back, and idempotent-retry all verified live, not
just unit-tested. Phase B is fully closed — no open follow-up remains.
