# Phase D validation record — Cloud Run API deployment

Dedicated VM (`causalops-test`, `asia-south1-c`, project
`project-7b68a103-d482-4418-a35`). Run 2026-09-10.

Phase D was scoped this session, not implemented straight from the original
lettered plan's "Pub/Sub worker + Cloud Run" framing — two decisions were
made before any code/infra:

1. **Pub/Sub dropped from scope.** `lab/docker-compose.yml`'s
   `gateway`/`orders`/`inventory`/`prometheus` all bind `127.0.0.1:<port>`
   (loopback-only), and the same compose file bind-mounts `../runs:/runs` —
   the exact directory `scenario_control.py`'s `root / "runs"` reads/writes.
   The replay worker cannot leave the VM without rewriting the lab itself,
   so only the owner-facing HTTP API is a real Cloud Run candidate. Pub/Sub
   would only replace `BackgroundControlPlaneWorkers`' existing 0.25s poll
   loop with a wake signal — a latency nicety, not a scaling win, since
   `firestore_control_plane.py`'s own `threading.Lock` concurrency model
   still needs replacing with real Firestore transactions before any
   multi-worker fan-out is safe. Deferred until that's actually needed.
2. **Clean-cut sqlite→Firestore restart, no migration script.** Moving the
   API to Cloud Run forces both the API and the VM worker onto the same
   control-plane backend (Cloud Run has no local disk to share a SQLite
   file with the VM) — this was the deferred Phase C decision, now made:
   whatever was `QUEUED`/`RUNNING`/`PAUSED_APPROVAL` in
   `results/control-plane.db` at cutover was abandoned, not migrated (in
   practice: 1 stale `QUEUED` row, 8 already-`FINALIZED` historical rows —
   nothing was actually in flight).

## 1. Scope

- `infra/phaseD/Dockerfile`: `python:3.12-slim`, installs via `uv sync
  --locked --no-dev`, runs the synced venv's own `uvicorn` binary directly
  (not `uv run uvicorn` — that re-verifies/re-syncs dependency groups on
  every container start and silently pulled the dev group, mypy/ruff
  included, back in at boot; direct venv invocation skips that).
- `infra/phase2/main.tf`: `google_project_service` for
  `run.googleapis.com`/`artifactregistry.googleapis.com`, a
  `google_artifact_registry_repository` (`causalops-api`, Docker format), a
  dedicated `causalops-api` service account (`roles/datastore.user` only —
  no GCS role, since the owner-facing routes never touch the artifact
  bucket directly, only `report_content` already stored in the Firestore
  document by the worker's own `finalize()`), a
  `google_cloud_run_v2_service` running the pushed image with
  `CAUSALOPS_CONTROL_PLANE_BACKEND=firestore` baked into its env, and
  `google_cloud_run_v2_service_iam_member` for public invocation (Google ID
  token auth still enforced at the application layer, same as the VM
  today — this grant only controls transport reachability).
- `google_billing_budget`: written but not applied this session —
  `var.billing_account_id` requires billing-account-level IAM the current
  terraform identity does not hold (`gcloud billing accounts list` failed
  with `cloudbilling.googleapis.com` disabled and `PERMISSION_DENIED` under
  the VM's own default compute SA). Left as an explicit follow-up, not
  silently skipped — the variable defaults empty and the resource is
  `count`-guarded off until it's set.

## 2. Pre-apply verification

- `terraform validate` / `terraform plan` (with `api_image`/
  `billing_account_id` unset) — confirmed the Cloud Run service and public-
  invoker resources are `count`-guarded to 0 until an image is actually
  pushed; base apply was 5 resources, no Cloud Run service yet.
- Local image smoke test (`docker run` on the VM, not Cloud Run) — boots in
  ~10-15s (real Firestore ADC resolution + a real
  `_reclaim_expired_investigations` query against the live database at
  worker-adjacent startup path, confirmed by the same `UserWarning` the
  live scripts already produce), serves `/docs` with `200`.

## 3. Terraform apply (two stages)

- **Base** (`api_image`/`billing_account_id` unset): `terraform apply` — 5
  added (project services, Artifact Registry repo, `causalops-api` SA, its
  Firestore IAM grant), 0 changed, 0 destroyed.
- Image build (`docker build -f infra/phaseD/Dockerfile .`) and
  `docker push` to the new repo — first push attempt failed
  (`artifactregistry.repositories.uploadArtifacts` denied for the VM's
  default compute SA, the identity `gcloud`/`docker` actually push as,
  distinct from terraform's own broader ADC identity that created the
  repo). Fixed with one more narrowly-scoped terraform resource
  (`google_artifact_registry_repository_iam_member`, `roles/
  artifactregistry.writer`, scoped to this one repository) — applied (1
  added), then the push succeeded.
- **Cloud Run** (`api_image`/`google_client_id`/`allowed_owners` set in
  the gitignored `terraform.tfvars`, matching the VM's own existing
  `.env.vm` values): `terraform apply` — 2 added (the service, the public-
  invoker grant). `cloud_run_api_url` output:
  `https://causalops-api-h23m2uzfta-uc.a.run.app`.

## 4. Cutover and live cross-surface proof

VM's `uvicorn` was not running at cutover time (no supervisor exists —
confirmed this session there never has been one; nothing was disrupted).
Started fresh: `CAUSALOPS_CONTROL_PLANE_BACKEND=firestore` exported
alongside the existing `.env.vm` values, backgrounded with `nohup`/
`disown` so it survives past this session (still no OS-level supervisor —
that's `infra/phaseD`'s own noted follow-up, not fixed here).

Both surfaces reachable: VM `http://127.0.0.1:8000/docs` → `200`, Cloud Run
`https://causalops-api-h23m2uzfta-uc.a.run.app/docs` → `200`.

**Decisive cross-surface proof** (no real Google ID token exchange needed
for this — the same "write directly through the real backend class"
approach Phase C's own live validation used): created one investigation
directly through a real `FirestoreReplayControlPlane` client, simulating
exactly what the Cloud Run API's `create()` route does, then watched the
VM's own already-running worker (no restart, no signal, purely polling)
discover, claim, and run it through the REAL lab (`docker compose`
services, all healthy) and REAL graph execution — not the "already
finalized artifact" fast path Phase C's own validation used.

| Check | Result |
|---|---|
| Created via a standalone Firestore client (simulating Cloud Run's `create()`) | `QUEUED`, real investigation id |
| VM worker (already running, untouched) claimed it | `RUNNING` within seconds, `scenario_reserved` event present — proves the poll loop sees writes from a process it never talked to directly |
| Full graph execution against the real lab | `COMPLETED`; events `investigation_queued, investigation_started, scenario_reserved, report_finalized, report_delivered` — the complete clean chain, first attempt |
| `report()` retrieval | 1491 characters, non-empty, matches a real finalized report |
| Cleanup | job/events/delivery documents for this synthetic-owner investigation deleted afterward |

## 5. Outcome

The API/worker split is live and proven consistent end-to-end: a request
landing on Cloud Run and a request landing on the VM's own `uvicorn` now
observe the same investigations, because both read/write the same
Firestore control plane. Not yet done, left as explicit follow-ups:

- `google_billing_budget` — needs billing-account IAM this session's
  identity doesn't hold.
- No CI/CD deploy job — this session's build/push/apply was manual; a
  `workflow_dispatch` `gcloud run deploy` step is still open.
- No process supervisor for the VM's own `uvicorn` (pre-existing gap, not
  introduced by Phase D) — Cloud Run itself doesn't have this problem
  (managed restart), but the VM-hosted worker still does.
- A real owner-authenticated HTTP round trip through the Cloud Run URL
  (Google sign-in + bearer token, matching `infra/phase2/VALIDATION.md`'s
  §4 methodology) was not run this session — the direct-Firestore-client
  proof above establishes the same control-plane-sharing claim without
  needing a fresh OAuth exchange, but a real end-user HTTP path through the
  Cloud Run URL specifically is still open.
