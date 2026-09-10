# Deploying the hosted API on your own GCP account

This covers the **hosted** deployment (a real API a browser can sign into,
backed by real GCP infrastructure) — not the local CLI, which needs none of
this (see the root `README.md`'s "Setup" section instead: `uv sync`,
`causalops doctor`, `causalops investigate`, all fully local and free).

Written from a real, reproducible deployment done this way — every command
below was actually run, including the mistakes and their fixes, called out
explicitly in "Gotchas" rather than smoothed over.

## Architecture, in one paragraph

One machine (a GCE VM, or any always-on machine you control) runs the
synthetic lab (`docker compose`) and a background worker process that
executes investigations against it — this machine needs real Docker access,
so it can't be serverless. Everything that machine and its worker touch is
optional beyond that: you can stop here with a fully working single-process
deployment (SQLite-backed, no GCP dependency at all beyond your own
machine), or go further and put the owner-facing HTTP API on Cloud Run
(stateless, autoscaling, no lab access needed) while the worker stays on
your machine — both then share one Firestore database as their common
source of truth. This guide covers both, in order, so you can stop after
either one.

## Prerequisites

- A GCP account with billing enabled on the project you'll use.
- `gcloud` CLI, authenticated (`gcloud auth login`).
- Terraform >= 1.8.
- Docker + Docker Compose.
- `uv` and Python 3.12 (see root `README.md`).
- One always-on machine with Docker access for the lab/worker — a small GCE
  VM (e2-small or similar) works fine; so does your own machine if it's
  reachable and stays on.

## Step 1 — GCP project

```bash
gcloud projects create YOUR_PROJECT_ID   # or use an existing one
gcloud config set project YOUR_PROJECT_ID
gcloud billing projects link YOUR_PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID
```

Authenticate Application Default Credentials — this is the identity
Terraform actually uses, separate from `gcloud`'s own active account:

```bash
gcloud auth application-default login
```

The account behind this needs enough IAM to create service accounts, IAM
bindings, a Firestore database, a GCS bucket, and (if you go that far) a
Cloud Run service — `roles/owner` or `roles/editor` on the project covers
all of it for a first-time setup. A narrower role set works too if you'd
rather scope it down; Terraform's own error messages name exactly which
permission is missing if you under-provision.

## Step 2 — Base infrastructure

```bash
cd infra/gcp
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: set project_id, region, artifact_bucket_name
# (bucket names are globally unique across all of GCS, not just your project)
terraform init
terraform plan    # review before applying
terraform apply
```

This creates, and only this: a Firestore Native database, a private
versioned GCS bucket, two dedicated service accounts (`causalops-replay-
control` for artifact writes, `causalops-api` for the optional Cloud Run
piece), their IAM grants, and an Artifact Registry repo (empty until you
push an image in Step 6). No Cloud Run service, no billing budget — both
stay off until their own variables are set (see `terraform.tfvars.example`'s
comments).

## Step 3 — The worker machine

On the machine that will run the lab and the worker:

```bash
git clone <this-repo-url> && cd CausalOps
uv sync --locked
docker compose -f lab/docker-compose.yml up -d
uv run causalops doctor   # confirms Docker, disk, checkpoint DB are all OK
```

Copy the environment template and fill it in:

```bash
cp .env.vm.example .env.vm
# edit .env.vm: CAUSALOPS_PROJECT_ROOT (absolute path to this checkout),
# CAUSALOPS_ALLOWED_OWNERS (your Google account email), leave
# CAUSALOPS_CONTROL_PLANE_BACKEND unset for now (sqlite) -- you can flip it
# to firestore once Step 4's OAuth client exists and you're ready to test
# the real sign-in flow.
```

If this machine is a GCE VM using its own default service account (rather
than a user's `gcloud auth application-default login`), see **Gotcha 1**
below before continuing — Firestore/Cloud-Run-facing calls will fail
otherwise with a scope error, not an obviously-named permission error.

## Step 4 — OAuth client (read this before clicking anything)

**The single most confusing failure mode in this whole setup is creating
the wrong client type.** In Google Cloud Console → APIs & Services →
Credentials → Create Credentials → OAuth client ID:

- Application type: **Web application** — not Desktop, not Android, not
  anything else. Only a Web application client exposes "Authorized
  JavaScript origins," which is what the dashboard's sign-in button
  actually needs (Google Identity Services, `accounts.google.com/gsi/
  client`). A Desktop-type client looks superficially fine (it's still an
  OAuth client, still has an ID) but fails at sign-in time with `Error 401:
  invalid_client` and gives no indication the client *type* is the
  problem — confirmed live, the hard way, this project's own first attempt
  used a pre-existing Desktop client and hit exactly this.
- Authorized JavaScript origins: add `http://localhost:8000` (Google
  Identity Services allows plain HTTP for localhost/127.0.0.1 specifically,
  for local testing) for now. If you're doing Step 6 (Cloud Run) later,
  you'll come back here and add that URL too, once you know it — you can't
  know it before the first Cloud Run deploy, since GCP generates it.
- If your project's OAuth consent screen is still in "Testing" (the
  default, and fine for a personal/demo deployment — publishing requires
  Google's app-verification review), add every `CAUSALOPS_ALLOWED_OWNERS`
  email as a test user there too. `CAUSALOPS_ALLOWED_OWNERS` is this
  application's own authorization boundary; the consent screen's test-user
  list is a *separate*, Google-enforced gate in front of it — both need to
  agree, or an otherwise-allowed owner still can't sign in.

Copy the resulting Client ID into `.env.vm`'s `CAUSALOPS_GOOGLE_CLIENT_ID`
(and, later, `terraform.tfvars`'s `google_client_id` if you deploy Cloud
Run — same client, both surfaces enforce the same allowlist).

## Step 5 — Run the worker

Quick foreground test first:

```bash
set -a; source .env.vm; set +a
uv run uvicorn causalops.api_runtime:app --factory --workers 1 --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in a browser, sign in, create a
`configuration_change` investigation, confirm it reaches `COMPLETED` — see
[`../docs/CLOUD_RUN_DEMO.md`](../docs/CLOUD_RUN_DEMO.md) for the full
owner-facing walkthrough (all four families, the pause/approve/reject
flow, talking points). Once confirmed, Ctrl-C it and run it for real via
systemd so it survives crashes and reboots:

```bash
cp infra/docker/causalops-api.service.example /tmp/causalops-api.service
# edit /tmp/causalops-api.service: replace YOUR_USER and every
# /absolute/path/to/... placeholder with your real values
sudo cp /tmp/causalops-api.service /etc/systemd/system/causalops-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now causalops-api
sudo systemctl status causalops-api   # should show "active (running)"
```

This is a complete, working deployment. Everything below is optional.

## Step 6 (optional) — Cloud Run API split

Only worth doing if you want the owner-facing API on a stateless, publicly
autoscaling surface instead of serving it from the worker machine directly.
Requires `CAUSALOPS_CONTROL_PLANE_BACKEND=firestore` on the worker (Cloud
Run has no local disk to share a SQLite file with it) — set that in
`.env.vm` now and restart the service (`sudo systemctl restart
causalops-api`) before continuing.

```bash
gcloud auth configure-docker REGION-docker.pkg.dev   # once per machine

docker build -f infra/docker/api.Dockerfile \
  -t REGION-docker.pkg.dev/YOUR_PROJECT_ID/causalops-api/causalops-api:v1 .
docker push REGION-docker.pkg.dev/YOUR_PROJECT_ID/causalops-api/causalops-api:v1
```

Edit `terraform.tfvars`: uncomment and fill in `api_image` (the exact image
reference you just pushed), `google_client_id` (Step 4's Web application
client ID), `allowed_owners` (same list as `.env.vm`'s
`CAUSALOPS_ALLOWED_OWNERS`, as a Terraform list).

```bash
cd infra/gcp
terraform apply
terraform output cloud_run_api_url
```

Take that URL back to Step 4's OAuth client and add it to Authorized
JavaScript origins. Sign in there too, confirm an investigation created
through the Cloud Run URL gets picked up and completed by the worker
machine (it will — they share the same Firestore database).

## Gotchas (all found live, all real)

1. **Running Terraform from a GCE VM's own default service account**: its
   OAuth scopes are narrow by default (`devstorage.read_only`,
   `logging.write`, ...) and `terraform apply` fails with
   `ACCESS_TOKEN_SCOPE_INSUFFICIENT`. Either authenticate as a real user
   account instead (`gcloud auth application-default login`, as in Step
   1), or widen the VM's attached scopes (`gcloud compute instances stop`,
   `gcloud compute instances set-service-account --scopes=cloud-platform`,
   `start` — a real restart, do this from outside the VM if you're running
   commands from inside it).
2. **OAuth client type** — see Step 4. This is the one to read twice.
3. **Don't run anything that calls `scenario_control.start_scenario()`
   directly on the local lab (the docker-marked integration tests, `causalops
   scenario start`, `causalops-evaluate`) while `causalops-api` is running
   on the same machine.** The background worker polls every 0.25s and
   reconciles any scenario marker with no matching Firestore job — correct
   for cleaning up genuinely orphaned state, but all of these bypass
   Firestore entirely, so the worker sees the marker as orphaned and
   releases it mid-run. The observed symptom is not an obvious "marker
   released" error: `start_scenario`'s own fault-injection phase reads that
   same marker on every request, so losing it mid-batch surfaces as `FAIL
   FAULT_NOT_OBSERVED` (or, in a `causalops-evaluate` run, an incident that
   silently investigates the wrong/no fault) — confusing unless you know to
   suspect the worker first. `sudo systemctl stop causalops-api` first, run
   the local work, then start it again.
4. **The synthetic lab isn't designed for indefinite uptime.** After many
   scenario cycles, fault injection can stop reliably producing a failing
   request (`LabError: FAULT_NOT_OBSERVED`). Fix: `docker compose -f
   lab/docker-compose.yml down && docker compose -f lab/docker-compose.yml
   up -d`. Do this before anything important (a demo, a fresh evaluation
   run), not just when something breaks.
5. **`docker push` can fail with a permission error even though
   `terraform apply` succeeded** — the identity that actually runs `docker
   push` (your `gcloud` CLI's active account) may differ from the one
   Terraform used (its own ADC identity). This repo's own
   `infra/gcp/artifact_registry.tf` already grants
   `roles/artifactregistry.writer` to the VM's default compute SA for
   this reason; if you're pushing as a different identity, it needs the
   same grant.
6. **Cloud Run must never run its own background worker.** Terraform
   already sets `CAUSALOPS_RUN_WORKER=false` on the Cloud Run service for
   you — if you ever deploy that image outside Terraform (a manual `gcloud
   run deploy`), set it explicitly, or Cloud Run will race the real
   worker for jobs it has no lab access to actually run.

## Cost

Cloud Run scales to zero (no traffic, no compute charge) since nothing here
sets `min_instance_count`. Firestore and GCS are pay-per-use, negligible at
low volume. The worker machine (VM or otherwise) is the one thing that
bills continuously if left running — stop it when you're not using it.

## Tearing down

```bash
cd infra/gcp
# unset api_image in terraform.tfvars first if you deployed Cloud Run,
# then:
terraform destroy
```

Terraform refuses to destroy the GCS bucket while it still holds objects
(versioning is on) — empty it first (`gsutil rm -r
gs://YOUR_BUCKET_NAME/**` or the Console) if `destroy` stops there.
Deleting the Firestore database deletes all investigation/checkpoint data
permanently — there's no undo. Stop (or delete) the worker machine
separately; Terraform doesn't manage it.
