terraform {
  required_version = ">= 1.8"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "current" {
  project_id = var.project_id
}

# Required for the VM's default service account to impersonate
# `replay_control_plane` at all (`google_service_account_iam_member.
# vm_impersonates_control_plane` below) -- without it, every impersonated
# credential request is refused with SERVICE_DISABLED, discovered live
# during Phase B validation.
resource "google_project_service" "iam_credentials" {
  project            = var.project_id
  service            = "iamcredentials.googleapis.com"
  disable_on_destroy = false
}

# This identity is for the replay control plane only. It has no owner role,
# service-account-key, compute, network, or provider-management permissions.
resource "google_service_account" "replay_control_plane" {
  account_id   = "causalops-replay-control"
  display_name = "CausalOps replay control plane"
}

resource "google_project_iam_member" "control_plane_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.replay_control_plane.email}"
}

# Final reports are versioned, private objects. Application code still enforces
# write-once finalization before it hands an artifact to the delivery outbox.
resource "google_storage_bucket" "replay_artifacts" {
  name                        = var.artifact_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }
}

locals {
  artifact_prefix = "projects/_/buckets/${google_storage_bucket.replay_artifacts.name}/objects/investigations/"
}

# Restrict both permissions to final-investigation objects. The API reads an
# exact known report path; it does not require bucket-wide access.
resource "google_storage_bucket_iam_member" "control_plane_artifact_writer" {
  bucket = google_storage_bucket.replay_artifacts.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.replay_control_plane.email}"

  condition {
    title       = "write_final_investigation_artifacts"
    description = "Create artifacts below the investigation prefix only."
    expression  = "resource.name.startsWith('${local.artifact_prefix}')"
  }
}

resource "google_storage_bucket_iam_member" "control_plane_artifact_reader" {
  bucket = google_storage_bucket.replay_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.replay_control_plane.email}"

  condition {
    title       = "read_final_investigation_artifacts"
    description = "Read artifacts below the investigation prefix only."
    expression  = "resource.name.startsWith('${local.artifact_prefix}')"
  }
}

# The VM's own runtime identity is its default Compute Engine service
# account, not `replay_control_plane` -- no `google_service_account_key`
# exists (see the bucket resource's own comment: no key ever created), so
# the application impersonates `replay_control_plane` for exactly as long
# as one GCS request instead. This is the only grant the VM's default SA
# receives; it still has no direct bucket role of its own.
resource "google_service_account_iam_member" "vm_impersonates_control_plane" {
  service_account_id = google_service_account.replay_control_plane.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"

  depends_on = [google_project_service.iam_credentials]
}

# Phase C (Firestore migration): `FirestoreCheckpointSaver`/
# `FirestoreReplayControlPlane` both resolve `firestore.Client()` against
# Application Default Credentials directly -- unlike `GcsArtifactStore`,
# neither impersonates `replay_control_plane`. The VM's default compute SA
# (already widened to the `cloud-platform` OAuth scope during Phase B) is
# therefore granted `roles/datastore.user` directly here, project-wide --
# Firestore has no per-collection IAM condition equivalent to the bucket's
# `resource.name.startsWith(...)` restriction above, so this is coarser
# than the GCS grant by necessity, not by choice.
resource "google_project_service" "firestore" {
  project            = var.project_id
  service            = "firestore.googleapis.com"
  disable_on_destroy = false
}

resource "google_firestore_database" "default" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.firestore]
}

resource "google_project_iam_member" "vm_firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"

  depends_on = [google_project_service.firestore]
}

# Phase D: Cloud Run hosts the owner-facing HTTP API only -- the replay
# worker (lab access, MCP spawning) stays VM-hosted, per infra/phaseD's own
# scoping notes (lab/docker-compose.yml binds services to 127.0.0.1 and
# shares a bind-mounted runs/ directory with the worker; neither survives a
# stateless container). Both the Cloud Run API and the VM worker must point
# at the same control-plane backend once this exists -- Cloud Run has no
# local disk to share a SQLite file with the VM -- so this only makes sense
# together with CAUSALOPS_CONTROL_PLANE_BACKEND=firestore on both sides.
resource "google_project_service" "run" {
  project            = var.project_id
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifact_registry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "api_images" {
  project       = var.project_id
  location      = var.region
  repository_id = "causalops-api"
  format        = "DOCKER"

  depends_on = [google_project_service.artifact_registry]
}

# The VM's default compute SA is what actually runs `docker push` (the
# gcloud CLI's active account, not terraform's own broader ADC identity) --
# without this it can create the repo but not push into it
# (artifactregistry.repositories.uploadArtifacts denied, confirmed live).
resource "google_artifact_registry_repository_iam_member" "vm_pushes_api_images" {
  project    = var.project_id
  location   = google_artifact_registry_repository.api_images.location
  repository = google_artifact_registry_repository.api_images.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"
}

# Dedicated identity for the Cloud Run revision -- deliberately not
# `replay_control_plane` (that one is scoped to GCS artifact writes/reads
# only) and not the VM's default compute SA (that one is scoped to
# Firestore + GCS-impersonation-granting only). Cloud Run needs Firestore
# access for the same reason the VM worker does; it needs no GCS role at
# all -- the owner-facing API never reads/writes the artifact bucket
# directly (report content is served out of the Firestore document,
# written there once by the VM worker's own `finalize()`).
resource "google_service_account" "cloud_run_api" {
  account_id   = "causalops-api"
  display_name = "CausalOps Cloud Run API"
}

resource "google_project_iam_member" "cloud_run_firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.cloud_run_api.email}"

  depends_on = [google_project_service.firestore]
}

# Empty var.api_image means "no image has been pushed yet" -- this whole
# resource (and the public-invoker grant below) is skipped rather than
# applying with a placeholder that would immediately fail to start.
resource "google_cloud_run_v2_service" "api" {
  count    = var.api_image == "" ? 0 : 1
  project  = var.project_id
  name     = "causalops-api"
  location = var.region

  template {
    service_account = google_service_account.cloud_run_api.email

    containers {
      image = var.api_image

      env {
        name  = "CAUSALOPS_ALLOWED_OWNERS"
        value = join(",", var.allowed_owners)
      }
      env {
        name  = "CAUSALOPS_GOOGLE_CLIENT_ID"
        value = var.google_client_id
      }
      env {
        name  = "CAUSALOPS_PROJECT_ROOT"
        value = "/app"
      }
      env {
        name  = "CAUSALOPS_CONTROL_PLANE_BACKEND"
        value = "firestore"
      }
      env {
        # Cloud Run has no lab/docker access -- without this, app()'s own
        # BackgroundControlPlaneWorkers would start here too and race the
        # VM's real worker for jobs it could never actually run (found
        # live this session as a real gap in the first Cloud Run deploy).
        name  = "CAUSALOPS_RUN_WORKER"
        value = "false"
      }

      ports {
        container_port = 8080
      }
    }
  }

  depends_on = [
    google_project_service.run,
    google_firestore_database.default,
    google_project_iam_member.cloud_run_firestore_user,
  ]
}

# The dashboard/API is meant to be internet-reachable; Google ID token auth
# is still enforced at the application layer (GoogleIdentityVerifier +
# CAUSALOPS_ALLOWED_OWNERS) exactly as it is on the VM today -- this grant
# only controls who can reach the Cloud Run *transport*, not who the
# application accepts as an owner.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count    = var.api_image == "" ? 0 : 1
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Sign-off gate for this phase's real recurring GCP cost -- a notification
# threshold, not a pre-spend block like cost_ledger.py's Claude-spend
# ceiling (GCP billing isn't triggered by a call this codebase controls).
# Empty var.billing_account_id skips this -- creating a budget needs
# billing-account-level IAM the project-scoped terraform identity may not
# hold; see infra/phaseD/VALIDATION.md for how this was actually applied.
resource "google_billing_budget" "phase_d_monthly" {
  count           = var.billing_account_id == "" ? 0 : 1
  billing_account = var.billing_account_id
  display_name    = "CausalOps Phase D monthly ceiling"

  budget_filter {
    projects = ["projects/${data.google_project.current.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_monthly_usd)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 1.0
  }
}
